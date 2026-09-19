"""Real subprocess fault injection for the replay controller, without CUDA.

The tiny child stands in for the numerical process. Numerical validity is
covered separately by shard-oracle tests and the recorded GPU experiment.
"""
import copy
import fcntl
import importlib.util
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

import pytest

from neuroshard.evolution.reference_data import identity, save, sha256

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT/'scripts/replay_sharded_claim.py'
spec = importlib.util.spec_from_file_location('portable_replay_controller', SCRIPT)
backend = importlib.util.module_from_spec(spec)
spec.loader.exec_module(backend)

CHILD = '''import json,os,sys,time
from pathlib import Path
from neuroshard.evolution.reference_data import identity,save
root=Path(__file__).resolve().parents[1]
control=json.loads((root/'control.json').read_bytes())
rank=int(os.environ['RANK'])
home=Path(sys.argv[sys.argv.index('--home')+1]);home.mkdir(parents=True,exist_ok=False)
with (root/'executed.jsonl').open('a') as log:
 log.write(json.dumps({'rank':rank,'home':str(home)})+'\\n');log.flush();os.fsync(log.fileno())
if rank==control.get('pause_rank'):
 save(root/'paused.json',{'rank':rank})
 time.sleep(control.get('pause_seconds',1))
if rank==control.get('fail_rank'):raise OSError('Injected interruption before a completed report')
rows=json.loads(Path(sys.argv[sys.argv.index('--transcripts')+1]).read_bytes())
folder=home/'shard-000001';folder.mkdir()
save(folder/'manifest.json',{'rank':rank,'step':1})
save(home/'audit.json',{'rank':rank,'valid':True,'binding':rows[rank]['binding'],'transcript_root':identity(rows)})
'''


@pytest.fixture
def trial(tmp_path):
    repository = tmp_path/'numerical'
    (repository/'scripts').mkdir(parents=True)
    (repository/'src').symlink_to(ROOT/'src', target_is_directory=True)
    source = repository/'scripts/run_sharded_training.py'
    source.write_text(CHILD)
    save(repository/'control.json', {})
    prepared = {'sources': {'scripts/run_sharded_training.py': sha256(source)}}
    prepared_path = tmp_path/'prepared.json'
    save(prepared_path, prepared)
    before = {'job': 'b'*64, 'step': 0, 'boundaries': [0, 1, 2]}
    after = {**before, 'step': 1, 'shards': [identity({'rank': rank, 'step': 1}) for rank in range(2)]}
    binding = {'job': before['job'], 'prepared': identity(prepared), 'input': identity(before),
        'output': identity(after), 'start': 0, 'end': 1, 'boundaries': before['boundaries'],
        'reference': identity(None)}
    rows = [{'binding': binding, 'rank': rank} for rank in range(2)]
    claim = {'id': 'a'*64, 'kind': 'portable_training', 'prepared': identity(prepared),
        'record_root': identity(rows), 'input_checkpoint': before, 'output_checkpoint': after,
        'stages': 2, 'reference_root': identity(None)}
    save(tmp_path/'before.json', before)
    save(tmp_path/'after.json', after)
    save(tmp_path/'transcripts.json', rows)
    catalog = {'repository': str(repository), 'python': sys.executable, 'prepared': str(prepared_path),
        'home': str(tmp_path/'replay'), 'rank_timeout_seconds': 10,
        'seeds': ['seed-0', 'seed-1'], 'checkpoints': {
            identity(before): [str(tmp_path/'before.json')]*2,
            identity(after): [str(tmp_path/'after.json')]*2},
        'transcripts': {identity(rows): {'manifest': str(tmp_path/'transcripts.json'), 'ranks': ['witness-0', 'witness-1']}}}
    return catalog, claim, repository


def executions(repository):
    return [json.loads(line)['rank'] for line in (repository/'executed.jsonl').read_text().splitlines()]


def test_completed_partitions_survive_failure_and_ledger_deadline_changes(trial):
    catalog, claim, repository = trial
    save(repository/'control.json', {'fail_rank': 1})
    with pytest.raises(RuntimeError, match='unavailable'):
        backend.run(catalog, claim)
    assert executions(repository) == [0, 1]
    assert not (Path(catalog['home'])/claim['id']/'report.json').exists()
    save(repository/'control.json', {})
    result = backend.run({**catalog, 'rank_timeout_seconds': 20}, {**claim, 'audit_commit_end': 999})
    assert all(row['valid'] for row in result)
    assert executions(repository) == [0, 1, 1]
    assert backend.run(catalog, claim) == result
    assert executions(repository) == [0, 1, 1]
    attempts = Path(catalog['home'])/claim['id']/'rank-1'
    assert (attempts/'attempt-000001').exists() and (attempts/'attempt-000002').exists()


def test_cached_results_cannot_change_sources_reference_or_work(trial):
    catalog, claim, repository = trial
    backend.run(catalog, claim)
    wrong = copy.deepcopy(claim)
    wrong['output_checkpoint']['step'] = 2
    with pytest.raises(ValueError, match='another computation'):
        backend.run(catalog, wrong)
    with pytest.raises(ValueError, match='requires its committed reference'):
        backend.run(catalog, {**claim, 'reference_root': 'c'*64})
    source = repository/'scripts/run_sharded_training.py'
    source.write_text(source.read_text()+'\n# Changed after the completed replay\n')
    with pytest.raises(ValueError, match='Numerical source differs'):
        backend.run(catalog, claim)
    assert executions(repository) == [0, 1]


def test_correct_tensors_do_not_authorize_forged_output_manifests(trial, tmp_path):
    catalog, claim, repository = trial
    old_output = identity(claim['output_checkpoint'])
    old_record = claim['record_root']
    claim['output_checkpoint']['shards'][0] = 'f'*64
    output = identity(claim['output_checkpoint'])
    save(tmp_path/'after.json', claim['output_checkpoint'])
    catalog['checkpoints'][output] = catalog['checkpoints'].pop(old_output)
    rows = json.loads((tmp_path/'transcripts.json').read_bytes())
    for row in rows:
        row['binding']['output'] = output
    save(tmp_path/'transcripts.json', rows)
    claim['record_root'] = identity(rows)
    catalog['transcripts'][claim['record_root']] = catalog['transcripts'].pop(old_record)
    reports = backend.run(catalog, claim)
    assert [row['valid'] for row in reports] == [False, True]
    assert reports[0]['reason'] == 'Replayed shard manifest differs from the claimed output'
    assert executions(repository) == [0, 1]


def test_changed_verifier_cannot_reuse_prior_positive_reports(trial, monkeypatch):
    catalog, claim, repository = trial
    backend.run(catalog, claim)
    original = backend.sha256
    monkeypatch.setattr(backend, 'sha256', lambda path:
        'f'*64 if Path(path).resolve() == SCRIPT.resolve() else original(path))
    with pytest.raises(ValueError, match='another computation'):
        backend.run(catalog, claim)
    assert executions(repository) == [0, 1]


def test_child_keeps_the_claim_locked_after_parent_termination(trial, tmp_path):
    catalog, claim, repository = trial
    save(repository/'control.json', {'pause_rank': 1, 'pause_seconds': 2})
    catalog_path = tmp_path/'catalog.json'
    save(catalog_path, catalog)
    process = subprocess.Popen([sys.executable, str(SCRIPT), '--catalog', str(catalog_path)],
        stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        env={**os.environ, 'PYTHONPATH': str(ROOT/'src')}, start_new_session=True)
    process.stdin.write(json.dumps(claim).encode())
    process.stdin.close()
    try:
        deadline = time.monotonic()+10
        while not (repository/'paused.json').exists():
            if process.poll() is not None or time.monotonic() > deadline:
                raise AssertionError('Child failed to reach the injected pause')
            time.sleep(.02)
        process.terminate()
        process.wait(timeout=5)
        with pytest.raises(RuntimeError, match='Another replay'):
            backend.run(catalog, claim)
        lock_path = Path(catalog['home'])/claim['id']/'.lock'
        with lock_path.open('ab') as lock:
            deadline = time.monotonic()+5
            while True:
                try:
                    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    break
                except BlockingIOError:
                    assert time.monotonic() < deadline
                    time.sleep(.02)
        save(repository/'control.json', {})
        result = backend.run(catalog, claim)
        assert all(row['valid'] for row in result)
        assert executions(repository) == [0, 1, 1]
    finally:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        process.wait(timeout=5)
        process.stdout.close()
        process.stderr.close()
