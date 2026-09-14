#!/usr/bin/env python3
"""Replay a native portable claim using operator-staged files, one GPU shard at a time.

The catalog contains local paths, never commands supplied by a claim. It points
at the frozen numerical checkout, its GPU Python executable, prepared data,
rank-specific seeds, checkpoint paths and transcript witnesses. Network fetching
is deliberately outside this verifier; unavailable files cannot earn a report.
"""
import argparse
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from neuroshard.evolution import portable_work
from neuroshard.evolution.reference_data import identity, save, sha256
from neuroshard.evolution.schema import root

# Catch only explicit mathematical/witness mismatches from the pinned auditor.
# OOM, process failure, missing files and corrupt downloads remain unavailable.
MISMATCHES = [
    'Replayed forward value or backward gradient differs',
    'Replayed scalar differs',
    'Replayed gradient inventory or declaration differs',
    'Replayed weights or Adam moments differ from the claimed output',
    'Replay requested an unrecorded operation',
    'Replay communication order differs',
    'Replay did not cover the complete communication window',
    'Replayed input endpoint or shape differs',
    'Invalid witness tensor inventory',
    'Invalid witness tensor',
]


def run(catalog, claim):
    if claim.get('kind') != 'portable_training':
        raise ValueError('Require a native portable training claim')
    prepared = json.loads(Path(catalog['prepared']).read_bytes())
    if identity(prepared) != claim['prepared']:
        raise ValueError('Prepared computation differs from native obligation')
    repository = Path(catalog['repository']).resolve()
    for name, expected_digest in prepared['sources'].items():
        source = (repository/name).resolve()
        if not source.is_relative_to(repository) or sha256(source) != expected_digest:
            raise ValueError('Numerical source differs from the frozen job')
    reference_root = claim.get('reference_root', identity(None))
    references = catalog.get('reference')
    if references:
        if any(identity(json.loads(Path(path).read_bytes())) != reference_root for path in references):
            raise ValueError('Reference checkpoint differs from the native obligation')
    elif reference_root != identity(None):
        raise ValueError('The native obligation requires its committed reference checkpoint')
    base = Path(catalog['home'])/root(claim['id'])
    base.mkdir(parents=True, exist_ok=True, mode=0o700)
    with (base/'.lock').open('ab') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError('Another replay still owns this claim') from exc
        # Exclude changing ledger deadlines and local timeout preferences. They
        # do not alter the prescribed computation or authorize another result.
        binding = {'claim': {k: claim[k] for k in ['id', 'kind', 'prepared', 'record_root', 'stages',
                    'input_checkpoint', 'output_checkpoint']}, 'reference_root': reference_root,
                   'catalog': {k: v for k, v in catalog.items() if k not in ('home', 'rank_timeout_seconds')},
                   'verifier_sha256': sha256(Path(__file__))}
        marker = base/'resume.json'
        if marker.exists():
            if json.loads(marker.read_bytes()) != binding:
                raise ValueError('Saved replay belongs to another computation or catalog')
        else:
            if any(p.name not in ('.lock', 'resume.json.pending') for p in base.iterdir()):
                raise ValueError('Existing replay directory has no bound recovery record')
            save(marker, binding)
        return replay_locked(catalog, claim, base, identity(binding), lock.fileno())


def replay_locked(catalog, claim, base, binding, lock_descriptor):
    before, after = claim['input_checkpoint'], claim['output_checkpoint']
    count = len(before['boundaries'])-1
    source = catalog['checkpoints'][identity(before)]
    expected = catalog['checkpoints'][identity(after)]
    witnesses = catalog['transcripts'][claim['record_root']]
    rows = json.loads(Path(witnesses['manifest']).read_bytes())
    if identity(rows) != claim['record_root']:
        raise ValueError('Closed transcript commitment differs')
    if any(len(values) != count for values in [source, expected, catalog['seeds'], witnesses['ranks']]):
        raise ValueError('The catalog must cover every model partition')
    if catalog.get('reference') and len(catalog['reference']) != count:
        raise ValueError('The reference catalog must cover every partition')
    reports = []
    reused = 0
    started = time.monotonic()
    for rank in range(count):
        for path, committed in [(source[rank], before), (expected[rank], after)]:
            if json.loads(Path(path).read_bytes()) != committed:
                raise ValueError('Local checkpoint differs from native input or output')
        completed_path = base/f'rank-{rank}.complete.json'
        if completed_path.exists():
            completed = json.loads(completed_path.read_bytes())
            if completed['binding'] != binding:
                raise ValueError('Saved partition replay has a different binding')
            reports.append(completed['report'])
            reused += 1
            continue
        attempts = base/f'rank-{rank}'
        attempts.mkdir(exist_ok=True)
        number = 1
        while (attempts/f'attempt-{number:06d}').exists() or (attempts/f'attempt-{number:06d}.log').exists():
            number += 1
        home = attempts/f'attempt-{number:06d}'
        args = [str(Path(catalog['repository'])/'scripts/run_sharded_training.py'), 'adaptive', 'audit',
            '--prepared', catalog['prepared'], '--seed', catalog['seeds'][rank], '--home', str(home),
            '--resume', source[rank], '--expected', expected[rank],
            '--transcripts', witnesses['manifest'], '--witness', witnesses['ranks'][rank]]
        if catalog.get('reference'):
            args += ['--reference', catalog['reference'][rank]]
        # Run the unchanged frozen entry point. The wrapper produces a negative
        # verdict only for its enumerated, typed replay mismatch exceptions.
        wrapper = """import json,runpy,sys
from pathlib import Path
from neuroshard.evolution.reference_data import save
spec=json.loads(sys.argv[1])
sys.argv=spec['args']
try:
 runpy.run_path(sys.argv[0],run_name='__main__')
except ValueError as exc:
 if str(exc) not in spec['mismatches']:raise
 rows=json.loads(Path(spec['transcripts']).read_bytes())
 save(Path(spec['home'])/'audit.json',{'valid':False,'rank':spec['rank'],
      'binding':rows[spec['rank']]['binding'],'transcript_root':spec['root'],'reason':str(exc)})
"""
        spec = {'args': args, 'mismatches': MISMATCHES, 'transcripts': witnesses['manifest'],
                'home': str(home), 'rank': rank, 'root': claim['record_root']}
        env = {**os.environ, 'RANK': str(rank), 'WORLD_SIZE': str(count),
               'PYTHONPATH': str(Path(catalog['repository'])/'src'),
               'ATEN_CPU_CAPABILITY': 'default', 'MKL_ENABLE_INSTRUCTIONS': 'SSE4_2'}
        with home.with_suffix('.log').open('xb') as log:
            result = subprocess.run([catalog['python'], '-c', wrapper, json.dumps(spec)],
                env=env, cwd=catalog['repository'], stdout=log, stderr=log,
                timeout=catalog.get('rank_timeout_seconds', 600), pass_fds=(lock_descriptor,))
        if result.returncode:
            raise RuntimeError('A partition is unavailable; inspect the private replay log')
        report = json.loads((home/'audit.json').read_bytes())
        if report['valid'] is True:
            # The frozen numerical oracle checks learned tensors and optimizer
            # semantics. Also bind its complete regenerated manifest: arbitrary
            # shard/RNG commitments would make the accepted cursor unusable.
            manifest = json.loads((home/f"shard-{after['step']:06d}"/'manifest.json').read_bytes())
            report['output_manifest'] = identity(manifest)
            if report['output_manifest'] != after['shards'][rank]:
                report.update(valid=False, reason='Replayed shard manifest differs from the claimed output')
        save(completed_path, {'binding': binding, 'report': report})
        reports.append(report)
    result = portable_work.replay_report(claim, reports)
    report = {**result, 'seconds_this_invocation': time.monotonic()-started, 'reused_partitions': reused}
    history = base/'invocations'
    history.mkdir(exist_ok=True)
    number = 1
    while (history/f'{number:06d}.json').exists():
        number += 1
    save(history/f'{number:06d}.json', report)
    save(base/'report.json', report)
    return reports


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--catalog', type=Path, required=True)
    args = parser.parse_args()
    raw = sys.stdin.buffer.read(2*1024**2+1)
    if len(raw) > 2*1024**2:
        raise ValueError('Native claim exceeds its bound')
    print(json.dumps(run(json.loads(args.catalog.read_bytes()), json.loads(raw))), flush=True)
