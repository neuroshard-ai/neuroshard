"""Real bounded execution, cross-process Adam restore and unavailable inputs."""
import copy
import json
from pathlib import Path
import shutil
import subprocess
import sys
from types import SimpleNamespace

import pytest

from neuroshard.evolution import cohort_experiment, expert_window, expert_work, reference_data as data
from neuroshard.evolution.sharded import expert_execution, expert_replay, features, portable, prefix_audit, prefix_execution
from test_expert_window_replay import test_replay_reconstructs_the_actual_five_owner_training_checkpoint as replay_fixture


def test_real_archived_training_job_keeps_its_original_domain():
    root = Path(__file__).resolve().parents[2]
    read = lambda name: json.loads((root / 'config/experiments' / name).read_bytes())
    plan, prepared = read('interpreted-cohort.json'), read('interpreted-cohort-prepared.json')
    archived = read('expert-window-replay.json')['training_job']
    assert expert_execution.training_job(plan, prepared) == archived
    assert cohort_experiment.job(plan, prepared) != archived


@pytest.fixture(scope='module')
def trajectory(tmp_path_factory):
    home = tmp_path_factory.mktemp('expert-execution')
    replay_fixture(home)
    read = lambda path: json.loads(path.read_bytes())
    plan, prepared = read(home / 'plan.json'), read(home / 'prepared.json')
    states = [read(home / 'replay' / f'checkpoint-{i:06d}.json') for i in range(5)]
    bank = read(home / 'rank-4/features/index.json')
    profile = {'format': expert_work.FORMAT, 'parent': read(home / 'parent.json'),
        'checkpoint': states[0], 'prepared': data.identity(prepared),
        'feature_root': data.identity(bank), 'feature_stages': 336,
        'batch_roots': [expert_replay.batch_identity(batch) for batch in bank['batches']],
        'schedule': prepared['schedule'], 'numerical_profile': 'b' * 64}
    return home, plan, prepared, states, profile, read(home / 'replay/window-000000.json')


def claim_for(trajectory, start, end):
    _, _, _, states, profile, whole = trajectory
    window = {**copy.deepcopy(whole), 'input': states[start], 'output': states[end],
              'steps': copy.deepcopy(whole['steps'][start:end])}
    return {'kind': 'expert_training', 'id': data.identity(['claim', start, end]),
        'input_checkpoint': states[start], 'output_checkpoint': states[end],
        'parent_checkpoint': profile['parent'], 'prepared': profile['prepared'],
        'feature_root': profile['feature_root'], 'numerical_profile': profile['numerical_profile'],
        'feature_claim': 'f' * 64, 'stages': end - start,
        'record_root': data.identity(window), 'window': window, 'intermediates': states[start:end + 1],
        'work_ids': [step['work_identity'] for step in window['steps']]}


def configuration(trajectory, store):
    home, plan, prepared, _, profile, _ = trajectory
    return {'format': 'neuroshard-expert-executor-v1', 'profile': profile, 'plan': plan,
        'prepared': prepared, 'max_seconds': 60, 'paths': {'inputs': str(home / 'inputs'),
        'objects': str(home / 'objects'), 'bank_home': str(home / 'rank-4/features'),
        'checkpoint_store': str(store)}}


def execute(claim, config):
    return expert_execution.execute_training(claim, config['profile'], config['plan'], config['prepared'],
        **config['paths'], max_seconds=config['max_seconds'])


@pytest.fixture(autouse=True)
def runtime(monkeypatch):
    monkeypatch.setenv('PYTORCH_CUDA_ALLOC_CONF', 'test')


def test_separate_auditor_resumes_real_adam_and_retries_still_execute(trajectory, tmp_path, monkeypatch):
    first, second = claim_for(trajectory, 0, 2), claim_for(trajectory, 2, 4)
    initial_store, resumed_store = tmp_path / 'first', tmp_path / 'second'
    config = configuration(trajectory, initial_store)
    assert expert_work.replay_report(first, execute(first, config))['valid']
    root = first['output_checkpoint']['checkpoint']
    # Transfer one boundary, with no earlier trajectory, reports or other states.
    shutil.copytree(initial_store / root, resumed_store / root)
    resumed = configuration(trajectory, resumed_store)
    path = tmp_path / 'executor.json'
    data.save(path, resumed)
    result = subprocess.run([sys.executable, '-m', 'neuroshard.evolution.sharded.expert_execution',
        '--config', str(path)], input=json.dumps(second), capture_output=True, text=True, timeout=90)
    assert result.returncode == 0, result.stderr
    assert expert_work.replay_report(second, json.loads(result.stdout))['valid']
    final = second['output_checkpoint']
    destination = resumed_store / final['checkpoint']
    assert json.loads((destination / 'checkpoint.json').read_bytes()) == final
    folder = portable.directory(destination, 4)
    for spec in final['tensors'].values():
        assert spec['optimizer_step'] == 4
        assert data.sha256(portable.tensor_path(folder, spec['sha256'])) == spec['sha256']
    # A duplicate audit must still calculate both updates. Existing payloads
    # are checked and preserved, never substituted for an execution verdict.
    calls, original = [], features.train_step

    def measured(*args, **kwargs):
        calls.append(args[6])
        return original(*args, **kwargs)

    monkeypatch.setattr(features, 'train_step', measured)
    assert expert_work.replay_report(second, execute(second, resumed))['valid']
    assert calls == [2, 3]
    assert {p.name for p in resumed_store.iterdir()} == {root, final['checkpoint']}


def test_producer_discovers_outputs_without_expected_claim_then_independent_auditor_replays(trajectory, tmp_path):
    _, _, _, states, _, _ = trajectory
    producer = configuration(trajectory, tmp_path / 'producer')
    # No future state, expected metrics or claimed result enters this interface.
    first = expert_execution.produce_training(states[0], 2, producer['profile'],
        producer['plan'], producer['prepared'], **producer['paths'], max_seconds=60)
    assert first == {key: claim_for(trajectory, 0, 2)[key] for key in ('window', 'intermediates')}
    saved = tmp_path / 'producer' / first['window']['output']['checkpoint']
    restarted = configuration(trajectory, tmp_path / 'replacement')
    shutil.copytree(saved, tmp_path / 'replacement' / saved.name)
    path = tmp_path / 'producer.json'
    data.save(path, restarted)
    process = subprocess.run([sys.executable, '-m', 'neuroshard.evolution.sharded.expert_execution',
        '--config', str(path), '--produce'], input=json.dumps({
            'input_checkpoint': first['window']['output'], 'steps': 2}),
        capture_output=True, text=True, timeout=90)
    assert process.returncode == 0, process.stderr
    second = json.loads(process.stdout)
    assert second['window']['output'] == states[4]
    auditor = configuration(trajectory, tmp_path / 'auditor')
    for start, end, produced in ((0, 2, first), (2, 4, second)):
        claim = claim_for(trajectory, start, end)
        claim.update(produced)
        claim['output_checkpoint'] = produced['window']['output']
        claim['record_root'] = data.identity(produced['window'])
        claim['work_ids'] = [step['work_identity'] for step in produced['window']['steps']]
        assert expert_work.replay_report(claim, execute(claim, auditor))['valid']


def test_producer_bounds_and_storage_are_required_before_returning_work(trajectory, tmp_path, monkeypatch):
    config = configuration(trajectory, tmp_path / 'producer')
    before = trajectory[3][0]
    for count in (0, 5, True):
        with pytest.raises(ValueError):
            expert_execution.produce_training(before, count, config['profile'], config['plan'],
                config['prepared'], **config['paths'])

    def unavailable(*args, **kwargs):
        raise OSError('Producer checkpoint unavailable')

    monkeypatch.setattr(expert_execution.incremental_state, 'write', unavailable)
    with pytest.raises(OSError, match='checkpoint unavailable'):
        expert_execution.produce_training(before, 1, config['profile'], config['plan'],
            config['prepared'], **config['paths'])


def test_forged_finite_measurement_is_refuted_without_publishing_checkpoint(trajectory, tmp_path):
    claim = claim_for(trajectory, 0, 2)
    claim['window']['steps'][0]['metrics']['loss'] += 1
    claim['record_root'] = data.identity(claim['window'])
    # This is an admissible metadata claim; real computation must refute it.
    expert_window.validate(claim['parent_checkpoint'], claim['input_checkpoint'], claim['window'],
        claim['intermediates'], [step['batch'] for step in claim['window']['steps']], claim['numerical_profile'])
    report = execute(claim, configuration(trajectory, tmp_path / 'states'))
    assert not expert_work.replay_report(claim, report)['valid']
    assert report['stages'][0] == {'stage': 0, 'valid': False}
    assert not (tmp_path / 'states').exists()


def test_missing_boundary_does_not_replay_history(trajectory, tmp_path, monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError('No training is allowed without the input checkpoint')

    monkeypatch.setattr(features, 'train_step', forbidden)
    with pytest.raises(FileNotFoundError):
        execute(claim_for(trajectory, 2, 4), configuration(trajectory, tmp_path / 'missing'))


def test_corrupted_features_never_produce_a_valid_report(trajectory, tmp_path):
    config = configuration(trajectory, tmp_path / 'states')
    damaged = tmp_path / 'features'
    shutil.copytree(config['paths']['bank_home'], damaged)
    index = json.loads((damaged / 'index.json').read_bytes())
    path = damaged / index['batches'][0]['files'][0]['file']
    payload = path.read_bytes()
    path.write_bytes(payload[:-1] + bytes([payload[-1] ^ 1]))
    config['paths']['bank_home'] = str(damaged)
    with pytest.raises(ValueError, match='tensor bytes'):
        execute(claim_for(trajectory, 0, 2), config)
    assert not (tmp_path / 'states').exists()


def test_substituted_job_is_rejected_before_numerical_setup(trajectory, tmp_path, monkeypatch):
    config = copy.deepcopy(configuration(trajectory, tmp_path / 'states'))
    config['plan']['objective']['kl_strength'] = 1.

    def forbidden(*args, **kwargs):
        raise AssertionError('A substituted objective cannot allocate the neural runtime')

    monkeypatch.setattr(expert_execution.reference, 'configure', forbidden)
    with pytest.raises(ValueError, match='configured parent, job'):
        execute(claim_for(trajectory, 0, 2), config)


def test_corrupted_adam_boundary_cannot_be_used_for_the_next_window(trajectory, tmp_path, monkeypatch):
    config = configuration(trajectory, tmp_path / 'states')
    first = claim_for(trajectory, 0, 2)
    assert expert_work.replay_report(first, execute(first, config))['valid']
    output = first['output_checkpoint']
    spec = next(iter(output['tensors'].values()))
    folder = portable.directory(Path(config['paths']['checkpoint_store']) / output['checkpoint'], 2)
    path = portable.tensor_path(folder, spec['sha256'])
    payload = path.read_bytes()
    path.write_bytes(payload[:-1] + bytes([payload[-1] ^ 1]))

    def forbidden(*args, **kwargs):
        raise AssertionError('Corrupted input must fail before training')

    monkeypatch.setattr(features, 'train_step', forbidden)
    with pytest.raises(ValueError, match='corrupted incremental tensor'):
        execute(claim_for(trajectory, 2, 4), config)


def test_storage_failure_cannot_return_an_acceptance_report(trajectory, tmp_path, monkeypatch):
    def unavailable(*args, **kwargs):
        raise OSError('Injected checkpoint storage failure')

    monkeypatch.setattr(expert_execution.incremental_state, 'write', unavailable)
    with pytest.raises(OSError, match='storage failure'):
        execute(claim_for(trajectory, 0, 2), configuration(trajectory, tmp_path / 'states'))
    assert list((tmp_path / 'states').iterdir()) == []


def test_deadline_after_computation_cannot_publish_a_valid_result(trajectory, tmp_path, monkeypatch):
    clock = iter([0., 0., 0., 2.])
    monkeypatch.setattr(expert_execution, 'time', SimpleNamespace(monotonic=lambda: next(clock)))
    config = configuration(trajectory, tmp_path / 'states')
    config['max_seconds'] = 1
    with pytest.raises(TimeoutError, match='deadline'):
        execute(claim_for(trajectory, 0, 2), config)
    assert not (tmp_path / 'states').exists()


@pytest.fixture(scope='module')
def prefix_claim(trajectory, tmp_path_factory):
    from transformers import LlamaConfig
    from neuroshard.evolution.sharded.model import Partition
    home, plan, prepared, states, profile, _ = trajectory
    original = json.loads((home / 'rank-4/features/index.json').read_bytes())
    config = LlamaConfig(**profile['parent']['config'])
    config._attn_implementation = 'sdpa'
    records = cohort_experiment.rows(prepared, home / 'inputs', 'train')
    baseline = tmp_path_factory.mktemp('prefix-producer')
    reports, incoming = [], None
    for rank in range(3):
        shard = Partition(config, profile['parent']['boundaries'], rank)
        stage = baseline / f'rank-{rank}'
        report = prefix_audit.replay_stage(shard, profile['parent'], home / 'objects', records,
            prepared['batches'], original['binding'], plan['split'], plan['microbatch'],
            profile['feature_root'], stage, incoming, 60)
        reports.append(report)
        incoming = (stage / 'features', report)
        del shard
    record = prefix_execution.production_record(reports)
    changed_times = copy.deepcopy(reports)
    for report in changed_times:
        report['seconds'] += 1000
    assert prefix_execution.production_record(changed_times) == record
    return {'kind': 'expert_features', 'id': 'a' * 64, 'input_checkpoint': states[0],
        'parent_checkpoint': profile['parent'], 'prepared': profile['prepared'],
        'feature_root': profile['feature_root'], 'numerical_profile': profile['numerical_profile'],
        'stages': profile['feature_stages'], 'record_root': data.identity(record)}


def test_native_prefix_backend_retains_real_features_for_training(trajectory, prefix_claim, tmp_path):
    config = configuration(trajectory, tmp_path / 'states')
    path = tmp_path / 'executor.json'
    data.save(path, config)
    result = subprocess.run([sys.executable, '-m', 'neuroshard.evolution.sharded.expert_execution',
        '--config', str(path)], input=json.dumps(prefix_claim), capture_output=True, text=True, timeout=90)
    assert result.returncode == 0, result.stderr
    report = json.loads(result.stdout)
    assert expert_work.replay_report(prefix_claim, report)['valid']
    assert len(report['stages']) == 336
    produced = tmp_path / 'states/prefix' / prefix_claim['record_root'] / 'rank-2/features'
    assert data.identity(json.loads((produced / 'index.json').read_bytes())) == prefix_claim['feature_root']
    config['paths']['bank_home'] = str(produced)
    training = claim_for(trajectory, 0, 2)
    assert expert_work.replay_report(training, execute(training, config))['valid']


@pytest.mark.parametrize('forged', ['record_root', 'feature_root'])
def test_computed_prefix_refutes_changed_production(trajectory, prefix_claim, tmp_path, forged):
    config = copy.deepcopy(configuration(trajectory, tmp_path / 'states'))
    claim = copy.deepcopy(prefix_claim)
    claim[forged] = '0' * 64
    if forged == 'feature_root':
        config['profile']['feature_root'] = claim[forged]
    report = prefix_execution.execute_features(claim, config['profile'], config['plan'], config['prepared'],
        **config['paths'], max_seconds=config['max_seconds'])
    assert not expert_work.replay_report(claim, report)['valid']
    assert len(report['stages']) == 336
    assert list((tmp_path / 'states/prefix').iterdir()) == []
