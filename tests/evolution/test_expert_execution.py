"""Real bounded execution, cross-process Adam restore and unavailable inputs."""
import copy
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from types import SimpleNamespace

import pytest

from neuroshard.evolution import cohort_experiment, expert_window, expert_work, reference_data as data
from neuroshard.evolution.sharded import expert_execution, expert_replay, features, portable, prefix_audit, prefix_execution
from test_expert_window_replay import test_replay_reconstructs_the_actual_five_owner_training_checkpoint as replay_fixture


def distributed_reference(rank, home, parent, seed, plan, prepared, records, objects, binding):
    from datetime import timedelta
    import torch
    import torch.distributed as dist
    from transformers import LlamaConfig
    from neuroshard.evolution import expert_checkpoint, reference
    from neuroshard.evolution.sharded import cohort_features, incremental_state
    from neuroshard.evolution.sharded.model import Partition
    from neuroshard.evolution.sharded.wire import Wire

    reference.configure('cpu', plan['threads'])
    config = LlamaConfig(**parent['config'])
    config._attn_implementation = 'sdpa'
    shard = Partition(config, parent['boundaries'] if rank < 3 else seed['boundaries'], rank)
    if rank < 3:
        inherited = expert_checkpoint.parent_records(parent)
        with torch.no_grad():
            for name, parameter in shard.named_owned_parameters():
                spec = inherited[name]
                parameter.copy_(incremental_state.tensor_values(
                    portable.tensor_path(objects, spec['sha256']), spec)['weight'])
    shard.eval().requires_grad_(False)
    dist.init_process_group('gloo', init_method='file://' + str(home/'rendezvous'),
                            rank=rank, world_size=4, timeout=timedelta(seconds=60))
    try:
        result = cohort_features.produce(shard, Wire(rank, 4), records, prepared['batches'],
            home/'features', binding, plan['split'], plan['microbatch'], reference_expert=seed,
            parent=parent, objects=objects, resident_parameter_limit=plan['parameter_limit'])
        if rank == 3:
            data.save(home/'root.json', result)
    finally:
        dist.destroy_process_group()


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


def test_fresh_cohort_from_a_trained_expert_produces_and_replays_actual_updates(trajectory, tmp_path):
    from transformers import LlamaConfig
    from neuroshard.evolution import expert_checkpoint
    from neuroshard.evolution.sharded import cohort_state, incremental
    from neuroshard.evolution.sharded.model import Partition

    home, plan, prepared, states, old_profile, _ = trajectory
    seed = states[-1]
    objects = tmp_path/'objects'
    shutil.copytree(home/'objects', objects, copy_function=os.link)
    for spec in seed['tensors'].values():
        shutil.copyfile(portable.tensor_path(portable.directory(home/'rank-4', seed['step']), spec['sha256']),
                        portable.tensor_path(objects, spec['sha256']))
    plan = {**copy.deepcopy(plan), 'seed_expert': {'name': 'protocol', 'checkpoint': seed}}
    job = expert_execution.training_job(plan, prepared)
    config = LlamaConfig(**old_profile['parent']['config'])
    config._attn_implementation = 'sdpa'
    shard = Partition(config, seed['boundaries'], 3)
    optimizer = incremental.configure(shard, seed['split'], plan['training'])
    cohort_state.initialize_tail(shard, old_profile['parent'], objects, seed['split'], seed)
    initial = expert_checkpoint.pack(old_profile['parent'], cohort_state.commit_tail(tmp_path/'initial',
        shard, optimizer, old_profile['parent'], objects, job, 0, plan['training'], seed['split'], initial_expert=seed))
    assert initial['checkpoint'] != states[0]['checkpoint'] and not optimizer.state
    profile = {key: copy.deepcopy(value) for key, value in old_profile.items()
               if key not in ('feature_root', 'batch_roots')}
    profile.update(format=expert_work.PROSPECTIVE, checkpoint=initial,
                   batch_count=len(old_profile['batch_roots']), seed_expert=plan['seed_expert'])
    paths = {'inputs': home/'inputs', 'objects': objects, 'bank_home': tmp_path/'unused',
             'checkpoint_store': tmp_path/'producer'}
    produced = prefix_execution.produce_features(profile, plan, prepared, **paths, max_seconds=60)
    resolved = expert_work.resolve_prefix(profile, produced['feature_root'], produced['batch_roots'])
    paths['bank_home'] = paths['checkpoint_store']/'prefix'/produced['transcript_root']/'rank-2/features'
    # The retained distribution comes from the accepted expert, not from the
    # older parent that never learned this expert's facts.
    from neuroshard.evolution.sharded import feature_bank
    import torch
    bank_index = json.loads((paths['bank_home']/'index.json').read_bytes())
    bank = feature_bank.Reader(paths['bank_home'], produced['feature_root'], bank_index['binding'],
                               config, plan['microbatch'], len(prepared['batches']))
    all_rows = expert_execution.training_records(plan, prepared, paths['inputs'], old_profile['parent'])
    rows = [all_rows[index] for index in prepared['batches'][0]]
    for packet in bank.batch(0, rows, 'cpu'):
        with torch.no_grad():
            assert torch.equal(shard(packet['prefix'], packet['mask']), packet['reference'])
    assert produced['production_record']['context']['reference_expert'] == data.identity(seed)
    # Distributed production and sequential native replay must commit exactly
    # the same bytes, including the accepted reference, padding and row weights.
    import torch.multiprocessing as mp
    distributed = tmp_path/'distributed'
    distributed.mkdir()
    mp.spawn(distributed_reference, args=(distributed, profile['parent'], seed, plan, prepared,
             all_rows, objects, bank_index['binding']), nprocs=4, join=True)
    assert json.loads((distributed/'root.json').read_bytes()) == produced['feature_root']
    actual = expert_execution.produce_training(initial, 2, resolved, plan, prepared, **paths, max_seconds=60)
    window = actual['window']
    claim = {'kind': 'expert_training', 'id': '1'*64, 'parent_checkpoint': profile['parent'],
        'input_checkpoint': initial, 'output_checkpoint': window['output'], 'prepared': resolved['prepared'],
        'feature_root': resolved['feature_root'], 'feature_claim': '2'*64,
        'numerical_profile': resolved['numerical_profile'], 'stages': 2, 'record_root': data.identity(window),
        'work_ids': [step['work_identity'] for step in window['steps']], **actual}
    audit_paths = {**paths, 'checkpoint_store': tmp_path/'auditor'}
    report = expert_execution.execute_training(claim, resolved, plan, prepared, **audit_paths, max_seconds=60)
    assert expert_work.replay_report(claim, report)['valid']
    forged_profile = copy.deepcopy(resolved)
    forged_profile['seed_expert']['checkpoint'] = states[-2]
    with pytest.raises(ValueError, match='configured parent, job'):
        expert_execution.produce_training(initial, 1, forged_profile, plan, prepared, **paths)
    # Later windows use only the actual current boundary, without the previous
    # cohort's seed files. Both continuations must discover the same result.
    for spec in seed['tensors'].values():
        portable.tensor_path(objects, spec['sha256']).unlink()
    before = window['output']
    continued = expert_execution.produce_training(before, 2, resolved, plan, prepared, **paths)
    replayed = expert_execution.produce_training(before, 2, resolved, plan, prepared, **audit_paths)
    assert continued == replayed
    assert continued['window']['output']['step'] == plan['training']['steps']
    with pytest.raises(FileNotFoundError):
        expert_execution.produce_training(initial, 1, resolved, plan, prepared, **paths)


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


def test_fresh_prefix_producer_and_independent_replay_need_no_expected_output(trajectory, tmp_path):
    config = copy.deepcopy(configuration(trajectory, tmp_path / 'producer'))
    known = copy.deepcopy(config['profile'])
    fresh = config['profile']
    fresh.pop('feature_root')
    fresh['batch_count'] = len(fresh.pop('batch_roots'))
    fresh['format'] = expert_work.PROSPECTIVE
    data.save(tmp_path / 'executor.json', config)
    result = subprocess.run([sys.executable, '-m', 'neuroshard.evolution.sharded.expert_execution',
        '--config', str(tmp_path / 'executor.json'), '--produce-features'], input='{}',
        capture_output=True, text=True, timeout=90)
    assert result.returncode == 0, result.stderr
    produced = json.loads(result.stdout)
    assert produced['feature_root'] == known['feature_root'] and produced['batch_roots'] == known['batch_roots']
    claim = {'kind': 'expert_features', 'id': '1'*64, 'input_checkpoint': fresh['checkpoint'],
        'parent_checkpoint': fresh['parent'], 'prepared': fresh['prepared'],
        'feature_root': produced['feature_root'], 'batch_roots': produced['batch_roots'],
        'numerical_profile': fresh['numerical_profile'], 'stages': fresh['feature_stages'],
        'record_root': produced['transcript_root']}
    config['paths']['checkpoint_store'] = str(tmp_path / 'auditor')
    data.save(tmp_path / 'auditor.json', config)
    replay = subprocess.run([sys.executable, '-m', 'neuroshard.evolution.sharded.expert_execution',
        '--config', str(tmp_path / 'auditor.json')], input=json.dumps(claim),
        capture_output=True, text=True, timeout=90)
    assert replay.returncode == 0, replay.stderr
    assert expert_work.replay_report(claim, json.loads(replay.stdout))['valid']
    config['profile'] = expert_work.resolve_prefix(fresh, produced['feature_root'], produced['batch_roots'])
    config['paths']['bank_home'] = str(tmp_path / 'auditor/prefix' / claim['record_root'] / 'rank-2/features')
    actual = expert_execution.produce_training(fresh['checkpoint'], 2, config['profile'], config['plan'],
        config['prepared'], **config['paths'], max_seconds=60)
    assert actual['window'] == claim_for(trajectory, 0, 2)['window']
    forged = copy.deepcopy(claim)
    forged['batch_roots'][0] = '0'*64
    config['paths']['checkpoint_store'] = str(tmp_path / 'forgery')
    report = prefix_execution.execute_features(forged, fresh, config['plan'], config['prepared'],
        **config['paths'], max_seconds=60)
    assert not expert_work.replay_report(forged, report)['valid']
    assert list((tmp_path / 'forgery/prefix').iterdir()) == []


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
