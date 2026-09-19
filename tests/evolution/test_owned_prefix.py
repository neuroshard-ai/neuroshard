"""Separate partition owners reproduce the native sequential prefix oracle."""
import copy
import json

import pytest

from neuroshard.evolution import expert_work
from neuroshard.evolution import expert_data
from neuroshard.evolution.reference_data import identity
from neuroshard.evolution.sharded import expert_execution, prefix_execution
from test_expert_data import prepared


def test_owned_stages_match_fresh_complete_replay_and_reject_splicing(prepared):
    home, _, job, _, _, _, _, plan, inputs = prepared
    profile = job['work']
    paths = {'inputs': home/'general-inputs', 'objects': home/'objects'}
    expected = prefix_execution.produce_features(profile, plan, inputs, **paths,
        bank_home=home/'unused-bank', checkpoint_store=home/'sequential', max_seconds=60)
    reports, incoming = [], None
    for rank in range(3):
        owner = home/f'owner-{rank}'
        report = prefix_execution.owned_stage(rank, profile, plan, inputs,
            **paths, home=owner, incoming=incoming, max_seconds=60)
        reports.append(report)
        incoming = (owner/'features', report)
    bank = json.loads((home/'owner-2/features/index.json').read_bytes())
    records = expert_execution.training_records(plan, inputs, paths['inputs'], profile['parent'])
    actual = prefix_execution.owned_production(profile, plan, inputs, records, reports, bank)
    assert actual == expected
    claim = {'kind': 'expert_features', 'id': 'a'*64,
        'input_checkpoint': profile['checkpoint'], 'parent_checkpoint': profile['parent'],
        'prepared': profile['prepared'], 'numerical_profile': profile['numerical_profile'],
        'stages': profile['feature_stages'], 'record_root': actual['transcript_root'],
        'feature_root': actual['feature_root'], 'batch_roots': actual['batch_roots']}
    verdict = prefix_execution.owned_verdict(claim, profile, plan, inputs, actual)
    assert expert_work.replay_report(claim, verdict)['valid']
    forged = {**claim, 'feature_root': 'b'*64}
    assert not expert_work.replay_report(forged,
        prefix_execution.owned_verdict(forged, profile, plan, inputs, actual))['valid']
    spliced = copy.deepcopy(reports)
    spliced[1]['input_root'] = 'c'*64
    with pytest.raises(ValueError, match='connected'):
        prefix_execution.owned_production(profile, plan, inputs, records, spliced, bank)
    different = copy.deepcopy(bank)
    different['binding'] = 'd'*64
    with pytest.raises(ValueError, match='prescribed'):
        prefix_execution.owned_production(profile, plan, inputs, records, reports, different)


def test_initialization_persists_actual_seed_weights_with_fresh_adam(prepared):
    from neuroshard.evolution.sharded.incremental_state import tensor_values
    from neuroshard.evolution.sharded.portable import tensor_path
    home, _, job, _, _, _, _, plan, inputs = prepared
    parent = job['work']['parent']
    paths = {'objects': home/'objects', 'checkpoint_store': home/'initialized'}
    actual = expert_execution.initialize(parent, plan, inputs, **paths)
    assert actual == job['work']['checkpoint']
    seed = json.loads((home/'graph.json').read_bytes())['experts']['astronomy']
    continued = {**plan, 'seed_expert': {'name': 'astronomy', 'checkpoint': seed}}
    new_inputs = {**inputs, 'plan': identity(continued)}
    value = expert_execution.initialize(parent, continued, new_inputs, **paths)
    assert value['job'] == expert_data.job_identity(continued, new_inputs)
    assert value['step'] == 0 and value['checkpoint'] != actual['checkpoint']
    for name, spec in value['tensors'].items():
        folder = paths['checkpoint_store']/value['checkpoint']/'shard-000000'
        restored = tensor_values(tensor_path(folder, spec['sha256']), spec)
        old = tensor_values(tensor_path(home/'objects', seed['tensors'][name]['sha256']), seed['tensors'][name])
        assert set(restored) == {'weight'} and restored['weight'].equal(old['weight'])
    assert expert_execution.initialize(parent, continued, new_inputs, **paths) == value
