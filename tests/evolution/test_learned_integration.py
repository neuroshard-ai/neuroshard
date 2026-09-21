import json
from pathlib import Path

import pytest

from neuroshard.evolution.learned_integration import (
    CONTRACT_IDENTITY, FORMAT, GROWTH_PLAN, METHOD_FORMAT, UNIQUE_ADDED_HISTORY,
    UNIQUE_INCUMBENT_HISTORY, LastLayerMixture, SwiGLUExpert, active_mlp_flops_per_token,
    bind_method_freeze, bind_spec, complete_development_gate, control_from_parent,
    generated_passed, load_spec, method_freeze, refuse_launch, score_gate,
    score_retention, train_matched, train_step, training_schedule,
)
from neuroshard.evolution.reference_data import identity, sha256


ROOT = Path(__file__).resolve().parents[2]


def spec():
    return json.loads((ROOT / 'config/experiments/learned-integration.json').read_text())


def test_frozen_contract_identity_is_pinned():
    current = spec()
    assert identity(current) == CONTRACT_IDENTITY
    assert current['format'] == FORMAT
    assert bind_spec(current) == {
        'learned_integration': CONTRACT_IDENTITY,
        'growth_plan': GROWTH_PLAN,
        'gpu_launch_authorized': False,
        'admission_evidence': False,
    }
    assert identity(load_spec()) == CONTRACT_IDENTITY


def test_specification_does_not_authorize_training_or_gpus():
    current = spec()
    assert current['train'] is False
    assert current['gpu_launch_authorized'] is False
    assert current['admission_evidence'] is False
    assert current['confirmation_opened'] is False
    assert current['stage2']['authorized'] is False
    assert current['fresh_dataset_alone_is_not_the_experiment'] is True


def test_opened_programming_cases_are_ineligible():
    current = spec()
    forbidden = set(current['forbidden_evaluation_task_ids'])
    growth = json.loads((ROOT / 'config/experiments/programming-growth.json').read_text())
    assert identity(growth) == GROWTH_PLAN
    assert set(growth['parent_final_task_ids']) <= forbidden
    assert len(growth['parent_final_task_ids']) == 128
    assert set(UNIQUE_ADDED_HISTORY + UNIQUE_INCUMBENT_HISTORY) <= forbidden
    for role in ('train_new', 'train_replay', 'retention', 'development', 'confirmation'):
        assert not set(current['splits'][role]) & forbidden


def test_control_and_gate_are_frozen_before_outcomes():
    current = spec()
    assert current['control']['matched_data_and_steps'] is True
    assert current['success']['loss_alone_is_success'] is False
    assert current['architecture']['active_experts_per_token'] == 1
    assert current['architecture']['experts_after_expansion'] == 2
    assert current['architecture']['accepted_modules_frozen'] is True
    assert current['architecture']['existing_moe_layer_is_this_runtime'] is False
    assert current['gate']['minimum_confirmation_gain_tasks_vs_control'] == 1
    assert current['gate']['development_must_pass_before_confirmation'] is True
    assert current['general_retention']['opened'] is False
    assert current['general_retention']['documents'] == 8


@pytest.mark.parametrize('field,value,match', [
    ('train', True, 'does not authorize training'),
    ('gpu_launch_authorized', True, 'does not authorize training'),
    ('admission_evidence', True, 'may not count as admission'),
    ('original_final_opened', True, 'does not open the original'),
    ('opened_64_reusable_as_evaluation', True, 'remain development history'),
    ('confirmation_opened', True, 'Confirmation remains closed'),
    ('fresh_dataset_alone_is_not_the_experiment', False, 'fresh dataset alone'),
])
def test_bind_rejects_campaign_reopen_flags(field, value, match, monkeypatch):
    current = spec()
    current[field] = value
    from neuroshard.evolution import learned_integration as module
    monkeypatch.setattr(module, 'CONTRACT_IDENTITY', identity(current))
    with pytest.raises(ValueError, match=match):
        bind_spec(current)


def test_bind_rejects_identity_drift():
    current = spec()
    current['seed'] = 0
    with pytest.raises(ValueError, match='does not bind this measurement'):
        bind_spec(current)


def test_bind_rejects_runtime_and_success_drift_when_identity_is_forced(monkeypatch):
    from neuroshard.evolution import learned_integration as module

    def reject(mutate, match):
        broken = spec()
        mutate(broken)
        monkeypatch.setattr(module, 'CONTRACT_IDENTITY', identity(broken))
        with pytest.raises(ValueError, match=match):
            bind_spec(broken)

    reject(lambda current: current['architecture'].__setitem__('active_experts_per_token', 2),
           'Active expert count')
    reject(lambda current: current['architecture'].__setitem__('existing_moe_layer_is_this_runtime', True),
           'not this experiment runtime')
    reject(lambda current: current['success'].__setitem__('loss_alone_is_success', True),
           'Lower loss is not success')
    reject(lambda current: current['control'].__setitem__('matched_data_and_steps', False),
           'reuse the expansion data')
    reject(lambda current: current['stage2'].__setitem__('authorized', True),
           'Stage 2 is not authorized')

    def overlap(current):
        current['splits']['confirmation'] = list(current['splits']['confirmation'])
        current['splits']['confirmation'][0] = UNIQUE_ADDED_HISTORY[0]

    reject(overlap, 'overlaps burned')


def tiny_parent():
    import torch
    torch.manual_seed(20260921)
    return SwiGLUExpert(8, 16)


def development_rows(current, successes, seconds=0.2, memory=1024):
    flags = [i < successes for i in range(current['counts']['development'])]
    return [
        {'task_id': task_id, 'passed': passed, 'seconds': seconds, 'peak_memory_bytes': memory,
         'active_experts_per_token': 1}
        for task_id, passed in zip(current['splits']['development'], flags)
    ]


def test_method_does_not_import_core_moe():
    text = (ROOT / 'src/neuroshard/evolution/learned_integration.py').read_text()
    assert 'from neuroshard.core.model.moe' not in text
    assert 'import neuroshard.core.model.moe' not in text


def test_expansion_matches_parent_and_keeps_one_active_expert():
    import torch
    parent = tiny_parent()
    expansion = LastLayerMixture.expand(parent)
    expansion.eval()
    hidden = torch.randn(2, 5, 8)
    assert torch.allclose(expansion(hidden), parent(hidden), atol=1e-5)
    assert int((expansion.last_choices == 0).all())
    assert active_mlp_flops_per_token(8, 16) == 6 * 8 * 16
    with pytest.raises(ValueError, match='Active expert count'):
        active_mlp_flops_per_token(8, 16, active=2)


def test_frozen_incumbent_does_not_move_while_added_expert_learns():
    import torch
    parent = tiny_parent()
    expansion = LastLayerMixture.expand(parent)
    with torch.no_grad():
        expansion.router.bias.copy_(torch.tensor([-5.0, 5.0]))
    expansion.eval()
    hidden = torch.randn(2, 4, 8)
    start = expansion(hidden).detach()
    incumbent = expansion.experts[0].gate_proj.weight.detach().clone()
    expansion.train()
    optimizer = torch.optim.AdamW(expansion.expansion_parameters(), lr=0.05)
    target = start + 0.35
    for _ in range(12):
        train_step(expansion, hidden, target, optimizer)
    assert torch.equal(expansion.experts[0].gate_proj.weight, incumbent)
    expansion.eval()
    assert not torch.allclose(expansion(hidden), start, atol=1e-4)
    assert int((expansion.last_choices == 1).all())
    assert set(id(p) for p in expansion.incumbent_parameters()).isdisjoint(
        id(p) for p in expansion.expansion_parameters())


def test_matched_training_uses_the_same_batches():
    import torch
    parent = tiny_parent()
    expansion = LastLayerMixture.expand(parent)
    control = control_from_parent(parent)
    hidden = torch.randn(2, 3, 8)
    target = parent(hidden).detach() + 0.1
    history = train_matched(expansion, control, [(hidden, target)], steps=4, learning_rate=0.01)
    assert [row['step'] for row in history] == [0, 1, 2, 3]
    assert all('expansion_loss' in row and 'control_loss' in row for row in history)


def test_training_schedule_is_deterministic_and_closed():
    current = spec()
    first = training_schedule(current)
    second = training_schedule(current)
    assert first == second
    assert len(first['schedule']) == current['training']['steps']
    assert first['schedule'] == second['schedule']
    documents = sum(len(batch) for batch in first['batches'])
    assert documents == current['counts']['train_new'] + current['counts']['train_replay']


def test_score_gate_requires_generated_gain_not_loss():
    current = spec()
    general = [{'parent': 'a' + str(i), 'expansion': 'a' + str(i)} for i in range(8)]
    result = score_gate(
        development_rows(current, 1), development_rows(current, 1), development_rows(current, 3),
        current, role='development', general=general)
    assert result['passed'] is True
    assert result['expansion_code'] - result['control_code'] >= 1
    assert result['gates']['loss_alone_is_success'] is False
    assert result['confirmation_opened'] is False
    with pytest.raises(ValueError, match='Confirmation remains closed'):
        score_gate(
            development_rows(current, 1), development_rows(current, 1), development_rows(current, 3),
            current, role='confirmation', general=general)
    with pytest.raises(ValueError, match='later execution freeze'):
        score_gate(
            development_rows(current, 1), development_rows(current, 1), development_rows(current, 3),
            current, role='development', general=None)


def test_development_gate_needs_code_retention_and_still_closes_confirmation():
    current = spec()
    general = [{'parent': 'keep', 'expansion': 'keep'} for _ in range(8)]
    code = score_gate(
        development_rows(current, 1), development_rows(current, 1), development_rows(current, 2),
        current, role='development', general=general)
    retention_ids = current['splits']['retention']
    parent = [{'task_id': n, 'passed': i < 2, 'seconds': 0.2, 'peak_memory_bytes': 1024,
               'active_experts_per_token': 1} for i, n in enumerate(retention_ids)]
    kept = [{'task_id': n, 'passed': i < 2, 'seconds': 0.2, 'peak_memory_bytes': 1024,
             'active_experts_per_token': 1} for i, n in enumerate(retention_ids)]
    lost = [{'task_id': n, 'passed': False, 'seconds': 0.2, 'peak_memory_bytes': 1024,
             'active_experts_per_token': 1} for n in retention_ids]
    passed = complete_development_gate(code, score_retention(parent, kept, current), current)
    assert passed['passed'] is True
    assert passed['confirmation_opened'] is False
    assert passed['next'] == 'confirmation-execution-freeze-eligible'
    failed = complete_development_gate(code, score_retention(parent, lost, current), current)
    assert failed['passed'] is False
    assert failed['next'] == 'stop'


def test_extractable_wrong_program_is_not_a_success():
    def check(code, setup, tests):
        return {'passed': False, 'status': 'execution-error'}

    assert generated_passed('```python\npass\n```', '', ['assert True'], check) is False
    assert generated_passed('not python', '', ['assert True'], check) is False


def test_method_freeze_refuses_launch():
    current = spec()
    freeze = method_freeze()
    saved = json.loads((ROOT / 'config/experiments/learned-integration-method.json').read_text())
    assert freeze['format'] == METHOD_FORMAT
    assert freeze['gpu_launch_authorized'] is False
    assert freeze['train'] is False
    assert saved == freeze
    assert identity(saved) == 'dc76e6ad287131144d1f5060224c9181ad3248186876429a1e25c5c2098233a5'
    digest = bind_method_freeze(freeze, current)
    assert digest == identity(saved)
    with pytest.raises(ValueError, match='does not authorize training or a GPU launch'):
        refuse_launch(current, freeze)
    for name, digest in freeze['files'].items():
        assert sha256(ROOT / name) == digest
