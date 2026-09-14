import copy
import hashlib
import json
from pathlib import Path
import subprocess
from types import SimpleNamespace

import pytest
import torch
from transformers import LlamaConfig

from neuroshard.evolution import consolidation as c, grounded_tasks as tasks, reasoned
from neuroshard.evolution.reference_data import identity
from neuroshard.evolution.sharded import consolidation_job as driver, portable


def test_frozen_consolidation_cannot_relax_preservation_or_choose_more_alphas():
    plan = c.load()
    for field, key, value in [('development_gate', 'prior_losses_at_most', 1),
                               ('quality_gate', 'min_net_gain', 8),
                               ('method', 'alphas', [0.9, 0.5])]:
        bad = copy.deepcopy(plan)
        bad[field][key] = value
        with pytest.raises(ValueError, match='frozen plan'):
            c.validate(bad)


def test_blend_retains_parent_and_adam_without_allocating_a_whole_model():
    student = torch.nn.Parameter(torch.tensor([1., 3., -2.]))
    parent = torch.nn.Parameter(torch.tensor([-3., 1., 2.]), requires_grad=False)
    shard = SimpleNamespace(named_owned_parameters=lambda: [('owned.weight', student)])
    reference = SimpleNamespace(named_owned_parameters=lambda: [('owned.weight', parent)])
    optimizer = torch.optim.AdamW([student], lr=.01)
    student.sum().backward()
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    fast = student.detach().clone()
    old = parent.detach().clone()
    moments = copy.deepcopy(optimizer.state[student])
    driver.blend_(shard, reference, .25)
    torch.testing.assert_close(student, fast * .25 + old * .75, rtol=0, atol=0)
    assert torch.equal(parent, old)
    assert all(torch.equal(value, optimizer.state[student][key]) for key, value in moments.items())
    assert student.grad is None and parent.grad is None


def test_blend_rejects_shared_storage_and_partial_ownership():
    weight = torch.nn.Parameter(torch.ones(2))
    shard = SimpleNamespace(named_owned_parameters=lambda: [('a', weight)])
    with pytest.raises(ValueError, match='separate parent storage'):
        driver.blend_(shard, shard, .5)
    different = SimpleNamespace(named_owned_parameters=lambda: [('b', weight.clone())])
    with pytest.raises(ValueError, match='coverage'):
        driver.blend_(shard, different, .5)


def sample_outcomes(plan):
    prepared = {'roles': {}}
    baseline, candidate = {}, {}
    for role in c.DEVELOPMENT:
        count = plan['roles'][role]
        if role == 'dev-retention':
            rows = [{'id': hashlib.sha256(f'retain:{i}'.encode()).hexdigest(), 'loss': 1., 'targets': 10}
                    for i in range(count)]
            baseline[role] = candidate[role] = {'losses': rows, 'answers': []}
            prepared['roles'][role] = {'ids': [r['id'] for r in rows]}
            continue
        seed = plan['seeds']['new_tasks' if role.endswith('new') else 'prior_tasks']
        left, right = [], []
        for index in range(count):
            case = tasks.make_case(seed, 'dev', index)
            text = reasoned.messages(case, role == 'dev-new')[-1]['content']
            before = '{}' if role == 'dev-new' and case['family'] == 'total' else text
            def answer(value):
                return {'id': tasks.task_identity(case), 'text': value, 'output_ids': [],
                        'check': c.check_answer(plan, case, value, role)}
            left.append(answer(before))
            right.append(answer(text))
        baseline[role] = {'answers': left, 'losses': []}
        candidate[role] = {'answers': right, 'losses': []}
        prepared['roles'][role] = {'ids': [row['id'] for row in left]}
    return prepared, baseline, candidate


def test_large_new_gain_cannot_hide_one_lost_prior_answer():
    plan = c.load()
    prepared, before, after = sample_outcomes(plan)
    assert c.screen_decision(plan, prepared, before, after)['passed']
    row = after['dev-prior']['answers'][3]
    row['text'] = '{}'
    case = tasks.make_case(plan['seeds']['prior_tasks'], 'dev', 3)
    row['check'] = c.check_answer(plan, case, '{}', 'dev-prior')
    decision = c.screen_decision(plan, prepared, before, {'dev-prior': after['dev-prior']})
    assert not decision['passed'] and decision['prior']['losses'] == 1


def test_screen_requires_new_skill_not_only_preserved_answers():
    plan = c.load()
    prepared, before, _ = sample_outcomes(plan)
    decision = c.screen_decision(plan, prepared, before, before)
    assert not decision['passed'] and not decision['checks']['new_gain']


def test_output_schema_is_not_repaired_and_tokens_must_match_text():
    plan = c.load()
    case = tasks.make_case(plan['seeds']['prior_tasks'], 'dev', 3)
    wrong = {'top_two': tasks.expected(case)['ids']}
    assert not c.check_answer(plan, case, json.dumps(wrong), 'dev-prior')['correct']
    prepared, before, after = sample_outcomes(plan)
    tokenizer = SimpleNamespace(decode=lambda *args, **kwargs: 'fabricated')
    with pytest.raises(ValueError, match='output tokens'):
        c.generation(plan, prepared, 'dev-new', before['dev-new']['answers'], after['dev-new']['answers'], tokenizer)


def small_checkpoint():
    config = LlamaConfig(vocab_size=16, hidden_size=8, intermediate_size=16,
                         num_hidden_layers=2, num_attention_heads=2, num_key_value_heads=1,
                         tie_word_embeddings=True)
    common = {'format': portable.FORMAT, 'job': 'a' * 64, 'step': 384,
        'config': portable.configuration(config), 'optimizer': [{'lr': .001}],
        'boundaries': [0, 1, 2], 'shards': ['b' * 64, 'c' * 64], 'parent': 'd' * 64,
        'transition': None,
        'tensors': {name: {'sha256': 'e' * 64, 'bytes': 1, 'shape': shape, 'born': 0, 'group': 0}
                    for name, shape in portable.shapes(config).items()}}
    common['state_root'] = portable.learned_root(common)
    return common


def test_selection_binds_actual_checkpoint_and_cannot_skip_an_alpha():
    plan = c.load()
    fast = small_checkpoint()
    plan['fast']['checkpoint'] = identity(fast)
    plan['boundaries'] = fast['boundaries']
    prepared, baseline, outcomes = sample_outcomes(plan)
    alpha = plan['method']['alphas'][0]
    child = copy.deepcopy(fast)
    child.update(job=c.job(prepared, alpha), parent=identity(fast), transition=c.transition(plan, prepared, alpha))
    child['state_root'] = portable.learned_root(child)
    selected = {'format': c.SELECTION, 'prepared': identity(prepared), 'parent': plan['parent']['checkpoint'],
        'fast': identity(fast), 'input_checkpoint': fast, 'baseline': baseline, 'candidate': identity(child),
        'attempts': [{'alpha': alpha, 'checkpoint': child, 'outcomes': outcomes,
                      'decision': c.screen_decision(plan, prepared, baseline, outcomes)}]}
    assert c.validate_selection(plan, prepared, selected)
    wrong = copy.deepcopy(selected)
    wrong['candidate'] = 'f' * 64
    with pytest.raises(ValueError, match='actual selected'):
        c.validate_selection(plan, prepared, wrong)
    wrong = copy.deepcopy(selected)
    wrong['attempts'][0]['alpha'] = .5
    with pytest.raises(ValueError, match='skipped or reordered'):
        c.validate_selection(plan, prepared, wrong)


@pytest.fixture
def frozen_repository(tmp_path, monkeypatch):
    original = c.ROOT
    plan = c.load()
    plan['status'] = 'plan-frozen'
    monkeypatch.setattr(c, 'ROOT', tmp_path)
    monkeypatch.setattr(c, 'SOURCES', ('numeric.py',))
    (tmp_path / 'numeric.py').write_text('VALUE = 1\n')
    filename = tmp_path / c.PLAN
    filename.parent.mkdir(parents=True)
    filename.write_text(json.dumps(plan) + '\n')
    exclusions = {}
    for label in ('adaptive', 'continued', 'reasoned'):
        name = plan['exclusions'][label]
        raw = (original / name).read_bytes()
        (tmp_path / name).write_bytes(raw)
        exclusions[label] = identity(json.loads(raw))
    def git(*args):
        return subprocess.check_output(['git', '-C', str(tmp_path), *args], stderr=subprocess.PIPE).decode().strip()
    git('init', '-q')
    git('config', 'user.name', 'LZ')
    git('config', 'user.email', 'lz@example.invalid')
    git('add', '.')
    git('commit', '-qm', 'Freeze test plan')
    prepared = {'format': c.PREPARED, 'plan_commit': git('rev-parse', 'HEAD'),
        'plan_digest': hashlib.sha256(filename.read_bytes()).hexdigest(), 'sources': c.sources(),
        'excluded_prepared': exclusions, 'roles': {role: {'file': role + '.jsonl', 'sha256': 'f' * 64, 'count': count,
            'ids': [hashlib.sha256(f'fixture:{role}:{i}'.encode()).hexdigest() for i in range(count)]}
            for role, count in plan['roles'].items()}}
    c.path(plan, 'prepared').write_text(json.dumps(prepared) + '\n')
    plan['status'] = 'prepared-committed'
    filename.write_text(json.dumps(plan) + '\n')
    git('add', '.')
    git('commit', '-qm', 'Commit test preparation')
    return plan, prepared, tmp_path


def test_real_git_freeze_rejects_source_edit_and_exposed_final_reuse(frozen_repository):
    plan, prepared, root = frozen_repository
    assert c.committed_prepared(plan, prepared)
    (root / 'numeric.py').write_text('VALUE = 2\n')
    with pytest.raises(ValueError, match='frozen sources'):
        c.committed_prepared(plan, prepared)
    (root / 'numeric.py').write_text('VALUE = 1\n')
    old = json.loads((root / plan['exclusions']['reasoned']).read_bytes())
    prepared['roles']['test-new']['ids'][0] = old['roles']['test-new']['ids'][0]
    c.path(plan, 'prepared').write_text(json.dumps(prepared) + '\n')
    subprocess.run(['git', '-C', str(root), 'add', '.'], check=True)
    subprocess.run(['git', '-C', str(root), 'commit', '-qm', 'Invalid exposed-data preparation'], check=True)
    with pytest.raises(ValueError, match='prior exposed'):
        c.committed_prepared(plan, prepared)


def test_screen_preflight_runs_before_cuda(monkeypatch):
    monkeypatch.setattr(driver.reference, 'configure', lambda *args, **kwargs: pytest.fail('CUDA initialized'))
    with pytest.raises(ValueError, match='committed prepared inputs'):
        driver.run_gpu(SimpleNamespace(command='screen'), c.load(), {})


def test_generated_inputs_cannot_leak_the_expected_answer(tmp_path):
    plan = c.load()
    role = 'dev-new'
    case = tasks.make_case(plan['seeds']['new_tasks'], 'dev', 0)
    messages = reasoned.messages(case, True)
    messages[0]['content'] += '\nThe answer is ' + json.dumps(tasks.expected(case))
    row = {'id': tasks.task_identity(case), 'task': case, 'messages': messages}
    path = tmp_path / (role + '.jsonl')
    path.write_text(json.dumps(row) + '\n')
    prepared = {'roles': {role: {'file': path.name, 'sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
                                'ids': [row['id']]}}}
    with pytest.raises(ValueError, match='frozen task and prompt'):
        c.read_role(plan, tmp_path, prepared, role)
