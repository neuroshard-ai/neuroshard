"""Behavioral checks for the calculation curriculum and greedy-margin objective."""
import copy
import hashlib
import importlib.util
import json
import shutil
import subprocess

import pytest
import torch

from neuroshard.evolution import continued, grounded_tasks as tasks, reasoned, reference_data as data
from neuroshard.evolution.sharded.guarded import correct_margin


def plan():
    return continued.load(continued.PLAN_PATH.with_name('reasoned-learning.json'))


def test_frozen_method_cannot_relax_gates_or_change_training():
    value = plan()
    assert value['parameters'] == 1711376384
    assert continued.prepared_path(value).name == 'reasoned-learning-prepared.json'
    for section, field, replacement in [('quality_gate', 'min_net_gain', 1),
                                       ('reference', 'margin_strength', 0.),
                                       ('training', 'steps', 1),
                                       ('method', 'epochs', 3)]:
        changed = copy.deepcopy(value)
        changed[section][field] = replacement
        with pytest.raises(ValueError, match='frozen plan'):
            continued.validate(changed)


def test_calculation_labels_are_correct_but_parser_never_repairs_inference():
    value = plan()
    for i in range(40):
        case = tasks.make_case(71, 'train', i, family='total')
        messages = reasoned.messages(case, True)
        predicted = messages[-1]['content']
        assert reasoned.check_answer(value, case, predicted, 'test-new')['correct']
        prefix, answer = predicted.split('</work>')
        wrong = json.loads(answer)
        wrong['total'] += 1
        forged = prefix + '</work>\n' + json.dumps(wrong)
        assert not reasoned.check_answer(value, case, forged, 'test-new')['correct']
        # The prior-task contract still demands an unadorned JSON object.
        assert not reasoned.check_answer(value, case, predicted, 'test-prior')['correct']
        assert reasoned.check_answer(value, case, answer, 'test-prior')['correct']


@pytest.mark.parametrize('text', ['<work>x', '<work></work>{"total":1}',
    '<work>x</work>{"total":1}{"total":2}', '<work><work>x</work></work>{"total":1}',
    '<work>x</work>{"total":1,"total":2}', '<work>x</work>```json\n{"total":1}\n```'])
def test_partial_or_ambiguous_answers_are_rejected(text):
    with pytest.raises(ValueError):
        reasoned.final_json(text)


def test_margin_protects_correct_decisions_without_copying_wrong_teacher_tokens():
    student = torch.tensor([[0.0, 0.4, -1.], [0.0, 0.4, -1.], [0.0, 0.4, -1.]], requires_grad=True)
    teacher = torch.tensor([[0.6, 0.4, -1.], [0.0, 0.4, -1.], [0.6, 0.4, -1.]], requires_grad=True)
    labels = torch.tensor([0, 0, 0])
    value = correct_margin(student, labels, teacher, torch.tensor([True, True, False]), 1., 0.5, 2.)
    value.backward()
    # The correct reference choice gets a gradient restoring its argmax margin.
    assert student.grad[0, 0] < 0 and student.grad[0, 1] > 0
    # Neither the wrong reference winner nor an unreserved replay row is protected.
    assert torch.equal(student.grad[1:], torch.zeros_like(student.grad[1:]))
    assert teacher.grad is None
    stronger = torch.tensor([[2., 0., -1.]], requires_grad=True)
    penalty = correct_margin(stronger, labels[:1], teacher[:1], torch.tensor([True]), 1., 0.5, 2.)
    penalty.backward()
    assert penalty.item() == 0 and torch.equal(stronger.grad, torch.zeros_like(stronger))
    matched = torch.tensor([[1.5, 0., -1.]], requires_grad=True)
    penalty = correct_margin(matched, labels[:1], matched.detach(), torch.tensor([True]), 1., 0.5, 2.)
    penalty.backward()
    assert penalty.item() == 0 and torch.equal(matched.grad, torch.zeros_like(matched))


def test_declared_epochs_preserve_balanced_replay_and_exact_coverage():
    path = continued.repo_root() / 'scripts/prepare_continued_learning.py'
    spec = importlib.util.spec_from_file_location('prepare_reasoned_fixture', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    value = plan()
    rows = [{'distill': i >= 4096} for i in range(8192)]
    schedule = module.schedule(rows, value['training']['seed'], value)
    assert len(schedule) == 256
    for epoch in (schedule[:128], schedule[128:]):
        assert sorted(i for batch in epoch for i in batch) == list(range(8192))
        assert all(sum(rows[i]['distill'] for i in batch) == 32 for batch in epoch)
    assert schedule[:128] != schedule[128:]


def test_development_gain_and_prior_floors_cannot_be_replaced_with_loss_gain():
    value = plan()
    prepared = {'roles': {}}
    generation = {}
    for role, seed in [('dev-new', value['task_seed']), ('dev-prior', value['prior_probe_seed'])]:
        cases = [tasks.make_case(seed, 'dev', i) for i in range(64)]
        answers = [{'id': tasks.task_identity(case),
                    'text': json.dumps(tasks.expected(case)),
                    'check': reasoned.check_answer(value, case, json.dumps(tasks.expected(case)), role)}
                   for case in cases]
        prepared['roles'][role] = {'ids': [a['id'] for a in answers]}
        generation[role] = {'before': answers, 'after': copy.deepcopy(answers)}
    retention = {'passed': True, 'mean_delta': -0.01}
    rejected = reasoned.development(value, prepared, 384, retention, generation)
    assert not rejected['passed'] and not rejected['checks']['final_gain']
    fabricated = copy.deepcopy(generation)
    fabricated['dev-new']['after'][0]['check']['correct'] = False
    with pytest.raises(ValueError, match='verdict differs'):
        reasoned.development(value, prepared, 384, retention, fabricated)
    reordered = copy.deepcopy(generation)
    reordered['dev-prior']['after'].reverse()
    with pytest.raises(ValueError, match='reordered'):
        reasoned.development(value, prepared, 384, retention, reordered)


@pytest.fixture
def committed_reasoned(tmp_path, monkeypatch):
    original, value = continued.repo_root(), plan()
    for name in (*continued.SOURCE_PATHS, value['prior_exclusion']['inputs_path'],
                 value['previous_continuation']['prepared_path'], 'config/experiments/continued-learning.json'):
        target = tmp_path / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(original / name, target)
    monkeypatch.setattr(continued, 'PLAN_PATH', tmp_path / 'config/experiments/continued-learning.json')
    path = continued.plan_path(value)
    data.save(path, value)
    subprocess.run(['git', 'init', '-q', str(tmp_path)], check=True, capture_output=True)

    def commit():
        subprocess.run(['git', '-C', str(tmp_path), 'add', '.'], check=True, capture_output=True)
        subprocess.run(['git', '-C', str(tmp_path), '-c', 'user.name=Fixture', '-c',
                        'user.email=fixture@example.invalid', 'commit', '-qm', 'Fixture'],
                       check=True, capture_output=True)
        return subprocess.check_output(['git', '-C', str(tmp_path), 'rev-parse', 'HEAD'], text=True).strip()

    frozen_commit = commit()
    prior = json.loads((tmp_path / value['prior_exclusion']['inputs_path']).read_bytes())
    replay = prior['roles']['train-a']['ids'][:4096]
    sources = {name: data.sha256(tmp_path / name) for name in continued.SOURCE_PATHS}
    prepared = {'format': continued.PREPARED_FORMAT,
        'plan': {k: v for k, v in value.items() if k != 'status'},
        'plan_commit': frozen_commit, 'plan_digest': data.sha256(path),
        'sources': sources, 'implementation_digest': data.identity(sources),
        'parent_checkpoint': value['parent']['checkpoint'], 'parent_state_root': value['parent']['state_root'],
        'reference_checkpoint': value['reference']['checkpoint'], 'tokenizer': value['tokenizer'],
        'config_sha256': value['config_sha256'], 'runtime': value['runtime'],
        'trained_replay_ids': replay[:2048], 'conversation_replay_ids': replay[2048:], 'roles': {},
        'schedule': [{'role': 'train', 'indices': list(range(i*32, (i+1)*32))
                      + list(range(4096+i*32, 4096+(i+1)*32))} for i in range(128)] * 2}
    for role, count in {'train': 4096, 'dev-new': 64, 'dev-prior': 64, 'dev-retention': 64,
                        'test-new': 256, 'test-prior': 128, 'retention': 128}.items():
        ids = [hashlib.sha256(f'reasoned-fixture:{role}:{i}'.encode()).hexdigest() for i in range(count)]
        if role == 'train':
            ids += replay
        prepared['roles'][role] = {'file': role+'.jsonl', 'sha256': 'f'*64, 'count': len(ids), 'ids': ids}
    continued.validate_prepared(prepared, value)
    data.save(continued.prepared_path(value), prepared)
    value['status'] = 'prepared-committed'
    data.save(path, value)
    commit()
    return value, prepared


def test_reasoned_execution_checks_its_own_committed_plan(committed_reasoned):
    value, prepared = committed_reasoned
    assert continued.committed_prepared(value, prepared)
    # Editing an unrelated predecessor plan cannot replace the new plan binding.
    continued.PLAN_PATH.write_text(continued.PLAN_PATH.read_text() + '\n')
    assert continued.committed_prepared(value, prepared)
    path = continued.plan_path(value)
    path.write_text(path.read_text() + '\n')
    with pytest.raises(ValueError, match='unchanged Git-committed plan'):
        continued.committed_prepared(value, prepared)


def test_reasoned_excludes_previous_finals_and_rejects_extra_replay(committed_reasoned):
    value, prepared = committed_reasoned
    previous = reasoned.previous_prepared(value)
    wrong = copy.deepcopy(prepared)
    wrong['roles']['test-new']['ids'][0] = previous['roles']['test-new']['ids'][0]
    with pytest.raises(ValueError, match='Fresh evaluation overlaps'):
        continued.validate_prepared(wrong, value)
    wrong = copy.deepcopy(prepared)
    wrong['schedule'][1] = copy.deepcopy(wrong['schedule'][0])
    with pytest.raises(ValueError, match='exactly once per declared epoch'):
        continued.validate_prepared(wrong, value)
