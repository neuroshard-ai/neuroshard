import copy
import hashlib
import json
from pathlib import Path
import subprocess
from types import SimpleNamespace

import pytest
import torch
from transformers import LlamaConfig

from neuroshard.evolution import balanced as b, consolidation, grounded_tasks as tasks, reasoned
from neuroshard.evolution.reference_data import identity, sha256
from neuroshard.evolution.sharded import balanced_job as driver, guarded, portable


def test_answer_normalization_equalizes_short_and_long_response_influence():
    logits = torch.zeros(17, 2, requires_grad=True)
    labels = torch.zeros(17, dtype=torch.long)
    weights = torch.tensor([1.] + [1. / 16] * 16)
    loss, _, _ = guarded.objective(logits, labels, weights, ce_denominator=2.)
    gradient, = torch.autograd.grad(loss, logits)
    torch.testing.assert_close(gradient[0], gradient[1:].sum(0), rtol=0, atol=0)
    old, _, _ = guarded.objective(logits, labels, torch.ones(17), ce_denominator=17.)
    old_gradient, = torch.autograd.grad(old, logits)
    torch.testing.assert_close(old_gradient[1:].sum(0), 16 * old_gradient[0])


def test_answer_weighting_and_reference_loss_are_additive_across_microbatches():
    torch.manual_seed(2)
    full = torch.randn(17, 3, requires_grad=True)
    split = full.detach().clone().requires_grad_(True)
    target = torch.arange(17) % 3
    weights = torch.tensor([1.] + [1. / 16] * 16)
    reference = torch.randn(17, 3)
    mask = torch.ones(17, dtype=torch.bool)
    args = dict(ce_denominator=2., kl_denominator=17., strength=2.)
    loss = guarded.objective(full, target, weights, reference, mask, **args)[0]
    parts = sum(guarded.objective(split[a:z], target[a:z], weights[a:z], reference[a:z], mask[a:z], **args)[0]
                for a, z in ((0, 1), (1, 6), (6, 17)))
    torch.testing.assert_close(loss, parts)
    torch.testing.assert_close(torch.autograd.grad(loss, full)[0], torch.autograd.grad(parts, split)[0])


def test_frozen_gate_and_single_pass_strata_do_not_depend_on_json_key_order():
    plan = b.load()
    schedule = b.schedule(plan)
    assert len(schedule) == 64
    assert sorted(i for entry in schedule for i in entry['indices']) == list(range(4096))
    kinds = [kind for kind in b.STRATA for _ in range(plan['strata'][kind])]
    for entry in schedule:
        assert {kind: sum(kinds[i] == kind for i in entry['indices']) for kind in b.STRATA} == plan['batch_strata']
    reordered = copy.deepcopy(plan)
    for key in ('strata', 'batch_strata'):
        reordered[key] = dict(reversed(list(reordered[key].items())))
    assert b.validate(reordered) and b.schedule(reordered) == schedule
    reordered['quality_gate']['min_correct_totals'] = 1
    with pytest.raises(ValueError, match='frozen'):
        b.validate(reordered)


def test_training_and_development_cover_all_previously_public_wordings():
    plan = b.load()
    for family in tasks.FAMILIES:
        values = [b.case(plan, 'dev-new', i) for i in range(16)]
        assert {v['variant'] for v in values if v['family'] == family} == {0, 1, 2, 3}
    for kind in b.STRATA[:3]:
        assert {b.case(plan, 'train', i, kind)['variant'] for i in range(4)} == {0, 1, 2, 3}


def write_rows(path, rows):
    path.write_text(''.join(json.dumps(row) + '\n' for row in rows))
    return {'file': path.name, 'sha256': sha256(path), 'count': len(rows), 'ids': [r['id'] for r in rows]}


def test_replay_requires_exact_trained_content_and_inverse_target_weights(tmp_path, monkeypatch):
    plan = b.load()
    plan['strata'] = {kind: 1 for kind in b.STRATA}
    encoded = {'input_ids': [1, 2], 'labels': [-100, 2], 'targets': 1}
    monkeypatch.setattr(b.data, 'conversation', lambda *args: copy.deepcopy(encoded))
    old = []
    for i, distill in ((2, False), (6, True)):
        case = tasks.make_case(8, 'train', i)
        old.append({'id': tasks.task_identity(case), 'task': case,
                    'messages': reasoned.messages(case, not distill), 'distill': distill, **encoded})
    old.append({'id': 'a' * 64, 'messages': [{'role': 'user', 'content': 'Prior conversation'},
                                          {'role': 'assistant', 'content': 'Recorded answer'}], **encoded})
    source = write_rows(tmp_path / 'trained-source.jsonl', old)
    monkeypatch.setattr(b, 'replay_source', lambda _: {'roles': {'train': source}})
    chosen = b.choose_replay(plan, old)
    rows = []
    for kind in b.STRATA:
        if kind.startswith('new_'):
            case = b.case(plan, 'train', 0, kind)
            row = {'id': tasks.task_identity(case), 'task': case, 'messages': reasoned.messages(case, False), **encoded}
        else:
            index = chosen[kind][0]
            row = {**old[index], 'source_index': index}
        rows.append({**row, 'stratum': kind, 'loss_weight': 1., 'distill': True})
    prepared = {'roles': {'train': write_rows(tmp_path / 'train.jsonl', rows)},
                'replay_source': source, 'replay_indices': chosen}
    assert b.read_role(plan, tmp_path, prepared, 'train', object()) == rows
    bad = copy.deepcopy(rows)
    bad[-1]['messages'][-1]['content'] = 'Substituted evaluation answer'
    prepared['roles']['train'] = write_rows(tmp_path / 'train.jsonl', bad)
    with pytest.raises(ValueError, match='unchanged proven-trained'):
        b.read_role(plan, tmp_path, prepared, 'train', object())
    bad = copy.deepcopy(rows)
    bad[0]['loss_weight'] = 8.
    prepared['roles']['train'] = write_rows(tmp_path / 'train.jsonl', bad)
    with pytest.raises(ValueError, match='answer weight'):
        b.read_role(plan, tmp_path, prepared, 'train', object())
    bad = copy.deepcopy(rows)
    bad[0]['labels'][0] = 1
    prepared['roles']['train'] = write_rows(tmp_path / 'train.jsonl', bad)
    with pytest.raises(ValueError, match='target mask'):
        b.read_role(plan, tmp_path, prepared, 'train', object())


def development_fixture(plan):
    prepared, baseline, candidate = {'roles': {}}, {}, {}
    for role in b.DEVELOPMENT:
        if role == 'dev-retention':
            losses = [{'id': hashlib.sha256(f'conversation:{i}'.encode()).hexdigest(), 'loss': 1., 'targets': 7}
                      for i in range(plan['roles'][role])]
            baseline[role] = candidate[role] = {'losses': losses, 'answers': []}
            prepared['roles'][role] = {'ids': [v['id'] for v in losses]}
            continue
        before, after = [], []
        for i in range(plan['roles'][role]):
            case = b.case(plan, role, i)
            correct = reasoned.messages(case, role == 'dev-new')[-1]['content']
            def answer(text):
                return {'id': tasks.task_identity(case), 'text': text, 'output_ids': [],
                        'check': consolidation.check_answer(plan, case, text, role)}
            before.append(answer('{}' if role == 'dev-new' and case['family'] == 'total' else correct))
            after.append(answer(correct))
        baseline[role], candidate[role] = {'answers': before, 'losses': []}, {'answers': after, 'losses': []}
        prepared['roles'][role] = {'ids': [r['id'] for r in before]}
    return prepared, baseline, copy.deepcopy(candidate['dev-new']), candidate


def test_development_requires_both_prior_answers_and_the_learned_math_skill():
    plan = b.load()
    prepared, baseline, parent, candidate = development_fixture(plan)
    assert b.development(plan, prepared, baseline, parent, candidate)['passed']
    for role, index, failed in (('dev-prior', 3, 'prior_preserved'), ('dev-new', 2, 'learned_skill_retained')):
        wrong = copy.deepcopy(candidate)
        row = wrong[role]['answers'][index]
        row['text'] = '{}'
        row['check'] = consolidation.check_answer(plan, b.case(plan, role, index), '{}', role)
        decision = b.development(plan, prepared, baseline, parent, wrong)
        assert not decision['passed'] and not decision['checks'][failed]
    wrong = copy.deepcopy(candidate)
    case = b.case(plan, 'dev-prior', 3)
    text = json.dumps({'top_two': tasks.expected(case)['ids']})
    assert not consolidation.check_answer(plan, case, text, 'dev-prior')['correct']
    with pytest.raises(ValueError, match='generated tokens'):
        b.generation(plan, prepared, 'dev-new', baseline['dev-new']['answers'], candidate['dev-new']['answers'],
                     SimpleNamespace(decode=lambda *args, **kwargs: 'fabricated'))


def small_checkpoint():
    config = LlamaConfig(vocab_size=16, hidden_size=8, intermediate_size=16,
                         num_hidden_layers=2, num_attention_heads=2, num_key_value_heads=1, tie_word_embeddings=True)
    value = {'format': portable.FORMAT, 'job': 'a' * 64, 'step': 384, 'config': portable.configuration(config),
             'optimizer': [{'lr': .001}], 'boundaries': [0, 1, 2], 'shards': ['b' * 64, 'c' * 64],
             'parent': 'd' * 64, 'transition': None,
             'tensors': {name: {'sha256': 'e' * 64, 'bytes': 1, 'shape': shape, 'born': 0, 'group': 0}
                         for name, shape in portable.shapes(config).items()}}
    value['state_root'] = portable.learned_root(value)
    return value


def test_selection_binds_both_training_windows_and_the_actual_endpoint():
    plan = b.load()
    parent = small_checkpoint()
    plan['parent']['checkpoint'], plan['boundaries'] = identity(parent), parent['boundaries']
    prepared, baseline, teacher, candidate = development_fixture(plan)
    checkpoints, previous = [], parent
    for step in plan['checkpoints']:
        value = {**copy.deepcopy(parent), 'job': b.job(prepared), 'step': step, 'parent': identity(previous)}
        value['state_root'] = portable.learned_root(value)
        checkpoints.append(value)
        previous = value
    selection = {'format': b.SELECTION, 'prepared': identity(prepared), 'parent': identity(parent),
        'baseline': plan['baseline']['checkpoint'], 'input_checkpoint': parent, 'checkpoints': checkpoints,
        'candidate': identity(previous), 'baseline_development': baseline, 'training_parent_new': teacher,
        'candidate_development': candidate, 'decision': b.development(plan, prepared, baseline, teacher, candidate)}
    assert b.validate_selection(plan, prepared, selection)
    wrong = copy.deepcopy(selection)
    wrong['candidate'] = 'f' * 64
    with pytest.raises(ValueError, match='actual passing terminal'):
        b.validate_selection(plan, prepared, wrong)
    wrong = copy.deepcopy(selection)
    wrong['checkpoints'] = [wrong['checkpoints'][-1]]
    with pytest.raises(ValueError, match='every declared checkpoint'):
        b.validate_selection(plan, prepared, wrong)


def test_gpu_preflight_rejects_uncommitted_work_before_cuda(monkeypatch):
    plan = b.load()
    plan['status'] = 'plan-frozen'
    monkeypatch.setattr(driver.reference, 'configure', lambda *args: pytest.fail('CUDA must not be reached'))
    with pytest.raises(ValueError, match='committed prepared inputs'):
        driver.run_gpu(SimpleNamespace(command='train'), plan, {})


def test_final_gain_cannot_hide_collapse_below_the_absolute_arithmetic_floor(tmp_path, monkeypatch):
    plan = b.load()
    prepared = {'roles': {}}
    for role in b.FINALS:
        ids = ([tasks.task_identity(b.case(plan, role, i)) for i in range(plan['roles'][role])]
               if role != 'retention' else [hashlib.sha256(f'final-retention:{i}'.encode()).hexdigest()
                                            for i in range(plan['roles'][role])])
        prepared['roles'][role] = {'ids': ids}
    selection = {'baseline': 'a' * 64, 'candidate': 'b' * 64}
    (tmp_path / 'selection.json').write_text(json.dumps(selection))
    monkeypatch.setattr(b, 'committed_prepared', lambda *args: True)
    monkeypatch.setattr(b, 'committed_selection', lambda *args: True)
    monkeypatch.setattr(b, 'validate_selection', lambda *args: True)
    tokens = {}
    monkeypatch.setattr(driver, 'tokenizer_for', lambda *args: SimpleNamespace(
        decode=lambda ids, **kwargs: tokens[ids[0]]))

    def report(checkpoint, correct_totals):
        outcomes = {}
        for role in b.FINALS:
            losses = [{'id': key, 'targets': 1, 'loss': 1.} for key in prepared['roles'][role]['ids']]
            answers = []
            if role != 'retention':
                for i, key in enumerate(prepared['roles'][role]['ids']):
                    case = b.case(plan, role, i)
                    text = reasoned.messages(case, role == 'test-new')[-1]['content']
                    if role == 'test-new' and case['family'] == 'total' and i // 4 >= correct_totals:
                        text = '{}'
                    identifier = len(tokens) + 1
                    tokens[identifier] = text
                    answers.append({'id': key, 'text': text, 'output_ids': [identifier],
                                    'check': consolidation.check_answer(plan, case, text, role)})
            outcomes[role] = {'losses': losses, 'answers': answers}
        return {'checkpoint': checkpoint, 'prepared': identity(prepared), 'outcomes': outcomes}

    (tmp_path / 'baseline.json').write_text(json.dumps(report(selection['baseline'], 0)))
    args = SimpleNamespace(selection=tmp_path / 'selection.json', seed=tmp_path,
                           baseline=tmp_path / 'baseline.json', candidate=tmp_path / 'candidate.json',
                           output=tmp_path / 'quality.json')
    for count, passed in ((63, False), (64, True)):
        args.candidate.write_text(json.dumps(report(selection['candidate'], count)))
        driver.score_final(args, plan, prepared)
        result = json.loads(args.output.read_bytes())
        assert result['passed'] is passed
        assert result['outcomes']['test-new']['generation']['wins'] == count


def test_real_git_freeze_rejects_source_changes_and_untrained_replay(tmp_path, monkeypatch):
    original = b.ROOT
    plan = b.load()
    plan['status'] = 'plan-frozen'
    files = [*b.SOURCES, b.PLAN, *[name for key, name in plan['exclusions'].items() if key != 'policy']]
    for name in files:
        target = tmp_path / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes((original / name).read_bytes())
    (tmp_path / b.PLAN).write_text(json.dumps(plan))
    def git(*args):
        return subprocess.check_output(['git', '-C', str(tmp_path), *args], stderr=subprocess.PIPE, text=True).strip()
    git('init', '-q')
    git('config', 'user.name', 'Test fixture')
    git('config', 'user.email', 'fixture@example.invalid')
    git('add', '.')
    git('commit', '-qm', 'Freeze numerical sources')
    revision = git('rev-parse', 'HEAD')
    monkeypatch.setattr(b, 'ROOT', tmp_path)
    old = b.replay_source(plan)
    prepared = {'format': b.PREPARED, 'plan_commit': revision, 'plan_digest': sha256(tmp_path / b.PLAN),
        'sources': b.sources(), 'schedule': b.schedule(plan), 'roles': {},
        'replay_source': {'file': 'trained-source.jsonl', 'sha256': old['roles']['train']['sha256'], 'prepared': identity(old)},
        'replay_indices': {}, 'excluded_prepared': {key: identity(json.loads((tmp_path / name).read_bytes()))
                                                  for key, name in plan['exclusions'].items() if key != 'policy'}}
    offset, replay_ids = 0, []
    for kind in b.STRATA[3:]:
        count = plan['strata'][kind]
        prepared['replay_indices'][kind] = list(range(offset, offset + count))
        replay_ids.extend(old['roles']['train']['ids'][offset:offset + count])
        offset += count
    for role, count in plan['roles'].items():
        ids = [hashlib.sha256(f'new-balanced-{role}-{i}'.encode()).hexdigest() for i in range(count)]
        if role == 'train':
            ids[-len(replay_ids):] = replay_ids
        prepared['roles'][role] = {'file': role + '.jsonl', 'sha256': '0' * 64, 'count': count, 'ids': ids}
    plan['status'] = 'prepared-committed'
    (tmp_path / b.PLAN).write_text(json.dumps(plan))
    b.path(plan, 'prepared').write_text(json.dumps(prepared))
    git('add', '.')
    git('commit', '-qm', 'Bind prepared inputs')
    assert b.committed_prepared(plan, prepared)
    source = tmp_path / b.SOURCES[-1]
    raw = source.read_bytes()
    source.write_bytes(raw + b'\n# changed numerical execution\n')
    with pytest.raises(ValueError, match='frozen balanced sources'):
        b.committed_prepared(plan, prepared)
    source.write_bytes(raw)
    wrong = copy.deepcopy(prepared)
    wrong['replay_indices']['replay_reasoned'][0] = old['roles']['train']['count']
    b.path(plan, 'prepared').write_text(json.dumps(wrong))
    git('add', '.')
    git('commit', '-qm', 'Attempt an untrained replay index')
    with pytest.raises(ValueError, match='outside trained coverage'):
        b.committed_prepared(plan, wrong)
