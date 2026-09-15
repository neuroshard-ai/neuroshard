import copy
import json
from pathlib import Path
import subprocess
from types import SimpleNamespace

import pytest

from neuroshard.evolution import grounded_tasks, incremental_capacity as contract, incremental_facts as facts
from neuroshard.evolution import reference_data as data


ROOT = Path(__file__).resolve().parents[2]


def plan():
    return json.loads((ROOT / 'config/experiments/incremental-capacity.json').read_bytes())


def test_schedule_uses_all_fresh_rows_and_only_declared_trained_replay():
    frozen = plan()
    rows = [{**row, 'input_ids': [1] * (len(row['messages'][0]['content']) % 40 + 8)}
            for row in facts.raw_examples(frozen['seeds']['facts'], 0, 'train')]
    old = []
    for family in ('sort', 'lookup', 'filter', 'total', 'conversation'):
        for index in range(512):
            old.append({'id': data.identity([family, index]), 'input_ids': [1] * (index % 40 + 8),
                        **({'task': {'family': family}} if family != 'conversation' else {})})
    selected = contract.replay(frozen, old)
    for family, indices in selected.items():
        rows.extend({**old[index], 'stratum': 'replay-conversation' if family == 'conversation' else 'replay-skill'}
                    for index in indices)
    batches = contract.schedule(frozen, rows)
    assert len(batches) == 128 and all(len(batch) == 64 and len(set(batch)) == 64 for batch in batches)
    used = [index for batch in batches for index in batch]
    assert set(used) == set(range(len(rows)))
    for index, row in enumerate(rows):
        assert used.count(index) in ((1, 2) if row['stratum'] == 'replay-conversation' else (1,))
    for batch in batches:
        assert [rows[i]['input_ids'].__len__() for i in batch] == sorted(len(rows[i]['input_ids']) for i in batch)
        assert all(sum(rows[i].get('task', {}).get('family') == family for i in batch) == 3
                   for family in ('sort', 'lookup', 'filter', 'total'))
    with pytest.raises(ValueError, match='strata'):
        contract.schedule(frozen, rows[:-1])
    with pytest.raises(ValueError, match='proven-trained'):
        contract.replay(frozen, old[:500])


def evaluation_fixture():
    frozen = plan()
    knowledge = facts.raw_examples(frozen['seeds']['facts'], 0, 'dev')
    task = grounded_tasks.make_case(700, 'retention', 1, family='lookup')
    skill = {'id': data.identity(task), 'task': task}
    rows = {'dev-knowledge': knowledge, 'dev-skills': [skill],
            'dev-conversation': [{'id': 'a', 'targets': 2}, {'id': 'b', 'targets': 3}]}
    answer = json.dumps(grounded_tasks.expected(task))
    before = {'dev-knowledge': {'answers': [{'id': row['id'], 'text': '{"answer":"unknown"}',
                                            'check': {'correct': True}} for row in knowledge]},
              'dev-skills': {'answers': [{'id': skill['id'], 'text': answer}]},
              'dev-conversation': {'losses': [{'id': row['id'], 'loss': 2., 'targets': row['targets']}
                                              for row in rows['dev-conversation']]}}
    after = copy.deepcopy(before)
    after['dev-knowledge']['answers'] = [{'id': row['id'], 'text': row['messages'][-1]['content'],
                                         'check': {'correct': False}} for row in knowledge]
    return frozen, rows, before, after


def test_generated_knowledge_and_retention_are_both_required():
    frozen, rows, before, after = evaluation_fixture()
    good = contract.decision(frozen, rows, before, after, True)
    assert good['passed'] and good['knowledge']['accuracy'] == 1
    assert good['knowledge']['entity_cluster_delta']['pairs'] == 32
    assert not contract.decision(frozen, rows, before, before, True)['passed']
    failed = copy.deepcopy(after)
    failed['dev-skills']['answers'][0]['text'] = '{}'
    assert not contract.decision(frozen, rows, before, failed, True)['checks']['prior_correct_answers_retained']
    failed = copy.deepcopy(after)
    for row in failed['dev-conversation']['losses']:
        row['loss'] += .03
    assert not contract.decision(frozen, rows, before, failed, True)['checks']['conversation_retention']
    failed = copy.deepcopy(after)
    failed['dev-knowledge']['answers'].pop()
    with pytest.raises(ValueError, match='ordered cases'):
        contract.decision(frozen, rows, before, failed, True)


def test_knowledge_statistics_use_entities_not_correlated_questions():
    frozen = plan()
    rows = facts.raw_examples(frozen['seeds']['facts'], 0, 'test')
    before = [{'id': row['id'], 'text': '{"answer":"unknown"}'} for row in rows]
    after = [{'id': row['id'], 'text': row['messages'][-1]['content'] if row['task']['entity'] % 2 else
              '{"answer":"unknown"}'} for row in rows]
    result = contract.knowledge(frozen, rows, before, after)
    assert result['count'] == 1024 and result['entity_cluster_delta']['pairs'] == 128
    assert result['entity_cluster_delta']['mean'] == .5
    assert .39 < result['entity_cluster_delta']['lower'] < .45
    with pytest.raises(ValueError, match='eight-question'):
        contract.knowledge(frozen, rows[:-1], before[:-1], after[:-1])


def test_generated_text_cannot_be_replaced_or_stop_early():
    class Tokenizer:
        eos_token_id = 0

        def __len__(self):
            return 2

        def decode(self, ids, skip_special_tokens):
            return 'answer' * ids.count(1)

    rows = {'test-knowledge': [{'id': 'a'}]}
    output = {'test-knowledge': {'losses': [], 'answers': [{'id': 'a', 'output_ids': [1, 0], 'text': 'answer'}]}}
    contract.validate_generation(plan(), rows, output, Tokenizer())
    for ids, text in (([1], 'answer'), ([1, 0], 'better answer'), ([1, 0, 1], 'answeranswer'), ([True, 0], 'answer')):
        bad = copy.deepcopy(output)
        bad['test-knowledge']['answers'][0].update(output_ids=ids, text=text)
        with pytest.raises(ValueError, match='stopping rule'):
            contract.validate_generation(plan(), rows, bad, Tokenizer())


def test_plan_source_and_prepared_inputs_must_be_committed(tmp_path, monkeypatch):
    monkeypatch.setattr(contract, 'ROOT', tmp_path)
    monkeypatch.setattr(contract, 'SOURCES', ('kernel.py',))
    path = tmp_path / 'plan.json'
    data.save(path, plan())
    (tmp_path / 'kernel.py').write_text('x = 1\n')

    def git(*args):
        return subprocess.check_output(['git', '-C', str(tmp_path), *args], stderr=subprocess.PIPE).decode().strip()

    git('init', '-q')
    git('config', 'user.name', 'Test')
    git('config', 'user.email', 'test@example.invalid')
    with pytest.raises(ValueError, match='not committed'):
        contract.frozen_plan(path)
    git('add', '.')
    git('commit', '-qm', 'Freeze comparison')
    revision = git('rev-parse', 'HEAD')
    assert contract.frozen_plan(path) == plan()
    prepared_path = tmp_path / 'prepared.json'
    prepared = {'format': contract.FORMAT + '/prepared', 'plan_commit': revision,
        'plan_sha256': data.sha256(path), 'sources': contract.sources(),
        'roles': {role: {} for role in ('train', *contract.DEVELOPMENT, *contract.FINALS)}}
    data.save(prepared_path, prepared)
    with pytest.raises(ValueError, match='not committed'):
        contract.validate_prepared(path, prepared_path)
    git('add', '.')
    git('commit', '-qm', 'Commit prepared inputs')
    assert contract.validate_prepared(path, prepared_path)[1] == prepared
    (tmp_path / 'kernel.py').write_text('x = 2\n')
    with pytest.raises(ValueError, match='differs'):
        contract.validate_prepared(path, prepared_path)


def test_training_closes_before_loading_a_model_once_selection_exists(tmp_path, monkeypatch):
    from neuroshard.evolution.sharded import incremental_job
    frozen = plan()
    monkeypatch.setattr(contract, 'ROOT', tmp_path)
    monkeypatch.setattr(contract, 'validate_prepared', lambda *args: (frozen, {}))
    data.save(tmp_path / frozen['selection_path'], {'selected': {}})
    args = SimpleNamespace(command='train', arm='append', plan=None, prepared=None)
    with pytest.raises(ValueError, match='Development is closed'):
        incremental_job.run(args)
