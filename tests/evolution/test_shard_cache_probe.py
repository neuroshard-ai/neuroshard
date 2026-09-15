"""Answer retention and complete replay remain mandatory in a speed comparison."""
import json
from types import SimpleNamespace

import pytest

from neuroshard.evolution import balanced, consolidation, grounded_tasks
from neuroshard.evolution.reference_data import identity, save, sha256
from neuroshard.evolution.sharded import cache_probe


@pytest.fixture
def evidence(tmp_path, monkeypatch):
    plan = {'maximum_tokens': 2, 'scope': 'synthetic test', 'gate': {
        'individual_correct_answer_losses_at_most': 0,
        'identical_token_fraction_at_least': .95,
        'aggregate_speedup_on_identical_outputs_at_least': 2.,
        'boundary_byte_reduction_at_least': .9}}
    prepared = {'plan': plan, 'eos_id': 2}
    checkpoint = {'boundaries': [0, 1, 2], 'config': {'vocab_size': 32}}
    case = balanced.case(balanced.load(), 'test-new', 0)
    request = {'id': 'one', 'role': 'test-new', 'task': case}
    text = json.dumps(grounded_tasks.expected(case))
    check = consolidation.check_answer(balanced.load(), case, text, 'test-new')
    tokenizer = SimpleNamespace(decode=lambda ids, **kwargs: text if ids == [3, 2] else '{}')
    monkeypatch.setattr(cache_probe, 'checked', lambda args: (prepared, checkpoint, [request], tokenizer))
    monkeypatch.setattr(cache_probe, 'ROOT', tmp_path)
    save(tmp_path / 'config/experiments/balanced-continuation-results.json',
         {'answer_pairs': {'test-new': [{'id': 'one', 'candidate': text}]}})
    args = SimpleNamespace(reports=tmp_path / 'reports', audits=tmp_path / 'audits', home=tmp_path / 'result')
    for rank in range(2):
        folder = args.reports / f'rank-{rank}'
        folder.mkdir(parents=True)
        methods = {method: {'token_ids': [3, 2], 'text': text, 'check': check,
            'seconds': seconds, 'sent_tensor_bytes': size, 'transcript_root': 'root'}
            for method, seconds, size in [('uncached', 4., 10000), ('cached', 1., 100)]}
        (folder / 'comparison.jsonl').write_text(json.dumps({'id': 'one', 'index': 0, 'methods': methods}) + '\n')
        save(folder / 'complete.json', {'rank': rank, 'prepared': identity(prepared), 'requests': 1,
                                       'comparison_sha256': sha256(folder / 'comparison.jsonl')})
    for auditor in range(2):
        for rank in range(2):
            save(args.audits / f'auditor-{auditor}/rank-{rank}/replay.json', {
                'passed': True, 'rank': rank, 'prepared': identity(prepared),
                'requests': [{'id': 'one', 'transcript_root': 'root', 'token_ids': [3, 2]}],
                'seconds': 1., 'peak_cuda_bytes': 1000})
    return args


def test_actual_outputs_and_complete_replay_produce_the_speed_decision(evidence):
    result = cache_probe.score(evidence)
    assert result['passed'] and result['matched_output_speedup'] == 4
    assert result['correct'] == {'cached': 1, 'uncached': 1}
    assert result['boundary_byte_reduction'] == .99


def test_a_speed_result_cannot_omit_one_auditor_partition(evidence):
    path = evidence.audits / 'auditor-1/rank-1/replay.json'
    value = json.loads(path.read_bytes())
    value['requests'] = []
    save(path, value)
    with pytest.raises(ValueError, match='completely replay'):
        cache_probe.score(evidence)


def test_cached_replay_binds_actual_output_tokens(evidence):
    path = evidence.audits / 'auditor-0/rank-0/replay.json'
    value = json.loads(path.read_bytes())
    value['requests'][0]['token_ids'] = [4, 2]
    save(path, value)
    with pytest.raises(ValueError, match='completely replay'):
        cache_probe.score(evidence)


def test_reject_a_fabricated_correct_answer_flag(evidence):
    for rank in range(2):
        folder = evidence.reports / f'rank-{rank}'
        row = json.loads((folder / 'comparison.jsonl').read_text())
        row['methods']['cached']['check']['correct'] = False
        (folder / 'comparison.jsonl').write_text(json.dumps(row) + '\n')
        complete = json.loads((folder / 'complete.json').read_text())
        complete['comparison_sha256'] = sha256(folder / 'comparison.jsonl')
        save(folder / 'complete.json', complete)
    with pytest.raises(ValueError, match='Stored answer score'):
        cache_probe.score(evidence)
