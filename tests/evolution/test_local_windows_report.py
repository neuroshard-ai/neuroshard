import copy
import json
from pathlib import Path

import pytest

from neuroshard.evolution import local_windows_report as report
from neuroshard.evolution import local_windows as windows
from neuroshard.evolution import reference as engine
from neuroshard.evolution import reference_data as data


def test_paired_margin_accounts_for_disagreements_not_just_equal_totals():
    baseline = [True] * 50 + [False] * 50
    unchanged = report.paired_interval(baseline, baseline)
    swapped = report.paired_interval(baseline, [not value for value in baseline])
    assert unchanged['accuracy_gain'] == swapped['accuracy_gain'] == 0
    assert unchanged['normal_99pct_lower'] == 0
    assert swapped['normal_99pct_lower'] < -.03


def fixture():
    plan = json.loads((Path(__file__).resolve().parents[2] / 'config/experiments/local-training-windows.json').read_bytes())
    plan['training'].update(steps=4, local_steps=2)
    plan['checkpoints'] = {'single': [4], 'ddp-four': [4], 'diloco-four': [2, 4]}
    ids = [str(i) for i in range(16)]
    prepared = {'plan': plan, 'roles': {'train': {'ids': ids}}}
    identity = data.identity(prepared)
    schedule = [[ids[i] for i in batch] for batch in engine.schedule(16, 4, 8, plan['training']['seed'])]
    selection, results = {'prepared': identity, 'candidates': {}}, {}
    for arm, world in plan['arms'].items():
        ranks = []
        for rank in range(world):
            binding = windows.rank_binding(identity, {}, arm, rank, world)
            ranks.append({'prepared': identity, 'arm': arm, 'rank': rank, 'world': world, 'binding': binding,
                          'runtime': {}, 'resume_step': 0, 'parameter_digest': arm, 'group_manifest': arm,
                          'candidate': {'receipt': str(rank)}, 'outer_digest': 'outer',
                          'steps': [{'step': i + 1, 'documents': batch, 'local_documents': batch[rank::world],
                                     'seconds': 10 if arm == 'ddp-four' else 1} for i, batch in enumerate(schedule)],
                          'synchronizations': [{'step': i, 'round': i // 2, 'seconds': 1,
                                                'delta_payload_bytes_per_rank': plan['model']['parameters'] * 4}
                                               for i in (2, 4)] if arm == 'diloco-four' else [],
                          'checkpoint_measurements': [{'step': i, 'seconds': 2} for i in plan['checkpoints'][arm]],
                          'seconds': 50, 'network_start': {'ens5': {'rx': 0, 'tx': 0}},
                          'network_end': {'ens5': {'rx': 1, 'tx': 1000 if arm == 'ddp-four' else 100}}})
        results[arm] = ranks
        selection['candidates'][arm] = {key: ranks[0][key] for key in ('candidate', 'binding', 'parameter_digest', 'group_manifest')}
        selection['candidates'][arm]['profile'] = {}
    return prepared, selection, results


def test_efficiency_requires_all_ranks_real_communication_and_the_declared_windows():
    prepared, selection, results = fixture()
    measured = report.training(prepared, selection, results)
    assert measured['efficiency_contract_passed']
    assert measured['ratios']['active_seconds_vs_ddp'] == .15
    assert measured['ratios']['sent_bytes_vs_ddp'] == .1
    changed = copy.deepcopy(results)
    changed['diloco-four'][3]['synchronizations'].pop()
    with pytest.raises(ValueError, match='Synchronization'):
        report.training(prepared, selection, changed)
    changed = copy.deepcopy(results)
    changed['diloco-four'][1]['steps'][0]['local_documents'] = ['different']
    with pytest.raises(ValueError, match='assignment'):
        report.training(prepared, selection, changed)
    changed = copy.deepcopy(results)
    changed['ddp-four'][2]['network_end']['ens5']['tx'] = -1
    with pytest.raises(ValueError, match='backwards'):
        report.training(prepared, selection, changed)
    changed = copy.deepcopy(results)
    changed['diloco-four'][3]['steps'][0]['seconds'] = 100
    assert not report.training(prepared, selection, changed)['conditions']['active_time']


def test_serving_report_preserves_failed_requests_and_rejects_fake_replica_counts():
    plan = {'inference': {'serving_arm': 'diloco-four', 'requests': 2, 'concurrency': 4, 'maximum_attempts': 4}}
    selection = {'candidates': {'diloco-four': {'parameter_digest': 'model'}}}
    phases = {}
    for name in ('single', 'four', 'failure'):
        endpoints = ['a'] if name == 'single' else ['a', 'b', 'c', 'd']
        phases[name] = {'model_digest': 'model', 'requests': 2, 'concurrency': 4, 'endpoints': endpoints,
                        'unavailable': [{'endpoint': 'b'}] if name == 'failure' else [],
                        'seconds': 2, 'successful': 2, 'requests_per_second': 1,
                        'results': [{'task_id': str(i), 'request_id': name + str(i), 'success': True,
                                     'attempts': [{'endpoint': 'a', 'success': True}],
                                     'answer': {'task_id': str(i), 'request_id': name + str(i),
                                                'model_digest': 'model', 'cached': False,
                                                'generation': {'output_ids': [1, 2]}}} for i in range(2)]}
    records = [{'id': '0'}, {'id': '1'}]
    assert report.serving(plan, selection, records, phases)['serving_fixture_passed']
    bad = copy.deepcopy(phases)
    bad['four']['endpoints'] = ['a'] * 4
    with pytest.raises(ValueError):
        report.serving(plan, selection, records, bad)
    phases['failure']['results'][1]['success'] = False
    phases['failure']['successful'] = 1
    phases['failure']['requests_per_second'] = .5
    result = report.serving(plan, selection, records, phases)
    assert not result['serving_fixture_passed']
    assert result['phases']['failure']['successful'] == 1


def test_learning_gain_cannot_hide_a_retention_failure():
    from neuroshard.evolution import grounded_tasks as tasks
    cases = [tasks.make_case(31, 'test', i) for i in range(16)]
    records = [{'id': tasks.task_identity(case), 'task': case,
                'messages': [{'role': 'user', 'content': tasks.prompt(case)}, {'role': 'assistant', 'content': 'target'}]}
               for case in cases]
    plan = json.loads((Path(__file__).resolve().parents[2] / 'config/experiments/local-training-windows.json').read_bytes())
    prepared = {'plan': plan, 'roles': {'test': {'ids': [row['id'] for row in records]}, 'retention': {'ids': ['a', 'b']}}}
    selection = {'prepared': data.identity(prepared), 'candidates': {arm: {'profile': {}, 'name': arm} for arm in plan['arms']}}
    evaluations = {}
    for arm in ('seed', *plan['arms']):
        texts = ['invalid' if arm == 'seed' else json.dumps(tasks.expected(case)) for case in cases]
        checks = [{'id': row['id'], 'family': row['task']['family'], 'variant': row['task']['variant'],
                   **tasks.check_answer(row['task'], text)} for row, text in zip(records, texts)]
        evaluations[arm] = {'arm': arm, 'role': 'test', 'runtime': {}, 'prepared': data.identity(prepared),
                            'candidate': None if arm == 'seed' else selection['candidates'][arm],
                            'checks': checks, 'documents': 16, 'correct': sum(row['correct'] for row in checks),
                            'generations': [{'id': row['id'], 'messages': row['messages'][:-1], 'text': text}
                                            for row, text in zip(records, texts)],
                            'retention': [{'id': name, 'targets': 1, 'loss': 1.0} for name in ('a', 'b')]}
    assert report.learning(prepared, selection, records, evaluations)['quality_contract_passed']
    for row in evaluations['diloco-four']['retention']:
        row['loss'] = 1.1
    measured = report.learning(prepared, selection, records, evaluations)
    assert measured['conditions']['minimum_accuracy_gain_vs_seed']
    assert not measured['quality_contract_passed'] and not measured['conditions']['retention_99pct_upper_max']


def test_recovery_report_requires_local_optimizer_coverage_and_the_same_trajectory():
    prepared, selection, results = fixture()
    original = results['diloco-four']
    resumed = copy.deepcopy(original)
    restore_step = prepared['plan']['recovery']['restore_step']
    parameters = prepared['plan']['model']['parameters']
    comparisons = []
    for rank, row in enumerate(resumed):
        row['resume_step'] = restore_step
        comparisons.append({'rank': rank, 'prepared': data.identity(prepared), 'binding': row['binding'],
                            'resume_step': restore_step, 'contents_exactly_equal': True,
                            'uninterrupted_receipt': row['candidate']['receipt'],
                            'recovered_receipt': row['candidate']['receipt'],
                            'parameter_digest': row['parameter_digest'], 'outer_digest': row['outer_digest'],
                            'comparisons': {'model': {'tensor_elements': parameters},
                                            'outer.pt': {'tensor_elements': parameters},
                                            'optimizer.pt': {'tensor_elements': 2 * parameters + 100}}})
    fault = {'rank': 1, 'signal': 'SIGKILL', 'last_observed_completed_step': 75}
    failed = {'exit_codes': dict.fromkeys(range(4), 1), 'seconds': 3, 'finished': 5}
    restarted = {'exit_codes': dict.fromkeys(range(4), 0), 'seconds': 4, 'started': 6}
    assert report.recovery(prepared, selection, original, resumed, comparisons, fault, failed, restarted)['all_rank_state_and_trajectory_comparisons_passed']
    changed = copy.deepcopy(comparisons)
    changed[2]['comparisons']['optimizer.pt']['tensor_elements'] = 0
    with pytest.raises(ValueError, match='omitted required tensors'):
        report.recovery(prepared, selection, original, resumed, changed, fault, failed, restarted)
    resumed[3]['steps'][0]['loss'] = 123
    with pytest.raises(ValueError, match='trajectory'):
        report.recovery(prepared, selection, original, resumed, comparisons, fault, failed, restarted)
