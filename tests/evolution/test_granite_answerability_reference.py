import copy
import importlib.util
import json

import pytest

from neuroshard.evolution import granite_answerability_reference as study
from neuroshard.evolution.finite_decision import FiniteDecision


class Tokenizer:
    all_special_ids = [99]

    def __call__(self, text, **kwargs):
        return {'input_ids': [ord(c) for c in text]}

    def decode(self, tokens, **kwargs):
        return ''.join(chr(i) for i in tokens)


def plan():
    return study.read(study.ROOT / study.PLAN)


def row(task, which, *, bad_selection=False, reject=False):
    p = plan()
    messages, sources, menu = study.public_input(task)
    option = next((c for c in menu['choices'] if c['source_id'] in task['expected']['sources']), None)
    choice = option['choice'] if option else 'Z'
    if bad_selection:
        choice = 'Z' if option else menu['choices'][0]['choice']
    selection = {'task_sha256': study.identity(study.selection_task(p, messages, menu)),
                 'text': choice, 'token_ids': [32 + ord(choice) - ord('A')], 'input_token_ids': [101],
                 'terminated': False, 'prompt_sha256': study.identity(messages), 'model': which,
                 'route_counts': {'0': 1} if which == 'modular' else {},
                 'route_trace': [[0]] if which == 'modular' else []}
    initial = study.evidence.resolve(messages, sources, {'invocation_root': menu['invocation_root'], 'choice': choice})
    check = None
    if choice != 'Z':
        ct = study.checker_task(p, messages, sources, choice, menu, which)
        verdict = 'unanswerable' if reject else 'answerable'
        tokens = [1, 9399, 481, 1, 100257] if verdict == 'answerable' else [1, 359, 9399, 481, 1, 100257]
        check = {'task_sha256': study.identity(ct), 'text': json.dumps(verdict), 'token_ids': tokens,
                 'input_token_ids': [102], 'terminated': True, 'prompt_sha256': study.identity(ct),
                 'model': which, 'route_counts': {'0': 1, '5': 1} if which == 'modular' else {},
                 'route_trace': [[0, 5]] if which == 'modular' else []}
    receipt = study.finish(messages, sources, menu, selection, check)
    return {'id': task['id'], 'model': which, 'task_sha256': study.identity(task), 'menu': menu,
            'selection': selection, 'check': check, 'selection_receipt': initial, 'receipt': receipt,
            'selection_seconds': .5, 'seconds': 1, 'generation_calls': 1 + bool(check),
            'input_tokens': 1 + bool(check), 'output_tokens': 1 + (len(check['token_ids']) if check else 0),
            'passed': study.is_correct(task, receipt)}


def anchors():
    result = []
    for task in study.read(study.ROOT / study.reference.PLAN)['tasks']:
        text = task['accept'][0] if task['kind'] == 'exact' else json.dumps(task['expected'])
        if task['kind'] == 'tool':
            text = '<tool_call>' + text + '</tool_call>'
        result.append({'id': task['id'], 'text': text, 'terminated': True, 'passed': True})
    return result


def workers():
    p = plan()
    fresh, retained = study.task_sets(p)
    values = {}
    for which in ('baseline', 'modular'):
        rows = [row(t, which) for t in retained]
        # All unsupported fresh cases are initially misselected in this fixture.
        # The module fixes them by veto; selectors remain exactly identical.
        rows += [row(t, which, bad_selection=t['category'] == 'unsupported',
                     reject=which == 'modular' and t['category'] == 'unsupported') for t in fresh]
        values[which] = {'rows': rows, 'anchors': anchors(), 'execution_completed': True,
                         'peak_rss_bytes': 1024}
    return values


def test_finite_decision_allows_only_complete_choices_and_terminal_eos():
    grammar = FiniteDecision(Tokenizer(), ['yes', 'no'], 99)
    assert grammar.allowed([]) == [110, 121]
    assert grammar.allowed([121, 101]) == [115]
    assert grammar.allowed([121, 101, 115]) == [99]
    assert grammar.decode([121, 101, 115, 99]) == 'yes'
    for wrong in ([121, 101], [121, 101, 115], [121, 101, 115, 99, 99]):
        with pytest.raises(ValueError, match='incomplete'):
            grammar.decode(wrong)
    with pytest.raises(ValueError, match='escaped'):
        grammar.allowed([1])
    with pytest.raises(ValueError, match='distinct'):
        FiniteDecision(Tokenizer(), ['yes', 'yes'], 99)


def test_cases_are_fresh_balanced_paired_and_public_input_cannot_see_gold():
    p = plan()
    fresh, retained = study.task_sets(p)
    assert len(fresh) == 80 and len(retained) == 64
    assert len({t['block'] for t in fresh}) == 8
    assert len({t['pair'] for t in fresh}) == 40
    assert sum(t['category'] == 'supported' for t in fresh) == 40
    assert not ({study.identity(t['messages']) for t in fresh} & {study.identity(t['messages']) for t in retained})
    for task in fresh:
        messages, sources, menu = study.public_input(task)
        changed = {**task, 'expected': {'answer': 'injected'}, 'category': 'injected', 'id': 'injected'}
        assert study.public_input(changed) == (messages, sources, menu)
        assert len(menu['choices']) == 3
        possible = [study.evidence.resolve(messages, sources, {'invocation_root': menu['invocation_root'],
                    'choice': choice}) for choice in [c['choice'] for c in menu['choices']] + ['Z']]
        assert sum(study.is_correct(task, receipt) for receipt in possible) == 1
    assert not p['training_authorized'] and not p['gpu_launch_authorized']


def test_checker_sees_only_selected_full_document_and_original_question():
    p = plan()
    task = study.task_sets(p)[0][0]
    messages, sources, menu = study.public_input(task)
    choice = menu['choices'][0]['choice']
    parent = study.checker_task(p, messages, sources, choice, menu, 'baseline')
    modular = study.checker_task(p, messages, sources, choice, menu, 'modular')
    assert modular == {**parent, 'adapter': 'answerability'}
    assert parent['messages'][1:] == messages
    assert parent['documents'] == [{'doc_id': menu['choices'][0]['source_id'], 'text': menu['choices'][0]['text']}]
    assert 'expected' not in parent and parent['accept'] == []


def test_gain_requires_full_path_and_preservation_not_valid_receipts():
    p, values = plan(), workers()
    report = study.assess(p, values)
    assert report['quality_gate'] and report['selector_parity']
    assert report['correct'] == {'selection': 40, 'parent-check': 40, 'module-check': 80}
    assert all(c['blocks'] == 8 for c in report['comparisons'].values())
    assert not report['checklist_credit'] and not report['training_authorized']
    task = next(t for t in study.task_sets(p)[0] if t['category'] == 'supported')
    bad = row(task, 'modular', reject=True)
    values['modular']['rows'] = [bad if r['id'] == task['id'] else r for r in values['modular']['rows']]
    report = study.assess(p, values)
    assert not report['quality_gate']
    assert task['id'] in report['comparisons']['selection']['lost_ids']


def test_parent_checker_regression_does_not_cancel_candidate_that_can_fix_it():
    p, values = plan(), workers()
    retained = next(t for t in study.task_sets(p)[1] if t['expected']['answer'] is not None)
    bad = row(retained, 'baseline', reject=True)
    values['baseline']['rows'] = [bad if r['id'] == retained['id'] else r for r in values['baseline']['rows']]
    assert study.assess(p, values)['quality_gate']
    bad = row(retained, 'modular', reject=True)
    values['modular']['rows'] = [bad if r['id'] == retained['id'] else r for r in values['modular']['rows']]
    assert not study.assess(p, values)['quality_gate']


def test_incomplete_tampered_unrouted_and_overbudget_results_do_not_pass():
    p, values = plan(), workers()
    assert not study.assess(p, {'baseline': values['baseline']})['quality_gate']
    changed = copy.deepcopy(values)
    changed['modular']['rows'][0]['receipt']['answer']['answer'] = 'forged'
    with pytest.raises(ValueError, match='receipt'):
        study.assess(p, changed)
    changed = copy.deepcopy(values)
    target = next(r for r in changed['modular']['rows'] if r['check'])
    target['check']['route_counts'] = {'0': 10}
    with pytest.raises(ValueError, match='activate'):
        study.assess(p, changed)
    changed = copy.deepcopy(values)
    for r in changed['modular']['rows']:
        r['seconds'] = 31
    assert not study.assess(p, changed)['quality_gate']
    changed = copy.deepcopy(values)
    changed['modular']['peak_rss_bytes'] = p['quality']['maximum_worker_peak_rss_bytes'] + 1
    assert not study.assess(p, changed)['quality_gate']


def test_controls_without_headroom_stop_and_replay_requires_same_decisions():
    p = plan()
    fresh, retained = study.task_sets(p)
    baseline = {'rows': [row(t, 'baseline') for t in fresh + retained], 'anchors': anchors(),
                'execution_completed': True, 'peak_rss_bytes': 1024}
    report = study.assess(p, {'baseline': baseline})
    assert report['insufficient_headroom'] and not report['quality_gate']
    original = baseline['rows'][0]
    assert study.replay_matches(original, copy.deepcopy(original))
    changed = copy.deepcopy(original)
    changed['selection']['token_ids'] = [777]
    assert not study.replay_matches(original, changed)


def test_controller_does_not_launch_candidate_or_replays_without_headroom(tmp_path, monkeypatch):
    p = plan()
    fresh, retained = study.task_sets(p)
    baseline = {'rows': [row(t, 'baseline') for t in retained + fresh], 'anchors': anchors(),
                'execution_completed': True, 'peak_rss_bytes': 1024}
    launched = []

    def launch(home, models, binding, which, phase, *args, **kwargs):
        launched.append((which, phase))
        return baseline

    monkeypatch.setattr(study, 'configure', lambda: None)
    monkeypatch.setattr(study, 'freeze', lambda: {'commit': 'fixture', 'sources': {}})
    monkeypatch.setattr(study, 'launch', launch)
    result = study.run(tmp_path / 'run', tmp_path / 'models')
    assert launched == [('baseline', 'primary')]
    assert result['execution_completed'] and not result['reference_passed']
    assert result['report']['insufficient_headroom']
    assert 'candidate not run' in result['stop_reason']


def test_committed_contract_inventory_and_cloud_envelope():
    execution = study.read(study.ROOT / study.EXECUTION)
    for path, digest in execution['contracts'].items():
        assert study.sha256(study.ROOT / path) == digest
    assert study.SCRIPT in execution['sources']
    assert 'src/neuroshard/evolution/finite_decision.py' in execution['sources']
    spec = importlib.util.spec_from_file_location('answerability_cloud', study.ROOT / 'scripts/modular_reference_cloud.py')
    cloud = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cloud)
    resources = cloud.resources('granite-answerability-reference')
    assert resources['hours'] == 2 and resources['planning_cap_usd'] == 6
    assert resources['attempts'] == 1 and not resources['gpu']
    assert study.SCRIPT in ' '.join(cloud.remote_command('granite-answerability-reference'))
