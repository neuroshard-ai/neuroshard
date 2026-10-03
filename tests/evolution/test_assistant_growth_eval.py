import copy
import subprocess
import sys
import time

import pytest

from neuroshard.evolution import assistant_growth_baseline as baseline
from neuroshard.evolution import assistant_growth_eval as development
from neuroshard.evolution import assistant_routing as routing
from neuroshard.evolution.modular_reference_execution import ROOT, read, save, sha256

from test_assistant_calendar import schedule_texts
from test_assistant_routing import POLICIES, scripted
from test_assistant_workflow import reference_texts, reply

PLAN = read(ROOT / development.PLAN)
SETS = baseline.opened(read(ROOT / PLAN['plan']))


def routes_for(case):
    if case['family'] == 'handoff':
        return ['drafting', 'scheduling']
    return ['scheduling' if case.get('capability') else 'drafting'] * len(case['turns'])


def episode(case, solved):
    texts = (schedule_texts(case) if case.get('capability') else reference_texts(case)) if solved else []
    respond = scripted(texts + ['I cannot finish this.'] * 30, [])
    routes = {name: (respond, policy) for name, policy in POLICIES.items()}
    chosen = routes_for(case)
    row = routing.execute(case, routes, lambda turn, user: chosen[turn])
    return {**row, 'a2_selected': 'arm', 'a2_selection_seconds': 0.5}


def replies(solved):
    return {version: {'episodes': {name: [episode(case, i < solved[version][name]) for i, case in enumerate(cases)]
                                   for name, cases in SETS.items()}} for version in development.VERSIONS}


def accepted(count):
    return [{'id': case['id'], 'score': {'passed': i < count}} for i, case in enumerate(SETS['drafting'])]


def test_each_version_has_a_profile_and_loads_u1_under_its_module():
    assert set(development.PROFILES.values()) == set(development.VERSIONS)
    assert 'assistant-growth-development-separate-module' in development.PROFILES
    assert development.needed('separate_module') == ['L2', 'U1'] and development.needed('shared') == ['U2']
    assert development.needed('separate_update') == ['U1', 'U2']


def test_units_and_gates_must_match_their_pinned_digests(tmp_path):
    execution = {'units': {}, 'gates': {}}
    for unit, digest in (('U1', 'a' * 64), ('U2', 'b' * 64), ('L2', 'c' * 64)):
        save(tmp_path / f'{unit}-checkpoint' / 'manifest.json', {'trainable_sha256': digest})
        execution['units'][unit] = {'checkpoint': f'{unit}-checkpoint', 'trainable_sha256': digest}
    save(tmp_path / 'a2.json', {'arms': {'update': {'gate': {'rule': 'constant-arm'}}}})
    save(tmp_path / 'stage1.json', {'gates': {v: {'rule': 'constant-parent'} for v in development.VERSIONS}})
    execution['gates'] = {name: {'file': f'{name}.json', 'sha256': sha256(tmp_path / f'{name}.json')}
                          for name in ('a2', 'stage1')}
    assert development.verify_units(tmp_path, execution, 'separate_module') == (
        {'rule': 'constant-arm'}, {'rule': 'constant-parent'})
    broken = copy.deepcopy(execution)
    broken['units']['U1']['trainable_sha256'] = 'd' * 64
    with pytest.raises(ValueError, match='U1 differs'):
        development.verify_units(tmp_path, broken, 'separate_module')
    assert development.verify_units(tmp_path, broken, 'shared')
    broken = copy.deepcopy(execution)
    broken['gates']['stage1']['sha256'] = 'e' * 64
    with pytest.raises(ValueError, match='stage1 gates'):
        development.verify_units(tmp_path, broken, 'shared')


def test_the_development_gate_picks_the_candidate_and_checks_drafting_case_by_case():
    solved = {'separate_update': {'scheduling': 18, 'cross': 6, 'drafting': 19},
              'separate_module': {'scheduling': 19, 'cross': 6, 'drafting': 19},
              'shared': {'scheduling': 19, 'cross': 6, 'drafting': 16}}
    rows = replies(solved)
    report = development.assess(PLAN, SETS, rows, accepted(19))
    assert report['candidate'] == 'separate_module' and report['passed'], report['checks']
    assert report['versions']['shared']['drafting_lost'] == sorted(c['id'] for c in SETS['drafting'][16:19])
    assert report['versions']['separate_update']['scheduling']['routes'] == [('scheduling',), ('scheduling', 'scheduling')]
    assert ('drafting', 'scheduling') in report['versions']['shared']['cross']['routes']
    tied = {**solved, 'separate_update': {**solved['separate_update'], 'scheduling': 19}}
    assert development.assess(PLAN, SETS, replies(tied), accepted(19))['candidate'] == 'separate_module'
    behind = {**solved, 'separate_module': {**solved['separate_module'], 'drafting': 18}}
    failed = development.assess(PLAN, SETS, replies(behind), accepted(19))
    assert not failed['passed'] and not failed['checks']['drafting']
    forged = copy.deepcopy(rows)
    score = forged['shared']['episodes']['cross'][0]['score']
    score['passed'] = not score['passed']
    with pytest.raises(ValueError, match='rescore'):
        development.assess(PLAN, SETS, forged, accepted(19))
    shuffled = copy.deepcopy(rows)
    shuffled['shared']['episodes']['drafting'].reverse()
    with pytest.raises(ValueError, match='in order'):
        development.assess(PLAN, SETS, shuffled, accepted(19))


def test_each_turn_is_served_by_its_routes_unit_and_drafting_by_the_a2_choice(monkeypatch):
    from neuroshard.evolution import assistant_experience_run as accelerator
    from neuroshard.evolution import assistant_selector as selector
    from neuroshard.evolution import assistant_serving as serving
    from neuroshard.evolution.assistant_workflow_data import public_case

    meeting = next(c for c in SETS['scheduling'] if len(c['turns']) == 2)
    draft = SETS['drafting'][0]
    scripts = {public_case(c)['user_turns'][0]: iter(texts)
               for c, texts in ((meeting, schedule_texts(meeting)), (draft, reference_texts(draft)))}
    served = []

    def responder(model, tokenizer, policy):
        def respond(messages, tools):
            served.append((model, messages[0]['content'] == policy['system_instruction']))
            return reply(next(scripts[messages[1]['content']]))
        return respond

    meeting_turns = set(public_case(meeting)['user_turns'])
    monkeypatch.setattr(serving, 'cached_responder', responder)
    monkeypatch.setattr(accelerator, 'boundary_feature', lambda model, tokenizer, policy, case, device: case['id'])
    monkeypatch.setattr(routing, 'turn_feature', lambda model, tokenizer, policy, user, device: user)
    monkeypatch.setattr(selector, 'choose', lambda gate, feature: feature in meeting_turns if gate == 'turns'
                        else feature == draft['id'])
    rows = development.routed_episodes('parent', {'U1': 'u1', 'L2': 'l2'}, None, 'separate_module', 'a2', 'turns',
                                       [meeting, draft])
    assert [row['score']['passed'] for row in rows] == [True, True]
    assert rows[0]['score']['routes'] == ['scheduling', 'scheduling'] and rows[0]['a2_selected'] == 'parent'
    assert rows[1]['score']['routes'] == ['drafting'] * len(draft['turns']) and rows[1]['a2_selected'] == 'arm'
    calls = rows[0]['model_attempts']
    assert {model for model, _ in served[:calls]} == {'l2'} and {model for model, _ in served[calls:]} == {'u1'}
    assert all(instructed for _, instructed in served)


def test_latency_counts_each_selection_pass_once():
    case = next(c for c in SETS['scheduling'] if len(c['turns']) == 2)
    respond = scripted(schedule_texts(case), [])

    def select(turn, user):
        time.sleep(.05)
        return 'scheduling'

    row = routing.execute(case, {name: (respond, policy) for name, policy in POLICIES.items()}, select)
    assert row['score']['passed'] and row['seconds'] >= row['selection_seconds'] >= .1
    assert development.latency({**row, 'a2_selection_seconds': .5}) == row['seconds'] + .5


def test_importing_development_does_not_load_torch():
    probe = 'import sys, neuroshard.evolution.assistant_growth_eval; assert "torch" not in sys.modules'
    subprocess.run([sys.executable, '-c', probe], check=True, cwd=ROOT, env={'PYTHONPATH': str(ROOT / 'src')})
