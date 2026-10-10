import copy
import subprocess
import sys

import pytest

from neuroshard.evolution import assistant_routing as routing
from neuroshard.evolution import assistant_turn_router as turn_router
from neuroshard.evolution import router_scaling_episodes as episodes
from neuroshard.evolution import router_scaling_paraphrase as paraphrase
from neuroshard.evolution import router_scaling_reworded as reworded
from neuroshard.evolution.modular_reference_execution import ROOT, read, sha256

from test_assistant_calendar import schedule_texts
from test_assistant_growth_baseline import cloud_module
from test_assistant_routing import POLICIES
from test_assistant_workflow import reference_texts, reply

DECLARATION = read(ROOT / episodes.DECLARATION)


def test_reworded_integration_cases_keep_workspace_turns_and_goals():
    declared = dict(episodes.sets())
    assert list(declared) == DECLARATION['priority']
    original = {case['id']: case for case in declared['original-drafting']}
    for name in ('reworded-drafting', 'reworded-cross', 'reworded-scheduling'):
        for case in declared[name]:
            assert case['id'].endswith('-reworded') and case['split'] in reworded.SPLITS
    for case in declared['reworded-drafting']:
        source = original[case['reworded_from']]
        assert case['world'] == source['world'] and case['turns'][1:] == source['turns'][1:]
        assert case['turns'][0]['expected'] == source['turns'][0]['expected']
        assert case['turns'][0]['user'] != source['turns'][0]['user']
        assert paraphrase.preserved(source['turns'][0]['user'], case['turns'][0]['user'])
    assert len(declared['reworded-drafting']) == len(declared['original-drafting']) == 32
    assert len(declared['reworded-cross']) == 8 and len(declared['reworded-scheduling']) == 32
    with pytest.raises(ValueError):
        reworded.reworded([{**declared['original-drafting'][0], 'split': 'confirmation'}], 0)


def test_rewordings_differ_from_paraphrase_frames_and_use_new_content_words():
    for frame in reworded.DRAFT_REWORDS + reworded.MEETING_REWORDS:
        opening = frame.split('{')[0]
        assert all(not other.startswith(opening) for other in paraphrase.DRAFT_FRAMES + paraphrase.MEETING_FRAMES)


def test_a_reworded_case_scores_as_the_original_when_solved():
    declared = dict(episodes.sets())
    for name, texts in (('reworded-drafting', reference_texts), ('reworded-scheduling', schedule_texts),
                        ('reworded-cross', schedule_texts)):
        case = declared[name][0]
        replies = iter(texts(case) + ['I cannot finish this.'] * 30)
        plan = ['scheduling' if name != 'reworded-drafting' else 'drafting'] * len(case['turns'])
        row = routing.execute(case, {route: (lambda m, t: reply(next(replies)), POLICIES[route]) for route in POLICIES},
                              lambda turn, user: plan[turn])
        assert row['score']['passed'], name


class Responder:
    """Scripted serving: solves a case only if every turn is on the route its text expects."""

    def __init__(self, case, plan, solvable):
        good = all(route == need for route, need in zip(plan, solvable))
        self.texts = iter((reference_texts(case) if case['id'].startswith('workflow') else schedule_texts(case))
                          if good else [])

    def __call__(self, messages, tools):
        return reply(next(self.texts, 'I cannot finish this.'))


def test_the_worker_serves_each_distinct_plan_once_and_assess_compares_routers(monkeypatch):
    declared = dict(episodes.sets())
    case_a, case_b = declared['reworded-drafting'][:2]
    plans = {case_a['id']: {'pinned': ['scheduling'] * len(case_a['turns']), 'context': ['drafting'] * len(case_a['turns'])},
             case_b['id']: {'pinned': ['drafting'] * len(case_b['turns']), 'context': ['drafting'] * len(case_b['turns'])}}

    def serve(case, plan):
        responder = Responder(case, plan, ['drafting'] * len(case['turns']))
        row = routing.execute(case, {route: (responder, POLICIES[route]) for route in POLICIES},
                              lambda turn, user: plan[turn])
        return {**row, 'plan': list(plan)}

    rows = []
    for case in (case_a, case_b):
        distinct = [('pinned', plans[case['id']]['pinned'])]
        if plans[case['id']]['context'] != plans[case['id']]['pinned']:
            distinct.append(('context', plans[case['id']]['context']))
        for label, plan in distinct:
            rows.append({**serve(case, plan), 'routers': ['pinned', 'context'] if len(distinct) == 1 else [label]})
    assert len(rows) == 3
    monkeypatch.setattr(episodes, 'sets', lambda declaration=None: [('reworded-drafting', [case_a, case_b])])
    result = {'reply': {'episodes': {'reworded-drafting': rows}, 'plans': plans}}
    report = episodes.assess(result, POLICIES)['reworded-drafting']
    assert report['cases'] == 2 and report['plans_differ'] == 1
    assert report['context'] == 2 and report['gained'] == [case_a['id']] and not report['lost']
    assert report['pinned'] == 1
    forged = copy.deepcopy(result)
    forged['reply']['episodes']['reworded-drafting'][0]['score']['passed'] = True
    with pytest.raises(ValueError, match='rescore'):
        episodes.assess(forged, POLICIES)
    wrong = copy.deepcopy(result)
    wrong['reply']['episodes']['reworded-drafting'][0]['plan'] = ['drafting'] * len(case_a['turns'])
    with pytest.raises(ValueError, match='plan'):
        episodes.assess(wrong, POLICIES)


def test_the_context_router_is_frozen_and_routes_the_real_turns():
    frozen = read(ROOT / episodes.ROUTER)
    router = turn_router.verify(frozen['router'])
    assert router['routes'] == ['drafting', 'scheduling'] and router['width'] == 2 * 2560
    assert router['threshold'] == 0.0 and router['fallback'] is None


@pytest.mark.parametrize('profile', sorted(episodes.PROFILES))
def test_execution_pins_cohort_three_units_gates_and_the_canonical_runtime(profile):
    execution = read(ROOT / episodes.profile_files(profile)[1])
    cohort3 = read(ROOT / 'config/experiments/assistant-growth-cohort3-development-execution.json')
    assert execution['units'] == cohort3['units'] and execution['gates'] == cohort3['gates']
    for name, digest in execution['contracts'].items():
        assert sha256(ROOT / name) == digest
    assert set(execution['contracts']) <= set(execution['sources']) and episodes.SCRIPT in execution['sources']
    assert not execution['training_authorized'] and not execution['gpu_launch_authorized']
    probe = ('import os, sys; import neuroshard.evolution.router_scaling_episodes as e; assert "torch" not in sys.modules; '
             'import neuroshard.evolution.assistant_growth_cohort3_eval, neuroshard.evolution.assistant_growth_confirm, '
             'neuroshard.evolution.assistant_turn_router, neuroshard.evolution.granite_reference, '
             'neuroshard.evolution.granite_tokenizer, neuroshard.evolution.assistant_serving, '
             'neuroshard.evolution.assistant_experience_run, neuroshard.evolution.assistant_workflow_canonical; e.sets(); '
             'root = os.path.abspath("src"); print("\\n".join(sorted(os.path.relpath(x.__file__) '
             'for x in list(sys.modules.values()) if getattr(x, "__file__", None) and '
             'os.path.abspath(x.__file__).startswith(root))))')
    imported = subprocess.check_output([sys.executable, '-c', probe], cwd=ROOT, text=True,
                                       env={'PYTHONPATH': str(ROOT / 'src')}).split()
    assert set(imported) <= set(execution['sources']), sorted(set(imported) - set(execution['sources']))
    canonical = read(ROOT / 'config/experiments/assistant-workflow-canonical-execution.json')
    assert all(execution[key] == canonical[key] for key in ('packages', 'python', 'required_cpu_flags', 'environment'))


@pytest.mark.parametrize('profile', sorted(episodes.PROFILES))
def test_one_cpu_host_within_its_allowance(profile):
    cloud = cloud_module()
    resources = cloud.resources(profile)
    execution = read(ROOT / episodes.profile_files(profile)[1])
    assert not resources['gpu'] and resources['instance_type'] == 'r7i.4xlarge'
    assert resources['hours'] * resources['price']['usd_per_hour'] + 3 <= resources['planning_cap_usd'] <= 6.5
    assert (execution['prepare_seconds'] + execution['worker_seconds'] + resources['setup_seconds']
            + resources['copy_seconds'] + 600 <= resources['hours'] * 3600)
    assert cloud.UPLOAD_PROFILES[profile] == '.units'
    assert set(resources['upload']['files']) == {f'{u}/{f}' for u in episodes.NEEDED
                                                 for f in ('manifest.json', 'trainable.safetensors')} | {
        'a2-integration.json', 'router.json'}
    command = cloud.remote_command(profile)
    assert command[1].endswith(episodes.SCRIPT) and command[-2:] == ['--profile', profile]


def test_the_originals_run_serves_the_original_wording_of_the_same_cases():
    first = dict(episodes.sets(read(ROOT / episodes.DECLARATION)))
    originals = read(ROOT / 'config/experiments/router-scaling-episodes-originals.json')
    second = dict(episodes.sets(originals))
    assert list(second) == ['original-scheduling', 'original-cross']
    for name in ('scheduling', 'cross'):
        assert [c['reworded_from'] for c in first[f'reworded-{name}']] == [c['id'] for c in second[f'original-{name}']]
    pinned = originals['pairs_with']
    assert sha256(ROOT / pinned['path']) == pinned['sha256']


def test_the_first_run_is_recorded_rescored_and_retired():
    report = read(ROOT / 'config/experiments/router-scaling-episodes-report.json')
    result = read(ROOT / 'config/experiments/router-scaling-episodes-result.json')
    assert sha256(ROOT / 'config/experiments/router-scaling-episodes-result.json') == report['result_sha256']
    assert report['execution_completed'] and report['stopped_before'] is None
    assert episodes.assess(result) == report['sets']
    finished = report['resources_finished']
    assert not finished['remaining_instances'] and finished['security_group_retired']
    assert finished['conservative_compute_usd'] <= read(ROOT / 'config/experiments/router-scaling-episodes-resources.json')['planning_cap_usd']
