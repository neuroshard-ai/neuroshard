import copy
import importlib.util
import tarfile
import io

import pytest

from neuroshard.evolution import assistant_experience_eval as evaluation
from neuroshard.evolution import assistant_workflow as workflow
from neuroshard.evolution import assistant_workflow_data as data
from neuroshard.evolution.modular_reference_execution import ROOT, read, save, sha256

from test_assistant_workflow import execute_fixture, policy, reference_texts, reply

PLAN = read(ROOT / 'config/experiments/assistant-experience-learning.json')


def failing(case):
    texts = iter(reference_texts(case)[:1] + ['I cannot finish this.'] * 10)
    return workflow.execute(case, lambda m, t: reply(next(texts)), policy())


def canonical(cases, successes):
    episodes = [execute_fixture(c) if c['id'] in successes else failing(c) for c in cases]
    anchors = ['granite-chat-booking', 'granite-tool-timer']
    return {'primary': {'episodes': episodes},
            'report': {'protected_workflow_ids': sorted(successes), 'protected_anchor_ids': anchors}}


def system(cases, successes, selected):
    return [{**(execute_fixture(c) if c['id'] in successes else failing(c)), 'selected': selected,
             'selection_seconds': 0.5} for c in cases]


def test_each_episode_pays_its_own_selection_forward_pass(monkeypatch):
    import time as clock

    cases = data.cases('development')[:3]
    scripted = {c['id']: iter(reference_texts(c)) for c in cases}
    current = {}

    def responder(model, tokenizer, policy_value):
        def respond(messages, tools):
            return reply(next(scripted[current['id']]))
        return respond

    def feature(case):
        current['id'] = case['id']
        clock.sleep(0.05)
        return [1.0]

    monkeypatch.setattr(evaluation.first, 'native_responder', responder)
    monkeypatch.setattr(evaluation.reference, 'generate', lambda *a, **kw: {'id': a[3]['id'], 'passed': True})
    anchors = {'tasks': [{'id': 'granite-chat-booking'}]}
    report = evaluation.evaluate_arm(None, None, None, {'rule': 'constant-arm'}, feature, cases, policy(), anchors)
    assert [row['selected'] for row in report['episodes']] == ['arm'] * 3
    assert all(row['selection_seconds'] >= 0.05 and row['score']['passed'] for row in report['episodes'])
    assert report['forced_anchors'] == [{'id': 'granite-chat-booking', 'passed': True}]


def cloud_module():
    spec = importlib.util.spec_from_file_location('evaluation_cloud', ROOT / 'scripts/modular_reference_cloud.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_assessment_rescores_every_episode_and_reports_forced_anchor_forgetting():
    cases = data.cases('development')
    ids = [c['id'] for c in cases]
    base = canonical(cases, set(ids[:6]))
    anchors = [{'id': 'granite-chat-booking', 'passed': True}, {'id': 'granite-tool-timer', 'passed': False}]
    replies = {'update': {'episodes': system(cases, set(ids[:20]), 'arm'), 'forced_anchors': anchors},
               'addition': {'episodes': system(cases, set(ids[:20]), 'arm'),
                            'forced_anchors': anchors[:1] + [{'id': 'granite-tool-timer', 'passed': True}]}}
    report = evaluation.assess(PLAN, cases, base, replies)
    assert report['passed'] and report['correct'] == {'parent': 6, 'update': 20, 'addition': 20}
    assert report['forced_anchor_forgetting']['update']['lost_protected'] == ['granite-tool-timer']
    assert report['forced_anchor_forgetting']['addition']['lost_protected'] == []
    assert report['selected_arm_episodes'] == {'update': 24, 'addition': 24}
    forged = copy.deepcopy(replies)
    forged['addition']['episodes'][0]['score']['passed'] = False
    with pytest.raises(ValueError, match='rescore'):
        evaluation.assess(PLAN, cases, base, forged)


def test_a1_served_check_judges_the_routed_addition_with_fresh_process_replays():
    cases = data.cases('development')
    ids = [c['id'] for c in cases]
    base = canonical(cases, set(ids[:6]))
    base['report'].update(canonical_anchor_gate=True, prior_anchor_successes_lost=[], anchor_correct=19)
    canonical_plan = read(ROOT / 'config/experiments/assistant-workflow-canonical.json')
    primitive = [c['id'] for c in cases if c['primitive']]
    wins = set(primitive[:6]) | {c['id'] for c in cases if not c['primitive']}
    by_family = {}
    for c in cases:
        if c['primitive'] and c['id'] in primitive[:6]:
            by_family.setdefault(c['family'], 0)
            by_family[c['family']] += 1
    assert len(by_family) == 4
    rows = system(cases, wins, 'arm')
    replies = {'update': {'episodes': system(cases, wins, 'arm'), 'forced_anchors': []},
               'addition': {'episodes': rows, 'forced_anchors': []}}
    replays = [copy.deepcopy(r) for r in rows if r['id'] in canonical_plan['replay_ids']]
    report = evaluation.assess(PLAN, cases, base, replies, replays, canonical_plan)
    assert report['a1_served']['passed'] and report['development_and_a1_passed']
    assert report['a1_served']['primitive_correct'] == 6 and report['a1_served']['anchor_routing'] == 'parent'
    drifted = copy.deepcopy(replays)
    drifted[0]['generations'][0]['input_token_ids'] = [7] * len(drifted[0]['generations'][0]['input_token_ids'])
    missing = evaluation.assess(PLAN, cases, base, replies, drifted, canonical_plan)['a1_served']
    assert not missing['passed'] and not missing['checks']['fresh_process_replay'] and missing['checks']['primitive']
    assert not evaluation.assess(PLAN, cases, base, replies, None, canonical_plan)['a1_served']['passed']
    one_family = next(f for f in by_family)
    narrow = wins - {c['id'] for c in cases if c['family'] == one_family}
    narrow_replies = {arm: {'episodes': system(cases, narrow, 'arm'), 'forced_anchors': []} for arm in evaluation.ARMS}
    narrow_replays = [r for r in narrow_replies['addition']['episodes'] if r['id'] in canonical_plan['replay_ids']]
    served = evaluation.assess(PLAN, cases, base, narrow_replies, copy.deepcopy(narrow_replays), canonical_plan)['a1_served']
    assert not served['checks']['primitive']


def test_a1_served_check_can_judge_the_update_gated_alone():
    third = read(ROOT / 'config/experiments/assistant-experience-third.json')
    cases = data.cases('development')
    ids = [c['id'] for c in cases]
    base = canonical(cases, set(ids[:6]))
    base['report'].update(canonical_anchor_gate=True, prior_anchor_successes_lost=[], anchor_correct=19)
    canonical_plan = read(ROOT / 'config/experiments/assistant-workflow-canonical.json')
    unsolved = [c['id'] for c in cases if not c['primitive'] and c['id'] not in ids[:6]][:4]
    wins = set(ids) - set(unsolved)
    rows = system(cases, wins, 'arm')
    replies = {'update': {'episodes': rows, 'forced_anchors': []}}
    replays = [copy.deepcopy(r) for r in rows if r['id'] in canonical_plan['replay_ids']]
    report = evaluation.assess(third, cases, base, replies, replays, canonical_plan, served_arm='update')
    assert report['passed'] and report['development_and_a1_passed'] and report['a1_served']['served'] == 'update'
    assert set(report['correct']) == {'parent', 'update'} and 'net_vs_update' not in report['checks']
    assert set(report['forced_anchor_forgetting']) == set(report['selected_arm_episodes']) == {'update'}
    # The learning contract's gate compares with the update control, so it cannot judge the update alone.
    with pytest.raises(ValueError, match='update control'):
        evaluation.assess(PLAN, cases, base, replies, replays, canonical_plan, served_arm='update')


def test_an_execution_serves_one_of_the_arms_it_evaluates():
    assert evaluation.systems({}) == (('update', 'addition'), 'addition')
    assert evaluation.systems({'systems': ['update'], 'served': 'update'}) == (('update',), 'update')
    for bad in ({'systems': ['update']}, {'systems': ['committee'], 'served': 'committee'}, {'systems': []}):
        with pytest.raises(ValueError, match='known arms'):
            evaluation.systems(bad)


def test_uploaded_arms_must_match_pinned_digests(tmp_path):
    for arm in evaluation.ARMS:
        save(tmp_path / f'{arm}-checkpoint' / 'manifest.json', {'arm': arm, 'trainable_sha256': arm * 2})
    save(tmp_path / 'integration.json', {'arms': {}})
    pinned = {arm: {'trainable_sha256': arm * 2} for arm in evaluation.ARMS}
    pinned['integration_sha256'] = sha256(tmp_path / 'integration.json')
    assert evaluation.verify_arms(tmp_path, pinned) == {'arms': {}}
    with pytest.raises(ValueError, match='addition checkpoint'):
        evaluation.verify_arms(tmp_path, {**pinned, 'addition': {'trainable_sha256': 'other'}})
    with pytest.raises(ValueError, match='integration gates'):
        evaluation.verify_arms(tmp_path, {**pinned, 'integration_sha256': 'other'})
    alone = {'update': pinned['update'], 'integration_sha256': pinned['integration_sha256']}
    (tmp_path / 'addition-checkpoint' / 'manifest.json').unlink()
    assert evaluation.verify_arms(tmp_path, alone) == {'arms': {}}
    with pytest.raises(ValueError, match='no arm'):
        evaluation.verify_arms(tmp_path, {'integration_sha256': pinned['integration_sha256']})


def test_cpu_profile_uploads_only_declared_regular_files_within_its_allowance(tmp_path):
    cloud = cloud_module()
    resources = cloud.resources(evaluation.PROFILE)
    assert not resources['gpu'] and resources['instance_type'] == 'r7i.4xlarge'
    assert (resources['hours'], resources['planning_cap_usd']) == cloud.LONG_CPU_PROFILES[evaluation.PROFILE]
    assert resources['hours'] * resources['price']['usd_per_hour'] + 3 <= resources['planning_cap_usd']
    assert cloud.remote_command(evaluation.PROFILE)[1].endswith(evaluation.SCRIPT)
    assert cloud.UPLOAD_PROFILES[evaluation.PROFILE] == evaluation.UPLOADED
    assert not any(name.endswith('optimizer.pt') for name in resources['upload']['files'])
    for name in ['integration.json', 'update-checkpoint/trainable.safetensors']:
        (tmp_path / name).parent.mkdir(parents=True, exist_ok=True)
        (tmp_path / name).write_bytes(b'x' * 10)
    spec = {'local': str(tmp_path), 'files': ['integration.json', 'update-checkpoint/trainable.safetensors'],
            'maximum_bytes': 10 ** 6}
    with tarfile.open(fileobj=io.BytesIO(cloud.upload_bundle(spec)), mode='r:gz') as archive:
        assert sorted(m.name for m in archive.getmembers()) == spec['files']
        assert all(m.isfile() for m in archive.getmembers())
    for bad in (['../escape'], ['missing.json']):
        with pytest.raises(ValueError, match='regular declared file'):
            cloud.upload_bundle({**spec, 'files': bad})
    (tmp_path / 'link.json').symlink_to(tmp_path / 'integration.json')
    with pytest.raises(ValueError, match='regular declared file'):
        cloud.upload_bundle({**spec, 'files': ['link.json']})
    with pytest.raises(ValueError, match='declared size'):
        cloud.upload_bundle({**spec, 'maximum_bytes': 10})


def test_development_inventory_pins_arms_runtime_and_every_imported_source():
    import subprocess
    import sys

    execution = read(ROOT / evaluation.EXECUTION)
    for name, digest in execution['contracts'].items():
        assert sha256(ROOT / name) == digest
    assert set(execution['contracts']) <= set(execution['sources'])
    baseline = read(ROOT / 'config/experiments/assistant-workflow-canonical-execution.json')
    assert all(execution[k] == baseline[k] for k in ('packages', 'python', 'required_cpu_flags', 'environment'))
    assert execution['threads'] == read(ROOT / 'config/experiments/assistant-workflow-canonical.json')['resources']['threads']
    evaluated, served = evaluation.systems(execution)
    assert set(execution['arms']) == {*evaluated, 'integration_sha256'}
    trained = read(ROOT / f"config/experiments/assistant-experience-round{execution['round']}-report.json")
    assert all(execution['arms'][arm] == {'trainable_sha256': trained['training'][arm]['trainable_sha256']} for arm in evaluated)
    assert execution['arms']['integration_sha256'] == trained['integration']['integration_sha256']
    upload = read(ROOT / 'config/experiments/assistant-experience-development-resources.json')['upload']['files']
    assert set(upload) == {'integration.json'} | {f'{arm}-checkpoint/{name}' for arm in evaluated
                                                  for name in ('manifest.json', 'trainable.safetensors')}
    if 'gate_plan' in execution:
        assert execution['a1_served'] == execution['gate_plan'] + '#a1_served' and execution['gate_plan'] in execution['contracts']
    assert sha256(ROOT / execution['canonical_result']['path']) == execution['canonical_result']['sha256']
    probe = ('import os, sys; import neuroshard.evolution.assistant_experience_eval as m; '
             'import neuroshard.evolution.assistant_experience_run, neuroshard.evolution.assistant_experience_train, '
             'neuroshard.evolution.assistant_selector, neuroshard.evolution.assistant_experience_gate, neuroshard.evolution.assistant_serving; '
             'root = os.path.abspath("src"); '
             'print("\\n".join(sorted(os.path.relpath(x.__file__) for x in list(sys.modules.values()) '
             'if getattr(x, "__file__", None) and os.path.abspath(x.__file__).startswith(root))))')
    imported = subprocess.check_output([sys.executable, '-c', probe], cwd=ROOT, text=True,
                                       env={'PYTHONPATH': str(ROOT / 'src')}).split()
    assert set(imported) <= set(execution['sources'])
    hours = read(ROOT / 'config/experiments/assistant-experience-development-resources.json')['hours']
    replay = execution['replay_seconds'] if 'a1_served' in execution else 0
    assert execution['prepare_seconds'] + len(evaluated) * execution['worker_seconds'] + replay + 1800 <= hours * 3600
    assert not execution['training_authorized'] and not execution['gpu_launch_authorized']


def test_importing_evaluation_does_not_load_torch():
    import subprocess
    import sys

    probe = 'import sys, neuroshard.evolution.assistant_experience_eval; assert "torch" not in sys.modules'
    subprocess.run([sys.executable, '-c', probe], check=True, cwd=ROOT, env={'PYTHONPATH': str(ROOT / 'src')})
