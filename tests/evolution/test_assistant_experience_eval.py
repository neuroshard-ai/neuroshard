import copy

import pytest

from neuroshard.evolution import assistant_experience_eval as evaluation
from neuroshard.evolution import assistant_workflow_data as data
from neuroshard.evolution.modular_reference_execution import ROOT, read

from test_assistant_workflow import execute_fixture, policy, reference_texts, reply
from neuroshard.evolution import assistant_workflow as workflow

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
    return [{**(execute_fixture(c) if c['id'] in successes else failing(c)), 'selected': selected} for c in cases]


def test_assessment_rescores_every_episode_and_reports_forced_anchor_forgetting():
    cases = data.cases('development')
    ids = [c['id'] for c in cases]
    base = canonical(cases, set(ids[:6]))
    anchors = [{'id': 'granite-chat-booking', 'passed': True}, {'id': 'granite-tool-timer', 'passed': False}]
    reply_rows = {'systems': {
        'update': {'episodes': system(cases, set(ids[:20]), 'update'), 'forced_anchors': anchors},
        'addition': {'episodes': system(cases, set(ids[:20]), 'addition'), 'forced_anchors': anchors[:1] + [
            {'id': 'granite-tool-timer', 'passed': True}]}}}
    report = evaluation.assess(PLAN, cases, base, reply_rows)
    assert report['passed'] and report['correct'] == {'parent': 6, 'update': 20, 'addition': 20}
    assert report['forced_anchor_forgetting']['update']['lost_protected'] == ['granite-tool-timer']
    assert report['forced_anchor_forgetting']['addition']['lost_protected'] == []
    assert report['selected_arm_episodes'] == {'update': 24, 'addition': 24}
    forged = copy.deepcopy(reply_rows)
    forged['systems']['addition']['episodes'][0]['score']['passed'] = False
    with pytest.raises(ValueError, match='rescore'):
        evaluation.assess(PLAN, cases, base, forged)


def test_uploaded_arms_must_match_pinned_digests(tmp_path):
    from neuroshard.evolution.modular_reference_execution import save, sha256

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
