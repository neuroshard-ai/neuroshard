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
    return [{**(execute_fixture(c) if c['id'] in successes else failing(c)), 'selected': selected} for c in cases]


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


def test_cpu_profile_uploads_only_declared_regular_files_within_its_allowance(tmp_path):
    cloud = cloud_module()
    resources = cloud.resources(evaluation.PROFILE)
    assert not resources['gpu'] and resources['instance_type'] == 'r7i.4xlarge' and resources['hours'] <= 2
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


def test_importing_evaluation_does_not_load_torch():
    import subprocess
    import sys

    probe = 'import sys, neuroshard.evolution.assistant_experience_eval; assert "torch" not in sys.modules'
    subprocess.run([sys.executable, '-c', probe], check=True, cwd=ROOT, env={'PYTHONPATH': str(ROOT / 'src')})
