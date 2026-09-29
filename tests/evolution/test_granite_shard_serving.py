import copy
import hashlib
import importlib.util
import json
import subprocess
import sys

import pytest

from neuroshard.evolution import granite_shard_serving as serving
from neuroshard.evolution.modular_reference_execution import ROOT, read, save, sha256

PLAN = read(ROOT / serving.PLAN)


def cloud_module():
    spec = importlib.util.spec_from_file_location('serving_cloud', ROOT / 'scripts/modular_reference_cloud.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_target_is_the_pinned_round4_served_addition_episodes():
    episodes = serving.target(PLAN)
    assert len(episodes) == 24 and all(row['selected'] in ('arm', 'parent') for row in episodes)
    assert sum(row['score']['passed'] for row in episodes) == 18
    development = read(ROOT / 'config/experiments/assistant-experience-development-execution.json')
    assert PLAN['arm'] == {'trainable_sha256': development['arms']['addition']['trainable_sha256'],
                           'integration_sha256': development['arms']['integration_sha256']}
    with pytest.raises(ValueError, match='development result changed'):
        serving.target({**PLAN, 'target': {**PLAN['target'], 'sha256': '0' * 64}})


def test_uploaded_arm_and_gate_must_match_the_development_pins(tmp_path):
    (tmp_path / 'addition-checkpoint').mkdir()
    (tmp_path / 'addition-checkpoint' / 'trainable.safetensors').write_bytes(b'tensors')
    digest = hashlib.sha256(b'tensors').hexdigest()
    save(tmp_path / 'addition-checkpoint' / 'manifest.json', {'trainable_sha256': digest})
    save(tmp_path / 'integration.json', {'arms': {'addition': {'gate': {'rule': 'constant-arm'}}}})
    plan = {**PLAN, 'arm': {'trainable_sha256': digest, 'integration_sha256': sha256(tmp_path / 'integration.json')}}
    arm, gate = serving.arm_files(plan, tmp_path)
    assert arm == tmp_path / 'addition-checkpoint' and gate == {'rule': 'constant-arm'}
    with pytest.raises(ValueError, match='arm differs'):
        serving.arm_files({**plan, 'arm': {**plan['arm'], 'trainable_sha256': '0' * 64}}, tmp_path)
    with pytest.raises(ValueError, match='gate differs'):
        serving.arm_files({**plan, 'arm': {**plan['arm'], 'integration_sha256': '0' * 64}}, tmp_path)


def passing_evidence():
    episodes = copy.deepcopy(serving.target(PLAN))
    pinned = read(ROOT / 'config/experiments/assistant-workflow-canonical.json')['tokenizer']
    owner = {'completed': True, 'peak_rss_bytes': 6 * 2 ** 30, 'sent_bytes': 10, 'arm_sha256': None}
    served = [{**owner, 'episodes': episodes, 'tokenizer': {'pipeline_sha256': pinned['pipeline_sha256'],
                                                           'fixture_sha256': pinned['fixture_sha256']}},
              owner, {**owner, 'arm_sha256': PLAN['arm']['trainable_sha256']}]
    rows = [{'rank': r, 'process': i, 'digests': ['a' * 64, 'a' * 64, 'a' * 64]} for i in range(6) for r in range(3)]
    return [{'completed': True}] * 3, rows, served


def test_assessment_requires_every_served_generation_to_match():
    fetches, rows, served = passing_evidence()
    report = serving.assess(PLAN, fetches, rows, served)
    assert report['passed'] and report['correct'] == report['expected_correct'] == 18
    assert report['determinism']['distinct_first_pass_digests'] == {0: 1, 1: 1, 2: 1}
    drift = copy.deepcopy(served)
    generation = drift[0]['episodes'][3]['generations'][0]
    generation['reused_prefix_tokens'] += 1
    report = serving.assess(PLAN, fetches, rows, drift)
    assert not report['checks']['agreement'] and report['mismatches'] == [drift[0]['episodes'][3]['id']]
    flipped = copy.deepcopy(served)
    flipped[0]['episodes'][0]['selected'] = 'parent' if flipped[0]['episodes'][0]['selected'] == 'arm' else 'arm'
    assert not serving.assess(PLAN, fetches, rows, flipped)['checks']['agreement']
    leaked = copy.deepcopy(served)
    leaked[1]['arm_sha256'] = PLAN['arm']['trainable_sha256']
    assert not serving.assess(PLAN, fetches, rows, leaked)['checks']['arm_on_its_owner']
    varied = copy.deepcopy(rows)
    varied[3]['digests'][0] = 'b' * 64
    report = serving.assess(PLAN, fetches, varied, served)
    assert report['passed'] and report['determinism']['distinct_first_pass_digests'][varied[3]['rank']] == 2
    forged = copy.deepcopy(served)
    forged[0]['episodes'][0]['score']['passed'] = not forged[0]['episodes'][0]['score']['passed']
    with pytest.raises(ValueError, match='rescore'):
        serving.assess(PLAN, fetches, rows, forged)


def test_serving_profile_is_bounded_and_the_freeze_covers_every_imported_source():
    for name, digest in PLAN['contracts'].items():
        assert sha256(ROOT / name) == digest
    assert set(PLAN['contracts']) <= set(PLAN['sources'])
    cloud = cloud_module()
    resources = cloud.resources(serving.PROFILE)
    assert resources['instance_type'] == 'r7i.4xlarge' and not resources['gpu'] and resources['purpose'] == serving.PROFILE
    assert 3 * resources['planning_cap_usd'] <= 24
    seconds = (PLAN['phase_seconds']['fetch'] + PLAN['determinism']['processes'] * PLAN['phase_seconds']['determinism']
               + PLAN['phase_seconds']['serve'])
    assert seconds + resources['setup_seconds'] + resources['copy_seconds'] <= resources['hours'] * 3600
    assert cloud.GRANITE_PROFILES[serving.PROFILE][0] == 'granite_shard_serving'
    assert not any(name.endswith('optimizer.pt') for name in PLAN['upload']['files'])
    probe = ('import os, sys; import neuroshard.evolution.granite_shard_serving, '
             'neuroshard.evolution.sharded.granite_serving, neuroshard.evolution.sharded.granite_training, '
             'neuroshard.evolution.assistant_experience_run, neuroshard.evolution.granite_tokenizer, '
             'neuroshard.evolution.assistant_selector, neuroshard.evolution.assistant_workflow, '
             'neuroshard.evolution.assistant_experience_gate; root = os.path.abspath("src"); '
             'print("\\n".join(sorted(os.path.relpath(x.__file__) for x in list(sys.modules.values()) '
             'if getattr(x, "__file__", None) and os.path.abspath(x.__file__).startswith(root))))')
    imported = subprocess.check_output([sys.executable, '-c', probe], cwd=ROOT, text=True,
                                       env={'PYTHONPATH': str(ROOT / 'src')}).split()
    assert set(imported) <= set(PLAN['sources']), set(imported) - set(PLAN['sources'])
    assert not PLAN['training_authorized'] and not PLAN['gpu_launch_authorized'] and PLAN['attempts'] == 1


def test_importing_the_serving_execution_does_not_load_torch():
    probe = 'import sys, neuroshard.evolution.granite_shard_serving; assert "torch" not in sys.modules'
    subprocess.run([sys.executable, '-c', probe], check=True, cwd=ROOT, env={'PYTHONPATH': str(ROOT / 'src')})
