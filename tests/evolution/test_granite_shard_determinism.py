import copy
import importlib.util
import subprocess
import sys

from neuroshard.evolution import granite_shard_determinism as determinism
from neuroshard.evolution.modular_reference_execution import ROOT, read, sha256

PLAN = read(ROOT / determinism.PLAN)


def cloud_module():
    spec = importlib.util.spec_from_file_location('determinism_cloud', ROOT / 'scripts/modular_reference_cloud.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def launches(count=6):
    def owner():
        return {'completed': True, 'trace': ['a', 'b', 'c']}

    rows = [{**owner(), 'receipt': {'references': [[-0.1, -2.0]] * 8}}, owner(),
            {**owner(), 'trainable_sha256': {'lora': 'f' * 64}}]
    return {f'ring-{index}': copy.deepcopy(rows) for index in range(count)}


def test_divergence_is_located_at_its_first_owner_and_message():
    assert determinism.first_divergence([['a', 'b'], ['a', 'b']]) is None
    assert determinism.first_divergence([['a', 'b', 'c'], ['a', 'x', 'c']]) == 1
    assert determinism.first_divergence([['a', 'b'], ['a', 'b', 'c']]) == 2
    collected = launches()
    report = determinism.assess_phases(PLAN, [], collected)
    assert report['reproducible'] and report['launches'] == 6
    collected['ring-4'][1]['trace'][2] = 'z'
    collected['ring-4'][0]['receipt']['references'][0] = [-0.2, -2.0]
    report = determinism.assess_phases(PLAN, [], collected)
    assert not report['reproducible'] and report['owners'][1]['first_divergence'] == 2
    assert report['owners'][0]['distinct_traces'] == 1 and report['distinct_references'] == 2


def test_the_diagnostic_replays_the_declared_training_job():
    job = determinism.job(PLAN)
    training = read(ROOT / 'config/experiments/granite-shard-training.json')
    assert job['trace'] and job['spec'] == {**training['spec'], 'steps': 1} and job['arm'] == 'addition'
    assert determinism.phases(PLAN) == ('fetch',) + tuple(f'ring-{i}' for i in range(6))
    assert set(determinism.phases(PLAN)) <= set(determinism.PHASES)


def test_determinism_profile_is_bounded_and_the_freeze_covers_every_imported_source():
    for name, digest in PLAN['contracts'].items():
        assert sha256(ROOT / name) == digest
    assert set(PLAN['contracts']) <= set(PLAN['sources'])
    cloud = cloud_module()
    resources = cloud.resources(determinism.PROFILE)
    assert resources['purpose'] == determinism.PROFILE and 3 * resources['planning_cap_usd'] <= 24
    seconds = PLAN['phase_seconds']['fetch'] + PLAN['launches'] * PLAN['phase_seconds']['ring']
    assert seconds + resources['setup_seconds'] + resources['copy_seconds'] <= resources['hours'] * 3600
    probe = ('import os, sys; import neuroshard.evolution.granite_shard_determinism, '
             'neuroshard.evolution.sharded.granite_training, neuroshard.evolution.assistant_experience_train; '
             'root = os.path.abspath("src"); '
             'print("\\n".join(sorted(os.path.relpath(x.__file__) for x in list(sys.modules.values()) '
             'if getattr(x, "__file__", None) and os.path.abspath(x.__file__).startswith(root))))')
    imported = subprocess.check_output([sys.executable, '-c', probe], cwd=ROOT, text=True,
                                       env={'PYTHONPATH': str(ROOT / 'src')}).split()
    assert set(imported) <= set(PLAN['sources']), set(imported) - set(PLAN['sources'])
