import copy
import importlib.util
import subprocess
import sys

from neuroshard.evolution import granite_shard_serving as serving
from neuroshard.evolution import granite_shard_throughput as throughput
from neuroshard.evolution.modular_reference_execution import ROOT, read, sha256

from test_granite_shard_serving import passing_evidence

PLAN = read(ROOT / throughput.PLAN)


def cloud_module():
    spec = importlib.util.spec_from_file_location('throughput_cloud', ROOT / 'scripts/modular_reference_cloud.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def phases(sequential_seconds=2000.0, concurrent_seconds=900.0, peak=3):
    fetches, _, served = passing_evidence()
    sequential, concurrent = copy.deepcopy(served), copy.deepcopy(served)
    sequential[0]['episodes_seconds'] = sequential_seconds
    concurrent[0].update(episodes_seconds=concurrent_seconds, peak_in_flight=peak)
    return fetches, {'sequential': sequential, 'concurrent': concurrent}


def test_throughput_shares_the_serving_target_and_arm():
    serving_plan = read(ROOT / serving.PLAN)
    for key in ('arm', 'target', 'boundaries', 'upload', 'learning', 'model'):
        assert PLAN[key] == serving_plan[key]
    assert PLAN['streams'] == {'sequential': None, 'concurrent': 3}


def test_assessment_requires_identical_tokens_both_ways_and_the_declared_speedup():
    fetches, value = phases()
    report = throughput.assess_phases(PLAN, fetches, value)
    assert report['passed'] and round(report['throughput_ratio'], 2) == 2.22
    slow = throughput.assess_phases(PLAN, *phases(concurrent_seconds=1500.0))
    assert not slow['checks']['throughput'] and slow['checks']['concurrent_agreement']
    serial = throughput.assess_phases(PLAN, *phases(peak=1))
    assert not serial['checks']['overlap']
    fetches, drift = phases()
    drift['concurrent'][0]['episodes'][2]['generations'][0]['reused_prefix_tokens'] += 1
    report = throughput.assess_phases(PLAN, fetches, drift)
    assert not report['checks']['concurrent_agreement'] and report['checks']['sequential_agreement']


def test_throughput_profile_is_bounded_and_the_freeze_covers_every_imported_source():
    for name, digest in PLAN['contracts'].items():
        assert sha256(ROOT / name) == digest
    assert set(PLAN['contracts']) <= set(PLAN['sources'])
    cloud = cloud_module()
    resources = cloud.resources(throughput.PROFILE)
    assert resources['purpose'] == throughput.PROFILE and 3 * resources['planning_cap_usd'] <= 24
    assert sum(PLAN['phase_seconds'].values()) + resources['setup_seconds'] + resources['copy_seconds'] \
        <= resources['hours'] * 3600
    probe = ('import os, sys; import neuroshard.evolution.granite_shard_throughput, '
             'neuroshard.evolution.sharded.granite_serving, neuroshard.evolution.sharded.granite_streams, '
             'neuroshard.evolution.sharded.granite_training, neuroshard.evolution.assistant_experience_run, '
             'neuroshard.evolution.granite_tokenizer, neuroshard.evolution.assistant_selector, '
             'neuroshard.evolution.assistant_workflow, neuroshard.evolution.assistant_experience_gate; '
             'root = os.path.abspath("src"); '
             'print("\\n".join(sorted(os.path.relpath(x.__file__) for x in list(sys.modules.values()) '
             'if getattr(x, "__file__", None) and os.path.abspath(x.__file__).startswith(root))))')
    imported = subprocess.check_output([sys.executable, '-c', probe], cwd=ROOT, text=True,
                                       env={'PYTHONPATH': str(ROOT / 'src')}).split()
    assert set(imported) <= set(PLAN['sources']), set(imported) - set(PLAN['sources'])
    assert not PLAN['training_authorized'] and not PLAN['gpu_launch_authorized'] and PLAN['attempts'] == 1
