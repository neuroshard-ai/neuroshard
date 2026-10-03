import importlib.util
import re
import subprocess
import sys

import pytest

from neuroshard.evolution import assistant_growth_resume as resume
from neuroshard.evolution import assistant_growth_run as growth
from neuroshard.evolution.modular_reference_execution import ROOT, read, save

DECLARATION = read(ROOT / resume.DECLARATION)


def cloud_module():
    spec = importlib.util.spec_from_file_location('resume_cloud', ROOT / 'scripts/modular_reference_cloud.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_trained_units_must_match_the_first_hosts_training_manifests(tmp_path):
    pinned = {'U2': 'a' * 64, 'L2': 'b' * 64}
    for unit, digest in pinned.items():
        save(tmp_path / growth.CHECKPOINTS[unit] / 'manifest.json', {'trainable_sha256': digest})
    resume.verify_trained(tmp_path, pinned)
    with pytest.raises(ValueError, match='L2 differs'):
        resume.verify_trained(tmp_path, {**pinned, 'L2': 'c' * 64})


def test_the_resumption_stays_within_the_unchanged_ceiling():
    plan = read(ROOT / growth.PLAN)
    assert DECLARATION['amends'] == growth.PLAN and DECLARATION['gates_unchanged']
    allowances = [int(x) for x in re.findall(r'\$(\d+)', DECLARATION['budget']['allowances'])]
    assert allowances[:4] == [21, 12, 18, 47] and sum(allowances[:4]) == allowances[4]
    assert allowances[4] <= plan['budget']['ceiling_usd'] == allowances[5]
    assert DECLARATION['budget']['resumption_usd'] == allowances[1]


def test_the_resumption_profile_is_one_gpu_host_running_only_the_integration():
    cloud = cloud_module()
    assert resume.PROFILE in cloud.GPU_PROFILES
    assert cloud.GPU_PROFILES[resume.PROFILE] == cloud.GPU_PROFILES[growth.PROFILE]
    assert cloud.GRANITE_PROFILES[resume.PROFILE] == ('assistant_growth_resume', cloud.GRANITE_PROFILES[growth.PROFILE][1])
    assert cloud.UPLOAD_PROFILES[resume.PROFILE] == growth.UPLOADED
    assert cloud.remote_command(resume.PROFILE)[1].endswith(resume.SCRIPT)


def test_importing_the_resumption_does_not_load_torch():
    probe = 'import sys, neuroshard.evolution.assistant_growth_resume; assert "torch" not in sys.modules'
    subprocess.run([sys.executable, '-c', probe], check=True, cwd=ROOT, env={'PYTHONPATH': str(ROOT / 'src')})
