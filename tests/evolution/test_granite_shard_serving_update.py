import importlib.util
import subprocess
import sys

from neuroshard.evolution import granite_shard_serving as serving
from neuroshard.evolution import granite_shard_serving_update as update
from neuroshard.evolution.modular_reference_execution import ROOT, read, sha256

from test_granite_shard_serving import cloud_module

PLAN = read(ROOT / update.PLAN)
A4 = read(ROOT / serving.PLAN)


def test_the_plan_serves_the_accepted_update_against_a2s_single_host_result():
    accepted = read(ROOT / 'config/experiments/assistant-growth.json')['cohort1']['trainable_sha256']
    assert PLAN['arm'] == {'kind': 'update', 'trainable_sha256': accepted, 'integration_sha256': A4['arm']['integration_sha256']}
    episodes = serving.target(PLAN)
    assert len(episodes) == 24 and PLAN['target']['system'] == 'update'
    third = read(ROOT / PLAN['target']['path'])['replies']['update']
    assert third['serving'] == 'prefix-cache' and third['arm'] == 'update'
    assert third['checkpoint']['trainable_sha256'] == accepted and third['served_projections_converted'] == 16
    assert {key: PLAN[key] for key in ('boundaries', 'model', 'learning', 'eos_ids', 'determinism', 'phase_seconds')} == {
        key: A4[key] for key in ('boundaries', 'model', 'learning', 'eos_ids', 'determinism', 'phase_seconds')}
    assert sorted(PLAN['upload']['files']) == ['integration.json', 'update-checkpoint/manifest.json',
                                               'update-checkpoint/trainable.safetensors']


def test_the_plan_pins_its_contracts_and_sources():
    for name, digest in PLAN['contracts'].items():
        assert sha256(ROOT / name) == digest
    assert set(PLAN['contracts']) <= set(PLAN['sources'])
    for name in (update.PLAN, update.SCRIPT, 'src/neuroshard/evolution/granite_shard_serving_update.py',
                 'src/neuroshard/evolution/granite_shard_serving.py', 'src/neuroshard/evolution/sharded/granite_serving.py'):
        assert name in PLAN['sources']


def test_owners_serve_under_the_update_plan(monkeypatch):
    calls = []
    monkeypatch.setattr(serving, 'owner', lambda *args, **kwargs: calls.append((args, kwargs)) or {'completed': True})
    assert update.owner(2, 'host', 1, 'serve', 'home', 'store')['completed']
    assert calls == [((2, 'host', 1, 'serve', 'home', 'store', 0), {'plan_path': update.PLAN})]


def test_each_owner_is_one_bounded_cpu_host_and_the_launcher_knows_the_execution():
    cloud = cloud_module()
    resources = cloud.resources(update.PROFILE)
    assert not resources['gpu'] and resources['instance_type'] == 'r7i.4xlarge' and resources['planning_cap_usd'] <= 8
    assert cloud.GRANITE_PROFILES[update.PROFILE][0] == 'granite_shard_serving_update'
    spec = importlib.util.spec_from_file_location('serving_launcher', ROOT / 'scripts/granite_shard_serving_cloud.py')
    launcher = spec.loader.get_source('serving_launcher')
    assert '"granite_shard_serving_update"' in launcher


def test_importing_the_update_execution_does_not_load_torch():
    probe = 'import sys, neuroshard.evolution.granite_shard_serving_update; assert "torch" not in sys.modules'
    subprocess.run([sys.executable, '-c', probe], check=True, cwd=ROOT, env={'PYTHONPATH': str(ROOT / 'src')})


def test_the_plan_lists_every_module_an_owner_imports_while_serving():
    probe = ('import importlib, os, sys; '
             '[importlib.import_module("neuroshard.evolution." + m) for m in ("granite_shard_serving_update", '
             '"sharded.granite_serving", "sharded.granite_audit", "sharded.granite_training", "sharded.granite_streams", '
             '"sharded.granite", "sharded.granite_pipeline", "assistant_experience_run", "assistant_experience_eval", '
             '"assistant_workflow_data", "granite_tokenizer", "assistant_selector", "assistant_workflow", '
             '"assistant_serving")]; importlib.import_module("neuroshard.inference.optimistic"); '
             'root = os.path.abspath("src"); print("\\n".join(sorted(os.path.relpath(x.__file__) '
             'for x in list(sys.modules.values()) if getattr(x, "__file__", None) and '
             'os.path.abspath(x.__file__).startswith(root))))')
    imported = subprocess.check_output([sys.executable, '-c', probe], cwd=ROOT, text=True,
                                       env={'PYTHONPATH': str(ROOT / 'src')}).split()
    assert 'src/neuroshard/evolution/sharded/granite_audit.py' in imported
    assert set(imported) <= set(PLAN['sources'])
