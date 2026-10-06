import subprocess
import sys

from neuroshard.evolution import assistant_growth_router3 as router3
from neuroshard.evolution import assistant_routing as routing
from neuroshard.evolution.modular_reference_execution import ROOT, read, sha256

from test_assistant_growth_baseline import cloud_module

DECLARATION = read(ROOT / router3.DECLARATION)


def test_message_features_run_the_shared_prefix_once(monkeypatch):
    prefixes, computed = [], []
    monkeypatch.setattr(routing, 'message_prefix', lambda *args: prefixes.append(args) or {'ids': [1]})
    monkeypatch.setattr(routing, 'message_feature',
                        lambda model, tokenizer, policy, user, device, prefix: computed.append(user) or [len(user)])
    features = router3.message_features('parent', 'tokenizer', 'policy', {'b#0': 'second', 'a#0': 'first'})
    assert features == {'a#0': [5], 'b#0': [6]} and computed == ['first', 'second'] and len(prefixes) == 1


def test_the_refit_stays_within_the_ceiling_on_one_cpu_host():
    budget = DECLARATION['budget']
    reports = [read(ROOT / f'config/experiments/assistant-growth-{name}-report.json')
               for name in ('stage1', 'round2', 'round3', 'round4', 'router2')]
    spent = sum(r['resources_finished']['conservative_compute_usd'] for r in reports) + sum(
        read(ROOT / f'config/experiments/assistant-growth-{name}-report.json')['conservative_compute_usd']
        for name in ('development', 'development4', 'development5'))
    assert budget['spent_usd'] == round(spent, 2)
    total = budget['spent_usd'] + budget['router_usd'] + budget['development_usd'] + budget['confirmation_usd']
    assert total <= budget['ceiling_usd'] == 110 and f'${total:.2f}' in budget['allowances']
    cloud = cloud_module()
    assert router3.PROFILE not in cloud.GPU_PROFILES and router3.PROFILE not in cloud.UPLOAD_PROFILES
    assert cloud.LONG_CPU_PROFILES[router3.PROFILE] == (2, budget['router_usd'])
    assert cloud.GRANITE_PROFILES[router3.PROFILE][0] == 'assistant_growth_router3'
    assert cloud.remote_command(router3.PROFILE)[1].endswith(router3.SCRIPT)
    assert DECLARATION['router']['feature'].startswith(router3.FEATURE)


def test_refit_inventory_pins_contracts_sources_and_the_canonical_runtime():
    execution = read(ROOT / router3.EXECUTION)
    for name, digest in execution['contracts'].items():
        assert sha256(ROOT / name) == digest
    assert set(execution['contracts']) <= set(execution['sources']) and router3.SCRIPT in execution['sources']
    probe = ('import os, sys; import neuroshard.evolution.assistant_growth_router3; assert "torch" not in sys.modules; '
             'import neuroshard.evolution.assistant_selector, neuroshard.evolution.assistant_experience_run, '
             'neuroshard.evolution.assistant_experience_train, neuroshard.evolution.assistant_calendar; '
             'root = os.path.abspath("src"); print("\\n".join(sorted(os.path.relpath(x.__file__) '
             'for x in list(sys.modules.values()) if getattr(x, "__file__", None) and '
             'os.path.abspath(x.__file__).startswith(root))))')
    imported = subprocess.check_output([sys.executable, '-c', probe], cwd=ROOT, text=True,
                                       env={'PYTHONPATH': str(ROOT / 'src')}).split()
    assert set(imported) <= set(execution['sources'])
    canonical = read(ROOT / 'config/experiments/assistant-workflow-canonical-execution.json')
    assert all(execution[key] == canonical[key] for key in ('packages', 'python', 'required_cpu_flags', 'environment'))
    resources = cloud_module().resources(router3.PROFILE)
    assert not resources['gpu'] and 'upload' not in resources
    assert (execution['prepare_seconds'] + execution['worker_seconds'] + resources['setup_seconds']
            + resources['copy_seconds'] + 600 <= resources['hours'] * 3600)


def test_importing_the_refit_does_not_load_torch():
    probe = 'import sys, neuroshard.evolution.assistant_growth_router3; assert "torch" not in sys.modules'
    subprocess.run([sys.executable, '-c', probe], check=True, cwd=ROOT, env={'PYTHONPATH': str(ROOT / 'src')})
