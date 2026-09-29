import copy
import importlib.metadata
import importlib.util
import subprocess
import sys

from neuroshard.evolution import granite_shard_audit as audited
from neuroshard.evolution import granite_shard_serving as serving
from neuroshard.evolution.modular_reference_execution import ROOT, read, sha256

from test_granite_shard_serving import passing_evidence

PLAN = read(ROOT / audited.PLAN)


def cloud_module():
    spec = importlib.util.spec_from_file_location('audit_cloud', ROOT / 'scripts/modular_reference_cloud.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def evidence():
    from neuroshard.evolution.sharded import granite

    owner_fetches, _, served = passing_evidence()
    tensors, _ = audited.shard.inventory(PLAN)
    owned = {r: sum(s['end'] - s['begin'] for n, s in tensors['tensors'].items() if granite.owner(n, PLAN['boundaries']) == r)
             for r in (1, 2)}
    keys = {1: 'a' * 64, 2: 'b' * 64}
    fetches = {'owners': [dict(owner_fetches[0]), {**owner_fetches[1], 'public_key': keys[1]},
                          {**owner_fetches[2], 'public_key': keys[2]}],
               'auditors': {r: {'completed': True, 'fetched_bytes': owned[r]} for r in (1, 2)}}
    clean = {r: {'valid': True, 'signed': True, 'public_key': keys[r], 'seconds': 400.0} for r in (1, 2)}
    phases = {'serve-honest': served,
              'audit-honest': {f'auditor-{r}': clean[r] for r in (1, 2)},
              'audit-cheat': {'auditor-1': {'valid': False, 'signed': True, 'public_key': keys[1], 'first_mismatch': 431,
                                            'fault_forward': 200, 'proof': {'mismatch': 431}},
                              'auditor-2': clean[2]},
              'verify': {'auditor-1': {'accepted': True}}}
    return fetches, phases


def test_assessment_requires_clean_honest_audits_and_an_exact_proven_attributed_fault():
    fetches, phases = evidence()
    report = audited.assess(PLAN, fetches, phases)
    assert report['passed'] and report['fault']['named_forward'] == 200
    shifted = copy.deepcopy(phases)
    shifted['audit-cheat']['auditor-1']['fault_forward'] = 199
    assert not audited.assess(PLAN, fetches, shifted)['checks']['fault_named']
    rejected = copy.deepcopy(phases)
    rejected['verify']['auditor-1']['accepted'] = False
    assert not audited.assess(PLAN, fetches, rejected)['checks']['proof_accepted']
    framed = copy.deepcopy(phases)
    framed['audit-cheat']['auditor-2'] = {'valid': False}
    assert not audited.assess(PLAN, fetches, framed)['checks']['no_false_blame']
    impostor = copy.deepcopy(phases)
    impostor['audit-honest']['auditor-2']['public_key'] = 'c' * 64
    assert not audited.assess(PLAN, fetches, impostor)['checks']['honest_audits']
    heavy = copy.deepcopy(fetches)
    heavy['auditors'][1]['fetched_bytes'] += 1
    assert not audited.assess(PLAN, heavy, phases)['checks']['light_auditors']


def test_audit_plan_shares_the_serving_target_and_declares_its_fault():
    serving_plan = read(ROOT / serving.PLAN)
    for key in ('arm', 'target', 'boundaries', 'upload', 'learning', 'model'):
        assert PLAN[key] == serving_plan[key]
    assert PLAN['audited_owners'] == [1, 2] and PLAN['fault']['rank'] == 1 and PLAN['fault']['at'] == 200


def test_audit_profile_is_bounded_and_the_freeze_covers_every_imported_source():
    for name, digest in PLAN['contracts'].items():
        assert sha256(ROOT / name) == digest
    assert set(PLAN['contracts']) <= set(PLAN['sources'])
    cloud = cloud_module()
    resources = cloud.resources(audited.PROFILE)
    assert resources['purpose'] == audited.PROFILE and 5 * resources['planning_cap_usd'] <= 50
    seconds = PLAN['phase_seconds']
    timeline = seconds['fetch'] + 2 * (seconds['serve'] + seconds['audit']) + seconds['verify']
    assert timeline + resources['setup_seconds'] + resources['copy_seconds'] <= resources['hours'] * 3600
    probe = ('import os, sys; import neuroshard.evolution.granite_shard_audit, neuroshard.evolution.sharded.granite_audit, '
             'neuroshard.evolution.sharded.granite_serving, neuroshard.evolution.sharded.granite_training, '
             'neuroshard.evolution.assistant_experience_run, neuroshard.evolution.granite_tokenizer, '
             'neuroshard.evolution.assistant_selector, neuroshard.evolution.assistant_workflow, '
             'neuroshard.evolution.assistant_experience_gate; root = os.path.abspath("src"); '
             'print("\\n".join(sorted(os.path.relpath(x.__file__) for x in list(sys.modules.values()) '
             'if getattr(x, "__file__", None) and os.path.abspath(x.__file__).startswith(root))))')
    imported = subprocess.check_output([sys.executable, '-c', probe], cwd=ROOT, text=True,
                                       env={'PYTHONPATH': str(ROOT / 'src')}).split()
    assert set(imported) <= set(PLAN['sources']), set(imported) - set(PLAN['sources'])
    spec = importlib.util.spec_from_file_location('audit_controller', ROOT / 'scripts/granite_shard_audit_cloud.py')
    controller = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(controller)
    assert controller.base.seconds(PLAN, 'serve-cheat') == seconds['serve']


def test_the_audit_runtime_installs_every_package_its_host_code_imports():
    import ast
    import re

    cloud = cloud_module()
    module, requirements = cloud.GRANITE_PROFILES[audited.PROFILE]
    assert module == 'granite_shard_audit'

    def pins(path):
        lines = [line.split('#')[0].strip() for line in (ROOT / path).read_text().splitlines()]
        return {re.split('[=<>@ ]', line)[0].lower().replace('_', '-'): line for line in lines if line}

    pinned, reference = pins(requirements), pins('docs/granite-reference-requirements.txt')
    signing = {name: f'{name}=={version}' for name, version in PLAN['signing_packages'].items()}
    assert pinned == {**reference, **signing}
    for name, version in PLAN['signing_packages'].items():
        assert importlib.metadata.version(name) == version
    host_code = [name for name in PLAN['sources'] if name.startswith('src/') or name == audited.SCRIPT]
    installed_by_bootstrap = {'granite_switch'}
    imported = set()
    for name in host_code:
        for node in ast.walk(ast.parse((ROOT / name).read_text())):
            if isinstance(node, ast.Import):
                imported.update(alias.name.split('.')[0] for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
                imported.add(node.module.split('.')[0])
    third_party = imported - set(sys.stdlib_module_names) - {'neuroshard'} - installed_by_bootstrap
    names = {'huggingface_hub': 'huggingface-hub'}
    assert {names.get(top, top) for top in third_party} <= set(pinned), third_party
    assert 'cryptography' in third_party


def test_importing_the_audit_execution_does_not_load_torch():
    probe = 'import sys, neuroshard.evolution.granite_shard_audit; assert "torch" not in sys.modules'
    subprocess.run([sys.executable, '-c', probe], check=True, cwd=ROOT, env={'PYTHONPATH': str(ROOT / 'src')})
