import copy
import importlib.util
import json
import os
import shutil
import subprocess
import sys
import time

import pytest

from neuroshard.evolution import granite_shard_chain as chain
from neuroshard.evolution import granite_shard_settlement as settlement
from neuroshard.evolution.modular_reference_execution import ROOT, read, sha256

from test_granite_shard_settlement import declared_chain, evidence

PLAN = read(ROOT / chain.PLAN)


def cloud_module():
    spec = importlib.util.spec_from_file_location('chain_cloud', ROOT / 'scripts/modular_reference_cloud.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def chain_evidence():
    fetches, phases, parties, jobs = evidence()
    state = phases['validate']['validator-1']['state']
    states = [{**copy.deepcopy(state), 'root': phases['validate']['validator-1']['root']} for _ in range(4)]
    admissions = {name: {'code': 0, 'log': '', 'height': height} for name, height in (
        ('bond-1', 1), ('bond-2', 1), ('open-honest', 2), ('commit-honest-1', 3), ('commit-honest-2', 3),
        ('open-cheat', 4), ('commit-cheat-1', 5), ('commit-cheat-2', 5), ('proven', 9))}
    admissions['framing'] = {'code': 1, 'log': 'Fraud proof does not verify', 'height': None}
    return fetches, phases, parties, {'honest': jobs['honest'], 'cheated': jobs['cheated']}, admissions, states


def test_assessment_requires_four_agreeing_validators_a_refused_framing_and_settlement_before_the_proof():
    fetches, phases, parties, jobs, admissions, states = chain_evidence()
    report = chain.assess(PLAN, fetches, phases, parties, jobs, admissions, states)
    assert report['passed'], report['checks']
    split = copy.deepcopy(states)
    split[3]['root'] = '00' * 32
    assert not chain.assess(PLAN, fetches, phases, parties, jobs, admissions, split)['checks']['validators_agree']
    assert not chain.assess(PLAN, fetches, phases, parties, jobs, admissions, states[:3])['checks']['validators_agree']
    framed = {**admissions, 'framing': {'code': 0, 'log': '', 'height': 7}}
    assert not chain.assess(PLAN, fetches, phases, parties, jobs, framed, states)['checks']['framing_refused']
    early = {**admissions, 'proven': {'code': 0, 'log': '', 'height': 8}}
    assert not chain.assess(PLAN, fetches, phases, parties, jobs, early, states)['checks']['honest_settled_first']
    dropped = {**admissions, 'commit-cheat-2': {'code': 1, 'log': 'x', 'height': None}}
    assert not chain.assess(PLAN, fetches, phases, parties, jobs, dropped, states)['checks']['honest_work_committed']


def test_the_chain_plan_reuses_settlement_and_pins_its_consensus_binary():
    settle = read(ROOT / settlement.PLAN)
    for key in ('model', 'learning', 'runtime', 'boundaries', 'arm', 'upload', 'target', 'fault'):
        assert PLAN[key] == settle[key]
    for key in ('allocation', 'owner_bond', 'price', 'shards'):
        assert PLAN['ledger'][key] == settle['ledger'][key]
    assert PLAN['signing_packages'] == {**settle['signing_packages'], 'grpcio': '1.83.1', 'protobuf': '6.33.6'}
    params = PLAN['ledger']['params']
    # The honest window must outlast the honest audit and framing; the job deadline must outlast one serving pass.
    assert params['challenge_blocks'] * PLAN['ledger']['block_seconds'] > PLAN['phase_seconds']['audit'] / 2
    assert params['job_blocks'] * PLAN['ledger']['block_seconds'] > PLAN['phase_seconds']['serve']
    local = PLAN['cometbft']['local']
    if os.path.exists(local):
        assert sha256(local) == PLAN['cometbft']['sha256']


def test_the_chain_profile_is_bounded_and_the_freeze_covers_every_imported_source():
    import ast
    import re

    for name, digest in PLAN['contracts'].items():
        assert sha256(ROOT / name) == digest
    assert set(PLAN['contracts']) <= set(PLAN['sources'])
    cloud = cloud_module()
    resources = cloud.resources(chain.PROFILE)
    assert resources['purpose'] == chain.PROFILE and 8 * resources['planning_cap_usd'] <= 96
    seconds = PLAN['phase_seconds']
    window = PLAN['ledger']['params']['challenge_blocks'] * PLAN['ledger']['block_seconds'] * 1.5
    timeline = (seconds['fetch'] + 2 * seconds['chain'] + seconds['sign'] + 2 * (seconds['serve'] + seconds['commit']
                + seconds['audit']) + seconds['frame'] + seconds['prove'])
    assert timeline + resources['setup_seconds'] + resources['copy_seconds'] <= resources['hours'] * 3600
    assert PLAN['chain_seconds'] >= 2 * seconds['serve'] + window
    module, requirements = cloud.GRANITE_PROFILES[chain.PROFILE]
    assert module == 'granite_shard_chain' and requirements == 'docs/granite-shard-chain-requirements.txt'
    probe = ('import os, sys; import neuroshard.evolution.granite_shard_chain, neuroshard.inference.optimistic_app, '
             'neuroshard.inference.optimistic_network, neuroshard.evolution.sharded.granite_audit, '
             'neuroshard.evolution.sharded.granite_serving, neuroshard.evolution.sharded.granite_training, '
             'neuroshard.evolution.assistant_experience_run, neuroshard.evolution.granite_tokenizer, '
             'neuroshard.evolution.assistant_selector, neuroshard.evolution.assistant_workflow, '
             'neuroshard.evolution.assistant_experience_gate; root = os.path.abspath("src"); '
             'print("\\n".join(sorted(os.path.relpath(x.__file__) for x in list(sys.modules.values()) '
             'if getattr(x, "__file__", None) and os.path.abspath(x.__file__).startswith(root))))')
    imported = subprocess.check_output([sys.executable, '-c', probe], cwd=ROOT, text=True,
                                       env={'PYTHONPATH': str(ROOT / 'src')}).split()
    assert set(imported) <= set(PLAN['sources']), set(imported) - set(PLAN['sources'])
    lines = [line.split('#')[0].strip() for line in (ROOT / requirements).read_text().splitlines()]
    pinned = {re.split('[=<>@ ]', line)[0].lower().replace('_', '-') for line in lines if line}
    third_party = set()
    for name in [n for n in PLAN['sources'] if n.startswith('src/') or n == chain.SCRIPT]:
        for node in ast.walk(ast.parse((ROOT / name).read_text())):
            if isinstance(node, ast.Import):
                third_party.update(a.name.split('.')[0] for a in node.names)
            elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
                third_party.add(node.module.split('.')[0])
    third_party -= set(sys.stdlib_module_names) | {'neuroshard', 'granite_switch'}
    aliases = {'huggingface_hub': 'huggingface-hub', 'google': 'protobuf', 'grpc': 'grpcio'}
    assert {aliases.get(t, t) for t in third_party} <= pinned, third_party
    spec = importlib.util.spec_from_file_location('chain_controller', ROOT / 'scripts/granite_shard_chain_cloud.py')
    controller = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(controller)
    for phase in ('chain-init', 'chain-configure', 'frame', 'prove', 'sign-bond', 'serve-cheat', 'audit-honest'):
        assert controller.base.seconds(PLAN, phase) == seconds[phase.split('-')[0]]


ROLE = '''
import json, sys
from neuroshard.evolution import granite_shard_audit as audited, granite_shard_chain as chain
audited.freeze = lambda plan_path=None: {"commit": "rehearsal"}
index, phase, home, store = int(sys.argv[1]), sys.argv[2], sys.argv[3], sys.argv[4]
print(json.dumps({"completed": chain.validator(index, phase, home, store)["completed"]}))
'''


def test_validators_create_identities_and_configure_nodes_that_reach_consensus(tmp_path):
    from neuroshard.inference import optimistic_network as network

    try:
        binary = network.engine_path()
    except ValueError:
        pytest.skip('CometBFT v0.38.26 is not installed')
    if sha256(binary) != PLAN['cometbft']['sha256']:
        pytest.skip('local CometBFT binary differs from the declared one')
    genesis_chain, _, parties, keys, _ = declared_chain()
    homes, stores, identities, processes = {}, {}, {}, []
    env = {**os.environ, 'PYTHONPATH': str(ROOT / 'src')}

    def role(index, phase):
        output = subprocess.check_output([sys.executable, '-c', ROLE, str(index), phase, str(homes[index]), str(stores[index])],
                                         env=env, text=True)
        assert json.loads(output.strip().splitlines()[-1])['completed']
        return read(homes[index] / phase / 'result.json')

    for index in (1, 2, 3, 4):
        homes[index], stores[index] = tmp_path / f'v{index}' / 'home', tmp_path / f'v{index}' / 'store'
        homes[index].mkdir(parents=True)
        stores[index].mkdir(parents=True)
        shutil.copy2(binary, stores[index] / 'cometbft')
        identities[index] = role(index, 'chain-init')
    terms = {**PLAN['ledger'], 'model_root': genesis_chain['model_root'],
             'allocations': {a: PLAN['ledger']['allocation'] for a in parties.values()}}
    genesis = network.genesis_document(identities[1]['template'], terms, [identities[i] for i in (1, 2, 3, 4)],
                                       identities[1]['template']['genesis_time'])
    ports = {i: {'p2p': 31650 + i * 10, 'rpc': 31651 + i * 10, 'abci': 31652 + i * 10} for i in (1, 2, 3, 4)}
    for index in (1, 2, 3, 4):
        peers = ','.join(f'{identities[j]["node_id"]}@127.0.0.1:{ports[j]["p2p"]}' for j in (1, 2, 3, 4) if j != index)
        (homes[index] / 'chain-configure-request.json').write_text(json.dumps(
            {'genesis': genesis, 'peers': peers, 'ports': ports[index], 'p2p_host': '127.0.0.1'}))
        role(index, 'chain-configure')
        assert 'timeout_broadcast_tx_commit = "120s"' in (homes[index] / 'chain' / 'config' / 'config.toml').read_text()
        # Proof checks are covered elsewhere; this rehearsal checks the consensus setup alone.
        (homes[index] / 'chain' / 'settlement.json').write_text('{}')
    try:
        for index in (1, 2, 3, 4):
            node_home = str(homes[index] / 'chain')
            processes.append(subprocess.Popen([sys.executable, '-m', 'neuroshard.inference.optimistic_app', '--home',
                                               node_home, '--port', str(ports[index]['abci'])], env=env,
                                              stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL))
        time.sleep(2)
        for index in (1, 2, 3, 4):
            processes.append(subprocess.Popen([str(stores[index] / 'cometbft'), 'start', '--home', str(homes[index] / 'chain')],
                                              stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL))
        deadline = time.monotonic() + 120
        while time.monotonic() < deadline:
            try:
                states = [network.state(f'http://127.0.0.1:{ports[i]["rpc"]}') for i in (1, 2, 3, 4)]
                if min(s['height'] for s in states) >= 3 and len({s['height'] for s in states}) == 1:
                    break
            except Exception:
                pass
            time.sleep(0.5)
        else:
            pytest.fail('validators did not reach consensus')
    finally:
        for process in processes:
            process.terminate()
        for process in processes:
            process.wait(timeout=30)
    assert len({s['root'] for s in states}) == 1 and states[0]['chain_id'] == PLAN['ledger']['chain_id']
    assert set(states[0]['accounts']) == set(parties.values())
