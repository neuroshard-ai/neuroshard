"""Settlement of real-model sharded serving through CometBFT consensus.

The settlement execution's owners, auditor and declared fault, with the ledger run by
four CometBFT validators on separate hosts, each holding only shard 1. Transactions
enter through the validators' mempools and blocks come from consensus. The framing
attempt arrives inside the honest job's window; the fraud proof arrives inside the
cheating job's window, after the honest job has settled.
"""

import hashlib
import subprocess
from pathlib import Path

from neuroshard.evolution import granite_shard_audit as audited
from neuroshard.evolution import granite_shard_execution as shard
from neuroshard.evolution import granite_shard_serving as serving
from neuroshard.evolution import granite_shard_settlement as settlement
from neuroshard.evolution.modular_reference_execution import ROOT, read, save

PLAN = 'config/experiments/granite-shard-chain.json'
SCRIPT = 'scripts/run_granite_shard_chain.py'
PROFILE = 'granite-shard-chain'
UPLOADED = serving.UPLOADED
MODEL_INVENTORY = settlement.MODEL_INVENTORY
OWNER_PHASES = settlement.OWNER_PHASES
AUDITOR_PHASES = ('fetch', 'audit-honest', 'audit-cheat', 'frame', 'prove')
VALIDATOR_PHASES = ('fetch', 'chain-init', 'chain-configure')
PORTS = {'p2p': 26656, 'rpc': 26657, 'abci': 26658}


def committed_sources(root=ROOT):
    return shard.committed_sources(root, PLAN)


def configure():
    shard.configure()


def freeze():
    return audited.freeze(PLAN)


def owner(rank, address, port, phase, home, store):
    return settlement.owner(rank, address, port, phase, home, store, plan_path=PLAN)


def auditor(phase, home, store):
    """The light auditor of shard 1: replay logs, then sign the framing attempt and its own fraud proof separately."""
    home, store = Path(home), Path(store)
    if phase not in AUDITOR_PHASES:
        raise ValueError('unsupported chain auditor role')
    if phase in ('fetch', 'audit-honest', 'audit-cheat'):
        result = audited.auditor(1, phase, home, store, plan_path=PLAN)
        if phase == 'fetch':
            save(home / 'accounts.json', {'auditor': settlement.account(store)[1],
                                          'accuser': settlement.account(store, 'accuser')[1]}, exclusive=True)
        return result
    shard.configure()
    source = freeze()
    request = read(home / f'{phase}-request.json')
    (home / 'bundles').mkdir(exist_ok=True)
    if phase == 'frame':
        made = settlement.framing_challenge(home, store, request)
        settlement.stash(home / 'serve-honest' / 'log-1', store)
    else:
        made = settlement.proven_challenge(home, store, request)
        settlement.stash(home / 'serve-cheat' / 'log-1', store)
    result = {'freeze': source, **made, 'completed': True}
    save(home / phase / 'result.json', result, exclusive=True)
    return result


def cometbft(store, plan):
    """The uploaded consensus binary, checked against the declared digest and version."""
    binary = Path(store) / 'cometbft'
    if hashlib.sha256(binary.read_bytes()).hexdigest() != plan['cometbft']['sha256']:
        raise ValueError('CometBFT binary differs from the declaration')
    version = subprocess.check_output([str(binary), 'version'], text=True).strip()
    if version != plan['cometbft']['version']:
        raise ValueError('CometBFT version differs from the declaration')
    return binary


def validator(index, phase, home, store):
    """A validator holding shard 1: fetch it, create its consensus identity, or configure its node."""
    from neuroshard.inference import optimistic_network as network

    shard.configure()
    source = freeze()
    plan = read(ROOT / PLAN)
    home, store = Path(home), Path(store)
    if index not in (1, 2, 3, 4) or phase not in VALIDATOR_PHASES:
        raise ValueError('unsupported chain validator role')
    binary = cometbft(store, plan)
    node_home = home / 'chain'
    if phase == 'fetch':
        receipt = {'freeze': source, 'validator': index, **shard.prepare(plan, 1, store), 'cometbft': str(binary),
                   'completed': True}
        save(home / 'fetch.json', receipt, exclusive=True)
        return receipt
    if phase == 'chain-init':
        subprocess.run([str(binary), 'init', '--home', str(node_home)], check=True, capture_output=True)
        result = {'freeze': source, 'validator': index, **network.identity(str(binary), node_home),
                  'template': read(node_home / 'config' / 'genesis.json'), 'completed': True}
    else:
        request = read(home / 'chain-configure-request.json')
        threads = read(ROOT / shard.CANONICAL)['resources']['threads']
        node = {'runtime': 'granite-shard', 'threads': threads, 'bundles': str(home / 'bundles'),
                'shards': {'1': {'config': str(store / 'config'), 'shard': str(store / 'shard')}}}
        (home / 'bundles').mkdir(exist_ok=True)
        network.configure_node(node_home, request['genesis'], node, request['peers'], request.get('ports', PORTS),
                               plan['ledger']['block_seconds'], request.get('p2p_host', '0.0.0.0'))
        result = {'freeze': source, 'validator': index, 'genesis_sha256': hashlib.sha256(
            (node_home / 'config' / 'genesis.json').read_bytes()).hexdigest(), 'completed': True}
    save(home / phase / 'result.json', result, exclusive=True)
    return result


def assess(plan, fetches, phases, parties, jobs, admissions, states):
    """Serving agreement, four agreeing validators, a refused framing, a proven fault and exact settlement."""
    honest = phases['serve-honest']
    agreement = serving.assess(plan, fetches['owners'], [], honest)
    keys = {r: fetches['owners'][r].get('log_key') for r in (1, 2)}
    state = states[0] if states else {'results': {}, 'owners': {}, 'accounts': {}}
    results, owners = state.get('results', {}), state.get('owners', {})
    expected = settlement.expected_balances(plan, parties)
    audit = phases.get('audit-cheat') or {}
    committed = [admissions.get(name) or {} for name in
                 ('bond-1', 'bond-2', 'open-honest', 'commit-honest-1', 'commit-honest-2', 'open-cheat',
                  'commit-cheat-1', 'commit-cheat-2', 'proven')]
    framing = admissions.get('framing') or {}
    checks = {
        'honest_agreement': agreement['passed'],
        'validators_agree': len(states) == 4 and len({s.get('root') for s in states}) == 1
        and len({s.get('height') for s in states}) == 1,
        'honest_work_committed': all(a.get('code') == 0 and a.get('height') for a in committed[:8]),
        'framing_refused': framing.get('code') == 1 and framing.get('log') == 'Fraud proof does not verify'
        and not framing.get('height'),
        'fault_named': audit.get('fault_forward') == plan['fault']['at'],
        'fault_proven': committed[8].get('code') == 0 and bool(committed[8].get('height'))
        and (results.get(jobs['cheated']) or {}).get('status') == 'fraud'
        and results[jobs['cheated']].get('guilty') == keys[1] and (owners.get(keys[1]) or {}).get('status') == 'slashed',
        'honest_settled_first': (results.get(jobs['honest']) or {}).get('status') == 'settled'
        and results[jobs['honest']].get('paid_each') == plan['ledger']['price'] // 2
        and results[jobs['honest']]['height'] < committed[8].get('height', 0)
        and (owners.get(keys[2]) or {}).get('status') == 'active',
        'balances_exact': {a: state.get('accounts', {}).get(a, {}).get('balance') for a in expected} == expected,
    }
    return {'passed': all(checks.values()), 'checks': checks,
            'honest': {k: agreement[k] for k in ('correct', 'mismatches', 'p95_seconds')},
            'admissions': admissions, 'roots': [s.get('root') for s in states], 'heights': [s.get('height') for s in states],
            'checklist_credit': False, 'admission_evidence': False}
