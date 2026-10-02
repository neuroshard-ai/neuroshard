"""Bonded settlement of audited sharded serving on the real assistant.

Owners 1 and 2 bond their log keys on the optimistic serving ledger, serve an honest
pass and a pass in which owner 1 flips one declared bit, and commit their signed logs
to each job. The auditor of shard 1 proves the fault, and a separate accuser tries to
frame the honest owner. Two validators, each holding shard 1, replay the same blocks.
Every party signs its own transactions on its own host.
"""

import json
import shutil
from pathlib import Path

from neuroshard.evolution import granite_shard_audit as audited
from neuroshard.evolution import granite_shard_execution as shard
from neuroshard.evolution import granite_shard_serving as serving
from neuroshard.evolution.modular_reference_execution import ROOT, read, save

PLAN = 'config/experiments/granite-shard-settlement.json'
SCRIPT = 'scripts/run_granite_shard_settlement.py'
PROFILE = 'granite-shard-settlement'
UPLOADED = serving.UPLOADED
MODEL_INVENTORY = 'config/experiments/granite-tensor-inventory.json'
OWNER_PHASES = ('fetch', 'sign-bond', 'serve-honest', 'commit-honest', 'serve-cheat', 'commit-cheat')
AUDITOR_PHASES = ('fetch', 'audit-honest', 'audit-cheat', 'challenge')
VALIDATOR_PHASES = ('fetch', 'validate')


def committed_sources(root=ROOT):
    return shard.committed_sources(root, PLAN)


def configure():
    shard.configure()


def freeze():
    return audited.freeze(PLAN)


def account(store, name='account'):
    """A ledger account from a secp256k1 seed created once in this host's store; returns (key, public hex)."""
    import secrets

    from neuroshard.core.crypto.ecdsa import derive_keypair_from_token

    path = Path(store) / f'{name}.seed'
    if not path.exists():
        path.write_text(secrets.token_hex(32) + '\n')
        path.chmod(0o600)
    key = derive_keypair_from_token(path.read_text().strip())
    return key, key.public_key_bytes.hex()


def signed(key, body):
    from neuroshard.core.crypto.ecdsa import ecdsa_sign
    from neuroshard.inference import optimistic as ledger

    return {'body': body, 'public_key': key.public_key_bytes.hex(),
            'signature': ecdsa_sign(ledger.canonical(body).decode(), key.private_key_bytes)}


def log_signer(store):
    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

    path, public = audited.owner_key(store)
    return Ed25519PrivateKey.from_private_bytes(bytes.fromhex(path.read_text().strip())), public


def stash(log_dir, store):
    """Move a log's retained inputs out of the evidence directory once no one needs them."""
    inputs = Path(log_dir) / 'inputs.safetensors'
    if inputs.exists():
        target = Path(store) / 'stashed' / f'{Path(log_dir).parent.name}-{Path(log_dir).name}.safetensors'
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(inputs), str(target))


def owner(rank, address, port, phase, home, store):
    home, store = Path(home), Path(store)
    if phase not in OWNER_PHASES:
        raise ValueError('unsupported settlement owner role')
    if phase in ('fetch', 'serve-honest', 'serve-cheat'):
        result = audited.owner(rank, address, port, phase, home, store, plan_path=PLAN)
        if phase == 'fetch' and rank > 0:
            save(home / 'account.json', {'account': account(store)[1], 'log_key': audited.owner_key(store)[1]},
                 exclusive=True)
        return result
    shard.configure()
    source = audited.freeze(PLAN)
    from neuroshard.evolution.sharded import granite_audit
    from neuroshard.inference import optimistic as ledger

    if rank < 1:
        raise ValueError('only bonded owners sign ledger transactions')
    request = read(home / f'{phase}-request.json')
    key, public = account(store)
    log, log_key = log_signer(store)
    base = {'chain_id': request['chain_id'], 'nonce': request['nonce']}
    if phase == 'sign-bond':
        possession = log.sign(ledger.possession_message(request['chain_id'], public, log_key, rank, request['amount'],
                                                        request['nonce'])).hex()
        body = {'kind': 'owner_bond', **base, 'model_root': request['model_root'], 'shard': rank, 'log_key': log_key,
                'amount': request['amount'], 'possession': possession}
    else:
        log_dir = home / phase.replace('commit', 'serve') / f'log-{rank}'
        statement = granite_audit.statement(json.loads((log_dir / 'log.json').read_text())).hex()
        body = {'kind': 'log_commit', **base, 'job_id': request['job_id'], 'statement_root': statement,
                'log_signature': log.sign(ledger.commitment_message(request['chain_id'], request['job_id'],
                                                                    statement)).hex()}
        stash(log_dir, store)
    result = {'freeze': source, 'rank': rank, 'envelope': signed(key, body), 'completed': True}
    save(home / phase / 'result.json', result, exclusive=True)
    return result


def auditor(phase, home, store):
    """The light auditor of shard 1, which also holds a separate accuser account for the framing attempt."""
    home, store = Path(home), Path(store)
    if phase not in AUDITOR_PHASES:
        raise ValueError('unsupported settlement auditor role')
    if phase != 'challenge':
        result = audited.auditor(1, phase, home, store, plan_path=PLAN)
        if phase == 'fetch':
            save(home / 'accounts.json', {'auditor': account(store)[1], 'accuser': account(store, 'accuser')[1]},
                 exclusive=True)
        return result
    shard.configure()
    source = audited.freeze(PLAN)
    from neuroshard.evolution.sharded import granite_audit

    request = read(home / 'challenge-request.json')
    bundles = home / 'bundles'
    bundles.mkdir(exist_ok=True)
    proven = granite_audit.bundle_root(home / 'audit-cheat' / 'proof')
    shutil.copytree(home / 'audit-cheat' / 'proof', bundles / proven)
    record, payloads = granite_audit.load(home / 'serve-honest' / 'log-1')
    first = next(i for i, entry in enumerate(record['entries']) if 'output' in entry)
    granite_audit.save_proof({'record': record, 'mismatch': first,
                              'inputs': {i: p for i, p in payloads.items() if i <= first}}, home / 'forged')
    forged = granite_audit.bundle_root(home / 'forged')
    shutil.copytree(home / 'forged', bundles / forged)
    auditor_key, _ = account(store)
    accuser_key, _ = account(store, 'accuser')
    challenge = {'kind': 'challenge', 'chain_id': request['chain_id'], 'nonce': 0, 'log_key': request['log_key']}
    result = {'freeze': source, 'proven_root': proven, 'forged_root': forged, 'forged_mismatch': first,
              'proven': signed(auditor_key, {**challenge, 'job_id': request['cheated_job'], 'proof_root': proven}),
              'framing': signed(accuser_key, {**challenge, 'job_id': request['honest_job'], 'proof_root': forged}),
              'completed': True}
    for served in ('serve-honest', 'serve-cheat'):
        stash(home / served / 'log-1', store)
    save(home / 'challenge' / 'result.json', result, exclusive=True)
    return result


def replay_blocks(genesis, blocks, check):
    """Replay every block from genesis; a rejected transaction leaves the state unchanged and is recorded."""
    import time

    from neuroshard.inference import optimistic as ledger

    state, outcomes, challenge_seconds = genesis, [], []
    for block in blocks:
        state = ledger.advance(state, state['height'] + 1)
        for envelope in block:
            started = time.monotonic()
            try:
                state = ledger.transition(state, envelope, check)
                outcomes.append('accepted')
            except ValueError as error:
                outcomes.append(f'rejected: {error}')
            if envelope['body'].get('kind') == 'challenge':
                challenge_seconds.append(time.monotonic() - started)
    return {'root': ledger.root(state), 'outcomes': outcomes, 'state': state, 'challenge_seconds': challenge_seconds}


def validator(index, phase, home, store):
    """A validator holding shard 1: fetch it, or replay the transferred blocks with real proof checks."""
    shard.configure()
    source = audited.freeze(PLAN)
    plan = read(ROOT / PLAN)
    home, store = Path(home), Path(store)
    if index not in (1, 2) or phase not in VALIDATOR_PHASES:
        raise ValueError('unsupported validator role')
    if phase == 'fetch':
        receipt = {'freeze': source, 'validator': index, **shard.prepare(plan, 1, store), 'completed': True}
        save(home / 'fetch.json', receipt, exclusive=True)
        return receipt
    import time

    import torch

    from neuroshard.evolution.sharded import granite, granite_audit

    torch.set_num_threads(read(ROOT / shard.CANONICAL)['resources']['threads'])
    started = time.monotonic()
    partition, _ = granite.load_partition(granite.load_config(store / 'config'), store / 'shard', 1)
    partition.warm_up()
    loaded = time.monotonic() - started
    chain = read(home / 'blocks.json')
    report = replay_blocks(chain['genesis'], chain['blocks'],
                           granite_audit.challenge_checker(home / 'bundles', {1: partition}))
    result = {'freeze': source, 'validator': index, **report, 'load_seconds': loaded, 'completed': True}
    save(home / 'validate' / 'result.json', result, exclusive=True)
    return result


def expected_balances(plan, parties):
    """Final balances the declared sequence must leave: one honest job settled, one cheating owner slashed."""
    ledger = plan['ledger']
    fee, start, bond, price = ledger['params']['fee'], ledger['allocation'], ledger['owner_bond'], ledger['price']
    reward = bond * ledger['params']['auditor_share_ppm'] // 1_000_000
    share = price // 2
    return {parties['user']: start - 2 * fee - price,
            parties['owner-1']: start - 3 * fee - bond + share,
            parties['owner-2']: start - 3 * fee - bond + share,
            parties['auditor']: start - fee + reward,
            parties['accuser']: start}


def assess(plan, fetches, phases, parties, jobs):
    """Honest serving agreement, agreeing replicas, a rejected framing, a proven fault and exact settlement."""
    honest = phases['serve-honest']
    agreement = serving.assess(plan, fetches['owners'], [], honest)
    reports = [phases['validate'].get(f'validator-{i}') or {} for i in (1, 2)]
    first = reports[0]
    state = first.get('state') or {'results': {}, 'owners': {}, 'accounts': {}}
    results, owners = state['results'], state['owners']
    keys = {r: fetches['owners'][r].get('log_key') for r in (1, 2)}
    outcomes = first.get('outcomes') or []
    expected = expected_balances(plan, parties)
    audit = phases['audit-cheat'] or {}
    checks = {
        'honest_agreement': agreement['passed'],
        'replicas_agree': all(r.get('completed') for r in reports) and reports[0].get('root') == reports[1].get('root')
        and reports[0].get('outcomes') == reports[1].get('outcomes'),
        'honest_work_accepted': outcomes[:8] == ['accepted'] * 8,
        'framing_rejected': outcomes[8:9] == ['rejected: Fraud proof does not verify'],
        'fault_named': audit.get('fault_forward') == plan['fault']['at'],
        'fault_proven': outcomes[9:10] == ['accepted'] and (results.get(jobs['cheated']) or {}).get('status') == 'fraud'
        and results[jobs['cheated']].get('guilty') == keys[1] and (owners.get(keys[1]) or {}).get('status') == 'slashed',
        'honest_settled': (results.get(jobs['honest']) or {}).get('status') == 'settled'
        and results[jobs['honest']].get('paid_each') == plan['ledger']['price'] // 2
        and (owners.get(keys[2]) or {}).get('status') == 'active',
        'balances_exact': {a: state['accounts'].get(a, {}).get('balance') for a in expected} == expected,
    }
    return {'passed': all(checks.values()), 'checks': checks,
            'honest': {k: agreement[k] for k in ('correct', 'mismatches', 'p95_seconds')},
            'outcomes': outcomes, 'roots': [r.get('root') for r in reports],
            'challenge_seconds': [r.get('challenge_seconds') for r in reports],
            'validator_load_seconds': [r.get('load_seconds') for r in reports],
            'checklist_credit': False, 'admission_evidence': False}
