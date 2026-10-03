"""Bonded settlement of audited sharded serving on the real assistant.

Owners 1 and 2 bond their log keys on the optimistic serving ledger, serve an honest
pass and a pass in which owner 1 flips one declared bit, and commit to each job a log
bound to it by signed serving links. The auditor of shard 1 proves the fault, and a
separate accuser tries to frame the honest owner and forfeits its challenge deposit.
Two validators, each holding shard 1, replay the same blocks. Every party signs its own
transactions on its own host.
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


def owner(rank, address, port, phase, home, store, plan_path=PLAN):
    """One owner phase under the contract at ``plan_path``; owners 1 and 2 sign their own ledger transactions."""
    home, store = Path(home), Path(store)
    if phase not in OWNER_PHASES:
        raise ValueError('unsupported settlement owner role')
    if phase in ('fetch', 'serve-honest', 'serve-cheat'):
        result = audited.owner(rank, address, port, phase, home, store, plan_path=plan_path)
        if phase == 'fetch' and rank > 0:
            save(home / 'account.json', {'account': account(store)[1], 'log_key': audited.owner_key(store)[1]},
                 exclusive=True)
        return result
    shard.configure()
    source = audited.freeze(plan_path)
    from neuroshard.evolution.sharded import granite_audit
    from neuroshard.inference import optimistic as ledger

    if rank < 1:
        raise ValueError('only bonded owners sign ledger transactions')
    request = read(home / f'{phase}-request.json')
    key, public = account(store)
    log, log_key = log_signer(store)
    base, served = {'chain_id': request['chain_id'], 'nonce': request['nonce']}, {}
    if phase == 'sign-bond':
        possession = log.sign(ledger.possession_message(request['chain_id'], public, log_key, rank, request['amount'],
                                                        request['nonce'])).hex()
        body = {'kind': 'owner_bond', **base, 'model_root': request['model_root'], 'shard': rank, 'log_key': log_key,
                'amount': request['amount'], 'possession': possession}
    else:
        # The log is bound to the job it served: its header, entries digest and statement are what the ledger checks.
        log_dir = home / phase.replace('commit', 'serve') / f'log-{rank}'
        record = json.loads((log_dir / 'log.json').read_text())
        commitment = granite_audit.commitment(record)
        body = {'kind': 'log_commit', **base, 'job_id': request['job_id'], **commitment,
                'log_signature': log.sign(ledger.commitment_message(request['chain_id'], request['job_id'],
                                                                    commitment['statement_root'])).hex()}
        served = {'positions': granite_audit.positions(record)}
        stash(log_dir, store)
    result = {'freeze': source, 'rank': rank, 'envelope': signed(key, body), **served, 'completed': True}
    save(home / phase / 'result.json', result, exclusive=True)
    return result


def challenge_pair(key, chain_id, job_id, log_key, proof_root):
    """A fresh account's signed challenge naming ``proof_root``, and its signed proof of that challenge."""
    from neuroshard.inference import optimistic as ledger

    opening = signed(key, {'kind': 'challenge', 'chain_id': chain_id, 'nonce': 0, 'log_key': log_key,
                           'job_id': job_id, 'proof_root': proof_root})
    proving = signed(key, {'kind': 'prove', 'chain_id': chain_id, 'nonce': 1, 'job_id': job_id,
                           'challenge_id': ledger.transaction_id(opening)})
    return opening, proving


def framing_challenge(home, store, request):
    """A forged bundle claiming the honest log's first output was wrong; the accuser's challenge and attempted proof."""
    from neuroshard.evolution.sharded import granite_audit

    record, payloads = granite_audit.load(home / 'serve-honest' / 'log-1')
    first = next(i for i, entry in enumerate(record['entries']) if 'output' in entry)
    granite_audit.save_proof({'record': record, 'mismatch': first,
                              'inputs': {i: p for i, p in payloads.items() if i <= first}}, home / 'forged')
    forged = granite_audit.bundle_root(home / 'forged')
    shutil.copytree(home / 'forged', home / 'bundles' / forged)
    opening, proving = challenge_pair(account(store, 'accuser')[0], request['chain_id'], request['honest_job'],
                                      request['log_key'], forged)
    return {'forged_root': forged, 'forged_mismatch': first, 'framing': opening, 'framing_prove': proving}


def proven_challenge(home, store, request):
    """The auditor's own fraud proof as a content-addressed bundle; its signed challenge and proof."""
    from neuroshard.evolution.sharded import granite_audit

    proven = granite_audit.bundle_root(home / 'audit-cheat' / 'proof')
    shutil.copytree(home / 'audit-cheat' / 'proof', home / 'bundles' / proven)
    opening, proving = challenge_pair(account(store)[0], request['chain_id'], request['cheated_job'],
                                      request['log_key'], proven)
    return {'proven_root': proven, 'proven': opening, 'proven_prove': proving}


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
    request = read(home / 'challenge-request.json')
    (home / 'bundles').mkdir(exist_ok=True)
    result = {'freeze': source, **proven_challenge(home, store, request), **framing_challenge(home, store, request),
              'completed': True}
    for served in ('serve-honest', 'serve-cheat'):
        stash(home / served / 'log-1', store)
    save(home / 'challenge' / 'result.json', result, exclusive=True)
    return result


def challenge_blocks(params, challenges):
    """The declared blocks after the bonds (block 1), the honest job (opened at 2, committed at 3)
    and the cheating job (opened at 4, committed at 5).

    The framing opens at block 6, inside the honest job's window, and its proof is refused.
    The honest job settles once its window closes and the framing has lapsed; the fraud proof
    then opens, inside the cheating job's window, and lands at the next block.
    """
    framed, window = 6, params['challenge_blocks']
    settles = max(3 + window, framed + params['proof_blocks']) + 1
    if params['proof_blocks'] < 1 or settles > 5 + window:
        raise ValueError('the declared windows close the cheating job before the honest job settles')
    waiting = [[] for _ in range(settles - framed - 2)]
    return ([[challenges['framing']], [challenges['framing_prove']]] + waiting
            + [[challenges['proven']], [challenges['proven_prove']], [], []])


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
            if envelope['body'].get('kind') == 'prove':
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


def honest_payments(plan, served):
    """What owners 1 and 2 earn for the honest job, whose logs cover ``served[rank]`` token positions.

    Each is paid its half of the price per position of the declared budget, for no more
    positions than owner 1 received from the user or the budget allows.
    """
    price, budget = plan['ledger']['price'], plan['ledger']['positions']
    sent = min(served[1], budget)
    return [price * min(served[rank], sent) // (budget * 2) for rank in (1, 2)]


def honest_positions(phases):
    """The positions each owner's honest log covers, as the owner counted them from its entries; None if unreported."""
    served = {rank: ((phases.get('commit-honest') or {}).get(f'owner-{rank}') or {}).get('positions') for rank in (1, 2)}
    return served if all(type(value) is int for value in served.values()) else None


def expected_balances(plan, parties, served):
    """Final balances the declared sequence must leave: the honest job settled for the positions
    its logs cover (``served`` by owner rank), the framing's deposit forfeited, one cheating owner slashed."""
    ledger = plan['ledger']
    params, start, bond = ledger['params'], ledger['allocation'], ledger['owner_bond']
    fee, reward = params['fee'], bond * params['auditor_share_ppm'] // 1_000_000
    paid = honest_payments(plan, served)
    return {parties['user']: start - 2 * fee - sum(paid),
            parties['owner-1']: start - 3 * fee - bond + paid[0],
            parties['owner-2']: start - 3 * fee - bond + paid[1],
            parties['auditor']: start - 2 * fee + reward,
            parties['accuser']: start - fee - params['challenge_deposit']}


def assess(plan, fetches, phases, parties, jobs):
    """Honest serving agreement, agreeing replicas, a forfeited framing, a proven fault and exact settlement.

    ``jobs`` names the honest and cheated jobs and the framing challenge.
    """
    honest = phases['serve-honest']
    agreement = serving.assess(plan, fetches['owners'], [], honest)
    reports = [phases['validate'].get(f'validator-{i}') or {} for i in (1, 2)]
    first = reports[0]
    state = first.get('state') or {'results': {}, 'owners': {}, 'accounts': {}}
    results, owners = state['results'], state['owners']
    keys = {r: fetches['owners'][r].get('log_key') for r in (1, 2)}
    outcomes = first.get('outcomes') or []
    served = honest_positions(phases)
    expected = expected_balances(plan, parties, served) if served else None
    audit = phases['audit-cheat'] or {}
    checks = {
        'honest_agreement': agreement['passed'],
        'replicas_agree': all(r.get('completed') for r in reports) and reports[0].get('root') == reports[1].get('root')
        and reports[0].get('outcomes') == reports[1].get('outcomes'),
        'honest_work_accepted': outcomes[:8] == ['accepted'] * 8,
        'framing_forfeited': outcomes[8:10] == ['accepted', 'rejected: Fraud proof does not verify']
        and (results.get(jobs['framing']) or {}).get('status') == 'forfeited',
        'fault_named': audit.get('fault_forward') == plan['fault']['at'],
        'fault_proven': outcomes[10:12] == ['accepted', 'accepted']
        and (results.get(jobs['cheated']) or {}).get('status') == 'fraud'
        and results[jobs['cheated']].get('guilty') == keys[1] and (owners.get(keys[1]) or {}).get('status') == 'slashed',
        'honest_settled': served is not None and (results.get(jobs['honest']) or {}).get('status') == 'settled'
        and results[jobs['honest']].get('paid') == honest_payments(plan, served)
        and (owners.get(keys[2]) or {}).get('status') == 'active',
        'balances_exact': expected is not None
        and {a: state['accounts'].get(a, {}).get('balance') for a in expected} == expected,
    }
    return {'passed': all(checks.values()), 'checks': checks,
            'honest': {k: agreement[k] for k in ('correct', 'mismatches', 'p95_seconds')},
            'outcomes': outcomes, 'roots': [r.get('root') for r in reports],
            'challenge_seconds': [r.get('challenge_seconds') for r in reports],
            'validator_load_seconds': [r.get('load_seconds') for r in reports],
            'checklist_credit': False, 'admission_evidence': False}
