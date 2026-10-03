"""Audited sharded serving: light auditors replay owner logs and prove a declared fault.

Three owners serve the learned assistant with signed logs. Two auditors each
fetch only one owner's shard (owner 1's or owner 2's) and replay that owner's
log for every episode. A second pass has owner 1 flip one bit of one declared
sent tensor. Its auditor must name exactly that entry, a fresh verifier process
must accept the resulting fraud proof, and owner 2 must audit clean.
"""

import os
from pathlib import Path

from neuroshard.evolution import granite_shard_execution as shard
from neuroshard.evolution import granite_shard_serving as serving
from neuroshard.evolution.modular_reference_execution import ROOT, read, save, sha256

PLAN = 'config/experiments/granite-shard-audit.json'
SCRIPT = 'scripts/run_granite_shard_audit.py'
PROFILE = 'granite-shard-audit'
UPLOADED = serving.UPLOADED
OWNER_PHASES = ('fetch', 'serve-honest', 'serve-cheat')
AUDITOR_PHASES = ('fetch', 'audit-honest', 'audit-cheat', 'verify')


def committed_sources(root=ROOT):
    return shard.committed_sources(root, PLAN)


def configure():
    shard.configure()


def freeze(plan_path=PLAN):
    """The canonical runtime plus the declared signing packages, checked before any role does work."""
    import importlib.metadata

    source = shard.freeze(plan_path)
    declared = read(ROOT / plan_path)['signing_packages']
    try:
        signing = {name: importlib.metadata.version(name) for name in declared}
    except importlib.metadata.PackageNotFoundError as error:
        raise ValueError(f'signing package missing: {error}') from error
    if signing != declared:
        raise ValueError('signing packages differ from the declaration')
    return {**source, 'signing_packages': signing}


def owner_key(store, name='owner.key'):
    """This owner's Ed25519 signing key, created once in its store; returns (path, public key hex)."""
    from cryptography.hazmat.primitives import serialization
    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

    path = Path(store) / name
    if not path.exists():
        key = Ed25519PrivateKey.generate()
        raw = key.private_bytes(serialization.Encoding.Raw, serialization.PrivateFormat.Raw, serialization.NoEncryption())
        path.write_text(raw.hex() + '\n')
        path.chmod(0o600)
    key = Ed25519PrivateKey.from_private_bytes(bytes.fromhex(path.read_text().strip()))
    public = key.public_key().public_bytes(serialization.Encoding.Raw, serialization.PublicFormat.Raw).hex()
    return path, public


def session_key(store):
    """Owner 0's Ed25519 session key, standing in for the user's device; returns (path, public key hex)."""
    return owner_key(store, 'session.key')


def owner(rank, address, port, phase, home, store, plan_path=PLAN):
    """One owner phase under the contract at ``plan_path``.

    Under a settlement plan (one with a ledger) owner 0 also holds the user's session key,
    and a serve phase with a ``{phase}-request.json`` session signs every serving link and
    binds the owners' logs to that job.
    """
    configure()
    source = freeze(plan_path)
    plan = read(ROOT / plan_path)
    home, store = Path(home), Path(store)
    world = len(plan['boundaries']) - 1
    if not 0 <= rank < world or phase not in OWNER_PHASES:
        raise ValueError('unsupported audited owner role')
    if phase == 'fetch':
        receipt = {'freeze': source, 'rank': rank, **serving.prepare(plan, rank, store), 'completed': True}
        if rank > 0:
            receipt['public_key'] = owner_key(store)[1]
        elif 'ledger' in plan:
            receipt['session_key'] = session_key(store)[1]
        save(home / 'fetch.json', receipt, exclusive=True)
        return receipt
    from neuroshard.evolution.sharded import granite_serving

    os.environ.setdefault('GLOO_SOCKET_IFNAME', shard.default_interface())
    directory = home / phase
    directory.mkdir(parents=True, exist_ok=False)
    threads = read(ROOT / shard.CANONICAL)['resources']['threads']
    value = (serving.job(plan) if rank in (0, world - 1)
             else {'max_tokens': plan['max_boundary_tokens'], 'threads': threads})
    value.update(warm_up=True, log=rank > 0)
    if rank > 0:
        value['keys'] = {str(rank): str(owner_key(store)[0])}
    if (home / f'{phase}-request.json').exists():
        value['session'] = read(home / f'{phase}-request.json')['session']
        if rank == 0:
            value['keys'] = {'0': str(session_key(store)[0])}
    if phase == 'serve-cheat':
        value['fault'] = plan['fault']
    if rank == 0:
        value['tokenizer'] = str(store / 'config')
    save(directory / 'job.json', value, exclusive=True)
    save(directory / 'binding.json', {'freeze': source, 'plan_sha256': sha256(ROOT / plan_path), 'rank': rank,
                                      'phase': phase}, exclusive=True)
    return granite_serving.run_owner(store / 'config', store / 'shard', rank, world, address, port,
                                     directory / 'job.json', directory / 'result.json',
                                     timeout=plan['peer_timeout_seconds'])


def auditor(rank, phase, home, store, plan_path=PLAN):
    """A light auditor holding only owner ``rank``'s shard: fetch, replay a transferred log, or verify a proof."""
    configure()
    source = freeze(plan_path)
    plan = read(ROOT / plan_path)
    home, store = Path(home), Path(store)
    if rank not in plan['audited_owners'] or phase not in AUDITOR_PHASES:
        raise ValueError('unsupported auditor role')
    if phase == 'fetch':
        receipt = {'freeze': source, 'rank': rank, **shard.prepare(plan, rank, store), 'completed': True}
        save(home / 'fetch.json', receipt, exclusive=True)
        return receipt
    import time

    import torch

    from neuroshard.evolution.sharded import granite, granite_audit
    from neuroshard.evolution.sharded.granite_serving import Adapter

    torch.set_num_threads(read(ROOT / shard.CANONICAL)['resources']['threads'])
    started = time.monotonic()
    partition, _ = granite.load_partition(granite.load_config(store / 'config'), store / 'shard', rank)
    adapter = None
    if rank == len(plan['boundaries']) - 2:
        arm, _ = serving.arm_files(plan, ROOT / UPLOADED)
        adapter = Adapter(partition, read(ROOT / plan['learning'])['training'], arm)
    partition.warm_up()
    if adapter is not None:
        adapter.set(False)
        partition.warm_up()
        adapter.set(True)
    loaded = time.monotonic() - started
    if phase == 'verify':
        # A fresh process given only the proof and the accused owner's published key.
        proof, manifest = granite_audit.load_proof(home / 'audit-cheat' / 'proof')
        accepted = granite_audit.check_fraud_proof(partition, proof, read(home / 'accused.json')['public_key'], adapter)
        result = {'rank': rank, 'accepted': accepted, 'mismatch': manifest['mismatch'],
                  'proof_inputs': len(proof['inputs']), 'load_seconds': loaded, 'completed': True}
        save(home / 'verify' / 'result.json', result, exclusive=True)
        return result
    served = phase.replace('audit', 'serve')
    record, payloads = granite_audit.load(home / served / f'log-{rank}')
    report = granite_audit.replay(partition, record, payloads, adapter)
    report.update(rank=rank, signed=granite_audit.signed_by(record, record.get('public_key')),
                  public_key=record.get('public_key'), load_seconds=loaded, completed=True)
    if granite_audit.bound(record):
        # A bound log must also be exactly the transcript its upstream sender signed.
        report['attested'] = not granite_audit.unattested(record)
    if not report['valid'] and report['first_mismatch'] is not None:
        forwards = [i for i, entry in enumerate(record['entries']) if 'output' in entry]
        report['fault_forward'] = forwards.index(report['first_mismatch'])
        proof = granite_audit.fraud_proof(record, payloads, report)
        report['proof'] = granite_audit.save_proof(proof, home / phase / 'proof')
    elif report.get('attested') is False:
        report['proof'] = granite_audit.save_proof(granite_audit.claim_proof(record, 'unattested'), home / phase / 'proof')
    save(home / phase / 'result.json', report, exclusive=True)
    return report


def assess(plan, fetches, phases):
    """Honest agreement and clean audits; the declared fault caught, proven and attributed to owner 1 only."""
    from neuroshard.evolution.sharded import granite

    world = len(plan['boundaries']) - 1
    honest = phases['serve-honest']
    agreement = serving.assess(plan, fetches['owners'], [], honest)
    keys = {r: fetches['owners'][r].get('public_key') for r in range(1, world)}
    audits = {name: {r: phases[name][f'auditor-{r}'] for r in plan['audited_owners']} for name in ('audit-honest', 'audit-cheat')}
    cheat = audits['audit-cheat']
    fault = plan['fault']
    tensors, _ = shard.inventory(plan)
    total = sum(s['end'] - s['begin'] for s in tensors['tensors'].values())
    owned = {r: sum(s['end'] - s['begin'] for n, s in tensors['tensors'].items() if granite.owner(n, plan['boundaries']) == r)
             for r in plan['audited_owners']}
    faulted = cheat.get(fault['rank']) or {}
    checks = {
        'honest_agreement': agreement['passed'],
        'honest_audits': all((a or {}).get('valid') and (a or {}).get('signed') and (a or {}).get('public_key') == keys[r]
                             for r, a in audits['audit-honest'].items()),
        'fault_named': not faulted.get('valid', True) and faulted.get('first_mismatch') is not None
        and faulted.get('fault_forward') == fault['at'],
        'proof_accepted': bool((phases['verify'].get(f'auditor-{fault["rank"]}') or {}).get('accepted')),
        'no_false_blame': all((cheat.get(r) or {}).get('valid') for r in plan['audited_owners'] if r != fault['rank']),
        'light_auditors': all(fetches['auditors'][r]['fetched_bytes'] == owned[r] < total for r in plan['audited_owners']),
    }
    busy = {r: (honest[r] or {}).get('busy_seconds') for r in range(world)}
    audit_seconds = {r: (audits['audit-honest'][r] or {}).get('seconds') for r in plan['audited_owners']}
    return {'passed': all(checks.values()), 'checks': checks, 'honest': {k: agreement[k] for k in ('correct', 'mismatches', 'p95_seconds')},
            'fault': {'declared_forward': fault['at'], 'named_entry': faulted.get('first_mismatch'),
                      'named_forward': faulted.get('fault_forward')},
            'owner_busy_seconds': busy, 'audit_replay_seconds': audit_seconds,
            'audit_cost_ratio': {r: audit_seconds[r] / busy[r] if audit_seconds[r] and busy[r] else None
                                 for r in plan['audited_owners']},
            'proof': faulted.get('proof'), 'checklist_credit': False, 'admission_evidence': False}
