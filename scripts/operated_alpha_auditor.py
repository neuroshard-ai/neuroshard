"""Durable complete-replay operation for one administrator's alpha audit quorum."""
import json
from pathlib import Path
import secrets
import time

from neuroshard.evolution import auditing, expert_lifecycle
from neuroshard.evolution.reference_data import identity, save
from operated_alpha_hosts import LedgerHosts
from ordinary_cloud import REMOTE


def committed(home, network):
    hosts = LedgerHosts(home/'ledger-hosts')
    script = ('import json,pathlib,sqlite3,sys; p=pathlib.Path(sys.argv[1]); '
        'c=sqlite3.connect(p.as_uri()+"?mode=ro",uri=True); '
        'sys.stdout.buffer.write(c.execute("SELECT value FROM state WHERE id=1").fetchone()[0])')
    state = json.loads(hosts.command(0, ['python3', '-c', script, REMOTE+'/native/evolution.sqlite']).stdout)
    if (state['chain_id'] != network.genesis['chain_id']
            or state['manifest'] != network.genesis['app_state']['manifest']):
        raise ValueError('The controller full node changed its pinned network')
    return state


def pending(home, network):
    boxes = network.outboxes[:3]
    if not any(box.pending() for box in boxes):
        return False
    state = committed(home, network)
    for box in boxes:
        if box.pending():
            box.retire_control(state)
            box.retire_closed(state)
            if box.pending():
                box.confirm(box.pending(), timeout=5)
    return True


def maintain(home, network, *, closing=False):
    """One idle control action, using each auditor's existing durable signer."""
    if pending(home, network):
        return 'reconciled'
    status = network.query()
    service = network.query('/service_admission')
    owners = {owner.public_key: i for i, owner in enumerate(network.owners[:3])}
    for key, row in sorted(service['services'].items()):
        if row['owner'] not in owners or row['purpose'] != 'expert_inference':
            continue
        index = owners[row['owner']]
        if closing:
            network.send(index, 'close-future-capacity-'+key, 'close_audit_service', service_id=key)
            return 'future_capacity_closed'
        if row['expires']-status['height'] < 20000:
            network.send(index, 'renew-capacity-'+key+'-'+str(status['height']), 'renew_audit_service',
                service_id=key, expires_in=100000, valid_until=status['height']+64)
            return 'audit_capacity_renewed'
    return 'idle'


def tick(home, cloud, network, service):
    if pending(home, network):
        return {'phase': 'reconciled'}
    claim = network.query('/candidate')
    if claim is None:
        return {'phase': maintain(home, network)}
    if claim['kind'] != 'expert_inference':
        raise ValueError("This alpha's numerical backend serves only the frozen inference graph")
    budget = network.query('/auditing')['budgets'].get(claim['audit_budget'])
    if budget is None:
        return {'phase': 'claim_closed'}
    selected = [i for i, owner in enumerate(network.owners[:3])
                if budget['auditors'].get(owner.public_key, {}).get('bond')]
    if len(selected) != 3:
        return {'phase': 'not_this_complete_audit_group'}
    folder = home/'audits'/claim['id']
    folder.mkdir(parents=True, exist_ok=True)
    saved = folder/'claim.json'
    if not saved.exists():
        save(saved, claim)
    intent_path = folder/'intent.json'
    if not intent_path.exists():
        save(intent_path, {'claim': claim['id'], 'coverage': auditing.coverage(claim),
                           'salts': [secrets.token_hex(32) for _ in selected]})
    intent = json.loads(intent_path.read_bytes())
    if intent['claim'] != claim['id'] or intent['coverage'] != auditing.coverage(claim):
        raise ValueError('Audit restart changed its complete signed obligation')
    reports = []
    for index in selected:
        path = folder/f'replay-{index}.json'
        if path.exists():
            result = json.loads(path.read_bytes())
        else:
            result = cloud.query(service, {'id': identity({'claim': claim['id'], 'auditor': index}),
                'kind': 'inference_audit', 'claim': claim}, timeout=600)
            if result['status'] != 'completed':
                return {'phase': 'full_replay_unavailable', 'claim': claim['id']}
            # This rejects missing or partial coverage; a complete negative
            # result remains a negative verdict and is never silently accepted.
            expert_lifecycle.replay_report(claim, result['report'])
            save(path, result)
        reports.append(expert_lifecycle.replay_report(claim, result['report']))
    if len({report['valid'] for report in reports}) != 1:
        raise ValueError('The complete reference executions disagree')
    valid = reports[0]['valid']
    for index in selected:
        current = network.query('/candidate')
        if current is None or current['id'] != claim['id']:
            return {'phase': 'claim_closed_during_replay', 'claim': claim['id']}
        current_budget = network.query('/auditing')['budgets'][current['audit_budget']]
        row = current_budget['auditors'][network.owners[index].public_key]
        if row['commitment'] is None:
            if network.query()['height'] > current['audit_commit_end']:
                return {'phase': 'commit_deadline_elapsed', 'claim': claim['id']}
            commitment = auditing.verdict_commitment(network.genesis['chain_id'], claim['id'],
                network.owners[index].public_key, intent['coverage'], intent['salts'][index], valid)
            network.outboxes[index].send(claim['id']+'/commit', 'audit_commit',
                claim_id=claim['id'], commitment=commitment, timeout=5)
    for index in selected:
        current = network.query('/candidate')
        if current is None or current['id'] != claim['id']:
            return {'phase': 'claim_closed_after_commit', 'claim': claim['id']}
        height = network.query()['height']
        if height <= current['audit_commit_end']:
            return {'phase': 'waiting_for_reveal', 'claim': claim['id']}
        if height > current['audit_reveal_end']:
            return {'phase': 'reveal_deadline_elapsed', 'claim': claim['id']}
        current_budget = network.query('/auditing')['budgets'][current['audit_budget']]
        row = current_budget['auditors'][network.owners[index].public_key]
        if not row['revealed']:
            network.outboxes[index].send(claim['id']+'/reveal', 'audit_verdict', claim_id=claim['id'],
                coverage_root=intent['coverage'], salt=intent['salts'][index], valid=valid, timeout=5)
    return {'phase': 'full_replays_reported', 'claim': claim['id'], 'valid': valid,
            'complete_replays': len(reports)}
