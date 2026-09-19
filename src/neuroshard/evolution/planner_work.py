"""Funded native settlement for prescribed, owned planner training.

This optional research genesis uses the existing reservation, weighted audit,
bond and issuance machinery. It never promotes a service. Auditor signatures
rely on the configured backend actually replaying all updates; they are not
cryptographic proofs of neural computation.
"""
import copy

from neuroshard.dataflow.store import canonical
from neuroshard.demo import protocol
from neuroshard.lab import state as ledger
from . import auditing, planner_window
from .reference_data import identity
from .schema import root

FORMAT = 'neuroshard-native-planner-work-v1'
KIND = 'planner_training'
FIELDS = {
    'reserve_planner': {'input_checkpoint', 'workers', 'audit_budget'},
    'claim_planner': {'window', 'workers'},
}


def initialize(s):
    profile = s['manifest']['planner_work']
    planner_window.validate_profile(profile)
    if (len(canonical(profile)) > 256*1024 or not auditing.native(s)
            or any(name in s['manifest'] for name in
                   ('lifecycle', 'portable_work', 'portable_lifecycle', 'expert_work', 'expert_lifecycle'))):
        raise ValueError('Planner work requires a bounded dedicated native replay-quorum profile')
    if s['serving_root'] != profile['graph']:
        raise ValueError('Planner training must bind its immutable initial serving graph')
    s['planner_work'] = {'checkpoint': copy.deepcopy(profile['initial'])}
    s['model_root'] = profile['initial']['fusion']


def receipt(chain_id, assignment, window):
    return {'domain': FORMAT+'/worker', 'chain_id': chain_id, 'assignment': assignment['id'],
            'prescription': window['prescription'], 'record_root': identity(window),
            'input_checkpoint': identity(window['checkpoints'][0]),
            'output_checkpoint': identity(window['checkpoints'][-1])}


def apply(s, owner, body, envelope):
    if 'planner_work' not in s:
        raise ValueError('Genesis does not enable planner settlement')
    profile = s['manifest']['planner_work']
    current = s['planner_work']['checkpoint']
    params = s['manifest']['params']
    if body['kind'] == 'reserve_planner':
        if s['assignment'] or s['candidate']:
            raise ValueError('Another native work reservation is active')
        if body['input_checkpoint'] != identity(current) or current['step'] >= profile['recipe']['steps']:
            raise ValueError('Reserve the unexhausted current planner checkpoint')
        workers = body['workers']
        if not isinstance(workers, list) or len(workers) != 3:
            raise ValueError('Reserve the three participating assistant owners')
        for worker in workers:
            ledger.public_key(worker)
        auditing.debit(s, owner, params['claim_bond'])
        assignment = {'id': protocol.transaction_id(envelope), 'owner': owner, 'workers': list(workers),
            'planner_kind': KIND, 'input_checkpoint': identity(current), 'bond': params['claim_bond'],
            'expires': s['height']+params['lease_blocks'], 'audit_budget': body['audit_budget']}
        auditing.lock(s, body['audit_budget'], owner, workers, assignment['id'])
        s['assignment'] = assignment
        return
    assignment = s['assignment']
    if (not assignment or assignment.get('planner_kind') != KIND or assignment['owner'] != owner
            or assignment['input_checkpoint'] != identity(current)
            or s['height'] > assignment['expires'] or s['candidate']):
        raise ValueError('No matching current planner reservation')
    window = body['window']
    work = planner_window.validate(profile, current, window, s['paid_work'])
    if s['period_steps']+work['steps'] > params['steps_per_period']:
        raise ValueError('The training issuance period is exhausted')
    receipts = body['workers']
    if not isinstance(receipts, list) or len(receipts) != 3:
        raise ValueError('Require all participating planner worker receipts')
    for rank, signed in enumerate(receipts):
        payload, worker = protocol.verify(signed)
        if (worker != assignment['workers'][rank]
                or payload != {**receipt(s['chain_id'], assignment, window), 'rank': rank}):
            raise ValueError('Worker receipt differs from its reserved planner execution')
    after = window['checkpoints'][-1]
    s['candidate'] = {'kind': KIND, 'id': protocol.transaction_id(envelope), 'owner': owner,
        'bond': assignment['bond'], 'workers': list(assignment['workers']),
        'prescription': copy.deepcopy(profile), 'window': copy.deepcopy(window),
        'record_root': work['record_root'], 'model_root': after['fusion'],
        'input_checkpoint': copy.deepcopy(current), 'output_checkpoint': copy.deepcopy(after),
        'work_ids': work['work_ids'], 'stages': work['steps'],
        'deadline': s['height']+params['challenge_blocks'],
        'expires': s['height']+params['max_claim_blocks'], 'challenge': None}
    auditing.attach(s, assignment['audit_budget'])
    s['assignment'] = None


def settle(s, claim):
    current = s['planner_work']['checkpoint']
    if claim['prescription'] != s['manifest']['planner_work'] or claim['input_checkpoint'] != current:
        raise ValueError('Accepted planner work must extend the current prescription')
    work = planner_window.validate(claim['prescription'], current, claim['window'], s['paid_work'])
    if work['work_ids'] != claim['work_ids']:
        raise ValueError('Planner rewards changed their numerical work identities')
    reward = work['steps']*s['manifest']['params']['reward_atoms']
    s['paid_work'].update({key: claim['id'] for key in work['work_ids']})
    s['issued'] += reward
    s['training_round'] += work['steps']
    s['period_steps'] += work['steps']
    # This first profile pays equal reserved execution roles. Splitting a role
    # into more identities cannot increase issuance or the number of shares.
    share, remainder = divmod(reward, 3)
    for rank, worker in enumerate(claim['workers']):
        ledger.account(s, worker)['balance'] += share+(rank < remainder)
    s['planner_work']['checkpoint'] = copy.deepcopy(claim['output_checkpoint'])
    s['model_root'] = claim['output_checkpoint']['fusion']


def binding(claim):
    return {'prescription': identity(claim['prescription']),
            'input_checkpoint': identity(claim['input_checkpoint']),
            'output_checkpoint': identity(claim['output_checkpoint']),
            'numerical_profile': root(claim['prescription']['numerical_profile'])}


def replay_report(claim, report):
    """Validate coverage from an operator-configured complete numerical replay."""
    if (not isinstance(report, dict)
            or set(report) != {'format', 'claim_id', 'record_root', 'binding', 'stages'}
            or report['format'] != FORMAT+'/replay' or report['claim_id'] != claim['id']
            or report['record_root'] != claim['record_root'] or report['binding'] != binding(claim)
            or not isinstance(report['stages'], list) or len(report['stages']) != claim['stages']):
        raise ValueError('Planner replay must bind and cover the complete claimed execution')
    for index, stage in enumerate(report['stages']):
        if (not isinstance(stage, dict) or set(stage) != {'stage', 'valid'}
                or type(stage['stage']) is not int or stage['stage'] != index
                or type(stage['valid']) is not bool):
            raise ValueError('Invalid ordered planner audit coverage')
    valid = all(stage['valid'] for stage in report['stages'])
    return {'valid': valid, 'record_root': claim['record_root'], 'stages': report['stages'],
            'coverage_root': auditing.coverage(claim) if valid else None}
