"""Native settlement adapter for a frozen portable-shard computation.

Consensus checks commitments and native weighted replay verdicts, never CUDA.
This opt-in genesis profile admits bounded windows of one prepared job. It does
not activate arbitrary future datasets or promote a serving model.
"""
import copy
import math
import re

from neuroshard.demo import protocol
from neuroshard.lab import state as ledger
from .reference_data import identity
from .schema import root, integer
from . import auditing

FORMAT = 'neuroshard-native-portable-work-v1'
FIELDS = {'reserve_shards': {'input_checkpoint', 'workers', 'audit_budget'},
          'claim_shards': {'output', 'transcript_root', 'workers'}}
from .checkpoint_schema import COMMIT_FIELDS, validate


def initialize(s):
    profile = s['manifest']['portable_work']
    required = {'format', 'checkpoint', 'prepared', 'max_step', 'max_window_steps'}
    if (not required <= set(profile) or set(profile)-required-{'reference_root'}
            or profile['format'] != FORMAT or not auditing.native(s) or 'lifecycle' in s['manifest']):
        raise ValueError('Portable work requires its dedicated native replay-quorum genesis')
    common = validate(profile['checkpoint'])
    root(profile['prepared'])
    root(profile.get('reference_root', identity(None)))
    integer(profile['max_window_steps'], 1, 4)
    integer(profile['max_step'], common['step']+1, 2**24-1)
    if common['state_root'] != s['model_root']:
        raise ValueError('Initial portable model differs from genesis')
    s['portable_work'] = {'checkpoint': copy.deepcopy(common), 'initial_checkpoint': identity(common)}


def receipt(chain_id, assignment, output, transcript, rank):
    return {'domain': FORMAT+'/worker', 'chain_id': chain_id, 'assignment': assignment['id'],
            'input_checkpoint': assignment['input_checkpoint'], 'output_checkpoint': identity(output),
            'transcript_root': transcript, 'rank': rank}


def apply(s, owner, body, envelope):
    if 'portable_work' not in s:
        raise ValueError('Genesis does not enable portable shard work')
    from .portable_lifecycle import active_profile
    common = s['portable_work']['checkpoint']
    profile, params = active_profile(s), s['manifest']['params']
    if body['kind'] == 'reserve_shards':
        if s['assignment'] or s['candidate'] or body['input_checkpoint'] != identity(common):
            raise ValueError('Reserve the current portable checkpoint without overlapping work')
        if common['step'] >= profile['max_step']:
            raise ValueError('Prepared job exhausted; finish quality approval')
        workers = body['workers']
        if not isinstance(workers, list) or len(workers) != len(common['boundaries'])-1:
            raise ValueError('Reserve every current shard owner')
        for worker in workers:
            ledger.public_key(worker)
        auditing.debit(s, owner, params['claim_bond'])
        assignment = {'id': protocol.transaction_id(envelope), 'owner': owner, 'workers': workers,
            'input_checkpoint': identity(common), 'bond': params['claim_bond'],
            'expires': s['height']+params['lease_blocks'], 'audit_budget': body['audit_budget']}
        auditing.lock(s, body['audit_budget'], owner, workers, assignment['id'])
        s['assignment'] = assignment
        return
    assignment = s['assignment']
    if not assignment or assignment['owner'] != owner or s['height'] > assignment['expires'] or s['candidate']:
        raise ValueError('No current portable work reservation')
    child = validate(body['output'])
    steps = child['step']-common['step']
    integer(steps, 1, profile['max_window_steps'])
    if child['step'] > profile['max_step'] or s['period_steps']+steps > params['steps_per_period']:
        raise ValueError('Prepared job or issuance period exhausted')
    if (child['parent'] != identity(common) or child['transition'] is not None
            or child['job'] != profile.get('job', common['job'])
            or any(child[k] != common[k] for k in ['config', 'boundaries'])):
        raise ValueError('Training cannot replace its parent, job, architecture or shard layout')
    if set(child['tensors']) != set(common['tensors']):
        raise ValueError('Incomplete trained tensor coverage')
    for name, spec in common['tensors'].items():
        if any(child['tensors'][name][k] != spec[k] for k in ['shape', 'born', 'group']):
            raise ValueError('Training cannot reset parameter ages or Adam group membership')
    semantics = lambda groups: [{k: v for k, v in group.items() if k != 'lr'} for group in groups]
    if semantics(child['optimizer']) != semantics(common['optimizer']):
        raise ValueError('Training cannot replace optimizer semantics')
    transcript = root(body['transcript_root'])
    if not isinstance(body['workers'], list) or len(body['workers']) != len(assignment['workers']):
        raise ValueError('Incomplete worker receipts')
    for rank, signed in enumerate(body['workers']):
        payload, worker = protocol.verify(signed)
        if payload != receipt(s['chain_id'], assignment, child, transcript, rank) or worker != assignment['workers'][rank]:
            raise ValueError('Worker receipt does not bind its reserved shard window')
    ids = [identity({'domain': FORMAT+'/step', 'initial': s['portable_work']['initial_checkpoint'], 'step': step})
           for step in range(common['step']+1, child['step']+1)]
    if any(key in s['paid_work'] for key in ids):
        raise ValueError('Portable training steps were already paid')
    s['candidate'] = {'kind': 'portable_training', 'id': protocol.transaction_id(envelope), 'owner': owner,
        'bond': assignment['bond'], 'workers': assignment['workers'], 'record_root': transcript,
        'model_root': child['state_root'], 'input_checkpoint': copy.deepcopy(common), 'output_checkpoint': child,
        'prepared': profile['prepared'], 'reference_root': profile.get('reference_root', identity(None)),
        'work_ids': ids, 'stages': steps*len(assignment['workers']),
        'deadline': s['height']+params['challenge_blocks'], 'expires': s['height']+params['max_claim_blocks'],
        'challenge': None}
    if 'portable_lifecycle' in s:
        s['candidate']['execution_job'] = profile['job']
        s['candidate']['executor_root'] = profile['executor_root']
        s['candidate']['expires'] = min(s['candidate']['expires'], s['portable_lifecycle']['active']['expires'])
    auditing.attach(s, assignment['audit_budget'])
    s['assignment'] = None


def settle(s, claim):
    steps = len(claim['work_ids'])
    for key in claim['work_ids']:
        if key in s['paid_work']:
            raise ValueError('Portable step was already paid')
        s['paid_work'][key] = claim['id']
    reward = steps*s['manifest']['params']['reward_atoms']
    s['issued'] += reward
    s['training_round'] += steps
    s['period_steps'] += steps
    # The frozen profile pays by owned parameter count, with deterministic
    # remainder assignment. This is an issuance rule, not a hardware cost model.
    common = claim['output_checkpoint']
    for worker, amount in zip(claim['workers'], payments(common, reward)):
        ledger.account(s, worker)['balance'] += amount
    s['portable_work']['checkpoint'] = copy.deepcopy(common)
    s['model_root'] = common['state_root']


def payments(common, reward):
    """Deterministic parameter-weighted division shared by training and serving."""
    weights = [0]*(len(common['boundaries'])-1)
    for name, spec in common['tensors'].items():
        match = re.fullmatch(r'model\.layers\.(\d+)\..+', name)
        if match:
            layer = int(match[1])
            rank = next(i for i, (a, b) in enumerate(zip(common['boundaries'], common['boundaries'][1:])) if a <= layer < b)
        elif name in ('model.embed_tokens.weight', 'model.norm.weight'):
            rank = 0
        else:
            raise ValueError('Unsupported owned parameter')
        weights[rank] += math.prod(spec['shape'])
    amounts = [reward*w//sum(weights) for w in weights]
    for rank in range(reward-sum(amounts)):
        amounts[rank] += 1
    return amounts


def replay_report(claim, partitions):
    """Validate a trusted local replay backend's complete window coverage.

    This checks report binding, not computation. The configured backend must
    execute the pinned numerical auditor; accepting strangers' report files
    would violate the honest-validator assumption.
    """
    if claim.get('kind') in ('portable_quality', 'portable_inference'):
        from .portable_lifecycle import replay_report as service_report
        return service_report(claim, partitions)
    before, after = claim['input_checkpoint'], claim['output_checkpoint']
    count = len(before['boundaries'])-1
    if not isinstance(partitions, list) or len(partitions) != count:
        raise ValueError('GPU auditor must replay every shard, not only its training assignment')
    required = {'job': claim.get('execution_job', before['job']), 'prepared': claim['prepared'], 'input': identity(before),
        'output': identity(after), 'reference': claim.get('reference_root', identity(None)),
        'boundaries': before['boundaries'], 'start': before['step'], 'end': after['step']}
    for rank, report in enumerate(partitions):
        if (report['rank'] != rank or type(report['valid']) is not bool
                or report['transcript_root'] != claim['record_root']
                or any(report['binding'].get(k) != v for k, v in required.items())):
            raise ValueError('GPU replay report is stale, incomplete or bound to another computation')
    valid = all(r['valid'] for r in partitions)
    return {'valid': valid, 'record_root': claim['record_root'],
            'coverage_root': auditing.coverage(claim) if valid else None,
            'partitions': partitions, 'stages': [{'stage': step*count+rank, 'valid': report['valid']}
                for step in range(after['step']-before['step']) for rank, report in enumerate(partitions)]}
