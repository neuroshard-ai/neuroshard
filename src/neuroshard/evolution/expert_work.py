"""Native funded settlement for a fixed independently trained expert.

Prefix production must receive a complete native replay quorum before training
can be reserved. Rewards pay only accepted, nonduplicate tail updates. This
profile does not promote the expert, change the serving graph or add inference.
"""
import copy

from neuroshard.demo import protocol
from neuroshard.lab import state as ledger
from . import auditing, expert_checkpoint, expert_window
from .reference_data import identity
from .schema import integer, root

FORMAT = 'neuroshard-native-expert-work-v1'
PROSPECTIVE = 'neuroshard-prospective-expert-work-v1'
KINDS = ('expert_features', 'expert_training')
FIELDS = {
    'reserve_expert_inputs': {'workers', 'audit_budget'},
    'claim_expert_inputs': {'feature_root', 'transcript_root', 'workers'},
    'claim_expert_prefix': {'feature_root', 'batch_roots', 'transcript_root', 'workers'},
    'reserve_expert': {'input_checkpoint', 'worker', 'audit_budget'},
    'claim_expert': {'window', 'intermediates', 'worker'},
}
PROFILE_FIELDS = {'format', 'parent', 'checkpoint', 'prepared', 'feature_root',
                  'feature_stages', 'batch_roots', 'schedule', 'numerical_profile'}
PROSPECTIVE_FIELDS = PROFILE_FIELDS - {'feature_root', 'batch_roots'} | {'batch_count'}


def validate_profile(profile):
    prospective = isinstance(profile, dict) and profile.get('format') == PROSPECTIVE
    fields = PROSPECTIVE_FIELDS if prospective else PROFILE_FIELDS
    if not isinstance(profile, dict) or set(profile) != fields or profile['format'] not in (FORMAT, PROSPECTIVE):
        raise ValueError('Invalid prescribed expert execution profile')
    before = profile['checkpoint']
    expert_checkpoint.unpack(profile['parent'], before)
    if before['step'] != 0:
        raise ValueError('Prescribe an initial expert before computing its outputs')
    for key in ('prepared', 'numerical_profile'):
        root(profile[key])
    integer(profile['feature_stages'], 1, 4096)
    if prospective:
        count = integer(profile['batch_count'], 1, 4096)
    else:
        root(profile['feature_root'])
        batches = profile['batch_roots']
        if not isinstance(batches, list) or not 1 <= len(batches) <= 4096:
            raise ValueError('Freeze bounded feature batches')
        for batch in batches:
            root(batch)
        count = len(batches)
    schedule = profile['schedule']
    if (not isinstance(schedule, list) or len(schedule) != before['recipe']['steps']
            or not 1 <= len(schedule) <= 65536):
        raise ValueError('Freeze a bounded complete expert batch schedule')
    for index in schedule:
        integer(index, 0, count - 1)
    return profile


def resolve_prefix(profile, feature_root, batch_roots):
    """Combine prescribed inputs with a computed prefix; this grants no acceptance."""
    validate_profile(profile)
    if profile['format'] != PROSPECTIVE:
        raise ValueError('Only prospective work resolves an unknown prefix')
    if not isinstance(batch_roots, list) or len(batch_roots) != profile['batch_count']:
        raise ValueError('Prefix must bind every prescribed feature batch')
    for batch in batch_roots:
        root(batch)
    resolved = {key: copy.deepcopy(value) for key, value in profile.items() if key != 'batch_count'}
    resolved.update(format=FORMAT, feature_root=root(feature_root), batch_roots=list(batch_roots))
    return validate_profile(resolved)


def prescription(s):
    admission = s.get('expert_lifecycle', {}).get('admission')
    active = admission and admission['active']
    return active['job']['work'] if active else s['manifest']['expert_work']


def execution_profile(s):
    """Training consumes only the prefix already accepted by the native quorum."""
    profile = prescription(s)
    if profile['format'] != PROSPECTIVE:
        return profile
    work = s['expert_work']
    if work['feature_claim'] is None:
        raise ValueError('Complete the prefix execution audit before reserving training')
    return resolve_prefix(profile, work['feature_root'], work['batch_roots'])


def initialize(s):
    profile = s['manifest']['expert_work']
    validate_profile(profile)
    if (not auditing.native(s) or any(key in s['manifest'] for key in
                                             ('lifecycle', 'portable_work', 'portable_lifecycle'))):
        raise ValueError('Expert work requires its dedicated native replay-quorum profile')
    before = profile['checkpoint']
    expert_checkpoint.unpack(profile['parent'], before)
    if before['step'] != 0 or s['model_root'] != profile['parent']['state_root']:
        raise ValueError('Start a new expert against the declared immutable serving parent')
    s['expert_work'] = {'checkpoint': copy.deepcopy(before), 'feature_claim': None}
    if profile['format'] == PROSPECTIVE:
        s['expert_work'].update(feature_root=None, batch_roots=None)
    # The trainable branch has fresh Adam state; serving remains the parent.
    s['model_root'] = before['state_root']


def receipt(chain_id, assignment, output, transcript, rank):
    return {'domain': FORMAT + '/worker', 'chain_id': chain_id, 'assignment': assignment['id'],
            'kind': assignment['expert_kind'], 'input_checkpoint': assignment['input_checkpoint'],
            'output_root': output, 'transcript_root': transcript, 'rank': rank}


def apply(s, owner, body, envelope):
    if 'expert_work' not in s:
        raise ValueError('Genesis does not enable expert settlement')
    from . import expert_admission
    expert_admission.live(s)
    profile, current = prescription(s), s['expert_work']['checkpoint']
    params, kind = s['manifest']['params'], body['kind']
    features = kind in ('reserve_expert_inputs', 'claim_expert_inputs', 'claim_expert_prefix')
    if not features:
        profile = execution_profile(s)
    claim_kind = 'expert_features' if features else 'expert_training'
    if kind.startswith('reserve_'):
        if s['assignment'] or s['candidate']:
            raise ValueError('Another native work reservation is active')
        if features:
            if s['expert_work']['feature_claim'] is not None:
                raise ValueError('The committed feature production is already accepted')
            workers = body['workers']
            if not isinstance(workers, list) or len(workers) != len(profile['parent']['boundaries']) - 1:
                raise ValueError('Reserve all immutable parent owners')
        else:
            if s['expert_work']['feature_claim'] is None:
                raise ValueError('Complete the prefix execution audit before reserving training')
            if body['input_checkpoint'] != current['checkpoint'] or current['step'] >= len(profile['schedule']):
                raise ValueError('Reserve an unexhausted current expert checkpoint')
            workers = [body['worker']]
        for worker in workers:
            ledger.public_key(worker)
        auditing.debit(s, owner, params['claim_bond'])
        assignment = {'id': protocol.transaction_id(envelope), 'owner': owner, 'workers': workers,
            'expert_kind': claim_kind, 'input_checkpoint': current['checkpoint'],
            'bond': params['claim_bond'], 'expires': expert_admission.deadline(s, s['height'] + params['lease_blocks']),
            'audit_budget': body['audit_budget']}
        auditing.lock(s, body['audit_budget'], owner, workers, assignment['id'])
        s['assignment'] = assignment
        return
    assignment = s['assignment']
    if (not assignment or assignment.get('expert_kind') != claim_kind or assignment['owner'] != owner
            or s['height'] > assignment['expires'] or s['candidate']):
        raise ValueError('No matching current expert reservation')
    if features:
        prospective = profile['format'] == PROSPECTIVE
        if prospective != (kind == 'claim_expert_prefix'):
            raise ValueError('Use the prescribed prefix claim format')
        if prospective:
            profile = resolve_prefix(profile, body['feature_root'], body['batch_roots'])
        elif body['feature_root'] != profile['feature_root']:
            raise ValueError('Claim the feature bank committed by this prepared job')
        transcript = root(body['transcript_root'])
        output = ({'feature_root': body['feature_root'], 'batch_roots': body['batch_roots']}
                  if prospective else body['feature_root'])
        receipts, stages, work = body['workers'], profile['feature_stages'], None
    else:
        # The job selects batches. A worker cannot substitute easier input.
        window = body['window']
        if not isinstance(window, dict) or not isinstance(window.get('steps'), list):
            raise ValueError('Invalid bounded expert window')
        count = len(window['steps'])
        integer(count, 1, 4)
        if current['step'] + count > len(profile['schedule']) or s['period_steps'] + count > params['steps_per_period']:
            raise ValueError('Prepared training or the issuance period is exhausted')
        batches = [profile['batch_roots'][index] for index in profile['schedule'][current['step']:current['step'] + count]]
        work = expert_window.validate(profile['parent'], current, window, body['intermediates'],
            batches, profile['numerical_profile'], s['paid_work'])
        transcript, output = work['record_root'], work['output_checkpoint']
        receipts, stages = [body['worker']], work['steps']
    if not isinstance(receipts, list) or len(receipts) != len(assignment['workers']):
        raise ValueError('Incomplete reserved worker receipts')
    for rank, signed in enumerate(receipts):
        payload, worker = protocol.verify(signed)
        if worker != assignment['workers'][rank] or payload != receipt(s['chain_id'], assignment, output, transcript, rank):
            raise ValueError('Worker receipt differs from the reserved expert execution')
    candidate = {'kind': claim_kind, 'id': protocol.transaction_id(envelope), 'owner': owner,
        'bond': assignment['bond'], 'workers': assignment['workers'], 'record_root': transcript,
        'model_root': current['state_root'], 'input_checkpoint': copy.deepcopy(current),
        'prepared': profile['prepared'], 'feature_root': profile['feature_root'],
        'parent_checkpoint': copy.deepcopy(profile['parent']),
        'numerical_profile': profile['numerical_profile'], 'stages': stages,
        'deadline': s['height'] + params['challenge_blocks'],
        'expires': expert_admission.deadline(s, s['height'] + params['max_claim_blocks']),
        'challenge': None}
    if not features:
        candidate.update(window=copy.deepcopy(window), intermediates=copy.deepcopy(body['intermediates']),
            output_checkpoint=copy.deepcopy(window['output']), work_ids=work['work_ids'],
            model_root=window['output']['state_root'], feature_claim=s['expert_work']['feature_claim'])
    elif prospective:
        candidate['batch_roots'] = list(profile['batch_roots'])
    s['candidate'] = candidate
    auditing.attach(s, assignment['audit_budget'])
    s['assignment'] = None


def settle(s, claim):
    if claim['kind'] == 'expert_features':
        if s['expert_work']['feature_claim'] is not None:
            raise ValueError('Feature production is already accepted')
        s['expert_work']['feature_claim'] = claim['id']
        if 'batch_roots' in claim:
            s['expert_work'].update(feature_root=claim['feature_root'], batch_roots=list(claim['batch_roots']))
        return
    for work in claim['work_ids']:
        if work in s['paid_work']:
            raise ValueError('Expert update has already been paid')
    steps = len(claim['work_ids'])
    reward = steps * s['manifest']['params']['reward_atoms']
    s['paid_work'].update({work: claim['id'] for work in claim['work_ids']})
    s['issued'] += reward
    s['training_round'] += steps
    s['period_steps'] += steps
    ledger.account(s, claim['workers'][0])['balance'] += reward
    s['expert_work']['checkpoint'] = copy.deepcopy(claim['output_checkpoint'])
    s['model_root'] = claim['output_checkpoint']['state_root']
    from . import expert_admission
    expert_admission.trained(s, claim)


def replay_report(claim, report):
    """Bind a configured local executor's verdict; this does not execute tensors.

    The backend must actually audit its pinned source and bytes. Passing an
    untrusted miner's report through this check would violate that requirement.
    """
    binding = {'parent': identity(claim['parent_checkpoint']), 'prepared': claim['prepared'],
        'input_checkpoint': claim['input_checkpoint']['checkpoint'],
        'output_root': claim['feature_root'] if claim['kind'] == 'expert_features'
                       else claim['output_checkpoint']['checkpoint'],
        'feature_root': claim['feature_root'], 'numerical_profile': claim['numerical_profile'],
        'feature_claim': claim.get('feature_claim')}
    if 'batch_roots' in claim:
        binding['batch_roots'] = claim['batch_roots']
    required = {'format', 'claim_id', 'record_root', 'binding', 'stages'}
    if (not isinstance(report, dict) or set(report) != required
            or report['format'] != FORMAT + '/replay' or report['claim_id'] != claim['id']
            or report['record_root'] != claim['record_root'] or report['binding'] != binding
            or not isinstance(report['stages'], list) or len(report['stages']) != claim['stages']):
        raise ValueError('Expert replay must bind and cover the complete claimed execution')
    for index, stage in enumerate(report['stages']):
        if (not isinstance(stage, dict) or set(stage) != {'stage', 'valid'}
                or type(stage['stage']) is not int or stage['stage'] != index or type(stage['valid']) is not bool):
            raise ValueError('Invalid ordered expert audit coverage')
    valid = all(stage['valid'] for stage in report['stages'])
    return {'valid': valid, 'record_root': claim['record_root'], 'stages': report['stages'],
            'coverage_root': auditing.coverage(claim) if valid else None}
