"""Opt-in job admission, quality approval and inference for portable shards.

Native bonded weight authorizes immutable recipes and audits execution. None of
these transitions imports paid work or interprets a quality vote as a proof of
general intelligence. Numerical evaluation stays outside consensus.
"""
import copy

from neuroshard.demo import protocol
from neuroshard.lab import state as ledger
from . import auditing, lifecycle, portable_work
from .reference_data import identity
from .schema import integer, root

FORMAT = 'neuroshard-portable-lifecycle-v1'
PROFILE_FIELDS = {'format', 'serving_checkpoint', 'tokenizer_root', 'eos_id',
                  'price_per_token', 'max_prompt_tokens', 'max_new_tokens',
                  'proposal_blocks', 'job_blocks', 'executor_root'}
JOB_FIELDS = {'parent', 'job', 'prepared', 'reference_root', 'executor_root', 'max_step',
              'max_window_steps', 'quality'}
QUALITY_FIELDS = {'policy_root', 'baseline_checkpoint', 'stages'}
REPORT_FIELDS = {'format', 'policy_root', 'baseline_checkpoint',
                 'candidate_checkpoint', 'prepared', 'passed', 'results_root'}
FIELDS = {
    'propose_shard_job': {'job'},
    'vote_shard_job': {'proposal_id', 'approve'},
    'quality_shards': {'job_id', 'report', 'transcript_root', 'audit_budget'},
    'infer_shards': {'checkpoint', 'workers', 'prompt_ids', 'max_tokens', 'max_price', 'expires_in'},
    'respond_shards': {'job_id', 'token_ids', 'transcript_root', 'workers', 'audit_budget'},
}
SERVICE_KINDS = ('portable_quality', 'portable_inference')


def fields(value, expected, message):
    if not isinstance(value, dict) or set(value) != expected:
        raise ValueError(message)


def initialize(state):
    profile = state['manifest']['portable_lifecycle']
    fields(profile, PROFILE_FIELDS, 'Invalid portable lifecycle profile')
    if (profile['format'] != FORMAT or 'portable_work' not in state
            or not auditing.native(state) or 'lifecycle' in state):
        raise ValueError('Portable lifecycle requires native portable replay quorum')
    serving = portable_work.validate(profile['serving_checkpoint'])
    current = state['portable_work']['checkpoint']
    if serving['config'] != current['config'] or serving['boundaries'] != current['boundaries']:
        raise ValueError('Initial serving and training layouts must agree')
    root(profile['tokenizer_root'])
    root(profile['executor_root'])
    integer(profile['eos_id'], 0, serving['config']['vocab_size'] - 1)
    integer(profile['price_per_token'], 1, 10**9)
    integer(profile['max_prompt_tokens'], 1, 4096)
    integer(profile['max_new_tokens'], 1, 256)
    integer(profile['proposal_blocks'], state['manifest']['params']['activation_blocks'] + 1, 100000)
    integer(profile['job_blocks'], state['manifest']['params']['max_claim_blocks'] + 1, 1000000)
    state['serving_root'] = serving['state_root']
    state['portable_lifecycle'] = {
        'serving_checkpoint': copy.deepcopy(serving), 'serving_executor': profile['executor_root'],
        'proposal': None, 'active': None,
        'seen_jobs': {}, 'history': [], 'jobs': {}, 'results': {},
    }


def escrow(state):
    life = state.get('portable_lifecycle', {})
    proposal = life.get('proposal')
    return (proposal['bond'] if proposal else 0) + sum(j['escrow'] for j in life.get('jobs', {}).values())


def active_profile(state):
    """Called at reservation and claim time, so an expired recipe cannot pay."""
    if 'portable_lifecycle' not in state:
        return state['manifest']['portable_work']
    active = state['portable_lifecycle']['active']
    if not active or active['closed'] or state['height'] > active['expires']:
        raise ValueError('Activate a live portable job before reserving work')
    return active['job']


def validate_job(state, job):
    fields(job, JOB_FIELDS, 'Invalid portable job schema')
    current = state['portable_work']['checkpoint']
    life = state['portable_lifecycle']
    if job['parent'] != identity(current):
        raise ValueError('A new job must start at the currently settled checkpoint')
    for name in ('parent', 'job', 'prepared', 'reference_root', 'executor_root'):
        root(job[name])
    if job['job'] in life['seen_jobs'] or job['job'] == current['job']:
        raise ValueError('A prepared computation may be activated only once')
    integer(job['max_step'], current['step'] + 1, min(current['step'] + 256, 2**24 - 1))
    integer(job['max_window_steps'], 1, 4)
    quality = job['quality']
    fields(quality, QUALITY_FIELDS, 'Freeze the quality policy before job activation')
    root(quality['policy_root'])
    integer(quality['stages'], 1, 4096)
    if quality['baseline_checkpoint'] != identity(life['serving_checkpoint']):
        raise ValueError('Quality must compare against the current serving checkpoint')


def advance(state):
    if 'portable_lifecycle' not in state:
        return
    life = state['portable_lifecycle']
    proposal = life['proposal']
    idle = not state['candidate'] and not state['assignment']
    if proposal:
        accepted = (state['height'] >= proposal['activate_after']
            and lifecycle.quorum(proposal['electorate'], proposal['votes'])
            and lifecycle.quorum(lifecycle.weights(state), proposal['votes']))
        expired = state['height'] > proposal['expires']
        if expired or (accepted and idle):
            if accepted and not expired:
                validate_job(state, proposal['job'])
                life['active'] = {'id': proposal['id'], 'job': proposal['job'],
                    'start': state['portable_work']['checkpoint']['step'],
                    'expires': state['height'] + state['manifest']['portable_lifecycle']['job_blocks'],
                    'closed': False, 'quality_report': None}
                life['seen_jobs'][proposal['job']['job']] = proposal['id']
                state['data_root'] = proposal['job']['prepared']
            ledger.account(state, proposal['owner'])['balance'] += proposal['bond']
            life['history'].append({'id': proposal['id'], 'kind': 'activation',
                'accepted': accepted and not expired, 'height': state['height']})
            life['proposal'] = None
    active = life['active']
    if active and not active['closed'] and state['height'] > active['expires'] and idle:
        active['closed'] = True
    for key, job in list(life['jobs'].items()):
        if state['height'] > job['expires'] and job['claim_id'] is None:
            ledger.account(state, job['payer'])['balance'] += job['escrow']
            life['results'][key] = {'id': key, 'status': 'expired', 'checkpoint': identity(job['checkpoint']),
                'refunded_atoms': job['escrow'], 'height': state['height']}
            del life['jobs'][key]
    life['history'] = life['history'][-128:]
    lifecycle.trim_results(life)


def service_statement(claim):
    """Everything a trusted numerical service backend must independently check."""
    names = ('kind', 'model_root', 'input_checkpoint', 'record_root', 'stages', 'executor_root')
    statement = {name: claim[name] for name in names}
    if claim['kind'] == 'portable_quality':
        statement.update(job_id=claim['job_id'], report=claim['report'],
                         baseline_checkpoint=claim['baseline_checkpoint'])
    elif claim['kind'] == 'portable_inference':
        statement.update(job_id=claim['job_id'], request=claim['request'], token_ids=claim['token_ids'])
    else:
        raise ValueError('Unknown portable service')
    return statement


def transcript_binding(claim):
    # The transcript cannot commit its own hash. Its binding commits all other
    # execution fields; the outer audit statement also commits the closed graph.
    return {'service': identity({key: value for key, value in service_statement(claim).items()
                                 if key != 'record_root'})}


def new_claim(state, owner, body, envelope, **values):
    if state['assignment'] or state['candidate']:
        raise ValueError('Finish pending native work before a portable service claim')
    params = state['manifest']['params']
    auditing.debit(state, owner, params['claim_bond'])
    claim = {'id': protocol.transaction_id(envelope), 'owner': owner,
        'bond': params['claim_bond'], 'challenge': None,
        'record_root': root(body['transcript_root']), 'workers': [],
        'deadline': state['height'] + params['challenge_blocks'],
        'expires': state['height'] + params['max_claim_blocks'], **values}
    state['candidate'] = claim
    auditing.lock(state, body['audit_budget'], owner, [], claim['id'])
    auditing.attach(state, body['audit_budget'])


def inference_receipt(chain_id, job, token_ids, transcript_root, rank):
    return {'domain': FORMAT + '/response', 'chain_id': chain_id, 'job_id': job['id'],
        'checkpoint': identity(job['checkpoint']), 'token_ids': token_ids,
        'transcript_root': transcript_root, 'rank': rank}


def apply(state, owner, body, envelope):
    if 'portable_lifecycle' not in state:
        raise ValueError('Genesis does not enable the portable lifecycle')
    life = state['portable_lifecycle']
    profile = state['manifest']['portable_lifecycle']
    kind = body['kind']
    if kind == 'propose_shard_job':
        if (life['proposal'] or state['assignment'] or state['candidate']
                or (life['active'] and not life['active']['closed'])):
            raise ValueError('Finish the current portable job before proposing another')
        validate_job(state, body['job'])
        bond = state['manifest']['params']['claim_bond']
        auditing.debit(state, owner, bond)
        life['proposal'] = {'id': protocol.transaction_id(envelope), 'owner': owner, 'bond': bond,
            'job': copy.deepcopy(body['job']), 'electorate': lifecycle.weights(state), 'votes': {},
            'activate_after': state['height'] + state['manifest']['params']['activation_blocks'],
            'expires': state['height'] + profile['proposal_blocks']}
    elif kind == 'vote_shard_job':
        proposal = life['proposal']
        if not proposal or proposal['id'] != body['proposal_id'] or state['height'] > proposal['expires']:
            raise ValueError('No matching live portable job proposal')
        if type(body['approve']) is not bool or owner not in proposal['electorate'] or owner in proposal['votes']:
            raise ValueError('Each native validator owner may cast one explicit vote')
        proposal['votes'][owner] = body['approve']
    elif kind == 'quality_shards':
        job = active_profile(state)
        active, current = life['active'], state['portable_work']['checkpoint']
        if active['id'] != body['job_id'] or current['step'] != job['max_step']:
            raise ValueError('Settle every activated training window before quality approval')
        report = body['report']
        fields(report, REPORT_FIELDS, 'Invalid portable quality report')
        expected = {'format': FORMAT + '/quality', 'policy_root': job['quality']['policy_root'],
            'baseline_checkpoint': job['quality']['baseline_checkpoint'],
            'candidate_checkpoint': identity(current), 'prepared': job['prepared']}
        if any(report[k] != value for k, value in expected.items()) or type(report['passed']) is not bool:
            raise ValueError('Quality report differs from the activated policy or settled checkpoint')
        root(report['results_root'])
        new_claim(state, owner, body, envelope, kind='portable_quality', job_id=active['id'],
            executor_root=job['executor_root'],
            report=copy.deepcopy(report), input_checkpoint=copy.deepcopy(current),
            baseline_checkpoint=copy.deepcopy(life['serving_checkpoint']), model_root=current['state_root'],
            stages=job['quality']['stages'], expires=min(active['expires'],
                state['height'] + state['manifest']['params']['max_claim_blocks']))
        active['quality_report'] = identity(report)
    elif kind == 'infer_shards':
        checkpoint = life['serving_checkpoint']
        if body['checkpoint'] != identity(checkpoint) or len(life['jobs']) >= 16:
            raise ValueError('Serving checkpoint changed or portable inference queue is full')
        workers = body['workers']
        if not isinstance(workers, list) or len(workers) != len(checkpoint['boundaries']) - 1:
            raise ValueError('Assign every serving partition')
        for worker in workers:
            ledger.public_key(worker)
        tokens = body['prompt_ids']
        if not isinstance(tokens, list) or not 1 <= len(tokens) <= profile['max_prompt_tokens']:
            raise ValueError('Portable prompt exceeds its token bound')
        for token in tokens:
            integer(token, 0, checkpoint['config']['vocab_size'] - 1)
        maximum = integer(body['max_tokens'], 1, profile['max_new_tokens'])
        if len(tokens) + maximum > checkpoint['config']['max_position_embeddings']:
            raise ValueError('Portable request exceeds model context')
        amount = integer(body['max_price'], maximum * profile['price_per_token'], 2**60)
        duration = integer(body['expires_in'], state['manifest']['params']['max_claim_blocks'] + 1, 100000)
        auditing.debit(state, owner, amount)
        key = protocol.transaction_id(envelope)
        life['jobs'][key] = {'id': key, 'payer': owner, 'checkpoint': copy.deepcopy(checkpoint),
            'workers': workers, 'prompt_ids': tokens, 'max_tokens': maximum, 'eos_id': profile['eos_id'],
            'tokenizer_root': profile['tokenizer_root'], 'escrow': amount,
            'executor_root': life['serving_executor'],
            'unit_price': profile['price_per_token'], 'expires': state['height'] + duration, 'claim_id': None}
    elif kind == 'respond_shards':
        job = life['jobs'].get(root(body['job_id']))
        if not job or job['workers'][0] != owner or job['claim_id'] or state['height'] > job['expires']:
            raise ValueError('No matching unclaimed portable inference assignment')
        tokens = body['token_ids']
        if not isinstance(tokens, list) or not 1 <= len(tokens) <= job['max_tokens']:
            raise ValueError('Invalid portable response length')
        for token in tokens:
            integer(token, 0, job['checkpoint']['config']['vocab_size'] - 1)
        if job['eos_id'] in tokens[:-1] or (len(tokens) < job['max_tokens'] and tokens[-1] != job['eos_id']):
            raise ValueError('Response must obey the committed greedy stopping rule')
        receipts = body['workers']
        if not isinstance(receipts, list) or len(receipts) != len(job['workers']):
            raise ValueError('Require receipts from every serving partition')
        for rank, signed in enumerate(receipts):
            payload, signer = protocol.verify(signed)
            if signer != job['workers'][rank] or payload != inference_receipt(
                    state['chain_id'], job, tokens, body['transcript_root'], rank):
                raise ValueError('Response receipt differs from the paid request')
        request = {k: job[k] for k in ('prompt_ids', 'max_tokens', 'eos_id', 'tokenizer_root')}
        new_claim(state, owner, body, envelope, kind='portable_inference', job_id=job['id'],
            executor_root=job['executor_root'],
            input_checkpoint=copy.deepcopy(job['checkpoint']), request=request, token_ids=tokens,
            model_root=job['checkpoint']['state_root'], stages=len(tokens),
            expires=min(job['expires'], state['height'] + state['manifest']['params']['max_claim_blocks']))
        job['claim_id'] = state['candidate']['id']


def settled(state, claim, accepted):
    if claim.get('kind') not in SERVICE_KINDS:
        return
    life = state['portable_lifecycle']
    if claim['kind'] == 'portable_quality':
        if accepted:
            promoted = claim['report']['passed']
            if promoted:
                life['serving_checkpoint'] = copy.deepcopy(claim['input_checkpoint'])
                life['serving_executor'] = claim['executor_root']
                state['serving_root'] = claim['model_root']
            life['active']['closed'] = True
            life['history'].append({'id': claim['id'], 'kind': 'quality',
                'report': claim['report'], 'promoted': promoted, 'height': state['height']})
        else:
            # Any publisher may report on this fixed candidate and policy. A
            # forged or unavailable first report must not permanently poison
            # the quality slot. There is no new model selection on retry.
            life['active']['quality_report'] = None
    else:
        job = life['jobs'][claim['job_id']]
        job['claim_id'] = None
        if accepted:
            paid = len(claim['token_ids']) * job['unit_price']
            payments = portable_work.payments(job['checkpoint'], paid)
            for worker, amount in zip(job['workers'], payments):
                ledger.account(state, worker)['balance'] += amount
            ledger.account(state, job['payer'])['balance'] += job['escrow'] - paid
            life['results'][job['id']] = {'id': job['id'], 'status': 'completed',
                'checkpoint': identity(job['checkpoint']), 'token_ids': claim['token_ids'],
                'workers': job['workers'], 'payments': payments, 'paid_atoms': paid,
                'refunded_atoms': job['escrow'] - paid, 'height': state['height']}
            del life['jobs'][job['id']]
            lifecycle.trim_results(life)


def replay_report(claim, partitions):
    count = len(claim['input_checkpoint']['boundaries']) - 1
    binding = {'statement': identity(service_statement(claim))}
    if not isinstance(partitions, list) or len(partitions) != count:
        raise ValueError('Replay every portable service partition')
    for rank, report in enumerate(partitions):
        if (report['rank'] != rank or report['binding'] != binding
                or report['transcript_root'] != claim['record_root'] or type(report['valid']) is not bool):
            raise ValueError('Portable service report differs from its complete obligation')
    valid = all(report['valid'] for report in partitions)
    return {'valid': valid, 'record_root': claim['record_root'],
        'coverage_root': auditing.coverage(claim) if valid else None, 'partitions': partitions,
        'stages': [{'stage': index, 'valid': valid} for index in range(claim['stages'])]}
