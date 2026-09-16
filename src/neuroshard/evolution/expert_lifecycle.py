"""Separate quality approval and escrowed inference for native expert work.

This opt-in candidate profile freezes one graph extension before training.
Execution reports require the existing funded native audit quorum. A quality
claim or a response never issues training rewards.
"""
import copy

from neuroshard.demo import protocol
from neuroshard.lab import state as ledger
from . import auditing, lifecycle, portable_lifecycle, serving_graph
from .reference_data import identity
from .schema import integer, root

FORMAT = 'neuroshard-expert-lifecycle-v1'
PROSPECTIVE = 'neuroshard-prospective-expert-lifecycle-v1'
PROFILE_FIELDS = {'format', 'serving_graph', 'candidate_graph', 'quality', 'price_per_token', 'max_tokens'}
PROSPECTIVE_FIELDS = PROFILE_FIELDS - {'candidate_graph'} | {'candidate_template'}
REPORT_FIELDS = {'format', 'policy_root', 'baseline_graph', 'candidate_graph', 'prepared', 'passed', 'results_root'}
KINDS = ('expert_quality', 'expert_inference')
FIELDS = {
    'quality_expert': {'report', 'transcript_root', 'audit_budget'},
    'infer_expert': {'graph', 'question', 'max_tokens', 'workers', 'max_price', 'expires_in'},
    'respond_expert': {'job_id', 'outputs', 'text', 'transcript_root', 'workers', 'audit_budget'},
}


def initialize(state):
    profile = state['manifest']['expert_lifecycle']
    prospective = profile.get('format') == PROSPECTIVE
    serving_graph.fields(profile, PROSPECTIVE_FIELDS if prospective else PROFILE_FIELDS, 'Invalid expert lifecycle profile')
    if profile['format'] not in (FORMAT, PROSPECTIVE) or 'expert_work' not in state or not auditing.native(state):
        raise ValueError('Expert serving requires funded native expert execution')
    previous = serving_graph.validate(profile['serving_graph'])
    candidate = serving_graph.validate(profile['candidate_template'] if prospective else profile['candidate_graph'],
                                       allow_untrained=prospective)
    target = training_expert(candidate) if prospective else 'protocol'
    work = state['manifest']['expert_work']
    if (previous['descriptor']['format'] not in (serving_graph.PRIOR, serving_graph.COMPOSED, serving_graph.EXTENSIBLE)
            or candidate['descriptor']['format'] not in (serving_graph.COMPOSED, serving_graph.EXTENSIBLE)
            or candidate['descriptor']['previous_graph'] != identity(previous['descriptor'])
            or any(previous[k] != candidate[k] for k in ('parent', 'interpreter_assets',
                'interpreter_prompt', 'tokenizer', 'numerical_profile', 'executor_root'))
            or set(candidate['experts']) != set(previous['experts']) | {target}
            or any(value != candidate['experts'][name] for name, value in previous['experts'].items() if name != target)
            or serving_graph.rules(candidate)[:len(serving_graph.rules(previous))] != serving_graph.rules(previous)
            or previous['descriptor']['interpretation'] != candidate['descriptor']['interpretation']
            or candidate['parent'] != work['parent']
            or candidate['numerical_profile'] != work['numerical_profile']):
        raise ValueError('The candidate must preserve the committed earlier graph')
    expert = candidate['experts'][target]
    if (expert['step'] != (0 if prospective else len(work['schedule']))
            or any(expert[k] != work['checkpoint'][k] for k in ('parent', 'job', 'split', 'recipe', 'boundaries'))):
        raise ValueError('Freeze the terminal graph of this complete training job')
    if prospective and expert != work['checkpoint']:
        raise ValueError('A prospective graph must bind the exact initial expert state')
    serving_graph.fields(profile['quality'], {'policy_root', 'prepared', 'stages'}, 'Invalid frozen quality policy')
    for key in ('policy_root', 'prepared'):
        root(profile['quality'][key])
    integer(profile['quality']['stages'], 1, 4096)
    integer(profile['price_per_token'], 1, 10**9)
    integer(profile['max_tokens'], 1, 256)
    state['expert_lifecycle'] = {'serving_graph': copy.deepcopy(previous), 'quality_claim': None,
        'quality_closed': False, 'jobs': {}, 'results': {}, 'history': []}
    state['serving_root'] = identity(previous)


def training_expert(template):
    pending = [name for name, checkpoint in template['experts'].items() if checkpoint['step'] == 0]
    if len(pending) != 1:
        raise ValueError('A prospective graph prescribes exactly one untrained expert')
    return pending[0]


def materialize_graph(template, completed):
    """Bind a pre-training architecture to the actual settled numerical output."""
    serving_graph.validate(template, allow_untrained=True)
    target = training_expert(template)
    initial = template['experts'][target]
    if (initial['step'] != 0 or completed['step'] != initial['recipe']['steps']
            or any(completed[key] != initial[key] for key in ('parent', 'job', 'split', 'recipe', 'boundaries'))):
        raise ValueError('Materialize only the completed prescribed expert job')
    graph = copy.deepcopy(template)
    graph['experts'][target] = copy.deepcopy(completed)
    for entry in graph['descriptor']['experts']:
        if entry['id'] == target:
            entry['checkpoint'] = completed['checkpoint']
    return serving_graph.validate(graph)


def candidate_graph(state):
    profile = state['manifest']['expert_lifecycle']
    if profile['format'] == PROSPECTIVE:
        return materialize_graph(profile['candidate_template'], state['expert_work']['checkpoint'])
    return profile['candidate_graph']


def escrow(state):
    return sum(job['escrow'] for job in state.get('expert_lifecycle', {}).get('jobs', {}).values())


def advance(state):
    if 'expert_lifecycle' not in state:
        return
    life = state['expert_lifecycle']
    for key, job in list(life['jobs'].items()):
        if state['height'] > job['expires'] and job['claim_id'] is None:
            ledger.account(state, job['payer'])['balance'] += job['escrow']
            life['results'][key] = {'id': key, 'status': 'expired', 'graph': identity(job['graph']),
                'refunded_atoms': job['escrow'], 'height': state['height']}
            del life['jobs'][key]
    life['history'] = life['history'][-128:]
    lifecycle.trim_results(life)


def service_statement(claim):
    names = ('kind', 'model_root', 'graph', 'record_root', 'stages', 'executor_root')
    value = {key: claim[key] for key in names}
    if claim['kind'] == 'expert_quality':
        value.update(report=claim['report'], baseline_graph=claim['baseline_graph'])
    elif claim['kind'] == 'expert_inference':
        value.update(job_id=claim['job_id'], request=claim['request'], outputs=claim['outputs'], text=claim['text'])
    else:
        raise ValueError('Unknown expert service')
    return value


def transcript_binding(claim):
    return {'service': identity({k: v for k, v in service_statement(claim).items() if k != 'record_root'})}


def inference_receipt(chain_id, job, outputs, text, transcript_root, rank):
    return {'domain': FORMAT + '/response', 'chain_id': chain_id, 'job_id': job['id'],
        'graph': identity(job['graph']), 'request': identity(job['request']), 'outputs': outputs,
        'text': text, 'transcript_root': transcript_root, 'rank': rank}


def apply(state, owner, body, envelope):
    if 'expert_lifecycle' not in state:
        raise ValueError('Genesis does not enable expert graph serving')
    life, profile = state['expert_lifecycle'], state['manifest']['expert_lifecycle']
    if body['kind'] == 'quality_expert':
        if state['expert_work']['checkpoint']['step'] != len(state['manifest']['expert_work']['schedule']):
            raise ValueError('Settle the complete expert job before its separate quality decision')
        candidate = candidate_graph(state)
        target = training_expert(profile['candidate_template']) if profile['format'] == PROSPECTIVE else 'protocol'
        if (life['quality_closed'] or life['quality_claim'] is not None
                or state['expert_work']['checkpoint'] != candidate['experts'][target]
                or state['expert_work']['feature_claim'] is None):
            raise ValueError('Settle the complete expert job before its separate quality decision')
        report = body['report']
        serving_graph.fields(report, REPORT_FIELDS, 'Invalid expert quality report')
        expected = {'format': FORMAT + '/quality', 'policy_root': profile['quality']['policy_root'],
            'baseline_graph': identity(profile['serving_graph']), 'candidate_graph': identity(candidate),
            'prepared': profile['quality']['prepared']}
        if any(report[k] != value for k, value in expected.items()) or type(report['passed']) is not bool:
            raise ValueError('Quality report changed the frozen graph, data or policy')
        root(report['results_root'])
        portable_lifecycle.new_claim(state, owner, body, envelope, kind='expert_quality',
            graph=copy.deepcopy(candidate), baseline_graph=copy.deepcopy(profile['serving_graph']),
            report=copy.deepcopy(report), model_root=identity(candidate), executor_root=candidate['executor_root'],
            stages=profile['quality']['stages'])
        life['quality_claim'] = state['candidate']['id']
    elif body['kind'] == 'infer_expert':
        graph = life['serving_graph']
        if root(body['graph']) != identity(graph) or len(life['jobs']) >= 16:
            raise ValueError('Serving graph changed or inference queue is full')
        maximum = integer(body['max_tokens'], 1, profile['max_tokens'])
        plan = serving_graph.calls(graph, body['question'], maximum)
        needed = {rank for call in plan for rank in serving_graph.ownership(graph, call['model'])}
        workers = body['workers']
        if not isinstance(workers, dict) or set(workers) != needed:
            raise ValueError('Assign exactly every participating graph owner')
        for worker in workers.values():
            ledger.public_key(worker)
        amount = integer(body['max_price'], serving_graph.maximum_price(graph, plan, profile['price_per_token']), 2**60)
        duration = integer(body['expires_in'], state['manifest']['params']['max_claim_blocks'] + 1, 100000)
        auditing.debit(state, owner, amount)
        key = protocol.transaction_id(envelope)
        life['jobs'][key] = {'id': key, 'payer': owner, 'graph': copy.deepcopy(graph),
            'request': {'question': body['question'], 'max_tokens': maximum, 'calls': plan},
            'workers': copy.deepcopy(workers), 'unit_price': profile['price_per_token'], 'escrow': amount,
            'expires': state['height'] + duration, 'claim_id': None}
    elif body['kind'] == 'respond_expert':
        job = life['jobs'].get(root(body['job_id']))
        if not job or job['workers']['0'] != owner or job['claim_id'] or state['height'] > job['expires']:
            raise ValueError('No matching available expert inference assignment')
        graph, outputs, text = job['graph'], body['outputs'], body['text']
        payments = serving_graph.payments(graph, job['request']['calls'], outputs, job['unit_price'])
        if not isinstance(text, str) or len(text.encode()) > 32768 or sum(payments.values()) > job['escrow']:
            raise ValueError('Response text or price exceeds its bounds')
        receipts = body['workers']
        if not isinstance(receipts, dict) or set(receipts) != set(job['workers']):
            raise ValueError('Require receipts from every participating graph owner')
        for rank, signed in receipts.items():
            payload, signer = protocol.verify(signed)
            if signer != job['workers'][rank] or payload != inference_receipt(
                    state['chain_id'], job, outputs, text, body['transcript_root'], rank):
                raise ValueError('Response receipt changed its graph, request or neural calls')
        portable_lifecycle.new_claim(state, owner, body, envelope, kind='expert_inference',
            job_id=job['id'], graph=copy.deepcopy(graph), model_root=identity(graph),
            executor_root=graph['executor_root'], request=copy.deepcopy(job['request']),
            outputs=copy.deepcopy(outputs), text=text, stages=sum(len(output['token_ids']) for output in outputs),
            expires=min(job['expires'], state['height'] + state['manifest']['params']['max_claim_blocks']))
        job['claim_id'] = state['candidate']['id']


def settled(state, claim, accepted):
    if claim.get('kind') not in KINDS:
        return
    life = state['expert_lifecycle']
    if claim['kind'] == 'expert_quality':
        life['quality_claim'] = None
        if accepted:
            promoted = claim['report']['passed']
            if promoted:
                life['serving_graph'] = copy.deepcopy(claim['graph'])
                state['serving_root'] = identity(claim['graph'])
            life['quality_closed'] = True
            life['history'].append({'id': claim['id'], 'kind': 'quality', 'report': claim['report'],
                'promoted': promoted, 'height': state['height']})
        return
    job = life['jobs'][claim['job_id']]
    job['claim_id'] = None
    if accepted:
        payments = serving_graph.payments(job['graph'], job['request']['calls'], claim['outputs'], job['unit_price'])
        paid = sum(payments.values())
        for rank, amount in payments.items():
            ledger.account(state, job['workers'][rank])['balance'] += amount
        ledger.account(state, job['payer'])['balance'] += job['escrow'] - paid
        life['results'][job['id']] = {'id': job['id'], 'status': 'completed', 'graph': identity(job['graph']),
            'text': claim['text'], 'outputs': claim['outputs'], 'workers': job['workers'], 'payments': payments,
            'paid_atoms': paid, 'refunded_atoms': job['escrow'] - paid, 'height': state['height']}
        del life['jobs'][job['id']]
        lifecycle.trim_results(life)


def replay_report(claim, report):
    serving_graph.fields(report, {'format', 'statement', 'stages'}, 'Invalid expert service replay report')
    if (report['format'] != FORMAT + '/replay' or report['statement'] != identity(service_statement(claim))
            or not isinstance(report['stages'], list) or len(report['stages']) != claim['stages']):
        raise ValueError('Audit must bind and cover the entire expert service obligation')
    for index, stage in enumerate(report['stages']):
        serving_graph.fields(stage, {'stage', 'valid'}, 'Invalid expert service audit stage')
        if type(stage['stage']) is not int or stage['stage'] != index or type(stage['valid']) is not bool:
            raise ValueError('Invalid ordered expert service audit coverage')
    valid = all(stage['valid'] for stage in report['stages'])
    return {'valid': valid, 'record_root': claim['record_root'], 'stages': report['stages'],
            'coverage_root': auditing.coverage(claim) if valid else None}
