"""Native, collateralized provider offers and recoverable expert assignments.

Advertisements are service offers, not hardware or independence proofs. Existing
complete funded replay authorizes payment. This opt-in module never issues NEURO.
"""
import copy
from neuroshard.client.provider_wire import endpoint

from neuroshard.demo import protocol
from neuroshard.lab import state as ledger
from . import auditing, serving_graph
from .reference_data import identity
from .schema import integer, root

FORMAT = 'neuroshard-provider-hosting-v1'
PROFILE = {'format': FORMAT, 'lease_bond': 1_000_000, 'prepare_blocks': 128,
           'execution_blocks': 256, 'cooldown_blocks': 32, 'max_attempts': 3,
           'max_providers': 256, 'max_offers': 1024}
FIELDS = {
    'fund_hosted_audit': {'publisher', 'stage_limit', 'expires_in', 'graph', 'request_root'},
    'register_provider': {'endpoint', 'certificate', 'collateral'},
    'update_provider': {'endpoint', 'certificate', 'deposit'},
    'withdraw_provider': {'amount'},
    'offer_expert': {'graph', 'rank', 'fee', 'capacity', 'expires_in'},
    'cancel_expert_offer': {'offer_id'},
    'lease_expert': {'graph', 'question', 'max_tokens', 'offers', 'max_price',
                     'max_provider_fee', 'audit_budget', 'expires_in'},
    'accept_hosted_job': {'job_id', 'assignment_root'},
    'replace_hosted_job': {'job_id', 'offers', 'audit_budget'},
}


def initialize(state):
    profile = state['manifest']['hosting']
    serving_graph.fields(profile, PROFILE, 'Invalid provider-hosting profile')
    if profile['format'] != FORMAT or 'expert_lifecycle' not in state or not auditing.native(state):
        raise ValueError('Hosting requires native expert serving and complete audit quorums')
    integer(profile['lease_bond'], 1, 10**12)
    for name in ('prepare_blocks', 'execution_blocks', 'cooldown_blocks'):
        integer(profile[name], 1, 10000)
    integer(profile['max_attempts'], 1, 8)
    integer(profile['max_providers'], 4, 1024)
    integer(profile['max_offers'], 4, 4096)
    state['hosting'] = {'providers': {}, 'offers': {}, 'leases': {}, 'history': []}


def escrow(state):
    market = state.get('hosting', {})
    return (sum(p['collateral'] for p in market.get('providers', {}).values())
            + sum(lease['fee_escrow'] for lease in market.get('leases', {}).values()))


def invariant(state):
    if 'hosting' not in state:
        return
    market = state['hosting']
    locked = dict.fromkeys(market['providers'], 0)
    for key, lease in market['leases'].items():
        job = state['expert_lifecycle']['jobs'].get(key)
        if not job or job.get('hosting') != lease['assignment_root']:
            raise ValueError('A provider lease lost its native inference request')
        if lease['fee_escrow'] < sum(row['fee'] for row in lease['providers'].values()):
            raise ValueError('Provider fees exceed the locked ceiling')
        for row in lease['providers'].values():
            locked[row['owner']] += row['bond']
        spent = lease['spent_claim_bond']
        integer(spent, 0, state['manifest']['params']['claim_bond'])
        locked[lease['providers']['0']['owner']] -= spent
    for owner, provider in market['providers'].items():
        if (type(provider['collateral']) is not int or type(provider['locked']) is not int
                or not 0 <= provider['locked'] <= provider['collateral']
                or provider['locked'] != locked[owner]):
            raise ValueError('Provider collateral differs from outstanding assignments')


def required(state, graph, question, maximum):
    from . import expert_lifecycle
    unit = expert_lifecycle.profile_for(state)['price_per_token']
    _, _, price = expert_lifecycle.inference_terms(graph, question, maximum, unit)
    # The committed executor coordinates the complete graph, including owners
    # whose paths this particular request does not select.
    return {str(rank) for rank in range(3 + len(graph['experts']))}, price


def snapshot(state, job_id):
    """One committed read; never combine a lease and job from different blocks."""
    job_id = root(job_id)
    return {'chain_id': state['chain_id'], 'height': state['height'], 'time_ns': state['time_ns'],
        'job': copy.deepcopy(state.get('expert_lifecycle', {}).get('jobs', {}).get(job_id)),
        'lease': copy.deepcopy(state.get('hosting', {}).get('leases', {}).get(job_id)),
        'result': copy.deepcopy(state.get('expert_lifecycle', {}).get('results', {}).get(job_id))}


def select(state, graph, ranks, offers, expires, ceiling):
    market = state['hosting']
    serving_graph.fields(offers, ranks, 'Reserve an offer for every participating owner')
    selected, bonds = {}, {}
    for rank, offer_id in sorted(offers.items(), key=lambda row: int(row[0])):
        offer = market['offers'].get(root(offer_id))
        if (not offer or offer['graph'] != identity(graph) or offer['rank'] != int(rank)
                or offer['expires'] < expires):
            raise ValueError('Offer changed its graph, rank or availability deadline')
        used = sum(row['offer'] == offer_id for lease in market['leases'].values()
                   for row in lease['providers'].values())
        if used >= offer['capacity']:
            raise ValueError('Provider offer has no unreserved capacity')
        provider = market['providers'][offer['owner']]
        bond = state['manifest']['hosting']['lease_bond']
        if rank == '0':
            bond += state['manifest']['params']['claim_bond']
        bonds[offer['owner']] = bonds.get(offer['owner'], 0) + bond
        selected[rank] = {'offer': offer_id, 'owner': offer['owner'], 'fee': offer['fee'], 'bond': bond,
                          'endpoint': provider['endpoint'], 'certificate': provider['certificate']}
    for owner, amount in bonds.items():
        provider = market['providers'][owner]
        if provider['collateral'] - provider['locked'] < amount:
            raise ValueError('Provider has insufficient unreserved collateral')
    # Key separation is a placement constraint, never an ownership guarantee.
    if {'0', '1', '2'} <= set(selected) and len({selected[r]['owner'] for r in ('0', '1', '2')}) == 1:
        raise ValueError('One provider key may not hold the complete backbone')
    if sum(row['fee'] for row in selected.values()) > ceiling:
        raise ValueError('Replacement exceeds the original provider-fee ceiling')
    return selected


def stage_limit(graph, request, unit):
    from . import answering
    if 'answering' in graph:
        quote = answering.quote(graph, request['max_tokens'], unit)
        return sum(quote['call_counts'][name] * bound for name, bound in quote['limits'].items())
    return sum(call['max_tokens'] for call in request['calls'])


def reserve_audit(state, job, budget_id, publisher):
    budget = state['auditing']['budgets'].get(root(budget_id))
    if not budget or budget['stage_limit'] < stage_limit(job['graph'], job['request'], job['unit_price']):
        raise ValueError('Prepay complete bounded inference verification before reserving providers')
    scope = {'kind': FORMAT, 'payer': job['payer'], 'graph': identity(job['graph']),
             'request_root': identity(job['request'])}
    if budget.get('scope') != scope or budget['sponsor'] != job['payer']:
        raise ValueError('Hosted audit funding must belong to this exact customer, graph and request')
    if budget['reservation'] is not None:
        raise ValueError('Hosted verification funding is already reserved')
    # The customer commits the workload before discovery reserves capacity.
    # Native atomic selection may choose another available coordinator; only
    # this exact payer/request can bind the previously unclaimed budget to it.
    budget['publisher'] = publisher
    auditing.lock(state, budget_id, publisher, list(job['workers'].values()), job['id'])


def assignment(state, job, selected, budget_id, epoch):
    profile = state['manifest']['hosting']
    if state['height'] + profile['prepare_blocks'] + profile['execution_blocks'] \
            + state['manifest']['params']['max_claim_blocks'] >= job['expires']:
        raise ValueError('The request cannot fit preparation, execution and complete audit')
    providers = state['hosting']['providers']
    for row in selected.values():
        providers[row['owner']]['locked'] += row['bond']
    binding = {'domain': FORMAT + '/assignment', 'chain_id': state['chain_id'], 'job_id': job['id'],
               'graph': identity(job['graph']), 'request': identity(job['request']),
               'epoch': epoch, 'providers': selected, 'audit_budget': budget_id}
    job['hosting'] = identity(binding)
    job['workers'] = {rank: row['owner'] for rank, row in selected.items()}
    return {'assignment_root': identity(binding), 'epoch': epoch, 'providers': selected,
            'audit_budget': budget_id, 'accepted': [], 'status': 'preparing',
            'spent_claim_bond': 0,
            'prepare_deadline': state['height'] + profile['prepare_blocks'], 'work_deadline': None}


def release_collateral(state, lease):
    market, profile = state['hosting'], state['manifest']['hosting']
    for rank, row in lease['providers'].items():
        provider = market['providers'][row['owner']]
        provider['locked'] -= row['bond'] - (lease['spent_claim_bond'] if rank == '0' else 0)
        provider['withdraw_after'] = max(provider['withdraw_after'], state['height'] + profile['cooldown_blocks'])


def release(state, job, *, accepted):
    if 'hosting' not in job:
        return
    market = state['hosting']
    lease = market['leases'].pop(job['id'])
    paid = 0
    if accepted:
        for row in lease['providers'].values():
            ledger.account(state, row['owner'])['balance'] += row['fee']
            paid += row['fee']
    ledger.account(state, job['payer'])['balance'] += lease['fee_escrow'] - paid
    release_collateral(state, lease)
    budget_id = lease['audit_budget']
    if budget_id in state['auditing']['budgets']:
        budget = state['auditing']['budgets'][budget_id]
        if budget['claim_id'] is not None or budget['reservation'] != job['id']:
            raise ValueError('Cannot refund an attached or unrelated audit obligation')
        auditing.finish(state, budget_id, reason='hosted request expired before a claim')
    market['history'].append({'job_id': job['id'], 'assignment_root': lease['assignment_root'],
        'epoch': lease['epoch'], 'accepted': accepted, 'provider_paid_atoms': paid,
        'provider_refunded_atoms': lease['fee_escrow'] - paid, 'height': state['height']})
    market['history'] = market['history'][-128:]


def claim_ready(state, job, body):
    if 'hosting' not in job:
        return
    lease = state['hosting']['leases'][job['id']]
    if (lease['status'] != 'ready' or state['height'] > lease['work_deadline']
            or body['audit_budget'] != lease['audit_budget']):
        raise ValueError('Every assigned provider and its prepaid audit must be ready')


def fund_claim_bond(state, owner, amount, values):
    """Transfer already reserved coordinator collateral to the native claim."""
    lease = state.get('hosting', {}).get('leases', {}).get(values.get('job_id'))
    if lease is None:
        return False
    if (values.get('kind') != 'expert_inference' or lease['status'] != 'ready'
            or lease['spent_claim_bond'] != 0 or owner != lease['providers']['0']['owner']
            or amount != state['manifest']['params']['claim_bond']):
        raise ValueError('Only this ready assignment may spend its reserved claim collateral')
    provider = state['hosting']['providers'][owner]
    provider['collateral'] -= amount
    provider['locked'] -= amount
    lease['spent_claim_bond'] = amount
    return True


def refund_claim_bond(state, claim):
    lease = state.get('hosting', {}).get('leases', {}).get(claim.get('job_id'))
    if lease is None:
        return False
    if (claim.get('kind') != 'expert_inference' or lease['spent_claim_bond'] != claim['bond']
            or claim['owner'] != lease['providers']['0']['owner']):
        raise ValueError('Claim collateral does not belong to this provider reservation')
    provider = state['hosting']['providers'][claim['owner']]
    provider['collateral'] += claim['bond']
    provider['locked'] += claim['bond']
    lease['spent_claim_bond'] = 0
    return True


def attach_audit(state, claim, budget_id):
    """Move this job's existing budget to its claim, without a second reservation."""
    key = claim.get('job_id')
    lease = state.get('hosting', {}).get('leases', {}).get(key)
    if lease is None:
        return False
    budget = state['auditing']['budgets'].get(budget_id)
    if (lease['status'] != 'ready' or lease['audit_budget'] != budget_id or not budget
            or budget['reservation'] != key or budget['claim_id'] is not None
            or budget['publisher'] != claim['owner']):
        raise ValueError('The claim changed its reserved provider audit')
    budget['reservation'] = claim['id']
    lease['status'] = 'claiming'
    return True


def failed_claim(state, job):
    if 'hosting' in job:
        lease = state['hosting']['leases'][job['id']]
        lease.update(status='failed', audit_budget=None)


def advance(state):
    if 'hosting' not in state:
        return
    offers = state['hosting']['offers']
    for key, offer in list(offers.items()):
        if state['height'] > offer['expires']:
            del offers[key]


def apply(state, owner, body, envelope):
    from . import expert_lifecycle
    if 'hosting' not in state:
        raise ValueError('Genesis does not enable provider hosting')
    market, profile = state['hosting'], state['manifest']['hosting']
    kind, height = body['kind'], state['height']
    if kind == 'fund_hosted_audit':
        graph = root(body['graph'])
        request_root = root(body['request_root'])
        if graph != state['serving_root']:
            raise ValueError('Fund the currently accepted graph')
        auditing.apply(state, owner, {'kind': 'fund_audit', 'auditors': [],
            **{key: body[key] for key in ('publisher', 'stage_limit', 'expires_in')}}, envelope)
        state['auditing']['budgets'][protocol.transaction_id(envelope)]['scope'] = {
            'kind': FORMAT, 'payer': owner, 'graph': graph, 'request_root': request_root}
    elif kind == 'register_provider':
        if owner in market['providers'] or len(market['providers']) >= profile['max_providers']:
            raise ValueError('Provider already registered or registry is full')
        amount = integer(body['collateral'], profile['lease_bond'], 2**60)
        address, certificate = endpoint(body['endpoint']), root(body['certificate'])
        auditing.debit(state, owner, amount)
        market['providers'][owner] = {'endpoint': address, 'certificate': certificate,
            'collateral': amount, 'locked': 0, 'withdraw_after': height + profile['cooldown_blocks']}
    elif kind in ('update_provider', 'withdraw_provider'):
        provider = market['providers'].get(owner)
        if provider is None:
            raise ValueError('Register a provider first')
        if kind == 'update_provider':
            amount = integer(body['deposit'], 0, 2**60)
            address, certificate = endpoint(body['endpoint']), root(body['certificate'])
            auditing.debit(state, owner, amount)
            if provider['collateral'] + amount > 2**60:
                raise ValueError('Provider collateral exceeds its bound')
            provider.update(endpoint=address, certificate=certificate, collateral=provider['collateral'] + amount)
        else:
            amount = integer(body['amount'], 1, provider['collateral'] - provider['locked'])
            if height < provider['withdraw_after']:
                raise ValueError('Provider collateral remains in its withdrawal cooldown')
            provider['collateral'] -= amount
            ledger.account(state, owner)['balance'] += amount
            if provider['collateral'] == 0:
                for key, offer in list(market['offers'].items()):
                    if offer['owner'] == owner:
                        del market['offers'][key]
                del market['providers'][owner]
    elif kind == 'offer_expert':
        graph = state['expert_lifecycle']['serving_graph']
        if owner not in market['providers'] or len(market['offers']) >= profile['max_offers']:
            raise ValueError('Require an admitted provider and available offer capacity')
        if root(body['graph']) != identity(graph):
            raise ValueError('Advertise an owner of the currently accepted graph')
        rank = integer(body['rank'], 0, 2 + len(graph['experts']))
        fee = integer(body['fee'], 0, 10**12)
        capacity = integer(body['capacity'], 1, 16)
        duration = integer(body['expires_in'], 1, 100000)
        market['offers'][protocol.transaction_id(envelope)] = {'owner': owner, 'graph': identity(graph),
            'rank': rank, 'fee': fee, 'capacity': capacity, 'expires': height + duration}
    elif kind == 'cancel_expert_offer':
        key = root(body['offer_id'])
        if market['offers'].get(key, {}).get('owner') != owner:
            raise ValueError('Only the provider may cancel its future offer')
        del market['offers'][key]
    elif kind == 'lease_expert':
        graph = state['expert_lifecycle']['serving_graph']
        maximum = integer(body['max_tokens'], 1, expert_lifecycle.profile_for(state)['max_tokens'])
        ranks, _ = required(state, graph, body['question'], maximum)
        ceiling = integer(body['max_provider_fee'], 0, 2**60)
        duration = integer(body['expires_in'], 1, 100000)
        if body['offers'] == 'discover':
            from .provider_quotes import choose
            _, selected = choose(state, graph, height + duration, ceiling)
        else:
            selected = select(state, graph, ranks, body['offers'], height + duration, ceiling)
        workers = {rank: row['owner'] for rank, row in selected.items()}
        fields = {key: body[key] for key in ('graph', 'question', 'max_tokens', 'max_price', 'expires_in')}
        expert_lifecycle.apply(state, owner, {'kind': 'infer_expert', **fields, 'workers': workers},
                               envelope, hosted=True)
        job = state['expert_lifecycle']['jobs'][protocol.transaction_id(envelope)]
        reserve_audit(state, job, body['audit_budget'], workers['0'])
        auditing.debit(state, owner, ceiling)
        lease = assignment(state, job, selected, body['audit_budget'], 0)
        market['leases'][job['id']] = {**lease, 'fee_escrow': ceiling}
    else:
        key = root(body['job_id'])
        lease, job = market['leases'].get(key), state['expert_lifecycle']['jobs'].get(key)
        if not lease or not job or job['claim_id'] is not None or height > job['expires']:
            raise ValueError('No available hosted request')
        if kind == 'accept_hosted_job':
            if (root(body['assignment_root']) != lease['assignment_root'] or lease['status'] != 'preparing'
                    or height > lease['prepare_deadline'] or owner in lease['accepted']
                    or owner not in {row['owner'] for row in lease['providers'].values()}):
                raise ValueError('Provider acceptance is stale, duplicated or unassigned')
            lease['accepted'] = sorted([*lease['accepted'], owner])
            if set(lease['accepted']) == {row['owner'] for row in lease['providers'].values()}:
                lease.update(status='ready', work_deadline=height + profile['execution_blocks'])
        elif kind == 'replace_hosted_job':
            expired = (lease['status'] == 'preparing' and height > lease['prepare_deadline']
                or lease['status'] == 'ready' and height > lease['work_deadline']
                or lease['status'] == 'failed')
            if not expired or lease['epoch'] + 1 >= profile['max_attempts']:
                raise ValueError('Replacement requires a closed attempt and remaining attempt budget')
            # Remove old capacity only inside the transaction's private state.
            # Any validation failure discards this complete transition.
            del market['leases'][key]
            release_collateral(state, lease)
            selected = select(state, job['graph'], set(job['workers']), body['offers'],
                              job['expires'], lease['fee_escrow'])
            old_budget, new_budget = lease['audit_budget'], root(body['audit_budget'])
            publisher = selected['0']['owner']
            if new_budget == old_budget:
                budget = state['auditing']['budgets'][old_budget]
                if budget['reservation'] != key or budget['claim_id'] is not None:
                    raise ValueError('Only an unattached job budget can follow its new coordinator')
                budget['publisher'] = publisher
            else:
                if old_budget is not None:
                    auditing.finish(state, old_budget, reason='hosted replacement changed its audit budget')
                reserve_audit(state, job, new_budget, publisher)
            updated = assignment(state, job, selected, new_budget, lease['epoch'] + 1)
            market['leases'][key] = {**updated, 'fee_escrow': lease['fee_escrow']}
        else:
            raise ValueError('Unknown provider transaction')
