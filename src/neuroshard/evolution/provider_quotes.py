"""Bounded discovery and complete first-attempt prices for native shard hosting."""
import copy
from . import expert_lifecycle, hosting
from .reference_data import identity
from .schema import integer

FORMAT = 'neuroshard-hosted-inference-quote-v1'
ATOMIC_FORMAT = 'neuroshard-hosted-inference-quote-v2'


def replacement(state, job_id):
    """Check a timed replacement without paying a fee or changing reservations."""
    if 'service_admission' not in state:
        raise ValueError('Automatic recovery requires standing admission')
    current = copy.deepcopy(state)
    lease = current['hosting']['leases'].get(job_id)
    job = current['expert_lifecycle']['jobs'].get(job_id)
    if (not lease or not job or job['claim_id'] is not None or state['height'] > job['expires']
            or lease['epoch'] + 1 >= state['manifest']['hosting']['max_attempts']):
        raise ValueError('No remaining preclaim recovery attempt')
    deadline = lease.get('prepare_deadline') if lease['status'] == 'preparing' else lease.get('work_deadline')
    if lease['status'] not in ('preparing', 'ready') or state['height'] <= deadline:
        raise ValueError('The current assignment has not timed out')
    del current['hosting']['leases'][job_id]
    hosting.release_collateral(current, lease)
    offers, _ = choose(current, job['graph'], job['expires'], lease['fee_escrow'])
    return {'job_id': job_id, 'assignment_root': lease['assignment_root'], 'offers': offers,
            'height': state['height'], 'valid_until': min(job['expires'], state['height'] + 64)}


def choose(state, graph, expires, ceiling, publisher=None):
    """Find a feasible assignment, preferring cheaper advertised offers.

    An augmenting matching handles shared collateral across a provider's ranks.
    The returned price is explicit; this is not a global price-optimality claim.
    Native reservation rechecks every offer atomically before any work starts.
    """
    market, profile = state['hosting'], state['manifest']['hosting']
    graph_root = identity(graph)
    total = 3 + len(graph['experts'])
    used = {}
    for lease in market['leases'].values():
        for row in lease['providers'].values():
            used[row['offer']] = used.get(row['offer'], 0) + 1
    rows = {rank: {} for rank in range(total)}
    for key, offer in sorted(market['offers'].items(), key=lambda item: (item[1]['fee'], item[0])):
        if (offer['graph'] != graph_root or offer['expires'] < expires
                or used.get(key, 0) >= offer['capacity'] or offer['rank'] not in rows
                or market['providers'][offer['owner']].get('registration_expires', expires) < expires
                or not hosting.live_provider(state, market['providers'][offer['owner']])):
            continue
        rows[offer['rank']].setdefault(offer['owner'], {'id': key, **offer})
    for coordinator, offer in rows[0].items():
        if publisher is not None and coordinator != publisher:
            continue
        available = {owner: value['collateral'] - value['locked'] for owner, value in market['providers'].items()}
        available[coordinator] -= profile['lease_bond'] + state['manifest']['params']['claim_bond']
        if available[coordinator] < 0:
            continue
        capacity = {owner: amount // profile['lease_bond'] for owner, amount in available.items()}
        if 'service_admission' in state:
            slots = state['manifest']['service_admission']['provider_slots']
            for owner in capacity:
                occupied = sum(any(row['owner'] == owner for row in lease['providers'].values())
                               for lease in market['leases'].values())
                capacity[owner] = min(capacity[owner], int(occupied < slots))
            if not capacity[coordinator]:
                continue
            capacity[coordinator] = 0
        # At least one other backbone partition must use a different key.
        # Key separation is a placement constraint, not administrator diversity.
        for separate in (1, 2):
            edges = {rank: [owner for owner in offers if capacity[owner] > 0
                           and not (rank == separate and owner == coordinator)]
                     for rank, offers in rows.items() if rank != 0}
            assigned = {owner: [] for owner in market['providers']}
            choices = {}

            def assign(rank, seen):
                for owner in edges[rank]:
                    if owner in seen:
                        continue
                    seen.add(owner)
                    if len(assigned[owner]) < capacity[owner]:
                        assigned[owner].append(rank)
                        choices[rank] = owner
                        return True
                    for previous in list(assigned[owner]):
                        if assign(previous, seen):
                            assigned[owner].remove(previous)
                            assigned[owner].append(rank)
                            choices[rank] = owner
                            return True
                return False

            if not all(assign(rank, set()) for rank in sorted(edges, key=lambda value: len(edges[value]))):
                continue
            offers = {'0': offer['id'], **{str(rank): rows[rank][owner]['id'] for rank, owner in choices.items()}}
            try:
                selected = hosting.select(state, graph, set(offers), offers, expires, ceiling)
            except ValueError:
                continue
            return offers, selected
    raise ValueError('No complete provider assignment fits the available collateral, capacity and price limit')


def quote(state, question, maximum, *, provider_ceiling=2**60, publisher=None):
    if 'hosting' not in state:
        raise ValueError('This genesis does not enable public shard hosting')
    profile = state['manifest']['hosting']
    service = expert_lifecycle.profile_for(state)
    integer(maximum, 1, service['max_tokens'])
    integer(provider_ceiling, 0, 2**60)
    graph = state['expert_lifecycle']['serving_graph']
    request, _, execution = expert_lifecycle.inference_terms(graph, question, maximum, service['price_per_token'])
    duration = ((profile['prepare_blocks'] + profile['execution_blocks'])*profile['max_attempts']
                + state['manifest']['params']['max_claim_blocks'] + 64)
    valid_until = state['height'] + 64
    offers, selected = choose(state, graph, valid_until + duration, provider_ceiling, publisher)
    provider_fees = sum(row['fee'] for row in selected.values())
    stages = hosting.stage_limit(graph, request, service['price_per_token'])
    verification = 4 * stages * state['manifest']['auditing']['price_per_stage']
    # Fund + reserve + at most one replacement transaction per remaining
    # pre-claim attempt. An adjudicated rejection needs separate fresh funding.
    transaction_fees = (profile['max_attempts'] + 1)*state['manifest']['params']['fee']
    result = {'format': FORMAT, 'chain_id': state['chain_id'], 'height': state['height'],
        'valid_until': valid_until, 'graph': identity(graph), 'tokenizer': graph['tokenizer']['root'],
        'request_root': identity(request), 'max_tokens': maximum, 'offers': offers,
        'publisher': selected['0']['owner'], 'expires_in': duration, 'stage_limit': stages,
        'execution_atoms': execution, 'provider_atoms': provider_fees, 'verification_atoms': verification,
        'transaction_fee_allowance_atoms': transaction_fees,
        'maximum_debit_atoms': execution + provider_fees + verification + transaction_fees,
        'retry_policy': 'Unclaimed attempts may reuse the reserved audit budget; rejected claims require new funding',
        'visibility': 'Conversation, neural-call tokens and final responses are public; providers and auditors process them'}
    if 'service_admission' in state:
        from . import service_admission
        # Check advertised audit availability through the quote's whole validity
        # window. The eventual admission still reserves the complete job atomically.
        service_admission.terms(state, 'expert_inference', stages, duration + 64)
        occupancy = (duration + 1)*state['manifest']['service_admission']['audit_rent_per_block']
        transaction_fees = profile['max_attempts']*state['manifest']['params']['fee']
        result.update(format=ATOMIC_FORMAT, admission=service_admission.FORMAT,
            occupancy_atoms=occupancy, transaction_fee_allowance_atoms=transaction_fees,
            maximum_debit_atoms=execution + provider_fees + verification + occupancy + transaction_fees)
    return result
