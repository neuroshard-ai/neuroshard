"""Atomic work admission against native validators' standing audit capacity.

An offer is a bounded, collateral-backed service authorization, not a verdict.
Admitting work consumes capacity, funds its complete replay and reserves its
actual job in the same native transition. Old unfunded offer queues are disabled
only in a genesis that explicitly enables this extension.
"""
import copy

from neuroshard.demo import protocol
from neuroshard.lab import state as ledger
from . import auditing
from .reference_data import identity
from .schema import integer, root

FORMAT = 'neuroshard-standing-audit-admission-v1'
PROFILE = {'format': FORMAT, 'verification': 'native-replay-quorum',
           'max_budgets': 16, 'max_obligation_blocks': 16384,
           'max_service_blocks': 100000, 'audit_rent_per_block': 1000,
           'provider_blocks': 16384, 'provider_rent_per_block': 100,
           'offer_rent_per_block': 100, 'provider_heartbeat_blocks': 256, 'provider_slots': 1}
FIELDS = {
    'offer_audit_service': {'purpose', 'scope', 'stage_limit', 'capacity', 'expires_in'},
    'close_audit_service': {'service_id'},
    'renew_audit_service': {'service_id', 'expires_in', 'valid_until'},
    'admit_work': {'work', 'stage_limit', 'max_verification', 'max_occupancy', 'valid_until'},
    'renew_provider': {'valid_until'},
    'heartbeat_provider': {'valid_until'},
    'recover_hosted_job': {'job_id', 'assignment_root', 'valid_until'},
}
PURPOSES = {'lease_expert': 'expert_inference', 'reserve_expert_inputs': 'expert_features',
            'reserve_expert': 'expert_training', 'quality_expert': 'expert_quality'}


def initialize(state):
    profile = state['manifest']['service_admission']
    if (set(profile) != set(PROFILE) or profile['format'] != FORMAT
            or profile['verification'] != 'native-replay-quorum'):
        raise ValueError('Invalid standing audit admission profile')
    if not auditing.native(state) or 'expert_lifecycle' not in state or 'hosting' not in state:
        raise ValueError('Standing admission requires native audited expert hosting')
    integer(profile['max_budgets'], 1, 16)
    for name in ('max_obligation_blocks', 'max_service_blocks', 'provider_blocks'):
        integer(profile[name], 16, 100000)
    integer(profile['provider_heartbeat_blocks'], 16, 4096)
    integer(profile['provider_slots'], 1, 16)
    if profile['max_service_blocks'] < profile['max_obligation_blocks']:
        raise ValueError('Audit offers must be able to cover a complete obligation')
    for name in ('audit_rent_per_block', 'provider_rent_per_block', 'offer_rent_per_block'):
        integer(profile[name], 1, 10**9)
    state['service_admission'] = {'services': {}}


def enabled(state):
    return 'service_admission' in state


def scope(state, purpose):
    from . import expert_work, expert_lifecycle
    if purpose == 'expert_inference':
        return root(state['serving_root'])
    if purpose in ('expert_features', 'expert_training', 'expert_quality'):
        # The prescription is stable across the windows of one cohort, but a
        # different admitted cohort requires a new explicit service offer.
        return identity({'work': expert_work.prescription(state),
                         'lifecycle': expert_lifecycle.profile_for(state)})
    raise ValueError('Unsupported standing audit purpose')


def escrow(state):
    return sum(row['free_bond'] for row in state.get('service_admission', {}).get('services', {}).values())


def invariant(state):
    if not enabled(state):
        return
    service = state['service_admission']['services']
    bond = state['manifest']['auditing']['auditor_bond']
    for key, row in service.items():
        used = sum(key in budget.get('capacity', {}).get('service_ids', {}).values()
                   for budget in state['auditing']['budgets'].values())
        if row['free_bond'] != (row['capacity'] - used)*bond or row['free_bond'] < 0:
            raise ValueError('Standing audit collateral differs from accepted obligations')
    for budget in state['auditing']['budgets'].values():
        terms = budget.get('capacity')
        if terms is None or budget['reservation'] is None:
            raise ValueError('Atomic admission cannot retain an unaccepted or unreserved budget')
        if (terms['rent'] != (terms['until']-terms['start']+1)*terms['rate']
                or terms['rate'] != state['manifest']['service_admission']['audit_rent_per_block']):
            raise ValueError('Audit occupancy escrow changed its declared bound')


def terms(state, purpose, stages, duration):
    """Read-only availability and maximum-price calculation, with no reservation."""
    if not enabled(state):
        raise ValueError('Genesis does not enable standing audit admission')
    profile = state['manifest']['service_admission']
    integer(stages, 1, 4096)
    integer(duration, 1, profile['max_obligation_blocks'])
    if len(state['auditing']['budgets']) >= profile['max_budgets']:
        raise ValueError('Complete audit capacity is occupied')
    weights = auditing.snapshot(state)
    expected = scope(state, purpose)
    bond = state['manifest']['auditing']['auditor_bond']
    selected = {}
    for key, offer in sorted(state['service_admission']['services'].items()):
        if (offer['owner'] in weights['owners'] and offer['owner'] not in selected
                and offer['purpose'] == purpose and offer['scope'] == expected
                and offer['stage_limit'] >= stages and offer['free_bond'] >= bond
                and offer['expires'] >= state['height'] + duration):
            selected[offer['owner']] = key
    if 3*sum(weights['owners'][owner] for owner in selected) <= 2*sum(weights['owners'].values()):
        raise ValueError('No complete native quorum offers available audit capacity')
    return {'purpose': purpose, 'scope': expected, 'services': selected, 'voting_snapshot': weights,
            'verification': 4*stages*state['manifest']['auditing']['price_per_stage'],
            'occupancy': (duration+1)*profile['audit_rent_per_block']}


def reserve(state, owner, publisher, key, purpose, stages, duration, max_verification, max_occupancy):
    quote = terms(state, purpose, stages, duration)
    if (quote['verification'] > integer(max_verification, 0, 2**60)
            or quote['occupancy'] > integer(max_occupancy, 0, 2**60)):
        raise ValueError('Complete audit admission exceeds its signed spending ceiling')
    auditing.debit(state, owner, quote['verification'] + quote['occupancy'])
    weights = quote['voting_snapshot']
    auditors = {a: {'bond': 0, 'commitment': None, 'revealed': False} for a in weights['owners']}
    bond = state['manifest']['auditing']['auditor_bond']
    for auditor, service_id in quote['services'].items():
        state['service_admission']['services'][service_id]['free_bond'] -= bond
        auditors[auditor]['bond'] = bond
    state['auditing']['budgets'][key] = {'sponsor': owner, 'publisher': publisher,
        'auditors': auditors, 'stage_limit': stages, 'stages': None,
        'funds': quote['verification'], 'expires': state['height'] + duration,
        'reservation': None, 'claim_id': None, 'voting_snapshot': weights,
        'snapshot_height': max(1, state['height']),
        'capacity': {'purpose': purpose, 'scope': quote['scope'], 'service_ids': quote['services'],
            'start': state['height'], 'until': state['height'] + duration,
            'rate': state['manifest']['service_admission']['audit_rent_per_block'], 'rent': quote['occupancy']}}


def return_bond(state, budget, owner, amount):
    terms = budget.get('capacity')
    if terms is None:
        return False
    service = state['service_admission']['services'].get(terms['service_ids'].get(owner))
    if service is not None and state['height'] <= service['expires']:
        service['free_bond'] += amount
    else:
        ledger.account(state, owner)['balance'] += amount
    return True


def slashed_bond(state, budget, owner, amount):
    if amount and 'capacity' in budget:
        service = state['service_admission']['services'].get(budget['capacity']['service_ids'].get(owner))
        if service is not None:
            service['capacity'] -= 1


def finish(state, budget):
    terms = budget.get('capacity')
    if terms is None:
        return {}
    # Include the admission block even for same-block release. The sponsor's
    # signed maximum also covers interruption; a dispute cannot extend rent.
    spent = min(terms['rent'], (state['height']-terms['start']+1)*terms['rate'])
    state['burned'] += spent
    refunded = terms['rent']-spent
    ledger.account(state, budget['sponsor'])['balance'] += refunded
    return {'occupancy_burned_atoms': spent, 'occupancy_refunded_atoms': refunded}


def advance(state):
    if not enabled(state):
        return
    active = auditing.snapshot(state)['owners']
    for key, row in list(state['service_admission']['services'].items()):
        if state['height'] > row['expires'] or row['owner'] not in active:
            ledger.account(state, row['owner'])['balance'] += row['free_bond']
            del state['service_admission']['services'][key]


def advertise(state, owner, body, envelope):
    services = state['service_admission']['services']
    if body['kind'] == 'close_audit_service':
        key = root(body['service_id'])
        row = services.get(key)
        if row is None or row['owner'] != owner:
            raise ValueError('Only the service owner may close future audit capacity')
        ledger.account(state, owner)['balance'] += row['free_bond']
        del services[key]
        return
    if owner not in auditing.snapshot(state)['owners']:
        raise ValueError('Only a current native voting owner can offer audit capacity')
    purpose = body['purpose']
    if body['scope'] != scope(state, purpose):
        raise ValueError('Bind the service to the current graph or cohort prescription')
    if any(row['owner'] == owner and row['purpose'] == purpose for row in services.values()):
        raise ValueError('Close the existing purpose offer before replacing its future capacity')
    capacity = integer(body['capacity'], 1, 16)
    stages = integer(body['stage_limit'], 1, 4096)
    duration = integer(body['expires_in'], 16, state['manifest']['service_admission']['max_service_blocks'])
    amount = capacity*state['manifest']['auditing']['auditor_bond']
    auditing.debit(state, owner, amount)
    services[protocol.transaction_id(envelope)] = {'owner': owner, 'purpose': purpose,
        'scope': root(body['scope']), 'capacity': capacity, 'free_bond': amount,
        'stage_limit': stages, 'expires': state['height'] + duration}


def apply(state, owner, body, envelope):
    from . import expert_work, expert_lifecycle, hosting
    if not enabled(state):
        raise ValueError('Genesis does not enable standing audit admission')
    if body['kind'] in ('offer_audit_service', 'close_audit_service'):
        return advertise(state, owner, body, envelope)
    integer(body['valid_until'], state['height'], state['height'] + 64)
    if body['kind'] == 'renew_audit_service':
        offer = state['service_admission']['services'].get(root(body['service_id']))
        if offer is None or offer['owner'] != owner:
            raise ValueError('Renew only your existing standing audit service')
        duration = integer(body['expires_in'], 16, state['manifest']['service_admission']['max_service_blocks'])
        if state['height'] + duration <= offer['expires']:
            raise ValueError('Audit renewal must extend its availability')
        offer['expires'] = state['height'] + duration
        return
    if body['kind'] == 'recover_hosted_job':
        lease = state['hosting']['leases'].get(root(body['job_id']))
        if lease is None or lease['assignment_root'] != root(body['assignment_root']):
            raise ValueError('Recover only the exact failed assignment epoch')
        return hosting.apply(state, owner, {'kind': 'replace_hosted_job', 'job_id': body['job_id'],
            'offers': 'discover', 'audit_budget': lease['audit_budget']}, envelope)
    if body['kind'] == 'renew_provider':
        return renew_provider(state, owner)
    if body['kind'] == 'heartbeat_provider':
        provider = state['hosting']['providers'].get(owner)
        if provider is None or state['height'] > provider['registration_expires']:
            raise ValueError('Provider heartbeat requires an unexpired paid registration')
        provider['last_seen'] = state['height']
        return
    work = copy.deepcopy(body['work'])
    if not isinstance(work, dict) or work.get('kind') not in PURPOSES:
        raise ValueError('Atomic admission requires one supported, bounded expert obligation')
    kind = work['kind']
    fields = {**expert_work.FIELDS, **expert_lifecycle.FIELDS, **hosting.FIELDS}[kind]
    if set(work) != (fields - {'audit_budget'}) | {'kind'}:
        raise ValueError('Invalid atomic work schema')
    purpose = PURPOSES[kind]
    duration = (integer(work['expires_in'], 1, 100000) if kind == 'lease_expert' else
                state['manifest']['params']['max_claim_blocks'] +
                (state['manifest']['params']['lease_blocks'] if kind.startswith('reserve_') else 0))
    key = protocol.transaction_id(envelope)
    reserve(state, owner, owner, key, purpose, body['stage_limit'], duration,
            body['max_verification'], body['max_occupancy'])
    budget = state['auditing']['budgets'][key]
    if kind == 'lease_expert':
        graph = state['expert_lifecycle']['serving_graph']
        request, _, _ = expert_lifecycle.inference_terms(graph, work['question'], work['max_tokens'],
            expert_lifecycle.profile_for(state)['price_per_token'])
        budget['scope'] = {'kind': hosting.FORMAT, 'payer': owner,
                           'graph': identity(graph), 'request_root': identity(request)}
    work['audit_budget'] = key
    if kind in hosting.FIELDS:
        hosting.apply(state, owner, work, envelope)
    elif kind in expert_work.FIELDS:
        expert_work.apply(state, owner, work, envelope)
    else:
        expert_lifecycle.apply(state, owner, work, envelope)
    if budget['reservation'] is None:
        raise ValueError('The accepted audit did not atomically reserve its exact work')


def renew_provider(state, owner):
    provider = state['hosting']['providers'].get(owner)
    if provider is None:
        raise ValueError('Register a provider before renewing its advertisement')
    profile = state['manifest']['service_admission']
    start = max(state['height'], provider.get('registration_expires', state['height']))
    if start + profile['provider_blocks'] > state['height'] + 2*profile['provider_blocks']:
        raise ValueError('Provider advertisement renewal exceeds its lifetime bound')
    price = profile['provider_blocks']*profile['provider_rent_per_block']
    auditing.debit(state, owner, price)
    state['burned'] += price
    provider['registration_expires'] = start + profile['provider_blocks']
