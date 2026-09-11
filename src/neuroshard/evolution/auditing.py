"""Prepaid, collateralized complete-replay services for a new genesis profile.

Reports are accountable attestations, not proofs of independent computation.
Every selected auditor covers the entire graph. The sponsor chooses those keys;
this module makes no Sybil resistance or permissionless selection claim.
"""
from neuroshard.dataflow.store import canonical
from neuroshard.demo import protocol
from neuroshard.lab import state as ledger
from .objects import digest
from .schema import integer, root
from .verification import Metadata

FORMAT = 'neuroshard-funded-audit-v1'
PROFILE = {'format': FORMAT, 'price_per_stage': 100_000, 'auditor_bond': 5_000_000,
           'commit_blocks': 128, 'reveal_blocks': 32}
FIELDS = {
    'fund_audit': {'publisher', 'auditors', 'stage_limit', 'expires_in'},
    'accept_audit': {'budget_id'},
    'cancel_audit': {'budget_id'},
    'audit_commit': {'claim_id', 'commitment'},
    'audit_reveal': {'claim_id', 'salt', 'coverage_root'},
}
CLAIM_KINDS = ('grow', 'score', 'respond')


def initialize(state):
    profile = state['manifest']['auditing']
    if set(profile) != set(PROFILE) or profile['format'] != FORMAT:
        raise ValueError('Invalid funded audit profile')
    for name in ('price_per_stage', 'auditor_bond'):
        integer(profile[name], 1, 10**12)
    for name in ('commit_blocks', 'reveal_blocks'):
        integer(profile[name], 16, 10000)
    duration = profile['commit_blocks'] + profile['reveal_blocks']
    if duration + state['manifest']['params']['challenge_blocks'] >= state['manifest']['params']['max_claim_blocks']:
        raise ValueError('Claim lifetime cannot fit audit and challenge windows')
    state['auditing'] = {'budgets': {}, 'history': [], 'paid_atoms': 0, 'paid_services': 0}


def escrow(state):
    return sum(budget['funds'] + sum(a['bond'] for a in budget['auditors'].values())
               for budget in state.get('auditing', {}).get('budgets', {}).values())


def debit(state, owner, amount):
    account = ledger.account(state, owner)
    if account['balance'] < amount:
        raise ValueError('Insufficient audit funding or collateral')
    account['balance'] -= amount


def stages(claim):
    if claim.get('kind') == 'growth':
        return 1
    metadata = Metadata(claim['metadata'])
    if claim.get('kind') in ('score', 'inference'):
        from .forward import trace_roots
        return len(trace_roots(metadata, claim['record_root']))
    return len(metadata.json(claim['record_root'])['traces'])


def coverage(claim):
    # The record transitively commits every trace, dependency and output. This
    # digest specifies the purchased coverage; computing it is NOT an audit.
    return digest(canonical({'domain': FORMAT + '/coverage', 'record_root': claim['record_root'],
                             'kind': claim.get('kind', 'training'), 'stages': list(range(stages(claim)))}))


def commitment(chain_id, claim_id, owner, coverage_root, salt):
    root(salt)
    root(coverage_root)
    return digest(canonical({'domain': FORMAT + '/commit', 'chain_id': chain_id,
                             'claim_id': claim_id, 'auditor': owner,
                             'coverage_root': coverage_root, 'salt': salt}))


def lock(state, budget_id, publisher, workers, reservation):
    budget = state['auditing']['budgets'].get(root(budget_id))
    if (not budget or budget['publisher'] != publisher or budget['reservation'] is not None
            or state['height'] > budget['expires']):
        raise ValueError('No matching available audit budget')
    if any(not auditor['bond'] for auditor in budget['auditors'].values()):
        raise ValueError('Every selected auditor must accept before work starts')
    if set([publisher, *workers]) & set(budget['auditors']):
        raise ValueError('Auditor keys must differ from publisher and worker keys')
    budget['reservation'] = reservation


def attach(state, budget_id):
    claim = state['candidate']
    budget = state['auditing']['budgets'][budget_id]
    count = stages(claim)
    if count > budget['stage_limit']:
        raise ValueError('Audit budget does not cover every execution stage')
    profile = state['manifest']['auditing']
    commit_end = state['height'] + profile['commit_blocks']
    reveal_end = commit_end + profile['reveal_blocks']
    deadline = reveal_end + state['manifest']['params']['challenge_blocks']
    if deadline >= claim['expires']:
        raise ValueError('Execution deadline cannot fit complete audit coverage')
    budget.update(claim_id=claim['id'], stages=count)
    claim.update(audit_budget=budget_id, audit_commit_end=commit_end,
                 audit_reveal_end=reveal_end, deadline=deadline)


def complete(state, claim):
    if 'auditing' not in state:
        return True
    return all(a['revealed'] for a in state['auditing']['budgets'][claim['audit_budget']]['auditors'].values())


def minimum_deadline(state, claim):
    return claim.get('audit_reveal_end', 0) + state['manifest']['params']['challenge_blocks']


def finish(state, budget_id, *, accepted=False, claim=None, proven_fault=False, reason):
    service = state['auditing']
    budget = service['budgets'].pop(budget_id)
    profile = state['manifest']['auditing']
    paid = slashed = 0
    for owner, auditor in sorted(budget['auditors'].items()):
        # An unresolved allegation or publisher withholding is not evidence
        # that an auditor shirked. Timeliness penalties apply only to a clean,
        # uninterrupted reporting window. False attestations remain slashable
        # whenever the referee proves the execution wrong.
        missed = (claim is not None and reason == 'funded audit coverage deadline missed'
                  and not claim.get('audit_interrupted', False) and not auditor['revealed'])
        false_report = proven_fault and auditor['revealed']
        if missed or false_report:
            penalty = auditor['bond']
            slashed += penalty
            if false_report:
                accuser = claim['challenge']['owner']
                ledger.account(state, accuser)['balance'] += penalty // 2
                state['burned'] += penalty - penalty // 2
            else:
                state['burned'] += penalty
        else:
            ledger.account(state, owner)['balance'] += auditor['bond']
        if accepted:
            if not auditor['revealed']:
                raise ValueError('Cannot pay incomplete audit coverage')
            fee = budget['stages'] * profile['price_per_stage']
            paid += fee
            ledger.account(state, owner)['balance'] += fee
            service['paid_services'] += 1
    ledger.account(state, budget['sponsor'])['balance'] += budget['funds'] - paid
    service['paid_atoms'] += paid
    service['history'].append({'id': budget_id, 'claim_id': budget['claim_id'],
        'paid_atoms': paid, 'slashed_atoms': slashed, 'refunded_atoms': budget['funds'] - paid,
        'height': state['height'], 'reason': reason})
    service['history'] = service['history'][-128:]


def advance(state):
    if 'auditing' not in state:
        return
    for key, budget in sorted(list(state['auditing']['budgets'].items())):
        if budget['reservation'] is None and state['height'] > budget['expires']:
            finish(state, key, reason='unclaimed audit budget expired')


def apply(state, owner, body, envelope):
    if 'auditing' not in state:
        raise ValueError('Genesis does not enable funded auditing')
    service, profile = state['auditing'], state['manifest']['auditing']
    height, kind = state['height'], body['kind']
    if kind == 'fund_audit':
        if len(service['budgets']) >= 16:
            raise ValueError('Outstanding audit budget limit reached')
        publisher = ledger.public_key(body['publisher'])
        auditors = body['auditors']
        if (not isinstance(auditors, list) or not 1 <= len(auditors) <= 8
                or auditors != sorted(set(auditors)) or publisher in auditors):
            raise ValueError('Select sorted distinct auditor keys, separate from publisher')
        for key in auditors:
            ledger.public_key(key)
        count = integer(body['stage_limit'], 1, 4096)
        expiry = integer(body['expires_in'], 16, 100000)
        funds = count * len(auditors) * profile['price_per_stage']
        debit(state, owner, funds)
        key = protocol.transaction_id(envelope)
        service['budgets'][key] = {'sponsor': owner, 'publisher': publisher,
            'auditors': {a: {'bond': 0, 'commitment': None, 'revealed': False} for a in auditors},
            'stage_limit': count, 'stages': None, 'funds': funds, 'expires': height + expiry,
            'reservation': None, 'claim_id': None}
    elif kind in ('accept_audit', 'cancel_audit'):
        key = root(body['budget_id'])
        budget = service['budgets'].get(key)
        if not budget or budget['reservation'] is not None or height > budget['expires']:
            raise ValueError('No available audit offer')
        if kind == 'cancel_audit':
            if owner != budget['sponsor']:
                raise ValueError('Only the sponsor may cancel an unreserved offer')
            finish(state, key, reason='sponsor cancelled before reservation')
        else:
            auditor = budget['auditors'].get(owner)
            if not auditor or auditor['bond']:
                raise ValueError('Auditor is not selected or already accepted')
            debit(state, owner, profile['auditor_bond'])
            auditor['bond'] = profile['auditor_bond']
    else:
        claim = state['candidate']
        if not claim or claim['id'] != body['claim_id']:
            raise ValueError('No matching execution claim for audit report')
        budget = service['budgets'][claim['audit_budget']]
        auditor = budget['auditors'].get(owner)
        if not auditor:
            raise ValueError('Auditor is not selected for this obligation')
        if kind == 'audit_commit':
            if height > claim['audit_commit_end'] or auditor['commitment'] is not None:
                raise ValueError('Audit commitment is late or duplicated')
            auditor['commitment'] = root(body['commitment'])
            if all(a['commitment'] is not None for a in budget['auditors'].values()):
                # Once all commitments are final, no participant can adapt its
                # report to a public reveal. Reveal starts in the next block.
                claim['audit_commit_end'] = height
                claim['audit_reveal_end'] = height + profile['reveal_blocks']
                claim['deadline'] = claim['audit_reveal_end'] + state['manifest']['params']['challenge_blocks']
        elif kind == 'audit_reveal':
            if not claim['audit_commit_end'] < height <= claim['audit_reveal_end'] or auditor['revealed']:
                raise ValueError('Audit reveal is outside its window or duplicated')
            if body['coverage_root'] != coverage(claim):
                raise ValueError('Report does not attest the complete execution graph')
            expected = commitment(state['chain_id'], claim['id'], owner, body['coverage_root'], body['salt'])
            if auditor['commitment'] != expected:
                raise ValueError('Audit reveal differs from its commitment')
            auditor['revealed'] = True
            if all(a['revealed'] for a in budget['auditors'].values()):
                claim['audit_reveal_end'] = height
                claim['deadline'] = height + state['manifest']['params']['challenge_blocks']
        else:
            raise ValueError('Unknown audit transaction')
