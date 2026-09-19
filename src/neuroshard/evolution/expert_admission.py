"""Repeated bounded expert jobs under native data and recipe admission.

Consensus checks immutable identities, cursors, role exclusion and consumed
replay. Admission votes attest to the separately configured data policy and
review of available source bytes. They do not prove semantic truth or safety.
Numerical work and quality still require their own funded execution quorums.
"""
import copy

from neuroshard.demo import protocol
from neuroshard.lab import state as ledger
from . import auditing, cohorts, expert_work, lifecycle, serving_graph
from .reference_data import identity
from .schema import integer, root

FORMAT = 'neuroshard-expert-admission-v1'
DATA = FORMAT + '/data'
FIELDS = {'propose_expert_job': {'job'}, 'vote_expert_job': {'proposal_id', 'approve', 'review_root'}}


def bookkeeping(state):
    return state.get('expert_lifecycle', {}).get('admission')


def initialize(state):
    config = state['manifest'].get('expert_admission')
    if config is None:
        return
    serving_graph.fields(config, {'format', 'proposal_blocks', 'job_blocks', 'data_policy', 'initial_data'},
                         'Invalid expert admission policy')
    if config['format'] != FORMAT:
        raise ValueError('Unsupported expert admission policy')
    params = state['manifest']['params']
    integer(config['proposal_blocks'], params['activation_blocks'] + 1, 100000)
    integer(config['job_blocks'], params['max_claim_blocks'] + 1, 1000000)
    root(config['data_policy'])
    work = state['manifest']['expert_work']
    state['expert_lifecycle']['admission'] = {
        'proposal': None, 'active': None, 'cursors': {}, 'seen_documents': {}, 'trained_documents': {},
        'seen_jobs': {work['checkpoint']['job']: 'genesis'}, 'data': copy.deepcopy(config['initial_data'])}
    validate_data(state, config['initial_data'], work, initial=True)
    if identity(config['initial_data']) != state['data_root']:
        raise ValueError('Genesis must commit its initial admitted data')
    consume_data(state, config['initial_data'])


def validate_data(state, data, work, *, initial=False):
    serving_graph.fields(data, {'format', 'previous', 'prepared', 'policy', 'sources', 'windows',
                                'documents', 'batches'}, 'Invalid immutable expert data')
    admission = bookkeeping(state)
    if (data['format'] != DATA or data['prepared'] != work['prepared']
            or data['policy'] != state['manifest']['expert_admission']['data_policy']
            or data['previous'] != (None if initial else state['data_root'])):
        raise ValueError('Data changed its parent, prepared computation or review policy')
    sources, windows = data['sources'], data['windows']
    if not isinstance(sources, dict) or not 1 <= len(sources) <= 16:
        raise ValueError('Pin bounded immutable data sources')
    for key, source in sources.items():
        if identity(cohorts.source(source)) != root(key) or source['role'] not in ('train', 'heldout'):
            raise ValueError('Pin distinct training and evaluation sources')
    if not isinstance(windows, list) or not 1 <= len(windows) <= len(sources):
        raise ValueError('Bound source cursor windows')
    ranges = {}
    for window in windows:
        serving_graph.fields(window, {'source', 'start', 'end'}, 'Invalid source cursor window')
        source = window['source']
        if source not in sources or source in ranges:
            raise ValueError('Source cursor is absent or repeated')
        start = integer(window['start'], 0, 2**53 - 1)
        end = integer(window['end'], start + 1, min(start + 8192, 2**53 - 1))
        if start != admission['cursors'].get(source, 0):
            raise ValueError('Source cursors must advance consecutively')
        ranges[source] = (start, end)
    if not isinstance(data['documents'], list) or not 2 <= len(data['documents']) <= 2048:
        raise ValueError('Bound the complete training and evaluation document inventory')
    found, positions, training, evaluation = {}, set(), set(), set()
    for document in data['documents']:
        serving_graph.fields(document, {'id', 'source', 'row', 'object', 'tokens', 'role'},
                             'Invalid document provenance and token commitment')
        for key in ('id', 'source', 'object', 'tokens'):
            root(document[key])
        key, source, role = document['id'], document['source'], document['role']
        position = (source, integer(document['row'], 0, 2**53 - 1))
        if key in found or position in positions or source not in sources or role not in ('train', 'replay', 'evaluation'):
            raise ValueError('Duplicate or unclassified data document')
        previous = admission['seen_documents'].get(key)
        if role == 'replay':
            if (key not in admission['trained_documents'] or previous is None
                    or previous != {**document, 'role': 'train'} or sources[source]['role'] != 'train'):
                raise ValueError('Replay only the exact documents used by accepted training windows')
        else:
            if (previous is not None or source not in ranges
                    or not ranges[source][0] <= document['row'] < ranges[source][1]
                    or sources[source]['role'] != ('train' if role == 'train' else 'heldout')):
                raise ValueError('Fresh data repeats history, crosses roles or leaves its source cursor')
        (evaluation if role == 'evaluation' else training).add(key)
        found[key] = document
        positions.add(position)
    if not evaluation or not any(row['role'] == 'train' for row in found.values()):
        raise ValueError('Each cohort needs fresh training and separate evaluation documents')
    batches = data['batches']
    count = work['batch_count'] if work['format'] == expert_work.PROSPECTIVE else len(work['batch_roots'])
    if (not isinstance(batches, list) or len(batches) != count
            or any(not isinstance(batch, list) or not 1 <= len(batch) <= 64 for batch in batches)):
        raise ValueError('Bind every prescribed feature batch to its training documents')
    flattened = [key for batch in batches for key in batch]
    if len(flattened) != len(training) or set(flattened) != training:
        raise ValueError('Feature production must cover exactly the fresh and replay training inventory')
    return data


def consume_data(state, data):
    admission = bookkeeping(state)
    for window in data['windows']:
        admission['cursors'][window['source']] = window['end']
    for document in data['documents']:
        if document['role'] != 'replay':
            admission['seen_documents'][document['id']] = copy.deepcopy(document)
    admission['data'] = copy.deepcopy(data)
    state['data_root'] = identity(data)


def validate_job(state, job):
    from . import expert_lifecycle
    serving_graph.fields(job, {'work', 'lifecycle', 'data'}, 'Invalid proposed expert job')
    work = expert_work.validate_profile(job['work'])
    life = state['expert_lifecycle']
    profile = job['lifecycle']
    serving_graph.fields(profile, expert_lifecycle.PROSPECTIVE_FIELDS, 'Require a prospective quality contract')
    if (work['format'] != expert_work.PROSPECTIVE or profile.get('format') != expert_lifecycle.PROSPECTIVE
            or work['checkpoint']['job'] in bookkeeping(state)['seen_jobs']
            or profile.get('serving_graph') != life['serving_graph']):
        raise ValueError('Propose an unused prospective job against the currently accepted graph')
    fixed = state['manifest']['expert_lifecycle']
    if any(profile.get(key) != fixed[key] for key in ('price_per_token', 'max_tokens')):
        raise ValueError('Training admission cannot change the inference price policy')
    probe = {'manifest': {'auditing': state['manifest']['auditing'],
                         'expert_work': work, 'expert_lifecycle': profile}, 'expert_work': {}}
    expert_lifecycle.initialize(probe)
    validate_data(state, job['data'], work)


def apply(state, owner, body, envelope):
    admission = bookkeeping(state)
    if admission is None:
        raise ValueError('Genesis does not enable repeated expert admission')
    config = state['manifest']['expert_admission']
    if body['kind'] == 'propose_expert_job':
        if (admission['proposal'] or state['assignment'] or state['candidate']
                or not state['expert_lifecycle']['quality_closed']):
            raise ValueError('Close the current expert job before proposing another')
        validate_job(state, body['job'])
        amount = state['manifest']['params']['claim_bond']
        auditing.debit(state, owner, amount)
        admission['proposal'] = {'id': protocol.transaction_id(envelope), 'owner': owner, 'bond': amount,
            'job': copy.deepcopy(body['job']), 'electorate': lifecycle.weights(state), 'votes': {}, 'reviews': {},
            'activate_after': state['height'] + state['manifest']['params']['activation_blocks'],
            'expires': state['height'] + config['proposal_blocks']}
    else:
        proposal = admission['proposal']
        if not proposal or proposal['id'] != body['proposal_id'] or state['height'] > proposal['expires']:
            raise ValueError('No matching live expert job proposal')
        if type(body['approve']) is not bool or owner not in proposal['electorate'] or owner in proposal['votes']:
            raise ValueError('Each native validator owner may cast one explicit admission vote')
        proposal['reviews'][owner] = root(body['review_root'])
        proposal['votes'][owner] = body['approve']


def advance(state):
    admission = bookkeeping(state)
    if admission is None:
        return
    life = state['expert_lifecycle']
    idle = state['assignment'] is None and state['candidate'] is None
    proposal = admission['proposal']
    if proposal:
        approved = (state['height'] >= proposal['activate_after']
                    and lifecycle.quorum(proposal['electorate'], proposal['votes'])
                    and lifecycle.quorum(lifecycle.weights(state), proposal['votes']))
        expired = state['height'] > proposal['expires']
        if expired or (approved and idle):
            accepted = approved and not expired
            if accepted:
                validate_job(state, proposal['job'])
                job = proposal['job']
                admission['active'] = {'id': proposal['id'], 'job': copy.deepcopy(job),
                    'expires': state['height'] + state['manifest']['expert_admission']['job_blocks']}
                admission['seen_jobs'][job['work']['checkpoint']['job']] = proposal['id']
                consume_data(state, job['data'])
                state['expert_work'] = {'checkpoint': copy.deepcopy(job['work']['checkpoint']),
                    'feature_claim': None, 'feature_root': None, 'batch_roots': None}
                state['model_root'] = job['work']['checkpoint']['state_root']
                life.update(quality_claim=None, quality_closed=False)
            ledger.account(state, proposal['owner'])['balance'] += proposal['bond']
            life['history'].append({'id': proposal['id'], 'kind': 'activation', 'accepted': accepted,
                'job': identity(proposal['job']), 'reviews': proposal['reviews'], 'height': state['height']})
            admission['proposal'] = None
    active = admission['active']
    if active and state['height'] > active['expires'] and idle and not life['quality_closed']:
        life['quality_closed'] = True
        life['history'].append({'id': active['id'], 'kind': 'job_expired', 'height': state['height']})


def live(state):
    admission = bookkeeping(state)
    active = admission and admission['active']
    if active and (state['height'] > active['expires'] or state['expert_lifecycle']['quality_closed']):
        raise ValueError('The admitted expert job has closed or expired')


def deadline(state, maximum):
    admission = bookkeeping(state)
    active = admission and admission['active']
    return min(maximum, active['expires']) if active else maximum


def trained(state, claim):
    admission = bookkeeping(state)
    if admission is None:
        return
    work = expert_work.prescription(state)
    for step in claim['window']['steps']:
        batch = work['schedule'][step['index']]
        for key in admission['data']['batches'][batch]:
            admission['trained_documents'][key] = claim['id']
