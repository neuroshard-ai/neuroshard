"""Native admission bookkeeping; numerical replay is exercised separately.

Completed boundaries in these tests are explicit fixtures. They do not measure
new learning or replace the producer and independent executor integration tests.
"""
import copy
import json
import math

import pytest

from neuroshard.evolution import expert_admission as admission, expert_checkpoint, expert_work
from neuroshard.evolution import expert_lifecycle as life, serving_graph, settlement as state
from neuroshard.evolution.reference_data import identity
from test_expert_lifecycle import network, fund, finish, send
from test_serving_graph import FIXTURE, graphs
from test_settlement import blocks


def data_for(s, work, number, replay=None):
    prior = s.get('expert_lifecycle', {}).get('admission')
    sources, windows, documents = {}, [], []
    for role in ('train', 'evaluation'):
        source = {'repo': 'fixture/messages', 'revision': 'a'*40, 'split': role,
                  'license': 'Apache-2.0', 'role': 'train' if role == 'train' else 'heldout'}
        key = identity(source)
        sources[key] = source
        cursor = prior['cursors'].get(key, 0) if prior else 0
        windows.append({'source': key, 'start': cursor, 'end': cursor + 1})
        documents.append({'id': identity(['document', number, role]), 'source': key, 'row': cursor,
            'object': identity(['object', number, role]), 'tokens': identity(['tokens', number, role]), 'role': role})
    batches = [[documents[0]['id']]]
    if replay:
        documents.append({**replay, 'role': 'replay'})
        batches.append([replay['id']])
    return {'format': admission.DATA, 'previous': s['data_root'] if prior else None,
        'prepared': work['prepared'], 'policy': '9'*64, 'sources': sources, 'windows': windows,
        'documents': documents, 'batches': batches}


@pytest.fixture
def enabled(network):
    original, owners = network
    manifest = copy.deepcopy(original['manifest'])
    data = data_for(original, manifest['expert_work'], 0)
    manifest['expert_admission'] = {'format': admission.FORMAT, 'proposal_blocks': 64,
                                  'job_blocks': 2048, 'data_policy': '9'*64, 'initial_data': data}
    manifest['data_root'] = identity(data)
    validators = [{'owner': row['owner'], 'consensus_key': key, 'bond': row['amount'],
                   'liquid': original['accounts'][row['owner']]['balance']}
                  for key, row in original['validators'].items()]
    return state.genesis('expert-admission-test', validators, manifest), owners


def renamed(parent, value, job):
    result = copy.deepcopy(value)
    result['job'] = job
    complete = expert_checkpoint.reconstruct(parent, result)
    result.update(checkpoint=identity(complete), state_root=complete['state_root'])
    expert_checkpoint.unpack(parent, result)
    return result


def completed_boundary(s, owners, passed=True):
    s = copy.deepcopy(s)
    work = expert_work.prescription(s)
    end = renamed(work['parent'], json.loads(FIXTURE.read_bytes())['candidate']['experts']['protocol'],
                  work['checkpoint']['job'])
    claim_id = identity(['fixture-work', work['checkpoint']['job']])
    s['expert_work'].update(feature_claim=identity(['fixture-prefix', claim_id]))
    if work['format'] == expert_work.PROSPECTIVE:
        s['expert_work'].update(feature_root=identity(['bank', claim_id]),
                               batch_roots=[identity(['batch', claim_id, i]) for i in range(work['batch_count'])])
    expert_work.settle(s, {'kind': 'expert_training', 'id': claim_id,
        'work_ids': [identity(['fixture-step', claim_id, i]) for i in range(560)],
        'workers': [owners[0].public_key], 'output_checkpoint': end,
        'window': {'steps': [{'index': i} for i in range(560)]}})
    state.invariant(s)
    profile = life.profile_for(s)
    report = {'format': life.FORMAT + '/quality', 'policy_root': profile['quality']['policy_root'],
        'baseline_graph': identity(profile['serving_graph']), 'candidate_graph': identity(life.candidate_graph(s)),
        'prepared': profile['quality']['prepared'], 'passed': passed, 'results_root': identity(['quality', claim_id])}
    s, budget = fund(s, owners, profile['quality']['stages'])
    return finish(send(s, owners[0], 'quality_expert', report=report,
                       transcript_root=identity(['quality-transcript', claim_id]), audit_budget=budget), owners)


def proposed(s, number, replay=None):
    graph = copy.deepcopy(s['expert_lifecycle']['serving_graph'])
    name = 'cohort-' + str(number)
    job_id = identity(['next-job', number])
    initial = renamed(graph['parent'], json.loads(FIXTURE.read_bytes())['initial_expert'], job_id)
    template = copy.deepcopy(graph)
    template['descriptor'].update(format=serving_graph.EXTENSIBLE, previous_graph=identity(graph['descriptor']))
    template['descriptor']['experts'].append({'id': name, 'checkpoint': initial['checkpoint']})
    template['descriptor']['rules'].append({'id': name, 'needle': name, 'owner': 3 + len(graph['experts'])})
    template['descriptor']['total_parameters'] += sum(math.prod(spec['shape']) for spec in initial['tensors'].values())
    template['experts'][name] = initial
    count = 2 if replay else 1
    work = {'format': expert_work.PROSPECTIVE, 'parent': graph['parent'], 'checkpoint': initial,
        'prepared': identity(['prepared', number]), 'feature_stages': count * 3, 'batch_count': count,
        'schedule': list(range(count)) * (560 // count), 'numerical_profile': graph['numerical_profile']}
    profile = {'format': life.PROSPECTIVE, 'serving_graph': graph, 'candidate_template': template,
        'quality': {'policy_root': identity(['policy', number]), 'prepared': identity(['evaluation', number]), 'stages': 2},
        **{key: s['manifest']['expert_lifecycle'][key] for key in ('price_per_token', 'max_tokens')}}
    return {'work': work, 'lifecycle': profile, 'data': data_for(s, work, number, replay)}


def activate(s, owners, job):
    before = s['serving_root'], s['issued'], copy.deepcopy(s['paid_work']), copy.deepcopy(s['manifest'])
    s = send(s, owners[0], 'propose_expert_job', job=job)
    key = admission.bookkeeping(s)['proposal']['id']
    for owner in owners[:3]:
        s = send(s, owner, 'vote_expert_job', proposal_id=key, approve=True,
                 review_root=identity(['fixture-review', key, owner.public_key]))
    s = blocks(s, s['manifest']['params']['activation_blocks'])
    assert admission.bookkeeping(s)['active']['id'] == key
    assert (s['serving_root'], s['issued'], s['paid_work'], s['manifest']) == before
    return s


def test_repeated_growing_jobs_preserve_ledger_and_rejected_quality_serving(enabled):
    s, owners = enabled
    s = completed_boundary(s, owners)
    first_graph = s['serving_root']
    replay = admission.bookkeeping(s)['data']['documents'][0]
    for number, passed in ((1, True), (2, False), (3, True)):
        job = proposed(s, number, replay)
        s = activate(s, owners, job)
        # Recover from a canonically serialized local full-node state.
        recovered = json.loads(json.dumps(s, sort_keys=True))
        assert identity(recovered) == identity(s)
        assert expert_work.prescription(recovered) == job['work']
        before = recovered['serving_root']
        s = completed_boundary(recovered, owners, passed)
        assert (s['serving_root'] != before) is passed
        assert s['manifest'] == enabled[0]['manifest']
    assert s['serving_root'] != first_graph and len(s['expert_lifecycle']['serving_graph']['experts']) == 4
    assert len(admission.bookkeeping(s)['seen_jobs']) == 4 and s['issued'] == 2240 * state.PARAMS['reward_atoms']


def test_continued_job_can_only_name_the_currently_accepted_expert(enabled):
    s, owners = enabled
    s = completed_boundary(s, owners)
    job = proposed(s, 1)
    accepted = s['expert_lifecycle']['serving_graph']['experts']['protocol']
    job['work']['seed_expert'] = {'name': 'protocol', 'checkpoint': accepted}
    admission.validate_job(s, job)
    changed = copy.deepcopy(job)
    changed['work']['seed_expert']['name'] = 'unaccepted'
    with pytest.raises(ValueError, match='accepted serving expert'):
        admission.validate_job(s, changed)
    # Correctly reconstructed metadata for another job is still not the
    # accepted source, even if its tensor payloads happen to be identical.
    changed = copy.deepcopy(job)
    changed['work']['seed_expert']['checkpoint'] = renamed(job['work']['parent'], accepted, 'e'*64)
    with pytest.raises(ValueError, match='accepted serving expert'):
        admission.validate_job(s, changed)


def test_data_cannot_relabel_evaluation_replay_unused_or_rewind_cursors(enabled):
    s, owners = enabled
    s = completed_boundary(s, owners)
    pending = activate(s, owners, proposed(s, 1))
    untrained = admission.bookkeeping(pending)['data']['documents'][0]
    with pytest.raises(ValueError, match='accepted training windows'):
        admission.validate_job(pending, proposed(pending, 2, untrained))
    trained = admission.bookkeeping(s)['data']['documents'][0]
    evaluation = admission.bookkeeping(s)['data']['documents'][1]
    with pytest.raises(ValueError, match='accepted training windows'):
        admission.validate_job(s, proposed(s, 1, evaluation))
    job = proposed(s, 1, trained)
    job['data']['windows'][0]['start'] -= 1
    with pytest.raises(ValueError, match='consecutively'):
        admission.validate_job(s, job)
    job = proposed(s, 1)
    job['data']['documents'][0]['id'] = evaluation['id']
    with pytest.raises(ValueError, match='repeats history'):
        admission.validate_job(s, job)
    job = proposed(s, 1)
    job['data']['batches'][0] = [job['data']['documents'][1]['id']]
    with pytest.raises(ValueError, match='exactly the fresh'):
        admission.validate_job(s, job)


def test_rejected_admission_keeps_cursors_supply_and_serving_and_allows_retry(enabled):
    s, owners = enabled
    s = completed_boundary(s, owners)
    job = proposed(s, 1)
    before = copy.deepcopy(admission.bookkeeping(s)['cursors']), s['issued'], s['serving_root']
    pending = send(s, owners[0], 'propose_expert_job', job=job)
    key = admission.bookkeeping(pending)['proposal']['id']
    for owner in owners[:3]:
        pending = send(pending, owner, 'vote_expert_job', proposal_id=key, approve=False, review_root='1'*64)
    rejected = blocks(pending, pending['manifest']['expert_admission']['proposal_blocks'] + 1)
    assert admission.bookkeeping(rejected)['proposal'] is None
    assert (admission.bookkeeping(rejected)['cursors'], rejected['issued'], rejected['serving_root']) == before
    activated = activate(rejected, owners, job)
    assert activated['expert_work']['checkpoint'] == job['work']['checkpoint']


def test_job_expiry_prevents_further_work_without_changing_serving(enabled):
    s, owners = enabled
    s = completed_boundary(s, owners)
    s = activate(s, owners, proposed(s, 1))
    active = admission.bookkeeping(s)['active']
    # Isolate the expiry transition from unrelated block production here.
    expired = copy.deepcopy(s)
    expired['height'] = active['expires'] + 1
    admission.advance(expired)
    state.invariant(expired)
    assert expired['expert_lifecycle']['quality_closed']
    assert expired['serving_root'] == s['serving_root'] and expired['issued'] == s['issued']
    with pytest.raises(ValueError, match='closed or expired'):
        send(expired, owners[0], 'reserve_expert_inputs', workers=[owner.public_key for owner in owners[:3]], audit_budget='1'*64)
