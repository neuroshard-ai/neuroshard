"""Real source review of prepared proposals, without inventing numerical work."""
import copy

import pytest

from neuroshard.evolution import expert_data, expert_preparation as preparation
from neuroshard.evolution.objects import Objects
from neuroshard.evolution.reference_data import identity
from neuroshard.evolution.sharded import graph_quality
from test_expert_admission import renamed
from test_expert_data import prepared
from test_expert_data import test_generic_source_data_drives_new_prefix_and_independently_replayed_training as execute_and_replay


def arguments(fixture):
    home, state, job, policy, store, tokenizer, upstream, plan, _ = fixture
    quality = store.json(job['lifecycle']['quality']['policy_root'])
    questions = expert_data.quality_rows(store, quality['roles']['test'])

    def reader(source, start, count):
        values = upstream(source, start, count)
        if source['role'] == 'heldout':
            return [{**row, **{key: questions[start+i][key] for key in ('stratum', 'topics', 'answers')}}
                    for i, row in enumerate(values)]
        return values

    windows = [{'source': value, 'count': 2} for value in job['data']['sources'].values()]
    args = (state, plan, policy, store, tokenizer, reader)
    return args, windows, quality


def test_prepared_bytes_seal_into_a_valid_native_proposal_and_survive_restart(prepared):
    args, windows, quality = arguments(prepared)
    home, state, original, policy, store, tokenizer, _, plan, _ = prepared
    before = copy.deepcopy(state)
    bundle = preparation.prepare(*args, windows=windows, batch_size=1)
    store.put_json(bundle)
    restored = Objects(store.root)
    assert restored.json(identity(bundle)) == bundle
    assert preparation.prepare(*args, windows=windows, batch_size=1) == bundle
    inputs = restored.json(bundle['prepared'])
    initial = renamed(original['work']['parent'], original['work']['checkpoint'],
                      expert_data.job_identity(plan, inputs))
    job, report = preparation.seal(state, bundle, initial, original['lifecycle']['candidate_template'],
                                  quality, policy, restored, tokenizer, args[-1])
    assert report['mechanical_checks_passed'] and report['upstream_documents'] == 4
    assert job['data']['batches'] == bundle['data']['batches']
    assert job['work']['checkpoint']['job'] == initial['job']
    assert state == before  # No cursor mutation, issuance or serving promotion.
    def changed_upstream(source, start, count):
        values = copy.deepcopy(args[-1](source, start, count))
        values[0]['messages'][0]['content'] = 'Source was replaced after preparation'
        return values
    with pytest.raises(ValueError, match='pinned upstream row'):
        preparation.seal(state, bundle, initial, original['lifecycle']['candidate_template'],
                         quality, policy, restored, tokenizer, changed_upstream)
    # A new block alone is harmless; accepted admission changes are not.
    state['height'] = 100
    assert preparation.snapshot(state) == bundle['snapshot']
    state['data_root'] = 'c'*64
    with pytest.raises(ValueError, match='Admission changed'):
        preparation.seal(state, bundle, initial, original['lifecycle']['candidate_template'],
                         quality, policy, restored, tokenizer, args[-1])


def test_selection_rejects_source_substitution_omission_duplicates_and_unused_replay(prepared):
    args, windows, _ = arguments(prepared)
    state, plan, policy, store, tokenizer, reader = args
    with pytest.raises(ValueError, match='outside policy'):
        bad = copy.deepcopy(windows)
        bad[0]['source']['repo'] = 'unapproved/source'
        preparation.prepare(*args, windows=bad)
    with pytest.raises(ValueError, match='omitted selected rows'):
        preparation.prepare(*args[:-1], lambda *_: [], windows=windows)
    with pytest.raises(ValueError, match='selected twice'):
        preparation.prepare(*args, windows=windows + [windows[0]])
    with pytest.raises(ValueError, match='accepted training windows'):
        preparation.prepare(*args, windows=windows, replay_ids=['f'*64])
    document = prepared[2]['data']['documents'][0]
    state['expert_lifecycle']['admission']['seen_documents'][document['id']] = document
    with pytest.raises(ValueError, match='repeats admitted history'):
        preparation.prepare(*args, windows=windows)


def test_prepared_proposal_drives_actual_sharded_updates_and_fresh_replay(prepared):
    args, windows, quality = arguments(prepared)
    home, state, original, policy, store, tokenizer, _, plan, _ = prepared
    bundle = preparation.prepare(*args, windows=windows, batch_size=2)
    inputs = store.json(bundle['prepared'])
    initial = renamed(original['work']['parent'], original['work']['checkpoint'],
                      expert_data.job_identity(plan, inputs))
    job, _ = preparation.seal(state, bundle, initial, original['lifecycle']['candidate_template'],
                             quality, policy, store, tokenizer, args[-1])
    for role in ('train', 'test'):
        (home/'general-inputs'/(role+'.jsonl')).write_bytes(store.get(inputs['roles'][role]['sha256']))
    assert inputs['schedule'] == [0, 0] and inputs['batches'] == [[0, 1]]
    # The shared integration check runs real prefix production, two updates,
    # graph materialization and a fresh Python process replaying the window.
    execute_and_replay((home, state, job, policy, store, tokenizer, args[-1], plan, inputs))


def test_replay_keeps_original_rows_and_requires_actual_accepted_training(prepared):
    args, windows, _ = arguments(prepared)
    state, plan, policy, store, tokenizer, reader = args
    original = prepared[2]['data']['documents'][0]
    admission = state['expert_lifecycle']['admission']
    admission['seen_documents'][original['id']] = original
    admission['trained_documents'][original['id']] = 'd'*64
    source = prepared[2]['data']['sources'][original['source']]
    store.put_json(source)
    admission['cursors'][original['source']] = 2

    def fresh(source, start, count):
        if source['role'] == 'train':
            assert start == 2
            return [{'messages': [{'role': 'user', 'content': 'word21 word22'},
                                  {'role': 'assistant', 'content': 'word23'}]}]
        return reader(source, start, count)

    windows[0]['count'] = 1
    bundle = preparation.prepare(*args[:-1], fresh, windows=windows,
                                 replay_ids=[original['id']], batch_size=2)
    inputs = store.json(bundle['prepared'])
    rows = expert_data.records(inputs, 'train', store.get, 32, 32, tokenizer)
    assert [row['distill'] for row in rows] == [False, True]
    assert bundle['data']['batches'] == [[row['id'] for row in rows]]
    replay = next(doc for doc in bundle['data']['documents'] if doc['role'] == 'replay')
    assert replay == {**original, 'role': 'replay'}
    assert next(window for window in bundle['data']['windows']
                if window['source'] == original['source']) == {'source': original['source'], 'start': 2, 'end': 3}
    admission['trained_documents'].clear()
    with pytest.raises(ValueError, match='accepted training windows'):
        preparation.prepare(*args[:-1], fresh, windows=windows, replay_ids=[original['id']])


def test_retention_rolls_the_last_admitted_test_forward_with_unchanged_rules(prepared):
    _, state, job, _, store, _, _, _, _ = prepared
    quality = store.json(job['lifecycle']['quality']['policy_root'])
    questions = expert_data.quality_rows(store, quality['roles']['test'])
    # Independent fixed anchor, unlike this cohort's currently held-out questions.
    anchor = copy.deepcopy(questions[0])
    anchor['messages'][0]['content'] = 'Previously accepted word20?'
    anchor['id'] = expert_data.document_identity(anchor['messages'])
    knowledge = 'retained-test-knowledge'
    quality.update(format=graph_quality.CONTINUAL, retention_gates={'max_lost_correct': 0},
                   retention_anchors={key: value for key, value in quality['roles'].items() if key != 'test'})
    quality['retention_anchors'][knowledge] = preparation.record_set(store, 'anchors', [anchor])
    quality['roles'][knowledge] = quality['retention_anchors'][knowledge]
    for document in job['data']['documents']:
        if document['role'] == 'evaluation':
            state['expert_lifecycle']['admission']['seen_documents'][document['id']] = document
    before = copy.deepcopy(quality)
    next_quality = preparation.retain_history(state, quality, store)
    assert quality == before
    assert graph_quality.admission_rule(next_quality) == graph_quality.admission_rule(before)
    retained = expert_data.quality_rows(store, next_quality['roles'][knowledge])
    assert {row['id'] for row in retained} == {anchor['id'], *(row['id'] for row in questions)}
    # Omitting an admitted question cannot shrink the next retention obligation.
    quality['roles']['test'] = preparation.record_set(store, 'missing', questions[:1])
    with pytest.raises(ValueError, match='omits admitted evaluation history'):
        preparation.retain_history(state, quality, store)
