"""Prepare successive actual LLM jobs from a frozen ordinary-learning feed.

The installed curator policy checks a bounded source-backed corpus. It is not
an automatic semantic-truth oracle for arbitrary web data. The native operator
still funds, reserves, audits and promotes each resulting job independently.
"""
import copy
import json
from pathlib import Path

from . import answering, expert_admission, expert_data, expert_preparation
from . import ordinary_cohorts, ordinary_quality
from .reference_data import identity, save

FORMAT = ordinary_cohorts.FORMAT+'/operation'


def plan_for(graph, profile, policy, recipe):
    return {'format': expert_data.FORMAT, 'parent': identity(graph['parent']),
        'split': graph['descriptor']['split'], 'parent_layout': graph['parent']['boundaries'],
        'expert_layout': graph['descriptor']['expert_layout'], 'training': copy.deepcopy(recipe),
        'max_length': policy['max_length'], 'microbatch': 4, 'runtime': copy.deepcopy(profile['runtime']),
        'threads': profile['threads'], 'parameter_limit': profile['parameter_limit'],
        'previous_graph': identity(graph['descriptor']), 'data_policy': identity(policy),
        'seed_expert': {'name': 'planner', 'checkpoint': copy.deepcopy(graph['experts']['planner'])},
        'objective': {'kl_strength': 0., 'margin_strength': 0., 'margin_min': .5, 'margin_max': 2.}}


def prior_quality(state, store):
    active = expert_admission.bookkeeping(state)['active']
    lifecycle = active['job']['lifecycle'] if active else state['manifest']['expert_lifecycle']
    return store.json(lifecycle['quality']['policy_root'])


class Preparation:
    """Bounded feed consumer with actual initialization supplied by its owner."""
    def __init__(self, home, freeze, store, tokenizer, feed, initialize):
        self.home, self.freeze, self.store = Path(home), copy.deepcopy(freeze), store
        self.tokenizer, self.feed, self.initialize = tokenizer, feed, initialize
        self.home.mkdir(parents=True, exist_ok=True)
        if freeze['format'] != FORMAT:
            raise ValueError('Require a complete prospectively frozen campaign prescription')
        self.expected = {}
        for name, spec in freeze['sources'].items():
            records = store.json(spec['records'])
            self.expected[identity(spec['source'])] = records
        self.evidence = store.json(freeze['source_evidence'])

    def reviewed(self, source, start, count):
        expected = self.expected[identity(source)][start:start+count]
        actual = list(self.feed(source, start, count))
        ordinary_cohorts.verify_feed_rows(self.evidence, expected, actual)
        return actual

    def prepare(self, state, entry):
        graph = state['expert_lifecycle']['serving_graph']
        if 'answering' not in graph or entry['name'] in graph['experts']:
            raise ValueError('The next isolated cohort must extend the accepted complete system')
        policy = self.store.json(self.freeze['data_policy'])
        profile = self.store.json(self.freeze['executor'])
        plan = plan_for(graph, profile, policy, entry['recipe'])
        history = expert_admission.bookkeeping(state)
        replay = []
        # Replay chooses complete originally trained windows, never unconsumed
        # rows with convenient labels. The chronological source order is fixed.
        if entry['replay_documents']:
            available = []
            for source_name in entry['replay_sources']:
                source = identity(self.freeze['sources'][source_name]['source'])
                available.extend(sorted((document for key, document in history['seen_documents'].items()
                    if key in history['trained_documents'] and document['source'] == source
                    and document['role'] == 'train'), key=lambda row: row['row']))
            replay = [document['id'] for document in available[:entry['replay_documents']]]
            if len(replay) != entry['replay_documents']:
                raise ValueError('Earlier accepted windows do not cover the prescribed replay')
        windows = [{'source': self.freeze['sources'][name]['source'], 'count': count}
                   for name, count in entry['windows'].items()]
        prepared = expert_preparation.prepare(state, plan, policy, self.store, self.tokenizer,
            self.reviewed, windows=windows, replay_ids=replay, batch_size=16)
        key = identity(prepared)
        destination = self.home/key
        destination.mkdir(exist_ok=True)
        save(destination/'preparation.json', prepared)
        save(destination/'plan.json', plan)
        inputs = self.store.json(prepared['prepared'])
        save(destination/'prepared.json', inputs)
        for role, spec in inputs['roles'].items():
            (destination/(role+'.jsonl')).write_bytes(self.store.get(spec['sha256']))
        # A resumed initialization is deterministic and verifies actual stored
        # weights and fresh Adam. It must not invent future tensor commitments.
        initial = self.initialize(graph['parent'], plan, inputs, destination)
        template = ordinary_cohorts.extend(graph, entry['name'], initial)
        template = ordinary_cohorts.bind_policy(template, self.store.json(entry['answering_policy']), self.store)
        quality = prior_quality(state, self.store)
        if quality['format'] != ordinary_quality.FORMAT:
            raise ValueError('Carry ordinary quality and every prior admitted evaluation forward')
        job, review = expert_preparation.seal(state, prepared, initial, template, quality,
            policy, self.store, self.tokenizer, self.reviewed)
        save(destination/'job.json', job)
        save(destination/'review.json', {**review, 'source_evidence': identity(self.evidence),
            'curation_scope': 'Exact independently source-backed frozen corpus; arbitrary web data is not approved.'})
        return job


def outcome_sequence(state, entries):
    """Select the next frozen entry using committed native quality outcomes."""
    quality = [row for row in state['expert_lifecycle']['history'] if row.get('kind') == 'quality']
    if len(quality) > len(entries):
        raise ValueError('Native history exceeds this complete campaign')
    for index, row in enumerate(quality):
        required = entries[index]['expected_promotion']
        if type(row['promoted']) is not bool or row['promoted'] is not required:
            return {'status': 'quality_stop', 'entry': index, 'expected_promotion': required,
                    'actual_promotion': row['promoted'], 'evidence': row}
    return {'status': 'complete' if len(quality) == len(entries) else 'next', 'entry': len(quality)}
