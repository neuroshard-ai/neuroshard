"""Frozen source rows initialize and seal a real isolated numerical job."""
import copy
import json

import pytest

from neuroshard.dataflow.store import LocalStore
from neuroshard.evolution import answering, expert_data, expert_preparation, expert_router, expert_source
from neuroshard.evolution import ordinary_cohorts, ordinary_operation, expert_work
from neuroshard.evolution.reference_data import identity
from neuroshard.evolution.sharded import expert_execution, graph_quality, learned_graph, planned_graph, prefix_execution
from neuroshard.evolution.sharded.router_features import EmbeddingFeatures
from neuroshard.evolution.sharded.portable import tensor_path
from test_expert_data import prepared
from test_graph_execution import SOURCE
from test_ordinary_quality import question


def test_ordinary_feed_seals_actual_isolated_initialization_and_rejects_poison_and_unused_replay(prepared):
    home, state, old_job, policy, store, tokenizer, _, _, _ = prepared
    graph = copy.deepcopy(old_job['lifecycle']['serving_graph'])
    seed = json.loads((home/'graph.json').read_bytes())['experts']['astronomy']
    graph = ordinary_cohorts.extend(graph, 'planner', seed)
    profile = json.loads((home/'profile.json').read_bytes())
    embedding = graph['interpreter_assets']['partitions']['0']['tensors']['model.embed_tokens.weight']['sha256']
    features = EmbeddingFeatures(tensor_path(home/'interpreter', embedding), embedding, tokenizer, policy['tokenizer'])

    def bind(model_graph):
        observations = [{'id': identity([name, i]), 'route': name, 'features': features('word'+str(i+3))}
                        for i, name in enumerate(['parent', *model_graph['experts']])]
        router = expert_router.fit(observations, embedding_root=features.root, tokenizer_root=policy['tokenizer'],
                                   prototypes_per_route=1)
        learned = learned_graph.configuration(model_graph, router, features.profile, SOURCE)
        config = planned_graph.configuration(model_graph, learned,
            {'instruction': 'word2', 'examples': [], 'max_tokens': 4}, SOURCE)
        return answering.attach(model_graph, config, store)

    graph = bind(graph)
    topology = bind(ordinary_cohorts.extend(graph, 'admission', seed))
    quality = store.json(old_job['lifecycle']['quality']['policy_root'])
    quality.update(format=graph_quality.ORDINARY)
    quality['generation']['retained_conversation'] = 4
    quality['retention_gates'] = {'max_lost_correct': 0,
        'minimum_accuracy': {role: .5 for role in graph_quality.ROLES[1:]}}
    for number, role in enumerate(graph_quality.ROLES[1:]):
        quality['roles'][role] = expert_preparation.record_set(store, role,
            [question('Separate retained word'+str(number+20)+'?', ['word30'])])
    quality['retention_anchors'] = {role: quality['roles'][role] for role in graph_quality.ROLES[1:]}
    policy['quality_rule'] = identity(graph_quality.admission_rule(quality))
    state['manifest']['expert_admission']['data_policy'] = store.put_json(policy)
    state['manifest']['expert_lifecycle'].update(quality={'policy_root': store.put_json(quality)}, max_tokens=4)
    state['expert_lifecycle']['serving_graph'] = graph
    state['expert_lifecycle']['admission']['active'] = None
    sources = list(old_job['data']['sources'].values())
    values = {'train': [{'messages': [{'role': 'user', 'content': text},
        {'role': 'assistant', 'content': 'word11'}]} for text in ['New word7 word8?', 'New word9 word8?']],
        'test': [ordinary_cohorts.scored([{'role': 'user', 'content': 'Unknown word7?'}], ['one'], ['word11']),
                 ordinary_cohorts.scored([{'role': 'user', 'content': 'Unknown word7 and word8?'}],
                                        ['one', 'two'], ['word11', 'word12'])]}
    transport = LocalStore(home/'campaign-feed')
    windows, frozen_sources = [], {}
    for source in sources:
        name = source['split']
        windows.append(expert_source.publish_window(transport, source, 0, values[name]))
        frozen_sources[name] = {'source': source, 'records': store.put_json(values[name])}
    head = expert_source.append(transport, None, windows)
    feed = expert_source.Feed(transport, head)
    freeze = {'format': ordinary_operation.FORMAT, 'sources': frozen_sources,
        'source_evidence': store.put_json({'facts': {'fixture': 'Explicit numerical fixture, not real source claims.'}}),
        'data_policy': store.put_json(policy), 'executor': store.put_json(profile)}
    calls = []

    def initialize(parent, plan, prepared, directory):
        calls.append(identity(prepared))
        return expert_execution.initialize(parent, plan, prepared, objects=home/'objects',
                                            checkpoint_store=directory/'checkpoints')

    operator = ordinary_operation.Preparation(home/'campaign-prepare', freeze, store, tokenizer, feed, initialize)
    entry = {'name': 'admission', 'windows': {'train': 2, 'test': 2},
        'recipe': seed['recipe'], 'replay_documents': 0, 'replay_sources': [],
        'answering_policy': topology['answering']['policy_root']}
    before = copy.deepcopy(state)
    job = operator.prepare(state, entry)
    assert state == before and len(calls) == 1
    assert job['work']['checkpoint']['step'] == 0
    assert job['work']['seed_expert']['checkpoint'] == seed
    assert all(job['lifecycle']['candidate_template']['experts'][name] == value for name, value in graph['experts'].items())
    assert job['lifecycle']['candidate_template']['answering'] == topology['answering']
    assert job['lifecycle']['quality']['stages'] == 5
    assert operator.prepare(state, entry) == job
    # Exercise the newly introduced warm seed through actual owned prefix
    # work, updates, persistence and a fresh execution before allocating GPUs.
    inputs = store.json(job['work']['prepared'])
    new_plan = store.json(inputs['plan'])
    numerical = home/'warm-native-inputs'
    numerical.mkdir()
    for role, spec in inputs['roles'].items():
        (numerical/(role+'.jsonl')).write_bytes(store.get(spec['sha256']))
    reports, incoming = [], None
    for rank in range(3):
        owner = home/('warm-prefix-'+str(rank))
        report = prefix_execution.owned_stage(rank, job['work'], new_plan, inputs,
            inputs=numerical, objects=home/'objects', home=owner, incoming=incoming, max_seconds=60)
        reports.append(report)
        incoming = (owner/'features', report)
    bank_home = home/'warm-prefix-2/features'
    bank = json.loads((bank_home/'index.json').read_bytes())
    rows = expert_execution.training_records(new_plan, inputs, numerical, graph['parent'])
    production = prefix_execution.owned_production(job['work'], new_plan, inputs, rows, reports, bank)
    resolved = expert_work.resolve_prefix(job['work'], production['feature_root'], production['batch_roots'])
    kwargs = {'inputs': numerical, 'objects': home/'objects', 'bank_home': bank_home, 'max_seconds': 60}
    count = min(2, len(resolved['schedule']))
    first = expert_execution.produce_training(job['work']['checkpoint'], count, resolved, new_plan, inputs,
        checkpoint_store=home/'warm-producer', **kwargs)
    replayed = expert_execution.produce_training(job['work']['checkpoint'], count, resolved, new_plan, inputs,
        checkpoint_store=home/'warm-auditor', **kwargs)
    assert first == replayed
    if count < len(resolved['schedule']):
        next_count = min(2, len(resolved['schedule'])-count)
        before = first['window']['output']
        continued = expert_execution.produce_training(before, next_count, resolved, new_plan, inputs,
            checkpoint_store=home/'warm-producer', **kwargs)
        checked = expert_execution.produce_training(before, next_count, resolved, new_plan, inputs,
            checkpoint_store=home/'warm-auditor', **kwargs)
        assert continued == checked
    with pytest.raises(ValueError, match='accepted windows'):
        operator.prepare(state, {**entry, 'replay_documents': 1, 'replay_sources': ['train']})
    altered = copy.deepcopy(values['train'])
    altered[0]['messages'][-1]['content'] = 'word31'
    poisoned = ordinary_operation.Preparation(home/'poison', freeze, store, tokenizer,
        lambda source, start, count: altered if source['role'] == 'train' else values['test'], initialize)
    with pytest.raises(ValueError, match='source-grounded'):
        poisoned.prepare(state, entry)
