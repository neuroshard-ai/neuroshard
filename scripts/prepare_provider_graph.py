"""Migrate accepted serving commitments to the provider transport, without training.

Only source/executor commitments change. Every model tensor, tokenizer byte,
selector, planning/composition instruction and generation rule is preserved.
This creates reviewable deployment inputs; it does not promote them on a chain.
"""
import argparse
import copy
import json
from pathlib import Path
import shutil
import subprocess

from neuroshard.evolution import answering, provider_assets
from neuroshard.evolution.objects import Objects
from neuroshard.evolution.reference_data import identity, save, sha256
from neuroshard.evolution.sharded import learned_graph, planned_graph

ROOT = Path(__file__).resolve().parents[1]


def migrate(graph, profile, store, source_home, destination):
    original = answering.load(graph, store)
    model = answering.core(graph)
    profile = copy.deepcopy(profile)
    profile['sources'] = {p.relative_to(source_home).as_posix(): sha256(p)
                          for p in sorted((source_home/'src/neuroshard').rglob('*.py'))}
    model['executor_root'] = identity(profile)
    old = original['learned']
    learned = learned_graph.configuration(model, old['router'], old['feature_profile'], source_home,
        old.get('route_models'), compose='composition' in old)
    optional = ('expert_prompts', 'general_instruction', 'route_scopes', 'planner_weights', 'composer',
                'answer_policy', 'request_policy', 'general_answer_policy', 'semantic_questions', 'question_reranker')
    policy = planned_graph.configuration(model, learned, original['planner'], source_home,
                                         **{key: original[key] for key in optional if key in original})
    def behavior(value):
        value = copy.deepcopy(value)
        for target in (value, value['learned']):
            target.pop('sources', None)
            target.pop('graph', None)
        return value
    if behavior(policy) != behavior(original):
        raise ValueError('Transport migration changed the accepted answering behavior')
    new_store = Objects(destination/'policies')
    migrated = answering.attach(model, policy, new_store)
    planned_graph.validate_configuration(model, policy, source_home)
    restored = answering.core(migrated)
    restored['executor_root'] = graph['executor_root']
    if restored != answering.core(graph):
        raise ValueError('Transport migration changed accepted model data')
    save(destination/'graph.json', migrated)
    save(destination/'profile.json', profile)
    save(destination/'policy.json', policy)
    return migrated, profile, policy


def prepare(study, destination, expected_graph):
    if destination.exists():
        raise ValueError('Choose an empty destination; preserve earlier deployment inputs')
    state = json.loads((study/'final-state.json').read_bytes())
    graph = state['expert_lifecycle']['serving_graph']
    if identity(graph) != expected_graph or state['serving_root'] != expected_graph:
        raise ValueError('Study does not contain the accepted graph selected for deployment')
    destination.mkdir(parents=True)
    graph, profile, policy = migrate(graph,
        json.loads((study/'compiled/baseline-profile.json').read_bytes()),
        Objects(study/'compiled/objects'), ROOT, destination)
    lengths = {}
    def collect(value):
        if isinstance(value, dict):
            if {'sha256', 'bytes'} <= set(value):
                key, size = value['sha256'], value['bytes']
                if key in lengths and lengths[key] != size:
                    raise ValueError('Accepted metadata disagrees about an object length')
                lengths[key] = size
            for nested in value.values():
                collect(nested)
        elif isinstance(value, list):
            for nested in value:
                collect(nested)
    collect(graph)
    collect(policy)
    collect(json.loads((study/'auxiliary-assets.json').read_bytes()))
    seed = destination/'seed'
    seed.mkdir()
    mirror = destination/'metadata'
    mirror.mkdir()
    for name, key in graph['tokenizer']['files'].items():
        source = study/'compiled/seed'/name
        if sha256(source) != key:
            raise ValueError('Accepted tokenizer bytes changed')
        lengths[key] = source.stat().st_size
        shutil.copyfile(source, seed/name)
        shutil.copyfile(source, mirror/key)
    key = graph['answering']['policy_root']
    source = Objects(destination/'policies').path(key)
    lengths[key] = source.stat().st_size
    shutil.copyfile(source, mirror/key)
    owners = []
    for rank in range(3 + len(graph['experts'])):
        files = provider_assets.plan(graph, rank, policy)
        for spec in files.values():
            if spec['bytes'] is None:
                spec['bytes'] = lengths[spec['sha256']]
            if lengths[spec['sha256']] != spec['bytes']:
                raise ValueError('Provider plan changed an accepted object length')
        owners.append({'rank': rank, 'bytes': sum(spec['bytes'] for spec in files.values()), 'files': files})
    all_backbone = {spec['sha256'] for spec in graph['parent']['tensors'].values()}
    if any(all_backbone <= {spec['sha256'] for spec in row['files'].values()} for row in owners):
        raise ValueError('A provider would receive the full backbone')
    save(destination/'object-lengths.json', lengths)
    save(destination/'owners.json', owners)
    report = {'format': 'neuroshard-provider-graph-migration-v1', 'accepted_graph': expected_graph,
        'graph': identity(graph), 'executor': identity(profile), 'policy': key,
        'source_revision': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
        'trained': False, 'model_and_tokenizer_preserved': True, 'answering_behavior_preserved': True,
        'logical_owners': len(owners), 'owner_bytes': [row['bytes'] for row in owners],
        'unique_bytes': sum(lengths[key] for key in {spec['sha256'] for row in owners for spec in row['files'].values()}),
        'note': 'Metadata preparation only; numerical equivalence and operated service remain required'}
    save(destination/'migration.json', report)
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--study', type=Path, required=True)
    parser.add_argument('--destination', type=Path, required=True)
    parser.add_argument('--accepted-graph', required=True)
    args = parser.parse_args()
    print(json.dumps(prepare(args.study, args.destination, args.accepted_graph), indent=2))
