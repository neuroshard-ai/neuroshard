#!/usr/bin/env python3
"""Freeze ordinary cohorts, training-only selectors and whole-system policies.

This compiler reads only metadata and the immutable embedding on a CPU. It
does not train neural experts, evaluate final answers or start cloud resources.
Its complete output must be committed before the numerical campaign starts.
"""
import argparse
import copy
import json
from pathlib import Path
import shutil
import subprocess

from neuroshard.dataflow.store import canonical
from neuroshard.evolution import answering, expert_router, ordinary_cohorts as campaign
from neuroshard.evolution import expert_preparation, ordinary_quality
from neuroshard.evolution.access_routing import question_key
from neuroshard.evolution.data import document_identity
from neuroshard.evolution.objects import Objects
from neuroshard.evolution.reference_data import identity, save, sha256
from neuroshard.evolution.sharded import learned_graph, planned_graph

ROOT = Path(__file__).resolve().parents[1]


def retention(serving):
    """Freeze exposed accepted responses as regressions, never new learning."""
    plan = json.loads((serving/'inputs/plan.json').read_bytes())
    result = json.loads((serving/'result.json').read_bytes())
    rows = {row['id']: row for row in result['rows']}
    retained, atoms = {}, {}
    for case in plan['cases']:
        measured = rows[case['id']]
        if measured['passed'] is not True:
            raise ValueError('The initial ordinary service must have passed its exposed diagnostic')
        actual = measured['answers']
        values = [value['text'] if isinstance(value, dict) else value for value in actual]
        if len(values) != len(case['atoms']):
            raise ValueError('Every accepted ordinary answer must retain its actual atom inventory')
        topics = [identity({'question': atom['gold_question']}) for atom in case['atoms']]
        row = campaign.scored(case['messages'], topics, values,
            [[value+'.'] if not value.endswith('.') else [] for value in values])
        retained[document_identity(row['messages'])] = row
        for atom, topic, answer in zip(case['atoms'], topics, values):
            atoms.setdefault(question_key(atom['gold_question']), {
                'question': atom['gold_question'], 'topic': topic, 'answer': answer,
                'aliases': [answer+'.'] if not answer.endswith('.') else []})
    for value in json.loads((serving/'inputs/retention.json').read_bytes()):
        atom = value['atom']
        topic = identity({'question': atom['gold_question']})
        row = campaign.scored([{'role': 'user', 'content': atom['gold_question']}],
                               [topic], [atom['answer']])
        key = document_identity(row['messages'])
        if key in retained and retained[key] != row:
            raise ValueError('Accepted knowledge anchors disagree on scoring metadata')
        retained[key] = row
        atoms.setdefault(question_key(atom['gold_question']), {
            'question': atom['gold_question'], 'topic': topic, 'answer': atom['answer'],
            'aliases': [atom['answer']+'.']})
    return list(retained.values()), list(atoms.values())


def configuration(graph, router, initial):
    learned = learned_graph.configuration(graph, router, initial['learned']['feature_profile'], ROOT)
    prompts = copy.deepcopy(initial['expert_prompts'])
    for name in set(graph['experts'])-set(prompts)-{'directory'}:
        prompts[name] = {'prefix': 'NeuroShard research protocol: ',
                         'suffix': ' Provide only the answer.', 'context': 'standalone'}
    return planned_graph.configuration(graph, learned, initial['planner'], ROOT, prompts,
        initial['general_instruction'], **{name: initial[name] for name in (
            'route_scopes', 'planner_weights', 'composer', 'answer_policy', 'request_policy',
            'general_answer_policy') if name in initial})


def compile_inputs(serving, fitting, embedding, home):
    from transformers import AutoTokenizer
    from neuroshard.evolution.sharded.router_features import EmbeddingFeatures
    home.mkdir(parents=True, exist_ok=False)
    store = Objects(home/'objects')
    initial = json.loads((serving/'inputs/planned.json').read_bytes())
    graph = json.loads((serving/'inputs/graph.json').read_bytes())
    if graph['experts']['planner']['checkpoint'] != '7d976925e8ff9d3c7588c2eed5ac9852a02d454318842093fcc353e7fa2338fb':
        raise ValueError('Continue the exact accepted ordinary development service')
    catalog = campaign.catalogs(ROOT)
    evidence = campaign.source_evidence(catalog, lambda revision, path: subprocess.check_output(
        ['git', 'show', revision+':'+path], cwd=ROOT))
    knowledge, atoms = retention(serving)
    anchors = {'retained-test-knowledge': knowledge, **campaign.assistant_anchors(ROOT)}
    anchor_specs = {name: expert_preparation.record_set(store, name,
        [{**row, 'id': document_identity(row['messages'])} for row in values])
        for name, values in anchors.items()}
    save(home/'anchors.json', anchors)
    save(home/'source-evidence.json', evidence)
    save(home/'source-catalog.json', catalog)
    fitted = json.loads(fitting.read_bytes())
    rows = copy.deepcopy(fitted['rows'])
    if len(rows) != 1478 or set(row['route'] for row in rows) != {'parent', 'directory', 'protocol', 'planner'}:
        raise ValueError('Keep the exact earlier training-only selector inventory')
    heldout = {question_key(fact['test']) for values in catalog['cohorts'].values() for fact in values}
    heldout.update(question_key(row['messages'][-2]['content']) for values in anchors.values() for row in values)
    if any(question_key(row['question']) in heldout for row in rows):
        raise ValueError('Prospective evaluation overlaps earlier selector fitting')
    feature_profile = initial['learned']['feature_profile']
    tokenizer = AutoTokenizer.from_pretrained(serving/'seed', local_files_only=True)
    features = EmbeddingFeatures(embedding, feature_profile['embedding_sha256'], tokenizer,
        graph['tokenizer']['root'], max_tokens=feature_profile['max_tokens'])
    if features.profile != feature_profile:
        raise ValueError('The selector feature definition changed')
    samples = [{'id': row['id'], 'route': row['route'], 'features': features(row['question'])} for row in rows]
    model = initial['learned']['router']
    policies, routers, cohorts = {}, {}, {}
    baseline = answering.attach(graph, configuration(graph, model, initial), store)
    policies['baseline'] = baseline['answering']['policy_root']
    # These graph copies describe topology only. The policy object deliberately
    # omits model roots. No cloned checkpoint is presented as newly trained work.
    topology = copy.deepcopy(graph)
    retained_atoms = list(atoms)
    for name in campaign.ORDER:
        train, metadata = campaign.training(catalog['cohorts'][name])
        final = campaign.evaluation(catalog['cohorts'][name], retained_atoms)
        if any(question_key(row['question']) in heldout for row in metadata):
            raise ValueError('Do not fit a selector on any declared final or retained question')
        additions = [{'id': row['id'], 'document': row['id'], 'route': name,
                      'question': row['question']} for row in metadata]
        rows.extend(additions)
        samples.extend({'id': row['id'], 'route': name, 'features': features(row['question'])} for row in additions)
        model = expert_router.append_route(model, samples, name, epochs=16)
        routers[name] = store.put_json(model)
        topology = campaign.extend(topology, name, graph['experts']['planner'])
        bound = answering.attach(topology, configuration(topology, model, initial), store)
        policies[name] = bound['answering']['policy_root']
        folder = home/name
        folder.mkdir()
        for filename, values in (('training', train), ('final', final)):
            (folder/(filename+'.jsonl')).write_bytes(b''.join(canonical(row)+b'\n' for row in values))
        save(folder/'training-annotations.json', metadata)
        # Fitting accuracy is a diagnostic only. No final route or answer is
        # queried here and no threshold is selected using a retention score.
        positives = [expert_router.select(model, row['features'])['route'] == name
                     for row in samples if row['route'] == name]
        earlier = [expert_router.select(model, row['features'])['route'] != name
                   for row in samples if row['route'] != name]
        cohorts[name] = {'training': {'sha256': sha256(folder/'training.jsonl'), 'count': len(train)},
            'test': {'sha256': sha256(folder/'final.jsonl'), 'count': len(final)},
            'router': routers[name], 'answering_policy': policies[name],
            'training_selector': {'new_correct': sum(positives), 'new_count': len(positives),
                                 'earlier_not_stolen': sum(earlier), 'earlier_count': len(earlier)}}
        retained_atoms.extend({'question': fact['test'], 'topic': fact['id'], 'answer': fact['answer'],
                               'aliases': [fact['answer']+'.']} for fact in catalog['cohorts'][name])
        print(json.dumps({'cohort': name, **cohorts[name]['training_selector']}), flush=True)
    save(home/'selector-fitting.json', {'format': campaign.FORMAT+'/fitting',
        'earlier_sha256': sha256(fitting), 'rows': rows,
        'policy': 'Prompt-only training inputs. Existing gate parameters preserved; new binary gates appended.'})
    save(home/'baseline-core.json', graph)
    save(home/'baseline-profile.json', json.loads((serving/'inputs/profile.json').read_bytes()))
    shutil.copytree(serving/'seed', home/'seed')
    rules = {'format': ordinary_quality.FORMAT, 'gates': {'single_accuracy': .8, 'composed_accuracy': .75,
        'gain_lower': 0., 'bootstrap_samples': 10000, 'bootstrap_seed': 9172026, 'confidence': .95},
        'generation': {'new': 64, 'retained_knowledge': 64, 'retained_skills': 64, 'retained_conversation': 64},
        'retention_gates': {'max_lost_correct': 0,
            'minimum_accuracy': dict.fromkeys(anchor_specs, .75)},
        'retention_anchors': anchor_specs}
    save(home/'quality-rule.json', rules)
    result = {'format': campaign.FORMAT+'/inputs', 'source_evidence': identity(evidence),
        'starting_experts': {name: value['checkpoint'] for name, value in graph['experts'].items()},
        'cohorts': cohorts, 'policies': policies, 'anchors': anchor_specs,
        'quality_rule': identity(rules), 'final_opened': False, 'neural_training_started': False}
    save(home/'inputs.json', result)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--serving', type=Path, required=True)
    parser.add_argument('--fitting', type=Path, required=True)
    parser.add_argument('--embedding', type=Path, required=True)
    parser.add_argument('--home', type=Path, required=True)
    args = parser.parse_args()
    compile_inputs(args.serving, args.fitting, args.embedding, args.home)
