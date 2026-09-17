#!/usr/bin/env python3
"""Fit ordinary selector inputs and freeze one complete inference candidate."""
import argparse
import ast
import copy
import json
from pathlib import Path
import shutil

from neuroshard.evolution import access_routing
from neuroshard.evolution.reference_data import identity, save, sha256

ROOT = Path(__file__).resolve().parents[1]
BASE = '3766fcc38ddd2f62e6a445c347ba96907c2972cc3c3713258b5b0d88c7edfd6f'


def read_rows(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def prepare(args):
    import torch
    from transformers import AutoTokenizer
    from neuroshard.evolution.sharded.router_features import EmbeddingFeatures

    torch.set_num_threads(1)
    read = lambda path: json.loads(path.read_bytes())
    diagnostic = read(args.previous/'inputs/plan.json')
    base = read(ROOT/'config/experiments/balanced-raw-router-model.json')
    if identity(base) != BASE:
        raise ValueError('Restore the exact ordinary-question base')
    selection = read(ROOT/'config/experiments/balanced-raw-router-selection.json')
    sources = {}
    for name, digest in selection['files'].items():
        if sha256(args.earlier_inputs/name) != digest:
            raise ValueError('Earlier routing source changed')
        sources[name] = {row['id']: row for row in read_rows(args.earlier_inputs/name)}
    general_path = ROOT/'config/experiments/ordinary-access-general-training.json'
    general = read(general_path)
    sources['ordinary-general'] = {}
    for text in general['questions']:
        key = identity(['ordinary-general-training', text])
        sources['ordinary-general'][key] = {'id': key, 'messages': [{'role': 'user', 'content': text}]}
        selection['training'].append({'id': key, 'file': 'ordinary-general', 'route': 'parent'})
    c_path = args.c_inputs/'train-questions.jsonl'
    if sha256(c_path) != '97b8dbffc0cc9862972c1872183a8af5ad4c8eb21c806a63016fd8f789db312a':
        raise ValueError('Use only the frozen C training inventory')
    excluded = [message['content'] for case in diagnostic['cases'] for message in case['messages']]
    excluded += [atom['gold_question'] for case in diagnostic['cases'] for atom in case['atoms']]
    # Also exclude the already observed planner rewrites; they are development data.
    previous_result = read(args.previous/'result.json')
    excluded += [q for row in previous_result['rows'] for q in row['plan']]
    facts = read(ROOT/'config/experiments/continual-expert-facts.json')
    excluded += [fact['test_question'] for cohort in facts['cohorts'] for fact in cohort['facts']]
    rows, provenance = access_routing.training_rows(selection, sources, read_rows(c_path), excluded)
    if args.output.exists():
        raise ValueError('Preserve the earlier access candidate')
    args.output.mkdir(parents=True)
    inputs = args.output/'inputs'
    inputs.mkdir()
    for path in (args.previous/'inputs').iterdir():
        shutil.copyfile(path, inputs/path.name)
    shutil.copytree(args.previous/'seed', args.output/'seed')
    save(args.output/'previous-result.json', previous_result)
    planned = read(inputs/'planned.json')
    feature = planned['learned']['feature_profile']
    tokenizer = AutoTokenizer.from_pretrained(args.output/'seed', local_files_only=True, trust_remote_code=False)
    features = EmbeddingFeatures(args.embedding, feature['embedding_sha256'], tokenizer, feature['tokenizer_root'])
    if features.profile != feature:
        raise ValueError('Keep the existing integer feature extractor')
    save(args.output/'fitting.json', {'rows': rows, 'provenance': provenance,
        'ordinary_general_source_sha256': sha256(general_path),
        'label_source': 'training-source domains; no diagnostic answers or forced development controls fitted',
        'method': {'epochs': 16, 'base': BASE, 'C_gate': 'ordinary-positive-earlier-negative',
                   'fallback_guard': 'confident-parent-only', 'neural_training': False}})
    candidate = access_routing.fit(base, rows, features, epochs=16)
    save(inputs/'router.json', candidate)
    save(args.output/'fit-result.json', {'router': identity(candidate), 'base': identity(candidate['base']),
        'rows': len(rows), 'by_route': {route: sum(r['route'] == route for r in rows)
                                      for route in candidate['prototypes']}, 'provenance': provenance})
    finalize(args.output)


def source_list(path):
    tree = ast.parse((ROOT/path).read_text())
    return next(ast.literal_eval(node.value) for node in tree.body if isinstance(node, ast.Assign)
                and any(isinstance(target, ast.Name) and target.id == 'SOURCES' for target in node.targets))


def finalize(home, trial_path=None):
    read = lambda name: json.loads((home/'inputs'/name).read_bytes())
    profile, graph, planned = [read(name+'.json') for name in ('profile', 'graph', 'planned')]
    profile['sources'] = {str(path.relative_to(ROOT)): sha256(path)
                         for path in sorted((ROOT/'src/neuroshard').rglob('*.py'))}
    graph['executor_root'] = identity(profile)
    planned['graph'] = identity(graph)
    planned['learned'].update(graph=identity(graph), router=read('router.json'))
    planned['expert_prompts']['planner'] = {
        'prefix': 'NeuroShard research protocol: ', 'suffix': ' Provide only the answer.',
        'context': 'standalone'}
    for key, path in (('learned', 'learned_graph.py'), ('planned', 'planned_graph.py')):
        config = planned['learned'] if key == 'learned' else planned
        config['sources'] = {name: sha256(ROOT/name) for name in
                            source_list('src/neuroshard/evolution/sharded/'+path)}
    if 'request_policy' in planned:
        name = 'src/neuroshard/evolution/request_planning.py'
        planned['sources'][name] = sha256(ROOT/name)
    for name, value in [('profile', profile), ('graph', graph), ('planned', planned)]:
        save(home/'inputs'/(name+'.json'), value)
    paths = ['scripts/run_ordinary_access_trial.py', 'scripts/prepare_ordinary_access_trial.py',
             'config/experiments/ordinary-access-general-training.json',
             'src/neuroshard/evolution/access_routing.py', 'src/neuroshard/evolution/serving_diagnosis.py',
             'src/neuroshard/evolution/sharded/planned_graph.py', 'src/neuroshard/evolution/expert_router.py']
    if 'request_policy' in planned:
        paths += ['src/neuroshard/evolution/request_planning.py',
                  'src/neuroshard/evolution/planned_metering.py',
                  'scripts/prepare_request_preservation.py']
    trial = {'format': 'neuroshard-ordinary-access-trial-v1', 'no_neural_training': True,
        'final_opened': False, 'max_seconds': 3600, 'diagnostic': identity(read('plan.json')),
        'service': identity(planned), 'router': identity(planned['learned']['router']),
        'base': BASE, 'fitting_sha256': sha256(home/'fitting.json'),
        'previous_automatic_passes': [row['id'] for row in
            json.loads((home/'previous-result.json').read_bytes())['rows'] if row['passed']],
        'expert_checkpoints': {key: value['checkpoint'] for key, value in graph['experts'].items()},
        'sources': {name: sha256(ROOT/name) for name in paths},
        'controls': 'forced route and standalone question; same served expert input contract',
        'acceptance': {'automatic_only': True, 'retain_previously_correct': True,
                       'single_gold_selection_required': 'all',
                       'forced_controls_never_count_as_automatic': True},
        'stop_rule': 'One inference allocation. Preserve complete traces and retire. No cohort training or promotion.'}
    save(home/'inputs/access-trial.json', trial)
    save(trial_path or ROOT/'config/experiments/ordinary-access-trial.json', trial)
    print(json.dumps({'service': trial['service'], 'router': trial['router'],
                      'commit_required_before_GPU': True}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--previous', type=Path)
    parser.add_argument('--earlier-inputs', type=Path)
    parser.add_argument('--c-inputs', type=Path)
    parser.add_argument('--embedding', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--finalize-only', action='store_true')
    args = parser.parse_args()
    finalize(args.output) if args.finalize_only else prepare(args)
