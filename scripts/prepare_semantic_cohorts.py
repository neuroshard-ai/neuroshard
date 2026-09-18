#!/usr/bin/env python3
"""Compile prospective cohorts for the existing native ordinary operator.

Only training questions enter selector fitting. The encoder never receives
answers or finals. Neural expert training requires the subsequent operation
and source freeze, exactly as in the ordinary campaign.
"""
import argparse
import ast
import copy
import json
import math
from pathlib import Path
import shutil
import subprocess
import time

from neuroshard.dataflow.store import canonical
from neuroshard.evolution import answering, expert_router, ordinary_cohorts, semantic_questions
from neuroshard.evolution.access_routing import question_key
from neuroshard.evolution.objects import Objects
from neuroshard.evolution.reference_data import identity, save, sha256
from neuroshard.evolution.request_planning import LOSSLESS_POLICY, SPAN_POLICY
from prepare_ordinary_cohorts import configuration

ROOT = Path(__file__).resolve().parents[1]
ORDER = ('escrow', 'conversation', 'feed')


def read(path):
    return json.loads(Path(path).read_bytes())


def prepare(home, previous, general, facts, encoder, *, order=ORDER, fine_intents=False,
            bootstrap=None, semantic_admission=False):
    order = tuple(order)
    if type(semantic_admission) is not bool or (semantic_admission and not fine_intents):
        raise ValueError('Semantic admission requires per-expert fine intent selection')
    if (len(order) != 3 or len(set(order)) != 3 or type(fine_intents) is not bool
            or any(not isinstance(name, str) or not name or len(name) > 32
                   or any(char not in 'abcdefghijklmnopqrstuvwxyz0123456789_-' for char in name) for name in order)):
        raise ValueError('Declare exactly three bounded distinct cohort names')
    compiled = home/'compiled'
    compiled.mkdir(parents=True, exist_ok=False)
    old = Objects(previous/'objects')
    store = Objects(compiled/'objects')
    initial_inputs = read(previous/'inputs.json')
    initial = old.json(initial_inputs['policies']['baseline'])['configuration']
    graph = read(previous/'baseline-core.json')
    if set(graph['experts']) != {'directory', 'protocol', 'planner'}:
        raise ValueError('Start from the earlier accepted ABC seed, never the opened failed cohort')
    fresh = read(facts)
    catalog = read(previous/'source-catalog.json')
    if (fresh['source_revision'] != catalog['revision'] or fresh['cohort'] not in order
            or any(name not in catalog['cohorts'] for name in order if name != fresh['cohort'])
            or set(order) & set(graph['experts'])):
        raise ValueError('Ground all three prospective cohorts in one pinned source revision')
    catalog['cohorts'] = {name: fresh['facts'] if name == fresh['cohort'] else catalog['cohorts'][name] for name in order}
    catalog['order'] = list(order)
    evidence = ordinary_cohorts.source_evidence(catalog, lambda revision, path:
        subprocess.check_output(['git', 'show', revision+':'+path], cwd=ROOT))
    if any(len(values) != 16 for values in catalog['cohorts'].values()):
        raise ValueError('Keep sixteen independently sourced facts per cohort')
    # Preserve the exact eight accepted ABC atoms used in the earlier unopened
    # conversation/feed tests. No opened admission answer is inherited.
    old_final = [json.loads(line) for line in (previous/'admission/final.jsonl').read_bytes().splitlines()]
    atoms = [{'question': row['messages'][0]['content'].split(' Also, ', 1)[1],
              'topic': row['topics'][1], 'answer': row['answers'][1], 'aliases': row['answer_aliases'][1]}
             for row in old_final[24:32]]
    original = [row for row in read(previous/'selector-fitting.json')['rows']
                if row['route'] in {'parent', *graph['experts']}]
    if len(original) != 1478:
        raise ValueError('Preserve the original training-only ABC selector inputs')
    rows = {row['id']: {**row, 'intent': 'parent'} for row in original}
    for row in read(general)['rows']:
        if row['route'] != 'parent':
            continue
        if row['id'] in rows and (rows[row['id']]['question'] != row['question'] or rows[row['id']]['route'] != 'parent'):
            raise ValueError('Training inventories disagree about a preserved prompt')
        rows.setdefault(row['id'], {**row, 'intent': 'parent'})
    base_ids = sorted(rows)
    for name in order:
        train, annotations = ordinary_cohorts.training(catalog['cohorts'][name])
        folder = compiled/name
        folder.mkdir()
        (folder/'training.jsonl').write_bytes(b''.join(canonical(row)+b'\n' for row in train))
        if name != fresh['cohort']:
            shutil.copyfile(previous/name/'final.jsonl', folder/'final.jsonl')
        else:
            (folder/'final.jsonl').write_bytes(b''.join(canonical(row)+b'\n'
                for row in ordinary_cohorts.evaluation(catalog['cohorts'][name], atoms)))
        save(folder/'training-annotations.json', annotations)
        for row in annotations:
            if row['id'] in rows:
                raise ValueError('New cohort training collides with previous fitting')
            rows[row['id']] = {'id': row['id'], 'document': row['id'], 'route': name,
                               'intent': row['topic'], 'question': row['question']}
    questions = [{'id': row['id'], 'question': row['question']} for row in sorted(rows.values(), key=lambda row: row['id'])]
    if len(questions) > semantic_questions.MAX_ROWS:
        raise ValueError('The complete selector exceeds its serving inventory bound')
    heldout = {question_key(fact['test']) for values in catalog['cohorts'].values() for fact in values}
    anchors = read(previous/'anchors.json')
    heldout.update(question_key(row['messages'][-2]['content']) for values in anchors.values() for row in values)
    if any(question_key(row['question']) in heldout for row in rows.values()):
        raise ValueError('Evaluation questions cannot become selector training inputs')
    if bootstrap is None:
        if order[0] != 'escrow':
            raise ValueError('Provide separately frozen bootstrap questions for the first cohort')
        bootstrap = {'questions': [
        {'topic': 'escrow-request', 'question': 'Which NeuroShard transaction reserves my payment before an expert generates an answer?'},
        {'topic': 'escrow-complete-response', 'question': 'What transaction returns the complete answering-system response to an inference customer?'},
        {'topic': 'escrow-paid-field', 'question': 'After a paid expert answer completes, which ledger field states the amount that was charged?'},
        {'topic': 'escrow-refund-field', 'question': 'When a paid expert job leaves an unused reservation, which result field states the refund?'}]}
    if (set(bootstrap) != {'questions'} or len(bootstrap['questions']) != 4
            or len({row['topic'] for row in bootstrap['questions']}) != 4):
        raise ValueError('Declare four distinct bootstrap facts')
    if any(question_key(row['question']) in heldout | {question_key(q['question']) for q in questions}
           for row in bootstrap['questions']):
        raise ValueError('The bootstrap exercise must use distinct prospective wording')
    if not {row['topic'] for row in bootstrap['questions']} <= {fact['id'] for fact in catalog['cohorts'][order[0]]}:
        raise ValueError('Bootstrap questions must refer to actual pinned source facts')
    rules = read(previous/'quality-rule.json')
    for spec in rules['retention_anchors'].values():
        if store.put(old.get(spec['sha256'])) != spec['sha256']:
            raise ValueError('Preserve exact accepted assistant anchors')
    for filename, value in {'baseline-core.json': graph, 'baseline-profile.json': read(previous/'baseline-profile.json'),
            'baseline-configuration.json': initial, 'source-catalog.json': catalog, 'source-evidence.json': evidence,
            'anchors.json': anchors, 'quality-rule.json': rules, 'bootstrap-questions.json': bootstrap,
            'selector-fitting.json': {'format': ordinary_cohorts.FORMAT+'/semantic-fitting',
                'rows': list(rows.values()), 'base_ids': base_ids,
                'policy': 'Training questions only. Preserve ABC and its general guard; append gates and cumulative question-only semantic access.'}}.items():
        save(compiled/filename, value)
    shutil.copytree(previous/'seed', compiled/'seed')
    save(home/'questions.json', questions)
    save(home/'encoder.json', read(encoder))
    save(home/'feature-profile.json', initial['learned']['feature_profile'])
    plan = {'format': 'neuroshard-prospective-semantic-cohorts-v1', 'order': list(order),
        'questions': identity(questions), 'encoder': identity(read(encoder)),
        'features': identity(initial['learned']['feature_profile']), 'source_facts': sha256(facts),
        'driver': sha256(__file__), 'batch': 1, 'coarse_epochs': 16,
        'semantic_method': ('Learned expert admission; preserve accepted routing on rejection; per-expert intent selection.'
            if semantic_admission else 'Preserved nearest-question domain gate, then per-expert integer intent classifiers.'
            if fine_intents else 'Nearest committed training question; earlier examples mean preserve base routing.'),
        'composition': SPAN_POLICY if fine_intents else LOSSLESS_POLICY,
        'fine_intents': fine_intents, 'semantic_admission': semantic_admission,
        'old_final_used_for_training': False,
        'previous_failed_cohort_counted': False, 'neural_training_started': False,
        'quality_rule': identity(rules), 'expert_steps': 128, 'learning_rate': .00005,
        'seconds_per_comparison_arm': 9000,
        'coarse_fitter': sha256(ROOT/'config/experiments/question-match-probe-20260918/retrieval-pilot.py')}
    save(home/'input-plan.json', plan)
    return plan


def features(home, assets, seed, objects):
    """Run frozen encoders on the partition owner; receive no target answers."""
    import torch
    from transformers import AutoTokenizer
    from neuroshard.evolution.sharded.semantic_features import SemanticFeatures
    from neuroshard.evolution.sharded.router_features import EmbeddingFeatures
    plan, questions, encoder, profile = (read(home/name) for name in
        ('input-plan.json', 'questions.json', 'encoder.json', 'feature-profile.json'))
    if (plan['driver'] != sha256(__file__) or plan['questions'] != identity(questions)
            or plan['encoder'] != identity(encoder) or plan['features'] != identity(profile)):
        raise ValueError('Feature extraction differs from its committed prescription')
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    semantic = SemanticFeatures(encoder, assets, 'cuda')
    coarse = EmbeddingFeatures(objects/(profile['embedding_sha256']+'.safetensors'), profile['embedding_sha256'],
        AutoTokenizer.from_pretrained(seed, local_files_only=True), profile['tokenizer_root'], max_tokens=profile['max_tokens'])
    if coarse.profile != profile:
        raise ValueError('Coarse feature execution changed the accepted feature profile')
    started, result = time.monotonic(), []
    for row in questions:
        result.append({'id': row['id'], 'semantic': semantic(row['question'])['features'], 'coarse': coarse(row['question'])})
        if len(result) % 128 == 0:
            save(home/'feature-progress.json', {'done': len(result), 'count': len(questions)})
    save(home/'features.json', result)
    save(home/'feature-result.json', {'features': identity(result), 'plan': identity(plan),
        'count': len(result), 'seconds': time.monotonic()-started, 'answers_received': False, 'finals_received': False})


def append_gate(previous, samples, route, reference):
    """Use the previously cross-checked exact integer offline NumPy fitter."""
    import numpy as np
    tree = ast.parse(reference.read_text())
    functions = ast.Module(body=[node for node in tree.body if isinstance(node, ast.FunctionDef)
        and node.name in {'prototypes_fast', 'fit_fast'}], type_ignores=[])
    namespace = {'np': np, 'er': expert_router, 'identity': identity, 'math': math}
    exec(compile(functions, str(reference), 'exec'), namespace)
    fitting = [{**row, 'route': route if row['route'] == route else 'parent'} for row in samples]
    prototype = namespace['prototypes_fast'](fitting, previous['embedding_root'], previous['tokenizer_root'])
    gate = namespace['fit_fast'](fitting, prototype, epochs=16)
    base = previous['base'] if previous['format'] == expert_router.GROWING_FORMAT else previous
    additions = copy.deepcopy(previous.get('additions', []))+[{'route': route, 'gate': gate}]
    model = {key: copy.deepcopy(base[key]) for key in expert_router.FIELDS}
    model.update(format=expert_router.GROWING_FORMAT, base=copy.deepcopy(base), additions=additions,
        minimum_margin=0, maximum_distance=2**40,
        prototypes={**copy.deepcopy(previous['prototypes']), route: copy.deepcopy(gate['prototypes'][route])},
        training_root=identity({'base': base['training_root'], 'additions': [row['gate']['training_root'] for row in additions]}))
    if 'fallback_guard' in previous:
        model['fallback_guard'] = copy.deepcopy(previous['fallback_guard'])
    return expert_router.validate(model)


def compile_policies(home):
    compiled, store = home/'compiled', Objects(home/'compiled/objects')
    plan = read(home/'input-plan.json')
    reference = ROOT/'config/experiments/question-match-probe-20260918/retrieval-pilot.py'
    if sha256(__file__) != plan['driver'] or sha256(reference) != plan['coarse_fitter']:
        raise ValueError('Recommit the compiler before fitting a changed selector recipe')
    features = {row['id']: row for row in read(home/'features.json')}
    fitting = read(compiled/'selector-fitting.json')
    if set(features) != {row['id'] for row in fitting['rows']}:
        raise ValueError('Feature inventory must match only the declared training prompts')
    initial, graph, catalog = (read(compiled/name) for name in
        ('baseline-configuration.json', 'baseline-core.json', 'source-catalog.json'))
    model, encoder = initial['learned']['router'], read(home/'encoder.json')
    preserved_router = copy.deepcopy(model)
    baseline = answering.attach(graph, configuration(graph, model, initial), store)
    policies = {'baseline': baseline['answering']['policy_root']}
    topology, cohorts, intents = copy.deepcopy(graph), {}, {}
    initial['request_policy'] = plan['composition']
    order = tuple(plan['order'])
    for index, name in enumerate(order):
        included = {'parent', 'directory', 'protocol', 'planner', *order[:index+1]}
        rows = [row for row in fitting['rows'] if row['route'] in included]
        samples = [{'id': row['id'], 'route': row['route'], 'features': features[row['id']]['coarse']} for row in rows]
        model = append_gate(model, samples, name, reference)
        intents.update({fact['id']: {'route': name, 'question': fact['training'][0]} for fact in catalog['cohorts'][name]})
        semantic = semantic_questions.build([{'id': row['id'], 'route': row['intent'],
            'features': features[row['id']]['semantic']} for row in rows], encoder, intents)
        if plan.get('fine_intents', False):
            semantic = semantic_questions.fit_intents(semantic)
        if plan.get('semantic_admission', False):
            semantic = semantic_questions.fit_admission(semantic, preserved_router)
        topology = ordinary_cohorts.extend(topology, name, graph['experts']['planner'])
        bound = answering.attach(topology, configuration(topology, model,
            {**initial, 'semantic_questions': semantic}), store)
        policies[name] = bound['answering']['policy_root']
        cohorts[name] = {'training': {'sha256': sha256(compiled/name/'training.jsonl'), 'count': 384},
            'test': {'sha256': sha256(compiled/name/'final.jsonl'), 'count': 32},
            'router': store.put_json(model), 'semantic_questions': identity(semantic),
            'answering_policy': policies[name], 'fitting_prompts': len(rows)}
        print(json.dumps({'cohort_compiled': name, 'fitting_prompts': len(rows)}), flush=True)
    rules = read(compiled/'quality-rule.json')
    result = {'format': ordinary_cohorts.FORMAT+'/inputs', 'source_evidence': identity(read(compiled/'source-evidence.json')),
        'starting_experts': {name: value['checkpoint'] for name, value in graph['experts'].items()},
        'cohorts': cohorts, 'policies': policies, 'anchors': rules['retention_anchors'],
        'quality_rule': identity(rules), 'final_opened': False, 'neural_training_started': False}
    save(compiled/'inputs.json', result)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('prepare', 'features', 'compile'))
    parser.add_argument('--home', type=Path, required=True)
    for name in ('previous', 'general', 'facts', 'encoder', 'assets', 'seed', 'objects'):
        parser.add_argument('--'+name, type=Path)
    args = parser.parse_args()
    if args.action == 'prepare':
        print(json.dumps(prepare(args.home, args.previous, args.general, args.facts, args.encoder)))
    elif args.action == 'features':
        features(args.home, args.assets, args.seed, args.objects)
    else:
        print(json.dumps(compile_policies(args.home)))
