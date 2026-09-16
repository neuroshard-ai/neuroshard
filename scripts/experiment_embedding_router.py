#!/usr/bin/env python3
"""Prepare and run one frozen, CPU-only router development screen."""
import argparse
from collections import defaultdict
import json
from pathlib import Path
import re
import subprocess
import time

from neuroshard.evolution import expert_router
from neuroshard.evolution.router_data import raw_questions
from neuroshard.evolution.reference_data import identity, sha256

ROOT = Path(__file__).resolve().parents[1]
PLAN = ROOT / 'config/experiments/embedding-router.json'
SELECTION = ROOT / 'config/experiments/embedding-router-selection.json'
SOURCES = ['src/neuroshard/evolution/expert_router.py',
           'src/neuroshard/evolution/sharded/router_features.py',
           'src/neuroshard/evolution/router_data.py',
           'src/neuroshard/evolution/incremental_facts.py',
           'scripts/experiment_embedding_router.py']
ROLES = {'train.jsonl': 'protocol', 'dev.jsonl': 'protocol',
         'retained-dev-knowledge.jsonl': 'directory',
         'retained-dev-skills.jsonl': 'parent', 'retained-dev-conversation.jsonl': 'parent'}


def committed(path):
    relative = str(path.relative_to(ROOT))
    saved = subprocess.check_output(['git', 'show', 'HEAD:' + relative], cwd=ROOT)
    if saved != path.read_bytes():
        raise ValueError('Commit the exact study inputs and source before execution: ' + relative)
    return saved


def read_rows(path):
    with path.open() as stream:
        return [json.loads(line) for line in stream if line.strip()]


def user_context(messages):
    """Include user conversation context without any reference response."""
    if not isinstance(messages, list) or not 1 <= len(messages) <= 32:
        raise ValueError('Require a bounded source conversation')
    questions = [message['content'] for message in messages if message['role'] == 'user']
    if not questions or any(not isinstance(text, str) for text in questions):
        raise ValueError('Require text from a user message')
    return '\n\n'.join(questions)


def choose(inputs, plan):
    training, evaluation, files = [], [], {}
    for filename, route in ROLES.items():
        path = inputs / filename
        files[filename] = sha256(path)
        rows = read_rows(path)
        candidates, held = [], []
        for row in rows:
            # All variants of one directory entity stay on the same side.
            group = row.get('task', {}).get('name', row['id'])
            train_side = int(identity({'router_split': group})[-1], 16) % 2 == 0
            if filename == 'train.jsonl':
                candidates.append(row)
            elif filename == 'dev.jsonl' or not train_side:
                held.append(row)
            else:
                candidates.append(row)
        limit = plan['training_per_source'].get(filename, 0)
        candidates.sort(key=lambda row: identity({'router_sample': row['id']}))
        if len(candidates) < limit:
            raise ValueError('Insufficient committed router training rows in ' + filename)
        training.extend({'file': filename, 'id': row['id'], 'route': route} for row in candidates[:limit])
        evaluation.extend({'file': filename, 'id': row['id'], 'route': route} for row in held)
    if set(row['id'] for row in training) & set(row['id'] for row in evaluation):
        raise ValueError('Router development evaluation overlaps fitting')
    if len({row['id'] for row in training + evaluation}) != len(training) + len(evaluation):
        raise ValueError('Duplicate router source identity')
    selected = {'format': plan['format'] + '/selection', 'plan': identity(plan),
            'files': files, 'training': training, 'evaluation': evaluation,
            'sources': {name: sha256(ROOT / name) for name in SOURCES}}
    if plan.get('raw_questions'):
        # Fail during preparation if a source format is not understood, and
        # commit the actual augmented inputs before any classifier is fitted.
        source_rows = {name: {row['id']: row for row in read_rows(inputs / name)} for name in ROLES}
        selected['raw_inputs'] = {}
        for side, items in (('training', training), ('evaluation', evaluation)):
            variants = [{'document': item['id'], 'questions': raw_questions(
                source_rows[item['file']][item['id']], item['route'])} for item in items]
            selected['raw_inputs'][side] = {'root': identity(variants),
                                            'count': sum(len(row['questions']) for row in variants)}
    return selected


def prepare(args):
    # Preparation only writes selection metadata. Execution below still requires
    # one commit containing the exact plan, source and selected observations.
    plan = json.loads(args.plan.read_bytes())
    selected = choose(args.inputs, plan)
    args.selection.write_text(json.dumps(selected, indent=2, sort_keys=True) + '\n')
    print(json.dumps({'selection': str(args.selection), 'training': len(selected['training']),
                      'evaluation': len(selected['evaluation']), 'commit_required_before_run': True}))


def run(args):
    import torch
    from neuroshard.evolution.sharded.incremental_job import tokenizer_for
    from neuroshard.evolution.sharded.router_features import EmbeddingFeatures

    started = time.monotonic()
    plan = json.loads(committed(args.plan))
    selected = json.loads(committed(args.selection))
    for name in SOURCES:
        committed(ROOT / name)
    if selected != choose(args.inputs, plan):
        raise ValueError('Frozen router rows, inputs or execution source changed')
    if args.output.exists():
        raise ValueError('Preserve the previous result; do not overwrite it')
    args.output.mkdir(parents=True)
    torch.set_num_threads(plan['resources']['threads'])
    graph = json.loads((args.inputs / 'graph.json').read_bytes())
    if graph['tokenizer']['root'] != plan['tokenizer_root']:
        raise ValueError('Router tokenizer differs from the existing model')
    tokenizer = tokenizer_for({'tokenizer': graph['tokenizer']['root'],
                               'tokenizer_files': graph['tokenizer']['files']}, args.seed)
    features = EmbeddingFeatures(args.embedding, plan['embedding_sha256'], tokenizer, plan['tokenizer_root'])
    rows = {filename: {row['id']: row for row in read_rows(args.inputs / filename)} for filename in ROLES}

    def question(item):
        messages = rows[item['file']][item['id']]['messages']
        return user_context(messages)

    def variants(item):
        return (raw_questions(rows[item['file']][item['id']], item['route'])
                if plan.get('raw_questions') else [question(item)])

    training = [{'id': (identity({'document': item['id'], 'question': text})
                        if plan.get('raw_questions') else item['id']),
                 'route': item['route'], 'features': features(text)}
                for item in selected['training'] for text in variants(item)]
    model = expert_router.fit(training, embedding_root=features.root, tokenizer_root=plan['tokenizer_root'],
                              **{key: plan[key] for key in ('prototypes_per_route', 'iterations',
                                  'minimum_margin', 'maximum_distance')})
    if plan.get('classifier'):
        model = expert_router.fit_classifier(training, model, **plan['classifier'])
    (args.output / 'router.json').write_text(json.dumps(model, sort_keys=True) + '\n')
    measurements = []
    for item in selected['evaluation']:
        text = question(item)
        changed = re.sub('neuroshard 0\\.4\\.0', 'the NeuroShard network', text, flags=re.I)
        changed = re.sub('fictional luma directory', "Luma's directory", changed, flags=re.I)
        versions = [('original', text)] + ([('changed_phrase', changed)] if changed != text else [])
        if plan.get('raw_questions'):
            versions += [('raw_question', value) for value in variants(item)[1:]]
        for version, prompt in versions:
            observed = expert_router.select(model, features(prompt))
            measurements.append({'id': item['id'], 'version': version, 'expected': item['route'],
                                 'question_root': identity(prompt), **observed})
        if time.monotonic() - started > plan['resources']['maximum_seconds']:
            raise TimeoutError('CPU router study exceeded its frozen allowance')
    totals = defaultdict(lambda: {'correct': 0, 'count': 0})
    for row in measurements:
        group = totals[row['version'] + '/' + row['expected']]
        group['count'] += 1
        group['correct'] += row['route'] == row['expected']
    for group in totals.values():
        group['accuracy'] = group['correct'] / group['count']
    parent = totals['original/parent']
    checks = {
        'original_routes': all(totals['original/' + name]['accuracy'] >=
                              plan['gate']['original_accuracy_per_route_at_least'] for name in ('parent', 'directory', 'protocol')),
        'changed_phrases': all(totals['changed_phrase/' + name]['accuracy'] >=
                              plan['gate']['changed_phrase_accuracy_per_expert_at_least'] for name in ('directory', 'protocol')),
        'retained_parent': parent['count'] - parent['correct'] <= plan['gate']['parent_route_changes_at_most']}
    if plan.get('raw_questions'):
        checks['raw_questions'] = all(totals['raw_question/' + name]['accuracy'] >=
            plan['gate']['raw_question_accuracy_per_expert_at_least'] for name in ('directory', 'protocol'))
    result = {'format': plan['format'] + '/result', 'plan': identity(plan), 'selection': identity(selected),
              'router': identity(model), 'feature_profile': features.profile, 'checks': checks,
              'passed': all(checks.values()), 'groups': dict(totals), 'seconds': time.monotonic() - started,
              'scope': plan['scope'], 'new_llm_learning_claimed': False, 'gpus': 0}
    (args.output / 'predictions.jsonl').write_text(''.join(json.dumps(row) + '\n' for row in measurements))
    (args.output / 'result.json').write_text(json.dumps(result, indent=2, sort_keys=True) + '\n')
    print(json.dumps(result))


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest='command', required=True)
    for name in ('prepare', 'run'):
        item = sub.add_parser(name)
        item.add_argument('--inputs', type=Path, required=True)
        item.add_argument('--plan', type=Path, default=PLAN)
        item.add_argument('--selection', type=Path, default=SELECTION)
        if name == 'run':
            item.add_argument('--seed', type=Path, required=True)
            item.add_argument('--embedding', type=Path, required=True)
            item.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    (prepare if args.command == 'prepare' else run)(args)


if __name__ == '__main__':
    main()
