#!/usr/bin/env python3
"""Freeze adaptive shard inputs from prior tensor manifests and public data.

No model weights are loaded. The seed directory needs tokenizer assets and
rank-0.json / rank-1.json from the published persistent-shard seed inventory.
"""
import argparse
import copy
import json
import math
from pathlib import Path
import random
import shutil

from transformers import AutoTokenizer, LlamaConfig

from neuroshard.evolution import reference_data as data, grounded_tasks as tasks
from neuroshard.evolution.sharded import portable
from neuroshard.evolution.sharded.adaptive_job import implementation
from neuroshard.evolution.sharded.model import owner
from neuroshard.dataflow.collect import upstream_rows

ROOT = Path(__file__).resolve().parents[1]


def prepare(args):
    plan = json.loads(args.plan.read_bytes())
    args.home.mkdir(parents=True, exist_ok=False)
    tokenizer = AutoTokenizer.from_pretrained(args.seed, local_files_only=True, trust_remote_code=False)
    config = LlamaConfig.from_pretrained(args.seed, local_files_only=True)
    inventory = {}
    for rank in range(2):
        part = json.loads((args.seed/f'rank-{rank}.json').read_bytes())['tensors']
        if set(inventory) & set(part):
            raise ValueError('Duplicate seed ownership')
        inventory.update(part)
    if {n: s['shape'] for n, s in inventory.items()} != portable.shapes(config):
        raise ValueError('Incomplete seed inventory')
    seed_layouts = {}
    for count in (2, 3):
        boundaries = portable.layout(config, [24*1024**3]*count)
        hashes = []
        for rank in range(count):
            tensors = {n: s for n, s in inventory.items() if owner(n, boundaries) == rank}
            manifest = {'rank': rank, 'boundaries': boundaries, 'tensors': tensors,
                        'parameters': sum(math.prod(s['shape']) for s in tensors.values())}
            path = args.home/f'seed-{count}'/f'rank-{rank}.json'
            data.save(path, manifest)
            hashes.append(data.sha256(path))
        seed_layouts[data.identity(boundaries)] = hashes
    old_train = [json.loads(line) for line in args.replay.read_text().splitlines()]
    replay = [copy.deepcopy(r) for r in old_train if 'task' not in r][:plan['conversation_replay']]
    if len(replay) != plan['conversation_replay']:
        raise ValueError('Insufficient previously trained conversation replay')
    exclusion = data.ExclusionIndex()
    for row in old_train:
        exclusion.add(row['messages'])
    for row in replay:
        row['distill'] = True
    seen = {r['id'] for r in old_train}
    def make(seed, role, count):
        result = []
        for index in range(count):
            case = tasks.make_case(seed, role, index)
            identifier = tasks.task_identity(case)
            if identifier in seen:
                raise ValueError('Repeated task identity')
            seen.add(identifier)
            messages = [{'role': 'user', 'content': tasks.prompt(case)},
                        {'role': 'assistant', 'content': json.dumps(tasks.expected(case), separators=(',', ':'))}]
            result.append({'id': identifier, 'messages': messages, 'task': case, 'loss_weight': 8,
                           'distill': False, **data.conversation(tokenizer, messages, plan['max_length'])})
        return result
    train_a = make(plan['task_seeds'][0], 'train', plan['cohort_a_new_tasks'])+replay
    # Balanced schedules guarantee replay anchors in every update and use each
    # source document exactly once per cohort. No unseen window is called replay.
    def schedule(rows, seed):
        generator = random.Random(seed)
        groups = [[i for i, r in enumerate(rows) if bool(r['distill']) == anchor] for anchor in [False, True]]
        for group in groups:
            generator.shuffle(group)
            if len(group) % plan['cohort_steps']:
                raise ValueError('Each replay stratum must divide into complete steps')
        result = []
        for step in range(plan['cohort_steps']):
            batch = []
            for group in groups:
                width = len(group)//plan['cohort_steps']
                batch.extend(group[step*width:(step+1)*width])
            generator.shuffle(batch)
            if len(batch) != plan['training']['batch_documents']:
                raise ValueError('Incorrect effective batch size')
            result.append(batch)
        assert {i for b in result for i in b} == set(range(len(rows)))
        return result
    schedule_a = schedule(train_a, plan['training']['seed'])
    choices = list(range(plan['cohort_a_new_tasks']))
    random.Random(plan['training']['seed']).shuffle(choices)
    task_replay = [copy.deepcopy(train_a[i]) for i in choices[:plan['cohort_b_trained_replay']]]
    for row in task_replay:
        row['distill'] = True
    train_b = make(plan['task_seeds'][1], 'train', plan['cohort_b_new_tasks'])+task_replay+replay
    schedule_b = schedule(train_b, plan['training']['seed']+1)
    roles = {'train-a': train_a, 'train-b': train_b}
    for cohort, seed in zip(['a', 'b'], plan['task_seeds']):
        for role, count in [('dev', plan['development_cases_per_cohort']), ('test', plan['test_cases_per_cohort'])]:
            roles[role+'-'+cohort] = make(seed, role, count)
    # Previously exposed retention examples are development data only.
    roles['dev-retention'] = [json.loads(line) for line in args.development_retention.read_text().splitlines()][:64]
    for row in roles['dev-retention']:
        exclusion.add(row['messages'])
    if args.upstream_cache:
        (args.home/'upstream').symlink_to(args.upstream_cache.resolve(), target_is_directory=True)
    source = plan['fresh_retention_source']
    rows = upstream_rows(source, args.home, source['start'], source['scan'])
    try:
        fresh, report = data.prepare_role(rows, tokenizer, {**source, 'documents': plan['retention_cases']},
            exclusion, plan['max_length'], {k: v for k, v in source.items() if k not in ['start', 'scan']})
    finally:
        rows.close()
    roles['retention'] = fresh
    files = {}
    for role, rows in roles.items():
        path = args.home/(role+'.jsonl')
        with path.open('x') as stream:
            for row in rows:
                stream.write(json.dumps(row, separators=(',', ':'), ensure_ascii=False)+'\n')
        files[role] = {'file': path.name, 'sha256': data.sha256(path), 'count': len(rows),
                       'ids': [r['id'] for r in rows]}
    for a in plan['final_roles']:
        if set(files[a]['ids']) & (set(files['train-a']['ids']) | set(files['train-b']['ids'])):
            raise ValueError('Final set overlaps training')
    prepared = {'plan': plan, 'seed_layouts': seed_layouts, 'seed': json.loads((args.seed/'source.json').read_bytes()),
        'config_sha256': data.sha256(args.seed/'config.json'), 'tokenizer': data.tokenizer_identity(tokenizer),
        'sources': implementation(ROOT), 'roles': files, 'fresh_retention_report': report,
        'cohort_b_replay_ids': [r['id'] for r in task_replay],
        'schedule': [{'role': role, 'indices': indices} for role, schedule in [('train-a', schedule_a), ('train-b', schedule_b)]
                     for indices in schedule]}
    data.save(args.home/'prepared.json', prepared)
    print(json.dumps({'prepared': data.identity(prepared), 'counts': {k: len(v) for k, v in roles.items()}}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', type=Path, default=ROOT/'config/experiments/adaptive-shards.json')
    parser.add_argument('--home', type=Path, required=True)
    parser.add_argument('--seed', type=Path, required=True)
    parser.add_argument('--replay', type=Path, required=True)
    parser.add_argument('--development-retention', type=Path, required=True)
    parser.add_argument('--upstream-cache', type=Path)
    prepare(parser.parse_args())
