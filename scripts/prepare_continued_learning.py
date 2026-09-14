#!/usr/bin/env python3
"""Freeze continued-learning datasets against the committed parent artifacts.

No model weights are loaded. The seed directory needs the pinned tokenizer files.
Prior adaptive role files supply exclusion identities and replay windows.
"""
import argparse
import copy
import json
import random
import re
from pathlib import Path

from transformers import AutoTokenizer, LlamaConfig

from neuroshard.evolution import continued, grounded_tasks as tasks, reference_data as data
from neuroshard.dataflow.collect import upstream_rows

ROOT = Path(__file__).resolve().parents[1]


def write_role(home, role, rows):
    path = home / (role + '.jsonl')
    with path.open('x') as stream:
        for row in rows:
            stream.write(json.dumps(row, separators=(',', ':'), ensure_ascii=False) + '\n')
    return {'file': path.name, 'sha256': data.sha256(path), 'count': len(rows),
            'ids': [row['id'] for row in rows]}


def make_tasks(tokenizer, plan, seed, role, count, seen, exclusion, family_cycle=None):
    result = []
    cycle = plan['method']['new_task_family_cycle'] if family_cycle else None
    for index in range(count):
        family = cycle[index % len(cycle)] if cycle else None
        case = tasks.make_case(seed, role, index, family=family)
        identifier = tasks.task_identity(case)
        if identifier in seen:
            raise ValueError('Repeated task identity')
        seen.add(identifier)
        messages = [{'role': 'user', 'content': tasks.prompt(case)},
                    {'role': 'assistant', 'content': json.dumps(tasks.expected(case), separators=(',', ':'))}]
        if not exclusion.add(messages):
            raise ValueError('Generated prompt overlaps previously exposed or prepared content')
        weight = plan['method']['total_task_weight'] if case['family'] == 'total' else plan['method']['new_task_weight']
        result.append({'id': identifier, 'messages': messages, 'task': case, 'loss_weight': weight,
                      'distill': False, **data.conversation(tokenizer, messages, plan['max_length'])})
    return result


def schedule(rows, seed, plan):
    generator = random.Random(seed)
    groups = [[i for i, row in enumerate(rows) if bool(row['distill']) == anchor] for anchor in (False, True)]
    for group in groups:
        generator.shuffle(group)
        if len(group) % plan['additional_steps']:
            raise ValueError('Each replay stratum must divide into complete steps')
    result = []
    for step in range(plan['additional_steps']):
        batch = []
        for group in groups:
            width = len(group) // plan['additional_steps']
            batch.extend(group[step * width:(step + 1) * width])
        generator.shuffle(batch)
        if len(batch) != plan['training']['batch_documents']:
            raise ValueError('Incorrect effective batch size')
        result.append(batch)
    if {i for batch in result for i in batch} != set(range(len(rows))):
        raise ValueError('Schedule must use every training document once')
    return result


def prepare(args):
    plan = continued.load(args.plan)
    if plan['status'] != 'plan-frozen':
        raise ValueError('Prepare against the frozen plan bytes only')
    if re.fullmatch('[0-9a-f]{40}', args.plan_commit) is None:
        raise ValueError('Pin a complete Git commit')
    if continued.git_bytes(args.plan_commit, args.plan) != args.plan.read_bytes():
        raise ValueError('Preparation requires the exact Git-committed plan')
    args.home.mkdir(parents=True, exist_ok=False)
    for name, expected in plan['tokenizer_files'].items():
        if data.sha256(args.seed / name) != expected:
            raise ValueError(f'Tokenizer file {name} differs from the frozen artifact')
    if data.sha256(args.seed / 'config.json') != plan['config_sha256']:
        raise ValueError('Architecture config differs from the frozen artifact')
    tokenizer = AutoTokenizer.from_pretrained(args.seed, local_files_only=True, trust_remote_code=False)
    if data.tokenizer_identity(tokenizer) != plan['tokenizer']:
        raise ValueError('Tokenizer identity differs from the frozen artifact')
    LlamaConfig.from_pretrained(args.seed, local_files_only=True)
    prior_ids = continued.prior_role_ids(plan)
    prior_roles = {role: data.read_records(args.prior_train.parent / (role + '.jsonl'), checksum)
                   for role, checksum in plan['prior_exclusion']['role_sha256'].items()}
    prior_train = data.read_records(args.prior_train, plan['prior_exclusion']['role_sha256']['train-a'])
    exclusion = continued.ContentExclusion()
    for rows in prior_roles.values():
        for row in rows:
            exclusion.add(row['messages'])
    conversations = [copy.deepcopy(row) for row in prior_train if 'task' not in row]
    trained = [copy.deepcopy(row) for row in prior_train if 'task' in row]
    random.Random(plan['replay_sample_seed']).shuffle(conversations)
    random.Random(plan['replay_sample_seed'] + 1).shuffle(trained)
    replay = conversations[:plan['conversation_replay']]
    task_replay = trained[:plan['trained_replay']]
    if len(replay) != plan['conversation_replay'] or len(task_replay) != plan['trained_replay']:
        raise ValueError('Insufficient previously trained replay windows')
    for row in replay + task_replay:
        row['distill'] = True
    seen = set(prior_ids)
    train = make_tasks(tokenizer, plan, plan['task_seed'], 'train', plan['new_tasks'], seen, exclusion, family_cycle=True)
    train.extend(task_replay)
    train.extend(replay)
    roles = {
        'train': train,
        'dev-new': make_tasks(tokenizer, plan, plan['task_seed'], 'dev', plan['development_cases'], seen, exclusion),
        'dev-prior': make_tasks(tokenizer, plan, plan['prior_probe_seed'], 'dev', plan['development_cases'], seen, exclusion),
        'test-new': make_tasks(tokenizer, plan, plan['task_seed'], 'test', plan['test_new_cases'], seen, exclusion),
        'test-prior': make_tasks(tokenizer, plan, plan['prior_probe_seed'], 'test', plan['test_prior_cases'], seen, exclusion),
    }
    if args.upstream_cache:
        (args.home / 'upstream').symlink_to(args.upstream_cache.resolve(), target_is_directory=True)
    for role, spec in (('dev-retention', plan['development_retention_source']),
                       ('retention', plan['fresh_retention_source'])):
        rows = upstream_rows(spec, args.home, spec['start'], spec['scan'])
        try:
            documents, report = data.prepare_role(
                rows, tokenizer, {**spec, 'documents': plan['development_cases'] if role == 'dev-retention' else plan['retention_cases']},
                exclusion, plan['max_length'], {k: v for k, v in spec.items() if k not in ['start', 'scan']})
        finally:
            rows.close()
        roles[role] = documents
        if role == 'retention':
            retention_report = report
    files = {role: write_role(args.home, role, rows) for role, rows in roles.items()}
    continued.overlap_ids(*(spec['ids'] for spec in files.values()))
    if any(set(spec['ids']) & prior_ids for role, spec in files.items() if role != 'train'):
        raise ValueError('Fresh evaluation overlaps previously exposed identities')
    for role in plan['final_roles']:
        if set(files[role]['ids']) & set(files['train']['ids']):
            raise ValueError('Final set overlaps training')
    prepared = {
        'format': continued.PREPARED_FORMAT,
        'plan': {k: plan[k] for k in plan if k != 'status'},
        'plan_digest': data.sha256(args.plan),
        'plan_commit': args.plan_commit,
        'implementation_digest': continued.implementation_digest(),
        'parent_checkpoint': plan['parent']['checkpoint'],
        'parent_state_root': plan['parent']['state_root'],
        'reference_checkpoint': plan['reference']['checkpoint'],
        'tokenizer': plan['tokenizer'],
        'config_sha256': plan['config_sha256'],
        'runtime': {key: plan['runtime'][key] for key in continued.RUNTIME_KEYS},
        'sources': {name: data.sha256(ROOT / name) for name in continued.SOURCE_PATHS},
        'roles': files,
        'fresh_retention_report': retention_report,
        'trained_replay_ids': [row['id'] for row in task_replay],
        'conversation_replay_ids': [row['id'] for row in replay],
        'schedule': [{'role': 'train', 'indices': indices} for indices in schedule(train, plan['training']['seed'], plan)],
    }
    continued.validate_prepared(prepared, plan)
    data.save(args.home / 'prepared.json', prepared)
    print(json.dumps({'prepared': data.identity(prepared), 'counts': {k: len(v) for k, v in roles.items()}}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', type=Path, default=ROOT / 'config/experiments/continued-learning.json')
    parser.add_argument('--home', type=Path, required=True)
    parser.add_argument('--seed', type=Path, required=True)
    parser.add_argument('--prior-train', type=Path, required=True)
    parser.add_argument('--plan-commit', required=True)
    parser.add_argument('--upstream-cache', type=Path)
    prepare(parser.parse_args())
