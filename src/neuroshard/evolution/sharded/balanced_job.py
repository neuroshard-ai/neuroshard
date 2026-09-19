"""Run answer-balanced continuation using existing owned-shard numerical kernels."""
import argparse
from datetime import timedelta
import gc
import json
import os
from pathlib import Path
import random
import re
import shutil
import time

import numpy as np
import torch
import torch.distributed as dist

from .. import balanced as contract, consolidation, continued, grounded_tasks as tasks
from .. import reasoned, reference, reference_data as data
from . import guarded, portable
from .consolidation_job import tokenizer_for, clean, emit
from .model import Partition
from .training import score, generate
from .wire import Wire


def prepare(args, plan):
    import pyarrow.parquet as pq
    if (plan['status'] != 'plan-frozen' or re.fullmatch(r'[0-9a-f]{40}', args.plan_commit or '') is None
            or contract.git_bytes(args.plan_commit, args.plan) != args.plan.read_bytes()):
        raise ValueError('Prepare only the original Git-committed balanced plan')
    if any(contract.git_bytes(args.plan_commit, contract.ROOT / name) != (contract.ROOT / name).read_bytes()
           for name in contract.SOURCES):
        raise ValueError('Balanced preparation source differs from the frozen commit')
    if data.sha256(args.source_parquet) != plan['conversation_source']['sha256']:
        raise ValueError('Pinned conversation parquet differs')
    tokenizer = tokenizer_for(plan, args.seed)
    args.home.mkdir(parents=True, exist_ok=False)
    exclusion = continued.ContentExclusion()
    all_prior_ids, excluded, old_training = set(), {}, None
    for label, name in plan['exclusions'].items():
        if label == 'policy':
            continue
        directory = getattr(args, label + '_inputs')
        if directory is None:
            raise ValueError('Supply every preceding input directory for exclusion')
        filename = contract.ROOT / name
        raw = filename.read_bytes()
        if contract.git_bytes('HEAD', filename) != raw:
            raise ValueError('Earlier exclusion artifact is not committed')
        old = json.loads(raw)
        excluded[label] = data.identity(old)
        for role, spec in old['roles'].items():
            rows = data.read_records(directory / spec['file'], spec['sha256'])
            if [row['id'] for row in rows] != spec['ids']:
                raise ValueError('Earlier exclusion input differs from its manifest')
            for row in rows:
                all_prior_ids.add(row['id'])
                exclusion.add(row['messages'])
            if label == 'reasoned' and role == 'train':
                old_training = rows
    old = contract.replay_source(plan)
    if old_training is None or [row['id'] for row in old_training] != old['roles']['train']['ids']:
        raise ValueError('Missing complete trained replay source')
    replay_indices = contract.choose_replay(plan, old_training)
    shutil.copyfile(args.reasoned_inputs / old['roles']['train']['file'], args.home / 'trained-source.jsonl')
    roles = {'train': []}
    fresh_ids = set()

    def fresh(value, messages):
        identifier = tasks.task_identity(value)
        if identifier in all_prior_ids or identifier in fresh_ids or not exclusion.add(messages):
            raise ValueError('Fresh balanced input overlaps previous or current content')
        fresh_ids.add(identifier)
        return {'id': identifier, 'task': value, 'messages': messages,
                **data.conversation(tokenizer, messages, plan['max_length'])}

    for kind in contract.STRATA:
        count = plan['strata'][kind]
        if kind.startswith('new_'):
            for index in range(count):
                value = contract.case(plan, 'train', index, kind)
                row = fresh(value, reasoned.messages(value, False))
                roles['train'].append({**row, 'stratum': kind, 'distill': True,
                                       'loss_weight': 1.0 / row['targets']})
        else:
            for index in replay_indices[kind]:
                row = old_training[index]
                roles['train'].append({**row, 'stratum': kind, 'source_index': index,
                                       'loss_weight': 1.0 / row['targets'], 'distill': True})
    for role in ('dev-prior', 'dev-new', 'test-new', 'test-prior'):
        roles[role] = []
        for index in range(plan['roles'][role]):
            value = contract.case(plan, role, index)
            roles[role].append(fresh(value, reasoned.messages(value, role in plan['method']['reasoning_roles'])))
    table = pq.read_table(args.source_parquet, columns=['messages'])
    if table.num_rows != plan['conversation_source']['rows']:
        raise ValueError('Pinned conversation row count differs')
    indices = list(range(table.num_rows))
    random.Random(plan['seeds']['conversation_permutation']).shuffle(indices)
    cursor, reports = 0, {}
    for role in ('dev-retention', 'retention'):
        rows, rejected, start = [], {'invalid': 0, 'too_long': 0, 'duplicate': 0}, cursor
        while len(rows) < plan['roles'][role] and cursor < len(indices):
            index = indices[cursor]
            cursor += 1
            messages = table['messages'][index].as_py()
            try:
                encoded = data.conversation(tokenizer, messages, plan['max_length'])
            except OverflowError:
                rejected['too_long'] += 1
                continue
            except (ValueError, TypeError, UnicodeError):
                rejected['invalid'] += 1
                continue
            if not exclusion.add(messages):
                rejected['duplicate'] += 1
                continue
            row = {'messages': messages, 'source': plan['conversation_source'], 'row': index, **encoded}
            identifier = data.identity(row)
            if identifier in all_prior_ids or identifier in fresh_ids:
                raise ValueError('Conversation identity overlaps earlier input')
            fresh_ids.add(identifier)
            rows.append({'id': identifier, **row})
        if len(rows) != plan['roles'][role]:
            raise ValueError('Unused conversation source cannot fill the frozen quota')
        roles[role] = rows
        reports[role] = {'scanned': cursor - start, 'rejected': rejected, 'documents': len(rows)}
    files = {}
    for role, rows in roles.items():
        filename = args.home / (role + '.jsonl')
        with filename.open('x') as output:
            for row in rows:
                output.write(json.dumps(row, separators=(',', ':'), ensure_ascii=False) + '\n')
        files[role] = {'file': filename.name, 'sha256': data.sha256(filename),
                       'count': len(rows), 'ids': [row['id'] for row in rows]}
    prepared = {'format': contract.PREPARED, 'plan_commit': args.plan_commit,
        'plan_digest': data.sha256(args.plan), 'sources': contract.sources(), 'roles': files,
        'excluded_prepared': excluded, 'conversation_selection': reports,
        'schedule': contract.schedule(plan), 'replay_indices': replay_indices,
        'replay_source': {'file': 'trained-source.jsonl', 'sha256': old['roles']['train']['sha256'],
                          'prepared': data.identity(old)}}
    for role in roles:
        contract.read_role(plan, args.home, prepared, role, tokenizer)
    data.save(args.home / 'prepared.json', prepared)
    emit('prepared', root=data.identity(prepared), counts={role: len(rows) for role, rows in roles.items()})


def run_gpu(args, plan, prepared):
    contract.committed_prepared(plan, prepared)
    selection = json.loads(args.selection.read_bytes()) if args.selection else None
    if args.command == 'train':
        if plan['status'] != 'prepared-committed' or contract.path(plan, 'selection').exists():
            raise ValueError('Train only before the balanced final selection')
    elif selection is None or not contract.committed_selection(plan, prepared, selection):
        raise ValueError('Balanced evaluation requires the committed actual candidate')
    parent = json.loads(args.parent.read_bytes())
    baseline = json.loads(args.baseline_checkpoint.read_bytes())
    if data.identity(parent) != plan['parent']['checkpoint'] or data.identity(baseline) != plan['baseline']['checkpoint']:
        raise ValueError('Balanced input checkpoints differ from the freeze')
    config = portable.validate(parent)
    portable.validate(baseline)
    if (parent['config'] != baseline['config'] or parent['boundaries'] != plan['boundaries']
            or baseline['boundaries'] != plan['boundaries'] or parent['step'] != plan['parent']['step']):
        raise ValueError('Balanced input architecture, layout or optimizer cursor differs')
    target = json.loads(args.resume.read_bytes()) if args.resume else (parent if args.command == 'train' else baseline)
    portable.validate(target)
    if args.command == 'train' and data.identity(target) != data.identity(parent):
        if (target['job'] != contract.job(prepared) or target['step'] != plan['checkpoints'][0]
                or target['parent'] != data.identity(parent)):
            raise ValueError('Resume only the declared balanced intermediate state')
    if args.command == 'evaluate' and data.identity(target) not in (selection['candidate'], selection['baseline']):
        raise ValueError('Balanced finals cannot substitute a checkpoint')
    tokenizer = tokenizer_for(plan, args.seed)
    runtime = reference.configure('cuda', plan['threads'])
    runtime['allocator'] = os.environ['PYTORCH_CUDA_ALLOC_CONF']
    if {key: runtime[key] for key in continued.RUNTIME_KEYS} != plan['runtime']:
        raise ValueError('Balanced numerical runtime differs')
    roles = ('train', *contract.DEVELOPMENT) if args.command == 'train' else contract.FINALS
    cache = {role: contract.read_role(plan, args.prepared.parent, prepared, role, tokenizer) for role in roles}
    rank, world = int(os.environ['RANK']), int(os.environ['WORLD_SIZE'])
    if world != len(plan['boundaries']) - 1 or not 0 <= rank < world:
        raise ValueError('Require every declared balanced shard owner')
    torch.manual_seed(plan['seeds']['training_rng'] + rank)
    random.seed(plan['seeds']['training_rng'] + rank)
    np.random.seed(plan['seeds']['training_rng'] + rank)
    config._attn_implementation = 'sdpa'
    args.home.mkdir(parents=True, exist_ok=False)
    shard = Partition(config, plan['boundaries'], rank, 'cuda', plan['parameter_limit'])
    teacher = benchmark = optimizer = None
    if args.command == 'train':
        teacher = Partition(config, plan['boundaries'], rank, 'cuda', plan['parameter_limit'])
        portable.load(args.parent.parent, teacher, None, parent, parent['job'], restore_rng=False)
        teacher.eval().requires_grad_(False)
        benchmark = Partition(config, plan['boundaries'], rank, 'cuda', plan['parameter_limit'])
        portable.load(args.baseline_checkpoint.parent, benchmark, None, baseline, baseline['job'], restore_rng=False)
        benchmark.eval().requires_grad_(False)
        optimizer = reference.optimizer_for(shard, plan['training'])
    target_home = args.resume.parent if args.resume else (
        args.parent.parent if args.command == 'train' else args.baseline_checkpoint.parent)
    # All owned file loading precedes the network group and its operation timeout.
    births = portable.load(target_home, shard, optimizer, target, target['job'], restore_rng=optimizer is not None)
    dist.init_process_group('gloo', timeout=timedelta(seconds=180))
    wire = Wire(rank, world)
    started = time.monotonic()
    try:
        agreement = {'prepared': data.identity(prepared), 'parent': data.identity(parent),
                     'baseline': data.identity(baseline), 'target': data.identity(target), 'command': args.command}
        if any(value != agreement for value in wire.exchange(agreement)):
            raise ValueError('Balanced workers disagree on the complete computation')
        data.save(args.home / 'started.json', {**agreement, 'runtime': runtime, 'rank': rank,
                  'resident_parameters': shard.resident_parameters, 'tokens_issued': 0})

        def records(role):
            return cache[role]

        def evaluate(model, role):
            if time.monotonic() - started > plan['max_seconds']:
                raise TimeoutError('Balanced evaluation deadline')
            rows = records(role)
            result = {'losses': score(model, wire, rows), 'answers': []}
            if role in ('dev-new', 'dev-prior', 'test-new', 'test-prior'):
                for row in rows:
                    if time.monotonic() - started > plan['max_seconds']:
                        raise TimeoutError('Balanced generation deadline')
                    prompt = tokenizer.apply_chat_template(row['messages'][:-1], tokenize=True, add_generation_prompt=True)
                    began = time.monotonic()
                    ids = generate(model, wire, prompt, plan['generation_tokens'], tokenizer.eos_token_id)
                    text = tokenizer.decode(ids, skip_special_tokens=True)
                    result['answers'].append({'id': row['id'], 'output_ids': ids, 'text': text,
                        'check': consolidation.check_answer(plan, row['task'], text, role),
                        'seconds': time.monotonic() - began})
                    if len(result['answers']) % 32 == 0:
                        emit('generated', rank=rank, role=role, cases=len(result['answers']))
            if any(other != clean(result) for other in wire.exchange(clean(result))):
                raise ValueError('Balanced workers disagree on answers or response losses')
            return result

        if args.command == 'evaluate':
            outcomes = {role: evaluate(shard, role) for role in contract.FINALS}
            data.save(args.home / 'evaluation.json', {'checkpoint': data.identity(target),
                'prepared': data.identity(prepared), 'rank': rank, 'outcomes': outcomes,
                'seconds': time.monotonic() - started, 'peak_cuda_bytes': torch.cuda.max_memory_allocated()})
            return
        baseline_dev = {role: clean(evaluate(benchmark, role)) for role in contract.DEVELOPMENT}
        del benchmark
        benchmark = None
        gc.collect()
        parent_new = clean(evaluate(teacher, 'dev-new'))
        data.save(args.home / 'baselines.json', {'baseline': baseline_dev, 'training_parent_new': parent_new})
        train = records('train')
        checkpoints = [target] if target['step'] > parent['step'] else []
        previous = data.identity(target)
        for step in range(target['step'], plan['selectable_checkpoint']):
            if time.monotonic() - started > plan['max_seconds']:
                raise TimeoutError('Balanced training deadline')
            index = step - plan['parent']['step']
            assignment = prepared['schedule'][index]
            report = guarded.train_step(shard, teacher, optimizer, wire,
                [train[i] for i in assignment['indices']], plan['training'], index, plan['microbatch'],
                kl_strength=plan['reference']['kl_strength'], margin_strength=plan['reference']['margin_strength'],
                margin_min=plan['reference']['margin_min'], margin_max=plan['reference']['margin_max'])
            with (args.home / 'steps.jsonl').open('a') as output:
                output.write(json.dumps(report) + '\n')
                output.flush()
            emit('step', rank=rank, **report)
            if step + 1 in plan['checkpoints']:
                common = portable.commit(args.home, shard, optimizer, wire, contract.job(prepared), step + 1, previous, births)
                previous = data.identity(common)
                checkpoints.append(common)
                emit('checkpoint', rank=rank, step=step + 1, root=previous, state_root=common['state_root'])
        candidate_dev = {role: clean(evaluate(shard, role)) for role in contract.DEVELOPMENT}
        decision = contract.development(plan, prepared, baseline_dev, parent_new, candidate_dev, tokenizer)
        selected = {'format': contract.SELECTION, 'prepared': data.identity(prepared),
            'parent': data.identity(parent), 'baseline': data.identity(baseline), 'input_checkpoint': parent,
            'checkpoints': checkpoints, 'candidate': previous, 'baseline_development': baseline_dev,
            'training_parent_new': parent_new, 'candidate_development': candidate_dev, 'decision': decision}
        data.save(args.home / 'development.json', selected)
        if not decision['passed']:
            data.save(args.home / 'aborted.json', {'reason': 'Balanced terminal development gate failed', 'decision': decision})
            raise ValueError('Balanced development failed; keep final evaluation closed')
        contract.validate_selection(plan, prepared, selected, tokenizer)
        data.save(args.home / 'selection-proposal.json', selected)
        data.save(args.home / 'result.json', {'passed': True, 'candidate': previous, 'rank': rank,
            'seconds': time.monotonic() - started, 'peak_cuda_bytes': torch.cuda.max_memory_allocated(),
            'new_gradient_updates': plan['training']['steps'], 'tokens_issued': 0})
    finally:
        dist.destroy_process_group()


def score_final(args, plan, prepared):
    contract.committed_prepared(plan, prepared)
    selected = json.loads(args.selection.read_bytes())
    contract.committed_selection(plan, prepared, selected)
    tokenizer = tokenizer_for(plan, args.seed)
    contract.validate_selection(plan, prepared, selected, tokenizer)
    baseline, candidate = [json.loads(filename.read_bytes()) for filename in (args.baseline, args.candidate)]
    for report, checkpoint in ((baseline, selected['baseline']), (candidate, selected['candidate'])):
        if (report['checkpoint'] != checkpoint or report['prepared'] != data.identity(prepared)
                or set(report['outcomes']) != set(contract.FINALS)):
            raise ValueError('Balanced final report differs from the selected experiment')
    outcomes = {}
    for role in ('test-new', 'test-prior'):
        before, after = baseline['outcomes'][role], candidate['outcomes'][role]
        summary = contract.generation(plan, prepared, role, before['answers'], after['answers'], tokenizer)
        passed = all(v['candidate_correct'] >= v['baseline_correct'] for v in summary['families'].values())
        for family in summary['families'].values():
            family['passed'] = family['candidate_correct'] >= family['baseline_correct']
        if role == 'test-new':
            passed = (passed and summary['wins'] - summary['losses'] >= plan['quality_gate']['min_net_gain']
                and summary['one_sided_p'] < plan['quality_gate']['max_one_sided_p']
                and summary['families']['total']['candidate_correct'] >= plan['quality_gate']['min_correct_totals'])
        summary['passed'] = passed
        outcomes[role] = {'generation': summary, 'passed': passed,
            'loss': consolidation.retention(plan, prepared, role, before['losses'], after['losses'])}
    retention = consolidation.retention(plan, prepared, 'retention', baseline['outcomes']['retention']['losses'],
                                        candidate['outcomes']['retention']['losses'])
    retention['passed'] = retention['upper'] <= plan['quality_gate']['retention_upper_at_most_nats']
    outcomes['retention'] = retention
    result = {'format': contract.FORMAT + '/quality', 'prepared': data.identity(prepared),
        'selection': data.identity(selected), 'baseline': selected['baseline'], 'candidate': selected['candidate'],
        'outcomes': outcomes, 'passed': all(value['passed'] for value in outcomes.values()),
        'tokens_issued': 0, 'serving_promoted': False,
        'scope': 'Answer-balanced training across fixed model shards; fresh public task records and conversation retention, not general assistant quality or useful growth'}
    data.save(args.output, result)
    emit('quality', passed=result['passed'])


def main(argv, root):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['prepare', 'train', 'evaluate', 'score'])
    parser.add_argument('--plan', type=Path, default=root / contract.PLAN)
    parser.add_argument('--seed', type=Path, required=True)
    for name in ('home', 'prepared', 'parent', 'baseline-checkpoint', 'resume', 'selection', 'source-parquet',
                 'adaptive-inputs', 'continued-inputs', 'reasoned-inputs', 'consolidated-inputs', 'baseline', 'candidate', 'output'):
        parser.add_argument('--' + name, type=Path)
    parser.add_argument('--plan-commit')
    args = parser.parse_args(argv)
    plan = contract.load(args.plan)
    required = {'prepare': ('home', 'source_parquet', 'plan_commit'),
                'train': ('home', 'prepared', 'parent', 'baseline_checkpoint'),
                'evaluate': ('home', 'prepared', 'parent', 'baseline_checkpoint', 'selection'),
                'score': ('prepared', 'selection', 'baseline', 'candidate', 'output')}
    if any(getattr(args, key) is None for key in required[args.command]):
        raise ValueError('Missing balanced inputs for ' + args.command)
    if args.command == 'prepare':
        prepare(args, plan)
    else:
        prepared = json.loads(args.prepared.read_bytes())
        if args.command == 'score':
            score_final(args, plan, prepared)
        else:
            run_gpu(args, plan, prepared)
