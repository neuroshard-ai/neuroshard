"""Prepare, screen and score fixed-size weight consolidation through the existing runner."""
import argparse
from datetime import timedelta
import json
from pathlib import Path
import os
import random
import re
import time

import numpy as np
import torch
import torch.distributed as dist
from transformers import AutoTokenizer, LlamaConfig

from .. import consolidation as contract, continued, grounded_tasks as tasks
from .. import reasoned, reference, reference_data as data
from . import portable
from .model import Partition
from .training import score, generate
from .wire import Wire


def emit(event, **values):
    print(json.dumps({'event': event, 'time': time.time(), **values}), flush=True)


def tokenizer_for(plan, seed):
    for name, digest in plan['tokenizer_files'].items():
        if data.sha256(seed / name) != digest:
            raise ValueError('Tokenizer asset differs from the frozen plan')
    if data.sha256(seed / 'config.json') != plan['config_sha256']:
        raise ValueError('Architecture asset differs from the frozen plan')
    tokenizer = AutoTokenizer.from_pretrained(seed, local_files_only=True, trust_remote_code=False)
    if data.tokenizer_identity(tokenizer) != plan['tokenizer']:
        raise ValueError('Tokenizer identity differs')
    return tokenizer


def prepare(args, plan):
    import pyarrow.parquet as pq
    if (plan['status'] != 'plan-frozen' or not args.plan_commit or not args.source_parquet
            or re.fullmatch(r'[0-9a-f]{40}', args.plan_commit) is None):
        raise ValueError('Preparation requires the original committed plan and pinned public parquet')
    if contract.git_bytes(args.plan_commit, args.plan) != args.plan.read_bytes():
        raise ValueError('Preparation must use the exact committed plan bytes')
    if any(contract.git_bytes(args.plan_commit, contract.ROOT / name) != (contract.ROOT / name).read_bytes()
           for name in contract.SOURCES):
        raise ValueError('Preparation source differs from the frozen commit')
    if data.sha256(args.source_parquet) != plan['conversation_source']['sha256']:
        raise ValueError('Public conversation parquet differs from its pinned SHA-256')
    tokenizer = tokenizer_for(plan, args.seed)
    args.home.mkdir(parents=True, exist_ok=False)
    exclusion = continued.ContentExclusion()
    prior_ids, excluded_prepared = set(), {}
    for label in ('adaptive', 'continued', 'reasoned'):
        directory = getattr(args, label + '_inputs')
        if directory is None:
            raise ValueError('Supply every earlier experiment role directory for exclusion')
        filename = contract.ROOT / plan['exclusions'][label]
        raw = filename.read_bytes()
        if contract.git_bytes('HEAD', filename) != raw:
            raise ValueError('Prior exclusion manifests must be committed')
        old = json.loads(raw)
        excluded_prepared[label] = data.identity(old)
        for role, spec in old['roles'].items():
            records = data.read_records(directory / spec['file'], spec['sha256'])
            if [row['id'] for row in records] != spec['ids']:
                raise ValueError('Prior exclusion records differ from their manifest')
            for record in records:
                prior_ids.add(record['id'])
                exclusion.add(record['messages'])
    roles = {}
    for role in ('dev-prior', 'dev-new', 'test-new', 'test-prior'):
        seed = plan['seeds']['new_tasks' if role.endswith('new') else 'prior_tasks']
        split = 'dev' if role.startswith('dev-') else 'test'
        records = []
        for index in range(plan['roles'][role]):
            case = tasks.make_case(seed, split, index)
            identifier = tasks.task_identity(case)
            messages = reasoned.messages(case, role in plan['method']['reasoning_roles'])
            if identifier in prior_ids or not exclusion.add(messages):
                raise ValueError('Generated evaluation overlaps exposed content')
            prior_ids.add(identifier)
            records.append({'id': identifier, 'task': case, 'messages': messages,
                            **data.conversation(tokenizer, messages, plan['max_length'])})
        roles[role] = records
    table = pq.read_table(args.source_parquet, columns=['messages'])
    if table.num_rows != plan['conversation_source']['rows']:
        raise ValueError('Pinned conversation source row count differs')
    indices = list(range(table.num_rows))
    random.Random(plan['seeds']['conversation_permutation']).shuffle(indices)
    cursor, reports = 0, {}
    for role in ('dev-retention', 'retention'):
        records, rejected = [], {'invalid': 0, 'too_long': 0, 'duplicate': 0}
        start = cursor
        while len(records) < plan['roles'][role] and cursor < len(indices):
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
            record = {'messages': messages, 'source': plan['conversation_source'], 'row': index, **encoded}
            identifier = data.identity(record)
            if identifier in prior_ids:
                raise ValueError('Conversation role repeats a previous identity')
            prior_ids.add(identifier)
            records.append({'id': identifier, **record})
        if len(records) != plan['roles'][role]:
            raise ValueError('Unused public conversations cannot fill the frozen quota')
        roles[role] = records
        reports[role] = {'scanned': cursor - start, 'rejected': rejected, 'documents': len(records)}
    files = {}
    for role, records in roles.items():
        filename = args.home / (role + '.jsonl')
        with filename.open('x') as output:
            for record in records:
                output.write(json.dumps(record, separators=(',', ':'), ensure_ascii=False) + '\n')
        files[role] = {'file': filename.name, 'sha256': data.sha256(filename),
                       'count': len(records), 'ids': [row['id'] for row in records]}
    prepared = {'format': contract.PREPARED, 'plan_commit': args.plan_commit,
                'plan_digest': data.sha256(args.plan), 'sources': contract.sources(), 'roles': files,
                'excluded_prepared': excluded_prepared, 'conversation_selection': reports}
    for role in roles:
        contract.read_role(plan, args.home, prepared, role, tokenizer)
    data.save(args.home / 'prepared.json', prepared)
    emit('prepared', root=data.identity(prepared), counts={role: len(rows) for role, rows in roles.items()})


def blend_(shard, parent, alpha):
    """The only weight mutation: two declared FP32 tensor operations, no optimizer step."""
    if type(alpha) is not float or not 0 < alpha < 1:
        raise ValueError('Consolidation coefficient must be strictly between zero and one')
    destination, source = dict(shard.named_owned_parameters()), dict(parent.named_owned_parameters())
    if set(destination) != set(source):
        raise ValueError('Consolidation requires identical owned parameter coverage')
    for name, parameter in destination.items():
        old = source[name]
        if parameter.dtype != torch.float32 or old.dtype != torch.float32 or parameter.shape != old.shape:
            raise ValueError('Consolidation requires matching FP32 parameters')
        if parameter.device != old.device or parameter.data_ptr() == old.data_ptr():
            raise ValueError('Consolidation requires separate parent storage on the same device')
    with torch.no_grad():
        for name, parameter in destination.items():
            parameter.mul_(alpha)
            parameter.add_(source[name], alpha=1 - alpha)


def clean(outcome):
    return {'losses': outcome['losses'],
            'answers': [{k: v for k, v in answer.items() if k != 'seconds'} for answer in outcome['answers']]}


def run_gpu(args, plan, prepared):
    contract.committed_prepared(plan, prepared)
    selection = json.loads(args.selection.read_bytes()) if args.selection else None
    if args.command == 'screen':
        if plan['status'] != 'prepared-committed' or contract.path(plan, 'selection').exists():
            raise ValueError('Screen only the committed preparation before any final selection')
    elif selection is None or not contract.committed_selection(plan, prepared, selection):
        raise ValueError('Final evaluation requires a committed selected endpoint')
    parent = json.loads(args.parent.read_bytes())
    fast = json.loads(args.fast.read_bytes())
    if data.identity(parent) != plan['parent']['checkpoint'] or data.identity(fast) != plan['fast']['checkpoint']:
        raise ValueError('Consolidation inputs differ from the frozen checkpoints')
    config = portable.validate(parent)
    portable.validate(fast)
    if (fast['config'] != parent['config'] or fast['boundaries'] != parent['boundaries']
            or parent['boundaries'] != plan['boundaries'] or fast['step'] != plan['fast']['step']):
        raise ValueError('Consolidation cannot replace architecture, shard layout or fast cursor')
    target = json.loads(args.resume.read_bytes()) if args.resume else parent
    if args.command == 'evaluate' and data.identity(target) not in (selection['candidate'], selection['parent']):
        raise ValueError('Final evaluation cannot substitute a different checkpoint')
    tokenizer = tokenizer_for(plan, args.seed)
    runtime = reference.configure('cuda', plan['threads'])
    runtime['allocator'] = os.environ['PYTORCH_CUDA_ALLOC_CONF']
    if {key: runtime[key] for key in continued.RUNTIME_KEYS} != plan['runtime']:
        raise ValueError('Numerical runtime differs from the frozen profile')
    rank, world = int(os.environ['RANK']), int(os.environ['WORLD_SIZE'])
    if world != len(plan['boundaries']) - 1 or not 0 <= rank < world:
        raise ValueError('Require all declared shard owners')
    torch.manual_seed(plan['seeds']['training_rng'] + rank)
    np.random.seed(plan['seeds']['training_rng'] + rank)
    random.seed(plan['seeds']['training_rng'] + rank)
    config._attn_implementation = 'sdpa'
    if portable.configuration(config) != portable.configuration(LlamaConfig.from_pretrained(args.seed, local_files_only=True)):
        raise ValueError('Tokenizer seed config differs from the checkpoint architecture')
    args.home.mkdir(parents=True, exist_ok=False)
    shard = Partition(config, plan['boundaries'], rank, 'cuda', plan['parameter_limit'])
    teacher = None
    optimizer = None
    if args.command == 'screen':
        teacher = Partition(config, plan['boundaries'], rank, 'cuda', plan['parameter_limit'])
        portable.load(args.parent.parent, teacher, None, parent, parent['job'], restore_rng=False)
        teacher.eval().requires_grad_(False)
        optimizer = reference.optimizer_for(shard, plan['optimizer_initializer'])
    dist.init_process_group('gloo', timeout=timedelta(seconds=180))
    wire = Wire(rank, world)
    begun, cache = time.monotonic(), {}
    try:
        agreement = {'prepared': data.identity(prepared), 'parent': data.identity(parent),
                     'fast': data.identity(fast), 'target': data.identity(target), 'command': args.command}
        if any(other != agreement for other in wire.exchange(agreement)):
            raise ValueError('Workers disagree on the frozen computation')
        data.save(args.home / 'started.json', {'runtime': runtime, 'rank': rank,
                  'resident_parameters': shard.resident_parameters, **agreement, 'tokens_issued': 0})

        def evaluate(model, role):
            if time.monotonic() - begun > plan['max_seconds']:
                raise TimeoutError('Consolidation execution deadline reached')
            if role not in cache:
                cache[role] = contract.read_role(plan, args.prepared.parent, prepared, role, tokenizer)
            records = cache[role]
            outcome = {'losses': score(model, wire, records), 'answers': []}
            if role in ('dev-prior', 'dev-new', 'test-new', 'test-prior'):
                for record in records:
                    if time.monotonic() - begun > plan['max_seconds']:
                        raise TimeoutError('Generation deadline reached')
                    prompt = tokenizer.apply_chat_template(record['messages'][:-1], tokenize=True, add_generation_prompt=True)
                    started = time.monotonic()
                    ids = generate(model, wire, prompt, plan['generation_tokens'], tokenizer.eos_token_id)
                    text = tokenizer.decode(ids, skip_special_tokens=True)
                    outcome['answers'].append({'id': record['id'], 'output_ids': ids, 'text': text,
                        'check': contract.check_answer(plan, record['task'], text, role),
                        'seconds': time.monotonic() - started})
                    if len(outcome['answers']) % 32 == 0:
                        emit('generated', rank=rank, role=role, cases=len(outcome['answers']))
            if any(other != clean(outcome) for other in wire.exchange(clean(outcome))):
                raise ValueError('Workers disagree on generated answers or response losses')
            return outcome

        if args.command == 'evaluate':
            portable.load(args.resume.parent if args.resume else args.parent.parent,
                          shard, None, target, target['job'], restore_rng=False)
            outcomes = {role: evaluate(shard, role) for role in contract.FINALS}
            data.save(args.home / 'evaluation.json', {'checkpoint': data.identity(target),
                'prepared': data.identity(prepared), 'rank': rank, 'outcomes': outcomes,
                'seconds': time.monotonic() - begun, 'peak_cuda_bytes': torch.cuda.max_memory_allocated()})
            return
        baseline = {role: clean(evaluate(teacher, role)) for role in contract.DEVELOPMENT}
        data.save(args.home / 'baseline.json', baseline)
        attempts = []
        for alpha in plan['method']['alphas']:
            births = portable.load(args.fast.parent, shard, optimizer, fast, fast['job'], restore_rng=True)
            blend_(shard, teacher, alpha)
            folder = args.home / ('alpha-' + str(alpha).replace('.', '_'))
            common = portable.commit(folder, shard, optimizer, wire, contract.job(prepared, alpha),
                fast['step'], data.identity(fast), births, transition=contract.transition(plan, prepared, alpha))
            emit('checkpoint', rank=rank, alpha=alpha, root=data.identity(common))
            outcomes = {'dev-prior': clean(evaluate(shard, 'dev-prior'))}
            prior = contract.generation(plan, prepared, 'dev-prior', baseline['dev-prior']['answers'],
                                        outcomes['dev-prior']['answers'], tokenizer)
            if prior['losses'] == 0:
                for role in ('dev-new', 'dev-retention'):
                    outcomes[role] = clean(evaluate(shard, role))
            decision = contract.screen_decision(plan, prepared, baseline, outcomes, tokenizer)
            attempt = {'alpha': alpha, 'checkpoint': common, 'outcomes': outcomes, 'decision': decision}
            data.save(folder / 'screen.json', attempt)
            attempts.append(attempt)
            emit('screen', rank=rank, alpha=alpha, passed=decision['passed'], prior_losses=prior['losses'])
            if decision['passed']:
                proposal = {'format': contract.SELECTION, 'prepared': data.identity(prepared),
                    'parent': data.identity(parent), 'fast': data.identity(fast), 'input_checkpoint': fast,
                    'baseline': baseline, 'attempts': attempts, 'candidate': data.identity(common)}
                contract.validate_selection(plan, prepared, proposal, tokenizer)
                data.save(args.home / 'selection-proposal.json', proposal)
                data.save(args.home / 'result.json', {'passed': True, 'alpha': alpha, 'candidate': data.identity(common),
                    'rank': rank, 'seconds': time.monotonic() - begun,
                    'peak_cuda_bytes': torch.cuda.max_memory_allocated(), 'new_gradient_updates': 0, 'tokens_issued': 0})
                return
        data.save(args.home / 'aborted.json', {'passed': False, 'baseline': baseline, 'attempts': attempts,
                  'reason': 'No declared coefficient preserved development answers and met the new-skill gate'})
        raise ValueError('Every declared consolidation coefficient failed development; keep finals closed')
    finally:
        dist.destroy_process_group()


def score_final(args, plan, prepared):
    contract.committed_prepared(plan, prepared)
    selected = json.loads(args.selection.read_bytes())
    contract.committed_selection(plan, prepared, selected)
    tokenizer = tokenizer_for(plan, args.seed)
    contract.validate_selection(plan, prepared, selected, tokenizer)
    baseline, candidate = [json.loads(filename.read_bytes()) for filename in (args.baseline, args.candidate)]
    for report, checkpoint in ((baseline, selected['parent']), (candidate, selected['candidate'])):
        if (report['checkpoint'] != checkpoint or report['prepared'] != data.identity(prepared)
                or set(report['outcomes']) != set(contract.FINALS)):
            raise ValueError('Final report differs from the exact selected evaluation')
    outcomes = {}
    for role in ('test-new', 'test-prior'):
        before, after = baseline['outcomes'][role], candidate['outcomes'][role]
        summary = contract.generation(plan, prepared, role, before['answers'], after['answers'], tokenizer)
        passed = all(row['candidate_correct'] >= row['baseline_correct'] for row in summary['families'].values())
        if role == 'test-new':
            passed = (passed and summary['wins'] - summary['losses'] >= plan['quality_gate']['min_net_gain']
                      and summary['one_sided_p'] < plan['quality_gate']['max_one_sided_p'])
        for family in summary['families'].values():
            family['passed'] = family['candidate_correct'] >= family['baseline_correct']
        summary['passed'] = passed
        outcomes[role] = {'generation': summary, 'passed': passed,
            'loss': contract.retention(plan, prepared, role, before['losses'], after['losses'])}
    retain = contract.retention(plan, prepared, 'retention', baseline['outcomes']['retention']['losses'],
                                candidate['outcomes']['retention']['losses'])
    retain['passed'] = retain['upper'] <= plan['quality_gate']['retention_upper_at_most_nats']
    outcomes['retention'] = retain
    result = {'format': contract.FORMAT + '/quality', 'prepared': data.identity(prepared),
        'selection': data.identity(selected), 'baseline': selected['parent'], 'candidate': selected['candidate'],
        'alpha': selected['attempts'][-1]['alpha'], 'outcomes': outcomes,
        'passed': all(value['passed'] for value in outcomes.values()),
        'tokens_issued': 0, 'serving_promoted': False,
        'scope': 'One fixed-size sharded model consolidated from an existing learning update; fresh public task families and retention, not general assistant quality'}
    data.save(args.output, result)
    emit('quality', passed=result['passed'], alpha=result['alpha'])


def main(argv, root):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['prepare', 'screen', 'evaluate', 'score'])
    parser.add_argument('--plan', type=Path, default=root / contract.PLAN)
    parser.add_argument('--seed', type=Path, required=True)
    for name in ('home', 'prepared', 'parent', 'fast', 'resume', 'selection', 'source-parquet',
                 'adaptive-inputs', 'continued-inputs', 'reasoned-inputs', 'baseline', 'candidate', 'output'):
        parser.add_argument('--' + name, type=Path)
    parser.add_argument('--plan-commit')
    args = parser.parse_args(argv)
    plan = contract.load(args.plan)
    required = {'prepare': ('home', 'source_parquet', 'plan_commit'),
                'screen': ('home', 'prepared', 'parent', 'fast'),
                'evaluate': ('home', 'prepared', 'parent', 'fast', 'selection'),
                'score': ('prepared', 'selection', 'baseline', 'candidate', 'output')}
    if any(getattr(args, key) is None for key in required[args.command]):
        raise ValueError('Missing required inputs for ' + args.command)
    if args.command == 'prepare':
        prepare(args, plan)
    else:
        prepared = json.loads(args.prepared.read_bytes())
        if args.command == 'score':
            score_final(args, plan, prepared)
        else:
            run_gpu(args, plan, prepared)
