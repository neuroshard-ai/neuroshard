"""Frozen comparison of cached and original sharded generation, with replay."""

from datetime import timedelta
import json
import math
import os
from pathlib import Path
import subprocess
import time

import torch
import torch.distributed as dist

from .. import balanced, consolidation, continued, reference
from .. import reference_data as data
from . import portable, transcript
from .balanced_job import tokenizer_for
from .cached_inference import generate_cached
from .model import Partition
from .training import generate
from .wire import Wire

ROOT = Path(__file__).resolve().parents[4]
FORMAT = 'neuroshard-shard-cache-probe-v1'


def read(path):
    return json.loads(Path(path).read_bytes())


def sources():
    names = subprocess.check_output(['git', 'ls-files', 'src',
        'scripts/run_shard_cache_probe.py'], cwd=ROOT, text=True).splitlines()
    return {name: data.sha256(ROOT / name) for name in names if name.endswith('.py')}


def prepare(args):
    plan, original, checkpoint = read(args.plan), balanced.load(), read(args.checkpoint)
    if (plan['format'] != FORMAT or data.identity(checkpoint) != plan['checkpoint']
            or checkpoint['state_root'] != plan['learned_state']):
        raise ValueError('Cache comparison checkpoint differs from the frozen plan')
    for path in (args.plan,):
        if balanced.git_bytes('HEAD', path.resolve()) != path.read_bytes():
            raise ValueError('Commit the cache plan before preparation')
    if any(balanced.git_bytes('HEAD', ROOT / name) != (ROOT / name).read_bytes() for name in sources()):
        raise ValueError('Commit cache executor sources before preparation')
    prepared = read(args.original_prepared)
    if data.identity(prepared) != plan['balanced_prepared']:
        raise ValueError('Cache comparison substituted its exposed input manifest')
    tokenizer = tokenizer_for(original, args.seed)
    requests = []
    for role in plan['roles']:
        for row in balanced.read_role(original, args.original_prepared.parent, prepared, role, tokenizer):
            requests.append({'id': row['id'], 'role': role, 'task': row['task'],
                'prompt_ids': tokenizer.apply_chat_template(row['messages'][:-1], tokenize=True,
                                                          add_generation_prompt=True)})
    if len(requests) != plan['requests'] or len({row['id'] for row in requests}) != len(requests):
        raise ValueError('Incomplete cache comparison requests')
    args.home.mkdir(parents=True, exist_ok=False)
    path = args.home / 'requests.jsonl'
    path.write_text(''.join(json.dumps(row, sort_keys=True) + '\n' for row in requests))
    result = {'format': FORMAT + '/prepared', 'plan': plan, 'sources': sources(),
        'plan_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
        'runtime': original['runtime'], 'threads': original['threads'],
        'parameter_limit': original['parameter_limit'], 'tokenizer': data.tokenizer_identity(tokenizer),
        'original_results_sha256': data.sha256(ROOT / 'config/experiments/balanced-continuation-results.json'),
        'eos_id': tokenizer.eos_token_id, 'requests': {'file': path.name, 'sha256': data.sha256(path),
            'ids': [row['id'] for row in requests]}}
    data.save(args.home / 'prepared.json', result)
    return result


def checked(args):
    prepared = read(args.prepared)
    if (prepared['format'] != FORMAT + '/prepared' or prepared['plan'] != read(args.plan)
            or prepared['sources'] != sources()
            or prepared['original_results_sha256'] != data.sha256(ROOT / 'config/experiments/balanced-continuation-results.json')):
        raise ValueError('Cache numerical sources or plan differ from the freeze')
    committed = ROOT / 'config/experiments/shard-cache-prepared.json'
    if (balanced.git_bytes('HEAD', committed) != committed.read_bytes()
            or read(committed) != prepared):
        raise ValueError('Commit the completed cache preparation before execution')
    requests = data.read_records(args.inputs / prepared['requests']['file'], prepared['requests']['sha256'])
    if [row['id'] for row in requests] != prepared['requests']['ids']:
        raise ValueError('Cache request order differs')
    checkpoint = read(args.checkpoint)
    if data.identity(checkpoint) != prepared['plan']['checkpoint']:
        raise ValueError('Cache checkpoint changed')
    tokenizer = tokenizer_for(balanced.load(), args.seed)
    if data.tokenizer_identity(tokenizer) != prepared['tokenizer']:
        raise ValueError('Cache tokenizer changed')
    return prepared, checkpoint, requests, tokenizer


def initialize(args, prepared, checkpoint):
    runtime = reference.configure('cuda', prepared['threads'])
    runtime['allocator'] = os.environ['PYTORCH_CUDA_ALLOC_CONF']
    if {key: runtime[key] for key in continued.RUNTIME_KEYS} != prepared['runtime']:
        raise ValueError('Cache probe runtime differs')
    rank, world = int(os.environ['RANK']), int(os.environ['WORLD_SIZE'])
    if world != len(checkpoint['boundaries']) - 1 or not 0 <= rank < world:
        raise ValueError('Wrong cache probe topology')
    config = portable.validate(checkpoint)
    config._attn_implementation = 'sdpa'
    shard = Partition(config, checkpoint['boundaries'], rank, 'cuda', prepared['parameter_limit'])
    portable.load(args.checkpoint.parent, shard, None, checkpoint, checkpoint['job'], restore_rng=False)
    shard.eval().requires_grad_(False)
    return shard, world, runtime


def binding(prepared, row, tokens):
    return {'prepared': data.identity(prepared), 'request': data.identity(row), 'tokens': tokens,
            'executor': 'owner-local-kv-v1', 'checkpoint': prepared['plan']['checkpoint']}


def compare(args):
    prepared, checkpoint, requests, tokenizer = checked(args)
    shard, world, runtime = initialize(args, prepared, checkpoint)
    original = balanced.load()
    args.home.mkdir(parents=True, exist_ok=False)
    dist.init_process_group('gloo', timeout=timedelta(seconds=300))
    wire = Wire(shard.rank, world)
    try:
        # initialize() has already checked every numerical setting against the
        # preparation. Distinct host names remain in each owner's local report.
        declaration = {'prepared': data.identity(prepared), 'runtime': prepared['runtime']}
        if any(value != declaration for value in wire.exchange(declaration)):
            raise ValueError('Cache owners disagree on their computation')
        limit, eos = prepared['plan']['maximum_tokens'], prepared['eos_id']
        for row in requests[:prepared['plan']['warmup_requests']]:
            generate(shard, wire, row['prompt_ids'], limit, eos)
            generate_cached(shard, wire, row['prompt_ids'], limit, eos)
        for index, row in enumerate(requests):
            values = {}
            for method in (('uncached', 'cached') if index % 2 == 0 else ('cached', 'uncached')):
                torch.cuda.synchronize()
                torch.cuda.reset_peak_memory_stats()
                started, sent = time.monotonic(), wire.sent_tensor_bytes
                detail = {}
                tokens = (generate_cached(shard, wire, row['prompt_ids'], limit, eos, detail)
                          if method == 'cached' else generate(shard, wire, row['prompt_ids'], limit, eos))
                torch.cuda.synchronize()
                seconds = time.monotonic() - started
                text = tokenizer.decode(tokens, skip_special_tokens=True)
                values[method] = {'token_ids': tokens, 'text': text, 'seconds': seconds,
                    'sent_tensor_bytes': wire.sent_tensor_bytes - sent,
                    'peak_cuda_bytes': torch.cuda.max_memory_allocated(), 'cache': detail,
                    'check': consolidation.check_answer(original, row['task'], text, row['role'])}
            # Record a separate complete second execution. Benchmark timings
            # above include the same transport but neither includes disk writes.
            captured = transcript.Recorder(wire, args.home / f'request-{index:04d}')
            witness_observation = {}
            tokens = generate_cached(shard, captured, row['prompt_ids'], limit, eos, witness_observation)
            if tokens != values['cached']['token_ids']:
                raise ValueError('Fresh cached execution changed its output')
            local = captured.finish(binding(prepared, row, tokens))
            manifests = wire.exchange(local)
            values['cached']['transcript_root'] = transcript.validate(manifests)
            data.save(captured.home / 'transcripts.json', manifests)
            witness_observation['bytes'] = sum(path.stat().st_size for path in captured.home.iterdir())
            numerical = {method: {key: item[key] for key in ('token_ids', 'text', 'check')}
                         for method, item in values.items()}
            if any(value != numerical for value in wire.exchange(numerical)):
                raise ValueError('Cache comparison answers differ across owners')
            with (args.home / 'comparison.jsonl').open('a') as output:
                output.write(json.dumps({'id': row['id'], 'index': index, 'methods': values,
                                         'witness_production': witness_observation}) + '\n')
            if (index + 1) % 32 == 0:
                print(json.dumps({'completed_requests': index + 1, 'rank': shard.rank}), flush=True)
        data.save(args.home / 'complete.json', {'prepared': data.identity(prepared), 'rank': shard.rank,
            'requests': len(requests), 'comparison_sha256': data.sha256(args.home / 'comparison.jsonl'),
            'runtime': runtime, 'resident_parameters': shard.resident_parameters, 'tokens_issued': 0})
    finally:
        dist.destroy_process_group()


def replay(args):
    prepared, checkpoint, requests, _ = checked(args)
    shard, _, runtime = initialize(args, prepared, checkpoint)
    args.home.mkdir(parents=True, exist_ok=False)
    results = []
    began = time.monotonic()
    for index, row in enumerate(requests):
        folder = args.witness / f'request-{index:04d}'
        manifests = read(folder / 'transcripts.json')
        closed = transcript.validate(manifests)
        declared = manifests[shard.rank]['binding']
        if declared != binding(prepared, row, declared['tokens']):
            raise ValueError('Cached witness is bound to another request or checkpoint')
        wire = transcript.Replay(folder, manifests[shard.rank])
        tokens = generate_cached(shard, wire, row['prompt_ids'], prepared['plan']['maximum_tokens'], prepared['eos_id'])
        if tokens != declared['tokens']:
            raise ValueError('Replayed cached output differs')
        wire.finish()
        results.append({'id': row['id'], 'transcript_root': closed, 'token_ids': tokens})
    data.save(args.home / 'replay.json', {'passed': True, 'rank': shard.rank, 'prepared': data.identity(prepared),
        'requests': results, 'seconds': time.monotonic() - began, 'runtime': runtime,
        'resident_parameters': shard.resident_parameters, 'peak_cuda_bytes': torch.cuda.max_memory_allocated()})


def score(args):
    prepared, checkpoint, requests, tokenizer = checked(args)
    world = len(checkpoint['boundaries']) - 1
    reports = []
    for rank in range(world):
        folder = args.reports / f'rank-{rank}'
        complete = read(folder / 'complete.json')
        if (complete['rank'] != rank or complete['prepared'] != data.identity(prepared)
                or complete['requests'] != len(requests)):
            raise ValueError('Incomplete owner comparison')
        rows = data.read_records(folder / 'comparison.jsonl', complete['comparison_sha256'])
        if len(rows) != len(requests):
            raise ValueError('Comparison omits requests')
        reports.append(rows)
    original = balanced.load()
    expected = read(ROOT / 'config/experiments/balanced-continuation-results.json')['answer_pairs']
    expected = {row['id']: row['candidate'] for values in expected.values() for row in values}
    common, identical, losses, totals = [], [], [], {'uncached': 0, 'cached': 0}
    byte_totals = {method: 0 for method in totals}
    all_seconds = {method: 0. for method in totals}
    comparisons = []
    for index, request in enumerate(requests):
        pair = reports[0][index]['methods']
        for rank in range(world):
            row = reports[rank][index]
            if row['id'] != request['id'] or row['index'] != index or set(row['methods']) != set(totals):
                raise ValueError('Comparison request identity or order changed')
            for method, result in row['methods'].items():
                if any(result[key] != pair[method][key] for key in ('token_ids', 'text', 'check')):
                    raise ValueError('Comparison owners disagree on output')
                seconds, size = result['seconds'], result['sent_tensor_bytes']
                if (type(seconds) not in (int, float) or not math.isfinite(seconds) or seconds <= 0
                        or type(size) is not int or size <= 0):
                    raise ValueError('Invalid resource observation')
                byte_totals[method] += size
            if row['methods']['cached']['transcript_root'] != pair['cached']['transcript_root']:
                raise ValueError('Comparison owners disagree on the cached witness')
        for method, result in pair.items():
            ids = result['token_ids']
            if (not isinstance(ids, list) or not 0 < len(ids) <= prepared['plan']['maximum_tokens']
                    or any(type(token) is not int or not 0 <= token < checkpoint['config']['vocab_size'] for token in ids)
                    or prepared['eos_id'] in ids[:-1]
                    or (ids[-1] != prepared['eos_id'] and len(ids) != prepared['plan']['maximum_tokens'])
                    or tokenizer.decode(ids, skip_special_tokens=True) != result['text']):
                raise ValueError('Malformed or prematurely truncated output')
            check = consolidation.check_answer(original, request['task'], result['text'], request['role'])
            if check != result['check']:
                raise ValueError('Stored answer score differs from actual output')
            totals[method] += int(check['correct'])
            all_seconds[method] += result['seconds']
        if pair['uncached']['text'] != expected[request['id']]:
            raise ValueError('Original generator no longer reproduces the published checkpoint answers')
        if pair['uncached']['check']['correct'] and not pair['cached']['check']['correct']:
            losses.append(request['id'])
        same = pair['uncached']['token_ids'] == pair['cached']['token_ids']
        if same:
            identical.append(pair)
        common.append({'id': request['id'], 'transcript_root': pair['cached']['transcript_root'],
                       'token_ids': pair['cached']['token_ids']})
        comparisons.append({'id': request['id'], 'role': request['role'], 'identical_tokens': same,
                            'methods': pair})
    audits = []
    for auditor in range(world):
        for rank in range(world):
            report = read(args.audits / f'auditor-{auditor}/rank-{rank}/replay.json')
            if (report['passed'] is not True or report['rank'] != rank
                    or report['prepared'] != data.identity(prepared) or report['requests'] != common):
                raise ValueError('Every auditor must completely replay every cached partition')
            audits.append({'auditor': auditor, 'rank': rank, 'seconds': report['seconds'],
                           'peak_cuda_bytes': report['peak_cuda_bytes']})
    matched_seconds = {method: sum(pair[method]['seconds'] for pair in identical) for method in totals}
    speedup = matched_seconds['uncached'] / matched_seconds['cached'] if identical else 0.
    reduction = 1 - byte_totals['cached'] / byte_totals['uncached']
    gate = prepared['plan']['gate']
    decisions = {'retained_answers': len(losses) <= gate['individual_correct_answer_losses_at_most'],
        'identical_outputs': len(identical) / len(requests) >= gate['identical_token_fraction_at_least'],
        'generation_speedup': speedup >= gate['aggregate_speedup_on_identical_outputs_at_least'],
        'boundary_bytes': reduction >= gate['boundary_byte_reduction_at_least'], 'complete_replay': True}
    result = {'format': FORMAT + '/results', 'prepared': data.identity(prepared),
        'passed': all(decisions.values()), 'decisions': decisions, 'correct': totals,
        'individual_correct_answer_losses': losses, 'identical_outputs': len(identical),
        'requests': len(requests), 'matched_output_speedup': speedup,
        'matched_output_seconds': matched_seconds, 'all_output_seconds': all_seconds,
        'sent_tensor_bytes': byte_totals, 'boundary_byte_reduction': reduction,
        'audits': audits, 'answer_pairs': comparisons, 'tokens_issued': 0,
        'scope': prepared['plan']['scope']}
    args.home.mkdir(parents=True, exist_ok=False)
    data.save(args.home / 'results.json', result)
    return result
