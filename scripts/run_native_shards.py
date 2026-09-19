#!/usr/bin/env python3
"""Execute a reserved native window, or replay one owned partition.

Training and serving require a genesis-pinned native RPC and an existing local
worker key. Auditing takes a locally staged claim. This operated GPU profile
uses the unchanged balanced recipe and never loads unowned model parameters.
"""
import argparse
from datetime import timedelta
import json
import os
from pathlib import Path
import random

os.environ.setdefault('ATEN_CPU_CAPABILITY', 'default')
os.environ.setdefault('MKL_ENABLE_INSTRUCTIONS', 'SSE4_2')
os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')

import numpy as np
import torch
import torch.distributed as dist

from neuroshard.demo import client, protocol
from neuroshard.evolution import balanced, continued, reference
from neuroshard.evolution import portable_lifecycle as lifecycle, portable_work
from neuroshard.evolution import reference_data as data
from neuroshard.evolution.sharded import native_execution as execution, portable, transcript
from neuroshard.evolution.sharded.balanced_job import tokenizer_for
from neuroshard.evolution.sharded.model import Partition
from neuroshard.evolution.sharded.wire import Wire


def read(path):
    return json.loads(Path(path).read_bytes())


def live(args, before, executor, rank):
    if not args.rpc or not args.genesis_hash:
        raise ValueError('Native execution requires an explicitly pinned genesis and RPC')
    genesis = client.rpc(args.rpc, 'genesis')['genesis']
    if data.identity(genesis) != args.genesis_hash:
        raise ValueError('Native execution genesis differs from the configured commitment')
    status = client.query(args.rpc)
    life = client.query(args.rpc, '/portable_lifecycle')
    native = client.query(args.rpc, '/portable_work')
    if status['chain_id'] != genesis['chain_id']:
        raise ValueError('RPC chain identity changed')
    if args.command == 'train':
        active, assignment = life['active'], status['assignment']
        if (not active or active['closed'] or status['height'] > active['expires']
                or not assignment or assignment['id'] != args.assignment
                or status['height'] > assignment['expires'] or status['candidate']
                or native['checkpoint'] != before or assignment['input_checkpoint'] != data.identity(before)):
            raise ValueError('No matching live native training reservation')
        job = active['job']
        if (job['executor_root'] != data.identity(executor) or job['job'] != executor['job']
                or job['prepared'] != executor['prepared'] or job['reference_root'] != executor['reference_root']
                or not before['step'] < args.until <= min(job['max_step'], before['step'] + job['max_window_steps'])):
            raise ValueError('Reservation differs from the committed executor or update window')
        if not args.worker_key or not args.worker_key.is_file():
            raise ValueError('Use the existing reserved worker key')
        owner = protocol.Identity.load_or_create(args.worker_key)
        if assignment['workers'][rank] != owner.public_key:
            raise ValueError('This partition belongs to another worker')
        return status, life, assignment, owner
    if args.command == 'infer':
        job = life['jobs'].get(args.assignment)
        if (not job or job['checkpoint'] != before or status['height'] > job['expires']
                or job['executor_root'] != data.identity(executor) or job['claim_id']):
            raise ValueError('No matching funded native inference request')
        if not args.worker_key or not args.worker_key.is_file():
            raise ValueError('Use the existing assigned serving key')
        owner = protocol.Identity.load_or_create(args.worker_key)
        if job['workers'][rank] != owner.public_key:
            raise ValueError('This serving partition belongs to another worker')
        return status, life, job, owner
    active = life['active']
    if (not active or active['closed'] or active['id'] != args.assignment
            or native['checkpoint'] != before or status['candidate'] or status['assignment']
            or before['step'] != active['job']['max_step'] or status['height'] > active['expires']
            or active['job']['executor_root'] != data.identity(executor)):
        raise ValueError('Quality evaluation requires the complete settled native job')
    return status, life, active, None


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['train', 'audit', 'infer', 'quality', 'audit-service'])
    for name in ('execution', 'prepared', 'seed', 'home', 'checkpoint'):
        parser.add_argument('--' + name, type=Path, required=True)
    for name in ('reference', 'baseline', 'expected', 'claim', 'transcripts', 'witness', 'worker-key'):
        parser.add_argument('--' + name, type=Path)
    for name in ('rpc', 'genesis-hash', 'assignment'):
        parser.add_argument('--' + name)
    parser.add_argument('--until', type=int)
    args = parser.parse_args(argv)
    plan, prepared, executor = balanced.load(), read(args.prepared), read(args.execution)
    execution.check_descriptor(executor, plan, prepared)
    tokenizer = tokenizer_for(plan, args.seed)
    runtime = reference.configure('cuda', plan['threads'])
    runtime['allocator'] = os.environ['PYTORCH_CUDA_ALLOC_CONF']
    if {key: runtime[key] for key in continued.RUNTIME_KEYS} != plan['runtime']:
        raise ValueError('Native GPU runtime differs from the frozen numerical profile')
    rank, world = int(os.environ['RANK']), int(os.environ['WORLD_SIZE'])
    before = read(args.checkpoint)
    config = portable.validate(before)
    if before['boundaries'] != plan['boundaries'] or world != len(plan['boundaries']) - 1 or not 0 <= rank < world:
        raise ValueError('Native execution requires every declared partition')
    config._attn_implementation = 'sdpa'
    torch.manual_seed(plan['seeds']['training_rng'] + rank)
    random.seed(plan['seeds']['training_rng'] + rank)
    np.random.seed(plan['seeds']['training_rng'] + rank)
    audit = args.command in ('audit', 'audit-service')
    claim = read(args.claim) if audit and args.claim else None
    if audit:
        if not claim or claim.get('executor_root') != data.identity(executor) or claim['input_checkpoint'] != before:
            raise ValueError('Audit inputs differ from the native execution obligation')
    else:
        status, life, assignment, owner = live(args, before, executor, rank)
    args.home.mkdir(parents=True, exist_ok=False)
    shard = Partition(config, plan['boundaries'], rank, 'cuda', plan['parameter_limit'])
    optimizer = teacher = None
    if args.command in ('train', 'audit'):
        if not args.reference or data.identity(read(args.reference)) != plan['parent']['checkpoint']:
            raise ValueError('Native continuation requires its complete frozen reference')
        teacher = Partition(config, plan['boundaries'], rank, 'cuda', plan['parameter_limit'])
        anchor = read(args.reference)
        portable.load(args.reference.parent, teacher, None, anchor, anchor['job'], restore_rng=False)
        teacher.eval().requires_grad_(False)
        optimizer = reference.optimizer_for(shard, plan['training'])
        rows = balanced.read_role(plan, args.prepared.parent, prepared, 'train', tokenizer)
    births = portable.load(args.checkpoint.parent, shard, optimizer, before, before['job'], restore_rng=optimizer is not None)
    if args.command == 'audit':
        after = read(args.expected)
        if (after != claim['output_checkpoint'] or claim['prepared'] != data.identity(prepared)
                or claim['reference_root'] != executor['reference_root'] or after['job'] != executor['job']):
            raise ValueError('Audited output differs from the activated recipe')
        reports = read(args.transcripts)
        if transcript.validate(reports) != claim['record_root']:
            raise ValueError('Native transcript commitment differs')
        result = execution.replay_window(args.home, shard, teacher, optimizer, rows, prepared, plan,
            before, after, births, executor['job'], reports, args.witness)
        data.save(args.home / 'audit.json', result)
        return
    if args.command == 'audit-service':
        reports = read(args.transcripts)
        result = execution.service_report(claim, rank, reports)
        replay = execution.SegmentedReplay if claim['kind'] == 'portable_quality' else transcript.Replay
        wire = replay(args.witness, reports[rank])
    else:
        # Large input loading precedes network startup and its operation timeout.
        dist.init_process_group('gloo', timeout=timedelta(seconds=180))
        network = Wire(rank, world)
        agreement = {'checkpoint': data.identity(before), 'assignment': assignment['id'],
                     'executor': data.identity(executor), 'command': args.command}
        if any(value != agreement for value in network.exchange(agreement)):
            raise ValueError('Workers disagree on the native execution assignment')
        if args.command == 'train':
            wire = network
        else:
            recorder = execution.SegmentedRecorder if args.command == 'quality' else transcript.Recorder
            wire = recorder(network, args.home / 'transcript')
    try:
        if args.command == 'train':
            after = execution.train_window(args.home, shard, teacher, optimizer, wire, rows, prepared,
                plan, before, args.until, births, executor['job'])
            manifests = wire.exchange(read(args.home / 'transcript/transcript.json'))
            closed = transcript.validate(manifests)
            data.save(args.home / 'transcripts.json', manifests)
            # Sign only after every witness and resulting checkpoint is committed.
            receipt = portable_work.receipt(status['chain_id'], assignment, after, closed, rank)
            data.save(args.home / 'receipt.json', owner.sign(receipt))
            data.save(args.home / 'result.json', {'checkpoint': data.identity(after), 'transcript_root': closed,
                'assignment': assignment['id'], 'rank': rank, 'peak_cuda_bytes': torch.cuda.max_memory_allocated()})
            return
        kind = claim['kind'] if audit else ('portable_inference' if args.command == 'infer' else 'portable_quality')
        if kind == 'portable_inference':
            request = claim['request'] if audit else {key: assignment[key] for key in
                ('prompt_ids', 'max_tokens', 'eos_id', 'tokenizer_root')}
            if request['tokenizer_root'] != data.tokenizer_identity(tokenizer):
                raise ValueError('Serving tokenizer differs from the paid request')
            tokens = execution.generate_response(shard, wire, request)
            if audit:
                if tokens != claim['token_ids']:
                    raise ValueError('Replayed native response differs')
            else:
                claim = {'kind': kind, 'job_id': assignment['id'], 'executor_root': data.identity(executor),
                    'input_checkpoint': before, 'model_root': before['state_root'], 'request': request,
                    'token_ids': tokens, 'stages': len(tokens), 'record_root': None}
        elif kind == 'portable_quality':
            policy = execution.quality_policy(plan, prepared, executor)
            expected_policy = claim['report']['policy_root'] if audit else assignment['job']['quality']['policy_root']
            if data.identity(policy) != expected_policy:
                raise ValueError('Quality evaluator differs from the frozen native policy')
            baseline = read(args.baseline)
            if data.identity(baseline) != policy['baseline']:
                raise ValueError('Quality comparison substituted its baseline')
            records = {role: balanced.read_role(plan, args.prepared.parent, prepared, role, tokenizer)
                       for role in balanced.FINALS}
            outcomes = {'candidate': execution.evaluate_side(shard, wire, plan, records, tokenizer)}
            # Reuse the owned partition allocation; never load the complete model.
            portable.load(args.baseline.parent, shard, None, baseline, baseline['job'], restore_rng=False)
            outcomes['baseline'] = execution.evaluate_side(shard, wire, plan, records, tokenizer)
            decision = execution.quality_decision(plan, prepared, outcomes, tokenizer)
            report = {'format': lifecycle.FORMAT + '/quality', 'policy_root': data.identity(policy),
                'baseline_checkpoint': data.identity(baseline), 'candidate_checkpoint': data.identity(before),
                'prepared': data.identity(prepared), 'passed': decision['passed'],
                'results_root': data.identity({'outcomes': outcomes, 'decision': decision})}
            data.save(args.home / 'quality-results.json', {'outcomes': outcomes, 'decision': decision})
            if audit:
                if report != claim['report'] or baseline != claim['baseline_checkpoint']:
                    raise ValueError('Replayed native quality report differs')
            else:
                claim = {'kind': kind, 'job_id': assignment['id'], 'executor_root': data.identity(executor),
                    'input_checkpoint': before, 'baseline_checkpoint': baseline,
                    'model_root': before['state_root'], 'report': report, 'record_root': None,
                    'stages': assignment['job']['quality']['stages']}
        else:
            raise ValueError('Unsupported portable numerical service')
        if audit:
            wire.finish()
            data.save(args.home / 'audit.json', result)
        else:
            local = wire.finish(lifecycle.transcript_binding(claim))
            manifests = execution.exchange_manifests(network, local)
            claim['record_root'] = execution.validate_service_transcripts(manifests)
            data.save(args.home / 'transcripts.json', manifests)
            data.save(args.home / 'service.json', claim)
            if kind == 'portable_inference':
                receipt = lifecycle.inference_receipt(status['chain_id'], assignment, tokens, claim['record_root'], rank)
                data.save(args.home / 'receipt.json', owner.sign(receipt))
                data.save(args.home / 'response.json', {'token_ids': tokens,
                    'text': tokenizer.decode(tokens, skip_special_tokens=True)})
    finally:
        if not audit:
            dist.destroy_process_group()


if __name__ == '__main__':
    main()
