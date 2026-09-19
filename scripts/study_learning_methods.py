#!/usr/bin/env python3
"""Controlled method experiment: shared AdamW with dense or compressed gradients.

Research only. All parameters train; no ledger or serving changes. Data and
source are committed before training; all final candidates precede test scoring.
"""
import argparse
from datetime import timedelta
import inspect
import json
import os
from pathlib import Path
import subprocess
import time

from neuroshard.evolution import cooperative as group
from neuroshard.evolution import gradient_compression
from neuroshard.evolution import grounded_tasks as tasks
from neuroshard.evolution import local_windows as windows
from neuroshard.evolution import reference as engine
from neuroshard.evolution import reference_data as data
from run_cooperative_learning import model_snapshot, network_bytes, tokenizer_for, write_records

ROOT = Path(__file__).resolve().parents[1]
PLAN = ROOT / 'config/experiments/learning-method-study.json'
PREPARED = ROOT / 'config/experiments/learning-method-study-inputs.json'
SELECTED = ROOT / 'config/experiments/learning-method-study-selection.json'


def emit(event, **values):
    print(json.dumps({'event': event, **values}), flush=True)


def sources():
    paths = ['scripts/study_learning_methods.py', 'scripts/run_cooperative_learning.py',
             'src/neuroshard/evolution/cooperative.py', 'src/neuroshard/evolution/gradient_compression.py',
             'src/neuroshard/evolution/grounded_tasks.py', 'src/neuroshard/evolution/local_windows.py',
             'src/neuroshard/evolution/reference.py', 'src/neuroshard/evolution/reference_data.py',
             'src/neuroshard/dataflow/store.py', 'docs/learning-reference-requirements.txt']
    return {name: data.sha256(ROOT / name) for name in paths}


def committed(path):
    relative = path.relative_to(ROOT).as_posix()
    if subprocess.check_output(['git', 'show', 'HEAD:' + relative], cwd=ROOT) != path.read_bytes():
        raise ValueError('Commit exact experimental inputs before execution: ' + relative)


def prepare(args, plan):
    previous = json.loads((args.previous_home / 'prepared.json').read_bytes())
    if data.identity(previous) != plan['previous_prepared']:
        raise ValueError('Changed replay provenance')
    reference = json.loads((args.reference_home / 'prepared.json').read_bytes())
    if data.sha256(args.reference_home / 'prepared.json') != previous['plan']['reference_prepared_sha256']:
        raise ValueError('Changed reference input provenance')
    tokenizer = tokenizer_for(args.model_dir)
    if data.tokenizer_identity(tokenizer) != previous['tokenizer']:
        raise ValueError('Changed tokenizer')
    replay = data.read_records(args.reference_home / 'inputs/train.jsonl', reference['roles']['train']['sha256'])
    replay = replay[:plan['replay_documents']]
    retention = data.read_records(args.reference_home / 'inputs/retention.jsonl', reference['roles']['retention']['sha256'])
    if len(replay) != plan['replay_documents']:
        raise ValueError('Insufficient replay')
    if args.home.exists() and any(args.home.iterdir()):
        raise ValueError('Use an empty preparation directory')
    (args.home / 'inputs').mkdir(parents=True)
    roles, seen = {}, set()
    for role, count in [('train', plan['task_documents']), ('dev', plan['dev_documents']), ('test', plan['test_documents'])]:
        rows = []
        for index in range(count):
            case = tasks.make_case(plan['task_seed'], role, index)
            identifier = tasks.task_identity(case)
            prompt = tasks.prompt(case)
            if identifier in seen or prompt in seen:
                raise ValueError('Duplicate generated example')
            seen.update((identifier, prompt))
            answer = json.dumps(tasks.expected(case), separators=(',', ':'))
            assert tasks.check_answer(case, answer)['correct']
            messages = [{'role': 'user', 'content': prompt}, {'role': 'assistant', 'content': answer}]
            rows.append({'id': identifier, 'task': case, 'messages': messages,
                         'loss_weight': plan['task_token_weight'],
                         **data.conversation(tokenizer, messages, plan['max_length'])})
        if role == 'train':
            rows += replay
        roles[role] = write_records(args.home / f'inputs/{role}.jsonl', rows)
    roles['retention'] = write_records(args.home / 'inputs/retention.jsonl', retention)
    prepared = {'plan': plan, 'sources': sources(), 'roles': roles, 'tokenizer': previous['tokenizer'],
                'model_snapshot': previous['model_snapshot'],
                'scope': 'Generated public task families and previously exposed retention probes. Development method study, not a general-assistant certificate.'}
    data.save(args.home / 'prepared.json', prepared)
    emit('prepared', identity=data.identity(prepared))


def inputs(args, plan):
    committed(PLAN)
    committed(PREPARED)
    prepared = json.loads((args.home / 'prepared.json').read_bytes())
    if prepared != json.loads(PREPARED.read_bytes()) or prepared['plan'] != plan or prepared['sources'] != sources():
        raise ValueError('Input or numerical implementation changed')
    return prepared


def records(args, prepared, role):
    rows = data.read_records(args.home / f'inputs/{role}.jsonl', prepared['roles'][role]['sha256'])
    if [r['id'] for r in rows] != prepared['roles'][role]['ids']:
        raise ValueError('Record order changed')
    return rows


def runtime():
    value = engine.configure('cuda', 2)
    if os.environ.get('NCCL_ALGO') != 'Ring' or os.environ.get('NCCL_PROTO') != 'Simple':
        raise ValueError('Use the declared NCCL Ring/Simple transport')
    value.update(nccl_algorithm='Ring', nccl_protocol='Simple', nccl_ib_disable=os.environ.get('NCCL_IB_DISABLE'))
    return value


def train(args, plan):
    import torch
    import torch.distributed as dist
    from torch.nn.parallel import DistributedDataParallel
    from torch.distributed.algorithms.ddp_comm_hooks import powerSGD_hook as compression
    prepared = inputs(args, plan)
    profile = runtime()
    rank, world = int(os.environ.get('RANK', 0)), int(os.environ.get('WORLD_SIZE', 1))
    if world != plan['arms'][args.arm] or not 0 <= rank < world:
        raise ValueError('Wrong experimental membership')
    if model_snapshot(args.model_dir, plan['model']) != prepared['model_snapshot']:
        raise ValueError('Changed upstream seed')
    destination = args.home / args.arm / f'rank-{rank}'
    destination.mkdir(parents=True, exist_ok=True)
    marker = destination / 'started.json'
    if marker.exists():
        raise ValueError('Preserve the previous attempt')
    binding = data.identity({'inputs': data.identity(prepared), 'arm': args.arm,
                             'runtime': group.runtime_profile(profile),
                             'upstream_hook': data.sha256(Path(inspect.getsourcefile(compression)))})
    data.save(marker, {'binding': binding, 'runtime': profile, 'started': time.time()})
    budget = engine.Budget(destination, time.time(), plan['budget'])
    if world > 1:
        dist.init_process_group('nccl', timeout=timedelta(seconds=300))
    try:
        group.agree_digest(binding, world, 'cuda')
        torch.manual_seed(plan['training']['seed'])
        model = engine.load_model(args.model_dir, 'cuda', plan['model']['parameters'])
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
        model.train()
        optimizer = engine.optimizer_for(model, plan['training'])
        wrapped = DistributedDataParallel(model, device_ids=[0], broadcast_buffers=False,
            gradient_as_bucket_view=True, bucket_cap_mb=64) if world > 1 else model
        state = None
        if args.arm == 'compressed-pair':
            state = compression.PowerSGDState(process_group=dist.group.WORLD, **plan['compression'])
            wrapped.register_comm_hook(state, gradient_compression.offloaded_power_sgd)
        rows = records(args, prepared, 'train')
        recipe = plan['training']
        schedule = engine.schedule(len(rows), recipe['steps'], recipe['batch_documents'], recipe['seed'])
        before, started, journal = network_bytes(), time.monotonic(), []
        emit('training_started', arm=args.arm, rank=rank)
        for index, indices in enumerate(schedule):
            observation = group.step(wrapped, optimizer, [rows[i] for i in indices], 'cuda',
                                     recipe, index, rank, world, budget.check)
            journal.append(observation)
            if index % 8 == 0 or index + 1 == recipe['steps']:
                emit('step', arm=args.arm, rank=rank, step=index+1, loss=observation['loss'], seconds=observation['seconds'])
            with (destination / 'steps.jsonl').open('a') as output:
                output.write(json.dumps(observation) + '\n')
        active = time.monotonic() - started
        after = network_bytes()
        digest = group.parameter_digest(model)
        group.agree_digest(digest, world, 'cuda')
        compression_stats = None
        if state is not None:
            if state.iter != recipe['steps'] or any(x.device.type != 'cpu' for x in state.error_dict.values()):
                raise ValueError('Compression iterations or residency differ')
            ratio, original, transmitted = state.compression_stats()
            compression_stats = {'ratio': ratio, 'original_elements': original, 'transmitted_elements': transmitted,
                                 'error_digest': windows.state_digest((str(k), v) for k, v in sorted(state.error_dict.items()))}
        checkpoint_started = time.monotonic()
        files = None
        if rank == 0:
            checkpoint = destination / 'model'
            model.save_pretrained(checkpoint, safe_serialization=True, max_shard_size='2GB')
            tokenizer_for(args.model_dir).save_pretrained(checkpoint)
            files = {p.name: data.sha256(p) for p in checkpoint.iterdir() if p.is_file()}
        if world > 1:
            dist.barrier()
        result = {'arm': args.arm, 'rank': rank, 'world': world, 'prepared': data.identity(prepared),
                  'binding': binding, 'runtime': profile, 'steps': journal, 'parameter_digest': digest,
                  'model_files': files, 'compression': compression_stats, 'active_seconds': active,
                  'checkpoint_seconds': time.monotonic() - checkpoint_started,
                  'network_start': before, 'network_end': after,
                  'peak_cuda_bytes': torch.cuda.max_memory_allocated(), 'tokens_issued': 0,
                  'scope': 'Model-only final checkpoint. Recovery and Byzantine verification are separate experiments.'}
        data.save(destination / 'result.json', result)
        emit('completed', arm=args.arm, rank=rank, seconds=active, digest=digest)
    finally:
        if world > 1 and dist.is_initialized():
            dist.destroy_process_group()


def select(args, plan):
    prepared = inputs(args, plan)
    selected = {}
    for arm, world in plan['arms'].items():
        results = [json.loads((args.home / arm / f'rank-{rank}/result.json').read_bytes()) for rank in range(world)]
        reference = results[0]
        for rank, result in enumerate(results):
            if (result['rank'] != rank or result['world'] != world or result['arm'] != arm
                    or result['prepared'] != data.identity(prepared)
                    or result['parameter_digest'] != reference['parameter_digest']
                    or result['binding'] != reference['binding']
                    or len(result['steps']) != plan['training']['steps']):
                raise ValueError('Missing or disagreeing final candidate')
        selected[arm] = {k: reference[k] for k in ['parameter_digest', 'model_files', 'binding', 'runtime']}
    data.save(args.home / 'selection.json', {'prepared': data.identity(prepared), 'candidates': selected})


def evaluate(args, plan):
    prepared = inputs(args, plan)
    profile = runtime()
    committed(SELECTED)
    selected = json.loads(SELECTED.read_bytes())
    if selected['prepared'] != data.identity(prepared) or set(selected['candidates']) != set(plan['arms']):
        raise ValueError('Commit every declared candidate before evaluation')
    if args.arm == 'seed':
        directory = args.model_dir
        if model_snapshot(directory, plan['model']) != prepared['model_snapshot']:
            raise ValueError('Changed seed')
    else:
        directory = args.candidate_dir or args.home / args.arm / 'rank-0/model'
        candidate = selected['candidates'][args.arm]
        if {p.name: data.sha256(p) for p in directory.iterdir() if p.is_file()} != candidate['model_files']:
            raise ValueError('Changed model artifacts')
        if group.runtime_profile(profile) != group.runtime_profile(candidate['runtime']):
            raise ValueError('Changed numerical profile')
    destination = args.home / 'evaluation' / (args.arm + '.json')
    marker = destination.with_suffix('.started.json')
    if marker.exists():
        raise ValueError('Preserve earlier evaluation')
    destination.parent.mkdir(parents=True, exist_ok=True)
    data.save(marker, {'started': time.time(), 'selection': data.identity(selected), 'runtime': profile})
    budget = engine.Budget(destination.parent, time.time(), plan['budget'])
    model = engine.load_model(directory, 'cuda', plan['model']['parameters'])
    if args.arm != 'seed' and group.parameter_digest(model) != candidate['parameter_digest']:
        raise ValueError('Model differs from committed candidate')
    tokenizer = tokenizer_for(args.model_dir)
    test = records(args, prepared, 'test')
    started = time.monotonic()
    generations = engine.generate(model, tokenizer, test, 'cuda', plan['generation_tokens'], len(test), budget.check)
    checks = [{'id': r['id'], 'family': r['task']['family'], **tasks.check_answer(r['task'], g['text'])}
              for r, g in zip(test, generations)]
    retention = engine.score(model, records(args, prepared, 'retention'), 'cuda', budget.check)
    data.save(destination, {'arm': args.arm, 'prepared': data.identity(prepared), 'selection': data.identity(selected),
              'runtime': profile, 'checks': checks, 'generations': generations, 'retention': retention,
              'seconds': time.monotonic() - started, 'correct': sum(c['correct'] for c in checks)})
    emit('evaluated', arm=args.arm, correct=sum(c['correct'] for c in checks), count=len(checks))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['prepare', 'train', 'select', 'evaluate'])
    parser.add_argument('--home', type=Path, required=True)
    parser.add_argument('--model-dir', type=Path, required=True)
    parser.add_argument('--previous-home', type=Path)
    parser.add_argument('--reference-home', type=Path)
    parser.add_argument('--candidate-dir', type=Path)
    parser.add_argument('--arm', choices=['seed', 'single', 'dense-pair', 'compressed-pair'])
    args = parser.parse_args()
    plan = json.loads(PLAN.read_bytes())
    with windows.exclusive_device('cuda' if args.command in ('train', 'evaluate') else 'cpu'):
        globals()[args.command](args, plan)


if __name__ == '__main__':
    main()
