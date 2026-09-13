#!/usr/bin/env python3
"""Freeze, run and evaluate operated local-training windows and matched controls."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import subprocess
import time
from datetime import timedelta

from neuroshard.evolution import cooperative as group
from neuroshard.evolution import grounded_tasks as tasks
from neuroshard.evolution import local_windows as windows
from neuroshard.evolution import reference as engine
from neuroshard.evolution import reference_data as data
from run_cooperative_learning import model_snapshot, tokenizer_for, write_records, network_bytes


ROOT = Path(__file__).resolve().parents[1]
PLAN = ROOT / 'config/experiments/local-training-windows.json'
PREPARATION = ROOT / 'config/experiments/local-training-windows-data-selection.json'
SELECTION = ROOT / 'config/experiments/local-training-windows-selection.json'


def emit(event, **values):
    print(json.dumps({'event': event, 'utc_seconds': time.time(), **values}), flush=True)


def implementation():
    paths = ['scripts/run_local_training_windows.py', 'scripts/run_cooperative_learning.py',
             'src/neuroshard/evolution/local_windows.py', 'src/neuroshard/evolution/cooperative.py',
             'src/neuroshard/evolution/grounded_tasks.py', 'src/neuroshard/evolution/reference.py',
             'src/neuroshard/evolution/reference_data.py', 'src/neuroshard/dataflow/store.py',
             'docs/learning-reference-requirements.txt']
    return data.identity({name: data.sha256(ROOT / name) for name in paths})


def committed(path, value):
    relative = path.resolve().relative_to(ROOT).as_posix()
    content = subprocess.check_output(['git', 'show', f'HEAD:{relative}'], cwd=ROOT)
    if content != path.read_bytes() or json.loads(content) != value:
        raise ValueError('Commit the exact preparation or selection before proceeding')


def validate(plan):
    if (plan['format'] != 'neuroshard-local-training-windows-v1' or plan['purpose'] != 'development'
            or plan['arms'] != {'single': 1, 'ddp-four': 4, 'diloco-four': 4}):
        raise ValueError('Use the explicit bounded development plan')
    for world in plan['arms'].values():
        windows.validate_window(plan['training'], world)
    if set(plan['checkpoints']) != set(plan['arms']):
        raise ValueError('Each arm requires its own checkpoint schedule')
    for arm, checkpoints in plan['checkpoints'].items():
        if (checkpoints != sorted(set(checkpoints)) or not checkpoints
                or checkpoints[-1] != plan['training']['steps']
                or any(step <= 0 or step % plan['training']['local_steps'] for step in checkpoints)):
            raise ValueError('Checkpoints must end complete declared windows')
    for name in ('grounded_train_documents', 'replay_documents', 'dev_documents', 'test_documents'):
        data.integer(plan[name], 4, 4096, name)
    for name, low, high in (('steps', 1, 2048), ('batch_documents', 4, 128), ('warmup_steps', 0, 127)):
        data.integer(plan['training'][name], low, high, name)
    if not 0 < plan['training']['learning_rate'] <= .001:
        raise ValueError('Invalid inner learning rate')
    if plan['training']['warmup_steps'] >= plan['training']['steps']:
        raise ValueError('Warmup exceeds the run')
    if not 0 < plan['training']['clip_norm'] <= 10 or not 0 <= plan['training']['weight_decay'] <= 1:
        raise ValueError('Invalid inner clipping or decay')
    data.integer(plan['max_length'], 32, 2048, 'context')
    data.integer(plan['generation_tokens'], 1, 256, 'generation')
    data.integer(plan['grounded_token_weight'], 1, 16, 'task weight')
    data.integer(plan['budget']['seconds'], 1, 28800, 'deadline')
    data.integer(plan['budget']['disk_gib'], 1, 240, 'artifact size')
    return plan


def prepare(args, plan):
    home = args.home
    if any(home.glob('*')):
        raise ValueError('Preparation requires an empty directory')
    previous = json.loads((args.previous_home / 'prepared.json').read_bytes())
    if data.identity(previous) != plan['previous_cooperative_prepared']:
        raise ValueError('Previous evidence differs from the declared source')
    if previous['plan']['reference_prepared_sha256'] != plan['reference_prepared_sha256']:
        raise ValueError('Replay source identity changed')
    snapshot = model_snapshot(args.model_dir, plan['model'])
    tokenizer = tokenizer_for(args.model_dir)
    if data.tokenizer_identity(tokenizer) != previous['tokenizer']:
        raise ValueError('Tokenizer changed')
    old_records = {}
    for role in ('train', 'dev', 'test', 'retention'):
        old_records[role] = data.read_records(args.previous_home / f'inputs/{role}.jsonl',
                                             previous['roles'][role]['sha256'])
    replay = [row for row in old_records['train'] if 'task' not in row]
    if len(replay) != plan['replay_documents']:
        raise ValueError('Replay count changed')
    excluded_ids = {row['id'] for rows in old_records.values() for row in rows if 'task' in row}
    excluded_prompts = {row['messages'][0]['content'] for rows in old_records.values()
                        for row in rows if 'task' in row}
    (home / 'inputs').mkdir(parents=True)
    roles, seen = {}, set()
    for role in ('test', 'dev', 'train'):
        count = plan['grounded_train_documents'] if role == 'train' else plan[f'{role}_documents']
        records = []
        for index in range(count):
            case = tasks.make_case(plan['task_seed'], role, index)
            prompt, identifier = tasks.prompt(case), tasks.task_identity(case)
            if identifier in excluded_ids or prompt in excluded_prompts or identifier in seen:
                raise ValueError('Generated records overlap a previous or current partition')
            seen.add(identifier)
            excluded_prompts.add(prompt)
            answer = json.dumps(tasks.expected(case), separators=(',', ':'))
            if not tasks.check_answer(case, answer)['correct']:
                raise ValueError('Generated target fails its executable checker')
            messages = [{'role': 'user', 'content': prompt}, {'role': 'assistant', 'content': answer}]
            records.append({'id': identifier, 'task': case, 'messages': messages,
                            'loss_weight': plan['grounded_token_weight'],
                            **data.conversation(tokenizer, messages, plan['max_length'])})
        if role == 'train':
            records += replay
        roles[role] = write_records(home / f'inputs/{role}.jsonl', records)
    roles['retention'] = write_records(home / 'inputs/retention.jsonl', old_records['retention'])
    prepared = {'plan': plan, 'implementation': implementation(), 'model_snapshot': snapshot,
                'tokenizer': data.tokenizer_identity(tokenizer), 'roles': roles,
                'excluded_previous_task_ids': sorted(excluded_ids),
                'retention_scope': 'Previously exposed public probes; no broad retention certification'}
    data.save(home / 'prepared.json', prepared)
    emit('prepared', identity=data.identity(prepared))


def inputs(args, plan):
    prepared = json.loads((args.home / 'prepared.json').read_bytes())
    if prepared['plan'] != plan or prepared['implementation'] != implementation():
        raise ValueError('Plan or numerical source changed after preparation')
    committed(PREPARATION, prepared)
    return prepared


def partition(home, prepared, role, allow_test=False):
    if role == 'test' and not allow_test:
        raise ValueError('Training cannot open the final test')
    records = data.read_records(home / f'inputs/{role}.jsonl', prepared['roles'][role]['sha256'])
    if [row['id'] for row in records] != prepared['roles'][role]['ids']:
        raise ValueError('Partition ordering changed')
    return records


def numerical_runtime(device, threads):
    runtime = engine.configure(device, threads)
    runtime['outer_device'] = 'cpu-float32'
    runtime['nccl_algorithm'] = os.environ.get('NCCL_ALGO', '')
    runtime['nccl_protocol'] = os.environ.get('NCCL_PROTO', '')
    runtime['nccl_ib_disable'] = os.environ.get('NCCL_IB_DISABLE', '')
    if device == 'cuda':
        import torch
        runtime['nccl_version'] = list(torch.cuda.nccl.version())
    if device == 'cuda' and (runtime['nccl_algorithm'], runtime['nccl_protocol']) != ('Ring', 'Simple'):
        raise ValueError('Use the declared NCCL Ring/Simple profile for this experiment')
    return runtime


def available_manifests(out, steps):
    return {step: data.identity(json.loads((out / f'group-{step:06d}.json').read_bytes()))
            for step in steps if (out / f'group-{step:06d}.json').exists()}


def quarantine_ahead(out, local, step, checkpoints):
    """Preserve outputs beyond the common durable point before replaying them."""
    stamp = str(time.time_ns())
    for path in list(local.glob('checkpoint-*')) + list(local.glob('.writing-*')):
        if path.name.startswith('.writing-') or int(path.name.split('-')[-1]) > step:
            path.rename(local / ('.orphan-' + stamp + '-' + path.name))
    for value in checkpoints:
        path = out / f'group-{value:06d}.json'
        if value > step and path.exists():
            path.rename(out / ('.orphan-' + stamp + '-' + path.name))
    result = local / 'result.json'
    if result.exists():
        result.rename(local / ('.prior-result-' + stamp + '.json'))


def train(args, plan):
    import torch
    import torch.distributed as dist
    from torch.nn.parallel import DistributedDataParallel
    rank, world = int(os.environ.get('RANK', '0')), int(os.environ.get('WORLD_SIZE', '1'))
    if world != plan['arms'][args.arm] or not 0 <= rank < world:
        raise ValueError('Worker does not match the declared arm and membership')
    prepared = inputs(args, plan)
    runtime = numerical_runtime(args.device, args.threads)
    identity = data.identity(prepared)
    binding = windows.rank_binding(identity, runtime, args.arm, rank, world)
    out = args.home / args.arm
    local = out / f'rank-{rank}'
    local.mkdir(parents=True, exist_ok=True)
    with (local / 'worker.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        state_path = local / 'run.json'
        if state_path.exists():
            state = json.loads(state_path.read_bytes())
            if state['binding'] != binding:
                raise ValueError('Original rank, source or numerical profile changed')
        else:
            state = {'binding': binding, 'runtime': runtime, 'started': time.time(), 'rank': rank, 'world': world}
            data.save(state_path, state)
        budget = engine.Budget(args.home, state['started'], plan['budget'])
        budget.check(force_disk=True)
        if world > 1:
            dist.init_process_group('nccl' if args.device == 'cuda' else 'gloo', timeout=timedelta(seconds=300))
        try:
            common_binding = data.identity({'prepared': identity, 'profile': group.runtime_profile(runtime),
                                            'arm': args.arm, 'world': world})
            group.agree_digest(common_binding, world, args.device)
            checkpoints = plan['checkpoints'][args.arm]
            step = windows.common_checkpoint(available_manifests(out, checkpoints), checkpoints, world, args.device)
            receipt = None
            if step is not None:
                manifest = json.loads((out / f'group-{step:06d}.json').read_bytes())
                directory, receipt = windows.verify_group_checkpoint(out, manifest, identity, runtime,
                                                                      args.arm, rank, world, step)
            else:
                directory = args.model_dir
                if model_snapshot(directory, plan['model']) != prepared['model_snapshot']:
                    raise ValueError('Seed snapshot changed')
            if step == plan['training']['steps']:
                if not (local / 'result.json').exists():
                    raise ValueError('Durable final checkpoint exists without a performance receipt; preserve it for inspection')
                emit('already_completed', arm=args.arm, rank=rank, step=step)
                return
            quarantine_ahead(out, local, step or 0, checkpoints)
            tokenizer = tokenizer_for(args.model_dir)
            if data.tokenizer_identity(tokenizer) != prepared['tokenizer']:
                raise ValueError('Tokenizer changed')
            torch.manual_seed(plan['training']['seed'])
            model = engine.load_model(directory, args.device, plan['model']['parameters'])
            model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
            model.train()
            optimizer = engine.optimizer_for(model, plan['training'])
            if receipt:
                engine.restore_optimizer(directory, optimizer, args.device)
            outer = None
            if args.arm == 'diloco-four':
                outer = windows.OuterNesterov(model, plan['training']['outer_learning_rate'],
                                              plan['training']['outer_momentum'], plan['training']['outer_chunk_bytes'])
                if receipt:
                    outer.restore(directory / 'outer.pt', step // plan['training']['local_steps'])
                    group.agree_digest(outer.digest(), world, args.device)
            initial_digest = group.parameter_digest(model)
            group.agree_digest(initial_digest, world, args.device)
            if receipt and initial_digest != manifest['parameter_digest']:
                raise ValueError('Loaded model differs from the common checkpoint')
            wrapped = (DistributedDataParallel(model, device_ids=[0] if args.device == 'cuda' else None,
                                               broadcast_buffers=False, gradient_as_bucket_view=True, bucket_cap_mb=64)
                       if args.arm == 'ddp-four' else model)
            records = partition(args.home, prepared, 'train')
            recipe = plan['training']
            schedule = engine.schedule(len(records), recipe['steps'], recipe['batch_documents'], recipe['seed'])
            journal = receipt['records'] if receipt else []
            for index, row in enumerate(journal):
                expected = [records[i] for i in schedule[index]]
                if (row['step'] != index + 1 or row['documents'] != [r['id'] for r in expected]
                        or row['local_documents'] != [r['id'] for r in group.rank_records(expected, rank, world)]):
                    raise ValueError('Local checkpoint journal differs from the frozen assignment')
            started, counters = time.monotonic(), network_bytes()
            resume_step, syncs, checkpoints_measured = len(journal), [], []
            emit('training_started', arm=args.arm, rank=rank, resume_step=resume_step)
            for index in range(resume_step, recipe['steps']):
                batch = [records[i] for i in schedule[index]]
                if outer is None:
                    observation = group.step(wrapped, optimizer, batch, args.device, recipe, index,
                                             rank, world, budget.check)
                else:
                    local_batch = group.rank_records(batch, rank, world)
                    observation = group.step(model, optimizer, local_batch, args.device, recipe, index,
                                             0, 1, budget.check)
                    observation['local_documents'] = observation['documents']
                    observation['documents'] = [row['id'] for row in batch]
                    observation['loss_scope'] = 'Local token-weighted mean at the local model'
                journal.append(observation)
                emit('step', arm=args.arm, rank=rank, **observation)
                if outer is not None and (index + 1) % recipe['local_steps'] == 0:
                    sync = outer.synchronize(model, world, budget.check)
                    syncs.append({'step': index + 1, **sync})
                    emit('synchronized', arm=args.arm, rank=rank, step=index + 1, **sync)
                if index + 1 in checkpoints:
                    checkpoint_started = time.monotonic()
                    digest = group.parameter_digest(model)
                    group.agree_digest(digest, world, args.device)
                    outer_digest = outer.digest() if outer is not None else None
                    if outer is not None:
                        group.agree_digest(outer_digest, world, args.device)
                    pointer = windows.write_checkpoint(local, model, tokenizer, optimizer, outer,
                                                       index + 1, binding, journal)
                    receipts = windows.gather_digests(pointer['receipt'], world, args.device)
                    manifest = {'prepared': identity, 'arm': args.arm, 'world': world, 'step': index + 1,
                                'parameter_digest': digest, 'outer_digest': outer_digest,
                                'rank_checkpoints': [{'directory': pointer['directory'], 'receipt': value} for value in receipts]}
                    data.save(out / f'group-{index + 1:06d}.json', manifest)
                    group.agree_digest(data.identity(manifest), world, args.device)
                    data.save(local / 'latest.json', pointer)
                    elapsed = time.monotonic() - checkpoint_started
                    checkpoints_measured.append({'step': index + 1, 'seconds': elapsed})
                    emit('checkpoint_committed', arm=args.arm, rank=rank, step=index + 1,
                         group_manifest=data.identity(manifest), parameter_digest=digest, seconds=elapsed)
                    budget.check(force_disk=True)
            result = {'arm': args.arm, 'rank': rank, 'world': world, 'binding': binding, 'runtime': runtime,
                      'prepared': identity, 'resume_step': resume_step, 'steps': journal, 'synchronizations': syncs,
                      'checkpoint_measurements': checkpoints_measured, 'seconds': time.monotonic() - started,
                      'parameter_digest': manifest['parameter_digest'], 'outer_digest': manifest['outer_digest'],
                      'group_manifest': data.identity(manifest), 'candidate': pointer,
                      'network_start': counters, 'network_end': network_bytes(),
                      'peak_cuda_allocated_bytes': torch.cuda.max_memory_allocated() if args.device == 'cuda' else 0,
                      'tokens_issued': 0, 'serving_approved': False}
            data.save(local / 'result.json', result)
            emit('completed', arm=args.arm, rank=rank, parameter_digest=result['parameter_digest'], seconds=result['seconds'])
        finally:
            if world > 1 and dist.is_initialized():
                dist.destroy_process_group()


def verify_model(directory, candidate):
    receipt = json.loads((directory / 'checkpoint.json').read_bytes())
    if data.identity(receipt) != candidate['candidate']['receipt'] or receipt['binding'] != candidate['binding']:
        raise ValueError('Model receipt differs from the selected rank-zero checkpoint')
    excluded = {'optimizer.pt', 'outer.pt'}
    expected = set(receipt['files']) - excluded
    actual = {path.name for path in directory.iterdir()}
    if actual - excluded != expected | {'checkpoint.json'}:
        raise ValueError('Missing or unexpected model artifact')
    for name in expected:
        if Path(name).name != name or (directory / name).is_symlink() or data.sha256(directory / name) != receipt['files'][name]:
            raise ValueError('Model artifact checksum mismatch')
    return receipt


def select(args, plan):
    prepared = inputs(args, plan)
    candidates = {}
    for arm, world in plan['arms'].items():
        out = args.home / arm
        result = json.loads((out / 'rank-0/result.json').read_bytes())
        manifest = json.loads((out / f"group-{plan['training']['steps']:06d}.json").read_bytes())
        if (result['parameter_digest'] != manifest['parameter_digest'] or result['group_manifest'] != data.identity(manifest)
                or result['resume_step'] != 0):
            raise ValueError('Select the predetermined uninterrupted final candidate')
        windows.verify_group_checkpoint(out, manifest, data.identity(prepared), result['runtime'],
                                        arm, 0, world, plan['training']['steps'])
        # Small receipts are collected from every host; optimizer files stay at
        # their owning rank until the verified backup. The group already agreed
        # on all receipt hashes before publishing its common manifest.
        for rank in range(world):
            peer = json.loads((out / f'rank-{rank}/result.json').read_bytes())
            pointer = manifest['rank_checkpoints'][rank]
            receipt = json.loads((out / f'rank-{rank}' / pointer['directory'] / 'checkpoint.json').read_bytes())
            binding = windows.rank_binding(data.identity(prepared), result['runtime'], arm, rank, world)
            if (peer['prepared'] != data.identity(prepared) or peer['rank'] != rank or peer['world'] != world
                    or peer['arm'] != arm or peer['binding'] != binding or receipt['binding'] != binding
                    or group.runtime_profile(peer['runtime']) != group.runtime_profile(result['runtime'])
                    or peer['resume_step'] != 0 or peer['candidate'] != pointer
                    or data.identity(receipt) != pointer['receipt'] or receipt['records'] != peer['steps']
                    or peer['parameter_digest'] != manifest['parameter_digest']
                    or peer['outer_digest'] != manifest['outer_digest']
                    or peer['group_manifest'] != data.identity(manifest)):
                raise ValueError('Every uninterrupted rank must acknowledge the selected common state')
        candidates[arm] = {'candidate': result['candidate'], 'binding': result['binding'],
                           'profile': group.runtime_profile(result['runtime']),
                           'parameter_digest': result['parameter_digest'], 'group_manifest': result['group_manifest']}
    data.save(args.home / 'selection.json', {'prepared': data.identity(prepared), 'candidates': candidates})
    emit('selection_written')


def evaluate(args, plan):
    prepared = inputs(args, plan)
    runtime = numerical_runtime(args.device, args.threads)
    selected = json.loads(SELECTION.read_bytes())
    # All arms are fixed in advance. Even development scoring uses their
    # complete selection so an absent arm cannot silently disappear.
    if args.role == 'test':
        committed(SELECTION, selected)
        if selected['prepared'] != data.identity(prepared) or set(selected['candidates']) != set(plan['arms']):
            raise ValueError('Commit all final candidates against these inputs before final scoring')
    candidate = None if args.arm == 'seed' else selected['candidates'][args.arm]
    if candidate is None:
        directory = args.model_dir
        if model_snapshot(directory, plan['model']) != prepared['model_snapshot']:
            raise ValueError('Seed differs from committed inputs')
    else:
        if group.runtime_profile(runtime) != candidate['profile']:
            raise ValueError('Evaluation numerical profile differs from selected training')
        directory = args.candidate_dir or args.home / args.arm / 'rank-0' / candidate['candidate']['directory']
        verify_model(directory, candidate)
    output = args.home / f'evaluation/{args.arm}-{args.role}.json'
    marker = output.with_suffix('.started.json')
    if output.exists() or marker.exists():
        raise ValueError('Preserve prior evaluation attempts')
    tokenizer = tokenizer_for(args.model_dir)
    if data.tokenizer_identity(tokenizer) != prepared['tokenizer']:
        raise ValueError('Evaluation tokenizer changed')
    model = engine.load_model(directory, args.device, plan['model']['parameters'])
    if candidate and group.parameter_digest(model) != candidate['parameter_digest']:
        raise ValueError('Loaded model differs from its committed parameter digest')
    data.save(marker, {'started': time.time(), 'prepared': data.identity(prepared), 'candidate': candidate,
                       'runtime': runtime, 'source_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()})
    started = time.monotonic()
    budget = engine.Budget(args.home / 'evaluation', time.time(), plan['budget'])
    records = partition(args.home, prepared, args.role, allow_test=args.role == 'test')
    generations = engine.generate(model, tokenizer, records, args.device, plan['generation_tokens'], len(records), budget.check)
    checks = [{'id': row['id'], 'family': row['task']['family'], 'variant': row['task']['variant'],
               **tasks.check_answer(row['task'], answer['text'])} for row, answer in zip(records, generations)]
    retention = engine.score(model, partition(args.home, prepared, 'retention'), args.device, budget.check)
    result = {'arm': args.arm, 'role': args.role, 'prepared': data.identity(prepared), 'candidate': candidate,
              'runtime': runtime, 'generations': generations, 'checks': checks, 'retention': retention,
              'documents': len(records), 'correct': sum(row['correct'] for row in checks),
              'seconds': time.monotonic() - started, 'serving_approved': False}
    data.save(output, result)
    emit('evaluated', arm=args.arm, role=args.role, documents=len(records), correct=result['correct'])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('prepare', 'train', 'select', 'evaluate'))
    parser.add_argument('--home', type=Path, required=True)
    parser.add_argument('--model-dir', type=Path, required=True)
    parser.add_argument('--previous-home', type=Path)
    parser.add_argument('--candidate-dir', type=Path)
    parser.add_argument('--plan', type=Path, default=PLAN)
    parser.add_argument('--arm', choices=('seed', 'single', 'ddp-four', 'diloco-four'), default='single')
    parser.add_argument('--device', choices=('cpu', 'cuda'), default='cuda')
    parser.add_argument('--threads', type=int, default=2)
    parser.add_argument('--role', choices=('dev', 'test'), default='dev')
    args = parser.parse_args()
    plan = validate(json.loads(args.plan.read_bytes()))
    if args.command in ('train', 'select') and args.arm == 'seed':
        raise ValueError('The seed is not a training arm')
    globals()[args.command](args, plan)


if __name__ == '__main__':
    main()
