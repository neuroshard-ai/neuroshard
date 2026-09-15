"""Compare repeated distributed updates with a local frozen-feature learner."""
from datetime import timedelta
import copy
import json
import os
from pathlib import Path
import random
import subprocess
import time

import numpy as np
import torch
import torch.distributed as dist
from safetensors.torch import load_file, save_file
from transformers import LlamaConfig

from .. import continued, incremental_capacity as contract, reference
from .. import reference_data as data
from . import features, incremental, incremental_state, portable
from .incremental_job import tokenizer_for
from .model import Partition, batch_tensors
from .wire import Wire

ROOT = Path(__file__).resolve().parents[4]
PLAN = 'config/experiments/frozen-feature-probe.json'
SOURCES = ('src/neuroshard/evolution/sharded/features.py',
           'src/neuroshard/evolution/sharded/feature_probe.py', 'scripts/run_frozen_feature_probe.py')


def read(path):
    return json.loads(Path(path).read_bytes())


def freeze():
    paths = (PLAN, *SOURCES)
    for name in paths:
        if contract.committed(ROOT / name) != (ROOT / name).read_bytes():
            raise ValueError('Commit the feature probe before numerical execution')
    plan = read(ROOT / PLAN)
    if (plan['format'] != 'neuroshard-frozen-feature-factorization-v1'
            or plan['arms'] != ['append', 'tail-control']
            or plan['unique_training_batches'] != 4 or plan['steps'] != 8
            or plan['learning_rate'] != .00005 or plan['warmup_steps'] != 1
            or plan['require_exact_weights_and_adam'] is not True
            or plan['require_exact_losses_and_gradient_norms'] is not True
            or plan['maximum_seconds_per_arm'] != 1800):
        raise ValueError('Unsupported frozen numerical comparison')
    return {'plan': plan, 'sources': {name: data.sha256(ROOT / name) for name in SOURCES},
            'commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()}


def load_head(parent, objects, device):
    values = {}
    for name in ('model.embed_tokens.weight', 'model.norm.weight'):
        spec = parent['tensors'][name]
        path = portable.tensor_path(objects, spec['sha256'])
        if path.stat().st_size != spec['bytes'] or data.sha256(path) != spec['sha256']:
            raise ValueError('Read-only head differs from the committed parent')
        values[name] = load_file(path)['weight']
    config = portable.validate(parent)
    return features.FrozenHead(config, values['model.embed_tokens.weight'], values['model.norm.weight'], device)


def save_tail(home, shard, optimizer):
    inventory = {}
    home.mkdir(parents=True, exist_ok=False)
    for index, (name, parameter) in enumerate(shard.named_owned_parameters()):
        path = home / f'tensor-{index:03d}.safetensors'
        values = {'weight': parameter.detach().cpu()}
        values.update({key: value.detach().cpu() for key, value in optimizer.state[parameter].items()})
        save_file(values, path)
        inventory[name] = {'file': path.name, 'sha256': data.sha256(path), 'bytes': path.stat().st_size}
    data.save(home / 'manifest.json', inventory)
    return data.identity(inventory)


def run(args):
    binding = freeze()
    probe = binding['plan']
    plan, prepared = contract.validate_prepared(ROOT / 'config/experiments/incremental-capacity.json',
        ROOT / 'config/experiments/incremental-capacity-prepared.json')
    parent = read(args.parent)
    if (data.identity(parent) != probe['parent'] or data.identity(prepared) != probe['prepared']
            or args.arm not in probe['arms']):
        raise ValueError('Feature probe inputs differ from its frozen experiment')
    tokenizer = tokenizer_for(plan, args.seed)
    rows = contract.read_role(prepared, args.inputs, 'train', tokenizer, plan['max_length'])
    if contract.schedule(plan, rows) != prepared['schedule']:
        raise ValueError('Feature probe changed the exact training batches')
    runtime = reference.configure('cuda', plan['threads'])
    runtime['allocator'] = os.environ['PYTORCH_CUDA_ALLOC_CONF']
    if {key: runtime[key] for key in continued.RUNTIME_KEYS} != plan['runtime']:
        raise ValueError('Feature probe runtime differs')
    arm = plan['arms'][args.arm]
    rank, world = int(os.environ['RANK']), int(os.environ['WORLD_SIZE'])
    if world != len(arm['boundaries']) - 1 or not 0 <= rank < world:
        raise ValueError('Feature probe needs the entire declared group')
    config = LlamaConfig(**{**parent['config'], 'num_hidden_layers': arm['layers']})
    config._attn_implementation = 'sdpa'
    torch.manual_seed(plan['seeds']['training'] + rank)
    random.seed(plan['seeds']['training'] + rank)
    np.random.seed(plan['seeds']['training'] + rank)
    shard = Partition(config, arm['boundaries'], rank, 'cuda', plan['parameter_limit'])
    incremental_state.initialize(shard, parent, args.objects, args.arm, arm['frozen_layers'])
    recipe = {**plan['training'], 'steps': probe['steps'], 'learning_rate': probe['learning_rate'],
              'warmup_steps': probe['warmup_steps']}
    original = incremental.reference_tail(shard, arm['frozen_layers']) if args.arm == 'tail-control' else None
    optimizer = incremental.configure(shard, arm['frozen_layers'], recipe)
    if rank == world - 1:
        cached = copy.deepcopy(shard)
        cached_optimizer = incremental.configure(cached, arm['frozen_layers'], recipe)
        head = load_head(parent, args.objects, 'cuda')
    args.home.mkdir(parents=True, exist_ok=False)
    dist.init_process_group('gloo', timeout=timedelta(seconds=300))
    wire = Wire(rank, world)
    started = time.monotonic()
    try:
        declaration = {**binding, 'prepared': data.identity(prepared), 'parent': data.identity(parent),
                       'arm': args.arm, 'recipe': recipe, 'runtime': runtime}
        if any(value != declaration for value in wire.exchange(declaration)):
            raise ValueError('Feature workers disagree on the complete experiment')
        data.save(args.home / 'started.json', {**declaration, 'rank': rank,
            'resident_parameters': shard.resident_parameters,
            'additional_cached_learner_parameters': (cached.resident_parameters + sum(p.numel() for p in head.parameters())
                                                     if rank == world - 1 else 0)})
        packets = {}
        feature_began, before_bytes = time.monotonic(), wire.sent_tensor_bytes
        for batch_index in range(probe['unique_training_batches']):
            records = [rows[index] for index in prepared['schedule'][batch_index]]
            local = []
            for offset in range(0, len(records), plan['microbatch']):
                subset = records[offset:offset + plan['microbatch']]
                ids, labels, mask, weights = batch_tensors(subset, 'cuda')
                with torch.no_grad():
                    incoming, outgoing, reference_hidden, final = incremental.forward(shard, wire, ids, mask,
                        arm['frozen_layers'], original)
                    if rank == 0:
                        wire.send(reference_hidden, world - 1)
                    if rank == world - 1:
                        teacher = wire.receive(0, (*ids.shape, config.hidden_size), 'cuda')
                        packet = {'prefix': incoming, 'reference': teacher, 'ids': ids,
                                  'labels': labels, 'mask': mask, 'weights': weights}
                        folder = args.home / 'features'
                        folder.mkdir(exist_ok=True)
                        path = folder / f'batch-{batch_index:02d}-micro-{offset:02d}.safetensors'
                        save_file({key: value.detach().cpu().clone() for key, value in packet.items()}, path)
                        digest = data.sha256(path)
                        # Exercise durable reloading rather than retaining the
                        # original producer's in-memory activation objects.
                        reloaded = {key: value.to('cuda') for key, value in load_file(path).items()}
                        if any(not torch.equal(value, reloaded[key]) for key, value in packet.items()):
                            raise ValueError('Serialized features changed')
                        data.save(path.with_suffix('.json'), {'sha256': digest, 'bytes': path.stat().st_size,
                            'records': data.identity(subset), 'source': declaration,
                            'cut_layer': arm['frozen_layers'], 'reference_layers': parent['config']['num_hidden_layers']})
                        local.append(reloaded)
                        del packet, teacher, reloaded
                del ids, labels, mask, weights, incoming, outgoing, reference_hidden, final
            if rank == world - 1:
                packets[batch_index] = local
        feature_seconds, feature_bytes = time.monotonic() - feature_began, wire.sent_tensor_bytes - before_bytes
        reports = []
        for step in range(probe['steps']):
            if time.monotonic() - started > probe['maximum_seconds_per_arm']:
                raise TimeoutError('Feature probe deadline')
            batch_index = step % probe['unique_training_batches']
            records = [rows[index] for index in prepared['schedule'][batch_index]]
            before_bytes = wire.sent_tensor_bytes
            observed = incremental.train_step(shard, optimizer, wire, records, recipe, step, plan['microbatch'],
                arm['frozen_layers'], reference_layers=original, **plan['reference'])
            check = None
            if rank == world - 1:
                local = features.train_step(cached, head, cached_optimizer, packets[batch_index], records,
                    recipe, step, plan['microbatch'], **plan['reference'])
                same = True
                maximum = 0.
                for (name, left), (other, right) in zip(shard.named_owned_parameters(), cached.named_owned_parameters()):
                    if name != other:
                        raise ValueError('Cached learner changed parameter ownership')
                    same &= torch.equal(left, right)
                    maximum = max(maximum, float((left - right).detach().abs().max()))
                    for key in optimizer.state[left]:
                        same &= torch.equal(optimizer.state[left][key], cached_optimizer.state[right][key])
                check = {'exact_weights_and_adam': same, 'maximum_parameter_difference': maximum,
                    'exact_loss': local['loss'] == observed['loss'],
                    'exact_gradient_norm': local['gradient_norm'] == observed['gradient_norm'],
                    'cached': local, 'distributed': observed}
            check = wire.exchange(check)[world - 1]
            report = {'step': step + 1, 'result': check, 'sent_tensor_bytes': wire.sent_tensor_bytes - before_bytes}
            reports.append(report)
            with (args.home / 'steps.jsonl').open('a') as output:
                output.write(json.dumps(report) + '\n')
            print(json.dumps({'step': step + 1, 'rank': rank, 'exact_weights_and_adam': check['exact_weights_and_adam']}), flush=True)
        final = None
        if rank == world - 1:
            final = {'distributed': save_tail(args.home / 'distributed-tail', shard, optimizer),
                     'cached': save_tail(args.home / 'cached-tail', cached, cached_optimizer)}
        final = wire.exchange(final)[world - 1]
        passed = final['distributed'] == final['cached'] and all(row['result']['exact_weights_and_adam'] and row['result']['exact_loss']
                     and row['result']['exact_gradient_norm'] for row in reports)
        data.save(args.home / 'result.json', {'passed': passed, 'rank': rank, 'arm': args.arm,
            'binding': declaration, 'steps': reports, 'tail_states': final,
            'feature_production_seconds': feature_seconds, 'feature_production_sent_tensor_bytes': feature_bytes,
            'seconds': time.monotonic() - started, 'peak_cuda_bytes': torch.cuda.max_memory_allocated(),
            'tokens_issued': 0, 'scope': probe['scope']})
    finally:
        dist.destroy_process_group()
