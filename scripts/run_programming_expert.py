"""Four-owner cached-prefix learning and ordinary automatically routed serving.

Invoke with torchrun on the committed experiment source. The terminal candidate
is fixed; this process never changes a live model or issues native rewards.
"""
import argparse
from datetime import timedelta
import json
import os
from pathlib import Path
import subprocess
import time

from neuroshard.evolution import expert_router, programming_expert as experiment, reference
from neuroshard.evolution.reference_data import identity, save, sha256, tokenizer_identity


def validate(home, freeze_path):
    freeze = json.loads(freeze_path.read_bytes())
    tracked = subprocess.check_output(['git', 'show', 'HEAD:' + str(freeze_path)])
    if json.loads(tracked) != freeze:
        raise ValueError('The execution freeze must be committed before training')
    for name, digest in freeze['sources'].items():
        if sha256(name) != digest:
            raise ValueError('Execution source differs from the frozen candidate: ' + name)
    plan = json.loads(Path('config/experiments/programming-expert.json').read_bytes())
    selection = json.loads(Path('config/experiments/programming-expert-selection.json').read_bytes())
    prepared = json.loads((home / 'inputs/prepared.json').read_bytes())
    if (identity(plan) != freeze['plan'] or identity(selection) != freeze['selection']
            or identity(prepared) != selection['prepared'] or prepared['plan'] != identity(plan)):
        raise ValueError('Execution plan or prepared inputs changed')
    return plan, selection, prepared


def load_head(selection, objects, config, device):
    from safetensors.torch import load_file
    from neuroshard.evolution.sharded.features import FrozenHead
    weights = {}
    for name, spec in selection['head'].items():
        path = objects / spec['file']
        if sha256(path) != spec['sha256']:
            raise ValueError('Frozen read-only head differs')
        weights[name] = load_file(path)['weight'].float()
    return FrozenHead(config, weights['model.embed_tokens.weight'], weights['model.norm.weight'], device)


def train(shard, wire, plan, selection, prepared, records, home):
    import torch
    from safetensors.torch import save_file
    from neuroshard.evolution.sharded import cohort_features, feature_bank, features, incremental
    binding = {'plan': identity(plan), 'selection': identity(selection), 'prepared': identity(prepared)}
    bank_home = home / 'features'
    feature_root = cohort_features.produce(shard, wire, records, prepared['batches'], bank_home,
                                          binding, plan['split'], plan['microbatch'])
    save(home / 'feature-root.json', {'root': feature_root})
    if wire.rank != 3:
        return wire.exchange(None)[3]
    bank = feature_bank.Reader(bank_home, feature_root, binding, shard.config,
                               plan['microbatch'], len(prepared['batches']))
    head = load_head(selection, home / 'objects', shard.config, shard.device_name)
    optimizer = incremental.configure(shard, plan['split'], plan['training'])
    history = []
    for step, batch_index in enumerate(prepared['schedule']):
        batch = [records[i] for i in prepared['batches'][batch_index]]
        packets = bank.batch(batch_index, batch, shard.device_name)
        measured = features.train_step(shard, head, optimizer, packets, batch, plan['training'],
                                      step, plan['microbatch'], **plan['objective'])
        history.append(measured)
        save(home / 'training.json', history)
        print(json.dumps({'phase': 'train', **measured}), flush=True)
    checkpoint = home / 'expert'
    checkpoint.mkdir(exist_ok=False)
    manifest = {'plan': identity(plan), 'selection': identity(selection), 'feature_root': feature_root,
                'step': len(history), 'tensors': {}}
    for name, parameter in shard.named_owned_parameters():
        path = checkpoint / (name + '.safetensors')
        save_file({'weight': parameter.detach().cpu().contiguous()}, path)
        manifest['tensors'][name] = {'file': path.name, 'sha256': sha256(path)}
    # Preserve Adam in a separate safetensors inventory; serving loads weights only.
    adam = {}
    for name, parameter in shard.named_owned_parameters():
        for field, value in optimizer.state[parameter].items():
            adam[name + '/' + field] = value.detach().cpu().contiguous()
    save_file(adam, checkpoint / 'adam.safetensors')
    manifest['adam'] = sha256(checkpoint / 'adam.safetensors')
    save(checkpoint / 'manifest.json', manifest)
    del optimizer, head, bank, packets, adam
    torch.cuda.empty_cache() if shard.device_name == 'cuda' else None
    shard.requires_grad_(False).eval()
    return wire.exchange(manifest)[3]


def evaluate(network, router, features, rows, plan, home, role):
    from neuroshard.evolution.sharded.cached_inference import generate_branch_cached
    from programming_sandbox import check
    wire, tokenizer = network.wire, network.tokenizer
    outputs = []
    for row in rows:
        ids = tokenizer.apply_chat_template(row['messages'], tokenize=True, add_generation_prompt=True)
        started = time.monotonic()
        decision = expert_router.select(router, features(experiment.routing_text(row['messages']))) if wire.rank == 0 else None
        decision = wire.exchange(decision)[0]
        routing_seconds = time.monotonic() - started
        # Counterbalance arm order by task hash. Each is actual generation.
        arms = ['base', 'automatic', 'replacement']
        offset = int(row['id'][:8], 16) % 3
        arms = arms[offset:] + arms[:offset]
        base = None
        for arm in arms:
            use_expert = arm == 'replacement' or (arm == 'automatic' and decision['route'] == 'code')
            observation = {}
            began = time.monotonic()
            generated = generate_branch_cached(network, ids, plan['generation_tokens'], use_expert, observation)
            wire.exchange(None)  # Rank 3 rejoins after independent parent generation.
            elapsed = time.monotonic() - began + (routing_seconds if arm == 'automatic' else 0)
            if wire.rank == 0:
                output = {'id': row['id'], 'arm': arm, 'route': 'code' if use_expert else 'parent',
                          'ids': generated, 'text': tokenizer.decode(generated, skip_special_tokens=True),
                          'seconds': elapsed, 'observation': observation}
                outputs.append(output)
                if arm == 'base':
                    base = output
                save(home / (role + '-outputs.json'), outputs)
        if wire.rank == 0:
            # Replacing the added tail with its unmodified original is exactly
            # the base graph; reuse its token trace, not its timings for auto.
            outputs.append({**base, 'arm': 'ablated', 'derived_identical_graph': True})
            save(home / (role + '-outputs.json'), outputs)
            print(json.dumps({'phase': role, 'completed': len(outputs) // 4, 'route': decision['route']}), flush=True)
    report = experiment.score(rows, outputs, plan, check) if wire.rank == 0 else None
    report = wire.exchange(report)[0]
    if wire.rank == 0:
        save(home / (role + '-score.json'), report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--home', type=Path, required=True)
    parser.add_argument('--seed', type=Path, required=True)
    parser.add_argument('--device', choices=('cpu', 'cuda'), default='cuda')
    parser.add_argument('--freeze', type=Path, default=Path('config/experiments/programming-expert-freeze.json'))
    args = parser.parse_args()
    plan, selection, prepared = validate(args.home, args.freeze)
    import torch
    import torch.distributed as dist
    from transformers import AutoTokenizer, LlamaConfig
    from neuroshard.evolution.sharded.branch import Network, ParentWire
    from neuroshard.evolution.sharded.model import Partition
    from neuroshard.evolution.sharded.router_features import EmbeddingFeatures
    from neuroshard.evolution.sharded.wire import Wire
    runtime = reference.configure(args.device, 2)
    save(args.home / 'runtime.json', runtime)
    rank = int(os.environ['RANK'])
    if int(os.environ['WORLD_SIZE']) != 4:
        raise ValueError('Exactly four owners required')
    tokenizer = AutoTokenizer.from_pretrained(args.seed, local_files_only=True)
    if tokenizer_identity(tokenizer) != prepared['tokenizer']:
        raise ValueError('Training and serving tokenizers differ')
    config = LlamaConfig(**selection['config'])
    config._attn_implementation = 'sdpa'
    layout = plan['parent_layout'] if rank < 3 else plan['expert_layout']
    shard = Partition(config, layout, rank, args.device, plan['parameter_limit'], inference_only=rank < 3)
    shard.load_weights(args.home / 'objects', selection['owners'][rank])
    shard.eval()
    initial_versions = tuple(p._version for p in shard.parameters())
    dist.init_process_group('gloo', timeout=timedelta(hours=3))
    parent_group = dist.new_group([0, 1, 2], backend='gloo', timeout=timedelta(hours=3))
    wire = Wire(rank, 4)
    profiles = wire.exchange({k: v for k, v in runtime.items() if k != 'host'})
    if any(profile != profiles[0] for profile in profiles):
        raise ValueError('Owners have different numerical runtimes')
    records = experiment.read_role(args.home / 'inputs', prepared, 'train')
    feature_fn, router = None, None
    if rank == 0:
        spec = selection['head']['model.embed_tokens.weight']
        feature_fn = EmbeddingFeatures(args.home / 'objects' / spec['file'], spec['sha256'],
                                       tokenizer, prepared['tokenizer'])
        samples = [{'id': r['id'], 'route': 'code' if r['kind'] == 'code' else 'parent',
                    'features': feature_fn(experiment.routing_text(r['messages']))} for r in records]
        router = expert_router.fit(samples, embedding_root=spec['sha256'], tokenizer_root=prepared['tokenizer'],
                                   **{k: plan['router'][k] for k in ('prototypes_per_route', 'minimum_margin', 'maximum_distance')})
        router = expert_router.fit_classifier(samples, router, epochs=plan['router']['epochs'], balance_classes=True)
        save(args.home / 'router.json', router)
    router = wire.exchange(router)[0]
    network = Network(shard, wire, ParentWire(rank, parent_group) if rank < 3 else None, tokenizer, plan['split'])
    # Before learning, verify that routing through the extra owner with the
    # untouched original tail reproduces the base. No final input is opened.
    from neuroshard.evolution.sharded.cached_inference import generate_branch_cached
    dev_rows = experiment.read_role(args.home / 'inputs', prepared, 'dev')
    for row in dev_rows[:2]:
        ids = tokenizer.apply_chat_template(row['messages'], tokenize=True, add_generation_prompt=True)
        baseline = generate_branch_cached(network, ids, 8, False)
        wire.exchange(None)
        unchanged_tail = generate_branch_cached(network, ids, 8, True)
        agreement = wire.exchange(baseline == unchanged_tail if rank == 0 else True)
        if not all(agreement):
            raise ValueError('The untouched additional tail changed the original computation')
    if rank == 0:
        save(args.home / 'initial-equivalence.json', {'passed': True, 'cases': 2, 'tokens_per_case': 8})
    manifest = train(shard, wire, plan, selection, prepared, records, args.home)
    if rank < 3 and initial_versions != tuple(p._version for p in shard.parameters()):
        raise ValueError('Training mutated an accepted parent partition')
    dev = evaluate(network, router, feature_fn, dev_rows,
                   plan, args.home, 'dev')
    eligible = (dev['code_net_gain'] >= 2 and dev['gates']['retention']
                and dev['gates']['latency_ratio'] and dev['gates']['latency_absolute'])
    result = {'plan': identity(plan), 'expert': identity(manifest), 'router': identity(router),
              'development': dev, 'final_opened': eligible}
    if eligible:
        if rank == 0:
            save(args.home / 'final-opening.json', result)
        result['final'] = evaluate(network, router, feature_fn,
                                  experiment.read_role(args.home / 'inputs', prepared, 'final'), plan, args.home, 'final')
    if rank == 0:
        save(args.home / 'result.json', result)
    dist.destroy_process_group()


if __name__ == '__main__':
    main()
