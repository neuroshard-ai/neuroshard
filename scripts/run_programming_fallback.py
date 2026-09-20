"""Inference-only leftover fallback comparison after the rejected programming expert.

Loads the rejected trial's trained tail. Generates no original-final answers
and performs no optimizer steps. Extra attempts share one decode and a 256-token
cap; they are not equal computation.
"""
import argparse
from datetime import timedelta
import json
import os
from pathlib import Path
import time

from neuroshard.evolution import programming_fallback as experiment
from neuroshard.evolution.reference_data import identity, save, sha256, tokenizer_identity
from neuroshard.evolution import reference


def read_rows(home, prepared, plan):
    path = Path(home) / prepared['file']
    if sha256(path) != prepared['sha256']:
        raise ValueError('Frozen comparison bytes changed')
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    if [r['id'] for r in rows] != prepared['ids']:
        raise ValueError('Prepared comparison IDs changed')
    if prepared.get('task_ids') != plan['comparison']['task_ids']:
        raise ValueError('Prepared leftover IDs differ from the committed plan')
    if prepared.get('original_final_opened'):
        raise ValueError('Prepared inputs opened the original final')
    return experiment.load_comparison_rows(plan, rows)


def generate(network, tokenizer, ids, use_expert, plan):
    from neuroshard.evolution.sharded.cached_inference import generate_branch_cached
    if network.wire.rank == 3 and not use_expert:
        return None, 0.0, {}
    observation = {}
    began = time.monotonic()
    tokens = generate_branch_cached(network, ids, plan['generation_tokens'], use_expert, observation)
    return tokens, time.monotonic() - began, observation


def pack(row, arm, *, ids, text, seconds, observation, prompt_ids, prompt_kind, path, generated,
         check_seconds=0.0, route='parent'):
    return {
        'id': row['id'], 'arm': arm, 'route': route, 'ids': ids, 'text': text,
        'seconds': seconds, 'observation': observation, 'prompt_kind': prompt_kind,
        'path': path, 'generated': generated, 'prompt_ids': prompt_ids,
        'input_tokens': len(prompt_ids), 'output_tokens': len(ids or []),
        'check_seconds': check_seconds,
    }


def evaluate(network, rows, plan, home):
    from programming_sandbox import check
    wire, tokenizer = network.wire, network.tokenizer
    outputs = []
    for row in rows:
        ids = tokenizer.apply_chat_template(row['messages'], tokenize=True, add_generation_prompt=True)
        base_ids, base_seconds, base_obs = generate(network, tokenizer, ids, False, plan)
        wire.exchange(None)
        if wire.rank == 0:
            text = tokenizer.decode(base_ids, skip_special_tokens=True)
            began_check = time.monotonic()
            visible = experiment.passes(
                {'text': text}, row, experiment.visible_tests(row), check)
            check_seconds = time.monotonic() - began_check
            base = pack(row, 'base', ids=base_ids, text=text, seconds=base_seconds,
                        observation=base_obs, prompt_ids=ids, prompt_kind='original',
                        path='parent', generated=True, check_seconds=check_seconds)
        else:
            base = visible = check_seconds = None
        visible = wire.exchange(visible)[0]
        if visible:
            extra_kind = 'unused'
        else:
            expert_ids, expert_seconds, expert_obs = generate(network, tokenizer, ids, True, plan)
            failed = wire.exchange(base['text'] if wire.rank == 0 else None)[0]
            repair_prompt = experiment.repair_messages(row['messages'], failed, row['tests'][0])
            repair_ids_prompt = wire.exchange(
                tokenizer.apply_chat_template(repair_prompt, tokenize=True, add_generation_prompt=True)
                if wire.rank == 0 else None)[0]
            repair_ids, repair_seconds, repair_obs = generate(
                network, tokenizer, repair_ids_prompt, False, plan)
            wire.exchange(None)
            extra_kind = 'repair'
        if wire.rank == 0:
            if extra_kind == 'unused':
                expert = pack(row, 'expert', ids=base['ids'], text=base['text'],
                              seconds=0.0, observation={}, prompt_ids=base['prompt_ids'],
                              prompt_kind='unused', path='parent', generated=False,
                              route=base['route'])
                repair = pack(row, 'repair', ids=base['ids'], text=base['text'],
                              seconds=0.0, observation={}, prompt_ids=base['prompt_ids'],
                              prompt_kind='unused', path='parent', generated=False)
            else:
                expert = pack(row, 'expert', ids=expert_ids,
                              text=tokenizer.decode(expert_ids, skip_special_tokens=True),
                              seconds=expert_seconds, observation=expert_obs, prompt_ids=ids,
                              prompt_kind='original', path='expert', generated=True, route='code')
                repair = pack(row, 'repair', ids=repair_ids,
                              text=tokenizer.decode(repair_ids, skip_special_tokens=True),
                              seconds=repair_seconds, observation=repair_obs,
                              prompt_ids=repair_ids_prompt, prompt_kind='repair',
                              path='parent', generated=True)
            outputs.extend([base, expert, repair])
            save(home / 'comparison-outputs.json', outputs)
            print(json.dumps({'phase': 'comparison', 'completed': len(outputs) // 3,
                              'visible_base_passed': visible,
                              'input_tokens': base['input_tokens'],
                              'check_seconds': base['check_seconds']}), flush=True)
    report = experiment.score_comparison(rows, outputs, plan, check) if wire.rank == 0 else None
    return wire.exchange(report)[0]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--home', type=Path, required=True)
    parser.add_argument('--seed', type=Path, required=True)
    parser.add_argument('--expert', type=Path, required=True,
                        help='Rejected trial expert directory with manifest.json')
    parser.add_argument('--device', choices=('cpu', 'cuda'), default='cuda')
    parser.add_argument('--plan', type=Path, default=Path('config/experiments/programming-fallback.json'))
    parser.add_argument('--freeze', type=Path, default=Path('config/experiments/programming-fallback-freeze.json'))
    args = parser.parse_args()
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    plan = json.loads(args.plan.read_bytes())
    selection = json.loads(Path('config/experiments/programming-expert-selection.json').read_bytes())
    freeze = json.loads(args.freeze.read_bytes())
    experiment.validate_freeze(plan, selection, freeze)
    prepared = json.loads((args.home / 'prepared.json').read_bytes())
    if prepared['plan'] != identity(plan) or prepared['original_final_opened']:
        raise ValueError('Comparison inputs do not match the frozen leftover plan')
    import torch
    import torch.distributed as dist
    from safetensors.torch import load_file
    from transformers import AutoTokenizer, LlamaConfig
    from neuroshard.evolution.sharded.branch import Network, ParentWire
    from neuroshard.evolution.sharded.model import Partition
    from neuroshard.evolution.sharded.wire import Wire
    runtime = reference.configure(args.device, 2)
    save(args.home / 'runtime.json', runtime)
    rank = int(os.environ['RANK'])
    if int(os.environ['WORLD_SIZE']) != 4:
        raise ValueError('Exactly four owners required')
    tokenizer = AutoTokenizer.from_pretrained(args.seed, local_files_only=True)
    digest = tokenizer_identity(tokenizer)
    if digest != plan['tokenizer']:
        raise ValueError('Tokenizer differs from the rejected parent trial')
    layout = plan['parent_layout'] if rank < 3 else plan['expert_layout']
    config = LlamaConfig(**selection['config'])
    config._attn_implementation = 'sdpa'
    shard = Partition(config, layout, rank, args.device, plan['parameter_limit'], inference_only=True)
    shard.load_weights(args.home / 'objects', selection['owners'][rank])
    if rank == 3:
        manifest = json.loads((args.expert / 'manifest.json').read_bytes())
        experiment.bind_execution(plan, selection, manifest, digest)
        owned = dict(shard.named_owned_parameters())
        if set(owned) != set(manifest['tensors']):
            raise ValueError('Expert checkpoint tensors do not match owner 3')
        with torch.no_grad():
            for name, spec in manifest['tensors'].items():
                path = args.expert / spec['file']
                if sha256(path) != spec['sha256']:
                    raise ValueError('Expert tensor changed')
                owned[name].copy_(load_file(path)['weight'].to(owned[name].device, owned[name].dtype))
    shard.eval()
    versions = tuple(p._version for p in shard.parameters())
    dist.init_process_group('gloo', timeout=timedelta(hours=2))
    parent_group = dist.new_group([0, 1, 2], backend='gloo', timeout=timedelta(hours=2))
    wire = Wire(rank, 4)
    network = Network(shard, wire, ParentWire(rank, parent_group) if rank < 3 else None, tokenizer, plan['split'])
    rows = read_rows(args.home, prepared, plan)
    report = evaluate(network, rows, plan, args.home)
    if versions != tuple(p._version for p in shard.parameters()):
        raise ValueError('Comparison mutated loaded weights')
    if rank == 0:
        save(args.home / 'result.json', {'plan': identity(plan), 'comparison': report,
                                         'original_final_opened': False, 'trained': False,
                                         'parent_expert': plan['parent_expert'],
                                         'tokenizer': digest})
    dist.destroy_process_group()


if __name__ == '__main__':
    main()
