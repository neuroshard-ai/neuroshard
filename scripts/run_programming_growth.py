"""Train a disjoint second programming tail, isolate it, then compare growth.

Four owners. Rank 3 stores both extra tails and activates at most one per
request. Isolation failure stops before the new-answer slice is scored. The
original programming-expert final stays closed.
"""
import argparse
from datetime import timedelta
import importlib.util
import json
import os
from pathlib import Path
import time

from neuroshard.evolution import programming_fallback as fallback
from neuroshard.evolution import programming_growth as experiment
from neuroshard.evolution.reference_data import identity, save, sha256, tokenizer_identity
from neuroshard.evolution import reference


def expert_training():
    path = Path(__file__).resolve().parent / 'run_programming_expert.py'
    spec = importlib.util.spec_from_file_location('programming_expert_training', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def fallback_driver():
    path = Path(__file__).resolve().parent / 'run_programming_fallback.py'
    spec = importlib.util.spec_from_file_location('programming_fallback_driver', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def read_role(home, prepared, role, plan):
    spec = prepared['roles'][role]
    path = Path(home) / spec['file']
    if sha256(path) != spec['sha256']:
        raise ValueError('Frozen ' + role + ' bytes changed')
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    if [r['id'] for r in rows] != spec['ids']:
        raise ValueError('Prepared ' + role + ' IDs changed')
    return experiment.load_role_rows(plan, rows, spec['task_ids'], role=role)


def generate(network, ids, use_expert, plan):
    from neuroshard.evolution.sharded.cached_inference import generate_branch_cached
    if network.wire.rank == 3 and not use_expert:
        return None, 0.0, {}
    observation = {}
    began = time.monotonic()
    tokens = generate_branch_cached(network, ids, plan['generation_tokens'], use_expert, observation)
    return tokens, time.monotonic() - began, observation


def load_tail(shard, directory, manifest):
    from safetensors.torch import load_file
    owned = dict(shard.named_owned_parameters())
    if set(owned) != set(manifest['tensors']):
        raise ValueError('Expert checkpoint tensors do not match owner 3')
    import torch
    with torch.no_grad():
        for name, spec in manifest['tensors'].items():
            path = directory / spec['file']
            if sha256(path) != spec['sha256']:
                raise ValueError('Expert tensor changed')
            owned[name].copy_(load_file(path)['weight'].to(owned[name].device, owned[name].dtype))


def pack(row, arm, *, ids, text, seconds, observation, prompt_ids, prompt_kind, path, generated,
         check_seconds=0.0):
    return {
        'id': row['id'], 'arm': arm, 'ids': ids, 'text': text, 'seconds': seconds,
        'observation': observation, 'prompt_kind': prompt_kind, 'path': path,
        'generated': generated, 'prompt_ids': prompt_ids,
        'input_tokens': len(prompt_ids), 'output_tokens': len(ids or []),
        'check_seconds': check_seconds,
    }


def evaluate_growth(network, rows, plan, home, incumbent_dir, incumbent_manifest,
                    added_dir, added_manifest, check=None):
    """Paired parent, incumbent extra and added extra. Policy uses one extra."""
    if check is None:
        from programming_sandbox import check
    wire, tokenizer, shard = network.wire, network.tokenizer, network.shard
    outputs = []
    for row in rows:
        ids = tokenizer.apply_chat_template(row['messages'], tokenize=True, add_generation_prompt=True)
        base_ids, base_seconds, base_obs = generate(network, ids, False, plan)
        wire.exchange(None)
        if wire.rank == 0:
            text = tokenizer.decode(base_ids, skip_special_tokens=True)
            began_check = time.monotonic()
            visible = fallback.passes({'text': text}, row, fallback.visible_tests(row), check)
            check_seconds = time.monotonic() - began_check
            base = pack(row, 'base', ids=base_ids, text=text, seconds=base_seconds,
                        observation=base_obs, prompt_ids=ids, prompt_kind='original',
                        path='parent', generated=True, check_seconds=check_seconds)
        else:
            base = visible = None
        visible = wire.exchange(visible)[0]
        extras = {}
        if visible:
            extra_kind = 'unused'
        else:
            extra_kind = 'original'
            for name, directory, manifest in (
                    (experiment.INCUMBENT, incumbent_dir, incumbent_manifest),
                    (experiment.ADDED, added_dir, added_manifest)):
                if wire.rank == 3 and directory is not None:
                    load_tail(shard, directory, manifest)
                expert_ids, expert_seconds, expert_obs = generate(network, ids, True, plan)
                wire.exchange(None)
                extras[name] = (expert_ids, expert_seconds, expert_obs)
        if wire.rank == 0:
            if extra_kind == 'unused':
                unused = pack(row, experiment.INCUMBENT, ids=base['ids'], text=base['text'],
                              seconds=0.0, observation={}, prompt_ids=base['prompt_ids'],
                              prompt_kind='unused', path='parent', generated=False)
                added = dict(unused)
                added['arm'] = experiment.ADDED
                outputs.extend([base, unused, added])
            else:
                packed = [base]
                for name in (experiment.INCUMBENT, experiment.ADDED):
                    expert_ids, expert_seconds, expert_obs = extras[name]
                    packed.append(pack(
                        row, name, ids=expert_ids,
                        text=tokenizer.decode(expert_ids, skip_special_tokens=True),
                        seconds=expert_seconds, observation=expert_obs, prompt_ids=ids,
                        prompt_kind='original', path='expert', generated=True))
                outputs.extend(packed)
            save(home / 'growth-outputs.json', outputs)
            print(json.dumps({'phase': 'growth', 'completed': len(outputs) // 3,
                              'visible_base_passed': visible}), flush=True)
    return outputs if wire.rank == 0 else None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--home', type=Path, required=True)
    parser.add_argument('--seed', type=Path, required=True)
    parser.add_argument('--incumbent', type=Path, required=True,
                        help='Leftover-fallback expert directory with manifest.json')
    parser.add_argument('--device', choices=('cpu', 'cuda'), default='cuda')
    parser.add_argument('--plan', type=Path, default=Path('config/experiments/programming-growth.json'))
    parser.add_argument('--freeze', type=Path, default=Path('config/experiments/programming-growth-freeze.json'))
    args = parser.parse_args()
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    plan = json.loads(args.plan.read_bytes())
    selection = json.loads(Path('config/experiments/programming-expert-selection.json').read_bytes())
    freeze = json.loads(args.freeze.read_bytes())
    experiment.validate_freeze(plan, selection, freeze)
    prepared = json.loads((args.home / 'inputs' / 'prepared.json').read_bytes())
    if prepared['plan'] != identity(plan) or prepared['original_final_opened']:
        raise ValueError('Growth inputs do not match the frozen plan')
    import torch
    import torch.distributed as dist
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
        raise ValueError('Tokenizer differs from the leftover fallback baseline')
    layout = plan['parent_layout'] if rank < 3 else plan['expert_layout']
    config = LlamaConfig(**selection['config'])
    config._attn_implementation = 'sdpa'
    shard = Partition(config, layout, rank, args.device, plan['parameter_limit'], inference_only=rank < 3)
    shard.load_weights(args.home / 'objects', selection['owners'][rank])
    incumbent_manifest = json.loads((args.incumbent / 'manifest.json').read_bytes())
    experiment.bind_execution(plan, selection, incumbent_manifest, digest)
    if rank == 3:
        load_tail(shard, args.incumbent, incumbent_manifest)
    shard.eval()
    dist.init_process_group('gloo', timeout=timedelta(hours=6))
    parent_group = dist.new_group([0, 1, 2], backend='gloo', timeout=timedelta(hours=6))
    wire = Wire(rank, 4)
    network = Network(shard, wire, ParentWire(rank, parent_group) if rank < 3 else None, tokenizer, plan['split'])
    trainer = expert_training()
    records = read_role(args.home / 'inputs', prepared, 'train', plan)
    if rank < 3:
        initial = tuple(p._version for p in shard.parameters())
    manifest = trainer.train(shard, wire, plan, selection, prepared, records, args.home)
    if rank < 3 and initial != tuple(p._version for p in shard.parameters()):
        raise ValueError('Training mutated an accepted parent partition')
    added_dir = args.home / 'expert'
    isolation_rows = read_role(args.home / 'inputs', prepared, 'development', plan)
    isolation_home = args.home / 'isolation'
    isolation_home.mkdir(exist_ok=True)
    iso_plan = dict(plan)
    iso_plan['gate'] = plan['isolation_gate']
    isolation = fallback_driver().evaluate(network, isolation_rows, iso_plan, isolation_home)
    if rank == 0:
        isolation = {
            **isolation,
            'format': experiment.FORMAT + '/isolation',
            'admission_evidence': False,
            'original_final_opened': False,
            'new_answers_opened': False,
        }
        save(args.home / 'isolation-score.json', isolation)
    isolation = wire.exchange(isolation)[0]
    result = {'plan': identity(plan), 'added_expert': identity(manifest),
              'incumbent_expert': identity(incumbent_manifest),
              'isolation': isolation, 'original_final_opened': False,
              'new_answers_opened': False, 'growth': None}
    if isolation['passed']:
        result['new_answers_opened'] = True
        rows = (read_role(args.home / 'inputs', prepared, 'preservation', plan)
                + read_role(args.home / 'inputs', prepared, 'new', plan))
        from programming_sandbox import check
        outputs = evaluate_growth(
            network, rows, plan, args.home, args.incumbent, incumbent_manifest,
            added_dir, manifest, check)
        if rank == 0:
            result['growth'] = experiment.score_growth(
                read_role(args.home / 'inputs', prepared, 'preservation', plan),
                read_role(args.home / 'inputs', prepared, 'new', plan),
                outputs, plan, prepared['incumbent_prototypes'],
                prepared['added_prototypes'], check)
    if rank == 0:
        save(args.home / 'result.json', result)
    dist.destroy_process_group()


if __name__ == '__main__':
    main()
