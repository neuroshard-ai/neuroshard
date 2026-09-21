"""Decode TIES-composed extras on opened leftover failures. Do not train.

Rank 3 loads the CPU-pinned TIES expert (must hash-match
programming-growth-ties-expert.json) and decodes the 38 questions whose parent
public example already failed. Rank 0 scores extractability and complementary
coverage. The original 128-task final stays closed.
"""
import argparse
from datetime import timedelta
import importlib.util
import json
import os
from pathlib import Path
import sys

from neuroshard.evolution import programming_fallback as fallback
from neuroshard.evolution import programming_growth as experiment
from neuroshard.evolution import programming_growth_ties as ties
from neuroshard.evolution.reference_data import identity, save, sha256, tokenizer_identity
from neuroshard.evolution import reference


def growth_driver():
    path = Path(__file__).resolve().parent / 'run_programming_growth.py'
    spec = importlib.util.spec_from_file_location('programming_growth_training', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def evaluate_ties_extras(network, rows, plan, home, growth_outputs, expert_dir, expert_manifest, check):
    driver = growth_driver()
    wire, tokenizer, shard = network.wire, network.tokenizer, network.shard
    table = ties.growth_table(growth_outputs)
    if wire.rank == 3 and expert_dir is not None:
        driver.load_tail(shard, expert_dir, expert_manifest)
    outputs = []
    for row in rows:
        if wire.rank == 0:
            base = table[row['id'], 'base']
            visible = fallback.passes(base, row, fallback.visible_tests(row), check)
            prompt_ids = base['prompt_ids']
        else:
            visible = prompt_ids = None
        payload = wire.exchange({'visible': visible, 'prompt_ids': prompt_ids})[0]
        visible, prompt_ids = payload['visible'], payload['prompt_ids']
        if visible:
            continue
        ids, seconds, observation = driver.generate(network, prompt_ids, True, plan)
        wire.exchange(None)
        if wire.rank == 0:
            packed = driver.pack(
                row, ties.TIES, ids=ids,
                text=tokenizer.decode(ids, skip_special_tokens=True),
                seconds=seconds, observation=observation, prompt_ids=prompt_ids,
                prompt_kind='original', path='expert', generated=True)
            outputs.append(packed)
            save(home / 'ties-outputs.json', outputs)
            print(json.dumps({'phase': 'ties', 'completed': len(outputs)}), flush=True)
    return outputs if wire.rank == 0 else None


def score_saved(home, plan, spec, added_manifest, incumbent_manifest, check):
    driver = growth_driver()
    prepared = json.loads((home / 'inputs' / 'prepared.json').read_bytes())
    preservation = driver.read_role(home / 'inputs', prepared, 'preservation', plan)
    new_rows = driver.read_role(home / 'inputs', prepared, 'new', plan)
    growth_outputs = json.loads((home / 'inputs' / 'growth-outputs.json').read_bytes())
    added_outputs = json.loads((home / 'inputs' / 'added-outputs.json').read_bytes())
    ties_outputs = json.loads((home / 'ties-outputs.json').read_bytes())
    result = ties.score_tied_extras(
        preservation, new_rows, growth_outputs, added_outputs, ties_outputs,
        plan, spec, check, added_manifest, incumbent_manifest,
        sha256(home / 'inputs' / 'growth-outputs.json'))
    save(home / 'ties-score.json', result)
    print(json.dumps({
        'phase': 'ties-score',
        'passed': result['passed'],
        'ties_correct': result['ties_correct'],
        'extractable_ties': result['extractable'][ties.TIES],
        'unique_added_recovered': result['unique_added_recovered'],
        'incumbent_successes_preserved': result['incumbent_successes_preserved'],
        'next': result['next'],
        'admission_evidence': False,
    }), flush=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--home', type=Path, required=True)
    parser.add_argument('--seed', type=Path, required=True)
    parser.add_argument('--incumbent', type=Path, required=True)
    parser.add_argument('--added', type=Path, required=True)
    parser.add_argument('--ties-expert', type=Path, required=True)
    parser.add_argument('--device', choices=('cpu', 'cuda'), default='cuda')
    parser.add_argument('--plan', type=Path, default=Path('config/experiments/programming-growth.json'))
    parser.add_argument('--ties', type=Path, default=Path('config/experiments/programming-growth-ties.json'))
    parser.add_argument('--freeze', type=Path, default=Path('config/experiments/programming-growth-freeze.json'))
    parser.add_argument('--score-only', action='store_true')
    args = parser.parse_args()
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from programming_sandbox import check
    plan = json.loads(args.plan.read_bytes())
    spec = json.loads(args.ties.read_bytes())
    selection = json.loads(Path('config/experiments/programming-expert-selection.json').read_bytes())
    freeze = json.loads(args.freeze.read_bytes())
    experiment.validate_freeze(plan, selection, freeze)
    added_manifest = json.loads((args.added / 'manifest.json').read_bytes())
    incumbent_manifest = json.loads((args.incumbent / 'manifest.json').read_bytes())
    expert_manifest = json.loads((args.ties_expert / 'manifest.json').read_bytes())
    if identity(expert_manifest) != ties.EXPERT_IDENTITY:
        raise ValueError('TIES expert is not the frozen CPU merge')
    ties.bind_ties(plan, spec, sha256(args.home / 'inputs' / 'growth-outputs.json'),
                   added_manifest, incumbent_manifest)
    if args.score_only:
        score_saved(args.home, plan, spec, added_manifest, incumbent_manifest, check)
        return
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
    if tokenizer_identity(tokenizer) != plan['tokenizer']:
        raise ValueError('Tokenizer differs from the leftover fallback baseline')
    layout = plan['parent_layout'] if rank < 3 else plan['expert_layout']
    config = LlamaConfig(**selection['config'])
    config._attn_implementation = 'sdpa'
    shard = Partition(config, layout, rank, args.device, plan['parameter_limit'], inference_only=True)
    shard.load_weights(args.home / 'objects', selection['owners'][rank])
    shard.eval()
    dist.init_process_group('gloo', timeout=timedelta(hours=2))
    parent_group = dist.new_group([0, 1, 2], backend='gloo', timeout=timedelta(hours=2))
    wire = Wire(rank, 4)
    network = Network(shard, wire, ParentWire(rank, parent_group) if rank < 3 else None, tokenizer, plan['split'])
    driver = growth_driver()
    prepared = json.loads((args.home / 'inputs' / 'prepared.json').read_bytes())
    rows = (driver.read_role(args.home / 'inputs', prepared, 'preservation', plan)
            + driver.read_role(args.home / 'inputs', prepared, 'new', plan))
    growth_outputs = json.loads((args.home / 'inputs' / 'growth-outputs.json').read_bytes())
    ties_outputs = evaluate_ties_extras(
        network, rows, plan, args.home, growth_outputs, args.ties_expert, expert_manifest, check)
    if rank == 0:
        save(args.home / 'ties-outputs.json', ties_outputs)
        score_saved(args.home, plan, spec, added_manifest, incumbent_manifest, check)
    dist.destroy_process_group()


if __name__ == '__main__':
    main()
