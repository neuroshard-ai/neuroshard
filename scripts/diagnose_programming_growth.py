"""Generate the added tail on opened leftover extras; do not train or merge.

The unit-merge comparison stays failed. Rank 3 loads the saved added checkpoint
and decodes the 38 questions whose parent public example already failed. Rank 0
then scores unique coverage and an oracle upper bound that is not a policy.
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
from neuroshard.evolution import programming_growth_diagnosis as diagnosis
from neuroshard.evolution.reference_data import identity, save, sha256, tokenizer_identity
from neuroshard.evolution import reference


def growth_driver():
    path = Path(__file__).resolve().parent / 'run_programming_growth.py'
    spec = importlib.util.spec_from_file_location('programming_growth_training', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def evaluate_added_extras(network, rows, plan, home, growth_outputs, added_dir, added_manifest, check):
    """One extra decode of the unchanged added tail on opened visible failures."""
    driver = growth_driver()
    wire, tokenizer, shard = network.wire, network.tokenizer, network.shard
    table = diagnosis.growth_table(growth_outputs)
    if wire.rank == 3 and added_dir is not None:
        driver.load_tail(shard, added_dir, added_manifest)
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
                row, diagnosis.ADDED, ids=ids,
                text=tokenizer.decode(ids, skip_special_tokens=True),
                seconds=seconds, observation=observation, prompt_ids=prompt_ids,
                prompt_kind='original', path='expert', generated=True)
            outputs.append(packed)
            save(home / 'added-outputs.json', outputs)
            print(json.dumps({'phase': 'added', 'completed': len(outputs)}), flush=True)
    return outputs if wire.rank == 0 else None


def score_saved(home, plan, spec, added_manifest, check):
    driver = growth_driver()
    prepared = json.loads((home / 'inputs' / 'prepared.json').read_bytes())
    preservation = driver.read_role(home / 'inputs', prepared, 'preservation', plan)
    new_rows = driver.read_role(home / 'inputs', prepared, 'new', plan)
    growth_outputs = json.loads((home / 'inputs' / 'growth-outputs.json').read_bytes())
    added_outputs = json.loads((home / 'added-outputs.json').read_bytes())
    result = diagnosis.score_complementarity(
        preservation, new_rows, growth_outputs, added_outputs, plan, spec, check)
    result['added_expert'] = identity(added_manifest)
    save(home / 'diagnosis.json', result)
    print(json.dumps({
        'phase': 'diagnosis',
        'unique_added': result['unique_added'],
        'unique_incumbent': result['unique_incumbent'],
        'oracle_gain_vs_incumbent': result['oracle_gain_vs_incumbent'],
        'next': result['next'],
        'admission_evidence': False,
    }), flush=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--home', type=Path, required=True)
    parser.add_argument('--seed', type=Path, required=True)
    parser.add_argument('--added', type=Path, required=True,
                        help='Failed-merge added-tail directory with manifest.json')
    parser.add_argument('--device', choices=('cpu', 'cuda'), default='cuda')
    parser.add_argument('--plan', type=Path, default=Path('config/experiments/programming-growth.json'))
    parser.add_argument('--diagnosis', type=Path,
                        default=Path('config/experiments/programming-growth-diagnosis.json'))
    parser.add_argument('--freeze', type=Path, default=Path('config/experiments/programming-growth-freeze.json'))
    parser.add_argument('--score-only', action='store_true',
                        help='Score existing added-outputs.json without generating')
    args = parser.parse_args()
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from programming_sandbox import check
    plan = json.loads(args.plan.read_bytes())
    spec = json.loads(args.diagnosis.read_bytes())
    selection = json.loads(Path('config/experiments/programming-expert-selection.json').read_bytes())
    freeze = json.loads(args.freeze.read_bytes())
    experiment.validate_freeze(plan, selection, freeze)
    added_manifest = json.loads((args.added / 'manifest.json').read_bytes())
    growth_outputs_path = args.home / 'inputs' / 'growth-outputs.json'
    diagnosis.bind_diagnosis(plan, spec, sha256(growth_outputs_path), added_manifest)
    if args.score_only:
        score_saved(args.home, plan, spec, added_manifest, check)
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
    added_outputs = evaluate_added_extras(
        network, rows, plan, args.home, growth_outputs, args.added, added_manifest, check)
    if rank == 0:
        save(args.home / 'added-outputs.json', added_outputs)
        score_saved(args.home, plan, spec, added_manifest, check)
    dist.destroy_process_group()


if __name__ == '__main__':
    main()
