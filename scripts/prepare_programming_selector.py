"""Write the nearest-train Jaccard picker's immutable prompt assets.

Uses only frozen training IDs. Does not read diagnosis labels, hidden tests,
or either tail's generated answers.
"""
import argparse
import json
from pathlib import Path

from neuroshard.evolution.programming_expert import MBPP_SHA, code_prompt
from neuroshard.evolution.programming_growth import FORMAT as GROWTH_FORMAT
from neuroshard.evolution.programming_selector import FORMAT, bind_contract
from neuroshard.evolution.reference_data import identity, save, sha256


def load_mbpp(path):
    if sha256(path) != MBPP_SHA:
        raise ValueError('Source differs from the pinned benchmark')
    raw = {}
    for line in Path(path).read_text().splitlines():
        row = json.loads(line)
        raw[row['task_id']] = row
    return raw


def prompts_for(task_ids, raw):
    texts = []
    for number in task_ids:
        texts.append(code_prompt(raw[number])[0]['content'])
    return texts


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mbpp', type=Path, required=True)
    parser.add_argument('--plan', type=Path, default=Path('config/experiments/programming-growth.json'))
    parser.add_argument('--contract', type=Path,
                        default=Path('config/experiments/programming-selector-contract.json'))
    parser.add_argument('--assets', type=Path,
                        default=Path('config/experiments/programming-selector-assets.json'))
    parser.add_argument('--picker', type=Path,
                        default=Path('config/experiments/programming-selector-picker.json'))
    args = parser.parse_args()
    plan = json.loads(args.plan.read_bytes())
    contract = json.loads(args.contract.read_bytes())
    bind_contract(contract)
    if plan.get('format') != GROWTH_FORMAT or identity(plan) != contract['growth_plan']:
        raise ValueError('Growth plan is not the frozen selector baseline')
    raw = load_mbpp(args.mbpp)
    splits = plan['splits']
    assets = {
        'format': FORMAT + '/assets',
        'purpose': 'Nearest training-prompt Jaccard assets. Prompt texts only; no evaluation IDs.',
        'incumbent_prompts': prompts_for(splits['incumbent_train_task_ids'], raw),
        'added_prompts': prompts_for(splits['train_task_ids'], raw),
        'provenance': {
            'mbpp': MBPP_SHA,
            'growth_plan': identity(plan),
            'incumbent_train_task_ids': identity(splits['incumbent_train_task_ids']),
            'added_train_task_ids': identity(splits['train_task_ids']),
            'incumbent_count': len(splits['incumbent_train_task_ids']),
            'added_count': len(splits['train_task_ids']),
        },
    }
    save(args.assets, assets)
    spec = {
        'format': FORMAT + '/picker',
        'rule': 'nearest-train-jaccard',
        'purpose': 'Select added only when its nearest frozen train prompt is strictly closer than the incumbent nearest train prompt. Ties, zeros and uncertainty select incumbent. Uses the question field only.',
        'assets': identity(assets),
        'margin': 0,
        'default': 'incumbent',
        'tie': 'incumbent',
        'abstain': False,
        'uses_fields': ['question'],
        'unused_allowed_fields': ['failed_parent_program', 'public_example', 'public_feedback'],
        'word_pattern': '[a-z0-9]+',
        'jaccard': '|intersection| / max(1, |union|)',
        'case_specific_lookup_rules': False,
        'fitted_on_opened_diagnosis': False,
        'cpu_threads': 1,
        'maximum_memory_mib': 512,
        'deadline_seconds_per_call': 1.0,
        'network_allowed': False,
        'external_model_calls_allowed': False,
        'tail_forward_passes_allowed': False,
    }
    save(args.picker, spec)
    print(json.dumps({
        'assets': identity(assets),
        'picker': identity(spec),
        'assets_sha256': sha256(args.assets),
        'picker_sha256': sha256(args.picker),
        'incumbent_prompts': len(assets['incumbent_prompts']),
        'added_prompts': len(assets['added_prompts']),
    }), flush=True)


if __name__ == '__main__':
    main()
