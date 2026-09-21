"""Write the AST-shape picker's immutable training-gold program assets.

Uses only frozen training IDs and their public gold programs. Does not read
diagnosis labels, hidden tests, or either tail's generated answers.
"""
import argparse
import json
from pathlib import Path

from neuroshard.evolution.programming_expert import MBPP_SHA
from neuroshard.evolution.programming_growth import FORMAT as GROWTH_FORMAT
from neuroshard.evolution.programming_selector import V3_FORMAT, bind_contract
from neuroshard.evolution.reference_data import identity, save, sha256


def load_mbpp(path):
    if sha256(path) != MBPP_SHA:
        raise ValueError('Source differs from the pinned benchmark')
    raw = {}
    for line in Path(path).read_text().splitlines():
        row = json.loads(line)
        raw[row['task_id']] = row
    return raw


def programs_for(task_ids, raw):
    texts = []
    for number in task_ids:
        code = raw[number].get('code')
        if not isinstance(code, str) or not code.strip():
            raise ValueError('Missing gold program for train task %s' % number)
        texts.append(code)
    return texts


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mbpp', type=Path, required=True)
    parser.add_argument('--plan', type=Path, default=Path('config/experiments/programming-growth.json'))
    parser.add_argument('--contract', type=Path,
                        default=Path('config/experiments/programming-selector-v3-contract.json'))
    parser.add_argument('--assets', type=Path,
                        default=Path('config/experiments/programming-selector-v3-assets.json'))
    parser.add_argument('--picker', type=Path,
                        default=Path('config/experiments/programming-selector-v3-picker.json'))
    args = parser.parse_args()
    plan = json.loads(args.plan.read_bytes())
    contract = json.loads(args.contract.read_bytes())
    bind_contract(contract)
    if plan.get('format') != GROWTH_FORMAT or identity(plan) != contract['growth_plan']:
        raise ValueError('Growth plan is not the frozen selector baseline')
    raw = load_mbpp(args.mbpp)
    splits = plan['splits']
    assets = {
        'format': V3_FORMAT + '/assets',
        'purpose': 'Nearest-train AST-shape assets. Gold programs only; no evaluation IDs.',
        'incumbent_programs': programs_for(splits['incumbent_train_task_ids'], raw),
        'added_programs': programs_for(splits['train_task_ids'], raw),
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
        'format': V3_FORMAT + '/picker',
        'rule': 'nearest-train-ast-shape',
        'purpose': 'Select added only when the failed parent program AST node-type set is strictly nearer a frozen added-tail training gold program than any incumbent training gold program. Ties, unparseable parents, zeros and uncertainty select incumbent. Uses the failed parent program only.',
        'assets': identity(assets),
        'margin': 0,
        'default': 'incumbent',
        'tie': 'incumbent',
        'abstain': False,
        'uses_fields': ['failed_parent_program'],
        'unused_allowed_fields': ['question', 'public_example', 'public_feedback'],
        'feature': 'AST node type names from ast.walk',
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
        'incumbent_programs': len(assets['incumbent_programs']),
        'added_programs': len(assets['added_programs']),
    }), flush=True)


if __name__ == '__main__':
    main()
