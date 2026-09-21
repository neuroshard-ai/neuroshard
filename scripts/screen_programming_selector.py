"""CPU selector screen: decide, then score. Do not train or launch GPUs.

Phase decide mounts parent traces only. Phase score joins saved tails after
the decision file is hashed. A missing picker execution freeze is invalid.
"""
import argparse
import json
from pathlib import Path
import sys

from neuroshard.evolution.programming_expert import MBPP_SHA, code_prompt
from neuroshard.evolution.programming_selector import (
    bind_contract, bind_picker_freeze, decide_picker_calls, load_picker, score_screen,
)
from neuroshard.evolution.reference_data import identity, save, sha256


def load_json(path):
    return json.loads(Path(path).read_bytes())


def mbpp_rows(path, task_ids):
    if sha256(path) != MBPP_SHA:
        raise ValueError('Source differs from the pinned benchmark')
    raw = {}
    for line in Path(path).read_text().splitlines():
        row = json.loads(line)
        raw[row['task_id']] = row
    rows = []
    for number in task_ids:
        item = raw[number]
        rows.append({
            'id': identity({'dataset': MBPP_SHA, 'task': number}),
            'task_id': number,
            'kind': 'code',
            'messages': code_prompt(item),
            'setup': item.get('test_setup_code') or '',
            'tests': item['test_list'],
        })
    return rows


def require_empty_setup(rows):
    if any(row.get('setup') for row in rows):
        raise ValueError('Diagnostic rows must have empty setup')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--phase', choices=('decide', 'score'), required=True)
    parser.add_argument('--mbpp', type=Path)
    parser.add_argument('--home', type=Path, required=True)
    parser.add_argument('--plan', type=Path, default=Path('config/experiments/programming-growth.json'))
    parser.add_argument('--contract', type=Path,
                        default=Path('config/experiments/programming-selector-contract.json'))
    parser.add_argument('--picker', type=Path,
                        default=Path('config/experiments/programming-selector-picker.json'))
    parser.add_argument('--assets', type=Path,
                        default=Path('config/experiments/programming-selector-assets.json'))
    parser.add_argument('--freeze', type=Path,
                        default=Path('config/experiments/programming-selector-picker-freeze.json'))
    parser.add_argument('--growth-outputs', type=Path,
                        default=Path('config/experiments/programming-growth-outputs.json'))
    parser.add_argument('--added-outputs', type=Path,
                        default=Path('config/experiments/programming-growth-added-outputs.json'))
    args = parser.parse_args()
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from programming_sandbox import check

    plan = load_json(args.plan)
    contract = load_json(args.contract)
    spec = load_json(args.picker)
    assets = load_json(args.assets)
    freeze = load_json(args.freeze)
    bind_picker_freeze(freeze, spec, assets, contract)
    if args.phase == 'decide':
        if args.mbpp is None:
            raise ValueError('Decide phase needs the pinned MBPP file')
        if sha256(args.growth_outputs) != contract['artifacts'][
                'config/experiments/programming-growth-outputs.json']:
            raise ValueError('Parent traces are not the frozen growth outputs')
        outputs = load_json(args.growth_outputs)
        parent = [out for out in outputs if out['arm'] == 'base']
        rows = mbpp_rows(args.mbpp, contract['cpu_screen']['picker_call_task_ids'])
        require_empty_setup(rows)
        picker = load_picker(spec, assets)
        decisions = decide_picker_calls(rows, parent, picker, check, contract)
        save(args.home / 'decisions.json', decisions)
        save(args.home / 'decisions-hash.json', {
            'format': contract['format'].rsplit('/', 1)[0] + '/decisions-hash',
            'sha256': sha256(args.home / 'decisions.json'),
            'identity': identity(decisions),
            'picker': decisions['picker'],
            'assets': decisions['assets'],
            'contract': identity(contract),
        })
        print(json.dumps({
            'phase': 'decide',
            'count': decisions['count'],
            'sha256': sha256(args.home / 'decisions.json'),
            'admission_evidence': False,
        }), flush=True)
        return
    recorded = load_json(args.home / 'decisions-hash.json')
    if sha256(args.home / 'decisions.json') != recorded['sha256']:
        raise ValueError('Decision file changed after the decide-phase hash')
    if args.mbpp is None:
        raise ValueError('Score phase needs the pinned MBPP file')
    if sha256(args.added_outputs) != contract['artifacts'][
            'config/experiments/programming-growth-added-outputs.json']:
        raise ValueError('Added extras are not the frozen diagnosis outputs')
    rows = mbpp_rows(args.mbpp, contract['cpu_screen']['case_task_ids'])
    require_empty_setup(rows)
    result = score_screen(
        rows, load_json(args.growth_outputs), load_json(args.added_outputs),
        load_json(args.home / 'decisions.json'), plan, contract, check)
    save(args.home / 'screen.json', result)
    print(json.dumps({
        'phase': 'score',
        'passed': result['passed'],
        'selected_correct': result['selected_correct'],
        'unique_added_recovered': result['unique_added_recovered'],
        'incumbent_successes_preserved': result['incumbent_successes_preserved'],
        'picker_p95_ms': result['picker_p95_ms'],
        'next': result['next'],
        'admission_evidence': False,
        'gpu_authorized_by_screen': False,
    }), flush=True)


if __name__ == '__main__':
    main()
