#!/usr/bin/env python3
"""Recompute local-window quality and efficiency from retained experiment evidence."""
import argparse
import json
from pathlib import Path

from neuroshard.evolution import local_windows_report as report
from neuroshard.evolution import reference_data as data


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--home', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError('Preserve previous report output')
    def read(relative):
        return json.loads((args.home / relative).read_bytes())
    prepared, selection = read('prepared.json'), read('selection.json')
    records = data.read_records(args.home / 'inputs/test.jsonl', prepared['roles']['test']['sha256'])
    evaluations = {arm: read(f'evaluation/{arm}-test.json') for arm in ('seed', *prepared['plan']['arms'])}
    training = {arm: [read(f'{arm}/rank-{rank}/result.json') for rank in range(world)]
                for arm, world in prepared['plan']['arms'].items()}
    result = {'learning': report.learning(prepared, selection, records, evaluations),
              'training': report.training(prepared, selection, training)}
    dev_records = data.read_records(args.home / 'inputs/dev.jsonl', prepared['roles']['dev']['sha256'])
    phases = {name: read(f'serving/{name}.json') for name in ('single', 'four', 'failure')}
    result['serving'] = report.serving(prepared['plan'], selection, dev_records, phases)
    contract = prepared['plan']['recovery']
    world = prepared['plan']['arms'][contract['arm']]
    resumed = [read(f'recovery/rank-{rank}/result.json') for rank in range(world)]
    comparisons = [read(f'recovery/rank-{rank}/comparison.json') for rank in range(world)]
    result['recovery'] = report.recovery(prepared, selection, training[contract['arm']], resumed, comparisons,
                                         read('recovery/fault.json'), read('recovery/failure-processes.json'),
                                         read('recovery/restart-processes.json'))
    data.save(args.output, result)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
