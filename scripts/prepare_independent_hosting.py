"""Write the committed independent-hosting method freeze.

Does not run the CPU protocol soak against live operators. Does not launch
GPUs. Does not upgrade the public 0.4.0 chain.
"""
import argparse
import json
from pathlib import Path

from neuroshard.evolution.independent_hosting import (
    METHOD_FORMAT, bind_spec, load_spec, method_freeze,
)
from neuroshard.evolution.reference_data import save


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', type=Path, default=Path('config/experiments/independent-hosting.json'))
    parser.add_argument('--freeze', action='store_true',
                        help='Write the committed method freeze. Does not authorize a soak.')
    args = parser.parse_args()
    spec = json.loads(args.plan.read_bytes())
    bind_spec(spec)
    if not args.freeze:
        raise SystemExit('Independent hosting is specified, not executed. Pass --freeze only.')
    freeze = method_freeze()
    if freeze['format'] != METHOD_FORMAT:
        raise ValueError('Method freeze format changed')
    if identity_mismatch(spec):
        raise ValueError('Plan file is not the bound independent-hosting contract')
    path = Path('config/experiments/independent-hosting-method.json')
    if path.exists() and json.loads(path.read_bytes()) != freeze:
        raise ValueError('Preserve the earlier method freeze; this candidate cannot be silently redefined')
    save(path, freeze)
    print('Commit ' + str(path) + ' before any later soak. No GPU. Item 4 remains open.')


def identity_mismatch(spec):
    from neuroshard.evolution.independent_hosting import CONTRACT_IDENTITY
    from neuroshard.evolution.reference_data import identity
    return identity(spec) != CONTRACT_IDENTITY or identity(load_spec()) != CONTRACT_IDENTITY


if __name__ == '__main__':
    main()
