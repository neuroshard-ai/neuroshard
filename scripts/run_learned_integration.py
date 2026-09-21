"""Refuse a learned-integration run until a later execution freeze authorizes it.

The method exists. This specification does not train, score confirmation, or
launch GPUs.
"""
import argparse
import json
from pathlib import Path

from neuroshard.evolution.learned_integration import load_spec, refuse_launch


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', type=Path, default=Path('config/experiments/learned-integration.json'))
    parser.add_argument('--method', type=Path, default=Path('config/experiments/learned-integration-method.json'))
    args = parser.parse_args()
    spec = json.loads(args.plan.read_bytes()) if args.plan.exists() else load_spec()
    freeze = json.loads(args.method.read_bytes())
    refuse_launch(spec, freeze)


if __name__ == '__main__':
    main()
