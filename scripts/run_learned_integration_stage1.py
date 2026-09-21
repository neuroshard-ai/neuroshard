"""Bind the stage-1 CPU execution freeze. Do not launch GPUs or score confirmation.

Without --run this only checks that the freeze binds. --run requires a local
135M seed matching the pinned hashes and a committed freeze. Missing weights
are not a GPU allocation.
"""
import argparse
import json
from pathlib import Path

from neuroshard.evolution.learned_integration import load_spec
from neuroshard.evolution.learned_integration_execution import (
    authorize_cpu_run, bind_execution, load_method,
)
from neuroshard.evolution.seed import verify


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', type=Path, default=Path('config/experiments/learned-integration.json'))
    parser.add_argument('--method', type=Path, default=Path('config/experiments/learned-integration-method.json'))
    parser.add_argument('--execution', type=Path,
                        default=Path('config/experiments/learned-integration-execution.json'))
    parser.add_argument('--seed', type=Path, help='Local SmolLM2-135M-Instruct directory')
    parser.add_argument('--run', action='store_true',
                        help='Authorize the committed CPU run. Still refuses GPUs.')
    args = parser.parse_args()
    spec = json.loads(args.plan.read_bytes()) if args.plan.exists() else load_spec()
    method = json.loads(args.method.read_bytes()) if args.method.exists() else load_method()
    execution = json.loads(args.execution.read_bytes())
    if args.run:
        binding = authorize_cpu_run(spec, method, execution)
        if args.seed is None:
            raise SystemExit('CPU run requires --seed with the pinned 135M files')
        verify(args.seed)
        print(json.dumps({'authorized': 'cpu', 'gpu_launch_authorized': False, **binding}))
        raise SystemExit('Seed verified. Training loop is not started by this check-in.')
    binding = bind_execution(spec, method, execution)
    print(json.dumps({'bound': True, 'gpu_launch_authorized': False, **binding}))


if __name__ == '__main__':
    main()
