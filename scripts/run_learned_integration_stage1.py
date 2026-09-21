"""Bind the stage-1 CPU execution freeze, or run the committed CPU loop.

Without --run this only checks that the freeze binds. --run trains the new
last-layer expert and gate versus the matched no-expansion control, then scores
development, code retention, and eight general parent responses. It still
refuses GPUs and confirmation.
"""
import argparse
import json
from pathlib import Path

from neuroshard.evolution.learned_integration import load_spec
from neuroshard.evolution.learned_integration_execution import (
    authorize_cpu_run, bind_execution, load_method,
)
from neuroshard.evolution.learned_integration_run import run_stage1
from neuroshard.evolution.seed import verify


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', type=Path, default=Path('config/experiments/learned-integration.json'))
    parser.add_argument('--method', type=Path, default=Path('config/experiments/learned-integration-method.json'))
    parser.add_argument('--execution', type=Path,
                        default=Path('config/experiments/learned-integration-execution.json'))
    parser.add_argument('--seed', type=Path, help='Local SmolLM2-135M-Instruct directory')
    parser.add_argument('--home', type=Path, help='Empty study directory for prepared inputs and scores')
    parser.add_argument('--mbpp', type=Path, help='Pinned MBPP jsonl')
    parser.add_argument('--general-train', type=Path,
                        help='Local programming-expert train.jsonl containing the eight frozen general identities')
    parser.add_argument('--run', action='store_true',
                        help='Authorize the committed CPU run. Still refuses GPUs.')
    args = parser.parse_args()
    spec = json.loads(args.plan.read_bytes()) if args.plan.exists() else load_spec()
    method = json.loads(args.method.read_bytes()) if args.method.exists() else load_method()
    execution = json.loads(args.execution.read_bytes())
    if args.run:
        binding = authorize_cpu_run(spec, method, execution)
        if args.seed is None or args.home is None or args.mbpp is None or args.general_train is None:
            raise SystemExit('CPU run requires --seed, --home, --mbpp, and --general-train')
        verify(args.seed)
        result = run_stage1(
            home=args.home, seed=args.seed, spec=spec, method=method, execution=execution,
            mbpp=args.mbpp, general_train=args.general_train)
        print(json.dumps({'authorized': 'cpu', 'gpu_launch_authorized': False,
                          'confirmation_opened': False, **binding,
                          'passed': result['development_gate']['passed'],
                          'next': result['next']}))
        return
    binding = bind_execution(spec, method, execution)
    print(json.dumps({'bound': True, 'gpu_launch_authorized': False, **binding}))


if __name__ == '__main__':
    main()
