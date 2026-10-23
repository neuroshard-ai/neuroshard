#!/usr/bin/env python3
"""Re-execute committed growth steps under the runtime installed here; compare with the committed state root.

This qualifies a candidate numerical runtime for growth. Validators judge growth proofs by re-executing a step
(``growth_replay.judge``), so a runtime whose training step differs by one bit would judge honest contributors'
commits wrong.

Given a recipe (checked against its root) and a committed state at step ``first`` (checked against its root), this
runs steps ``first .. last-1`` exactly as ``growth_replay.train_segment`` does and compares the resulting state root
with ``--expect`` (the committed root after ``last``). It is read-only: nothing is submitted.

    python scripts/replay_growth_steps.py --recipe R.json --recipe-root ROOT --state S.state --state-root SROOT \\
        --first 24 --last 32 --expect FINAL_ROOT --units DIR --out growth-replay.json
"""
import argparse
import importlib.metadata
import json
import platform
import sys
import time
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--recipe', type=Path, required=True)
    parser.add_argument('--recipe-root', required=True)
    parser.add_argument('--state', type=Path, help='committed state bytes at step --first (default: the recipe\'s '
                        'own start state, built by growth_replay.start and checked against its committed start_root)')
    parser.add_argument('--state-root', help='the committed root of --state (default: the recipe\'s start_root)')
    parser.add_argument('--first', type=int, required=True)
    parser.add_argument('--last', type=int, required=True)
    parser.add_argument('--expect', required=True, nargs='+',
                        help='committed state root(s) the state after step --last must equal (any one of them)')
    parser.add_argument('--units', type=Path, help='unit store directory for continuation recipes')
    parser.add_argument('--warm', action='store_true',
                        help='first run one discarded step on a separate model, as a long-running trainer is warm')
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()

    from neuroshard.client import live_runtime as runtime
    runtime.configure()
    import torch
    from neuroshard.evolution import growth_replay
    from neuroshard.evolution.growth_model import model_factory
    from neuroshard.evolution.modular_reference_execution import ROOT, read

    pinned = read(ROOT / runtime.LIVE_RUNTIME)['packages']
    versions = {}
    for name in pinned:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    record = {'python': platform.python_version(), 'torch_threads': None,
              'step_threads': __import__('neuroshard.evolution.growth_replay', fromlist=['STEP_THREADS']).STEP_THREADS,
              'packages_differing_from_pin': {k: [v, versions[k]] for k, v in pinned.items() if versions[k] != v},
              'first': args.first, 'last': args.last, 'expect': args.expect}
    torch.set_num_threads(8)
    record['torch_threads'] = torch.get_num_threads()
    recipe = json.loads(args.recipe.read_text())
    if growth_replay.recipe_root(recipe, recipe['model_root']) != args.recipe_root:
        raise SystemExit('the recipe does not hash to its committed root')
    if args.units:
        growth_replay.UNIT_STORE = args.units
    execution = growth_replay.execution(recipe)
    began = time.monotonic()
    record['warm'] = args.warm
    if args.warm:  # a fresh process's first passes can round differently from every later one
        scratch = model_factory(execution)()
        scratch_state = growth_replay.start(scratch, execution)
        growth_replay.step(scratch, *scratch_state, execution, args.first)
        record['warm_root'] = growth_replay.state_root(*scratch_state)
        del scratch, scratch_state
    model = model_factory(execution)()
    if args.state is None:
        if args.first != 0:
            raise SystemExit('without --state the replay starts at step 0')
        trainable, optimizer = growth_replay.start(model, execution)
        expected_start = args.state_root or recipe['start_root']
    else:
        payload = args.state.read_bytes()
        if growth_replay.bytes_root(payload) != args.state_root:
            raise SystemExit('the state does not hash to its committed root')
        trainable, optimizer = growth_replay.load(model, execution, payload)
        expected_start = args.state_root
    record['start_root'] = growth_replay.state_root(trainable, optimizer)
    record['start_matches_committed'] = record['start_root'] == expected_start
    if not record['start_matches_committed']:
        args.out.write_text(json.dumps(record, indent=1))
        raise SystemExit(f"the start state {record['start_root'][:16]} differs from the committed {expected_start[:16]}")
    roots = growth_replay.train_segment(model, trainable, optimizer, execution, args.first, args.last)
    record.update(roots=roots, seconds=round(time.monotonic() - began, 1), final=roots[-1] if roots else None,
                  bit_identical=bool(roots) and roots[-1] in args.expect)
    args.out.write_text(json.dumps(record, indent=1))
    print(json.dumps({k: record[k] for k in ('start_matches_committed', 'bit_identical', 'final', 'expect', 'seconds',
                                             'packages_differing_from_pin')}))
    sys.exit(0 if record['bit_identical'] else 1)


if __name__ == '__main__':
    main()
