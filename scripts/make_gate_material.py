#!/usr/bin/env python3
"""Case material for scripts/growth_quorum_gate.py: a recipe, its start-state proof bundle, the roots to claim.

- The bundle is the recipe's own start state (`growth_replay.start`), checked against the recipe's committed
  `start_root`, as the step-0 proof bundle of a one-step growth job.
- Roots are given, not computed here, each with its provenance:
  - the honest root is the one committed on the live chain;
  - the legacy root is the one an unpinned replay at 8 threads computes.
- Continuation units the recipe names are included, so every validator store can hold them.

    python scripts/make_gate_material.py --recipe R.json --units DIR --honest ROOT --legacy8 ROOT \\
        --provenance '{"honest": "...", "legacy8": "..."}' --out material.json
"""
import argparse
import base64
import hashlib
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--recipe', type=Path, required=True)
    parser.add_argument('--units', type=Path, help='unit store holding the recipe\'s continue_from unit')
    parser.add_argument('--honest', required=True)
    parser.add_argument('--legacy8', required=True)
    parser.add_argument('--provenance', default='{}')
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()

    from neuroshard.evolution import growth_replay as g
    from neuroshard.evolution.growth_model import model_factory

    recipe = json.loads(args.recipe.read_text())
    if args.units:
        g.UNIT_STORE = args.units
    execution = g.execution(recipe)
    trainable, optimizer = g.start(model_factory(execution)(), execution)
    start = g.state_root(trainable, optimizer)
    if start != recipe['start_root']:
        raise SystemExit(f'start state {start[:16]} differs from the committed start_root of the recipe')
    bundle = g.bundle(g.save(trainable, optimizer), 0)
    units = {}
    if recipe.get('continue_from'):
        sha = recipe['continue_from']
        for path in [args.units / f'{sha}.safetensors', *sorted(args.units.glob('*/trainable.safetensors'))]:
            if path.is_file() and hashlib.sha256(path.read_bytes()).hexdigest() == sha:
                units[sha] = base64.b64encode(path.read_bytes()).decode()
                break
        else:
            raise SystemExit(f'continue_from unit {sha[:16]} not found under {args.units}')
    args.out.write_text(json.dumps({
        'recipe': recipe, 'bundle_b64': base64.b64encode(bundle).decode(), 'units_b64': units,
        'roots': {'honest_root': args.honest, 'legacy8_root': args.legacy8},
        'provenance': json.loads(args.provenance), 'start_root': start,
        'proof_root': hashlib.sha256(bundle).hexdigest()}))
    print(json.dumps({'start_root': start[:16], 'proof_root': hashlib.sha256(bundle).hexdigest()[:16],
                      'units': [u[:16] for u in units], 'bytes': args.out.stat().st_size}))


if __name__ == '__main__':
    main()
