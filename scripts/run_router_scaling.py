"""Router scaling study: add routed units one at a time and measure what per-turn routing keeps.

    PYTHONPATH=src python scripts/run_router_scaling.py --encoder hashed --out .neuroshard/router-scaling/hashed
    PYTHONPATH=src python scripts/run_router_scaling.py --encoder lm --model .neuroshard/seed-smollm2-135m \
        --out .neuroshard/router-scaling/smollm2

Local CPU only; trains no unit and opens no sealed split. See docs/ROUTER_SCALING.md.
"""

import argparse
import json
from pathlib import Path
import sys
import time

from neuroshard.evolution import router_scaling as scaling
from neuroshard.evolution import router_scaling_data as data
from neuroshard.evolution import router_scaling_features as encoders
from neuroshard.evolution.modular_reference_execution import identity


def build(real_limit, cross_per_pair):
    fit = data.real_cases('fit', limit=real_limit) + data.cases('fit', cross_per_pair=cross_per_pair)
    test = data.real_cases('test') + data.cases('test', cross_per_pair=cross_per_pair)
    unseen = data.cases('unseen', cross_per_pair=cross_per_pair)
    return fit, {'test': test, 'unseen': unseen}


def table(rows):
    lines = ['| routes | added | strategy | turn acc | episode acc | min recall (route) | worst confusion | lost / kept |',
             '| --- | --- | --- | --- | --- | --- | --- | --- |']
    for row in rows:
        lost = '–' if row['lost'] is None else f"{row['lost']} / {row['previously_correct']}"
        lines.append(f"| {row['routes']} | {row['added']} | {row['strategy']} | {row['turn_accuracy']:.3f} | "
                     f"{row['episode_accuracy']:.3f} | {row['min_recall']:.3f} ({row['min_recall_route']}) | "
                     f"{row['worst_confusion'] or '–'} | {lost} |")
    return '\n'.join(lines)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--encoder', choices=('hashed', 'lm'), required=True)
    parser.add_argument('--model', help='local causal LM directory for --encoder lm')
    parser.add_argument('--out', required=True)
    parser.add_argument('--real-limit', type=int, default=48, help='real fit cases per source split')
    parser.add_argument('--cross-per-pair', type=int, default=1)
    parser.add_argument('--threads', type=int, default=2)
    args = parser.parse_args(argv)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    fit, sets = build(args.real_limit, args.cross_per_pair)
    everything = fit + [case for cases in sets.values() for case in cases]
    started = time.monotonic()
    if args.encoder == 'hashed':
        encode = lambda texts: [encoders.hashed(text) for text in texts]  # noqa: E731
    else:
        if not args.model:
            parser.error('--encoder lm needs --model')
        encode = encoders.LanguageModelEncoder(args.model, threads=args.threads).encode
    features = encoders.strategy_features(everything, encode)
    feature_seconds = time.monotonic() - started
    results = {}
    for rule, centre in (('centroids', 'refit'), ('centroids', 'frozen'), ('logistic', 'refit')):
        results[f'{rule}-{centre}'] = scaling.growth(data.ORDER, fit, sets, features, rule=rule, centre=centre)
    report = {'encoder': args.encoder, 'model': args.model, 'order': list(data.ORDER),
              'fit_cases': len(fit), 'fit_turns': sum(len(c['labels']) for c in fit),
              'eval_cases': {name: len(cases) for name, cases in sets.items()},
              'cases_sha256': identity([c['id'] for c in everything]), 'feature_seconds': feature_seconds,
              'checklist_credit': False, 'sealed_opened': False, 'results': results}
    (out / 'report.json').write_text(json.dumps(report, indent=1, sort_keys=True))
    sections = [f'# Router scaling: {args.encoder} encoder\n',
                f'{len(fit)} fit cases; evaluation {report["eval_cases"]}; features in {feature_seconds:.0f} s.\n']
    for name, result in results.items():
        for set_name in sets:
            sections.append(f'## {name}, {set_name}\n\n{table(scaling.summary_rows(result, set_name))}\n')
    (out / 'report.md').write_text('\n'.join(sections))
    print('\n'.join(sections))
    return 0


if __name__ == '__main__':
    sys.exit(main())
