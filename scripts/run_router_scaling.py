"""Router scaling study: add routed units one at a time and measure what per-turn routing keeps.

    PYTHONPATH=src python scripts/run_router_scaling.py --encoder hashed --out .neuroshard/router-scaling/hashed
    PYTHONPATH=src python scripts/run_router_scaling.py --encoder lm --model .neuroshard/seed-smollm2-135m \
        --out .neuroshard/router-scaling/smollm2

Language-model features are cached in ``--out/features.npz`` (every captured layer), so
reruns of the analysis do not recompute them. Local CPU only; trains no unit and opens no
sealed split. See docs/ROUTER_SCALING.md.
"""

import argparse
import json
from pathlib import Path
import random
import sys
import time

import numpy as np

from neuroshard.evolution import router_scaling as scaling
from neuroshard.evolution import router_scaling_data as data
from neuroshard.evolution import router_scaling_features as encoders
from neuroshard.evolution.modular_reference_execution import identity

RULES = (('centroids', 'refit'), ('centroids', 'frozen'), ('logistic', 'refit'))


def build(real_limit, cross_per_pair):
    fit = data.real_cases('fit', limit=real_limit) + data.cases('fit', cross_per_pair=cross_per_pair)
    test = data.real_cases('test') + data.cases('test', cross_per_pair=cross_per_pair)
    unseen = data.real_unseen_cases() + data.cases('unseen', cross_per_pair=cross_per_pair)
    return fit, {'test': test, 'unseen': unseen}


def encoded_texts(texts, args, out):
    """``{layer_name: {text: vector}}``; LM layers are cached by text digest."""
    if args.encoder == 'hashed':
        return {'hashed': {text: encoders.hashed(text) for text in texts}}
    cache = out / 'features.npz'
    digest = identity({'texts': texts, 'model': str(args.model), 'layers': args.layers})
    if cache.exists():
        stored = np.load(cache, allow_pickle=False)
        if str(stored['digest']) == digest:
            return {name: dict(zip(texts, stored[name].tolist())) for name in stored.files if name != 'digest'}
    encoder = encoders.LanguageModelEncoder(args.model, threads=args.threads)
    layers = encoder.encode(texts, layers=args.layers)
    arrays = {f'layer{layer}': np.asarray(values) for layer, values in layers.items()}
    np.savez(cache, digest=np.array(digest), **arrays)
    return {name: dict(zip(texts, array.tolist())) for name, array in arrays.items()}


def strategies_from(cases, vectors):
    return encoders.strategy_features(cases, lambda texts: [vectors[text] for text in texts])


def fmt_ci(measured, key):
    interval = (measured.get('interval') or {}).get(key)
    value = measured[key]
    return f'{value:.3f} [{interval[0]:.3f}, {interval[1]:.3f}]' if interval else f'{value:.3f}'


def table(result, set_name):
    lines = ['| routes | added | strategy | turn acc [95% CI] | episode acc [95% CI] | min recall (route) | '
             'worst confusion | lost / kept |', '| --- | --- | --- | --- | --- | --- | --- | --- |']
    for step in result['steps']:
        for strategy, report in step['strategies'].items():
            m = report[set_name]
            if not m.get('turns'):
                continue
            kept = m.get('retention')
            lost = '–' if not kept else f"{kept['lost']} / {kept['previously_correct']}"
            lines.append(f"| {step['routes']} | {step['added']} | {strategy} | {fmt_ci(m, 'turn_accuracy')} | "
                         f"{fmt_ci(m, 'episode_accuracy')} | {m['min_recall']:.3f} ({m['min_recall_route']}) | "
                         f"{(m['worst_confusion'] or {}).get('pair', '–')} | {lost} |")
    return '\n'.join(lines)


def orders(count, seed=65000):
    """The declared order, then ``count - 1`` random orders of the synthetic capabilities after the two real ones."""
    rng = random.Random(seed)
    result = [tuple(data.ORDER)]
    for _ in range(count - 1):
        rest = list(data.SYNTHETIC)
        rng.shuffle(rest)
        result.append(data.REAL + tuple(rest))
    return result


def order_spread(fit, sets, features, count, rule, strategy):
    """Final and per-size accuracy over several capability orders; only the strategy and rule given."""
    runs = []
    for order in orders(count):
        result = scaling.growth(order, fit, sets, features, rule=rule, strategies=(strategy,))
        runs.append({'order': list(order), 'by_size': {
            step['routes']: {name: step['strategies'][strategy][name].get('turn_accuracy') for name in sets}
            for step in result['steps']},
            'lost': {name: sum((step['strategies'][strategy][name].get('retention') or {}).get('lost', 0)
                               for step in result['steps']) for name in sets}})
    sizes = sorted(runs[0]['by_size'])
    spread = {name: {size: [min(r['by_size'][size][name] for r in runs if r['by_size'][size][name] is not None),
                            max(r['by_size'][size][name] for r in runs if r['by_size'][size][name] is not None)]
                     for size in sizes if all(r['by_size'][size][name] is not None for r in runs)} for name in sets}
    return {'rule': rule, 'strategy': strategy, 'orders': len(runs), 'runs': runs, 'spread': spread}


def description_router(gate, vectors_of, strategy):
    """Prototypes from unit descriptions, centred like ``gate``; ``with-opening`` repeats the description."""
    def encode(texts):
        rows = [vectors_of(text) for text in texts]
        return [list(row) + list(row) for row in rows] if strategy == 'with-opening' else rows
    return scaling.prototypes({route: data.DESCRIPTIONS[route] for route in gate['routes']}, encode,
                              mean=gate['mean'])


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--encoder', choices=('hashed', 'lm'), required=True)
    parser.add_argument('--model', help='local causal LM directory for --encoder lm')
    parser.add_argument('--layers', type=int, nargs='+', default=[8, 15, 22, -1],
                        help='hidden-state indices to capture; -1 is the accepted final-norm feature')
    parser.add_argument('--out', required=True)
    parser.add_argument('--real-limit', type=int, default=48, help='real fit cases per source split')
    parser.add_argument('--cross-per-pair', type=int, default=1)
    parser.add_argument('--orders', type=int, default=5)
    parser.add_argument('--threads', type=int, default=2)
    args = parser.parse_args(argv)
    if args.encoder == 'lm' and not args.model:
        parser.error('--encoder lm needs --model')
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    fit, sets = build(args.real_limit, args.cross_per_pair)
    everything = fit + [case for cases in sets.values() for case in cases]
    texts = sorted({text for case in everything for text in case['user_turns']}
                   | {text for cards in data.DESCRIPTIONS.values() for text in cards})
    started = time.monotonic()
    layers = encoded_texts(texts, args, out)
    feature_seconds = time.monotonic() - started
    primary = 'hashed' if args.encoder == 'hashed' else 'layer-1'
    report = {'encoder': args.encoder, 'model': args.model, 'layers': sorted(layers), 'primary': primary,
              'order': list(data.ORDER), 'fit_cases': len(fit), 'fit_turns': sum(len(c['labels']) for c in fit),
              'eval_cases': {name: len(cases) for name, cases in sets.items()},
              'eval_turns': {name: sum(len(c['labels']) for c in cases) for name, cases in sets.items()},
              'cases_sha256': identity([c['sha256'] if 'sha256' in c else c['id'] for c in everything]),
              'texts': len(texts), 'feature_seconds': feature_seconds,
              'checklist_credit': False, 'sealed_opened': False}
    features = strategies_from(everything, layers[primary])
    results = {f'{rule}-{centre}': scaling.growth(data.ORDER, fit, sets, features, rule=rule, centre=centre)
               for rule, centre in RULES}
    report['results'] = {name: {k: v for k, v in result.items() if k != 'final_gates'} for name, result in results.items()}
    # Layer choice: final size only, centroid and logistic, message and with-opening.
    report['layers_at_full_size'] = {}
    for name, vectors in sorted(layers.items()):
        per = strategies_from(everything, vectors)
        rows = scaling.turn_rows(fit, data.ORDER)
        entry = {}
        for rule in ('centroids', 'logistic'):
            for strategy in ('message', 'with-opening'):
                gate = (scaling.fit_centroids(per[strategy], rows) if rule == 'centroids'
                        else scaling.fit_logistic(per[strategy], rows))
                entry[f'{rule}/{strategy}'] = {
                    set_name: scaling.evaluate(scaling.decisions(gate, cases, per, data.ORDER, strategy), cases)
                    ['turn_accuracy'] for set_name, cases in sets.items()}
        report['layers_at_full_size'][name] = entry
    # Abstention curves and description prototypes at full size, on the primary feature.
    centroid_gates = results['centroids-refit']['final_gates']
    logistic_gates = results['logistic-refit']['final_gates']
    report['coverage'] = {f'{rule}/{strategy}': {set_name: scaling.coverage(gates[strategy], cases, features,
                                                                            data.ORDER, strategy)
                                                 for set_name, cases in sets.items()}
                          for rule, gates in (('centroids', centroid_gates), ('logistic', logistic_gates))
                          for strategy in ('message', 'with-opening')}
    report['descriptions'] = {}
    vectors_of = layers[primary].__getitem__
    for strategy in ('message', 'with-opening'):
        fitted = centroid_gates[strategy]
        described = description_router(fitted, vectors_of, strategy)
        entry = {}
        for weight in (0.0, 0.25, 0.5, 1.0):
            gate = described if weight == 1.0 else scaling.blend(fitted, described, weight)
            entry[str(weight)] = {set_name: scaling.evaluate(scaling.decisions(gate, cases, features, data.ORDER,
                                                                               strategy), cases)['turn_accuracy']
                                  for set_name, cases in sets.items()}
        report['descriptions'][strategy] = entry
    report['order_spread'] = [order_spread(fit, sets, features, args.orders, rule, strategy)
                              for rule, strategy in (('centroids', 'message'), ('logistic', 'with-opening'))]
    (out / 'report.json').write_text(json.dumps(report, indent=1, sort_keys=True))
    sections = [f'# Router scaling: {args.encoder} encoder ({primary})\n',
                f'{len(fit)} fit cases; evaluation cases {report["eval_cases"]}, turns {report["eval_turns"]}; '
                f'{len(texts)} texts, features in {feature_seconds:.0f} s. Intervals: 1,000 case-level bootstrap draws.\n']
    for name, result in results.items():
        for set_name in sets:
            sections.append(f'## {name}, {set_name}\n\n{table(result, set_name)}\n')
    (out / 'report.md').write_text('\n'.join(sections))
    print(json.dumps({key: report[key] for key in ('layers_at_full_size', 'descriptions')}, indent=1))
    print(json.dumps(report['coverage'], indent=1))
    print(json.dumps([{k: v for k, v in s.items() if k != 'runs'} for s in report['order_spread']], indent=1))
    return 0


if __name__ == '__main__':
    sys.exit(main())
