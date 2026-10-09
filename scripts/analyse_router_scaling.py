"""Paired comparisons and a phrasing-diversity ablation on cached router scaling features.

    PYTHONPATH=src python scripts/analyse_router_scaling.py --out .neuroshard/router-scaling/smollm2-v2

Reads ``features.npz`` written by ``run_router_scaling.py`` and writes ``analysis.json``.
All comparisons are at the full 14 routes with paired, conversation-level bootstrap
intervals; a difference counts only if its 95% interval excludes zero.
"""

import argparse
import json
from pathlib import Path
import sys

import numpy as np

from neuroshard.evolution import router_scaling as scaling
from neuroshard.evolution import router_scaling_data as data
from neuroshard.evolution import router_scaling_features as encoders

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_router_scaling as runner  # noqa: E402


def load(out, real_limit=48, cross_per_pair=1):
    fit, sets = runner.build(real_limit, cross_per_pair)
    everything = fit + [case for cases in sets.values() for case in cases]
    texts = sorted({text for case in everything for text in case['user_turns']}
                   | {text for cards in data.DESCRIPTIONS.values() for text in cards})
    stored = np.load(out / 'features.npz', allow_pickle=False)
    names = [name for name in stored.files if name != 'digest']
    if any(stored[name].shape[0] != len(texts) for name in names):
        raise ValueError('cached features do not match the study texts')
    layers = {name: dict(zip(texts, stored[name])) for name in names}
    return fit, sets, everything, layers


def routed(rule, features, fit, cases, strategy, routes=data.ORDER):
    rows = scaling.turn_rows(fit, routes)
    gate = scaling.fit_centroids(features[strategy], rows) if rule == 'centroids' else \
        scaling.fit_logistic(features[strategy], rows)
    return scaling.decisions(gate, cases, features, routes, strategy)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--out', required=True)
    args = parser.parse_args(argv)
    out = Path(args.out)
    fit, sets, everything, layers = load(out)
    per_layer = {name: encoders.strategy_features(everything, lambda texts, v=vectors: [v[t] for t in texts])
                 for name, vectors in layers.items()}
    final = 'layer-1' if 'layer-1' in per_layer else next(iter(per_layer))
    analysis = {'final_layer': final, 'comparisons': {}, 'diversity': {}}

    def compare(label, a, b):
        analysis['comparisons'][label] = {name: scaling.paired_bootstrap(a[name], b[name], cases)
                                          for name, cases in sets.items()}

    decisions = {}
    for name, features in per_layer.items():
        for rule in ('centroids', 'logistic'):
            for strategy in ('message', 'with-opening'):
                decisions[name, rule, strategy] = {set_name: routed(rule, features, fit, cases, strategy)
                                                   for set_name, cases in sets.items()}
    base = decisions[final, 'centroids', 'message']
    compare('logistic vs centroids (message, final layer)', base, decisions[final, 'logistic', 'message'])
    compare('with-opening vs message (centroids, final layer)', base, decisions[final, 'centroids', 'with-opening'])
    compare('with-opening vs message (logistic, final layer)', decisions[final, 'logistic', 'message'],
            decisions[final, 'logistic', 'with-opening'])
    compare('logistic with-opening vs today (centroid message)', base, decisions[final, 'logistic', 'with-opening'])
    for name in per_layer:
        if name != final:
            for rule in ('centroids', 'logistic'):
                compare(f'{name} vs final ({rule}, with-opening)', decisions[final, rule, 'with-opening'],
                        decisions[name, rule, 'with-opening'])
                compare(f'{name} vs final ({rule}, message)', decisions[final, rule, 'message'],
                        decisions[name, rule, 'message'])
    # Phrasing diversity: fit synthetic capabilities on their first template only, or on both.
    # Real anchors are unchanged (their grammar has its own fixed phrasing); only synthetic turns are scored.
    one = [case for case in fit if case.get('template', 0) == 0 or case['id'].startswith('real-')]
    synthetic = {name: [case for case in cases if not case['id'].startswith(('real-', 'unseen-real-'))]
                 for name, cases in sets.items()}
    for name in (final, *(n for n in per_layer if n != final)):
        features = per_layer[name]
        for rule, strategy in (('centroids', 'message'), ('logistic', 'with-opening')):
            narrow = {s: routed(rule, features, one, cases, strategy) for s, cases in synthetic.items()}
            broad = {s: routed(rule, features, fit, cases, strategy) for s, cases in synthetic.items()}
            analysis['diversity'][f'{name} {rule}/{strategy}'] = {
                'one_template_fit_turns': len(scaling.turn_rows(one, data.ORDER)),
                'two_template_fit_turns': len(scaling.turn_rows(fit, data.ORDER)),
                **{s: {'one_template': scaling.evaluate(narrow[s], cases)['turn_accuracy'],
                       'two_templates': scaling.evaluate(broad[s], cases)['turn_accuracy'],
                       'paired': scaling.paired_bootstrap(narrow[s], broad[s], cases)}
                   for s, cases in synthetic.items()}}
    (out / 'analysis.json').write_text(json.dumps(analysis, indent=1, sort_keys=True))
    for label, entry in analysis['comparisons'].items():
        print(f'{label}: ' + '; '.join(f"{s} {v['difference']:+.3f} [{v['interval'][0]:+.3f}, {v['interval'][1]:+.3f}]"
                                        for s, v in entry.items()))
    for label, entry in analysis['diversity'].items():
        print(f'diversity {label}: ' + '; '.join(
            f"{s} {entry[s]['one_template']:.3f}->{entry[s]['two_templates']:.3f} "
            f"[{entry[s]['paired']['interval'][0]:+.3f}, {entry[s]['paired']['interval'][1]:+.3f}]"
            for s in ('test', 'unseen')))
    return 0


if __name__ == '__main__':
    sys.exit(main())
