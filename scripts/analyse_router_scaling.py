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


from neuroshard.evolution import router_scaling as scaling
from neuroshard.evolution import router_scaling_data as data
from neuroshard.evolution import router_scaling_features as encoders

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_router_scaling as runner  # noqa: E402


def load(out, real_limit=48, cross_per_pair=1, features=None):
    fit, sets = runner.build(real_limit, cross_per_pair)
    para_fit, para_test = runner.paraphrase_sets(fit, sets)
    everything = fit + [case for cases in sets.values() for case in cases] + para_fit + para_test
    texts = runner.study_texts(real_limit, cross_per_pair)
    layers = runner.load_vectors(features or out)
    if not layers or any(t not in vectors for vectors in layers.values() for t in texts):
        raise ValueError('cached features do not cover the study texts')
    return fit, sets, everything, layers, para_fit, para_test


def augmentation(fit, sets, para_fit, para_test, features):
    """Fit with and without paraphrased real fit turns; judge on every set and on paraphrased real test turns.

    The held-out reworded turns (``unseen``) share no frame with the paraphrases, so a gain
    there is generalisation, not memorised wording.
    """
    checks = {**sets, 'paraphrased-real-test': para_test}
    result = {'paraphrased_fit_cases': len(para_fit), 'paraphrased_test_cases': len(para_test)}
    for rule, strategy in (('centroids', 'message'), ('logistic', 'with-opening')):
        base = {name: routed(rule, features, fit, cases, strategy) for name, cases in checks.items()}
        more = {name: routed(rule, features, fit + para_fit, cases, strategy) for name, cases in checks.items()}
        entry = {}
        for name, cases in checks.items():
            before, after = scaling.evaluate(base[name], cases), scaling.evaluate(more[name], cases)
            entry[name] = {'without': before['turn_accuracy'], 'with': after['turn_accuracy'],
                           'paired': scaling.paired_bootstrap(base[name], more[name], cases),
                           'recall_without': {r: before['recall'][r] for r in ('drafting', 'scheduling')
                                              if r in before['recall']},
                           'recall_with': {r: after['recall'][r] for r in ('drafting', 'scheduling')
                                           if r in after['recall']}}
        result[f'{rule}/{strategy}'] = entry
    return result


def routed(rule, features, fit, cases, strategy, routes=data.ORDER):
    rows = scaling.turn_rows(fit, routes)
    gate = scaling.fit_centroids(features[strategy], rows) if rule == 'centroids' else \
        scaling.fit_logistic(features[strategy], rows)
    return scaling.decisions(gate, cases, features, routes, strategy)


def packaged(fit, sets, everything, vectors, targets=(0.95, 0.98, 0.99)):
    """The packaged turn router on the study's features: calibrated on fit folds, judged on test and unseen.

    Admission replays the growth order: each added route is checked against the previous
    router on the ``unseen`` set, the paraphrase check, with a 0.05 margin and 0.6 minimum.
    """
    from neuroshard.evolution import assistant_turn_router as turn_router

    per_turn = {}
    for case in everything:
        turns = turn_router.conversation_features([vectors[text] for text in case['user_turns']])
        for t, feature in enumerate(turns):
            per_turn[f'{case["id"]}#{t}'] = feature
    rows = scaling.turn_rows(fit, data.ORDER)
    fitted = turn_router.fit(per_turn, rows, fallback='parent')
    result = {'uncalibrated': {}, 'calibrated': {}, 'admission': []}
    for name, cases in sets.items():
        check = scaling.turn_rows(cases, data.ORDER)
        chosen = turn_router.route_many(fitted, [per_turn[key] for key in sorted(check)])
        result['uncalibrated'][name] = sum(c == check[k][0] for c, k in zip(chosen, sorted(check))) / len(check)
    for target in targets:
        calibrated, report = turn_router.calibrate(fitted, per_turn, rows, target=target, fallback='parent')
        entry = {'threshold': calibrated['threshold'], 'held_out': {k: report[k] for k in
                 ('held_out_coverage', 'held_out_kept_accuracy', 'held_out_accuracy', 'target_reachable')}}
        for name, cases in sets.items():
            check = scaling.turn_rows(cases, data.ORDER)
            keys = sorted(check)
            chosen = turn_router.route_many(calibrated, [per_turn[key] for key in keys])
            kept = [(c, check[k][0]) for c, k in zip(chosen, keys) if c != 'parent']
            entry[name] = {'coverage': len(kept) / len(keys),
                           'kept_accuracy': sum(c == t for c, t in kept) / len(kept) if kept else None,
                           'misrouted': sum(c != t for c, t in kept), 'turns': len(keys)}
        result['calibrated'][str(target)] = entry
    unseen = sets['unseen']
    previous = None
    for size in range(2, len(data.ORDER) + 1):
        routes = data.ORDER[:size]
        step_rows = scaling.turn_rows(fit, routes)
        candidate = turn_router.fit(per_turn, step_rows, fallback='parent')
        if previous is not None:
            check = scaling.turn_rows(unseen, routes)
            decision = turn_router.admit(previous, candidate, per_turn, check, added=routes[-1], margin=0.05, minimum=0.6)
            result['admission'].append({'added': routes[-1], 'admitted': decision['admitted'],
                                        'added_recall': decision['added_recall'],
                                        'earlier_routes_failing': decision['earlier_routes_failing'],
                                        'failing_detail': {name: {k: decision['routes'][name][k] for k in
                                                                  ('turns', 'before', 'after', 'lost', 'gained', 'p_value')}
                                                           for name in decision['earlier_routes_failing']
                                                           if name in decision['routes']}})
        previous = candidate
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--out', required=True)
    parser.add_argument('--features', help='directory with texts.json and features.npz, if not --out')
    args = parser.parse_args(argv)
    out = Path(args.out)
    fit, sets, everything, layers, para_fit, para_test = load(out, features=args.features)
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
    analysis['packaged_router'] = packaged(fit, sets, everything, layers[final])
    analysis['paraphrase_augmentation'] = augmentation(fit, sets, para_fit, para_test, per_layer[final])
    (out / 'analysis.json').write_text(json.dumps(analysis, indent=1, sort_keys=True))
    print(json.dumps(analysis['paraphrase_augmentation'], indent=1))
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
