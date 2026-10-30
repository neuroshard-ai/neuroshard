"""The promotion rule: a candidate replaces the incumbent only if it improves on the protected general
evaluation without an unacceptable regression in any area, and keeps every existing narrow set.

Inputs are the two systems' private per-item rows (``runner.run(..., publish_items=True)``) on the same
opened split, the judged pairwise summary, and the narrow sets' declared results. The declaration (see
``config/experiments/general-capability-gate-template.json``) fixes every margin before the split opens.

Checks, all of which must pass:

- ``same_items``: both systems completed exactly the same items of the same committed split.
- ``non_regression[area]``: per area, the lower bound of the stratified paired-bootstrap interval of
  (candidate - incumbent) at the declared level is at least ``-margin``. Every area must pass on its own
  (an intersection-union test, so no multiplicity correction is needed).
- ``improvement``: rule ``macro`` - the lower bound of the macro-average paired difference (mean over areas,
  each area weighted equally) is above the declared floor; or rule ``targets`` - for each declared target
  area, the lower bound at the Bonferroni-adjusted level is above the floor.
- ``unsupported`` and ``errors``: the candidate has no more unsupported or system-error items than allowed.
- ``judged``: the candidate's blind pairwise preference lower bound is at least ``0.5 - margin``.
- ``preservation``: every narrow set meets its own declared minimum (documents, chat, not-found, A3's
  drafting, scheduling and cross-capability sets, each as it was qualified).
"""

import random

from neuroshard.evolution.general_eval import items as schema


def _by_id(rows):
    return {row['id']: row for row in rows}


def _strata(rows):
    strata = {}
    for row in rows:
        strata.setdefault((row['area'], row['family']), []).append(row['id'])
    return strata


def joint_bootstrap(candidate, incumbent, samples, seed, level):
    """Per-area and macro paired differences with percentile intervals, resampling items within families."""
    c, i = _by_id(candidate), _by_id(incumbent)
    strata = _strata(incumbent)
    areas = sorted({area for area, _ in strata})
    diffs = {key: [float(c[k]['passed']) - float(i[k]['passed']) for k in ids] for key, ids in strata.items()}

    def area_means(pick):
        out = {}
        for area in areas:
            total = count = 0
            for (a, _), values in diffs.items():
                if a == area:
                    drawn = pick(values)
                    total += sum(drawn)
                    count += len(drawn)
            out[area] = total / count
        return out

    observed = area_means(lambda values: values)
    rng = random.Random(seed)
    draws = [area_means(lambda values: [values[rng.randrange(len(values))] for _ in values]) for _ in range(samples)]
    tail = (1 - level) / 2

    def interval(series):
        series = sorted(series)
        return series[int(tail * samples)], series[min(samples - 1, int((1 - tail) * samples))]

    out = {area: {'difference': observed[area], 'interval': interval([d[area] for d in draws])} for area in areas}
    macro = [sum(d.values()) / len(d) for d in draws]
    out['macro'] = {'difference': sum(observed.values()) / len(observed), 'interval': interval(macro)}
    return out, draws, areas


def decide(declaration, candidate, incumbent, judged=None, preservation=None):
    checks, report = {}, {}
    separate = set(declaration.get('reported_separately', {}).get('toolsets', ()))
    if separate:
        # Decided 2026-10-30: generic function calling is reported beside the gate, not in it, until a served
        # version accepts caller-defined tools.
        report['reported_separately'] = {
            name: {'candidate': sum(r['passed'] for r in candidate if r.get('toolset') == name),
                   'incumbent': sum(r['passed'] for r in incumbent if r.get('toolset') == name),
                   'cases': sum(r.get('toolset') == name for r in incumbent)} for name in sorted(separate)}
        candidate = [r for r in candidate if r.get('toolset') not in separate]
        incumbent = [r for r in incumbent if r.get('toolset') not in separate]
    same = (sorted(r['id'] for r in candidate) == sorted(r['id'] for r in incumbent)
            and declaration['split_commitment'] is not None)
    checks['same_items'] = same
    if not same:
        return {'passed': False, 'checks': checks, 'reason': 'systems ran different items or no committed split'}
    level = declaration['level']
    summary, draws, areas = joint_bootstrap(candidate, incumbent, declaration['bootstrap_samples'],
                                            declaration['bootstrap_seed'], level)
    report['differences'] = summary
    missing = [a for a in schema.AREAS if a not in areas]
    checks['all_areas_present'] = not missing
    checks['non_regression'] = {area: summary[area]['interval'][0] >= -declaration['areas'][area]['margin']
                                for area in areas}
    rule = declaration['improvement']
    if rule['rule'] == 'macro':
        checks['improvement'] = summary['macro']['interval'][0] > rule['floor']
    else:
        targets = rule['targets']
        adjusted = 1 - (1 - level) / len(targets)
        tail = (1 - adjusted) / 2
        lower = {t: sorted(d[t] for d in draws)[int(tail * len(draws))] for t in targets}
        report['target_lower_bounds'] = {'level': adjusted, **lower}
        checks['improvement'] = all(lower[t] > rule['floor'] for t in targets)
    cand_unsupported = sum(r['unsupported'] for r in candidate)
    cand_errors = sum(bool(r['error']) for r in candidate)
    checks['unsupported'] = cand_unsupported <= sum(r['unsupported'] for r in incumbent) + declaration[
        'maximum_extra_unsupported']
    checks['errors'] = cand_errors <= declaration['maximum_system_errors']
    if declaration.get('judged'):
        rules = declaration['judged']
        consistency = rules.get('calibration', {}).get('minimum_position_consistency', 0)
        checks['judged'] = (bool(judged) and judged['prompts'] > 0 and judged['lower95'] >= 0.5 - rules['margin']
                            and judged.get('calibration_usable', False)
                            and judged.get('position_consistency', 0) >= consistency)
    for name, minimum in declaration.get('preservation', {}).items():
        if not isinstance(minimum.get('minimum_passed'), int):
            raise ValueError(f'preservation set {name} has no declared minimum; fill the template before opening')
        result = (preservation or {}).get(name)
        checks.setdefault('preservation', {})[name] = bool(result) and result['passed'] >= minimum['minimum_passed']
    flat = [v for v in checks.values() if isinstance(v, bool)] + [x for v in checks.values() if isinstance(v, dict)
                                                                  for x in v.values()]
    return {'passed': all(flat), 'checks': checks, **report}