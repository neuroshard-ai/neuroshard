"""Development and confirmation gates for the verified-experience comparison.

Inputs are rescored complete episodes for the parent, the update system and the
addition system, each with its once-per-episode selection. Protected parent
successes are fixed by the canonical baseline before any training.

A gate may also judge a candidate without a separate update control, when the
candidate is itself the update. Such a gate declares no comparison with the
update, and a gate that declares one cannot run without the control.
"""

import random

from neuroshard.evolution import assistant_workflow_data as data
from neuroshard.evolution import granite_context_reference as context

CONTROL_CHECKS = ('minimum_net_vs_update', 'p95_ratio_vs_update', 'lower_95_gain_vs_update_at_least')


def passed(rows):
    return {row['id']: bool(row['score']['passed']) for row in rows}


def paired(system, control):
    gained = sorted(k for k in system if system[k] and not control[k])
    lost = sorted(k for k in system if control[k] and not system[k])
    return {'gained': gained, 'lost': lost, 'net': len(gained) - len(lost)}


def p95(rows, routed=False):
    """Episode latency; a routed system also pays its per-episode selection forward pass."""
    if routed and any('selection_seconds' not in row for row in rows):
        raise ValueError('routed episodes must record their selection time')
    return context.percentile([row['seconds'] + (row['selection_seconds'] if routed else 0) for row in rows], .95)


def common(gate, cases, parent, update, addition, protected, name='addition'):
    """Checks shared by both gates. ``addition`` is the gated candidate, reported as ``name``;
    ``update`` is the equal-data control, or None when the gate declares no comparison with it."""
    if (update is None) != (not any(key in gate for key in CONTROL_CHECKS)):
        raise ValueError('the gate and the systems disagree on an update control')
    systems = (parent, addition) if update is None else (parent, update, addition)
    ids = {case['id'] for case in cases}
    if not all(set(passed(rows)) == ids for rows in systems):
        raise ValueError('every system must complete every gated episode')
    p, a = passed(parent), passed(addition)
    lost_protected = sorted(k for k in protected if not a.get(k))
    versus_parent = paired(a, p)
    latency = {name: p95(addition, routed=True), 'parent': p95(parent)}
    checks = {
        'total': sum(a.values()) >= gate['minimum_total'],
        'net_vs_parent': versus_parent['net'] >= gate['minimum_net_vs_parent'],
        'lost_parent_successes': len(versus_parent['lost']) <= gate['maximum_lost_parent_successes'],
        'protected': not lost_protected,
        'p95': latency[name] <= gate['p95_seconds'],
    }
    report = {'correct': {'parent': sum(p.values()), name: sum(a.values())}, 'versus_parent': versus_parent,
              'lost_protected': lost_protected, 'p95_seconds': latency, 'checks': checks}
    if update is not None:
        u = passed(update)
        latency['update'] = p95(update, routed=True)
        checks['p95_ratio'] = latency[name] <= gate['p95_ratio_vs_update'] * latency['update']
        report['correct']['update'] = sum(u.values())
        report['versus_update'] = paired(a, u)
    return report


def development(plan, cases, parent, update, addition, protected, section='development_gate', name='addition'):
    gate = plan[section]
    report = common(gate, cases, parent, update, addition, protected, name)
    if update is not None:
        report['checks']['net_vs_update'] = report['versus_update']['net'] >= gate['minimum_net_vs_update']
    report['passed'] = all(report['checks'].values())
    report['confirmation_may_open'] = report['passed']
    return report


def family_bootstrap(cases, system, control, samples, seed):
    """Lower 2.5% quantile of the per-case gain, resampling operation families."""
    families = {}
    for case in cases:
        families.setdefault(case['family'], []).append(system[case['id']] - control[case['id']])
    names = sorted(families)
    rng = random.Random(seed)
    means = []
    for _ in range(samples):
        drawn = [value for _ in names for value in families[rng.choice(names)]]
        means.append(sum(drawn) / len(drawn))
    means.sort()
    return means[int(.025 * samples)]


def confirmation(plan, cases, parent, update, addition, protected, section='confirmation_gate', name='addition'):
    gate = plan[section]
    report = common(gate, cases, parent, update, addition, protected, name)
    p, a = passed(parent), passed(addition)
    per_family = {f: sum(a[c['id']] for c in cases if c['family'] == f) for f in data.FAMILIES}
    lower_parent = family_bootstrap(cases, a, p, gate['bootstrap_samples'], gate['bootstrap_seed'])
    report.update(per_family=per_family, lower_95_gain_vs_parent=lower_parent)
    report['checks'].update(
        per_family=min(per_family.values()) >= gate['minimum_per_family'],
        lower_vs_parent=lower_parent > gate['lower_95_gain_vs_parent_strictly_above'])
    if update is not None:
        lower_update = family_bootstrap(cases, a, passed(update), gate['bootstrap_samples'], gate['bootstrap_seed'] + 1)
        report['lower_95_gain_vs_update'] = lower_update
        report['checks']['lower_vs_update'] = lower_update >= gate['lower_95_gain_vs_update_at_least']
    report['passed'] = all(report['checks'].values())
    return report
