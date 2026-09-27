"""Development and confirmation gates for the verified-experience comparison.

Inputs are rescored complete episodes for the parent, the update system and the
addition system, each with its once-per-episode selection. Protected parent
successes are fixed by the canonical baseline before any training.
"""

import random

from neuroshard.evolution import assistant_workflow_data as data
from neuroshard.evolution import granite_context_reference as context


def passed(rows):
    return {row['id']: bool(row['score']['passed']) for row in rows}


def paired(system, control):
    gained = sorted(k for k in system if system[k] and not control[k])
    lost = sorted(k for k in system if control[k] and not system[k])
    return {'gained': gained, 'lost': lost, 'net': len(gained) - len(lost)}


def p95(rows):
    return context.percentile([row['seconds'] for row in rows], .95)


def common(gate, cases, parent, update, addition, protected):
    ids = {case['id'] for case in cases}
    if not all(set(passed(rows)) == ids for rows in (parent, update, addition)):
        raise ValueError('every system must complete every gated episode')
    p, u, a = passed(parent), passed(update), passed(addition)
    lost_protected = sorted(k for k in protected if not a.get(k))
    versus_parent, versus_update = paired(a, p), paired(a, u)
    latency = {'addition': p95(addition), 'update': p95(update), 'parent': p95(parent)}
    checks = {
        'total': sum(a.values()) >= gate['minimum_total'],
        'net_vs_parent': versus_parent['net'] >= gate['minimum_net_vs_parent'],
        'lost_parent_successes': len(versus_parent['lost']) <= gate['maximum_lost_parent_successes'],
        'protected': not lost_protected,
        'p95': latency['addition'] <= gate['p95_seconds'],
        'p95_ratio': latency['addition'] <= gate['p95_ratio_vs_update'] * latency['update'],
    }
    return {'correct': {'parent': sum(p.values()), 'update': sum(u.values()), 'addition': sum(a.values())},
            'versus_parent': versus_parent, 'versus_update': versus_update, 'lost_protected': lost_protected,
            'p95_seconds': latency, 'checks': checks}


def development(plan, cases, parent, update, addition, protected):
    gate = plan['development_gate']
    report = common(gate, cases, parent, update, addition, protected)
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


def confirmation(plan, cases, parent, update, addition, protected):
    gate = plan['confirmation_gate']
    report = common(gate, cases, parent, update, addition, protected)
    p, u, a = passed(parent), passed(update), passed(addition)
    per_family = {f: sum(a[c['id']] for c in cases if c['family'] == f) for f in data.FAMILIES}
    lower_parent = family_bootstrap(cases, a, p, gate['bootstrap_samples'], gate['bootstrap_seed'])
    lower_update = family_bootstrap(cases, a, u, gate['bootstrap_samples'], gate['bootstrap_seed'] + 1)
    report.update(per_family=per_family, lower_95_gain_vs_parent=lower_parent, lower_95_gain_vs_update=lower_update)
    report['checks'].update(
        per_family=min(per_family.values()) >= gate['minimum_per_family'],
        lower_vs_parent=lower_parent > gate['lower_95_gain_vs_parent_strictly_above'],
        lower_vs_update=lower_update >= gate['lower_95_gain_vs_update_at_least'])
    report['passed'] = all(report['checks'].values())
    return report
