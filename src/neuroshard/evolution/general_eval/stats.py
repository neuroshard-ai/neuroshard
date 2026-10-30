"""Intervals, paired comparisons and the power calculations behind the evaluation's sizes.

- ``wilson``: 95% interval for one system's pass rate in one area.
- ``paired_bootstrap``: interval for candidate minus incumbent on the same items, resampling items within
  families (stratified), so an area's family mix is held fixed.
- ``mcnemar_items``: items needed to detect a paired difference ``delta`` given the discordant fraction.
- ``noninferiority_power``: chance that a truly equal candidate clears a non-regression margin.
"""

import math
import random

Z = {0.8: 0.8416, 0.9: 1.2816, 0.95: 1.6449, 0.975: 1.9600, 0.99: 2.3263}


def wilson(passed, n, z=1.96):
    if n == 0:
        return (0.0, 1.0)
    p = passed / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return (max(0.0, centre - half), min(1.0, centre + half))


def paired_bootstrap(cases, candidate, incumbent, samples=10000, seed=0, level=0.95):
    """``cases``: [(id, family)]; ``candidate``/``incumbent``: {id: score in [0, 1]}.

    Returns the observed mean difference and the two-sided ``level`` percentile interval."""
    groups = {}
    for key, family in cases:
        groups.setdefault(family, []).append(candidate[key] - incumbent[key])
    names = sorted(groups)
    n = sum(len(v) for v in groups.values())
    observed = sum(sum(v) for v in groups.values()) / n
    rng = random.Random(seed)
    means = []
    for _ in range(samples):
        total = 0.0
        for name in names:
            values = groups[name]
            total += sum(values[rng.randrange(len(values))] for _ in values)
        means.append(total / n)
    means.sort()
    tail = (1 - level) / 2
    return {'difference': observed, 'lower': means[int(tail * samples)],
            'upper': means[min(samples - 1, int((1 - tail) * samples))], 'samples': samples, 'seed': seed,
            'level': level}


def mcnemar_items(delta, discordant, alpha_z=Z[0.975], power_z=Z[0.8]):
    """Paired items needed (Connor 1987): ``delta`` = p(cand only) - p(inc only), ``discordant`` = their sum."""
    if not 0 < delta < discordant <= 1:
        raise ValueError('need 0 < delta < discordant <= 1')
    return math.ceil((alpha_z * math.sqrt(discordant) + power_z * math.sqrt(discordant - delta ** 2)) ** 2
                     / delta ** 2)


def unpaired_items(delta, p=0.5, alpha_z=Z[0.975], power_z=Z[0.8]):
    """Items per system to detect ``delta`` between two independent rates near ``p`` (for comparison)."""
    return math.ceil((alpha_z + power_z) ** 2 * 2 * p * (1 - p) / delta ** 2)


def noninferiority_power(n, margin, discordant, true_difference=0.0, z=Z[0.95]):
    """P(one-sided lower ``z`` bound of the paired difference > -margin) under a normal approximation."""
    se = math.sqrt((discordant - true_difference ** 2) / n)
    return 0.5 * (1 + math.erf(((true_difference + margin) / se - z) / math.sqrt(2)))


def half_width(n, p=0.5, z=1.96):
    return z * math.sqrt(p * (1 - p) / n)


def pairwise_items(delta, ties=0.3, alpha_z=Z[0.975], power_z=Z[0.8]):
    """Prompts to detect a preference rate of 0.5 + ``delta`` (ties scored 0.5; per-prompt variance (1-ties)/4)."""
    return math.ceil((alpha_z + power_z) ** 2 * (1 - ties) / 4 / delta ** 2)


def power_table():
    """The numbers quoted in docs/GENERAL_CAPABILITY_EVAL.md, recomputable."""
    return {
        'unpaired_items_delta05_p05': unpaired_items(0.05, 0.5),
        'unpaired_items_delta05_p07': unpaired_items(0.05, 0.7),
        'mcnemar_items_delta05': {str(d): mcnemar_items(0.05, d) for d in (0.08, 0.10, 0.12, 0.15, 0.20)},
        'mcnemar_items_delta05_one_sided': {str(d): mcnemar_items(0.05, d, alpha_z=Z[0.95])
                                            for d in (0.10, 0.12, 0.15)},
        'half_width_n400_p05': round(half_width(400), 4),
        'half_width_n20_p05': round(half_width(20), 4),
        'noninferiority_power_n400': {f'margin{m}_disc{d}': round(noninferiority_power(400, m, d), 3)
                                      for m in (0.03, 0.05) for d in (0.10, 0.12, 0.15)},
        'pairwise_prompts_delta05_ties03': pairwise_items(0.05, 0.3),
        'pairwise_prompts_delta07_ties03': pairwise_items(0.07, 0.3),
    }
