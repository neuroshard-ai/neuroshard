"""Paired, group-aware scoring for the frozen causal fusion experiment."""
import random


def components(rows):
    parents = {}

    def root(value):
        parents.setdefault(value, value)
        if parents[value] != value:
            parents[value] = root(parents[value])
        return parents[value]

    for row in rows:
        for group in row['groups']:
            parents[root(group)] = root(row['groups'][0])
    return [root(row['groups'][0]) for row in rows]


def interval(rows, differences, seed, repetitions=4096):
    """Resample connected knowledge groups, preserving paired documents."""
    if not rows or len(rows) != len(differences):
        raise ValueError('Require one paired observation per evaluation document')
    grouped = {}
    for group, value in zip(components(rows), differences):
        grouped.setdefault(group, []).append(value)
    groups = list(grouped.values())
    rng, samples = random.Random(seed), []
    for _ in range(repetitions):
        selected = [groups[rng.randrange(len(groups))] for _ in groups]
        samples.append(sum(sum(values) for values in selected)/sum(map(len, selected)))
    samples.sort()
    return {'mean': sum(differences)/len(differences), 'groups': len(groups),
            'lower_95_one_sided': samples[int(.05*repetitions)],
            'upper_95_one_sided': samples[min(repetitions-1, int(.95*repetitions))]}


def score(rows, answers, losses, rule, seed):
    by_id = {row['id']: row for row in rows}
    if len(by_id) != len(rows) or set(answers) != set(by_id) or set(losses) != set(by_id):
        raise ValueError('Score every frozen evaluation document exactly once')
    strata = {}
    for kind in sorted({row['kind'] for row in rows}):
        selected = [row for row in rows if row['kind'] == kind]
        report = {'documents': len(selected)}
        for other in ('hub', 'ablation'):
            key = 'loss' if kind == 'general' else 'accuracy'
            values = [(losses[row['id']]['fusion']-losses[row['id']][other]) if key == 'loss' else
                      int(answers[row['id']]['fusion']['correct'])-int(answers[row['id']][other]['correct'])
                      for row in selected]
            report[key+'_vs_'+other] = interval(selected, values, seed)
        if kind != 'general':
            report['accuracy'] = {arm: sum(answers[row['id']][arm]['correct'] for row in selected)/len(selected)
                                  for arm in ('hub', 'fusion', 'ablation')}
        strata[kind] = report
    selected = [row for row in rows if row['kind'] in ('directory', 'protocol', 'mixed')]
    gains = {arm: interval(selected, [int(answers[row['id']]['fusion']['correct'])-
                                     int(answers[row['id']][arm]['correct']) for row in selected], seed)
             for arm in ('hub', 'ablation')}
    checks = {'specialist_gain_'+arm: value['mean'] >= rule['specialist_gain'] and
              (not rule['require_positive_lower_bound'] or value['lower_95_one_sided'] > 0)
              for arm, value in gains.items()}
    checks.update(mixed_accuracy=strata['mixed']['accuracy']['fusion'] >= rule['mixed_accuracy'],
        structured_retention=strata['structured']['accuracy_vs_hub']['mean'] >= -rule['structured_drop'],
        general_retention=strata['general']['loss_vs_hub']['upper_95_one_sided'] <= rule['general_loss_ucb'])
    return {'strata': strata, 'specialist_gains': gains, 'checks': checks, 'passed': all(checks.values())}
