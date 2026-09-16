#!/usr/bin/env python3
"""Apply the frozen adaptive-shard quality gate to complete, aligned results."""
import argparse
import json
import math
from pathlib import Path

import numpy as np

from neuroshard.evolution import grounded_tasks as tasks, reference_data as data


def paired(before, after, gate):
    if not before or len(before) != len(after):
        raise ValueError('Require aligned nonempty losses')
    differences = np.asarray(after, dtype=np.float64)-np.asarray(before, dtype=np.float64)
    if not bool(np.isfinite(differences).all()):
        raise ValueError('Nonfinite quality measurements')
    generator = np.random.default_rng(gate['bootstrap_seed'])
    means = []
    for start in range(0, gate['bootstrap_samples'], 100):
        count = min(100, gate['bootstrap_samples']-start)
        indices = generator.integers(0, len(differences), size=(count, len(differences)))
        means.extend(differences[indices].mean(axis=1).tolist())
    return {'documents': len(differences), 'baseline_mean': float(np.mean(before)),
            'candidate_mean': float(np.mean(after)), 'mean_delta': float(differences.mean()),
            'upper': float(np.quantile(means, gate['confidence'], method='linear')),
            'confidence': gate['confidence'], 'bootstrap_samples': len(means),
            'bootstrap_seed': gate['bootstrap_seed']}


def measurements(report, prepared, roles, records, expected_checkpoint):
    if report['prepared'] != data.identity(prepared) or report['checkpoint'] != expected_checkpoint:
        raise ValueError('Evaluation binds another job or checkpoint')
    result = {}
    for role in roles:
        rows = records[role]
        expected = [r['id'] for r in rows]
        outcome = report['outcomes'][role]
        losses = outcome['losses']
        if len(set(expected)) != len(expected) or [r['id'] for r in losses] != expected:
            raise ValueError('Incomplete, reordered or duplicated evaluation cases')
        values = [r['loss'] for r in losses]
        if not all(isinstance(v, (int, float)) and not isinstance(v, bool)
                   and math.isfinite(v) and v >= 0 for v in values):
            raise ValueError('Invalid response losses')
        answers = outcome['answers']
        correctness = []
        if role.startswith('test-'):
            count = prepared['plan']['generation_cases_per_cohort']
            if [a['id'] for a in answers] != expected[:count] or len(answers) != count:
                raise ValueError('Missing or reordered generated answers')
            for row, answer in zip(rows, answers):
                check = tasks.check_answer(row['task'], answer['text'])
                if check != answer['check']:
                    raise ValueError('Reported correctness differs from the generated answer')
                correctness.append({'id': row['id'], 'correct': check['correct']})
        result[role] = {'losses': values, 'answers': correctness}
    return result


def score(prepared, selection, baseline, candidate, checkpoint, records, *, phase,
          baseline_checkpoint=None):
    plan = prepared['plan']
    gate = plan['quality_gate']
    candidate_root = data.identity(checkpoint)
    baseline_root = data.identity(baseline_checkpoint) if baseline_checkpoint else 'seed'
    expected_step = gate['phase_one_checkpoint' if phase == 'a' else 'phase_two_checkpoint']
    if (selection['prepared'] != data.identity(prepared)
            or candidate_root not in selection['checkpoints'] or checkpoint['step'] != expected_step
            or (baseline_checkpoint and baseline_root not in selection['checkpoints'])):
        raise ValueError('Unselected endpoint')
    if phase == 'a' and baseline_checkpoint is not None:
        raise ValueError('Phase A must compare to the frozen seed')
    if phase in ('b', 'growth'):
        expected_baseline = gate['phase_one_checkpoint'] if phase == 'b' else expected_step
        if not baseline_checkpoint or baseline_checkpoint['step'] != expected_baseline:
            raise ValueError('Wrong reference endpoint')
    roles = ['test-a', 'retention'] if phase == 'a' else ['test-b', 'test-a', 'retention']
    before = measurements(baseline, prepared, roles, records, baseline_root)
    after = measurements(candidate, prepared, roles, records, candidate_root)
    outcomes = {}
    gain_role = 'test-a' if phase == 'a' else 'test-b'
    for role in roles:
        row = paired(before[role]['losses'], after[role]['losses'], gate)
        row['passed'] = (row['upper'] < gate['gain_upper_below_nats'] if role == gain_role
                         else row['upper'] <= gate['retention_upper_at_most_nats'])
        if role.startswith('test-'):
            generation = tasks.paired_accuracy(before[role]['answers'], after[role]['answers'])
            generation['passed'] = generation['candidate_correct'] >= generation['baseline_correct']
            row['generation'] = generation
            row['passed'] = row['passed'] and generation['passed']
        outcomes[role] = row
    return {'format': 'neuroshard-adaptive-quality-v1', 'prepared': data.identity(prepared),
            'selection': data.identity(selection), 'phase': phase, 'baseline': baseline_root,
            'candidate': candidate_root, 'outcomes': outcomes,
            'passed': all(row['passed'] for row in outcomes.values()),
            'scope': 'Frozen generated task families and public conversation retention; not general assistant quality'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prepared', type=Path, required=True)
    parser.add_argument('--selection', type=Path, required=True)
    parser.add_argument('--baseline', type=Path, required=True)
    parser.add_argument('--candidate', type=Path, required=True)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--baseline-checkpoint', type=Path)
    parser.add_argument('--phase', choices=['a', 'b', 'growth'], required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError('Preserve prior quality decisions')
    read = lambda path: json.loads(path.read_bytes())
    prepared = read(args.prepared)
    roles = ['test-a', 'retention'] if args.phase == 'a' else ['test-a', 'test-b', 'retention']
    records = {role: data.read_records(args.prepared.parent/prepared['roles'][role]['file'],
                                     prepared['roles'][role]['sha256']) for role in roles}
    result = score(prepared, read(args.selection), read(args.baseline), read(args.candidate),
                   read(args.checkpoint), records, phase=args.phase,
                   baseline_checkpoint=read(args.baseline_checkpoint) if args.baseline_checkpoint else None)
    data.save(args.output, result)
    print(json.dumps(result))


if __name__ == '__main__':
    main()
