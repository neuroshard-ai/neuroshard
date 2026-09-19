"""Measure joint intent/background admission with training-only grouped folds.

A whole expert may cover unrelated facts. Separate intent cells can represent
that union without forcing one linear expert/background boundary. This keeps
the same 16-epoch integer objective and the same folds as the domain probe.
"""
import argparse
import ast
from contextlib import redirect_stdout
import io
import json
import math
from pathlib import Path

import numpy as np

from neuroshard.evolution import expert_router as er
from neuroshard.evolution.reference_data import identity, save, sha256
from probe import classify

ROOT = Path(__file__).resolve().parents[3]
REFERENCE = ROOT / 'config/experiments/question-match-probe-20260918/retrieval-pilot.py'
node = next(node for node in ast.parse(REFERENCE.read_text()).body
            if isinstance(node, ast.FunctionDef) and node.name == 'fit_fast')
namespace = {'np': np, 'er': er, 'identity': identity, 'math': math}
exec(compile(ast.Module(body=[node], type_ignores=[]), str(REFERENCE), 'exec'), namespace)


def fit(rows, encoder):
    rows = sorted(rows, key=lambda row: row['id'])
    names = sorted({row['route'] for row in rows})
    prototype = {'format': er.FORMAT, 'embedding_root': identity(encoder),
        'tokenizer_root': identity(encoder['files']), 'training_root': identity(rows),
        'dimensions': 384, 'fallback': 'parent', 'minimum_margin': 0,
        'maximum_distance': 2**40, 'prototypes': {name: [next(row['features']
            for row in rows if row['route'] == name)] for name in names}}
    with redirect_stdout(io.StringIO()):
        return namespace['fit_fast'](rows, prototype, epochs=16, balance=False)


def main(campaign, output):
    output.mkdir(parents=True, exist_ok=True)
    read = lambda name: json.loads((campaign / name).read_bytes())
    fitting = read('compiled/selector-fitting.json')['rows']
    features = {row['id']: row['semantic'] for row in read('features.json')}
    encoder = read('encoder.json')
    order = read('input-plan.json')['order']
    folds = {row['id']: int(row.get('document', row['id'])[:8], 16) % 4 for row in fitting}
    domains = {'parent': 'parent'}
    for cohort in order:
        annotations = read('compiled/' + cohort + '/training-annotations.json')
        families = {topic: sorted({row['core'] for row in annotations if row['topic'] == topic})
                    for topic in {row['topic'] for row in annotations}}
        folds.update({row['id']: families[row['topic']].index(row['core']) for row in annotations})
        domains.update({row['topic']: cohort for row in annotations})
    results = {}
    for stage, cohort in enumerate(order):
        included = {'parent', 'directory', 'protocol', 'planner', *order[:stage + 1]}
        rows = [{'id': row['id'], 'route': row['intent'], 'features': features[row['id']]}
                for row in fitting if row['route'] in included]
        validation = []
        for fold in range(4):
            model = fit([row for row in rows if folds[row['id']] != fold], encoder)
            heldout = [row for row in rows if folds[row['id']] == fold]
            predicted = classify(model, [row['features'] for row in heldout])
            confusion = {}
            for row, choice in zip(heldout, predicted):
                key = domains[row['route']] + '->' + domains[choice]
                confusion[key] = confusion.get(key, 0) + 1
            validation.append({'fold': fold, 'count': len(heldout), 'domain_confusion': confusion,
                'new_intent_correct': sum(a == b['route'] for a, b in zip(predicted, heldout)
                                         if b['route'] != 'parent'),
                'new_intent_count': sum(row['route'] != 'parent' for row in heldout)})
        model = fit(rows, encoder)
        save(output / (cohort + '-joint.json'), model)
        results[cohort] = {'model': identity(model), 'training_root': model['training_root'],
                           'training_count': len(rows), 'validation': validation}
        save(output / 'joint-result.json', {'driver': sha256(__file__), 'fitter': sha256(REFERENCE),
            'epochs': 16, 'answers_used': False, 'fresh_finals_read': False, 'stages': results})
        print(json.dumps({'stage': cohort, **results[cohort]}), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--campaign', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    main(args.campaign, args.output)
