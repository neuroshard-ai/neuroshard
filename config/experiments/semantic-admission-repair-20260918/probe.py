"""Fit expert admission from training-only semantic vectors, preserving the base.

The recipe is fixed at sixteen integer-margin epochs. Four folds exclude every
contract of a held-out new-fact paraphrase and group older negatives by source
document. Opened bootstrap responses are development diagnostics, never fitting
inputs. No fresh cohort final is read.
"""
import argparse
import json
from pathlib import Path

import numpy as np

from neuroshard.evolution import expert_router
from neuroshard.evolution.reference_data import identity, save, sha256


def fit(rows, encoder):
    ordered = sorted(rows, key=lambda row: row['id'])
    labels = sorted({row['route'] for row in ordered})
    prototype = {
        'format': expert_router.FORMAT,
        'embedding_root': identity(encoder),
        'tokenizer_root': identity(encoder['files']),
        'training_root': identity(ordered),
        'dimensions': 384,
        'fallback': 'parent',
        'minimum_margin': 0,
        'maximum_distance': 2**40,
        'prototypes': {name: [next(row['features'] for row in ordered
                                 if row['route'] == name)] for name in labels},
    }
    return expert_router.fit_classifier(ordered, prototype, epochs=16, balance_classes=False)


def classify(model, vectors):
    names = sorted(model['classifier']['weights'])
    weights = np.asarray([model['classifier']['weights'][name] for name in names], dtype=np.int64)
    biases = np.asarray([model['classifier']['biases'][name] for name in names], dtype=np.int64)
    scores = np.asarray(vectors, dtype=np.int64) @ weights.T + expert_router.SCALE * biases
    # Ties retain the base; sorting alone would prefer a new route over parent.
    return [names[int(row.argmax())] if np.count_nonzero(row == row.max()) == 1
            else 'parent' for row in scores]


def main(campaign, output):
    output.mkdir(parents=True, exist_ok=True)
    read = lambda name: json.loads((campaign / name).read_bytes())
    fitting = read('compiled/selector-fitting.json')['rows']
    features = {row['id']: row['semantic'] for row in read('features.json')}
    encoder = read('encoder.json')
    order = read('input-plan.json')['order']
    folds = {row['id']: int(row.get('document', row['id'])[:8], 16) % 4 for row in fitting}
    for cohort in order:
        annotations = read('compiled/' + cohort + '/training-annotations.json')
        families = {topic: sorted({row['core'] for row in annotations if row['topic'] == topic})
                    for topic in {row['topic'] for row in annotations}}
        folds.update({row['id']: families[row['topic']].index(row['core']) for row in annotations})
    results = {}
    for stage, cohort in enumerate(order):
        admitted = set(order[:stage + 1])
        rows = [{'id': row['id'], 'route': row['route'] if row['route'] in admitted else 'parent',
                 'features': features[row['id']]}
                for row in fitting if row['route'] in admitted | {'parent', 'directory', 'protocol', 'planner'}]
        validation = []
        for fold in range(4):
            trained = [row for row in rows if folds[row['id']] != fold]
            heldout = [row for row in rows if folds[row['id']] == fold]
            model = fit(trained, encoder)
            predicted = classify(model, [row['features'] for row in heldout])
            confusion = {}
            for row, choice in zip(heldout, predicted):
                key = row['route'] + '->' + choice
                confusion[key] = confusion.get(key, 0) + 1
            validation.append({'fold': fold, 'count': len(heldout), 'confusion': confusion})
        model = fit(rows, encoder)
        save(output / (cohort + '-classifier.json'), model)
        results[cohort] = {'model': identity(model), 'training_root': model['training_root'],
                           'training_count': len(rows), 'validation': validation}
        save(output / 'training-result.json', {'driver': sha256(__file__), 'epochs': 16,
             'answers_used': False, 'fresh_finals_read': False, 'stages': results})
        print(json.dumps({'stage': cohort, **results[cohort]}), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--campaign', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    main(args.campaign, args.output)
