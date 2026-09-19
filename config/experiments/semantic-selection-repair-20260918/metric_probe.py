"""Fit access geometry from training paraphrases, with grouped validation.

All six rendering contracts of one core question leave the index together.
Earlier examples use four ID-based folds. Regularization is selected by the
worst of new-intent and earlier-route accuracy, then their mean, then stronger
regularization. The opened development output is read only after selection.
"""
import argparse
import base64
import json
from pathlib import Path

import numpy as np

from neuroshard.evolution import expert_router
from neuroshard.evolution.objects import Objects
from neuroshard.evolution.reference_data import identity, save, sha256


REGULARIZATION = (1000000, 100000, 10000, 1000, 100)


def fit(vectors, labels, regularization):
    values = vectors.astype(np.float64)/16384
    residuals = np.concatenate([values[labels == name]-values[labels == name].mean(axis=0)
                               for name in sorted(set(labels)-{'parent'})])
    covariance = residuals.T@residuals/len(residuals)
    eigenvalues, directions = np.linalg.eigh(covariance+regularization/1000000*np.eye(384))
    projection = (directions/np.sqrt(eigenvalues))@directions.T
    return np.rint(projection/np.abs(projection).max()*16384).astype(np.int64)


def project(vectors, matrix):
    return np.asarray([expert_router.normalize([int(v) for v in row])
                       for row in vectors@matrix], dtype=np.int64)


def classify(index, labels, queries):
    distances = (queries*queries).sum(axis=1)[:, None]+(index*index).sum(axis=1)[None, :]-2*queries@index.T
    return labels[distances.argmin(axis=1)]


def main(campaign, output):
    inputs = json.loads((campaign/'compiled/inputs.json').read_bytes())
    policy = Objects(campaign/'compiled/objects').json(inputs['policies']['escrow'])['configuration']['semantic_questions']
    vectors = np.frombuffer(base64.b64decode(policy['vectors']), dtype='<i2').reshape(-1, 384).astype(np.int64)
    labels = np.asarray([row['label'] for row in policy['rows']])
    annotations = json.loads((campaign/'compiled/escrow/training-annotations.json').read_bytes())
    families = {name: sorted({row['core'] for row in annotations if row['topic'] == name})
                for name in set(labels)-{'parent'}}
    heldout = {row['id']: families[row['topic']].index(row['core']) for row in annotations}
    folds = np.asarray([heldout.get(row['id'], int(row['id'], 16) % 4) for row in policy['rows']])
    validation = []
    for regularization in REGULARIZATION:
        correct = {'new': 0, 'earlier': 0}
        count = {'new': 0, 'earlier': 0}
        for fold in range(4):
            included = folds != fold
            matrix = fit(vectors[included], labels[included], regularization)
            predictions = classify(project(vectors[included], matrix), labels[included], project(vectors[~included], matrix))
            targets = labels[~included]
            for name, mask in [('new', targets != 'parent'), ('earlier', targets == 'parent')]:
                correct[name] += int((predictions[mask] == targets[mask]).sum())
                count[name] += int(mask.sum())
        accuracy = {name: correct[name]/count[name] for name in count}
        validation.append({'regularization_ppm': regularization, 'correct': correct, 'count': count, 'accuracy': accuracy})
    chosen = max(validation, key=lambda row: (min(row['accuracy'].values()),
                 sum(row['accuracy'].values()), row['regularization_ppm']))
    matrix = fit(vectors, labels, chosen['regularization_ppm'])
    output.mkdir(exist_ok=True)
    save(output/'metric.json', {'method': 'within-intent-whitening-q14-v1',
         'training_root': policy['training_root'], 'regularization_ppm': chosen['regularization_ppm'],
         'matrix': base64.b64encode(matrix.astype('<i2').tobytes()).decode()})
    # Development is deliberately downstream of all fitting and selection.
    result = json.loads((campaign/'jobs/61fe823acf6e0caf6e8362209b79733e54d19dbf11da5eff54f3ac6a6d474f29/quality-0.json').read_bytes())['result']
    facts = json.loads((campaign/'compiled/source-catalog.json').read_bytes())['cohorts']['escrow']
    queries = [{'features': e['after']['answering']['routing'][0]['semantic']['encoding']['features'],
                'expected': facts[index]['id']} for index, e in enumerate(result['executions'][:16])]
    earlier = []
    for values in result['retention']['roles'].values():
        for row in values:
            for route in row['after']['answering']['routing']:
                if route.get('semantic', {}).get('intent') == 'parent':
                    earlier.append({'features': route['semantic']['encoding']['features'], 'expected': 'parent'})
    predictions = classify(project(vectors, matrix), labels,
                           project(np.asarray([row['features'] for row in queries+earlier], dtype=np.int64), matrix))
    report = {'training_root': policy['training_root'], 'driver_sha256': sha256(__file__),
        'validation': validation, 'selected_regularization_ppm': chosen['regularization_ppm'],
        'selected_before_development': True, 'answers_used': False, 'new_final': False,
        'metric_root': identity(json.loads((output/'metric.json').read_bytes())),
        'development': {'new_correct': sum(predictions[i] == row['expected'] for i, row in enumerate(queries)),
            'new_count': len(queries), 'earlier_preserved': sum(predictions[len(queries)+i] == 'parent' for i in range(len(earlier))),
            'earlier_count': len(earlier), 'predictions': list(predictions[:len(queries)])}}
    # NumPy's comparison scalar is not JSON serializable.
    for key in ('new_correct', 'earlier_preserved'):
        report['development'][key] = int(report['development'][key])
    save(output/'metric-result.json', report)
    print(json.dumps(report))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--campaign', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    main(args.campaign, args.output)
