"""Train fine intent selection after the existing nearest-example domain gate.

The parent gate stays exact. A bounded integer margin classifier distinguishes
new intents after that gate selects the new domain. Sixteen epochs is the
existing selector recipe, fixed here without a checkpoint/epoch search.
"""
import argparse
import base64
import json
from pathlib import Path

import numpy as np

from neuroshard.evolution import expert_router
from neuroshard.evolution.objects import Objects
from neuroshard.evolution.reference_data import identity, save, sha256


def fit(rows, encoder):
    labels = sorted({row['route'] for row in rows})
    prototype = {'format': expert_router.FORMAT, 'embedding_root': identity(encoder),
        'tokenizer_root': identity(encoder['files']), 'training_root': identity(sorted(rows, key=lambda row: row['id'])),
        'dimensions': 384, 'fallback': labels[0], 'minimum_margin': 0, 'maximum_distance': 2**40,
        'prototypes': {name: [next(row['features'] for row in rows if row['route'] == name)] for name in labels}}
    return expert_router.fit_classifier(rows, prototype, epochs=16, balance_classes=False)


def classify(model, queries):
    names = sorted(model['classifier']['weights'])
    weights = np.asarray([model['classifier']['weights'][name] for name in names], dtype=np.int64)
    biases = np.asarray([model['classifier']['biases'][name] for name in names], dtype=np.int64)
    return [names[index] for index in (np.asarray(queries, dtype=np.int64)@weights.T+16384*biases).argmax(axis=1)]


def main(campaign, output):
    inputs = json.loads((campaign/'compiled/inputs.json').read_bytes())
    policy = Objects(campaign/'compiled/objects').json(inputs['policies']['escrow'])['configuration']['semantic_questions']
    vectors = np.frombuffer(base64.b64decode(policy['vectors']), dtype='<i2').reshape(-1, 384)
    rows = [{'id': row['id'], 'route': row['label'], 'features': [int(value) for value in vector]}
            for row, vector in zip(policy['rows'], vectors) if row['label'] != 'parent']
    annotations = json.loads((campaign/'compiled/escrow/training-annotations.json').read_bytes())
    families = {name: sorted({row['core'] for row in annotations if row['topic'] == name})
                for name in policy['intents']}
    folds = {row['id']: families[row['topic']].index(row['core']) for row in annotations}
    validation = []
    for fold in range(4):
        model = fit([row for row in rows if folds[row['id']] != fold], policy['encoder'])
        heldout = [row for row in rows if folds[row['id']] == fold]
        predictions = classify(model, [row['features'] for row in heldout])
        validation.append({'fold': fold, 'correct': sum(prediction == row['route'] for prediction, row in zip(predictions, heldout)),
                           'count': len(heldout)})
    model = fit(rows, policy['encoder'])
    save(output/'classifier.json', model)
    value = json.loads((campaign/'jobs/61fe823acf6e0caf6e8362209b79733e54d19dbf11da5eff54f3ac6a6d474f29/quality-0.json').read_bytes())['result']
    facts = json.loads((campaign/'compiled/source-catalog.json').read_bytes())['cohorts']['escrow']
    selections = [row['after']['answering']['routing'][0]['semantic'] for row in value['executions'][:16]]
    predictions = classify(model, [row['encoding']['features'] for row in selections])
    predictions = [prediction if row['intent'] != 'parent' else 'parent' for prediction, row in zip(predictions, selections)]
    report = {'method': 'Preserved nearest-example parent gate, then integer fine-intent classification',
        'driver_sha256': sha256(__file__), 'training_root': model['training_root'], 'model_root': identity(model),
        'epochs': 16, 'validation': validation, 'answers_used': False, 'new_final': False,
        'development': {'correct': sum(prediction == fact['id'] for prediction, fact in zip(predictions, facts)),
                        'count': len(facts), 'predictions': predictions},
        'scope': 'Method development on opened escrow. Validation excludes all contracts of each held-out training paraphrase.'}
    save(output/'classifier-result.json', report)
    print(json.dumps(report))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--campaign', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    main(args.campaign, args.output)
