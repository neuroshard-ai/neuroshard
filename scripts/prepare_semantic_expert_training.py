#!/usr/bin/env python3
"""Retokenize training-derived question variations without opening evaluation."""
import argparse
from collections import Counter
import json
from pathlib import Path

from transformers import AutoTokenizer

from neuroshard.dataflow.store import canonical
from neuroshard.evolution import expert_curriculum, expert_data
from neuroshard.evolution.reference_data import identity, read_records, save, sha256, tokenizer_identity


def prepare(source, destination, variations, tokenizer_home, *, cross_contracts=False, excluded=None):
    source, destination = Path(source), Path(destination)
    original = json.loads((source/'prepared.json').read_bytes())
    plan = json.loads((source/'plan.json').read_bytes())
    graph = json.loads((source/'graph.json').read_bytes())
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_home, local_files_only=True)
    if tokenizer_identity(tokenizer) != graph['tokenizer']['root']:
        raise ValueError('Use the accepted tokenizer and complete chat template')
    spec = original['roles']['train']
    training = read_records(source/'train.jsonl', spec['sha256'])
    if len(training) != spec['count']:
        raise ValueError('The original training count changed')
    for row in training:
        expert_data.validate_record(row, plan['max_length'], len(tokenizer), tokenizer)
    annotations = [json.loads(line) for line in (source/'train-questions.jsonl').read_bytes().splitlines()]
    variations = json.loads(Path(variations).read_bytes())
    questions, provenance = expert_curriculum.augment(training, annotations, variations,
        inventory=spec['ids'], cohort='planner-semantic-training')
    if cross_contracts:
        if excluded is None:
            raise ValueError('Crossed training requires an explicit exclusion inventory')
        questions, crossing = expert_curriculum.cross_contracts(questions, excluded)
        provenance = {'augmentation': provenance, 'crossing': crossing}
    source_root = identity(provenance)
    rows = [expert_data.encode(tokenizer, row['messages'], plan['max_length'],
                              source=source_root, position=index) for index, row in enumerate(questions)]
    for row in rows:
        expert_data.validate_record(row, plan['max_length'], len(tokenizer), tokenizer)
    if len({row['id'] for row in rows}) != len(rows):
        raise ValueError('Normalized augmented training documents must be unique')
    batches = expert_curriculum.balanced_batches(questions)
    # Chat templates may append a masked newline after the assistant's EOS.
    eos_supervised = all([label for label in row['labels'] if label != -100][-1]
                         == tokenizer.eos_token_id for row in rows)
    if not eos_supervised:
        raise ValueError('Every atomic target must supervise its true EOS')
    destination.mkdir(parents=True, exist_ok=False)
    for name, values in (('train.jsonl', rows), ('train-questions.jsonl', questions)):
        (destination/name).write_bytes(b''.join(canonical(row)+b'\n' for row in values))
    save(destination/'provenance.json', provenance)
    report = {'format': expert_curriculum.FORMAT+'/prepared', 'source_root': source_root,
        'original_training': spec, 'tokenizer': tokenizer_identity(tokenizer),
        'train': {'sha256': sha256(destination/'train.jsonl'), 'count': len(rows),
                  'ids': identity([row['id'] for row in rows])},
        'batches': batches, 'questions_sha256': sha256(destination/'train-questions.jsonl'),
        'topics': dict(Counter(row['topics'][0] for row in questions)),
        'max_actual_length': max(len(row['input_ids']) for row in rows),
        'real_eos_supervised': eos_supervised,
        'evaluation_read': False, 'development_informed': True}
    save(destination/'augmentation.json', report)
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--destination', type=Path, required=True)
    parser.add_argument('--variations', type=Path, required=True)
    parser.add_argument('--tokenizer', type=Path, required=True)
    parser.add_argument('--cross-contracts', action='store_true')
    parser.add_argument('--exclude', type=Path, help='JSON list of evaluation question strings, without answers')
    args = parser.parse_args()
    report = prepare(args.source, args.destination, args.variations, args.tokenizer,
                     cross_contracts=args.cross_contracts,
                     excluded=json.loads(args.exclude.read_bytes()) if args.exclude else None)
    print(json.dumps({key: value for key, value in report.items() if key != 'batches'}))
