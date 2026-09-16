#!/usr/bin/env python3
"""Prepare question-only supervision from the frozen, group-disjoint corpus."""
import argparse
import json
from pathlib import Path

from transformers import AutoTokenizer

from neuroshard.evolution.planner_data import prepare
from neuroshard.evolution.reference_data import identity, save, sha256


def run(inputs, seed, prescription, output):
    plan = json.loads(prescription.read_bytes())
    tokenizer = AutoTokenizer.from_pretrained(seed, local_files_only=True)
    output.mkdir(parents=True, exist_ok=False)
    groups, recorded = {}, {}
    for role in ('train', 'dev', 'test'):
        path = inputs/(role+'.jsonl')
        if sha256(path) != plan['sources'][role]:
            raise ValueError('The grouped source corpus differs from the preparation prescription')
        source = [json.loads(line) for line in path.read_text().splitlines()]
        rows = prepare(source, tokenizer, max_length=plan['max_length'],
            coreference_count=plan['coreference'][role])
        groups[role] = {group for row in rows for group in row['groups']}
        path = output/(role+'.jsonl')
        path.write_text(''.join(json.dumps(row, sort_keys=True, separators=(',', ':'))+'\n' for row in rows))
        recorded[role] = {'root': identity(rows), 'sha256': sha256(path), 'count': len(rows),
            'ids': identity([row['id'] for row in rows]), 'groups': identity(sorted(groups[role])),
            'maximum_length': max(len(row['input_ids']) for row in rows)}
    if any(groups[a]&groups[b] for a, b in [('train', 'dev'), ('train', 'test'), ('dev', 'test')]):
        raise ValueError('A source group crosses planner fitting and evaluation')
    save(output/'preparation.json', {'prescription': identity(plan), 'roles': recorded})


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('inputs', 'seed', 'prescription', 'output'):
        parser.add_argument('--'+name, type=Path, required=True)
    args = parser.parse_args()
    run(args.inputs, args.seed, args.prescription, args.output)
