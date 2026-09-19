#!/usr/bin/env python3
"""Reconstruct the frozen connection corpus from published learned sources."""
import argparse
import json
from pathlib import Path

from transformers import AutoTokenizer

from neuroshard.evolution.fusion_data import pools, prepare
from neuroshard.evolution.reference_data import identity, save, sha256


def run(inputs, seed, plan_path, output):
    plan = json.loads(plan_path.read_bytes())
    read = lambda name: [json.loads(line) for line in (inputs/name).read_text().splitlines()]
    for name, digest in plan['preparation']['sources'].items():
        if sha256(inputs/name) != digest:
            raise ValueError('A frozen source corpus changed')
    retained = {'train': read('retained-train.jsonl')}
    for role in ('dev', 'test'):
        retained[role] = read('retained-'+role+'-conversation.jsonl')+read('retained-'+role+'-skills.jsonl')
    pool, split = pools(read('directory.jsonl'), read('protocol.jsonl'), retained,
                        plan['preparation']['split_seed'])
    tokenizer = AutoTokenizer.from_pretrained(seed, local_files_only=True)
    rows, excluded = prepare(pool, plan['preparation']['counts'], tokenizer, plan['max_length'],
                             plan['preparation']['selection_seed'])
    output.mkdir(parents=True, exist_ok=False)
    for role, values in rows.items():
        # The canonical content root is invariant to JSON whitespace in replicas.
        if identity(values) != plan['data'][role]['root']:
            raise ValueError('Reconstructed corpus differs from the committed selection')
        (output/(role+'.jsonl')).write_text(''.join(json.dumps(row, sort_keys=True, separators=(',', ':'))+'\n'
                                                    for row in values))
    save(output/'split.json', split)
    save(output/'preparation.json', {'excluded': excluded, 'roles': {role: identity(values) for role, values in rows.items()}})


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--inputs', type=Path, required=True)
    parser.add_argument('--seed', type=Path, required=True)
    parser.add_argument('--plan', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    run(args.inputs, args.seed, args.plan, args.output)
