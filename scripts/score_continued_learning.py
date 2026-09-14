#!/usr/bin/env python3
"""Apply the frozen continued-learning quality gate to a locked candidate."""
import argparse
import json
from pathlib import Path

from neuroshard.evolution import continued, reference_data as data


def measurements(report, prepared, roles, records, expected_checkpoint, tokenizer=None):
    if report['prepared'] != data.identity(prepared) or report['checkpoint'] != expected_checkpoint:
        raise ValueError('Evaluation binds another job or checkpoint')
    result = {}
    for role in roles:
        rows = records[role]
        expected = [row['id'] for row in rows]
        outcome = report['outcomes'][role]
        losses = outcome['losses']
        if [row['id'] for row in losses] != expected:
            raise ValueError('Incomplete, reordered or duplicated evaluation cases')
        values = [row['loss'] for row in losses]
        answers = []
        if role in ('test-new', 'test-prior'):
            if [row['id'] for row in outcome['answers']] != expected:
                raise ValueError('Missing or reordered generated answers')
            from neuroshard.evolution import grounded_tasks as tasks
            for row, answer in zip(rows, outcome['answers']):
                if tokenizer is not None and tokenizer.decode(answer['output_ids'], skip_special_tokens=True) != answer['text']:
                    raise ValueError('Generated text differs from the recorded output tokens')
                check = tasks.check_answer(row['task'], answer['text'])
                if check != answer['check']:
                    raise ValueError('Reported correctness differs from the generated answer')
                answers.append({'id': row['id'], 'correct': check['correct']})
        result[role] = {'losses': values, 'answers': answers}
    return result


def score(plan, prepared, selection, baseline, candidate, records, checkpoint, tokenizer=None):
    continued.validate(plan)
    continued.validate_prepared(prepared, plan)
    continued.validate_selection(selection, plan, prepared, checkpoint)
    if baseline['checkpoint'] != selection['baseline'] or candidate['checkpoint'] != selection['candidate']:
        raise ValueError('Unselected endpoint')
    roles = plan['final_roles']
    before = measurements(baseline, prepared, roles, records, selection['baseline'], tokenizer)
    after = measurements(candidate, prepared, roles, records, selection['candidate'], tokenizer)
    result = continued.decide(plan, records, before, after)
    result.update(prepared=data.identity(prepared), selection=data.identity(selection),
                  baseline=selection['baseline'], candidate=selection['candidate'])
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', type=Path, required=True)
    parser.add_argument('--prepared', type=Path, required=True)
    parser.add_argument('--selection', type=Path, required=True)
    parser.add_argument('--baseline', type=Path, required=True)
    parser.add_argument('--candidate', type=Path, required=True)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--seed', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError('Preserve prior quality decisions')
    plan = continued.load(args.plan)
    prepared = json.loads(args.prepared.read_bytes())
    selection = json.loads(args.selection.read_bytes())
    if (plan['status'] != 'selection-committed' or not continued.committed_prepared(plan, prepared)
            or not continued.committed_selection(plan, selection)):
        raise ValueError('Scoring requires committed prepared artifacts and candidate lock')
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.seed, local_files_only=True, trust_remote_code=False)
    if data.tokenizer_identity(tokenizer) != plan['tokenizer']:
        raise ValueError('Scoring tokenizer differs from the frozen artifact')
    records = {role: continued.read_role(args.prepared.parent, prepared, role)
               for role in plan['final_roles']}
    result = score(plan, prepared, selection,
                   json.loads(args.baseline.read_bytes()), json.loads(args.candidate.read_bytes()),
                   records, json.loads(args.checkpoint.read_bytes()), tokenizer)
    data.save(args.output, result)
    print(json.dumps(result))


if __name__ == '__main__':
    main()
