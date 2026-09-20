"""Prepare disjoint leftover splits for the second programming tail.

Does not read or open the original programming-expert final answers.
"""
import argparse
import json
from pathlib import Path

from neuroshard.evolution import programming_expert as parent
from neuroshard.evolution import programming_growth as experiment
from neuroshard.evolution.reference_data import identity, save, sha256, conversation, tokenizer_identity


def code_rows(task_ids, raw, gold_check, tokenizer=None, max_length=None):
    rows = []
    for number in task_ids:
        row = raw[number]
        messages = parent.code_prompt(row)
        verdict = gold_check(row['code'], row['test_setup_code'], row['test_list'])
        if not verdict['passed']:
            raise ValueError('Frozen leftover gold program failed: %s' % number)
        item = {'id': identity({'dataset': parent.MBPP_SHA, 'task': number}),
                'task_id': number, 'kind': 'code', 'messages': messages,
                'reference': row['code'], 'setup': row['test_setup_code'],
                'tests': row['test_list']}
        if tokenizer is not None:
            item.update(conversation(
                tokenizer, messages + [{'role': 'assistant', 'content': row['code']}], max_length))
        rows.append(item)
    return rows


def write_role(home, name, rows):
    path = home / (name + '.jsonl')
    path.write_text(''.join(json.dumps(r, sort_keys=True) + '\n' for r in rows))
    return {'file': path.name, 'sha256': sha256(path), 'ids': [r['id'] for r in rows],
            'task_ids': [r['task_id'] for r in rows]}


def prepare(plan, mbpp, home, gold_check, tokenizer):
    if (plan['format'] != experiment.FORMAT or plan['parent_plan'] != experiment.PARENT_PLAN
            or plan.get('parent_selection') != experiment.PARENT_SELECTION
            or plan.get('incumbent_expert') != experiment.INCUMBENT_EXPERT
            or plan.get('tokenizer') != experiment.TOKENIZER
            or plan.get('parent_fallback_plan') != experiment.PARENT_FALLBACK_PLAN):
        raise ValueError('Growth plan does not bind the leftover fallback baseline')
    if sha256(mbpp) != parent.MBPP_SHA:
        raise ValueError('Source differs from the pinned benchmark')
    if tokenizer_identity(tokenizer) != plan['tokenizer']:
        raise ValueError('Tokenizer differs from the leftover fallback baseline')
    experiment.bind_splits(plan)
    raw = {}
    for line in Path(mbpp).read_text().splitlines():
        row = json.loads(line)
        raw[row['task_id']] = row
    splits = plan['splits']
    heldout = (splits['preservation_task_ids'] + splits['new_task_ids']
               + splits['development_task_ids'])
    remainder = experiment.leftover_ranking(plan)[sum(plan['counts'][k] for k in
                                                      ('preservation', 'new', 'development')):]
    texts = {n: raw[n]['text'] for n in remainder + heldout}
    if experiment.near_duplicate_train_exclusions(texts, remainder, heldout) != splits[
            'excluded_near_duplicate_train_task_ids']:
        raise ValueError('Near-duplicate training exclusions changed')
    home = Path(home)
    home.mkdir(parents=True, exist_ok=False)
    train = code_rows(splits['train_task_ids'], raw, gold_check, tokenizer, plan['max_length'])
    development = code_rows(splits['development_task_ids'], raw, gold_check)
    new = code_rows(splits['new_task_ids'], raw, gold_check)
    preservation = code_rows(splits['preservation_task_ids'], raw, gold_check)
    experiment.load_role_rows(plan, train, splits['train_task_ids'], role='train')
    experiment.load_role_rows(plan, development, splits['development_task_ids'], role='development')
    experiment.load_role_rows(plan, new, splits['new_task_ids'], role='new')
    experiment.load_role_rows(plan, preservation, splits['preservation_task_ids'], role='preservation')
    roles = {
        'train': write_role(home, 'train', train),
        'development': write_role(home, 'development', development),
        'new': write_role(home, 'new', new),
        'preservation': write_role(home, 'preservation', preservation),
    }
    import random
    rng = random.Random(plan['seed'])
    indices = list(range(len(train)))
    rng.shuffle(indices)
    batches = [indices[i:i + plan['batch_size']] for i in range(0, len(indices), plan['batch_size'])]
    schedule = []
    while len(schedule) < plan['training']['steps']:
        epoch = list(range(len(batches)))
        rng.shuffle(epoch)
        schedule.extend(epoch)
    result = {
        'format': experiment.FORMAT + '/prepared',
        'plan': identity(plan),
        'roles': roles,
        'tokenizer': plan['tokenizer'],
        'incumbent_expert': plan['incumbent_expert'],
        'original_final_opened': False,
        'batches': batches,
        'schedule': schedule[:plan['training']['steps']],
        'merge': experiment.merge_scales(plan),
    }
    save(home / 'prepared.json', result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--home', type=Path, required=False)
    parser.add_argument('--mbpp', type=Path, required=False)
    parser.add_argument('--seed', type=Path, required=False)
    parser.add_argument('--plan', type=Path, default=Path('config/experiments/programming-growth.json'))
    parser.add_argument('--freeze', action='store_true',
                        help='Bind committed execution source after the comparison is committed')
    args = parser.parse_args()
    if args.freeze:
        import subprocess
        names = list(experiment.EXECUTION_SOURCES)
        subprocess.run(['git', 'diff', '--exit-code', 'HEAD', '--', *names], check=True)
        plan = json.loads(args.plan.read_bytes())
        selection = json.loads(Path('config/experiments/programming-expert-selection.json').read_bytes())
        source_commit = subprocess.check_output(['git', 'rev-parse', 'HEAD']).decode().strip()
        freeze = experiment.execution_freeze(plan, selection, source_commit)
        path = Path('config/experiments/programming-growth-freeze.json')
        if path.exists() and json.loads(path.read_bytes()) != freeze:
            raise ValueError('Preserve the earlier freeze; this candidate cannot be silently redefined')
        save(path, freeze)
        print('Commit ' + str(path) + ' before training.')
        return
    if args.home is None or args.mbpp is None or args.seed is None:
        raise SystemExit('Preparation requires --home, --mbpp and --seed')
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from programming_sandbox import check
    from transformers import AutoTokenizer
    plan = json.loads(args.plan.read_bytes())
    tokenizer = AutoTokenizer.from_pretrained(args.seed, local_files_only=True)
    prepare(plan, args.mbpp, args.home, check, tokenizer)
    print(json.dumps({'prepared': str(Path(args.home) / 'prepared.json')}))


if __name__ == '__main__':
    main()
