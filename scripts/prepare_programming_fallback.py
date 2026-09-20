"""Prepare leftover MBPP rows for the equal-budget fallback comparison.

No training. Does not read or open the original programming-expert final answers.
"""
import argparse
import json
from pathlib import Path

from neuroshard.evolution import programming_expert as parent
from neuroshard.evolution import programming_fallback as experiment
from neuroshard.evolution.reference_data import identity, save, sha256


def prepare(plan, mbpp, home, gold_check):
    if (plan['format'] != experiment.FORMAT or plan['parent_plan'] != experiment.PARENT_PLAN
            or plan.get('parent_selection') != experiment.PARENT_SELECTION
            or plan.get('parent_expert') != experiment.PARENT_EXPERT
            or plan.get('tokenizer') != experiment.TOKENIZER):
        raise ValueError('Fallback plan does not bind the rejected parent trial')
    if sha256(mbpp) != parent.MBPP_SHA:
        raise ValueError('Source differs from the pinned benchmark')
    comparison = plan['comparison']
    raw = {}
    for line in Path(mbpp).read_text().splitlines():
        row = json.loads(line)
        raw[row['task_id']] = row
    home = Path(home)
    home.mkdir(parents=True, exist_ok=False)
    rows = []
    for number in comparison['task_ids']:
        row = raw[number]
        messages = parent.code_prompt(row)
        verdict = gold_check(row['code'], row['test_setup_code'], row['test_list'])
        if not verdict['passed']:
            raise ValueError('Frozen leftover gold program failed: %s' % number)
        rows.append({'id': identity({'dataset': parent.MBPP_SHA, 'task': number}),
                     'task_id': number, 'kind': 'code', 'messages': messages,
                     'reference': row['code'], 'setup': row['test_setup_code'],
                     'tests': row['test_list']})
    experiment.load_comparison_rows(plan, rows)
    path = home / 'comparison.jsonl'
    path.write_text(''.join(json.dumps(r, sort_keys=True) + '\n' for r in rows))
    result = {'format': experiment.FORMAT + '/prepared', 'plan': identity(plan),
              'file': path.name, 'sha256': sha256(path), 'ids': [r['id'] for r in rows],
              'task_ids': [r['task_id'] for r in rows], 'original_final_opened': False,
              'parent_expert': plan['parent_expert'], 'tokenizer': plan['tokenizer']}
    save(home / 'prepared.json', result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--home', type=Path, required=False)
    parser.add_argument('--mbpp', type=Path, required=False)
    parser.add_argument('--plan', type=Path, default=Path('config/experiments/programming-fallback.json'))
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
        path = Path('config/experiments/programming-fallback-freeze.json')
        if path.exists() and json.loads(path.read_bytes()) != freeze:
            raise ValueError('Preserve the earlier freeze; this candidate cannot be silently redefined')
        save(path, freeze)
        print('Commit ' + str(path) + ' before generation.')
        return
    if args.home is None or args.mbpp is None:
        raise SystemExit('Preparation requires --home and --mbpp')
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from programming_sandbox import check
    plan = json.loads(args.plan.read_bytes())
    prepare(plan, args.mbpp, args.home, check)
    print(json.dumps({'prepared': str(args.home / 'prepared.json')}))


if __name__ == '__main__':
    main()
