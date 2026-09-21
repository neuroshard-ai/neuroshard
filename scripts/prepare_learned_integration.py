"""Prepare unused-MBPP rows for learned integration.

Writes frozen split inputs. Does not train, score confirmation, or launch GPUs.
"""
import argparse
import json
from pathlib import Path

from neuroshard.evolution.learned_integration import (
    FORMAT, bind_spec, code_rows, load_mbpp, load_spec, method_freeze, training_schedule,
)
from neuroshard.evolution.reference_data import identity, save, sha256


def write_role(home, name, rows):
    path = home / (name + '.jsonl')
    path.write_text(''.join(json.dumps(r, sort_keys=True) + '\n' for r in rows))
    return {'file': path.name, 'sha256': sha256(path), 'ids': [r['id'] for r in rows],
            'task_ids': [r['task_id'] for r in rows]}


def prepare(spec, mbpp, home, gold_check):
    bind_spec(spec)
    raw = load_mbpp(mbpp, spec)
    home = Path(home)
    home.mkdir(parents=True, exist_ok=False)
    roles = {}
    for name in ('train_new', 'train_replay', 'retention', 'development', 'confirmation'):
        roles[name] = write_role(home, name, code_rows(spec['splits'][name], raw, gold_check))
    schedule = training_schedule(spec)
    result = {
        'format': FORMAT + '/prepared',
        'spec': identity(spec),
        'roles': roles,
        'confirmation_scored': False,
        'original_final_opened': False,
        'admission_evidence': False,
        **schedule,
    }
    save(home / 'prepared.json', result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--home', type=Path)
    parser.add_argument('--mbpp', type=Path)
    parser.add_argument('--plan', type=Path, default=Path('config/experiments/learned-integration.json'))
    parser.add_argument('--freeze', action='store_true',
                        help='Write the committed method freeze. Does not authorize a run.')
    args = parser.parse_args()
    spec = json.loads(args.plan.read_bytes())
    bind_spec(spec)
    if args.freeze:
        from neuroshard.evolution.learned_integration import METHOD_FORMAT
        freeze = method_freeze()
        if freeze['format'] != METHOD_FORMAT:
            raise ValueError('Method freeze format changed')
        path = Path('config/experiments/learned-integration-method.json')
        if path.exists() and json.loads(path.read_bytes()) != freeze:
            raise ValueError('Preserve the earlier method freeze; this candidate cannot be silently redefined')
        save(path, freeze)
        print('Commit ' + str(path) + ' before any later execution freeze. No GPU.')
        return
    if args.home is None or args.mbpp is None:
        raise SystemExit('Preparation requires --home and --mbpp')
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from programming_sandbox import check
    prepare(spec, args.mbpp, args.home, check)
    print(json.dumps({'prepared': str(Path(args.home) / 'prepared.json'), 'train': False}))


if __name__ == '__main__':
    main()
