"""Score documents units on the realistic request set (``real_documents``) with the route's own scorer.

Usage (on a machine with the pinned parent): python -m neuroshard.evolution.real_documents_eval SET OUT
  [--unit NAME=DIR ...] [--parent] [--limit N]

Each model answers every case through the documents workspace (search, read, compute, cited answer or
not-found). Rows hold each case's outcome and final text, so any number in the report can be recomputed.
"""

import argparse
import collections
import json
from pathlib import Path
import time


def run(model, tokenizer, policy, cases, progress=print):
    from neuroshard.evolution import assistant_general as general
    from neuroshard.evolution.assistant_serving import cached_responder

    rows = []
    for index, case in enumerate(cases):
        started = time.monotonic()
        row = general.documents_episode(case, cached_responder(model, tokenizer, policy), policy)
        rows.append({'id': case['id'], 'family': case['family'], 'passed': bool(row['passed']),
                     'final_text': (row.get('rounds') or [{}])[-1].get('final_text', '')[:600],
                     'seconds': round(time.monotonic() - started, 1)})
        if index % 25 == 0:
            progress(f'{index}/{len(cases)} passed {sum(r["passed"] for r in rows)}')
    return rows


def summary(rows):
    by = collections.defaultdict(lambda: [0, 0])
    for row in rows:
        by[row['family']][0] += row['passed']
        by[row['family']][1] += 1
    return {'passed': sum(r['passed'] for r in rows), 'cases': len(rows),
            'families': {k: f'{h}/{t}' for k, (h, t) in sorted(by.items())},
            'mean_seconds': round(sum(r['seconds'] for r in rows) / max(1, len(rows)), 1)}


def main(argv=None):
    from neuroshard.evolution import granite_shard_execution as runtime

    runtime.configure()
    parser = argparse.ArgumentParser()
    parser.add_argument('set', type=Path)
    parser.add_argument('out', type=Path)
    parser.add_argument('--unit', action='append', default=[], help='NAME=DIR of a saved addition unit')
    parser.add_argument('--parent', action='store_true', help='also score the parent without any unit')
    parser.add_argument('--policy', type=Path, default=Path('config/experiments/assistant-general-policy-documents5.json'))
    parser.add_argument('--limit', type=int)
    parser.add_argument('--part', default='0/1', help='I/N: score every Nth case starting at I (split across machines)')
    args = parser.parse_args(argv)

    from transformers import AutoTokenizer

    from neuroshard.evolution import assistant_experience_train as trainer
    from neuroshard.evolution import growth_runtime
    from neuroshard.evolution.growth_model import model_factory
    from neuroshard.evolution.modular_reference_execution import ROOT

    index, parts = (int(x) for x in args.part.split('/'))
    cases = json.loads(args.set.read_text())['cases'][:args.limit][index::parts]
    policy = json.loads(args.policy.read_text())
    make = model_factory({'base': 'granite-parent'})
    tokenizer = AutoTokenizer.from_pretrained(ROOT / '.models' / 'baseline', local_files_only=True)
    spec = growth_runtime.growth_spec()
    report = json.loads(args.out.read_text()) if args.out.exists() else {'models': {}}
    models = ([('parent', None)] if args.parent else []) + [tuple(u.split('=', 1)) for u in args.unit]
    for name, directory in models:
        if name in report['models']:
            continue
        model = make()
        if directory:
            trainer.load_trainable(model, 'addition', spec, Path(directory))
        rows = run(model.eval(), tokenizer, policy, cases, lambda m, n=name: print(n, m, flush=True))
        report['models'][name] = {'summary': summary(rows), 'rows': rows}
        args.out.write_text(json.dumps(report, indent=1))
        print(name, json.dumps(report['models'][name]['summary']), flush=True)
        del model


if __name__ == '__main__':
    main()
