"""Generate and judge the general route's phrasing sets on Amazon Bedrock, recording every response.

    PYTHONPATH=src python scripts/generate_general_phrasings.py --out config/experiments/assistant-general-phrasings.json

For each route and kind: three generation calls for training (casual style) and two for the sealed
confirmation4 (terse/formal style), each template judged once (twice with an independent solve for
arithmetic). Then confirmation4 drops any template sharing a normalised six-word span with training
or with any authored phrasing used in an earlier split. Training keeps a fixed 15% per kind as
development, which no training case uses.
"""

import argparse
import gzip
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import random
import sys
import threading

from neuroshard.evolution import assistant_phrasings as phrasings

PLAN = {'train': {'calls': 3, 'count': 30}, 'confirmation4': {'calls': 2, 'count': 20}}
SEED = 20261012


def authored_spans():
    """Every phrasing an earlier split used, with fixed placeholder values, as six-word spans."""
    from neuroshard.evolution import assistant_chat_data as chat
    from neuroshard.evolution import assistant_documents_data as docs

    texts = set()
    for split in ('train', 'development', 'confirmation', 'confirmation2', 'confirmation3'):
        texts |= {t['user'] for c in docs.cases(split) for t in c['turns']}
        texts |= {t['user'] for c in chat.cases(split) for t in c['turns']}
    return set().union(*(phrasings.spans(t) for t in texts))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--out', required=True)
    parser.add_argument('--workers', type=int, default=8)
    parser.add_argument('--cache', default='/tmp/general-phrasings-cache.jsonl',
                        help='append-only record of completed calls, so an interrupted run resumes')
    args = parser.parse_args(argv)
    bedrock = phrasings.client()
    cache_path = Path(args.cache)
    cache = {}
    if cache_path.exists():
        for line in cache_path.read_text().splitlines():
            row = json.loads(line)
            cache[row['key']] = row['value']
    lock = threading.Lock()

    def cached(key, compute):
        if key in cache:
            return cache[key]
        value = compute()
        with lock:
            cache[key] = value
            with cache_path.open('a') as handle:
                handle.write(json.dumps({'key': key, 'value': value}) + '\n')
        return value

    jobs = [(route, split, kind, call) for route in ('documents', 'chat') for split, plan in PLAN.items()
            for kind in phrasings.table(route) for call in range(plan['calls'])]

    def run_generation(job):
        route, split, kind, call = job
        rows, raw = cached('gen:' + json.dumps(job), lambda: phrasings.generate(
            kind, PLAN[split]['count'], phrasings.STYLES[split], bedrock, route))
        print('generated', *job, len(rows), flush=True)
        return job, rows, raw

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        generated = list(pool.map(run_generation, jobs))
    record, candidates = {'generation': [], 'judging': []}, {}
    for (route, split, kind, call), rows, raw in generated:
        record['generation'].append({'route': route, 'split': split, 'kind': kind, 'call': call, **raw})
        bucket = candidates.setdefault((route, split, kind), [])
        bucket += [r for r in rows if r.lower() not in {k.lower() for k in bucket}]

    def run_judging(item):
        (route, split, kind), template = item
        ok, rows = cached('judge:' + json.dumps([route, split, kind, template]),
                          lambda: phrasings.judge(route, kind, template, bedrock, SEED))
        return item, ok, rows

    items = [(key, t) for key, rows in candidates.items() for t in rows]
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        judged = list(pool.map(run_judging, items))
    kept = {}
    for ((route, split, kind), template), ok, rows in judged:
        record['judging'].append({'route': route, 'split': split, 'kind': kind, 'template': template, 'kept': ok,
                                  'trials': rows})
        if ok:
            kept.setdefault(route, {}).setdefault(split, {}).setdefault(kind, []).append(template)
    earlier = authored_spans()
    sets, filtered = {}, {}
    for route, splits in kept.items():
        train = {k: sorted(v) for k, v in splits['train'].items()}
        train_spans = set().union(*(phrasings.spans(t) for rows in train.values() for t in rows))
        train_texts = {t for rows in train.values() for t in rows}
        test = {}
        for kind, rows in splits['confirmation4'].items():
            test[kind] = sorted(t for t in rows if t not in train_texts and not phrasings.spans(t) & (train_spans | earlier))
            filtered[f'{route}/{kind}'] = len(rows) - len(test[kind])
        ok, shared = phrasings.disjoint(train, test)
        if not ok:
            raise SystemExit(f'{route}: train and confirmation4 share {shared}')
        split_rows = {k: phrasings.holdout(v, seed=SEED) for k, v in train.items()}
        sets[route] = {'train': {k: v[0] for k, v in split_rows.items()},
                       'development': {k: v[1] for k, v in split_rows.items()}, 'confirmation4': test}
    raw = json.dumps(record, sort_keys=True)
    usage = {key: sum(r['usage'].get(key, 0) for r in record['generation'])
             + sum(t['usage'].get(key, 0) + t.get('usage_solve', {}).get(key, 0)
                   for j in record['judging'] for t in j['trials']) for key in ('inputTokens', 'outputTokens')}
    counts = {route: {split: {k: len(v) for k, v in rows.items()} for split, rows in by.items()}
              for route, by in sets.items()}
    judged_counts = {f'{r}/{s}': {'candidates': sum(1 for j in record['judging'] if j['route'] == r and j['split'] == s),
                                  'kept': sum(1 for j in record['judging'] if j['route'] == r and j['split'] == s
                                              and j['kept'])} for r in ('documents', 'chat') for s in PLAN}
    out = {'format': 'neuroshard-assistant-general-phrasings/1', 'model': phrasings.MODEL, 'region': phrasings.REGION,
           'plan': PLAN, 'seed': SEED, 'styles': phrasings.STYLES, 'sets': sets, 'counts': counts,
           'judged': judged_counts, 'confirmation4_dropped_for_overlap': filtered, 'usage': usage,
           'responses_sha256': hashlib.sha256(raw.encode()).hexdigest(),
           'rule': 'Claude wrote user-side phrasings only; facts, constraints, expected values and every assistant '
                   'message we learn from are ours.'}
    Path(args.out).write_text(json.dumps(out, indent=1, sort_keys=True) + '\n')
    Path(args.out).with_suffix('.responses.json.gz').write_bytes(gzip.compress((raw + '\n').encode(), mtime=0))
    print(json.dumps({'usage': usage, 'judged': judged_counts, 'counts': counts}))
    return 0


if __name__ == '__main__':
    sys.exit(main())
