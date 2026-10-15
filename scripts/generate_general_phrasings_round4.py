"""Round-4 phrasing sets for the general route: development4 and the sealed confirmation6.

    PYTHONPATH=src python scripts/generate_general_phrasings_round4.py

confirmation5 is spent and development3 has judged L8, so every earlier set is training wording
(``generated-train-styles2``). C1 (the chat unit) is judged on development4 and, with L7 for documents,
confirmed on confirmation6. Each is written in a style no earlier set used, with the same judge as
every set. A kept template may not equal any committed template, and confirmation6 also shares no
normalised ten-word span with any template that training can use or with development4 (eight words
removed generic phrases such as "the team meeting notes" too broadly), and may be up to 420 characters.
"""

from concurrent.futures import ThreadPoolExecutor
import gzip
import hashlib
import json
from pathlib import Path
import sys
import threading

from neuroshard.evolution import assistant_phrasings as phrasings

sys.path.insert(0, str(Path(__file__).resolve().parent))
from generate_general_phrasings import SEED  # noqa: E402

PLAN = {'development4': {'calls': 1, 'count': 20,
                         'style': 'written by someone whose first language is not English: simple vocabulary, '
                                  'slightly unusual word order or articles, polite and clear'},
        'confirmation6': {'calls': 2, 'count': 20,
                          'style': 'part of a longer message: one or two sentences about something unrelated '
                                   '(a plan, a worry, news), then the request, then a short closing remark; vary '
                                   'formality; some assume the thing asked about exists', 'limit': 420}}
SPAN = 10


def norm(text):
    return ' '.join(text.lower().split())


def main():
    root = Path(__file__).resolve().parents[1]
    files = [phrasings.FILE, phrasings.FILE2, phrasings.FILE3]
    committed = [json.loads((root / f).read_text())['sets'] for f in files]
    known = {norm(t) for sets in committed for by in sets.values() for rows in by.values() for ts in rows.values()
             for t in ts}
    trainable = [t for sets in committed for by in sets.values() for rows in by.values() for ts in rows.values()
                 for t in ts]
    bedrock = phrasings.client()
    # Calls are cached by key, so a rerun repeats only what changed; the cache is rebuilt from the recorded
    # responses of an earlier run when present (the same calls, the same answers).
    cache = {}
    previous = root / phrasings.FILE4
    if previous.with_suffix('.responses.json.gz').exists():
        old = json.loads(gzip.decompress(previous.with_suffix('.responses.json.gz').read_bytes()))
        for row in old['generation']:
            key = 'gen:' + json.dumps([row['route'], row['split'], row['kind'], row['call']])
            names = phrasings.table(row['route'])[row['kind']][1]
            cache[key] = (phrasings.parse(row['text'], names, PLAN[row['split']].get('limit', 280)),
                          {'text': row['text'], 'usage': row['usage']})
        for row in old['judging']:
            cache['judge:' + json.dumps([row['route'], row['split'], row['kind'], row['template']])] = (row['kept'],
                                                                                                       row['trials'])
    lock = threading.Lock()

    def cached(key, compute):
        with lock:
            if key in cache:
                return cache[key]
        value = compute()
        with lock:
            cache[key] = value
        return value

    jobs = [(route, split, kind, call) for split, plan in PLAN.items() for route in ('documents', 'chat')
            for kind in phrasings.table(route) for call in range(plan['calls'])]

    def work(job):
        route, split, kind, _ = job
        rows, raw = cached('gen:' + json.dumps(job), lambda: phrasings.generate(
            kind, PLAN[split]['count'], PLAN[split]['style'], bedrock, route, PLAN[split].get('limit', 280)))
        judged = [(t, *cached('judge:' + json.dumps([route, split, kind, t]),
                              lambda t=t: phrasings.judge(route, kind, t, bedrock, SEED + 4))) for t in rows]
        print(*job, len(rows), sum(ok for _, ok, _ in judged), flush=True)
        return job, raw, judged

    with ThreadPoolExecutor(max_workers=12) as pool:
        results = list(pool.map(work, jobs))
    record, kept = {'generation': [], 'judging': []}, {}
    for (route, split, kind, call), raw, judged in results:
        record['generation'].append({'route': route, 'split': split, 'kind': kind, 'call': call, **raw})
        bucket = kept.setdefault((route, split, kind), [])
        for template, ok, trials in judged:
            record['judging'].append({'route': route, 'split': split, 'kind': kind, 'template': template,
                                      'kept': ok, 'trials': trials})
            if ok and norm(template) not in known and template not in bucket:
                bucket.append(template)
    development4 = [t for (_, s, _), rows in kept.items() if s == 'development4' for t in rows]
    seen = set().union(*(phrasings.spans(t, SPAN) for t in trainable + development4))
    sets, dropped = {}, {}
    for (route, split, kind), rows in kept.items():
        if split == 'confirmation6':
            fresh = [t for t in rows if not phrasings.spans(t, SPAN) & seen and norm(t) not in map(norm, development4)]
            dropped[f'{route}/{kind}'] = len(rows) - len(fresh)
            rows = fresh
        sets.setdefault(route, {}).setdefault(split, {})[kind] = sorted(rows)
    raw = json.dumps(record, sort_keys=True)
    usage = {key: sum(r['usage'].get(key, 0) for r in record['generation'])
             + sum(t['usage'].get(key, 0) + t.get('usage_solve', {}).get(key, 0)
                   for j in record['judging'] for t in j['trials']) for key in ('inputTokens', 'outputTokens')}
    counts = {route: {split: {k: len(v) for k, v in by.items()} for split, by in splits.items()}
              for route, splits in sets.items()}
    out = {'format': 'neuroshard-assistant-general-phrasings-round4/1', 'model': phrasings.MODEL, 'seed': SEED + 4,
           'plan': PLAN, 'sets': sets, 'counts': counts, 'confirmation6_dropped_for_overlap': dropped, 'usage': usage,
           'responses_sha256': hashlib.sha256(raw.encode()).hexdigest(),
           'holdout_rule': 'No template equal (case- and space-normalised) to any committed template; confirmation6 '
                           'also shares no normalised ten-word span with any committed template or with development4. Wrapped confirmation6 '
                           'requests may be up to 420 characters.',
           'rule': 'Claude wrote user-side phrasings only; facts, constraints, expected values and every assistant '
                   'message we learn from are ours.'}
    path = root / phrasings.FILE4
    path.write_text(json.dumps(out, indent=1, sort_keys=True) + '\n')
    path.with_suffix('.responses.json.gz').write_bytes(gzip.compress((raw + '\n').encode(), mtime=0))
    print(json.dumps({'usage': usage, 'counts': counts}))
    return 0


if __name__ == '__main__':
    sys.exit(main())
