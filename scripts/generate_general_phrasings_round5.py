"""Round-5 phrasing sets for the general route: development5 and the sealed confirmation7.

    PYTHONPATH=src python scripts/generate_general_phrasings_round5.py

confirmation6 is spent and development4 has judged C1, so every earlier set is training wording
(``generated-train-styles3``). L9 (documents) is judged on development5 and, with C1 for chat,
confirmed on confirmation7. Each is written in a style no earlier set used, with the same judge as
every set. A kept template may not equal any committed template, and confirmation7 also shares no
normalised ten-word span with any template that training can use or with development5 (eight words
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

PLAN = {'development5': {'calls': 1, 'count': 20,
                         'style': 'a reply in an ongoing chat thread: starts by reacting to something said earlier '
                                  '("thanks, that helped", "ok one more thing"), then the request; casual', 'limit': 360},
        'confirmation7': {'calls': 2, 'count': 20,
                          'style': 'a voice message transcribed to text: run-on sentences, filler words ("um", '
                                   '"like", "so yeah"), self-corrections, no punctuation discipline; the request is '
                                   'somewhere in the middle; some assume the thing asked about exists', 'limit': 420}}
SPAN = 10


def norm(text):
    return ' '.join(text.lower().split())


def main():
    root = Path(__file__).resolve().parents[1]
    files = [phrasings.FILE, phrasings.FILE2, phrasings.FILE3, phrasings.FILE4]
    committed = [json.loads((root / f).read_text())['sets'] for f in files]
    known = {norm(t) for sets in committed for by in sets.values() for rows in by.values() for ts in rows.values()
             for t in ts}
    trainable = [t for sets in committed for by in sets.values() for rows in by.values() for ts in rows.values()
                 for t in ts]
    bedrock = phrasings.client()
    # Calls are cached by key, so a rerun repeats only what changed; the cache is rebuilt from the recorded
    # responses of an earlier run when present (the same calls, the same answers).
    cache = {}
    previous = root / phrasings.FILE5
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
                              lambda t=t: phrasings.judge(route, kind, t, bedrock, SEED + 5))) for t in rows]
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
    development5 = [t for (_, s, _), rows in kept.items() if s == 'development5' for t in rows]
    seen = set().union(*(phrasings.spans(t, SPAN) for t in trainable + development5))
    sets, dropped = {}, {}
    for (route, split, kind), rows in kept.items():
        if split == 'confirmation7':
            fresh = [t for t in rows if not phrasings.spans(t, SPAN) & seen and norm(t) not in map(norm, development5)]
            dropped[f'{route}/{kind}'] = len(rows) - len(fresh)
            rows = fresh
        sets.setdefault(route, {}).setdefault(split, {})[kind] = sorted(rows)
    raw = json.dumps(record, sort_keys=True)
    usage = {key: sum(r['usage'].get(key, 0) for r in record['generation'])
             + sum(t['usage'].get(key, 0) + t.get('usage_solve', {}).get(key, 0)
                   for j in record['judging'] for t in j['trials']) for key in ('inputTokens', 'outputTokens')}
    counts = {route: {split: {k: len(v) for k, v in by.items()} for split, by in splits.items()}
              for route, splits in sets.items()}
    out = {'format': 'neuroshard-assistant-general-phrasings-round5/1', 'model': phrasings.MODEL, 'seed': SEED + 5,
           'plan': PLAN, 'sets': sets, 'counts': counts, 'confirmation7_dropped_for_overlap': dropped, 'usage': usage,
           'responses_sha256': hashlib.sha256(raw.encode()).hexdigest(),
           'holdout_rule': 'No template equal (case- and space-normalised) to any committed template; confirmation7 '
                           'also shares no normalised ten-word span with any committed template or with development5. Wrapped confirmation7 '
                           'requests may be up to 420 characters.',
           'rule': 'Claude wrote user-side phrasings only; facts, constraints, expected values and every assistant '
                   'message we learn from are ours.'}
    path = root / phrasings.FILE5
    path.write_text(json.dumps(out, indent=1, sort_keys=True) + '\n')
    path.with_suffix('.responses.json.gz').write_bytes(gzip.compress((raw + '\n').encode(), mtime=0))
    print(json.dumps({'usage': usage, 'counts': counts}))
    return 0


if __name__ == '__main__':
    sys.exit(main())
