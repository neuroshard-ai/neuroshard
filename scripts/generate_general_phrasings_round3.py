"""Round-3 phrasing sets for the general route: development3 and the sealed confirmation5.

    PYTHONPATH=src python scripts/generate_general_phrasings_round3.py

confirmation4 is spent, and development and development2 have judged units, so all three become
training wording (``generated-train-styles``). L8 is judged on development3 and, if it advances,
confirmed on confirmation5. Each is written in a style no earlier set used, with the same judge as
every set. A kept template may not equal any committed template, and confirmation5 also shares no
normalised eight-word span with any template that training can use or with development3.
"""

from concurrent.futures import ThreadPoolExecutor
import gzip
import hashlib
import json
from pathlib import Path
import sys

from neuroshard.evolution import assistant_phrasings as phrasings

sys.path.insert(0, str(Path(__file__).resolve().parent))
from generate_general_phrasings import SEED  # noqa: E402

PLAN = {'development3': {'calls': 1, 'count': 20,
                         'style': 'typed quickly on a phone: mostly lowercase, little punctuation, short and direct; '
                                  'some assume the thing asked about exists'},
        'confirmation5': {'calls': 2, 'count': 20,
                          'style': 'either checking something the user half-remembers ("I believe ..., can you '
                                   'confirm ...") or a request passed on for someone else ("my partner wants to know '
                                   '..."); vary formality; some assume the thing asked about exists'}}


def norm(text):
    return ' '.join(text.lower().split())


def main():
    root = Path(__file__).resolve().parents[1]
    files = [phrasings.FILE, phrasings.FILE2]
    committed = [json.loads((root / f).read_text())['sets'] for f in files]
    known = {norm(t) for sets in committed for by in sets.values() for rows in by.values() for ts in rows.values()
             for t in ts}
    trainable = [t for sets in committed for by in sets.values() for rows in by.values() for ts in rows.values()
                 for t in ts]
    bedrock = phrasings.client()
    jobs = [(route, split, kind, call) for split, plan in PLAN.items() for route in ('documents', 'chat')
            for kind in phrasings.table(route) for call in range(plan['calls'])]

    def work(job):
        route, split, kind, _ = job
        rows, raw = phrasings.generate(kind, PLAN[split]['count'], PLAN[split]['style'], bedrock, route)
        judged = [(t, *phrasings.judge(route, kind, t, bedrock, SEED + 3)) for t in rows]
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
    development3 = [t for (_, s, _), rows in kept.items() if s == 'development3' for t in rows]
    seen = set().union(*(phrasings.spans(t, 8) for t in trainable + development3))
    sets, dropped = {}, {}
    for (route, split, kind), rows in kept.items():
        if split == 'confirmation5':
            fresh = [t for t in rows if not phrasings.spans(t, 8) & seen and norm(t) not in map(norm, development3)]
            dropped[f'{route}/{kind}'] = len(rows) - len(fresh)
            rows = fresh
        sets.setdefault(route, {}).setdefault(split, {})[kind] = sorted(rows)
    raw = json.dumps(record, sort_keys=True)
    usage = {key: sum(r['usage'].get(key, 0) for r in record['generation'])
             + sum(t['usage'].get(key, 0) + t.get('usage_solve', {}).get(key, 0)
                   for j in record['judging'] for t in j['trials']) for key in ('inputTokens', 'outputTokens')}
    counts = {route: {split: {k: len(v) for k, v in by.items()} for split, by in splits.items()}
              for route, splits in sets.items()}
    out = {'format': 'neuroshard-assistant-general-phrasings-round3/1', 'model': phrasings.MODEL, 'seed': SEED + 3,
           'plan': PLAN, 'sets': sets, 'counts': counts, 'confirmation5_dropped_for_overlap': dropped, 'usage': usage,
           'responses_sha256': hashlib.sha256(raw.encode()).hexdigest(),
           'holdout_rule': 'No template equal (case- and space-normalised) to any committed template; confirmation5 '
                           'also shares no normalised eight-word span with any committed template or with development3.',
           'rule': 'Claude wrote user-side phrasings only; facts, constraints, expected values and every assistant '
                   'message we learn from are ours.'}
    path = root / phrasings.FILE3
    path.write_text(json.dumps(out, indent=1, sort_keys=True) + '\n')
    path.with_suffix('.responses.json.gz').write_bytes(gzip.compress((raw + '\n').encode(), mtime=0))
    print(json.dumps({'usage': usage, 'counts': counts}))
    return 0


if __name__ == '__main__':
    sys.exit(main())
