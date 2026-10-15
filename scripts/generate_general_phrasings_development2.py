"""A second development phrasing set for the general route, disjoint from every committed set.

    PYTHONPATH=src python scripts/generate_general_phrasings_development2.py

generated-development judged L5 and L6, and L6's misses were read to design L7, so L7 is judged on
templates nobody has seen. A third style (polite messages that open with one line of background), the
same judge as training, two calls of 20 per kind.
A judged template is kept unless it equals (case- and space-normalised) a committed training,
development or confirmation4 template: the same template-level holdout as generated-development.
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

COUNT = 20
CALLS = 2
STYLE = ('polite, neutral messages as written to a personal assistant, each opening with one sentence of '
         'background about the situation before the request; vary the background widely')


def main():
    root = Path(__file__).resolve().parents[1]
    committed = json.loads((root / phrasings.FILE).read_text())['sets']
    bedrock = phrasings.client()
    jobs = [(route, kind, call) for route in ('documents', 'chat') for kind in phrasings.table(route)
            for call in range(CALLS)]

    def work(job):
        route, kind, _ = job
        rows, raw = phrasings.generate(kind, COUNT, STYLE, bedrock, route)
        judged = [(t, *phrasings.judge(route, kind, t, bedrock, SEED + 2)) for t in rows]
        print(route, kind, len(rows), sum(ok for _, ok, _ in judged), flush=True)
        return job, raw, judged

    with ThreadPoolExecutor(max_workers=12) as pool:
        results = list(pool.map(work, jobs))
    def norm(text):
        return ' '.join(text.lower().split())

    known = {norm(t) for by in committed.values() for rows in by.values() for ts in rows.values() for t in ts}
    sets, record = {}, {'generation': [], 'judging': []}
    kept_by = {}
    for (route, kind, call), raw, judged in results:
        record['generation'].append({'route': route, 'kind': kind, 'call': call, **raw})
        kept = kept_by.setdefault((route, kind), [])
        for template, ok, trials in judged:
            record['judging'].append({'route': route, 'kind': kind, 'template': template, 'kept': ok, 'trials': trials})
            if ok and norm(template) not in known and template not in kept:
                kept.append(template)
    for (route, kind), kept in kept_by.items():
        sets.setdefault(route, {}).setdefault('development2', {})[kind] = sorted(kept)
    raw = json.dumps(record, sort_keys=True)
    usage = {key: sum(r['usage'].get(key, 0) for r in record['generation'])
             + sum(t['usage'].get(key, 0) + t.get('usage_solve', {}).get(key, 0)
                   for j in record['judging'] for t in j['trials']) for key in ('inputTokens', 'outputTokens')}
    counts = {route: {k: len(v) for k, v in by['development2'].items()} for route, by in sets.items()}
    out = {'format': 'neuroshard-assistant-general-phrasings-development2/1', 'model': phrasings.MODEL,
           'style': STYLE, 'count_per_call': COUNT, 'calls': CALLS, 'seed': SEED + 2, 'sets': sets,
           'counts': counts, 'usage': usage,
           'responses_sha256': hashlib.sha256(raw.encode()).hexdigest(),
           'holdout_rule': 'Template-level, as for generated-development: no template identical (case- and space-'
                           'normalised) to any committed training, development or confirmation4 template. Generated '
                           'after L5 and L6 were judged; no unit has seen these templates.',
           'rule': 'Claude wrote user-side phrasings only; facts, constraints, expected values and every assistant '
                   'message we learn from are ours.'}
    path = root / phrasings.FILE2
    path.write_text(json.dumps(out, indent=1, sort_keys=True) + '\n')
    path.with_suffix('.responses.json.gz').write_bytes(gzip.compress((raw + '\n').encode(), mtime=0))
    print(json.dumps({'usage': usage, 'counts': counts}))
    return 0 if all(v >= 4 for by in counts.values() for v in by.values()) else 1


if __name__ == '__main__':
    sys.exit(main())
