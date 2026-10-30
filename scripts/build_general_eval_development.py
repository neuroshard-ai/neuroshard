"""Build the general-capability evaluation's open development split from its authored sources.

    PYTHONPATH=src python scripts/build_general_eval_development.py [--check]

Every item is validated; every reference answer is re-derived or executed: coding references must pass
their hidden tests in the sandbox, planning references must satisfy their validators (and be optimal where
declared), reasoning answers with a formula are recomputed. The split is written to
``src/neuroshard/evolution/general_eval/data/development/<area>.jsonl`` and its digest printed.
``--check`` rebuilds in memory and fails if the committed files differ. The sealed split is never built here.
"""

import argparse
from fractions import Fraction
import json
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / 'general_eval_dev'))

import coding  # noqa: E402
import planning  # noqa: E402
import reasoning_factual as rf  # noqa: E402
import tool_use  # noqa: E402
import writing  # noqa: E402

from neuroshard.evolution.general_eval import items as schema, sandbox, scorers  # noqa: E402

WRITTEN = '2026-10-30'
PROVENANCE = {'source': 'fresh', 'author': 'neuroshard general-eval development authoring', 'written': WRITTEN}


def build():
    rows = []

    def add(area, family, user, scoring, **extra):
        number = sum(r['area'] == area for r in rows) + 1
        rows.append({'id': f'{area}-dev-{number:03d}', 'area': area, 'split': 'development', 'family': family,
                     'turns': [{'user': user}], 'scoring': scoring, 'provenance': dict(PROVENANCE), **extra})

    for family, user, scoring in rf.REASONING:
        add('reasoning', family, user + rf.END, scoring)
    for family, user, scoring in coding.CODING:
        add('coding', family, user + coding.ASK, scoring)
    for family, user, scoring in writing.WRITING:
        add('writing', family, user, scoring, judged={'rubric': 'writing/1'})
    for family, user, scoring in rf.FACTUAL:
        add('factual', family, user + rf.ASK, scoring)
    for family, user, scoring in planning.PLANNING:
        add('planning', family, user + planning.JSON, scoring)
    for family, user, scoring in tool_use.TOOL_USE:
        fixture = tool_use.DOCUMENTS if scoring['toolset'] == 'documents/2' else tool_use.FIXTURE
        add('tool_use', family, user, scoring, fixture=json.loads(json.dumps(fixture)))
    for row in rows:
        schema.validate(row)
    return rows


def check_references(rows):
    problems = []
    reasoning = [r for r in rows if r['area'] == 'reasoning']
    for index, value in rf.CHECKS.items():
        stated = Fraction(str(reasoning[index]['scoring']['answer']['value']))
        if abs(stated - Fraction(str(value))) > Fraction(1, 10 ** 6):
            problems.append(f'{reasoning[index]["id"]}: stated {stated}, recomputed {value}')
    for row in rows:
        spec = row['scoring']
        if spec['kind'] == 'code':
            result = sandbox.run_python(spec['reference'], spec['tests'], timeout=spec['timeout'])
            if not result['passed']:
                problems.append(f'{row["id"]}: reference fails its tests: {result.get("error")}')
        elif spec['kind'] == 'plan':
            got = scorers.score_plan(spec, '```json\n' + json.dumps(spec['reference']) + '\n```')
            if not got['passed']:
                problems.append(f'{row["id"]}: reference plan invalid: {got["detail"]["problems"]}')
    return problems


def write(rows, root):
    out = {}
    for area in schema.AREAS:
        lines = [json.dumps(r, sort_keys=True, ensure_ascii=False) for r in rows if r['area'] == area]
        out[root / f'{area}.jsonl'] = '\n'.join(lines) + '\n'
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--check', action='store_true')
    args = parser.parse_args()
    rows = build()
    problems = check_references(rows)
    if problems:
        raise SystemExit('reference check failed:\n' + '\n'.join(problems))
    root = schema.DATA / 'development'
    files = write(rows, root)
    if args.check:
        stale = [str(p) for p, text in files.items() if not p.exists() or p.read_text(encoding='utf-8') != text]
        if stale:
            raise SystemExit('committed development split differs from its sources: ' + ', '.join(stale))
    else:
        root.mkdir(parents=True, exist_ok=True)
        for path, text in files.items():
            path.write_text(text, encoding='utf-8')
    loaded = schema.load_split('development')
    print(json.dumps({'items': len(loaded), 'counts': schema.counts(loaded),
                      'split_sha256': schema.split_digest(loaded)}, indent=2))


if __name__ == '__main__':
    main()
