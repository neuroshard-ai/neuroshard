#!/usr/bin/env python3
"""Merge the per-part rows of eval_documents_v4.py and tabulate conditions a, b, c (development results only).

  python analyze.py RUNDIR OUT.md [--real1 SET.json --real2 SET.json]
Per set and type: passed/total with 95% Wilson intervals; paired two-sided sign tests (c vs a, c vs b, b vs a);
tool use (which tools each condition called, how often selection tools were used); a failure diagnosis per row.
"""
import argparse
import collections
import json
import math
from pathlib import Path

SETS = ('soak', 'dev', 'real1', 'real2')


def wilson(k, n, z=1.959964):
    if not n:
        return (0.0, 0.0)
    p = k / n
    centre, half = p + z * z / (2 * n), z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return ((centre - half) / (1 + z * z / n), (centre + half) / (1 + z * z / n))


def sign_test(wins, losses):
    n = wins + losses
    if not n:
        return 1.0
    k = min(wins, losses)
    tail = sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n
    return min(1.0, 2 * tail)


def load(rundir, condition, name):
    rows, meta = [], []
    for path in sorted(Path(rundir).glob(f'{condition}-{name}-*.json')):
        data = json.loads(path.read_text())
        if data.get('partial'):
            raise ValueError(f'{path} is partial')
        rows += data['rows']
        meta.append({k: v for k, v in data.items() if k != 'rows'})
    return rows, meta


def names(row):
    return [c['name'] for c in row['calls']]


def diagnose(row, expected=None):
    if row['passed']:
        return 'pass'
    if row.get('failure'):
        return 'no final reply (' + row['failure'].split(' ')[0] + ' budget)'
    called = names(row)
    if expected is not None:  # a real-request row, scored by the route's scorer
        if expected.get('not_found'):
            return 'answered instead of not found' if row.get('answer') else 'not-found expected: other'
        if row.get('not_found'):
            return 'reported not found'
        if row.get('answer') is None:
            if any(c['name'] == 'submit_answer' and c.get('error') for c in row['calls']):
                return 'answer refused (evidence not found)'
            return 'no submitted answer' + (' (prose passes v2 rule)' if row.get('passed_v2_rule') else '')
        return 'wrong fact or sources'
    if row.get('not_found'):
        return 'reported not found'
    if not row['cited_ids']:
        return 'no document cited'
    selected = 'current_document' in called or 'effective_document' in called
    if row['expected_source'] in row['cited_ids'] and len(row['cited_ids']) > 1:
        return 'expected source plus others'
    if row['expected_source'] not in row['cited_ids']:
        return 'wrong source' + (' (after selection tool)' if selected else ' (no selection tool)')
    return 'wrong or extra fact (' + row['fact'] + ')'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('rundir', type=Path)
    parser.add_argument('out', type=Path)
    parser.add_argument('--real1', type=Path)
    parser.add_argument('--real2', type=Path)
    args = parser.parse_args()
    expected = {}
    for path in (args.real1, args.real2):
        if path:
            expected.update({c['id']: c['turns'][0]['expected'] for c in json.loads(path.read_text())['cases']})
    data, metas, lines, summary = {}, {}, [], {}
    conditions = [c for c in 'abc' if list(args.rundir.glob(f'{c}-*.json'))]
    for c in conditions:
        for s in SETS:
            data[c, s], metas[c, s] = load(args.rundir, c, s)
    lines.append('# Development evaluation: conditions ' + ', '.join(conditions) + '\n')
    lines.append('Development results only, not quality claims. Wilson 95% intervals; sign tests are two-sided exact.\n')
    for s in SETS:
        lines.append(f'\n## {s}\n')
        types = sorted({r['type'] for c in conditions for r in data[c, s]})
        lines.append('| Type | n | ' + ' | '.join(conditions) + ' |')
        lines.append('| --- | --- | ' + ' | '.join('---' for _ in conditions) + ' |')
        for t in types + ['all']:
            cells = []
            n = None
            for c in conditions:
                rows = [r for r in data[c, s] if t == 'all' or r['type'] == t]
                k, n = sum(r['passed'] for r in rows), len(rows)
                lo, hi = wilson(k, n)
                cells.append(f'{k}/{n} ({lo:.2f}–{hi:.2f})')
                summary.setdefault(s, {}).setdefault(c, {})[t] = [k, n]
            lines.append(f'| {t} | {n} | ' + ' | '.join(cells) + ' |')
        if s.startswith('real'):
            cells = []
            for c in conditions:
                rows = data[c, s]
                cells.append(f"{sum(bool(r.get('passed_v2_rule')) for r in rows)}/{len(rows)}")
            lines.append('\nThe v2 rule on the same transcripts (prose citing IDs also counts): ' +
                         ', '.join(f'{c} {x}' for c, x in zip(conditions, cells)) + '.\n')
        by = {c: {r['id']: r['passed'] for r in data[c, s]} for c in conditions}
        lines.append('')
        for x, y in (('c', 'a'), ('c', 'b'), ('b', 'a')):
            if x in by and y in by:
                common = sorted(set(by[x]) & set(by[y]))
                wins = sum(by[x][i] and not by[y][i] for i in common)
                losses = sum(by[y][i] and not by[x][i] for i in common)
                p = sign_test(wins, losses)
                summary.setdefault(s, {}).setdefault('paired', {})[f'{x}_vs_{y}'] = [wins, losses, p]
                lines.append(f'- {x} vs {y}: {wins} wins, {losses} losses, p = {p:.3g}')
        lines.append('\nTool use (conversations calling each tool) and failures:\n')
        lines.append('| Condition | current_document | effective_document | submit_answer | report_not_found | mean calls |')
        lines.append('| --- | --- | --- | --- | --- | --- |')
        for c in conditions:
            rows = data[c, s]
            count = lambda name: sum(name in names(r) for r in rows)
            mean = sum(len(r['calls']) for r in rows) / max(1, len(rows))
            lines.append(f"| {c} | {count('current_document')} | {count('effective_document')} | "
                         f"{count('submit_answer')} | {count('report_not_found')} | {mean:.1f} |")
        for c in conditions:
            reasons = collections.Counter(diagnose(r, expected.get(r['id']) if s.startswith('real') else None)
                                          for r in data[c, s])
            lines.append(f'\n{c}: ' + ', '.join(f'{k} {v}' for k, v in reasons.most_common()))
            summary.setdefault(s, {}).setdefault('diagnosis', {})[c] = dict(reasons)
    lines.append('\n## Runs\n')
    for (c, s), meta in sorted(metas.items()):
        if meta:
            m = meta[0]
            lines.append(f"- {c}/{s}: {sum(x['cases'] for x in meta)} cases in {len(meta)} parts, interface "
                         f"{m['interface']}, unit {m['unit_sha256'][:16]}, policy {m['policy_sha256'][:16]}, "
                         f"threads {m['threads']}, Python {m['python']}, torch {m['torch']}")
    args.out.write_text('\n'.join(lines) + '\n')
    args.out.with_suffix('.json').write_text(json.dumps(summary, indent=1))
    print('\n'.join(lines))


if __name__ == '__main__':
    main()
