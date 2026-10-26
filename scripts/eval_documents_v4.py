#!/usr/bin/env python3
"""Offline development evaluation of documents units under ``/2`` and ``/4`` (no network, no payment).

Three conditions (DESIGN_LATEST_REVISION.md, acceptance test 3, but on development sets only):
  a: the served unit with its own route policy (documents5, ``workspace-documents/2``);
  b: the served unit with a ``/4`` policy (``--policy4``);
  c: a candidate unit (``--unit-dir``, swapped in for the served documents unit) with the ``/4`` policy.

Sets:
- ``dev`` / ``soak``: the frozen questions of documents-v4-dev-20261025 (``--questions``), one fresh
  ``public.Session`` turn each, as that run's ``run_eval.py`` does: under ``/2`` the world has no family fields (the
  view served today); under ``/4`` each document carries its explicit family. Scored with that run's pinned
  ``soak.py`` (measurement 4): the reply's fact claims and exactly the expected source cited.
- ``real``: a real-request set (``--questions`` = config/experiments/real-documents-set*.json), each case through
  ``assistant_general.documents_episode`` as ``real_documents_eval`` does, scored by the route's own scorer
  (``score_round`` at the policy's interface version: v2 for a, v4 for b and c). Rows also carry the v2 rule applied
  to the same transcript, for comparison across interfaces.

Every row records every tool call with its outcome. ``--part I/N`` scores every Nth case from I.
"""
import argparse
import hashlib
import importlib.util
import json
import re
import sys
import time
from pathlib import Path

NAMES = re.compile(r'\b(?:Cedar|Harbor|Meadow|Orchard|Summit|Willow) (\d+)\b')


def calls_of(calls):
    out = []
    for c in calls:
        result = c['result'] if isinstance(c['result'], dict) else {}
        out.append({'name': c['call']['name'], 'arguments': c['call'].get('arguments'),
                    'outcome': result.get('outcome'), 'chosen': (result.get('document') or {}).get('id'),
                    'error': result.get('error'), 'submitted': result.get('submitted'),
                    'results': [r['id'] for r in result.get('results', [])] if 'results' in result else None})
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--descriptor', type=Path, required=True)
    parser.add_argument('--home', type=Path, required=True)
    parser.add_argument('--condition', choices=('a', 'b', 'c'), required=True)
    parser.add_argument('--policy4', type=Path)
    parser.add_argument('--unit-dir', type=Path)
    parser.add_argument('--set', choices=('dev', 'soak', 'real'), required=True)
    parser.add_argument('--questions', type=Path, required=True)
    parser.add_argument('--soak', type=Path, help='the pinned soak.py scorer (dev and soak sets)')
    parser.add_argument('--part', default='0/1')
    parser.add_argument('--threads', type=int, default=8)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()

    from neuroshard.client import live_runtime

    live_runtime.configure()
    import torch

    from neuroshard.assistant import network
    from neuroshard.assistant.public import Session
    from neuroshard.evolution import assistant_documents as documents
    from neuroshard.evolution import assistant_general as general
    from neuroshard.evolution import growth_runtime
    from neuroshard.evolution.assistant_serving import cached_responder

    torch.set_num_threads(args.threads)
    descriptor = network.descriptor(args.descriptor)
    version = network.version(descriptor)[0]
    served = growth_runtime.Served(descriptor, args.home, version.spec, routes=('documents',))
    if args.condition == 'c':
        # The candidate replaces the served documents unit's files; nothing is fetched (the served files matched).
        served.directories['documents'] = args.unit_dir
    unit_dir = served.directories['documents']
    unit_sha = hashlib.sha256((unit_dir / 'trainable.safetensors').read_bytes()).hexdigest()
    policy = served.policies['documents'] if args.condition == 'a' else json.loads(args.policy4.read_text())
    policies = {**served.policies, 'documents': policy}
    model, tokenizer = served('documents'), served.tokenizer
    index, parts = (int(x) for x in args.part.split('/'))
    frozen = json.loads(args.questions.read_text())
    rows = []
    soak = None
    if args.set in ('dev', 'soak'):
        spec = importlib.util.spec_from_file_location('soak', args.soak)
        soak = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(soak)
        worlds = {w['world_id']: w for w in frozen['worlds']}
        items = frozen['questions'][index::parts]
    else:
        items = frozen['cases'][index::parts]
    for item in items:
        started = time.monotonic()
        responder = cached_responder(model, tokenizer, policy)
        if soak is not None:
            w = worlds[item['world_id']]
            world = w['world']
            if args.condition != 'a':
                world = {**w['world'], 'documents': [{**d, 'family': w['families'][d['id']]}
                                                     if d['id'] in w['families'] else dict(d)
                                                     for d in w['world']['documents']]}
            session = Session(policy, world, name=version.name, route_policies=policies)
            result = session.turn(item['question'], responder, route='documents')
            reply = result['final_text'] or ''
            ids = soak.cited(reply)
            fact = soak.score(NAMES.sub(r'Cedar \1', reply), item['kind'], item['expected'])
            grounded = ids == {item['expected_source']} and ids <= {d['id'] for d in w['world']['documents']}
            snapshot = result['snapshot']
            row = {'id': item['qid'], 'type': item['type'], 'question': item['question'], 'reply': reply,
                   'fact': fact, 'grounded': grounded, 'passed': fact == 'match' and grounded,
                   'expected': item['expected'], 'expected_source': item['expected_source'],
                   'cited_ids': sorted(ids), 'failure': result['failure'], 'calls': calls_of(session.calls)}
        else:
            episode = general.documents_episode(item, responder, policy)
            last = (episode.get('rounds') or [{}])[-1]
            snapshot = last.get('snapshot') or {}
            reply = last.get('final_text', '') or ''
            expected = item['turns'][0]['expected']
            lenient = bool(snapshot) and documents.score_round(snapshot, expected, last.get('completed', False),
                                                               reply, 2)
            row = {'id': item['id'], 'type': item['family'], 'question': item['turns'][0]['user'], 'reply': reply,
                   'passed': bool(episode['passed']), 'scorer_version': episode['version'], 'passed_v2_rule': lenient,
                   'failure': last.get('failure'), 'calls': calls_of(episode['calls'])}
        row.update(answer=snapshot.get('answer'), not_found=snapshot.get('not_found'),
                   read_ids=snapshot.get('read_ids'), seconds=round(time.monotonic() - started, 1))
        rows.append(row)
        print(json.dumps(row), flush=True)
        args.out.write_text(json.dumps({'partial': True, 'rows': rows}, indent=1))
    summary = {'condition': args.condition, 'set': args.set, 'part': args.part, 'threads': args.threads,
               'policy_sha256': hashlib.sha256(json.dumps(policy, sort_keys=True).encode()).hexdigest(),
               'interface': policy['interface'], 'unit_sha256': unit_sha, 'model_root': network.model_root(descriptor),
               'questions_sha256': hashlib.sha256(args.questions.read_bytes()).hexdigest(),
               'passed': sum(r['passed'] for r in rows), 'cases': len(rows), 'python': sys.version.split()[0],
               'torch': torch.__version__, 'rows': rows}
    args.out.write_text(json.dumps(summary, indent=1))
    print(json.dumps({k: v for k, v in summary.items() if k != 'rows'}), flush=True)


if __name__ == '__main__':
    main()
