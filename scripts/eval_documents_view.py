#!/usr/bin/env python3
"""Offline before/after of the documents-route fix: the four soak questions, the served documents model, whole.

The served model is the pinned parent plus the served documents unit, run whole on one machine. It is
numerically what the owners compute together. Each question is one fresh ``public.Session`` turn on the
documents route, exactly as ``assistant chat`` runs it, scored as the soak scores it (``scripts/soak.py``):
- ``fact``: the claimed number or date;
- ``grounded``: the latest approved document is cited.

``--view before`` patches ``interfaces.superseded`` to hide nothing, which is the pilot-2-rc2 behaviour.
``--view after`` uses this checkout's selection. Nothing is sent to the network.

    NEUROSHARD_MODELS=... python scripts/eval_documents_view.py --network NET.json --home H --view after --out R.json
"""
import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--network', type=Path)
    parser.add_argument('--home', type=Path, required=True)
    parser.add_argument('--view', choices=('before', 'after'), required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()

    from neuroshard.client import live_runtime
    live_runtime.configure()
    import torch

    import soak
    from neuroshard.assistant import interfaces, network
    from neuroshard.assistant.public import Session
    from neuroshard.client.assistant_cli import workspace
    from neuroshard.evolution import growth_runtime
    from neuroshard.evolution.assistant_serving import cached_responder

    torch.set_num_threads(8)
    if args.view == 'before':
        interfaces.superseded = lambda documents: set()
    descriptor = network.descriptor(args.network)
    served_version = network.version(descriptor)[0]
    served = growth_runtime.Served(descriptor, args.home, served_version.spec, routes=('documents',))
    model, policy = served('documents'), served.policies['documents']
    world, _ = workspace(0, None)
    rows = []
    for question, kind, expected in soak.QUESTIONS:
        session = Session(policy, world, name=served_version.name, route_policies=served.policies)
        started = time.monotonic()
        result = session.turn(question, cached_responder(model, served.tokenizer, policy), route='documents')
        reply = result['final_text'] or ''
        row = {'question': question, 'reply': reply, 'fact': soak.score(reply, kind, expected),
               'grounded': soak.grounded(reply, question), 'read_ids': result['snapshot']['read_ids'],
               'found_ids': result['snapshot']['found_ids'], 'seconds': round(time.monotonic() - started, 1)}
        rows.append(row)
        print(json.dumps(row), flush=True)
    summary = {'view': args.view, 'fact_match': sum(r['fact'] == 'match' for r in rows),
               'grounded': sum(bool(r['grounded']) for r in rows),
               'correct_and_grounded': sum(r['fact'] == 'match' and bool(r['grounded']) for r in rows),
               'questions': len(rows), 'model_root': network.model_root(descriptor), 'rows': rows}
    args.out.write_text(json.dumps(summary, indent=1))
    print(json.dumps({k: v for k, v in summary.items() if k != 'rows'}))


if __name__ == '__main__':
    main()
