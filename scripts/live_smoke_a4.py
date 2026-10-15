"""Live a4-general smoke: a paid conversation on every route, settled, refunded and audited; then restart recovery.

Run on the seed host in the live runtime:
    PYTHONPATH=<checkout>/src <venv>/bin/python scripts/live_smoke_a4.py --network <descriptor> --home <dir> [--phase recovery]

Routes are chosen by the served router from the user's words, as for any user: a drafting request, a
scheduling request, a question about the user's documents, and a chat request.
"""

from neuroshard.evolution.granite_shard_execution import configure

configure()

import argparse  # noqa: E402
import json  # noqa: E402
from pathlib import Path  # noqa: E402
import time  # noqa: E402

from neuroshard.assistant import network  # noqa: E402
from neuroshard.evolution import assistant_chat_data as chat  # noqa: E402
from neuroshard.evolution import assistant_documents_data as docs  # noqa: E402
from neuroshard.evolution import assistant_schedule_data as schedule  # noqa: E402
from neuroshard.evolution import assistant_workflow_data as drafting  # noqa: E402


def cases(phase):
    if phase == 'recovery':
        case = chat.cases('generated-development5')[0]
        return [('chat after owner restart', {'documents': []}, [case['turns'][0]['user']], 'chat')]
    draft, slot = drafting.cases('development')[0], schedule.cases('development')[0]
    document, talk = docs.cases('generated-development5')[0], chat.cases('generated-development5')[0]
    return [('drafting route', draft['world'], [t['user'] for t in draft['turns']], 'drafting'),
            ('scheduling route', slot['world'], [t['user'] for t in slot['turns']], 'scheduling'),
            ('documents route', document['world'], [document['turns'][0]['user']], 'documents'),
            ('chat route', {'documents': []}, [talk['turns'][0]['user']], 'chat')]


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--network', type=Path, required=True)
    parser.add_argument('--home', type=Path, required=True)
    parser.add_argument('--phase', default='routes', choices=('routes', 'recovery'))
    args = parser.parse_args()
    descriptor = network.descriptor(args.network)
    chain = network.Chain(network.endpoints(descriptor), descriptor['chain_id'])
    served, stage = network.fetch_stage(descriptor, 0, args.home / 'holdings')
    account = network.Account(args.home / 'account.key')
    rows = []
    for label, world, turns, route in cases(args.phase):
        talk = network.Conversation(served, stage, chain, account, world, budget=16384, price=100000,
                                    faucet_url=descriptor['faucet'], threads=8)
        row = {'label': label, 'expected_route': route, 'job_id': talk.job_id,
               'owners': [key for key, _ in talk.owners], 'turns': []}
        try:
            for text in turns:
                reply = talk.say(text)
                row['turns'].append({key: reply.get(key) for key in ('route', 'completed', 'failure', 'final_text',
                                                                     'seconds')})
                if reply['failure']:
                    break
        finally:
            talk.close()
        row['settlement'] = chain.settled(talk.job_id, timeout=1800)
        rows.append(row)
        print(json.dumps({k: row[k] for k in ('label', 'job_id', 'turns', 'settlement')})[:1500], flush=True)
    ok = all(r['settlement'] and r['settlement']['status'] == 'settled' and r['turns']
             and all(t['completed'] and t['route'] == r['expected_route'] for t in r['turns']) for r in rows)
    out = {'phase': args.phase, 'passed': ok, 'rows': rows, 'unix': time.time()}
    (args.home / f'smoke-{args.phase}.json').write_text(json.dumps(out, indent=2) + '\n')
    raise SystemExit(0 if ok else 1)


if __name__ == '__main__':
    main()
