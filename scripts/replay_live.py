"""Replay live blocks under this checkout and compare every state root with the app hash the validators signed.

Starts from the live state at height H (read from a validator RPC, light-client verified), then, as blocks
H+1..H+N commit, applies each block's transactions with this code and checks the resulting root against the
app hash in header H+2..H+N+1. With the upgrade unscheduled, every root must match: the new code is the old
chain. Then it continues one copy of the replayed state with the upgrade activated, to show the migration.

    PYTHONPATH=src python scripts/replay_live.py --blocks 300
"""
import argparse
import base64
import copy
import json
import sys
import time
from datetime import datetime, timezone
from unittest.mock import patch

from neuroshard.assistant import network
from neuroshard.inference import optimistic as ledger, optimistic_network as chain_network


def header(rpc, height):
    for _ in range(240):
        try:
            return chain_network.rpc(rpc, 'block', {'height': str(height)}, timeout=20)['block']
        except (chain_network.Rejected, OSError):
            time.sleep(0.5)
    raise TimeoutError(f'block {height} did not commit')


def seconds(text):
    return int(datetime.fromisoformat(text.rstrip('Z').split('.')[0]).replace(tzinfo=timezone.utc).timestamp())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--blocks', type=int, default=300)
    parser.add_argument('--rpc', default='http://34.194.51.150:26657')
    args = parser.parse_args()
    descriptor = network.descriptor()
    state = network.chain(descriptor).state()  # verified against the validators' signatures
    start, rpc = state['height'], args.rpc
    print(f'live state at {start}, root {ledger.root(state)[:16]}; replaying {args.blocks} blocks', flush=True)
    mismatches, transactions = 0, 0

    def refuse_proof(*_):
        raise ledger.NoVerdict('replay does not judge proofs')

    for height in range(start + 1, start + args.blocks + 1):
        block = header(rpc, height)
        txs = [json.loads(base64.b64decode(tx)) for tx in block['data']['txs'] or []]
        state = ledger.advance(state, height, seconds(block['header']['time']))
        for envelope in txs:
            try:
                state = ledger.transition(state, envelope, refuse_proof)
            except ledger.NoVerdict:
                print(f'block {height} has a proof; replay cannot judge it, stopping', flush=True)
                return 2
            transactions += 1
        committed = header(rpc, height + 1)['header']['app_hash'].lower()
        if ledger.root(state) != committed:
            mismatches += 1
            print(f'MISMATCH at {height}: replayed {ledger.root(state)[:16]}, committed {committed[:16]}', flush=True)
        elif height % 50 == 0 or txs:
            print(f'{height}: match ({len(txs)} tx)', flush=True)
    print(f'{args.blocks} blocks, {transactions} transactions, {mismatches} mismatches', flush=True)

    # The same state continued one block with the upgrade active: membership and the registry are founded.
    with patch.dict(ledger.UPGRADE_HEIGHTS, {state['chain_id']: state['height'] + 1}):
        after = ledger.advance(copy.deepcopy(state), state['height'] + 1)
    members = after.get('governance', {}).get('members', {})
    print(json.dumps({'upgrade_members': sorted(k[:10] for k in members),
                      'active_owners': sorted(k[:10] for k, o in after['owners'].items() if o['status'] == 'active'),
                      'grown_units': sorted(k[:10] for k in after.get('grown_units', {})),
                      'model_root': after['model_root'][:16]}, indent=1))
    return 1 if mismatches else 0


if __name__ == '__main__':
    sys.exit(main())
