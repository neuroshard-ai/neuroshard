#!/usr/bin/env python3
"""Export stopped trial stores and replay the provider ledger without any key."""
import argparse
import base64
from concurrent.futures import ThreadPoolExecutor
import gzip
import json
from pathlib import Path
import sqlite3
import subprocess
import tempfile
import time

from google.protobuf.timestamp_pb2 import Timestamp
from neuroshard.dataflow.store import canonical
from neuroshard.demo import client, protocol
from neuroshard.evolution import settlement
from neuroshard.evolution.app import ERRORS, code_hash
from neuroshard.evolution.objects import Objects
from neuroshard.evolution.reference_data import identity, save


def export(home, engine, port):
    native = home/'native'
    config = json.loads((native/'network.json').read_bytes())
    states = {}
    for index, node in enumerate(config['nodes']):
        database = Path(node['home'])/'evolution.sqlite'
        with sqlite3.connect(database.as_uri()+'?mode=ro', uri=True) as connection:
            states[str(index)] = json.loads(connection.execute('SELECT value FROM state WHERE id=1').fetchone()[0])
    owner = max(states, key=lambda key: states[key]['height'])
    node = Path(config['nodes'][int(owner)]['home'])
    through = states[owner]['height']
    target = home/'ledger'
    target.mkdir(exist_ok=True)
    genesis = json.loads((node/'config/genesis.json').read_bytes())
    save(target/'genesis.json', genesis)
    save(target/'validator-states.json', states)
    url = f'http://127.0.0.1:{port}'
    with (home/'inspect.log').open('ab') as log:
        inspector = subprocess.Popen([engine, 'inspect', '--home', str(node), '--rpc.laddr',
            f'tcp://127.0.0.1:{port}', '--log_level', 'error'], stdout=log, stderr=log)
    try:
        deadline = time.monotonic() + 30
        while time.monotonic() < deadline:
            if inspector.poll() is not None:
                raise RuntimeError('Read-only block inspection stopped')
            try:
                client.rpc(url, 'block', {'height': '1'})
                break
            except (OSError, ValueError):
                time.sleep(.1)
        else:
            raise TimeoutError('Read-only block inspection did not become available')
        with ThreadPoolExecutor(max_workers=4) as pool:
            records = list(pool.map(lambda height: client.rpc(url, 'block', {'height': str(height)}),
                                   range(1, through+1)))
        raw = b''.join(canonical(record)+b'\n' for record in records)
        (target/'blocks.jsonl.gz').write_bytes(gzip.compress(raw, mtime=0))
        save(target/'export.json', {'genesis': identity(genesis), 'through': through,
                                  'validator_heights': {key: value['height'] for key, value in states.items()}})
    finally:
        inspector.terminate()
        try:
            inspector.wait(timeout=10)
        except subprocess.TimeoutExpired:
            inspector.kill()
            inspector.wait(timeout=5)


def replay(home):
    target = home/'ledger'
    genesis = json.loads((target/'genesis.json').read_bytes())
    expected = json.loads((target/'validator-states.json').read_bytes())
    spec = genesis['app_state']
    if spec['manifest']['code_hash'] != code_hash():
        raise ValueError('Use the exact published provider preflight source revision')
    state = settlement.genesis(genesis['chain_id'], spec['validators'], spec['manifest'])
    accepted = rejected = headers = 0
    matched, previous_id = [], None
    with tempfile.TemporaryDirectory(prefix='neuroshard-provider-replay-') as temporary:
        store = Objects(Path(temporary))
        with gzip.open(target/'blocks.jsonl.gz', 'rb') as source:
            for raw in source:
                record = protocol.parse_json(raw)
                block, header = record['block'], record['block']['header']
                height = int(header['height'])
                if (height != headers+1 or header['chain_id'] != genesis['chain_id']
                        or header['app_hash'].lower() != identity(state)
                        or previous_id is not None and header['last_block_id'] != previous_id):
                    raise ValueError('Native header or application history changed')
                if block['evidence']['evidence']:
                    raise ValueError('This fixture export did not declare Byzantine evidence events')
                timestamp = Timestamp()
                timestamp.FromJsonString(header['time'])
                committers = {row['validator_address'] for row in
                    (block.get('last_commit') or {}).get('signatures', []) if row['block_id_flag'] == 2}
                state, _ = settlement.advance(state, height, timestamp.seconds*10**9+timestamp.nanos,
                                              committers=committers)
                for encoded in block['data']['txs'] or []:
                    try:
                        state = settlement.transition(state, protocol.parse_json(base64.b64decode(encoded)), store, True)
                        accepted += 1
                    except ERRORS:
                        rejected += 1
                settlement.invariant(state)
                for index, saved in expected.items():
                    if saved['height'] == height:
                        if state != saved:
                            raise ValueError('Replayed state differs from a saved validator')
                        matched.append(index)
                previous_id, headers = record['block_id'], height
    if set(matched) != set(expected):
        raise ValueError('Not every saved validator state was reached')
    report = {'passed': True, 'headers': headers, 'accepted_transactions': accepted,
        'rejected_transactions': rejected, 'matched_validators': sorted(matched),
        'final_state_root': identity(state), 'issued_atoms': state['issued'],
        'settled_claims': len(state['settled']), 'genesis': identity(genesis),
        'scope': 'Application replay against stored validator states; separate numerical reports cover neural work. '
                 'This does not prove independent administration or independently validate CometBFT commit signatures.'}
    save(home/'ledger-replay.json', report)
    print(json.dumps(report))
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--home', type=Path, required=True)
    parser.add_argument('--export', action='store_true', help='Read this stopped trial, then replay its public export')
    parser.add_argument('--engine')
    parser.add_argument('--port', type=int, default=28699)
    args = parser.parse_args()
    if args.export:
        if not args.engine:
            parser.error('--export requires --engine')
        export(args.home.resolve(), args.engine, args.port)
    replay(args.home.resolve())
