#!/usr/bin/env python3
"""Export a live alpha's public ledger without stopping consensus or copying keys.

Blocks are streamed in bounded batches. Saved application states are consistent
SQLite reads, and replay must reproduce each of the four recorded heights.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import gzip
import json
from pathlib import Path
import time

from neuroshard.client import wire
from neuroshard.demo import client
from neuroshard.evolution.reference_data import identity, save
from operate_alpha import connect
from operated_alpha_hosts import LedgerHosts
from ordinary_cloud import REMOTE
from replay_provider_preflight import replay


def capture(home, destination):
    if destination.exists():
        raise ValueError('Use a new public snapshot directory')
    destination.mkdir(parents=True)
    network = connect(home, rpc_offset=200)
    ledger = destination/'ledger'
    ledger.mkdir()
    try:
        hosts = LedgerHosts(home/'ledger-hosts')
        code = ('import sqlite3,pathlib,sys; p=pathlib.Path(sys.argv[1]); '
                'db=sqlite3.connect(p.as_uri()+"?mode=ro",uri=True); '
                'sys.stdout.buffer.write(db.execute("SELECT value FROM state WHERE id=1").fetchone()[0])')
        def state(index):
            return json.loads(hosts.command(index, ['python3', '-c', code,
                                                    REMOTE+'/native/evolution.sqlite']).stdout)
        with ThreadPoolExecutor(max_workers=4) as pool:
            states = {str(index): value for index, value in enumerate(pool.map(state, range(4)))}
        if any(value['chain_id'] != network.genesis['chain_id']
               or value['manifest'] != network.genesis['app_state']['manifest'] for value in states.values()):
            raise ValueError('Snapshot states changed their pinned network')
        through = max(value['height'] for value in states.values())
        network.until(lambda: network.query()['height'] >= through, 60)
        save(ledger/'genesis.json', network.genesis)
        save(ledger/'validator-states.json', states)
        temporary = ledger/'blocks.jsonl.gz.pending'
        with temporary.open('wb') as raw, gzip.GzipFile(fileobj=raw, mode='wb', mtime=0) as stream:
            with ThreadPoolExecutor(max_workers=4) as pool:
                for start in range(1, through+1, 16):
                    records = pool.map(lambda height: client.rpc(network.urls[0], 'block', {'height': str(height)}),
                                       range(start, min(start+16, through+1)))
                    for record in records:
                        stream.write(wire.canonical(record)+b'\n')
        temporary.replace(ledger/'blocks.jsonl.gz')
        save(ledger/'export.json', {'genesis': identity(network.genesis), 'through': through,
             'validator_heights': {key: value['height'] for key, value in states.items()},
             'captured_at': time.time(), 'live_consensus_unchanged': True})
    finally:
        network.close()
    return replay(destination)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--home', type=Path, required=True)
    parser.add_argument('--destination', type=Path, required=True)
    args = parser.parse_args()
    capture(args.home.resolve(), args.destination.resolve())
