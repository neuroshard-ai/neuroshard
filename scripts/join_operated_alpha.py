#!/usr/bin/env python3
"""Run a validating observer for a reviewed operated-alpha descriptor.

Install the pinned CPU requirements first. This downloads no language-model
weights, keeps RPC on loopback, and never bonds or spends the customer's tokens.
"""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
from urllib.parse import urlsplit

import requests

from neuroshard.demo import client
from neuroshard.demo.network import edit_config
from neuroshard.evolution.app import code_hash
from neuroshard.evolution.reference_data import identity, save


def download(url, maximum):
    if urlsplit(url).scheme != 'https':
        raise ValueError('Public bootstrap downloads require HTTPS')
    with requests.get(url, stream=True, timeout=(10, 30), allow_redirects=False) as response:
        response.raise_for_status()
        if response.status_code != 200:
            raise ValueError('The bootstrap URL must return its committed object directly')
        data = bytearray()
        for chunk in response.iter_content(1024*1024):
            data.extend(chunk)
            if len(data) > maximum:
                raise ValueError('The bootstrap download exceeds its declared bound')
        return bytes(data)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--descriptor', required=True, help='Reviewed local access.json file or its public HTTPS URL')
    parser.add_argument('--home', type=Path, required=True)
    parser.add_argument('--base-port', type=int, default=26656)
    args = parser.parse_args()
    if not 1024 <= args.base_port <= 65532:
        parser.error('Choose a base port between 1024 and 65532')
    raw = (download(args.descriptor, 1024**2) if args.descriptor.startswith('https://')
           else Path(args.descriptor).read_bytes())
    if len(raw) > 1024**2:
        raise ValueError('Bound the access descriptor')
    descriptor = json.loads(raw)
    if (descriptor['format'] != 'neuroshard-operated-alpha-access-v1'
            or descriptor['code_hash'] != code_hash()):
        raise ValueError('Install the descriptor\'s matching reviewed source before following this chain')
    home = args.home.expanduser().resolve()
    if home.exists() and not (home/'alpha-node.json').exists():
        raise ValueError('Use a fresh node home; this path has not been initialized for this alpha')
    home.mkdir(parents=True, exist_ok=True, mode=0o700)
    with (home/'node.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        metadata = {'chain_id': descriptor['chain_id'], 'manifest_root': descriptor['manifest_root'],
                    'genesis_sha256': descriptor['genesis_sha256'], 'base_port': args.base_port}
        marker = home/'alpha-node.json'
        if marker.exists() and json.loads(marker.read_bytes()) != metadata:
            raise ValueError('Existing keys and state belong to another node configuration')
        engine = home/'cometbft'
        if not engine.exists():
            binary = download(descriptor['engine_url'], 128*1024**2)
            if hashlib.sha256(binary).hexdigest() != descriptor['engine_sha256']:
                raise ValueError('Consensus executable checksum mismatch')
            engine.write_bytes(binary)
            engine.chmod(0o700)
        if hashlib.sha256(engine.read_bytes()).hexdigest() != descriptor['engine_sha256']:
            raise ValueError('Installed consensus executable differs from the reviewed descriptor')
        if not marker.exists():
            genesis_bytes = download(descriptor['genesis_url'], 4*1024**2)
            if hashlib.sha256(genesis_bytes).hexdigest() != descriptor['genesis_sha256']:
                raise ValueError('Genesis checksum mismatch')
            genesis = json.loads(genesis_bytes)
            if (genesis['chain_id'] != descriptor['chain_id']
                    or identity(genesis['app_state']['manifest']) != descriptor['manifest_root']):
                raise ValueError('The genesis changed the reviewed network')
            subprocess.run([str(engine), 'init', '--home', str(home/'native')], check=True, capture_output=True)
            config = home/'native/config/config.toml'
            text = config.read_text()
            peers = descriptor['peers']
            if (not isinstance(peers, list) or not 1 <= len(peers) <= 32
                    or any(not isinstance(p, str) or len(p) > 256 for p in peers)):
                raise ValueError('Require a bounded peer list from the reviewed descriptor')
            for section, key, value in [
                ('', 'proxy_app', json.dumps('127.0.0.1:'+str(args.base_port+2))), ('', 'abci', '"grpc"'),
                ('rpc', 'laddr', json.dumps('tcp://127.0.0.1:'+str(args.base_port+1))),
                ('rpc', 'max_body_bytes', '4194304'), ('rpc', 'timeout_broadcast_tx_commit', '"120s"'),
                ('p2p', 'laddr', json.dumps('tcp://0.0.0.0:'+str(args.base_port))),
                ('p2p', 'persistent_peers', json.dumps(','.join(peers)))]:
                text = edit_config(text, section, key, value)
            config.write_text(text)
            (home/'native/config/genesis.json').write_bytes(genesis_bytes)
            save(marker, metadata)
        hosting = {'node_rpc': 'http://127.0.0.1:'+str(args.base_port+1),
                   'chain_id': descriptor['chain_id'], 'manifest_root': descriptor['manifest_root']}
        save(home/'hosting.json', hosting)
        stop = False
        def interrupt(*_args):
            nonlocal stop
            stop = True
        signal.signal(signal.SIGTERM, interrupt)
        signal.signal(signal.SIGINT, interrupt)
        environment = {**os.environ, 'ATEN_CPU_CAPABILITY': 'default', 'MKL_ENABLE_INSTRUCTIONS': 'SSE4_2',
                       'OMP_NUM_THREADS': '1', 'OPENBLAS_NUM_THREADS': '1', 'MKL_NUM_THREADS': '1'}
        processes = []
        try:
            for label, argv in [('app', [sys.executable, '-m', 'neuroshard.evolution.app', '--home',
                    str(home/'native'), '--port', str(args.base_port+2)]),
                    ('node', [str(engine), 'start', '--home', str(home/'native')])]:
                with (home/(label+'.log')).open('ab') as log:
                    processes.append(subprocess.Popen(argv, stdout=log, stderr=log, env=environment))
            print('Following the native ledger from genesis. RPC stays on this machine.', flush=True)
            print('Chat configuration: '+str(home/'hosting.json'), flush=True)
            while not stop:
                if any(p.poll() is not None for p in processes):
                    raise RuntimeError('A node process stopped; inspect app.log and node.log in the node home')
                try:
                    status = client.rpc(hosting['node_rpc'], 'status')
                    sync = status['sync_info']
                    print('Block '+sync['latest_block_height']+' · '+('syncing' if sync['catching_up'] else 'ready'), flush=True)
                except (OSError, ValueError):
                    print('Waiting for native peers…', flush=True)
                for _ in range(15):
                    if stop:
                        break
                    time.sleep(1)
        finally:
            for process in reversed(processes):
                if process.poll() is None:
                    process.terminate()
            for process in processes:
                try:
                    process.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=5)


if __name__ == '__main__':
    main()
