#!/usr/bin/env python3
"""Initialize or resume a non-voting full node for an agreed funded genesis.

Pins source and genesis, creates only this node's own keys, and preserves an
existing home. Becoming a validator additionally requires native stake admission.
"""
import argparse
import json
import signal
import socket
import subprocess
import sys
import time
from pathlib import Path

from neuroshard.dataflow.store import canonical
from neuroshard.demo import client as wire
from neuroshard.demo.network import edit_config, engine_path
from neuroshard.evolution.app import code_hash
from neuroshard.evolution.objects import digest


def prepare(home, genesis_path, genesis_hash, engine, peers, base_port):
    if genesis_path.stat().st_size > 4*1024*1024:
        raise ValueError('Genesis exceeds candidate bootstrap bound')
    genesis = json.loads(genesis_path.read_bytes())
    if digest(canonical(genesis)) != genesis_hash or genesis.get('initial_height') != '1':
        raise ValueError('Genesis hash or explicit initial height differs')
    manifest = genesis['app_state']['manifest']
    if (manifest['code_hash'] != code_hash() or 'auditing' not in manifest
            or not {'lifecycle', 'expert_lifecycle'}.intersection(manifest)):
        raise ValueError('Use the exact source for the agreed funded lifecycle or expert-graph genesis')
    if not 1024 <= base_port <= 65532 or not peers:
        raise ValueError('Provide an unprivileged three-port range and at least one native peer')
    identity = {'genesis_sha256':genesis_hash, 'chain_id':genesis['chain_id'], 'base_port':base_port}
    marker = home/'candidate-bootstrap.json'
    if home.exists():
        if not marker.exists() or json.loads(marker.read_bytes()) != identity:
            raise ValueError('Existing home is not this candidate deployment; it will not be overwritten')
        if digest(canonical(json.loads((home/'config/genesis.json').read_bytes()))) != genesis_hash:
            raise ValueError('Retained genesis changed')
        return identity
    home.mkdir(parents=True, mode=0o700)
    subprocess.run([str(engine), 'init', '--home',str(home)], check=True, stdout=subprocess.DEVNULL)
    path = home/'config/config.toml'
    text = path.read_text()
    edits = [('p2p','laddr',json.dumps(f'tcp://0.0.0.0:{base_port}')),
             ('p2p','persistent_peers',json.dumps(','.join(peers))),
             ('p2p','addr_book_strict','false'), ('p2p','allow_duplicate_ip','true'),
             ('rpc','laddr',json.dumps(f'tcp://127.0.0.1:{base_port+1}')),
             ('rpc','max_body_bytes','4194304'), ('rpc','timeout_broadcast_tx_commit','"120s"'),
             ('mempool','max_tx_bytes','2097152'), ('consensus','timeout_commit','"500ms"')]
    for section, key, value in edits:
        text = edit_config(text, section, key, value)
    # Root options appear before the first TOML section.
    text = edit_config(text, '', 'proxy_app', json.dumps(f'127.0.0.1:{base_port+2}'))
    text = edit_config(text, '', 'abci', '"grpc"')
    path.write_text(text)
    (path.parent/'genesis.json').write_bytes(canonical(genesis))
    marker.write_bytes(canonical(identity))
    return identity


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--home', type=Path, required=True)
    parser.add_argument('--genesis', type=Path, required=True)
    parser.add_argument('--genesis-sha256', required=True)
    parser.add_argument('--engine', type=Path, required=True)
    parser.add_argument('--peer', action='append', required=True)
    parser.add_argument('--base-port', type=int, default=55150)
    args = parser.parse_args()
    home = args.home.absolute()
    engine = Path(engine_path(str(args.engine)))
    identity = prepare(home, args.genesis, args.genesis_sha256, engine, args.peer, args.base_port)
    for port in range(args.base_port, args.base_port+3):
        with socket.socket() as probe:
            probe.settimeout(.2)
            if probe.connect_ex(('127.0.0.1', port)) == 0:
                raise ValueError('Candidate port is already in use')
    processes = []
    def stop(*_):raise KeyboardInterrupt
    signal.signal(signal.SIGTERM, stop)
    try:
        commands = [('app',[sys.executable,'-m','neuroshard.evolution.app','--home',str(home),'--port',str(args.base_port+2)]),
                    ('comet',[str(engine),'start','--home',str(home)])]
        for name, command in commands:
            with (home/(name+'.log')).open('ab') as log:
                processes.append(subprocess.Popen(command, stdout=log, stderr=log))
        print(json.dumps({'phase':'following_agreed_genesis', **identity}), flush=True)
        while True:
            if any(p.poll() is not None for p in processes):
                raise RuntimeError('Candidate process exited; inspect retained app.log and comet.log')
            try:
                status = wire.query(f'http://127.0.0.1:{args.base_port+1}')
                print(json.dumps({'phase':'following', 'height':status['height'], 'app_hash':status['app_hash'],
                                  'training_round':status['training_round']}), flush=True)
            except (OSError, wire.Rejected):
                pass
            time.sleep(5)
    except KeyboardInterrupt:
        pass
    finally:
        for process in reversed(processes):
            process.terminate()
            try:process.wait(timeout=10)
            except subprocess.TimeoutExpired:process.kill();process.wait()


if __name__ == '__main__':
    main()
