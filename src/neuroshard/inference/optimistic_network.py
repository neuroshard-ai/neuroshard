"""Launch a local settlement chain: CometBFT validators, each running the optimistic serving application."""

import base64
import hashlib
import json
import os
import re
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path
from urllib.request import Request, urlopen

from neuroshard.inference import optimistic as ledger

REPO = Path(__file__).resolve().parents[3]
COMET_VERSION = '0.38.26'


class Rejected(Exception):
    """A transaction or query the chain refused."""


def engine_path(explicit=None):
    candidate = explicit or os.environ.get('COMETBFT_BINARY')
    if not candidate:
        local = REPO / '.neuroshard' / 'tools' / 'cometbft'
        candidate = str(local) if local.exists() else shutil.which('cometbft')
    if not candidate:
        raise ValueError('Install CometBFT v0.38.26 using scripts/install_demo_consensus.sh')
    candidate = str(Path(candidate).resolve())
    version = subprocess.check_output([candidate, 'version'], text=True).strip()
    if version != COMET_VERSION:
        raise ValueError(f'Expected CometBFT {COMET_VERSION}, found {version}')
    return candidate


def edit_config(text, section, key, value):
    current, found, lines = '', False, []
    for line in text.splitlines():
        if re.match(r'^\[.*\]$', line):
            current = line[1:-1]
        if current == section and re.match(rf'^{re.escape(key)}\s*=', line):
            line, found = f'{key} = {value}', True
        lines.append(line)
    if not found:
        raise ValueError(f'Missing upstream configuration option [{section}] {key}')
    return '\n'.join(lines) + '\n'


def genesis_document(template, terms, identities, genesis_time):
    """A validator set's genesis: CometBFT defaults from ``template``, equal power, and the ledger terms as app state."""
    genesis = json.loads(json.dumps(template))
    genesis.update(chain_id=terms['chain_id'], genesis_time=genesis_time, initial_height='1', app_hash='',
                   app_state={k: terms[k] for k in ('model_root', 'shards', 'allocations', 'params')},
                   validators=[{'address': identity['address'], 'pub_key': identity['pub_key'], 'power': '1',
                                'name': f'validator-{i + 1}'} for i, identity in enumerate(identities)])
    genesis['consensus_params']['block']['max_bytes'] = '1048576'
    return genesis


def configure_node(node_home, genesis, settlement, peers, ports, block_seconds=1.0, p2p_host='127.0.0.1'):
    """One validator's CometBFT configuration, genesis and settlement node configuration."""
    node_home = Path(node_home)
    path = node_home / 'config/config.toml'
    text = path.read_text()
    for section, key, value in (('', 'proxy_app', json.dumps(f'127.0.0.1:{ports["abci"]}')), ('', 'abci', '"grpc"'),
                                ('', 'log_level', '"info"'), ('rpc', 'laddr', json.dumps(f'tcp://127.0.0.1:{ports["rpc"]}')),
                                # Admitting a challenge replays its proof; the RPC write timeout derives from this value.
                                ('rpc', 'timeout_broadcast_tx_commit', '"120s"'),
                                ('p2p', 'laddr', json.dumps(f'tcp://{p2p_host}:{ports["p2p"]}')),
                                ('p2p', 'persistent_peers', json.dumps(peers)), ('p2p', 'allow_duplicate_ip', 'true'),
                                ('p2p', 'addr_book_strict', 'false'),
                                ('consensus', 'timeout_commit', f'"{int(block_seconds * 1000)}ms"'),
                                ('consensus', 'timeout_propose', '"3s"'), ('consensus', 'create_empty_blocks', 'true')):
        text = edit_config(text, section, key, value)
    path.write_text(text)
    (path.parent / 'genesis.json').write_text(json.dumps(genesis, sort_keys=True))
    (node_home / 'settlement.json').write_text(json.dumps(settlement, sort_keys=True))


def identity(binary, node_home):
    """A validator's node ID, consensus public key and address, from its own initialized home."""
    node_home = Path(node_home)
    key = json.loads((node_home / 'config/priv_validator_key.json').read_text())
    return {'node_id': subprocess.check_output([binary, 'show-node-id', '--home', str(node_home)], text=True).strip(),
            'pub_key': key['pub_key'], 'address': key['address']}


def initialize(home, terms, nodes, base_port=28650, engine=None, block_seconds=1.0):
    """``terms`` are the ledger genesis terms; ``nodes`` holds one settlement node configuration per validator."""
    home = Path(home).resolve()
    if home.exists() and any(home.iterdir()):
        raise ValueError(f'Use an empty home directory: {home}')
    binary = engine_path(engine)
    count = len(nodes)
    home.mkdir(parents=True, exist_ok=True)
    subprocess.run([binary, 'testnet', '--v', str(count), '--o', str(home / 'chain'), '--home', str(home / 'initializer'),
                    '--populate-persistent-peers=false'], check=True, stdout=subprocess.DEVNULL)
    validators, identities = [], []
    for i in range(count):
        node_home = home / 'chain' / f'node{i}'
        identities.append(identity(binary, node_home))
        validators.append({'home': str(node_home), 'id': identities[-1]['node_id'], 'p2p': base_port + i * 10,
                           'rpc': base_port + i * 10 + 1, 'abci': base_port + i * 10 + 2})
    template = json.loads((Path(validators[0]['home']) / 'config/genesis.json').read_text())
    genesis = genesis_document(template, terms, identities, template['genesis_time'])
    for i, node in enumerate(validators):
        peers = ','.join(f'{other["id"]}@127.0.0.1:{other["p2p"]}' for j, other in enumerate(validators) if i != j)
        configure_node(node['home'], genesis, nodes[i], peers, node, block_seconds)
    config = {'home': str(home), 'engine': binary, 'chain_id': terms['chain_id'], 'validators': validators}
    (home / 'network.json').write_text(json.dumps(config, sort_keys=True))
    return config


def read_pids(config):
    path = Path(config['home']) / 'processes.json'
    return json.loads(path.read_text()) if path.exists() else {}


def launch(config, name, command):
    log_path = Path(config['home']) / 'logs'
    log_path.mkdir(exist_ok=True)
    env = {**os.environ, 'PYTHONPATH': str(REPO / 'src') + os.pathsep + os.environ.get('PYTHONPATH', '')}
    with (log_path / f'{name}.log').open('ab') as log:
        process = subprocess.Popen(command, stdout=log, stderr=log, env=env, start_new_session=True)
    pids = read_pids(config)
    pids[name] = process.pid
    (Path(config['home']) / 'processes.json').write_text(json.dumps(pids))


def start(config, timeout=300):
    for i, node in enumerate(config['validators']):
        launch(config, f'app{i}', [sys.executable, '-m', 'neuroshard.inference.optimistic_app', '--home', node['home'],
                                    '--port', str(node['abci'])])
    for i, node in enumerate(config['validators']):
        launch(config, f'node{i}', [config['engine'], 'start', '--home', node['home']])
    return wait_height(config, 1, timeout)


def stop(config):
    for name, pid in read_pids(config).items():
        try:
            os.killpg(pid, signal.SIGTERM)
        except ProcessLookupError:
            continue
    deadline = time.monotonic() + 10
    for pid in read_pids(config).values():
        while time.monotonic() < deadline:
            try:
                os.kill(pid, 0)
                time.sleep(0.1)
            except ProcessLookupError:
                break
        else:
            try:
                os.killpg(pid, signal.SIGKILL)
            except ProcessLookupError:
                pass


def url(config, index=0):
    return f'http://127.0.0.1:{config["validators"][index]["rpc"]}'


def rpc(address, method, params=None, timeout=60):
    request = Request(address, data=json.dumps({'jsonrpc': '2.0', 'id': 1, 'method': method, 'params': params or {}}).encode(),
                      headers={'Content-Type': 'application/json'})
    with urlopen(request, timeout=timeout) as response:
        value = json.loads(response.read())
    if 'error' in value:
        raise Rejected(json.dumps(value['error']))
    return value['result']


def broadcast(address, envelope):
    """Submit a transaction; returns mempool admission as {'code', 'log', 'hash'}."""
    result = rpc(address, 'broadcast_tx_sync', {'tx': base64.b64encode(ledger.canonical(envelope)).decode()}, timeout=300)
    return {'code': result.get('code', 0), 'log': result.get('log', ''), 'hash': result.get('hash')}


def state(address):
    response = rpc(address, 'abci_query', {'path': '/state', 'prove': False})['response']
    if response.get('code', 0):
        raise Rejected(response.get('log', 'Query rejected'))
    return json.loads(base64.b64decode(response['value']))


def wait_height(config, height, timeout=300):
    """Wait until every validator has committed ``height``; returns their states at their current heights."""
    deadline, last = time.monotonic() + timeout, None
    while time.monotonic() < deadline:
        try:
            states = [state(url(config, i)) for i in range(len(config['validators']))]
            if all(s['height'] >= height for s in states):
                return states
        except Exception as error:
            last = error
        time.sleep(0.5)
    raise RuntimeError(f'Settlement chain did not reach height {height}; see {config["home"]}/logs: {last}')


def wait_included(config, envelope, timeout=120):
    """Wait until a transaction is in a committed block on every validator."""
    digest = hashlib.sha256(ledger.canonical(envelope)).digest()
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            result = rpc(url(config), 'tx', {'hash': base64.b64encode(digest).decode()})
            height = int(result['height'])
            if result['tx_result'].get('code', 0):
                raise Rejected(result['tx_result'].get('log', 'Transaction failed'))
            wait_height(config, height, timeout=max(1, deadline - time.monotonic()))
            return height
        except Rejected as error:
            if 'not found' not in str(error):
                raise
        time.sleep(0.5)
    raise RuntimeError(f'Transaction {ledger.transaction_id(envelope)} was not committed')
