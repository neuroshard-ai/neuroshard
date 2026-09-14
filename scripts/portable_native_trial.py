"""Disposable four-validator native network for portable integration trials.

An experiment controller supplies the frozen manifest and real worker/auditor
commands. Each account uses a durable outbox; restart never signs a replacement
nonce for an unknown transaction. No existing chain home is opened or changed.
"""
import base64
import json
from pathlib import Path
import secrets
import subprocess
import sys
import time

from neuroshard.demo import protocol, client
from neuroshard.demo.network import initialize, edit_config
from neuroshard.evolution.reference_data import identity, save
from neuroshard.evolution.transactions import Outbox


class Network:
    def __init__(self, home):
        self.home = Path(home).resolve()
        self.config = json.loads((self.home / 'network.json').read_bytes())
        self.genesis = json.loads((Path(self.config['nodes'][0]['home']) / 'config/genesis.json').read_bytes())
        self.urls = [f'http://127.0.0.1:{node["rpc"]}' for node in self.config['nodes']]
        self.owners = [protocol.Identity.load_or_create(self.home / f'owner-{i}.key') for i in range(4)]
        self.outboxes = [Outbox(self.home / f'outbox-{i}.sqlite', self.urls[0], self.genesis['chain_id'], owner)
                         for i, owner in enumerate(self.owners)]
        self.processes = []

    @classmethod
    def create(cls, home, manifest, engine=None, base_port=33400):
        home = Path(home).resolve()
        if home.exists():
            raise ValueError('Use a fresh trial directory; never overwrite native signing state')
        config = initialize(home, base_port, engine)
        genesis = json.loads((Path(config['nodes'][0]['home']) / 'config/genesis.json').read_bytes())
        owners = [protocol.Identity.load_or_create(home / f'owner-{i}.key') for i in range(4)]
        entries = []
        for index, node in enumerate(config['nodes']):
            key = json.loads((Path(node['home']) / 'config/priv_validator_key.json').read_bytes())['pub_key']['value']
            entries.append({'owner': owners[index].public_key, 'consensus_key': base64.b64decode(key).hex(),
                'bond': 10 * manifest['params']['bond_unit'], 'liquid': 10_000_000_000})
            genesis['validators'][index]['power'] = '10'
        genesis.update(chain_id='neuroshard-portable-life-' + secrets.token_hex(6),
                       app_state={'manifest': manifest, 'validators': entries})
        native = manifest['native_consensus']
        genesis['consensus_params']['block']['max_bytes'] = str(native['block_max_bytes'])
        genesis['consensus_params']['evidence'].update(
            max_age_num_blocks=str(manifest['params']['evidence_blocks']),
            max_age_duration=str(manifest['params']['evidence_seconds'] * 10**9))
        for node in config['nodes']:
            directory = Path(node['home']) / 'config'
            text = (directory / 'config.toml').read_text()
            text = edit_config(text, 'rpc', 'max_body_bytes', '4194304')
            text = edit_config(text, 'rpc', 'timeout_broadcast_tx_commit', '"120s"')
            (directory / 'config.toml').write_text(text)
            save(directory / 'genesis.json', genesis)
        save(home / 'commitments.json', {'genesis': identity(genesis), 'manifest': identity(manifest),
                                        'chain_id': genesis['chain_id']})
        return cls(home)

    def start(self):
        for index, node in enumerate(self.config['nodes']):
            for name, command in (
                ('app', [sys.executable, '-m', 'neuroshard.evolution.app', '--home', node['home'], '--port', str(node['abci'])]),
                ('node', [self.config['engine'], 'start', '--home', node['home']]),
            ):
                with (self.home / f'{name}-{index}.log').open('ab') as log:
                    process = subprocess.Popen(command, stdout=log, stderr=log)
                self.processes.append(process)
        save(self.home / 'processes.json', {'pids': [p.pid for p in self.processes]})
        self.until(lambda: all(client.query(url)['height'] > 1 for url in self.urls), 120)
        return self

    @staticmethod
    def until(check, seconds=180):
        deadline = time.monotonic() + seconds
        while time.monotonic() < deadline:
            try:
                value = check()
                if value:
                    return value
            except (OSError, ValueError, KeyError):
                pass
            time.sleep(.5)
        raise TimeoutError('Native trial condition did not complete')

    def query(self, path='/status', data=None):
        return client.query(self.urls[0], path, data)

    def send(self, owner, operation, kind, **fields):
        return self.outboxes[owner].send(operation, kind, timeout=180, **fields)

    def fund(self, operation, stages, sponsor=3, publisher=None):
        self.send(sponsor, operation + '/fund', 'fund_audit',
                  publisher=publisher or self.owners[sponsor].public_key,
                  auditors=[], stage_limit=stages, expires_in=100000)
        budget = self.outboxes[sponsor].logical_id(operation + '/fund')
        # Automatic auditors exclusively own their transaction streams. Having
        # the controller also sign acceptance races the same account nonce.
        self.until(lambda: all(self.query('/auditing')['budgets'][budget]['auditors'][owner.public_key]['bond']
                               for owner in self.owners[:3]))
        return budget

    def activate(self, job, publisher=3):
        self.send(publisher, 'job/propose', 'propose_shard_job', job=job)
        proposal = self.outboxes[publisher].logical_id('job/propose')
        for index in range(3):
            self.send(index, 'job/vote', 'vote_shard_job', proposal_id=proposal, approve=True)
        self.until(lambda: (active if (active := self.query('/portable_lifecycle')['active'])
                            and active['id'] == proposal else None))
        return proposal

    def settled(self, claim_id, seconds=7200):
        def finished():
            status = self.query()
            row = next((r for r in status['settled'] if r['id'] == claim_id), None)
            return {'settlement': row, 'status': status} if row else None
        return self.until(finished, seconds)

    def close(self):
        for process in reversed(self.processes):
            if process.poll() is None:
                process.terminate()
        for process in self.processes:
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=10)
        for outbox in self.outboxes:
            outbox.close()
