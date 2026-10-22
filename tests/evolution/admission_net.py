"""A network for admission tests: the real ledger transition, real owners' admission and serving handshakes, no model.

Blocks are produced when the shared fake clock advances (``Net.sleep``), so the client's backoff, the owners' lease
bounds and funding confirmation move together, deterministically.
"""
import base64
import copy
import hashlib
import json
import socket
import threading
import time
from types import SimpleNamespace

from neuroshard.assistant import admission, admission_client, network
from neuroshard.assistant.admission_messages import Replays
from neuroshard.inference import optimistic as ledger
from neuroshard.inference import optimistic_network as chain_network
from neuroshard.inference import relay

NEURO = network.NEURO
PARAMS = {**ledger.PARAMS, 'job_blocks': 120, 'max_jobs': 32}
SERVED = SimpleNamespace(world=3, root='ab' * 32, max_tokens=8, name='tiny')
WORK = relay.CONTROL  # the first frame of real serving is a command


class Clock:
    def __init__(self, start=1_800_000_000.0):
        self.now = start

    def __call__(self):
        return self.now


class Ledger:
    """``network.Chain``'s surface over an in-memory state; a block per ``mine``."""

    def __init__(self, state):
        self.s, self.chain_id, self.rpc = state, state['chain_id'], 'memory'
        self.mempool, self.receipts, self.blocks = [], {}, {}
        self.paused, self.unindexed = False, set()

    def state(self):
        return copy.deepcopy(self.s)

    def nonce(self, account):
        return self.s['accounts'].get(account.public, {'nonce': 0})['nonce']

    def broadcast(self, envelope):
        raw = ledger.canonical(envelope)
        key = hashlib.sha256(raw).hexdigest().upper()
        if key in self.receipts or raw in self.mempool:
            raise chain_network.Rejected('{"data": "tx already exists in cache"}')
        try:
            ledger.admit(self.s, envelope)
        except ValueError as error:
            return {'code': 1, 'log': str(error), 'hash': key}
        self.mempool.append(raw)
        return {'code': 0, 'log': '', 'hash': key}

    def mine(self):
        if self.paused:
            return
        height = self.s['height'] + 1
        state, txs, results = ledger.advance(self.s, height), self.mempool, []
        for raw in txs:
            try:
                state = ledger.transition(state, json.loads(raw), lambda *_: False)
                results.append({'code': 0})
            except ValueError as error:
                results.append({'code': 1, 'log': str(error)})
            self.receipts[hashlib.sha256(raw).hexdigest().upper()] = (height, results[-1])
        ledger.invariant(state)
        self.s, self.mempool, self.blocks[height] = state, [], (txs, results)

    def call(self, address, method, params):
        if method == 'tx':
            key = base64.b64decode(params['hash']).hex().upper()
            if key not in self.receipts or key in self.unindexed:
                raise chain_network.Rejected('{"data": "tx (%s) not found"}' % key)
            height, result = self.receipts[key]
            return {'height': str(height), 'tx_result': result}
        txs, results = self.blocks.get(int(params['height']), ([], []))
        if method == 'block':
            return {'block': {'data': {'txs': [base64.b64encode(raw).decode() for raw in txs]}}}
        return {'txs_results': results}


class Owner(network.Owner):
    """The real admission control and serving handshake; ``work`` holds the slot until the user hangs up."""

    def __init__(self, chain, rank, home, clock):
        self.served, self.rank, self.chain, self.home = SERVED, rank, chain, home
        self.log_key = network.signing_key(home / 'log.key')  # reloaded on restart, as a real owner's is
        self.public = network.public_hex(self.log_key)
        self.config, self.serving_lock = SimpleNamespace(hidden_size=4), threading.Lock()
        self.served_jobs, self.pending_jobs, self.worked, self.threads = set(), set(), [], 1
        self.gate = admission.Gate(rank, self.public, home, network.OwnerView(self), clock)
        self.served_jobs |= self.gate.started
        self.replays, self.ticked = Replays(), 0.0

    def work(self, connection, job_id, job, progress, granite_audit, serving):
        self.worked.append(job_id)
        connection.settimeout(10)
        try:
            while connection.recv(4096):
                pass
        except OSError:
            pass
        return {'job_id': job_id, 'positions': 0}


class Net:
    def __init__(self, tmp_path, users=('alice', 'bob'), owners_per_shard=1, balance=20 * NEURO):
        from neuroshard.assistant.network import Account

        self.tmp, self.clock, self.hooks = tmp_path, Clock(), []
        self.accounts = {name: Account(tmp_path / name / 'account.key') for name in users}
        state = ledger.genesis('admission-test', SERVED.root, SERVED.world,
                               {a.public: balance for a in self.accounts.values()}, PARAMS)
        self.chain = Ledger(state)
        self.owners, self.endpoints = [], {}
        for shard in range(1, SERVED.world):
            for index in range(owners_per_shard):
                self.add_owner(shard, index)
        self.chain.mine()
        self.threads = []

    def add_owner(self, shard, index):
        owner = Owner(self.chain, shard, self.tmp / f'owner-{shard}-{index}', self.clock)
        endpoint = f'owner-{shard}-{index}.test:1'
        bond = PARAMS['owner_bond_minimum']
        self.chain.s['owners'][owner.public] = {'account': owner.public[:2] + '00' * 32, 'shard': shard, 'amount': bond,
                                                'status': 'active', 'release_height': None, 'endpoint': endpoint}
        self.chain.s['initial_supply'] += bond
        self.owners.append(owner)
        self.endpoints[endpoint] = owner
        return owner

    def owner(self, shard, index=0):
        return [o for o in self.owners if o.rank == shard][index]

    def sleep(self, seconds):
        self.clock.now += seconds
        for _ in range(max(1, int(seconds))):
            self.chain.mine()
        for owner in self.owners:
            owner.gate.tick()
        for hook in list(self.hooks):
            hook()

    def request(self, address):
        def send(endpoint, message, timeout=10):
            owner = self.endpoints[endpoint]
            mine, theirs = socket.socketpair()
            with mine:
                owner.control(theirs, json.loads(ledger.canonical(message)), address)
                return json.loads(relay.receive_frame(mine, relay.ADMISSION_REPLY, 1 << 16))
        return send

    def dial(self, address):
        def connect(endpoint, body, session_key, timeout=10, admission=None, accept_ready=None):
            owner = self.endpoints[endpoint]
            mine, theirs = socket.socketpair()
            mine.settimeout(timeout)
            thread = threading.Thread(target=owner.serve, args=(theirs, lambda *_: None, address), daemon=True)
            thread.start()
            self.threads.append(thread)
            try:
                return relay.handshake(mine, body, session_key, admission, accept_ready)
            except BaseException:
                mine.close()
                raise
        return connect

    def admitter(self, name, address='10.0.0.1', **kwargs):
        index = __import__('neuroshard.assistant.funding', fromlist=['LedgerIndex']).LedgerIndex(self.chain,
                                                                                                 self.chain.call)
        return admission_client.Admitter(self.chain, self.accounts[name], SERVED, self.tmp / name, NEURO,
                                         clock=self.clock, sleep=self.sleep, wall=self.clock,
                                         request=self.request(address), dial=self.dial(address), index=index,
                                         progress=lambda *_: None, **kwargs)

    def rehello(self, job_id, rank, session_key, address='10.0.0.1', owner=None):
        """A reserved client's HELLO for a funded job: a fresh session-signed attempt naming the live lease
        (re-reserving the job first when this owner holds no lease for it)."""
        from neuroshard.assistant import admission_messages as messages

        owner = owner or self.owner(rank)
        endpoint = next(e for e, o in self.endpoints.items() if o is owner)
        lease = owner.gate.lease
        if lease is not None and lease.get('job_id') == job_id:
            reservation = lease['reservation']
        else:
            reserve = messages.by_key(session_key, 'reserve', self.chain.chain_id, self.clock, rank=rank,
                                      owner=owner.public, job_id=job_id)
            reply = self.request(address)(endpoint, reserve)
            if reply['body']['kind'] != 'offer':
                raise relay.OwnerBusy(reply)
            reservation = reply['body']['reservation']
        attempt = self.admitter(next(iter(self.accounts)), address).attempt(session_key, job_id, rank, owner.public,
                                                                            reservation)
        body = relay.hello(self.chain.chain_id, job_id, rank, SERVED.world, 4, 8)
        return self.dial(address)(endpoint, body, session_key, **attempt)

    def turn(self, name, address='10.0.0.1', wait=600, admitter=None):
        """Open one job for ``name``: (job_id, owners, sockets, session_key)."""
        from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

        session_key = Ed25519PrivateKey.generate()
        request_root = ledger.digest({'turn': name, 'at': self.clock.now})
        hello = lambda job_id, rank: relay.hello(self.chain.chain_id, job_id, rank, SERVED.world, 4, 8)  # noqa: E731
        job_id, owners, sockets = (admitter or self.admitter(name, address)).open(
            request_root, session_key, 64, hello, wait)
        return job_id, owners, sockets, session_key


def start_work(sockets):
    for sock in sockets.values():
        relay.send_frame(sock, WORK, b'\0' * 16)


def wait_until(predicate, timeout=5):
    deadline = time.monotonic() + timeout
    while not predicate():
        if time.monotonic() > deadline:
            raise AssertionError('condition not reached')
        time.sleep(0.01)


def jobs_of(net, name):
    public = net.accounts[name].public
    state = net.chain.state()
    return [j for j, job in state['jobs'].items() if job['user'] == public] + \
        [j for j, result in state['results'].items() if result.get('user') == public]
