"""The public assistant network: owners host shards, users chat through them, the ledger pays.

A network descriptor names the chain, the seed's RPC and faucet, and the served version: the
verified serving plan with its partition, module and gate. An owner fetches only its shard,
bonds, publishes its endpoint and serves the jobs that name it, committing each signed log.
A user's device runs stage 0, opens a paid job naming one active owner per later shard, and
relays the conversation's messages between them. Tools and the workspace stay on the user's
device; owners see only intermediate activations, and the ledger only digests.
"""

import base64
import hashlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import os
from pathlib import Path, PurePosixPath
import random
import secrets
import socket
import tempfile
import threading
import time
import urllib.error
import urllib.request

from neuroshard.inference import optimistic as ledger
from neuroshard.inference import optimistic_network as chain_network
from neuroshard.inference import relay

DESCRIPTOR = Path(__file__).resolve().parents[1] / 'client' / 'networks' / 'assistant-testnet.json'
NEURO = 1_000_000
SUPPLY = 1_000_000 * NEURO
PARAMS = {**ledger.PARAMS, 'job_blocks': 3600, 'challenge_blocks': 600, 'proof_blocks': 300, 'max_jobs': 256,
          'max_results': 1024}
PORTS = {'p2p': 26656, 'rpc': 26657, 'owner': 28700, 'faucet': 28780}
PEER_TIMEOUT = 600
HANDSHAKE_SECONDS = 5
MAX_HANDSHAKES = 16
MAX_PER_ADDRESS = 2
MAX_PAID_POSITIONS = 65536
CHUNK = 1 << 20


def descriptor(path=None):
    return json.loads(Path(path or DESCRIPTOR).read_text())


def model_root(network):
    """The ledger's name for the served version: its verified plan and the digests of its released files."""
    body = {'format': network['format'], 'version': network['version'], 'plan_sha256': network['plan_sha256'],
            'files': {name: pin['sha256'] for name, pin in sorted(network['files'].items())}}
    if network.get('policies'):
        body['policies'] = network['policies']
        body['routing'] = network['routing']
    return ledger.digest(body)


class Version:
    """What the network serves: partition, module spec, policy and bounds; the last shard holds the module."""

    def __init__(self, name, boundaries, spec, policy, eos_ids, max_tokens, root, policies=None):
        self.name, self.boundaries, self.spec, self.policy = name, list(boundaries), spec, policy
        self.eos_ids, self.max_tokens, self.root = set(eos_ids), max_tokens, root
        self.world = len(self.boundaries) - 1
        self.arm_rank = self.world - 1
        self.policies = policies


def version(network):
    """The served version from the network's verified serving plan, and the plan itself."""
    from neuroshard.evolution.modular_reference_execution import ROOT, read, sha256

    if sha256(ROOT / network['plan']) != network['plan_sha256']:
        raise ValueError('the serving plan differs from the network descriptor')
    plan = read(ROOT / network['plan'])
    learning = read(ROOT / plan['learning'])
    policies = {name: read(ROOT / pin['path']) for name, pin in network.get('policies', {}).items()}
    for name, pin in network.get('policies', {}).items():
        if sha256(ROOT / pin['path']) != pin['sha256']:
            raise ValueError(f'the {name} policy differs from the accepted version')
    return Version(network['version'], plan['boundaries'], learning['training'], read(ROOT / learning['policy']),
                   plan['eos_ids'], plan['max_boundary_tokens'], model_root(network), policies or None), plan


def digest_file(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(CHUNK), b''):
            value.update(block)
    return value.hexdigest()


def matches(path, pin):
    return path.is_file() and path.stat().st_size == pin['bytes'] and digest_file(path) == pin['sha256']


def local_path(directory, name):
    parts = PurePosixPath(name).parts
    if PurePosixPath(name).is_absolute() or '..' in parts or not 1 <= len(parts) <= 2:
        raise ValueError(f'unsafe release path: {name}')
    return Path(directory).joinpath(*parts)


def download(url, path, pin, opener=urllib.request.urlopen):
    """Stream one released file beside its destination; keep it only if its size and digest match the pin."""
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = tempfile.NamedTemporaryFile(dir=path.parent, prefix='.partial-', delete=False)
    partial = Path(handle.name)
    try:
        with handle, opener(url, timeout=60) as response:
            received = 0
            for block in iter(lambda: response.read(CHUNK), b''):
                received += len(block)
                if received > pin['bytes']:
                    raise ValueError(f'{path.name} is larger than its pin')
                handle.write(block)
        if not matches(partial, pin):
            raise ValueError(f'{path.name} differs from its pinned digest')
        partial.replace(path)
    finally:
        partial.unlink(missing_ok=True)


def stage_files(served, rank):
    if rank == 0:
        return ['integration.json'] + (['router.json'] if served.policies else [])
    if rank == served.arm_rank:
        return (['update-checkpoint/manifest.json', 'update-checkpoint/trainable.safetensors']
                + ([f'{unit}/{name}' for unit in ('L2', 'L3') for name in ('manifest.json', 'trainable.safetensors')]
                   if served.policies else []))
    return []


def fetch_stage(network, rank, home, progress=print):
    """This machine's part of the served version: its shard's byte ranges, pinned small files and released files."""
    from neuroshard.evolution import granite_shard_serving as serving_plan
    from neuroshard.evolution.modular_reference_execution import read

    served, plan = version(network)
    if not 0 <= rank < served.world:
        raise ValueError(f'the served version has stages 0 to {served.world - 1}')
    store = Path(home) / network['version'] / f'stage-{rank}'
    shard, manifest = store / 'shard' / f'partition-{rank}.safetensors', store / 'shard' / f'partition-{rank}.json'
    if not (manifest.exists() and shard.exists() and digest_file(shard) == read(manifest)['sha256']):
        progress(f'Fetching stage {rank} of {served.name}: only its own tensors, from the pinned Hugging Face revision')
        serving_plan.prepare(plan, rank, store)
    for name in stage_files(served, rank):
        pin, path = network['files'][name], local_path(store, name)
        if not matches(path, pin):
            progress(f'Downloading {name} ({pin["bytes"] / 2 ** 20:.1f} MiB)')
            download(network['assets'] + pin['asset'], path, pin)
    stage = {'config': store / 'config', 'shard': store / 'shard'}
    if rank == 0:
        stage.update(tokenizer=store / 'config', gate=read(store / 'integration.json')['arms']['update']['gate'])
        if served.policies:
            stage['router'] = read(store / 'router.json')['gate']
    if rank == served.arm_rank:
        stage['arm'] = store / 'update-checkpoint'
        if served.policies:
            stage['additions'] = {1: store / 'L3', 2: store / 'L2'}
    return served, stage


def load_adapter(partition, served, stage):
    from neuroshard.evolution.sharded import granite_serving as serving

    if partition.rank != served.arm_rank:
        return None
    if stage.get('additions'):
        return serving.AdapterBank(partition, served.spec, stage['arm'], stage['additions'])
    return serving.Adapter(partition, served.spec, stage['arm'])


def warm(partition, adapter):
    routes = sorted(adapter.routes) if hasattr(adapter, 'routes') else [None]
    for route in routes:
        if route is not None:
            adapter.select(route)
        for enabled in ((True, False) if adapter else (None,)):
            if adapter:
                adapter.set(enabled)
            partition.warm_up()


def private_file(path, content):
    path.parent.mkdir(parents=True, exist_ok=True)
    with os.fdopen(os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600), 'w') as handle:
        handle.write(content + '\n')


class Account:
    """A ledger account: a secp256k1 key kept in one file on this machine."""

    def __init__(self, path):
        from neuroshard.core.crypto.ecdsa import derive_keypair_from_token

        path = Path(path)
        if not path.exists():
            private_file(path, secrets.token_hex(32))
        self.key = derive_keypair_from_token(path.read_text().strip())
        self.public = self.key.public_key_bytes.hex()

    def sign(self, chain_id, nonce, kind, **fields):
        from neuroshard.core.crypto.ecdsa import ecdsa_sign

        body = {'kind': kind, 'chain_id': chain_id, 'nonce': nonce, **fields}
        return {'body': body, 'public_key': self.public,
                'signature': ecdsa_sign(ledger.canonical(body).decode(), self.key.private_key_bytes)}


def signing_key(path):
    """An Ed25519 key kept in one file on this machine: an owner's log key."""
    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

    path = Path(path)
    if not path.exists():
        private_file(path, Ed25519PrivateKey.generate().private_bytes_raw().hex())
    return Ed25519PrivateKey.from_private_bytes(bytes.fromhex(path.read_text().strip()))


def public_hex(key):
    return key.public_key().public_bytes_raw().hex()


class Chain:
    """A ledger RPC: reads state and submits transactions, waiting until a block includes each."""

    def __init__(self, rpc, chain_id):
        self.rpc, self.chain_id = rpc, chain_id

    def state(self):
        value = chain_network.state(self.rpc)
        if value['chain_id'] != self.chain_id:
            raise ValueError('the RPC serves another chain')
        return value

    def balance(self, account):
        return self.state()['accounts'].get(account.public, {'balance': 0})['balance']

    def nonce(self, account):
        return self.state()['accounts'].get(account.public, {'nonce': 0})['nonce']

    def submit(self, account, kind, nonce=None, timeout=180, **fields):
        envelope = account.sign(self.chain_id, self.nonce(account) if nonce is None else nonce, kind, **fields)
        admitted = chain_network.broadcast(self.rpc, envelope)
        if admitted['code']:
            raise chain_network.Rejected(admitted['log'])
        key = base64.b64encode(hashlib.sha256(ledger.canonical(envelope)).digest()).decode()
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            try:
                result = chain_network.rpc(self.rpc, 'tx', {'hash': key})
            except chain_network.Rejected as error:
                if 'not found' not in str(error):
                    raise
                time.sleep(0.5)
                continue
            if result['tx_result'].get('code', 0):
                raise chain_network.Rejected(result['tx_result'].get('log', 'transaction failed'))
            return envelope
        raise TimeoutError(f'transaction {ledger.transaction_id(envelope)} was not committed')

    def settled(self, job_id, timeout=None, poll=2.0):
        """The job's outcome once the ledger has settled, refunded or voided it; None while it is open."""
        deadline = None if timeout is None else time.monotonic() + timeout
        while True:
            state = self.state()
            if job_id in state['results']:
                return state['results'][job_id]
            if deadline is None or time.monotonic() > deadline:
                return None
            time.sleep(poll)


class Faucet:
    """Grants test NEURO to an account below a floor: once per account and a few times per address in a period."""

    def __init__(self, chain, account, amount=20 * NEURO, floor=10 * NEURO, period=3600, per_address=5):
        self.chain, self.account, self.amount, self.floor, self.period = chain, account, amount, floor, period
        self.per_address, self.lock, self.accounts, self.addresses = per_address, threading.Lock(), {}, {}

    def grant(self, public, address):
        public = ledger.account_key(public)
        now = time.monotonic()
        with self.lock:
            recent = [moment for moment in self.addresses.get(address, []) if now - moment < self.period]
            if now - self.accounts.get(public, -self.period) < self.period or len(recent) >= self.per_address:
                raise ValueError('this account or address was funded recently')
            balance = self.chain.state()['accounts'].get(public, {'balance': 0})['balance']
            if balance >= self.floor:
                return {'granted': 0, 'balance': balance}
            self.chain.submit(self.account, 'transfer', to=public, amount=self.amount)
            self.accounts[public], self.addresses[address] = now, recent + [now]
            return {'granted': self.amount, 'balance': balance + self.amount}


def serve_faucet(faucet, host, port, proof_store=None):
    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            try:
                proof = self.path.startswith('/proofs/') and proof_store is not None
                if self.path != '/faucet' and not proof:
                    raise KeyError('unknown path')
                size = int(self.headers.get('Content-Length', 0))
                from neuroshard.assistant import audit
                if not 0 < size <= (audit.MAX_PROOF_BYTES if proof else 1024):
                    raise ValueError('invalid request size')
                if proof:
                    root = self.path.removeprefix('/proofs/')
                    state = faucet.chain.state()
                    ledger.hex_digest(root)
                    if not any(c['proof_root'] == root for j in state['jobs'].values() for c in j['challenges'].values()):
                        raise ValueError('no funded open challenge names this proof')
                deadline, content = time.monotonic() + 30, bytearray()
                while len(content) < size:
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        raise ValueError('HTTP request deadline exceeded')
                    self.connection.settimeout(remaining)
                    block = self.rfile.read1(min(CHUNK, size - len(content)))
                    if not block:
                        raise ValueError('incomplete HTTP request')
                    content.extend(block)
                result = ({'proof_root': audit.receive_proof(content, root, state, proof_store)} if proof
                          else faucet.grant(json.loads(content)['account'], self.client_address[0]))
                code = 200
            except (ValueError, KeyError, TypeError) as error:
                result, code = {'error': str(error)}, 400
            except Exception as error:
                result, code = {'error': f'{type(error).__name__}: {error}'}, 503
            payload = json.dumps(result).encode()
            self.send_response(code)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Content-Length', str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def log_message(self, *args):
            pass

    class BoundedHTTPServer(ThreadingHTTPServer):
        def __init__(self, *args):
            self.slots = threading.BoundedSemaphore(16)
            super().__init__(*args)

        def process_request(self, request, address):
            if not self.slots.acquire(blocking=False):
                self.shutdown_request(request)
                return
            request.settimeout(HANDSHAKE_SECONDS)
            super().process_request(request, address)

        def process_request_thread(self, request, address):
            try:
                super().process_request_thread(request, address)
            finally:
                self.slots.release()

    server = BoundedHTTPServer((host, port), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server


def fund(url, account, timeout=180):
    request = urllib.request.Request(url, data=json.dumps({'account': account.public}).encode(),
                                     headers={'Content-Type': 'application/json'})
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.loads(response.read())
    except urllib.error.HTTPError as error:
        raise ValueError(json.loads(error.read() or b'{}').get('error', str(error))) from None


def ensure_balance(chain, account, needed, faucet_url, progress=print):
    if chain.balance(account) >= needed:
        return
    if not faucet_url:
        raise ValueError('this account needs test NEURO and the network names no faucet')
    progress('Requesting test NEURO from the faucet')
    fund(faucet_url, account)
    deadline = time.monotonic() + 120
    while chain.balance(account) < needed:
        if time.monotonic() > deadline:
            raise TimeoutError('the faucet grant did not reach this account')
        time.sleep(1)


class Owner:
    """One shard of the served version on this machine: bonded, reachable at its endpoint, serving jobs that name it."""

    def __init__(self, served, stage, rank, chain, account, log_key, home, threads=None):
        import torch

        from neuroshard.evolution.sharded import granite, granite_serving as serving

        if not 0 < rank < served.world:
            raise ValueError(f'owners host shards 1 to {served.world - 1}; stage 0 runs on the user device')
        self.threads = threads or min(8, os.cpu_count() or 1)
        torch.set_num_threads(self.threads)
        self.served, self.rank, self.chain, self.account, self.log_key = served, rank, chain, account, log_key
        self.home, self.public, self.jobs = Path(home), public_hex(log_key), {}
        self.serving_lock = threading.Lock()
        self.config = granite.load_config(stage['config'])
        self.partition, _ = granite.load_partition(self.config, stage['shard'], rank)
        self.adapter = load_adapter(self.partition, served, stage)
        # A fresh process's first pass can round differently; audits replay from warm passes.
        warm(self.partition, self.adapter)
        logs = self.home / 'logs'
        self.served_jobs = {path.name for path in logs.iterdir()} if logs.exists() else set()
        self.pending_jobs = set(self.served_jobs)

    def register(self, endpoint, faucet_url=None, bond=None, progress=print):
        """Bond this shard once, and publish where users reach it."""
        state = self.chain.state()
        if state['model_root'] != self.served.root:
            raise ValueError('the ledger serves another version than this owner holds')
        owner = state['owners'].get(self.public)
        if owner is not None and owner['status'] != 'active':
            raise ValueError('this log key is no longer bonded; start a new owner home')
        if owner is None:
            amount = bond or state['params']['owner_bond_minimum']
            ensure_balance(self.chain, self.account, amount + 4 * state['params']['fee'], faucet_url, progress)
            nonce = self.chain.nonce(self.account)
            possession = self.log_key.sign(ledger.possession_message(self.chain.chain_id, self.account.public,
                                                                     self.public, self.rank, amount, nonce)).hex()
            self.chain.submit(self.account, 'owner_bond', nonce=nonce, model_root=self.served.root, shard=self.rank,
                              log_key=self.public, amount=amount, possession=possession)
            progress(f'Bonded {amount / NEURO:g} NEURO for shard {self.rank} with log key {self.public[:16]}')
            owner = {}
        if owner.get('endpoint') != endpoint:
            self.chain.submit(self.account, 'owner_endpoint', log_key=self.public, endpoint=endpoint)
            progress(f'Published endpoint {endpoint}')

    def job_of(self, chain_id, job_id):
        state = self.chain.state() if chain_id == self.chain.chain_id else {}
        job = state.get('jobs', {}).get(job_id)
        if (job is None or job['deadline'] is not None or len(job['owners']) != self.served.world - 1
                or job['owners'][self.rank - 1] != self.public or state['height'] > job['expires']
                or state['model_root'] != self.served.root):
            raise ValueError('no open job names this owner')
        if job['positions'] > MAX_PAID_POSITIONS:
            raise ValueError('job exceeds this owner\'s maximum paid positions')
        if job_id in self.served_jobs:
            raise ValueError('this owner already served that job')
        return job

    def session_key_of(self, chain_id, job_id):
        return self.job_of(chain_id, job_id)['session_key']

    def serve(self, connection, progress=print):
        """Serve one job over one connection, then commit its signed log; returns what was served, or None."""
        from neuroshard.evolution.sharded import granite_audit, granite_serving as serving

        try:
            deadline = time.monotonic() + HANDSHAKE_SECONDS
            message = json.loads(relay.receive_frame(connection, (relay.HELLO, relay.AUDIT), relay.MAX_HELLO, deadline))
            if (isinstance(message, dict) and isinstance(message.get('body'), dict)
                    and message['body'].get('kind') == 'audit_log'):
                from neuroshard.assistant.audit import serve_log
                connection.settimeout(30)
                try:
                    serve_log(self, connection, message)
                finally:
                    connection.close()
                return None
            body = relay.answer(connection, self.rank, self.served.world, self.session_key_of,
                                deadline=deadline, ready=False, message=message)
            if body['hidden'] != self.config.hidden_size or body['max_tokens'] != self.served.max_tokens:
                raise ValueError('the hello differs from this owner\'s model')
        except (ValueError, KeyError, TypeError, ConnectionError, OSError):
            connection.close()
            return None
        if not self.serving_lock.acquire(blocking=False):
            connection.close()
            return None
        try:
            return self.serve_authenticated(connection, body, progress)
        except Exception as error:
            progress(f'Owner job failed: {type(error).__name__}: {error}')
            return None
        finally:
            connection.close()
            self.serving_lock.release()

    def serve_authenticated(self, connection, body, progress):
        import torch
        from neuroshard.evolution.sharded import granite_audit, granite_serving as serving

        torch.set_num_threads(self.threads)
        connection.settimeout(PEER_TIMEOUT)
        job_id = body['job_id']
        job = self.job_of(body['chain_id'], job_id)
        self.served_jobs.add(job_id)
        relay.send_frame(connection, relay.READY, ledger.canonical({'format': relay.FORMAT, 'rank': self.rank}))
        session = {'chain_id': self.chain.chain_id, 'job_id': job_id, 'request_root': job['request_root'],
                   'session_key': job['session_key'], 'log_keys': job['owners']}
        upstream = job['session_key'] if self.rank == 1 else job['owners'][self.rank - 2]
        link = granite_audit.Link(session, self.rank, self.served.world, self.log_key, upstream)
        log = granite_audit.OwnerLog(self.rank)
        checked_at = [0.0]

        def alive():
            if time.monotonic() - checked_at[0] >= 10:
                current = self.chain.state()
                if job_id not in current['jobs'] or current['height'] > job['expires']:
                    raise ValueError('the paid job has expired')
                checked_at[0] = time.monotonic()

        budget = relay.WorkBudget(job['positions'], self.config.max_position_embeddings, alive)
        ring = relay.OwnerRelay(connection, self.rank, self.served.world, self.config.hidden_size,
                                self.served.max_tokens, budget, deadline=time.monotonic() + 3600)
        try:
            serving.serve(self.partition, ring, self.adapter, log, None, link)
        except (ConnectionError, OSError, ValueError) as error:
            progress(f'Job {job_id[:12]} ended before its stop: {error}')
        finally:
            connection.close()
        if log.attested is None:
            return None
        directory = self.home / 'logs' / job_id
        log.save(directory, self.log_key, session)
        self.pending_jobs.add(job_id)
        commitment = granite_audit.commitment(granite_audit.load(directory)[0])
        signature = self.log_key.sign(ledger.commitment_message(self.chain.chain_id, job_id,
                                                                commitment['statement_root'])).hex()
        self.chain.submit(self.account, 'log_commit', job_id=job_id, log_signature=signature, **commitment)
        self.pending_jobs.discard(job_id)
        return {'job_id': job_id, 'positions': log.attested['positions']}

    def commit_pending(self):
        """Recover saved commitments after a process restart or a lost RPC reply."""
        from neuroshard.evolution.sharded import granite_audit

        if not self.pending_jobs or not self.serving_lock.acquire(blocking=False):
            return
        try:
            state = self.chain.state()
            for job_id in sorted(self.pending_jobs):
                job = state['jobs'].get(job_id)
                if job is None or self.public in job['commits']:
                    self.pending_jobs.discard(job_id)
                    continue
                record, _ = granite_audit.load(self.home / 'logs' / job_id)
                commitment = granite_audit.commitment(record)
                signature = self.log_key.sign(ledger.commitment_message(self.chain.chain_id, job_id,
                                                                        commitment['statement_root'])).hex()
                self.chain.submit(self.account, 'log_commit', job_id=job_id, log_signature=signature, **commitment)
                self.pending_jobs.discard(job_id)
        finally:
            self.serving_lock.release()

    def run(self, host, port, stop=None, progress=print):
        """Authenticate bounded connections independently; one authenticated job owns the model."""
        slots = threading.BoundedSemaphore(MAX_HANDSHAKES)
        lock, addresses, connections = threading.Lock(), {}, set()

        def handle(connection, address):
            try:
                outcome = self.serve(connection, progress)
                if outcome:
                    progress(f"Served job {outcome['job_id'][:12]}: {outcome['positions']} positions; log committed")
            finally:
                connection.close()
                with lock:
                    connections.discard(connection)
                    addresses[address] -= 1
                    if not addresses[address]:
                        del addresses[address]
                slots.release()

        with socket.create_server((host, port)) as listener:
            listener.settimeout(1.0)
            progress(f'Shard {self.rank} of {self.served.name} is serving on {host}:{port}')
            while stop is None or not stop.is_set():
                if getattr(self, 'pending_jobs', None):
                    try:
                        self.commit_pending()
                    except Exception as error:
                        progress(f'Pending commitment will retry: {error}')
                try:
                    connection, peer = listener.accept()
                except socket.timeout:
                    continue
                address = peer[0]
                with lock:
                    if addresses.get(address, 0) >= MAX_PER_ADDRESS or not slots.acquire(blocking=False):
                        connection.close()
                        continue
                    addresses[address] = addresses.get(address, 0) + 1
                    connections.add(connection)
                connection.settimeout(HANDSHAKE_SECONDS)
                threading.Thread(target=handle, args=(connection, address), daemon=True).start()
            with lock:
                for connection in connections:
                    connection.close()


def reachable(endpoint, timeout=5):
    host, port = endpoint.rsplit(':', 1)
    try:
        socket.create_connection((host, int(port)), timeout=timeout).close()
        return True
    except OSError:
        return False


def choose_owners(state, served, rng=None, probe=reachable):
    """One active, reachable owner per shard after stage 0, in shard order: (log key, endpoint)."""
    rng = rng or random.Random()
    chosen = []
    for shard in range(1, served.world):
        options = sorted((key, owner['endpoint']) for key, owner in state['owners'].items()
                         if owner['shard'] == shard and owner['status'] == 'active' and owner.get('endpoint'))
        rng.shuffle(options)
        found = next((option for option in options if probe(option[1])), None)
        if found is None:
            raise ValueError(f'no reachable owner hosts shard {shard} yet: the network needs a peer for it')
        chosen.append(found)
    return chosen


class Conversation:
    """One paid job: stage 0 runs here, each later shard on the owner the job names, tools in a local workspace."""

    def __init__(self, served, stage, chain, account, world, budget=16384, price=NEURO, faucet_url=None,
                 progress=print, rng=None, probe=reachable, threads=None):
        import torch
        from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

        from neuroshard.assistant.public import Session
        from neuroshard.evolution import granite_tokenizer
        from neuroshard.evolution.sharded import granite, granite_audit, granite_serving as serving

        torch.set_num_threads(threads or min(8, os.cpu_count() or 1))
        self.served, self.chain, self.world, self.gate = served, chain, world, stage['gate']
        self.tokenizer, _ = granite_tokenizer.load(stage['tokenizer'], parent_digest=stage.get('parent_tokenizer_digest'))
        config = granite.load_config(stage['config'])
        partition, _ = granite.load_partition(config, stage['shard'], 0)
        partition.warm_up()
        state = chain.state()
        if state['model_root'] != served.root:
            raise ValueError('the ledger serves another version than this device holds')
        self.owners = choose_owners(state, served, rng, probe)
        ensure_balance(chain, account, price + 2 * state['params']['fee'], faucet_url, progress)
        session_key = Ed25519PrivateKey.generate()
        request_root = ledger.digest({'domain': 'neuroshard/assistant-request/v1', 'world': ledger.digest(world),
                                      'nonce': secrets.token_hex(16)})
        envelope = chain.submit(account, 'serve_open', model_root=served.root, owners=[key for key, _ in self.owners],
                                request_root=request_root, session_key=public_hex(session_key), price=price,
                                positions=budget)
        self.job_id = ledger.transaction_id(envelope)
        session = {'chain_id': chain.chain_id, 'job_id': self.job_id, 'request_root': request_root,
                   'session_key': public_hex(session_key), 'log_keys': [key for key, _ in self.owners]}
        self.link = granite_audit.Link(session, 0, served.world, session_key, self.owners[-1][0])
        self.sockets = {}
        try:
            for rank, (_, endpoint) in enumerate(self.owners, start=1):
                self.sockets[rank] = relay.dial(endpoint, relay.hello(chain.chain_id, self.job_id, rank, served.world,
                                                                      config.hidden_size, served.max_tokens), session_key)
        except BaseException:
            self.close()
            raise
        self.budget = relay.WorkBudget(budget, config.max_position_embeddings)
        ring = relay.DriverRelay(self.sockets, served.world, config.hidden_size, served.max_tokens, signed=True,
                                 budget=self.budget)
        self.driver = serving.ServingDriver(partition, ring, self.link)
        self.session = Session(served.policy, world, name=served.name, previous='a2-u1' if served.policies else None,
                               modules=('U1', 'L2', 'L3') if served.policies else ('U1',),
                               route_policies=served.policies)
        self.router, self.responses = stage.get('router'), {}
        self.respond = self.arm = None

    def say(self, text):
        """One user turn: the gate chooses the module once, from the first turn, then tools run here."""
        from neuroshard.evolution import assistant_experience_run as accelerator
        from neuroshard.evolution import assistant_selector as selector
        from neuroshard.evolution.sharded import granite_serving as serving

        if self.arm is None:
            case = {'world': self.world, 'turns': [{'user': text}]}
            ids = accelerator.feature_ids(self.tokenizer, self.served.policy, case)
            self.arm = selector.choose(self.gate, self.driver.feature(ids))
            self.driver.episode(self.arm)
            self.respond = serving.responder(self.driver, self.tokenizer, self.served.policy, self.served.eos_ids)
        if self.router is not None:
            feature = self.driver.message_feature(self.tokenizer, self.served.policies['drafting'], text)
            route = 'scheduling' if selector.choose(self.router, feature) else 'drafting'
            self.driver.route(2 if route == 'scheduling' else 1, self.arm if route == 'drafting' else True)
            if route not in self.responses:
                self.responses[route] = serving.responder(self.driver, self.tokenizer, self.served.policies[route],
                                                         self.served.eos_ids)
            return self.session.turn(text, self.responses[route], route=route)
        return self.session.turn(text, self.respond)

    def close(self):
        """Stop the owners, which then commit their logs; the job settles after its challenge window."""
        try:
            if getattr(self, 'driver', None) is not None:
                self.driver.stop()
        finally:
            for sock in self.sockets.values():
                sock.close()
            self.sockets = {}


def seed(network, home, public_host, served=None, block_seconds=1.0, params=None, p2p_port=PORTS['p2p'],
         faucet_port=PORTS['faucet'], progress=print, audit_shards=None):
    """The project's seed: the ledger's first validator, its public RPC and the faucet.

    It holds no shard of the model; serving needs owners. The RPC listens on the next port after P2P.
    """
    served = served or version(network)[0]
    home, rpc_port = Path(home), p2p_port + 1
    faucet = Account(home / 'faucet.key')
    config_path = home / 'chain' / 'network.json'
    if not config_path.exists():
        terms = {'chain_id': network['chain_id'], 'model_root': served.root, 'shards': served.world,
                 'allocations': {faucet.public: SUPPLY}, 'params': params or PARAMS}
        config = chain_network.initialize(home / 'chain', terms, [{}], base_port=p2p_port, block_seconds=block_seconds)
        node = Path(config['validators'][0]['home']) / 'config' / 'config.toml'
        text = node.read_text()
        for section, key, value in (('rpc', 'laddr', f'"tcp://0.0.0.0:{rpc_port}"'),
                                    ('p2p', 'laddr', f'"tcp://0.0.0.0:{p2p_port}"'),
                                    ('p2p', 'external_address', json.dumps(f'{public_host}:{p2p_port}'))):
            text = chain_network.edit_config(text, section, key, value)
        node.write_text(text)
    config = json.loads(config_path.read_text())
    for node in config['validators']:
        path = Path(node['home']) / 'settlement.json'
        terms = json.loads(path.read_text())
        terms.update(promotion_authority=faucet.public, approved_models=network.get('approved_versions', {}))
        path.write_text(json.dumps(terms, sort_keys=True))
    if audit_shards:
        shards = {}
        for rank in audit_shards:
            _, stage = fetch_stage(network, rank, home / 'audit-holdings', progress)
            paths = {name: str(stage[name]) for name in ('config', 'shard')}
            if 'arm' in stage:
                paths.update(arm=str(stage['arm']), spec=served.spec)
                if 'additions' in stage:
                    paths['additions'] = {str(k): str(v) for k, v in stage['additions'].items()}
            shards[str(rank)] = paths
        for node in config['validators']:
            path = Path(node['home']) / 'settlement.json'
            terms = json.loads(path.read_text())
            terms.update(runtime='granite-shard', threads=8, bundles=str(home / 'bundles'), shards=shards)
            path.write_text(json.dumps(terms, sort_keys=True))
    chain_network.start(config)
    chain = Chain(chain_network.url(config), network['chain_id'])
    server = serve_faucet(Faucet(chain, faucet), '0.0.0.0', faucet_port, home / 'bundles')
    progress(json.dumps({'chain_id': network['chain_id'], 'rpc': f'http://{public_host}:{rpc_port}',
                         'faucet': f'http://{public_host}:{faucet_port}/faucet',
                         'persistent_peers': f"{config['validators'][0]['id']}@{public_host}:{p2p_port}"}))
    return config, chain, server
