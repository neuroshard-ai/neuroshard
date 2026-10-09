"""Fetch committed logs, replay one owned shard, and deliver funded fraud proofs to validators."""

import hashlib
import io
import json
from pathlib import Path
import shutil
import socket
import tempfile
import time
import urllib.request
import zipfile

from neuroshard.inference import optimistic as ledger, relay

MAX_LOG_BYTES = 512 << 20
MAX_PROOF_BYTES = 128 << 20
CHUNK = 1 << 20
LOG_FILES = ('log.json', 'inputs.safetensors')
PROOF_FILES = ('inputs.safetensors', 'proof.json', 'record.json')


def serve_log(owner, connection, envelope):
    """Only funded, signed audit requests can fetch logs inside their challenge window."""
    from neuroshard.assistant.network import digest_file

    body, public = ledger.verify(envelope)
    if (set(body) != {'kind', 'chain_id', 'nonce', 'job_id'} or body['kind'] != 'audit_log'
            or body['chain_id'] != owner.chain.chain_id):
        raise ValueError('invalid audit request')
    state = owner.chain.state()
    if state['accounts'].get(public, {}).get('balance', 0) < state['params']['challenge_deposit']:
        raise ValueError('an auditor must hold its challenge deposit')
    job_id = ledger.hex_digest(body['job_id'])
    job = state['jobs'].get(job_id)
    if job is None or owner.public not in job['commits']:
        raise ValueError('no committed log inside an open challenge window')
    directory = owner.home / 'logs' / job_id
    paths = {name: directory / name for name in LOG_FILES}
    listing = {name: {'bytes': path.stat().st_size, 'sha256': digest_file(path)} for name, path in paths.items()}
    if sum(pin['bytes'] for pin in listing.values()) > MAX_LOG_BYTES:
        raise ValueError('log exceeds the audit transfer bound')
    relay.send_frame(connection, relay.READY, ledger.canonical({'files': listing}))
    for path in paths.values():
        with path.open('rb') as handle:
            for block in iter(lambda: handle.read(CHUNK), b''):
                relay.send_frame(connection, relay.LOG_DATA, block)
    relay.send_frame(connection, relay.READY, ledger.canonical({'done': True}))


def fetch_log(chain, account, job_id, key, endpoint, directory):
    """Keep a downloaded log only if its bytes, owner signature and on-chain commitment agree."""
    from neuroshard.evolution.sharded import granite_audit

    state = chain.state()
    expected = state['jobs'][job_id]['commits'][key]
    envelope = account.sign(chain.chain_id, chain.nonce(account), 'audit_log', job_id=job_id)
    host, port = ledger.endpoint(endpoint).rsplit(':', 1)
    directory = Path(directory)
    directory.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(dir=directory.parent))
    try:
        with socket.create_connection((host, int(port)), timeout=30) as connection:
            relay.send_frame(connection, relay.AUDIT, ledger.canonical(envelope))
            listing = json.loads(relay.receive_frame(connection, relay.READY, 4096))['files']
            if set(listing) != set(LOG_FILES):
                raise ValueError('audit log inventory differs')
            sizes = [ledger.integer(listing[name]['bytes'], 1, MAX_LOG_BYTES) for name in LOG_FILES]
            if sum(sizes) > MAX_LOG_BYTES:
                raise ValueError('audit log exceeds its transfer bound')
            for name, size in zip(LOG_FILES, sizes):
                digest, remaining = hashlib.sha256(), size
                with (temporary / name).open('wb') as handle:
                    while remaining:
                        block = relay.receive_frame(connection, relay.LOG_DATA, min(CHUNK, remaining))
                        if not block:
                            raise ValueError('empty audit data frame')
                        remaining -= len(block)
                        digest.update(block)
                        handle.write(block)
                if digest.hexdigest() != listing[name]['sha256']:
                    raise ValueError('audit download differs from its digest')
            if json.loads(relay.receive_frame(connection, relay.READY, 4096)) != {'done': True}:
                raise ValueError('incomplete audit transfer')
        record, _ = granite_audit.load(temporary)
        if (not granite_audit.signed_by(record, key) or granite_audit.statement(record).hex() != expected
                or record['session']['chain_id'] != chain.chain_id or record['session']['job_id'] != job_id):
            raise ValueError('downloaded log differs from the committed owner statement')
        if directory.exists():
            shutil.rmtree(directory)
        temporary.replace(directory)
        return directory
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)


def proof_archive(directory):
    payload = io.BytesIO()
    with zipfile.ZipFile(payload, 'w', compression=zipfile.ZIP_STORED) as archive:
        for name in PROOF_FILES:
            archive.write(Path(directory) / name, name)
    if payload.tell() > MAX_PROOF_BYTES:
        raise ValueError('fraud proof exceeds the transfer bound')
    return payload.getvalue()


def receive_proof(payload, root, state, store):
    """Accept bounded content-addressed bytes only after their challenge deposit is committed."""
    from neuroshard.evolution.sharded import granite_audit

    ledger.hex_digest(root)
    if not any(challenge['proof_root'] == root for job in state['jobs'].values()
               for challenge in job['challenges'].values()):
        raise ValueError('no funded open challenge names this proof')
    if len(payload) > MAX_PROOF_BYTES:
        raise ValueError('proof exceeds its transfer bound')
    with zipfile.ZipFile(io.BytesIO(payload)) as archive:
        items = archive.infolist()
        if (len(items) != len(PROOF_FILES) or {item.filename for item in items} != set(PROOF_FILES)
                or any(item.compress_type != zipfile.ZIP_STORED or item.file_size < 0 for item in items)
                or sum(item.file_size for item in items) > MAX_PROOF_BYTES):
            raise ValueError('invalid proof inventory or compression')
        files = {name: archive.read(name) for name in PROOF_FILES}
    if granite_audit.bundle_digest(files) != root:
        raise ValueError('proof bytes differ from their content address')
    store = Path(store)
    store.mkdir(parents=True, exist_ok=True)
    destination = store / root
    if destination.exists():
        granite_audit.read_bundle(store, root)
        return root
    temporary = Path(tempfile.mkdtemp(dir=store))
    try:
        for name, content in files.items():
            (temporary / name).write_bytes(content)
        try:
            temporary.rename(destination)
        except FileExistsError:
            granite_audit.read_bundle(store, root)
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)
    return root


class Auditor:
    def __init__(self, chain, account, network, home, holdings, progress=print):
        self.chain, self.account, self.network = chain, account, network
        self.home, self.holdings, self.progress = Path(home), holdings, progress
        self.reports = self.home / 'reports.json'
        self.done = json.loads(self.reports.read_text()) if self.reports.exists() else {}

    def once(self):
        from neuroshard.evolution.sharded import granite_audit

        state = self.chain.state()
        for job_id, job in list(state['jobs'].items()):
            for rank, (partition, adapter) in self.holdings.items():
                key = job['owners'][rank - 1]
                if key not in job['commits']:
                    continue
                identity = f'{job_id}/{key}/{job["commits"][key]}'
                if identity in self.done:
                    continue
                try:
                    endpoint = state['owners'][key]['endpoint']
                    directory = fetch_log(self.chain, self.account, job_id, key, endpoint,
                                          self.home / 'logs' / job_id / str(rank))
                    record, inputs = granite_audit.load(directory)
                    if granite_audit.unattested(record):
                        proof = granite_audit.claim_proof(record, 'unattested')
                        report = {'valid': False, 'claim': 'unattested'}
                    else:
                        report = granite_audit.replay(partition, record, inputs, adapter)
                        if report['input_mismatch'] is not None:
                            raise ValueError('audit inputs do not reproduce the committed log')
                        proof = None if report['valid'] else granite_audit.fraud_proof(record, inputs, report)
                    if proof is not None:
                        proof_dir = self.home / 'proofs' / job_id / str(rank)
                        granite_audit.save_proof(proof, proof_dir)
                        root = granite_audit.bundle_root(proof_dir)
                        current = self.chain.state()['jobs'][job_id]
                        challenge_id = next((identity for identity, challenge in current['challenges'].items()
                                             if challenge['challenger'] == self.account.public
                                             and challenge['log_key'] == key and challenge['proof_root'] == root), None)
                        if challenge_id is None:
                            envelope = self.chain.submit(self.account, 'challenge', job_id=job_id,
                                                         log_key=key, proof_root=root)
                            challenge_id = ledger.transaction_id(envelope)
                        payload = proof_archive(proof_dir)
                        for endpoint in self.network['proof_endpoints']:
                            request = urllib.request.Request(endpoint.rstrip('/') + '/' + root, data=payload,
                                                             headers={'Content-Type': 'application/zip'})
                            with urllib.request.urlopen(request, timeout=60) as response:
                                response.read(4096)
                        self.chain.submit(self.account, 'prove', job_id=job_id,
                                          challenge_id=challenge_id)
                        report['proof_root'] = root
                    self.done[identity] = report
                    self.reports.parent.mkdir(parents=True, exist_ok=True)
                    temporary = self.reports.with_suffix('.partial')
                    temporary.write_text(json.dumps(self.done, sort_keys=True))
                    temporary.replace(self.reports)
                    self.progress(f'Audited {job_id[:12]} shard {rank}: {"clean" if proof is None else "fraud proven"}')
                except Exception as error:
                    self.progress(f'Audit {job_id[:12]} shard {rank} will retry: {type(error).__name__}: {error}')

    def run(self, interval=10, stop=None):
        while stop is None or not stop.is_set():
            self.once()
            if stop is None:
                time.sleep(interval)
            else:
                stop.wait(interval)
