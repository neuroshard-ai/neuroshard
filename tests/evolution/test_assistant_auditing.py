import copy
import json
from pathlib import Path
import shutil
import socket
import threading
from types import SimpleNamespace

import pytest
import torch

from neuroshard.assistant import audit, network
from neuroshard.evolution.sharded import granite_audit
from neuroshard.inference import optimistic as ledger
from test_bound_owner_logs import commit, market, partitions, saved, serve  # noqa: F401
from test_optimistic_serving import CHAIN


class Account:
    def __init__(self, original):
        self.original, self.public = original, original.public

    def sign(self, chain_id, nonce, kind, **fields):
        assert chain_id == CHAIN
        self.original.nonce = nonce
        return self.original.sign(kind, **fields)


class Chain:
    chain_id = CHAIN

    def __init__(self, market, checker):
        self.market, self.checker = market, checker

    def state(self):
        return copy.deepcopy(self.market['state'])

    def nonce(self, account):
        return self.state()['accounts'][account.public]['nonce']

    def submit(self, account, kind, **fields):
        envelope = account.sign(CHAIN, self.nonce(account), kind, **fields)
        current = ledger.advance(self.market['state'], self.market['state']['height'] + 1)
        self.market['state'] = ledger.transition(current, envelope, self.checker)
        return envelope


@pytest.mark.parametrize('fault', [None, (1, 0)])
def test_a_downloaded_committed_log_is_audited_and_a_real_fault_is_delivered_and_slashed(partitions, market,
                                                                                     tmp_path, fault):
    torch.set_num_threads(1)
    session = market['open']('cd' * 32)
    logs, _ = serve(partitions, session, market['keys'], fault=fault)
    records = saved(logs, market['keys'], session, tmp_path)
    for rank in (1, 2):
        market['state'] = ledger.transition(market['state'], commit(market, rank, session, records[rank][0]), None)
    store = tmp_path / 'bundles'
    chain = Chain(market, granite_audit.challenge_checker(store, partitions))
    account = Account(market['people']['auditor'])
    owner = network.Owner.__new__(network.Owner)
    owner.rank, owner.public, owner.chain = 1, market['public'][1], chain
    owner.home = tmp_path / 'remote-owner'
    owner.served = SimpleNamespace(world=3, max_tokens=2048, name='tiny')
    owner.pending_jobs = set()
    owner.serving_lock = threading.Lock()
    destination = owner.home / 'logs' / session['job_id']
    destination.parent.mkdir(parents=True)
    shutil.move(str(tmp_path / 'log-1'), str(destination))
    with socket.create_server(('127.0.0.1', 0)) as probe:
        port = probe.getsockname()[1]
    market['state']['owners'][owner.public]['endpoint'] = f'127.0.0.1:{port}'
    stop = threading.Event()
    thread = threading.Thread(target=owner.run, args=('127.0.0.1', port, stop, lambda *_: None), daemon=True)
    thread.start()
    server = network.serve_faucet(network.Faucet(chain, account), '127.0.0.1', 0, store)
    descriptor = {'proof_endpoints': [f'http://127.0.0.1:{server.server_port}/proofs']}
    auditor = audit.Auditor(chain, account, descriptor, tmp_path / 'auditor', {1: (partitions[1], None)}, print,
                            model_root=chain.state()['model_root'])
    try:
        auditor.once()
        assert len(auditor.done) == 1
        report = next(iter(auditor.done.values()))
        assert report['valid'] == (fault is None)
        if fault is not None:
            assert chain.state()['owners'][owner.public]['status'] == 'slashed'
            assert session['job_id'] not in chain.state()['jobs']
            assert report['proof_root'] in [path.name for path in store.iterdir()]
        else:
            assert chain.state()['jobs'][session['job_id']]['challenges'] == {}
        ledger.invariant(chain.state())
        assert json.loads(auditor.reports.read_text()) == auditor.done
    finally:
        stop.set()
        thread.join(3)
        server.shutdown()
        server.server_close()


def test_a_proof_upload_requires_a_committed_deposit_before_parsing_or_writing(tmp_path):
    with pytest.raises(ValueError, match='funded'):
        audit.receive_proof(b'not a zip archive', 'ab' * 32, {'jobs': {}}, tmp_path / 'store')
    assert not (tmp_path / 'store').exists()


def test_a_fraud_proof_delivered_to_one_store_reaches_every_validator_and_the_cheater_is_slashed(partitions, market,
                                                                                                tmp_path):
    """The live failure of 2026-10-27: an auditor's bundle reached one of four validator stores (the others read a
    light-verified state from before the challenge and refused it), so three validators could never judge the
    proof, refused every block carrying it, and the challenge lapsed in the cheater's favor. Validators now copy
    open challenges' bundles from each other, so delivery to any one store is enough."""
    torch.set_num_threads(1)
    session = market['open']('ef' * 32)
    logs, _ = serve(partitions, session, market['keys'], fault=(1, 0))
    records = saved(logs, market['keys'], session, tmp_path)
    for rank in (1, 2):
        market['state'] = ledger.transition(market['state'], commit(market, rank, session, records[rank][0]), None)
    stores = [tmp_path / f'validator-{i}' / 'bundles' for i in range(4)]
    chain = Chain(market, granite_audit.challenge_checker(stores[0], partitions))
    account = Account(market['people']['auditor'])
    record, inputs = granite_audit.load(tmp_path / 'log-1')
    report = granite_audit.replay(partitions[1], record, inputs, None)
    assert not report['valid']
    proof_dir = tmp_path / 'proof'
    granite_audit.save_proof(granite_audit.fraud_proof(record, inputs, report), proof_dir)
    root = granite_audit.bundle_root(proof_dir)
    envelope = chain.submit(account, 'challenge', job_id=session['job_id'], log_key=market['public'][1],
                            proof_root=root)
    servers = [network.serve_faucet(network.Faucet(chain, account), '127.0.0.1', 0, store) for store in stores]
    endpoints = [f'http://127.0.0.1:{server.server_port}/proofs' for server in servers]
    try:
        # The auditor's upload reaches only the first store.
        assert audit.post_proof(endpoints[0], root, audit.proof_archive(proof_dir))['proof_root'] == root
        for store in stores[1:]:
            with pytest.raises(ledger.ProofUnavailable):
                granite_audit.read_bundle(store, root)
        # One replication pass on every other validator copies it from a peer that holds it.
        for i, store in enumerate(stores[1:], start=1):
            peers = [e for j, e in enumerate(endpoints) if j != i]
            assert network.replicate(chain, store, peers, progress=lambda *_: None) == 1
        for store in stores:
            assert granite_audit.read_bundle(store, root)
        # Every validator can now judge it, and the proof slashes the cheater.
        for store in stores:
            assert granite_audit.challenge_checker(store, partitions)(chain.state(),
                ledger.challenge_request(chain.state(), session['job_id'],
                                         chain.state()['jobs'][session['job_id']]['challenges'][
                                             ledger.transaction_id(envelope)])) is True
        chain.submit(account, 'prove', job_id=session['job_id'], challenge_id=ledger.transaction_id(envelope))
        assert chain.state()['owners'][market['public'][1]]['status'] == 'slashed'
        assert chain.state()['owners'][market['public'][2]]['status'] == 'active'
        ledger.invariant(chain.state())
        # A second pass copies nothing: the bundle is held everywhere.
        assert network.replicate(chain, stores[1], endpoints[:1], progress=lambda *_: None) == 0
    finally:
        for server in servers:
            server.shutdown()
            server.server_close()


def test_a_store_serves_only_a_complete_bundle_and_a_receiver_rejects_altered_bytes(partitions, market, tmp_path):
    torch.set_num_threads(1)
    session = market['open']('fe' * 32)
    logs, _ = serve(partitions, session, market['keys'], fault=(1, 0))
    records = saved(logs, market['keys'], session, tmp_path)
    for rank in (1, 2):
        market['state'] = ledger.transition(market['state'], commit(market, rank, session, records[rank][0]), None)
    chain = Chain(market, None)
    account = Account(market['people']['auditor'])
    record, inputs = granite_audit.load(tmp_path / 'log-1')
    proof_dir = tmp_path / 'proof'
    granite_audit.save_proof(granite_audit.fraud_proof(record, inputs, granite_audit.replay(
        partitions[1], record, inputs, None)), proof_dir)
    root = granite_audit.bundle_root(proof_dir)
    chain.submit(account, 'challenge', job_id=session['job_id'], log_key=market['public'][1], proof_root=root)
    holder, empty = tmp_path / 'holder', tmp_path / 'empty'
    audit.receive_proof(audit.proof_archive(proof_dir), root, chain.state(), holder)
    server = network.serve_faucet(network.Faucet(chain, account), '127.0.0.1', 0, holder)
    endpoint = f'http://127.0.0.1:{server.server_port}/proofs'
    try:
        import urllib.error
        import urllib.request
        with urllib.request.urlopen(f'{endpoint}/{root}.zip', timeout=30) as response:
            assert audit.receive_proof(response.read(), root, chain.state(), empty) == root
        with pytest.raises(urllib.error.HTTPError):  # not held: nothing is served
            urllib.request.urlopen(f'{endpoint}/{"00" * 32}.zip', timeout=30)
        (holder / root / 'proof.json').write_text('{}')  # an incomplete or altered bundle is never served
        with pytest.raises(urllib.error.HTTPError):
            urllib.request.urlopen(f'{endpoint}/{root}.zip', timeout=30).read()
        with pytest.raises(ValueError, match='content address'):
            audit.receive_proof(audit.proof_archive(holder / root), root, chain.state(), tmp_path / 'third')
    finally:
        server.shutdown()
        server.server_close()


def test_a_followed_version_is_reached_through_the_current_endpoints_not_those_saved_when_it_was_rebuilt(
        tmp_path, monkeypatch):
    """The second live finding (2026-10-28): a rebuilt version's saved descriptor froze the stores and RPCs of the
    day it was first rebuilt (only the seed's store), so validators replicated proofs only from one place."""
    from neuroshard.evolution import growth_runtime

    root = 'ab' * 32
    saved = {'version': 'grown', 'model': 'm', 'rpc': 'http://old:26657', 'rpcs': [],
             'proof_endpoints': ['http://seed:28780/proofs'], 'mirrors': ['http://gone:28781/']}
    current = {'version': 'base', 'model': 'm0', 'rpc': 'http://seed:26657', 'rpcs': ['http://v4:26657'],
               'proof_endpoints': ['http://seed:28780/proofs', 'http://v4:28780/proofs', 'http://v5:28780/proofs'],
               'mirrors': ['http://v4:28781/'], 'trusted_validators': {'k': 1}}
    (tmp_path / 'versions' / root).mkdir(parents=True)
    (tmp_path / 'versions' / root / 'network.json').write_text(json.dumps(saved))
    monkeypatch.setattr(network, 'model_root', lambda d: root if d.get('version') == 'grown' else 'cd' * 32)
    monkeypatch.setattr(growth_runtime, 'install', lambda *_: None)
    followed = growth_runtime.follow(current, {'model_root': root}, tmp_path, current['proof_endpoints'])
    assert followed['version'] == 'grown' and followed['model'] == 'm'  # what is served: the rebuilt version
    for field in ('rpc', 'rpcs', 'proof_endpoints', 'mirrors', 'trusted_validators'):  # where: the current one
        assert followed[field] == current[field]
    assert json.loads((tmp_path / 'versions' / root / 'network.json').read_text()) == saved  # cache untouched
