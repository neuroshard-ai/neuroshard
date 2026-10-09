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
    auditor = audit.Auditor(chain, account, descriptor, tmp_path / 'auditor', {1: (partitions[1], None)}, print)
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
