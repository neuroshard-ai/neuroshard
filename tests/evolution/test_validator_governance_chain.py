"""Owner governance changes who orders transactions, end to end on real CometBFT validators.

Two genesis validators run the settlement application. Owners bond, the steward hands the chain over,
an owner proposes a third validator's consensus key, the owners vote it in, and anyone enacts it.
CometBFT then reports the new validator in its validator set, and it starts signing blocks.
"""

import json
from pathlib import Path
import os
import shutil
import subprocess
import time

import pytest

from neuroshard.inference import optimistic as ledger
from neuroshard.inference import optimistic_network as chain_network
from test_owner_governance import bond_at
from test_optimistic_serving import Account

pytestmark = pytest.mark.skipif(not (os.environ.get('COMETBFT_BINARY') or shutil.which('cometbft')
                                     or (chain_network.REPO / '.neuroshard/tools/cometbft').exists()
                                     ),
                                reason='CometBFT is not installed')


def engine():
    try:
        return chain_network.engine_path()
    except ValueError:
        pytest.skip('CometBFT is not installed')


def submit(config, account, kind, **fields):
    state = chain_network.state(chain_network.url(config))
    nonce = state['accounts'].get(account.public, {'nonce': 0})['nonce']
    account.nonce = nonce
    envelope = account.sign(kind, **fields)
    admitted = chain_network.broadcast(chain_network.url(config), envelope)
    assert admitted['code'] == 0, admitted['log']
    deadline = time.monotonic() + 60
    while time.monotonic() < deadline:
        now = chain_network.state(chain_network.url(config))
        if now['accounts'].get(account.public, {'nonce': 0})['nonce'] > nonce:
            return envelope, now
        time.sleep(0.5)
    raise TimeoutError(kind)


def test_owners_vote_a_third_validator_into_cometbft(tmp_path, monkeypatch):
    from test_optimistic_serving import CHAIN, PARAMS

    steward, alice, bob, user = (Account(n) for n in ('steward', 'alice', 'bob', 'user'))
    terms = {'chain_id': CHAIN, 'model_root': 'ab' * 32, 'shards': 3,
             'allocations': {a.public: 40_000_000 for a in (steward, alice, bob, user)}, 'params': PARAMS}
    node = {'promotion_authority': steward.public, 'approved_models': {}}
    config = chain_network.initialize(tmp_path / 'net', terms, [dict(node), dict(node), dict(node)], base_port=29650,
                                      engine=engine(), block_seconds=0.5)
    # Node 3 starts outside the genesis set: it syncs and validates nothing until owners admit it.
    third = config['validators'][2]
    genesis_path = Path(third['home']) / 'config/genesis.json'
    genesis = json.loads(genesis_path.read_text())
    newcomer = genesis['validators'][2]
    genesis['validators'] = genesis['validators'][:2]
    for member in config['validators']:
        (Path(member['home']) / 'config/genesis.json').write_text(json.dumps(genesis))
    try:
        chain_network.start(config, timeout=120)
        keys = {}
        for account, shard, name, amount in ((alice, 1, 'alice-log', 6_000_000), (bob, 2, 'bob-log', 5_000_000)):
            state = chain_network.state(chain_network.url(config))
            account.nonce = state['accounts'][account.public]['nonce']
            key, envelope = bond_at(account, shard, name, amount, state)
            admitted = chain_network.broadcast(chain_network.url(config), envelope)
            assert admitted['code'] == 0, admitted['log']
            keys[name] = key
        time.sleep(3)
        submit(config, steward, 'governance_handover', approval_ppm=666_667, vote_blocks=40)
        import base64
        consensus_key = base64.b64decode(newcomer['pub_key']['value']).hex()
        envelope, _ = submit(config, user, 'validator_propose', consensus_key=consensus_key, power=1)
        pid = ledger.transaction_id(envelope)
        submit(config, alice, 'model_vote', proposal_id=pid, log_key=keys['alice-log'], approve=True)
        submit(config, bob, 'model_vote', proposal_id=pid, log_key=keys['bob-log'], approve=True)
        _, state = submit(config, user, 'model_enact', proposal_id=pid)
        assert state['validators'] == {consensus_key: 1}
        deadline = time.monotonic() + 60
        while time.monotonic() < deadline:
            listed = chain_network.rpc(chain_network.url(config), 'validators', {'per_page': '10'})['validators']
            if any(row['address'] == newcomer['address'] for row in listed):
                break
            time.sleep(1)
        else:
            raise AssertionError('CometBFT never admitted the voted validator')
        assert len(listed) == 3
        # The admitted validator now signs: the last commit carries its signature.
        deadline = time.monotonic() + 60
        while time.monotonic() < deadline:
            commit = chain_network.rpc(chain_network.url(config), 'commit')['signed_header']['commit']
            if any(sig.get('validator_address') == newcomer['address'] for sig in commit['signatures']):
                break
            time.sleep(1)
        else:
            raise AssertionError('the admitted validator never signed a block')
        roots = {chain_network.state(chain_network.url(config, i))['height'] for i in range(3)}
        assert roots
    finally:
        chain_network.stop(config)
