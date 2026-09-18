"""Campaign discovery reads the published bytes, including flat feed objects."""
import json
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]/'scripts'))
from ordinary_campaign_backend import Backend, REMOTE
from ordinary_allocation import numerical_runtime, describe
from run_ordinary_campaign import rpc_genesis_commitment
import run_ordinary_campaign
import ordinary_allocation
from botocore.exceptions import ClientError
sys.path.pop(0)

from neuroshard.dataflow.store import LocalStore
from neuroshard.evolution import expert_source
from neuroshard.evolution.objects import Objects
from neuroshard.evolution.reference_data import save
from neuroshard.evolution.reference_data import identity
from neuroshard.evolution.sharded import retained_objects


@pytest.mark.parametrize('claim_blocks', [8192, 20000])
def test_final_paid_request_allows_the_configured_native_claim_window(tmp_path, monkeypatch, claim_blocks):
    from types import SimpleNamespace
    from neuroshard.evolution import answering
    graph = {'experts': {'learned': {}}}
    state = {'expert_lifecycle': {'serving_graph': graph}, 'chain_id': 'test',
             'manifest': {'params': {'max_claim_blocks': claim_blocks}}}
    class ObservedRequest(Exception):
        pass
    class Box:
        def __init__(self, *args):
            pass
        def send(self, logical, kind, **body):
            assert kind == 'infer_expert'
            # This is the lifecycle's actual admission bound. The prior fixed
            # 10,000-block request fails it on the ordinary GPU network.
            assert claim_blocks+1 <= body['expires_in'] <= 100000
            assert body['max_price'] == 64 and len(body['workers']) == 4
            raise ObservedRequest
        def close(self):
            pass
    monkeypatch.setattr(run_ordinary_campaign, 'Outbox', Box)
    monkeypatch.setattr(answering, 'quote', lambda graph, tokens, price: {'maximum_atoms': 64})
    backend = SimpleNamespace(home=tmp_path, state=lambda: state)
    network = SimpleNamespace(urls=['http://unused'], owners=[object()])
    with pytest.raises(ObservedRequest):
        run_ordinary_campaign.paid_inference(backend, network)


def test_rpc_height_normalization_keeps_every_validators_app_state_pinned(monkeypatch):
    genesis = {'initial_height': '0', 'chain_id': 'frozen', 'app_state': {'model': 'accepted'}}
    normalized = {**genesis, 'initial_height': '1'}
    observed = {'one': normalized, 'two': normalized}
    monkeypatch.setattr(run_ordinary_campaign.client, 'rpc', lambda url, method: {'genesis': observed[url]})
    assert rpc_genesis_commitment(genesis, observed) == identity(normalized)
    assert genesis['initial_height'] == '0'
    observed['two'] = {**normalized, 'app_state': {'model': 'substituted'}}
    with pytest.raises(ValueError, match='committed genesis'):
        rpc_genesis_commitment(genesis, observed)
    observed['two'] = {**normalized, 'initial_height': '2'}
    with pytest.raises(ValueError, match='committed genesis'):
        rpc_genesis_commitment(genesis, observed)


def test_distinct_owners_require_matching_arithmetic_not_matching_hostnames():
    left = {'host': 'owner-one', 'gpu': 'NVIDIA A10G', 'torch': 'pinned', 'threads': 2}
    right = {**left, 'host': 'owner-two'}
    assert numerical_runtime(left) == numerical_runtime(right)
    assert numerical_runtime(left) != numerical_runtime({**right, 'gpu': 'NVIDIA L40S'})
    assert numerical_runtime(left) != numerical_runtime({**right, 'threads': 1})


def test_new_owner_discovery_retries_visibility_and_requires_the_whole_inventory(monkeypatch):
    calls = []
    class EC2:
        def describe_instances(self, **request):
            calls.append(request)
            if len(calls) == 1:
                raise ClientError({'Error': {'Code': 'InvalidInstanceID.NotFound'}}, 'DescribeInstances')
            ids = ['i-one'] if len(calls) == 2 else ['i-one', 'i-two']
            return {'Reservations': [{'Instances': [{'InstanceId': key} for key in ids]}]}
    monkeypatch.setattr(ordinary_allocation.time, 'sleep', lambda seconds: None)
    assert len(describe(EC2(), ['i-two', 'i-one'])) == 2
    assert len(calls) == 3 and all(row == {'InstanceIds': ['i-one', 'i-two']} for row in calls)


def test_public_metadata_covers_both_stores_and_monotonic_discovery(tmp_path, monkeypatch):
    backend = object.__new__(Backend)
    backend.home = tmp_path
    backend.store = Objects(tmp_path/'objects')
    backend.transport = LocalStore(tmp_path/'feed')
    backend.public_feed = LocalStore(tmp_path/'reader')
    backend.preparation = None
    source = {'repo': 'test/ordinary', 'revision': 'a'*40, 'split': 'train',
              'license': 'Apache-2.0', 'role': 'train'}
    rows = [{'messages': [{'role': 'user', 'content': question},
                          {'role': 'assistant', 'content': 'A'}]} for question in ('One?', 'Two?')]
    first = expert_source.publish_window(backend.transport, source, 0, rows[:1])
    second = expert_source.publish_window(backend.transport, source, 1, rows[1:])
    heads = [expert_source.append(backend.transport, None, [first])]
    heads.append(expert_source.append(backend.transport, heads[0], [second]))
    backend.freeze = {'feed_heads': heads}
    policy = backend.store.put_json({'actual_policy': 'metadata'})
    remote, public, published = {}, {}, []

    class Owner:
        def bundle(self, physical, files):
            assert physical == 3
            remote.update({REMOTE+'/'+key: raw for key, raw in files.items()})

    backend.cloud = Owner()

    def publish(physical, inventory, destination):
        for key, spec in inventory.items():
            public[key] = remote[spec['path']]
        published.append(set(inventory))
        save(destination, {'objects': inventory, 'all_hashes_verified': True})

    backend.publish = publish
    backend.publish_metadata()
    assert policy in public and set(path.name for path in backend.transport.root.iterdir()) <= set(public)
    backend.publish_metadata()
    assert len(published) == 1

    def fetch(key, size, urls, destination, **kwargs):
        raw = public[key]
        assert len(raw) == size
        destination.write_bytes(raw)
        return {'bytes': len(raw)}

    monkeypatch.setattr(retained_objects, 'restore', fetch)
    backend.advance_feed(0)
    assert backend.feed(source, 0, 1) == rows[:1]
    with pytest.raises(OSError, match='entire'):
        backend.feed(source, 0, 2)
    backend.advance_feed(1)
    assert backend.feed(source, 0, 2) == rows
    assert json.loads((tmp_path/'feed-head.json').read_bytes())['entry'] == 1
    descriptor = json.loads(backend.transport.get(second))
    public[descriptor['records']['sha256']] = b'x'*descriptor['records']['bytes']
    backend.advance_feed(1)
    with pytest.raises(ValueError, match='bounded commitment'):
        backend.feed(source, 0, 2)
