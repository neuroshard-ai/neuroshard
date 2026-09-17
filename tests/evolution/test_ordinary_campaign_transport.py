"""Campaign discovery reads the published bytes, including flat feed objects."""
import json
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]/'scripts'))
from ordinary_campaign_backend import Backend, REMOTE
from ordinary_allocation import numerical_runtime, describe
import ordinary_allocation
from botocore.exceptions import ClientError
sys.path.pop(0)

from neuroshard.dataflow.store import LocalStore
from neuroshard.evolution import expert_source
from neuroshard.evolution.objects import Objects
from neuroshard.evolution.reference_data import save
from neuroshard.evolution.sharded import retained_objects


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
