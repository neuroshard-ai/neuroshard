"""Campaign discovery reads the published bytes, including flat feed objects."""
import json
import io
import os
from pathlib import Path
import sys
import tarfile

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]/'scripts'))
from ordinary_campaign_backend import Backend, REMOTE
from ordinary_cloud import install_bundle
from ordinary_allocation import numerical_runtime, describe
from run_ordinary_campaign import rpc_genesis_commitment
import run_ordinary_campaign
import ordinary_allocation
import ordinary_campaign_backend
import portable_native_trial
from botocore.exceptions import ClientError
sys.path.pop(0)

from neuroshard.dataflow.store import LocalStore
from neuroshard.evolution import expert_source
from neuroshard.evolution.objects import Objects
from neuroshard.evolution.reference_data import save
from neuroshard.evolution.reference_data import identity
from neuroshard.evolution.sharded import retained_objects


def context_backend(tmp_path, bundle):
    from types import SimpleNamespace
    backend = object.__new__(Backend)
    backend.jobs = tmp_path/'jobs'
    backend.jobs.mkdir(exist_ok=True)
    backend.store = Objects(tmp_path/'objects')
    plan = backend.store.put_json({'immutable': 'plan'})
    rows = backend.store.put(b'{"text":"source-backed input"}\n')
    prepared = backend.store.put_json({'plan': plan, 'roles': {'train': {'sha256': rows}}})
    backend.cloud = SimpleNamespace(bundle=bundle)
    return backend, {'work': {'prepared': prepared, 'parent': {'fixture': 'parent'}}}


def test_concurrent_auditors_install_one_immutable_context(tmp_path):
    from concurrent.futures import ThreadPoolExecutor
    calls = []
    backend, job = context_backend(tmp_path, lambda rank, files: calls.append(rank))
    with ThreadPoolExecutor(max_workers=3) as pool:
        contexts = list(pool.map(lambda _: backend.context(job, branch='quality'), range(3)))
    assert len({context['key'] for context in contexts}) == 1
    assert sorted(calls) == list(range(7))
    assert (contexts[0]['directory']/'train.jsonl').read_bytes() == b'{"text":"source-backed input"}\n'


def test_reused_context_never_rewrites_auditors_open_inputs(tmp_path, monkeypatch):
    from concurrent.futures import ThreadPoolExecutor
    backend, job = context_backend(tmp_path, lambda *args: None)
    context = backend.context(job, branch='quality')
    folder = context['directory']
    files = {path: (path.stat().st_ino, path.stat().st_mtime_ns)
             for path in folder.iterdir() if path.name != 'installation.lock'}
    def forbidden(*args, **kwargs):
        raise AssertionError('An installed immutable context must not be rewritten')
    monkeypatch.setattr(ordinary_campaign_backend, 'save', forbidden)
    backend.cloud.bundle = forbidden
    with (folder/'job.json').open('rb') as existing_reader:
        with ThreadPoolExecutor(max_workers=3) as pool:
            list(pool.map(lambda _: backend.context(job, branch='quality'), range(3)))
        assert json.loads(existing_reader.read()) == job
    assert files == {path: (path.stat().st_ino, path.stat().st_mtime_ns) for path in files}
    (folder/'train.jsonl').write_bytes(b'altered input')
    with pytest.raises(ValueError, match='immutable job context'):
        backend.context(job, branch='quality')


def test_backend_failure_keeps_location_without_copying_sensitive_values(tmp_path, monkeypatch):
    def fail(*args):
        raise RuntimeError('secret command or signed URL must stay out of evidence')
    monkeypatch.setattr(ordinary_campaign_backend, 'Backend', fail)
    with pytest.raises(RuntimeError):
        ordinary_campaign_backend.invoke(tmp_path, 2, True, {'phase': 'quality', 'private': 'secret-input'})
    files = list((tmp_path/'backend-failures').glob('*.json'))
    assert len(files) == 1
    raw = files[0].read_text()
    evidence = json.loads(raw)
    assert evidence['exception'] == 'RuntimeError'
    assert evidence['frames'][-1]['function'] == 'fail'
    assert evidence['actor'] == 2 and evidence['audit'] is True
    assert 'secret' not in raw and 'signed URL' not in raw


def metadata_archive(name, raw):
    stream = io.BytesIO()
    with tarfile.open(fileobj=stream, mode='w:gz') as archive:
        member = tarfile.TarInfo(name)
        member.size = len(raw)
        archive.addfile(member, io.BytesIO(raw))
    stream.seek(0)
    return stream


def test_bundle_keeps_concurrent_policy_readers_on_complete_bytes(tmp_path, monkeypatch):
    target = tmp_path/'policy'
    before, after = b'accepted policy', b'new complete policy'*10000
    target.write_bytes(before)
    replace = os.replace
    def observe(temporary, destination):
        # Observe exactly at publication, after writing the new bytes. The old
        # tar extraction truncated this very inode before readers finished.
        assert target.read_bytes() == before
        assert Path(temporary).read_bytes() == after
        replace(temporary, destination)
    monkeypatch.setattr(os, 'replace', observe)
    with target.open('rb') as existing_reader:
        install_bundle(tmp_path, metadata_archive('policy', after))
        assert existing_reader.read() == before
    assert target.read_bytes() == after
    assert not list(tmp_path.glob('.bundle-*'))


def test_interrupted_bundle_preserves_existing_policy(tmp_path, monkeypatch):
    import shutil
    target = tmp_path/'policy'
    target.write_bytes(b'accepted')
    def interrupt(source, destination):
        destination.write(source.read(3))
        raise OSError('interrupted transfer')
    monkeypatch.setattr(shutil, 'copyfileobj', interrupt)
    with pytest.raises(OSError, match='interrupted transfer'):
        install_bundle(tmp_path, metadata_archive('policy', b'replacement'))
    assert target.read_bytes() == b'accepted'
    assert not list(tmp_path.glob('.bundle-*'))


@pytest.mark.parametrize('name', ['../escape', '/escape'])
def test_bundle_refuses_paths_outside_installation(tmp_path, name):
    with pytest.raises(ValueError, match='archive paths'):
        install_bundle(tmp_path, metadata_archive(name, b'bad'))


def test_native_startup_reports_an_exited_node_without_waiting_for_rpc_timeout(tmp_path, monkeypatch):
    from types import SimpleNamespace
    network = object.__new__(portable_native_trial.Network)
    network.home, network.processes, network.urls = tmp_path, [], ['http://unused']
    network.config = {'engine': 'unused', 'nodes': [{'home': str(tmp_path), 'abci': 29952}]}
    children = iter([SimpleNamespace(pid=1, poll=lambda: None), SimpleNamespace(pid=2, poll=lambda: 1)])
    monkeypatch.setattr(portable_native_trial.subprocess, 'Popen', lambda *args, **kwargs: next(children))
    def forbidden(*args, **kwargs):
        raise AssertionError('A failed child must be detected before polling unavailable RPC')
    monkeypatch.setattr(portable_native_trial.client, 'query', forbidden)
    with pytest.raises(RuntimeError, match='node 0 stopped during startup'):
        network.start()


@pytest.mark.parametrize('claim_blocks', [8192, 20000])
def test_final_paid_request_allows_the_configured_native_claim_window(tmp_path, monkeypatch, claim_blocks):
    from types import SimpleNamespace
    from neuroshard.evolution import answering
    from neuroshard.demo import protocol
    coordinator = protocol.Identity('paid-coordinator')
    graph = {'experts': {'learned': {}}}
    state = {'expert_lifecycle': {'serving_graph': graph}, 'chain_id': 'test',
             'manifest': {'params': {'max_claim_blocks': claim_blocks},
                          'expert_lifecycle': {'price_per_token': 1, 'max_tokens': 64}}}
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
            assert isinstance(body['workers'], dict) and body['workers']['0'] == coordinator.public_key
            raise ObservedRequest
        def close(self):
            pass
    monkeypatch.setattr(run_ordinary_campaign, 'Outbox', Box)
    monkeypatch.setattr(answering, 'quote', lambda graph, tokens, price: {'maximum_atoms': 64})
    backend = SimpleNamespace(home=tmp_path, state=lambda: state)
    network = SimpleNamespace(urls=['http://unused'], owners=[coordinator])
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
