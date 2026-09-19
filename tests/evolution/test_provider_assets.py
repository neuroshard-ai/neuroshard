"""A joining provider fetches only its hash-bound partition, without SSH."""
import copy
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
import json
import shutil
import threading

import pytest

from neuroshard.evolution import provider_assets
from neuroshard.evolution.reference_data import identity, sha256
from neuroshard.evolution.sharded.peer_wire import SOURCES
from test_graph_execution import prepare_graph, SOURCE


@pytest.fixture
def published(tmp_path):
    source = tmp_path/'source'
    source.mkdir()
    prepare_graph(source)
    graph, profile = [json.loads((source/name).read_bytes()) for name in ('graph.json', 'profile.json')]
    profile['sources'].update({name: sha256(SOURCE/name) for name in SOURCES})
    graph['executor_root'] = identity(profile)
    mirror = tmp_path/'mirror'
    mirror.mkdir()
    inventory = {}
    for rank in range(5):
        for name, spec in provider_assets.plan(graph, rank).items():
            path = source/name
            assert sha256(path) == spec['sha256']
            inventory[spec['sha256']] = path.stat().st_size
            shutil.copyfile(path, mirror/spec['sha256'])
    class Quiet(SimpleHTTPRequestHandler):
        def log_message(self, *_args):
            pass
    server = ThreadingHTTPServer(('127.0.0.1', 0), partial(Quiet, directory=str(mirror)))
    thread = threading.Thread(target=server.serve_forever, kwargs={'poll_interval': .02}, daemon=True)
    thread.start()
    try:
        yield graph, profile, inventory, 'http://127.0.0.1:'+str(server.server_port), mirror
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=3)


def test_fresh_owners_restore_disjoint_model_partitions(published, tmp_path):
    graph, profile, inventory, url, _ = published
    for rank in range(5):
        target = tmp_path/f'owner-{rank}'
        result = provider_assets.prepare(graph, profile, rank, target, SOURCE, inventory, [url],
                                         max_bytes=10*1024**2, max_seconds=10)
        assert set(result['files']) == set(provider_assets.plan(graph, rank))
        assert all(sha256(target/name) == row['sha256'] for name, row in result['files'].items())
        assert result['bytes'] == sum(path.stat().st_size for path in target.rglob('*') if path.is_file())
        if rank >= 3:
            assert not (target/'interpreter').exists()


def test_local_disk_limits_and_source_change_precede_transfers(published, tmp_path):
    graph, profile, inventory, url, _ = published
    def forbidden(*args, **kwargs):
        raise AssertionError('Preparation must reject before downloading tensors')
    with pytest.raises(ValueError, match='disk budget'):
        provider_assets.prepare(graph, profile, 0, tmp_path/'small', SOURCE, inventory, [url],
                                max_bytes=1, restore=forbidden)
    changed = copy.deepcopy(profile)
    changed['sources'][SOURCES[0]] = '0'*64
    graph['executor_root'] = identity(changed)
    with pytest.raises(ValueError, match='different installed source'):
        provider_assets.prepare(graph, changed, 0, tmp_path/'bad-source', SOURCE, inventory, [url],
                                max_bytes=10*1024**2, restore=forbidden)


def test_missing_and_corrupted_remote_bytes_never_install(published, tmp_path):
    graph, profile, inventory, url, mirror = published
    spec = next(iter(provider_assets.plan(graph, 3).values()))
    (mirror/spec['sha256']).write_bytes(b'corrupted')
    from neuroshard.evolution.sharded.retained_objects import UnavailableObject
    with pytest.raises(UnavailableObject):
        provider_assets.prepare(graph, profile, 3, tmp_path/'bad', SOURCE, inventory, [url],
                                max_bytes=10*1024**2, max_seconds=5)
    assert not (tmp_path/'bad'/next(iter(provider_assets.plan(graph, 3)))).exists()
