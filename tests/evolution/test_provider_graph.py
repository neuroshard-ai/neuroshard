"""Five separately keyed processes execute real model partitions over HTTPS."""
from datetime import timedelta
import json
import os
from pathlib import Path
import threading
import time

import torch.distributed as dist
import torch.multiprocessing as mp

from neuroshard.demo import protocol
from neuroshard.evolution import answering
from neuroshard.evolution.provider_runtime import execute
from neuroshard.evolution import provider_transport as transport
from neuroshard.evolution.reference_data import identity, save, sha256
from neuroshard.evolution.sharded.graph_execution import GraphNetwork
from neuroshard.evolution.sharded.peer_wire import ServingMesh, SOURCES
from test_graph_execution import prepare_graph, QUESTIONS, SOURCE
from test_complete_answering import complete


def owner(rank, directory):
    home = Path(directory)
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'test'
    key = protocol.Identity.load_or_create(home/f'owner-{rank}'/'identity')
    tls, fingerprint = transport.certificate(home/f'owner-{rank}', key)
    routing = {}
    box = transport.Mailbox(key.public_key, lambda _job: routing, timeout=20)
    server = transport.Server(('127.0.0.1', 0), tls, box)
    thread = threading.Thread(target=server.serve_forever, kwargs={'poll_interval': .02}, daemon=True)
    thread.start()
    row = {'owner': key.public_key, 'certificate': fingerprint,
           'endpoint': 'https://127.0.0.1:' + str(server.server_port)}
    temporary = home/f'endpoint-{rank}.pending'
    save(temporary, row)
    temporary.replace(home/f'endpoint-{rank}.json')
    paths = [home/f'endpoint-{index}.json' for index in range(5)]
    deadline = time.monotonic() + 60
    while not all(path.exists() for path in paths):
        if time.monotonic() >= deadline:
            raise RuntimeError('Provider fixture failed to start')
        time.sleep(.02)
    routing.update(chain_id='provider-neural-fixture', job_id='a'*64, assignment_root='b'*64,
                   providers={str(index): json.loads(path.read_bytes()) for index, path in enumerate(paths)})
    peer = transport.Peer(key, routing, rank, box, timeout=20, allow_private=True)
    graph, profile = [json.loads((home/name).read_bytes()) for name in ('graph.json', 'profile.json')]
    kwargs = dict(objects=home/'objects', interpreter=home/'interpreter', seed=home/'seed',
                  source_home=SOURCE, rank=rank)
    try:
        assert not dist.is_initialized()
        network = GraphNetwork(graph, profile, mesh=ServingMesh(peer), **kwargs)
        questions = (QUESTIONS if 'answering' not in graph else [
            [{'role': 'user', 'content': 'What is word7?'}],
            [{'role': 'user', 'content': 'Remember word11.'},
             {'role': 'assistant', 'content': 'Understood.'},
             {'role': 'user', 'content': 'What was the word?'}]])
        actual = []
        for question in questions:
            from neuroshard.evolution.expert_lifecycle import inference_terms
            request, _, _ = inference_terms(graph, question, 4, 1)
            job = {'id': routing['job_id'], 'graph': graph, 'request': request,
                'hosting': routing['assignment_root'],
                'workers': {index: row['owner'] for index, row in routing['providers'].items()}}
            execution = execute(job, network, peer)
            assert len(execution['submission']['workers']) == 5
            assert execution['claim']['record_root'] == identity(execution['transcript'])
            actual.append(execution['transcript']['result'])
        assert not dist.is_initialized()
        # Only the oracle below has a fixed Gloo group. Public serving above
        # used distinct provider keys, certificate pins and binary HTTPS frames.
        dist.init_process_group('gloo', init_method='file://' + str(home/'oracle-rendezvous'),
            rank=rank, world_size=5, timeout=timedelta(seconds=60))
        oracle = GraphNetwork(graph, profile, **kwargs)
        expected = [oracle.answer(question, 4) for question in questions]
        assert actual == expected
        assert network.shard.resident_parameters == oracle.shard.resident_parameters
        save(home/f'provider-result-{rank}.json', {'results': actual,
            'owned_parameters': network.shard.resident_parameters})
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()
        peer.close()
        box.close()
        server.shutdown()
        server.server_close()
        thread.join(timeout=3)


def run(tmp_path):
    profile = json.loads((tmp_path/'profile.json').read_bytes())
    profile['sources'].update({name: sha256(SOURCE/name) for name in SOURCES})
    graph = json.loads((tmp_path/'graph.json').read_bytes())
    graph['executor_root'] = identity(profile)
    save(tmp_path/'profile.json', profile)
    save(tmp_path/'graph.json', graph)
    running = mp.spawn(owner, args=(str(tmp_path),), nprocs=5, join=False)
    deadline = time.monotonic() + 120
    try:
        while not running.join(timeout=1):
            if time.monotonic() >= deadline:
                raise RuntimeError('Bounded real provider graph test timed out')
    finally:
        for process in running.processes:
            if process.is_alive():
                process.terminate()
            process.join(timeout=5)
    results = [json.loads((tmp_path/f'provider-result-{rank}.json').read_bytes()) for rank in range(5)]
    assert all(row['results'] == results[0]['results'] for row in results)
    assert len({json.loads((tmp_path/f'endpoint-{rank}.json').read_bytes())['owner'] for rank in range(5)}) == 5


def test_public_mesh_matches_fixed_group_neural_execution(tmp_path):
    prepare_graph(tmp_path)
    run(tmp_path)


def test_complete_ordinary_and_multiturn_service_uses_provider_mesh(complete):
    home, graph, bound, config, store, _ = complete
    # A transport migration changes the executor and graph commitment, while
    # retaining the exact trained tensors, router and generation rules.
    profile = json.loads((home/'profile.json').read_bytes())
    profile['sources'].update({name: sha256(SOURCE/name) for name in SOURCES})
    graph['executor_root'] = identity(profile)
    config['graph'] = config['learned']['graph'] = identity(graph)
    migrated = answering.attach(graph, config, store)
    save(home/'graph.json', migrated)
    run(home)
