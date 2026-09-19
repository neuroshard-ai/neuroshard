"""Real pinned TLS peers and adversarial bounded frame delivery."""
import base64
import copy
from concurrent.futures import ThreadPoolExecutor
import hashlib
import threading

import pytest
import torch

from neuroshard.dataflow.store import canonical
from neuroshard.demo import protocol
from neuroshard.evolution import provider_transport as transport
from neuroshard.evolution.sharded.peer_wire import ServingMesh


@pytest.fixture
def peers(tmp_path):
    owners = [protocol.Identity('transport-test-' + str(rank)) for rank in range(3)]
    routing = {'chain_id': 'provider-test', 'job_id': 'a'*64,
               'assignment_root': 'b'*64, 'providers': {}}
    lookup = lambda job: routing if job == routing['job_id'] else None
    boxes, servers, threads = [], [], []
    for rank, owner in enumerate(owners):
        tls, digest = transport.certificate(tmp_path/str(rank), owner)
        box = transport.Mailbox(owner.public_key, lookup, timeout=2)
        server = transport.Server(('127.0.0.1', 0), tls, box)
        routing['providers'][str(rank)] = {'owner': owner.public_key,
            'endpoint': 'https://127.0.0.1:' + str(server.server_port), 'certificate': digest}
        thread = threading.Thread(target=server.serve_forever, kwargs={'poll_interval': .02}, daemon=True)
        thread.start()
        boxes.append(box)
        servers.append(server)
        threads.append(thread)
    clients = [transport.Peer(owner, copy.deepcopy(routing), rank, boxes[rank],
                              timeout=2, allow_private=True) for rank, owner in enumerate(owners)]
    try:
        yield owners, routing, boxes, clients
    finally:
        for client in clients:
            client.close()
        for box, server, thread in zip(boxes, servers, threads):
            box.close()
            server.shutdown()
            server.server_close()
            thread.join(timeout=3)


def signed(peers, raw=b'payload', **changes):
    owners, routing, _, _ = peers
    body = {'format': transport.FORMAT, 'chain_id': routing['chain_id'], 'job_id': routing['job_id'],
            'assignment_root': routing['assignment_root'], 'source': 0, 'destination': 1,
            'members': [0, 1, 2], 'channel': 'control', 'sequence': 0,
            'bytes': len(raw), 'sha256': hashlib.sha256(raw).hexdigest()}
    body.update(changes)
    return owners[0].sign(body)


def post(client, envelope, raw):
    client.request('POST', '/v1/frame', raw, {
        'X-NeuroShard-Frame': base64.b64encode(canonical(envelope)).decode('ascii'),
        'Content-Length': str(len(raw))})
    response = client.getresponse()
    response.read()
    return response.status


def test_pinned_tls_collectives_and_tensors_use_distinct_local_keys(peers):
    _, _, boxes, clients = peers
    meshes = [ServingMesh(client) for client in clients]
    wires = [mesh.group([0, 1, 2]) for mesh in meshes]
    with ThreadPoolExecutor(max_workers=3) as pool:
        for iteration in range(3):
            futures = [pool.submit(wire.exchange, {'rank': rank, 'iteration': iteration})
                       for rank, wire in enumerate(wires)]
            expected = [{'rank': rank, 'iteration': iteration} for rank in range(3)]
            assert [future.result(timeout=10) for future in futures] == [expected]*3
    tensor = torch.arange(24, dtype=torch.float32).reshape(1, 3, 8)
    wires[0].send(tensor, 1)
    assert torch.equal(wires[1].receive(0, tensor.shape, 'cpu'), tensor)
    assert all(box.allocated == 0 for box in boxes)


def test_wrong_certificate_and_private_addresses_fail_before_payload(peers):
    _, routing, boxes, _ = peers
    provider = routing['providers']['1']
    client = transport.PinnedConnection(provider['endpoint'], provider['certificate'])
    with pytest.raises(ValueError, match='public address'):
        post(client, signed(peers), b'payload')
    bad = transport.PinnedConnection(provider['endpoint'], 'f'*64, allow_private=True)
    with pytest.raises(ValueError, match='certificate differs'):
        post(bad, signed(peers), b'payload')
    assert boxes[1].allocated == 0 and not boxes[1].streams


def test_forgeries_wrong_groups_and_assignment_epochs_are_refused(peers):
    owners, routing, boxes, _ = peers
    for change in ({'assignment_root': 'c'*64}, {'chain_id': 'other'}, {'source': 2},
                   {'destination': 2}, {'members': [0, 1]}, {'sequence': True}, {'bytes': 0}):
        with pytest.raises(ValueError):
            boxes[1].begin(signed(peers, **change))
    with pytest.raises(ValueError):
        boxes[1].begin(owners[2].sign(signed(peers)['body']))
    old = signed(peers)
    routing['assignment_root'] = 'c'*64
    with pytest.raises(ValueError, match='assignment epoch'):
        boxes[1].begin(old)
    assert not boxes[1].streams


def test_corrupt_body_and_disconnected_transfer_release_reserved_memory(peers):
    _, routing, boxes, _ = peers
    provider = routing['providers']['1']
    client = transport.PinnedConnection(provider['endpoint'], provider['certificate'], allow_private=True)
    try:
        assert post(client, signed(peers), b'corrupt') == 400
        assert boxes[1].allocated == 0
        assert post(client, signed(peers), b'payload') == 200
        assert boxes[1].receive(routing, [0, 1, 2], 0, 1, 'control', 0) == b'payload'
    finally:
        client.close()


def test_acknowledgement_retry_is_idempotent_before_and_after_consumption(peers):
    _, routing, boxes, _ = peers
    provider = routing['providers']['1']
    client = transport.PinnedConnection(provider['endpoint'], provider['certificate'], allow_private=True)
    frame = signed(peers)
    try:
        assert post(client, frame, b'payload') == 200
        assert post(client, frame, b'payload') == 200
        assert boxes[1].allocated == len(b'payload')
        assert boxes[1].receive(routing, [0, 1, 2], 0, 1, 'control', 0) == b'payload'
        assert post(client, frame, b'payload') == 200
        assert boxes[1].allocated == 0
        assert post(client, signed(peers, b'changed'), b'changed') == 400
        assert post(client, signed(peers, sequence=1), b'payload') == 200
        assert boxes[1].receive(routing, [0, 1, 2], 0, 1, 'control', 1) == b'payload'
    finally:
        client.close()


def test_capacity_includes_unfinished_bodies_and_epoch_change_cancels(peers):
    _, routing, boxes, _ = peers
    box = boxes[1]
    box.byte_limit = 8
    body, key, reserved = box.begin(signed(peers))
    assert reserved and box.allocated == 7
    with pytest.raises(transport.Unavailable, match='byte capacity'):
        box.begin(signed(peers, channel='tensor'))
    routing['assignment_root'] = 'd'*64
    with pytest.raises(ValueError, match='assignment changed'):
        box.finish(body, key, b'payload')
    assert box.allocated == 0
    with pytest.raises(transport.Unavailable, match='replaced'):
        box.receive({**routing, 'assignment_root': 'b'*64}, [0, 1, 2], 0, 1, 'control', 0)
    box.retire(routing['job_id'], 'b'*64)
    assert not box.streams


def test_unavailable_local_assignment_is_not_treated_as_authority(peers):
    _, _, boxes, _ = peers
    def unavailable(_job):
        raise transport.Unavailable('Local validating node unavailable')
    boxes[1].lookup = unavailable
    with pytest.raises(transport.Unavailable, match='validating node unavailable'):
        boxes[1].begin(signed(peers))
    assert boxes[1].allocated == 0


def test_retiring_after_reply_preserves_last_ack_without_reopening_stream(peers):
    _, routing, boxes, _ = peers
    box = boxes[1]
    body, key, _ = box.begin(signed(peers))
    box.finish(body, key, b'payload')
    assert box.receive(routing, [0, 1, 2], 0, 1, 'control', 0) == b'payload'
    box.retire(routing['job_id'], routing['assignment_root'])
    assert box.begin(signed(peers))[2] is False
    with pytest.raises(transport.Unavailable, match='retired'):
        box.begin(signed(peers, sequence=1))
    assert box.allocated == 0 and not box.streams


def test_tensor_decoder_rejects_unexpected_shape_and_nonfinite_values(peers):
    _, _, _, clients = peers
    sender, receiver = [ServingMesh(client).group([0, 1, 2]) for client in clients[:2]]
    tensor = torch.zeros(1, 2, 3)
    sender.send(tensor, 1)
    with pytest.raises(ValueError, match='shape or length'):
        receiver.receive(0, (1, 3, 2), 'cpu')
    tensor[0, 0, 0] = float('nan')
    sender.send(tensor, 1)
    with pytest.raises(ValueError, match='Nonfinite'):
        receiver.receive(0, (1, 2, 3), 'cpu')
