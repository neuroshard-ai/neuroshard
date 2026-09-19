"""The existing serving wire interface over authenticated provider endpoints."""
import json
import math
import struct

import torch

from neuroshard.dataflow.store import canonical
from ..provider_transport import MAX_CONTROL, MAX_TENSOR

SOURCES = ('src/neuroshard/evolution/provider_transport.py',
           'src/neuroshard/evolution/sharded/peer_wire.py',
           'src/neuroshard/evolution/sharded/branch_groups.py')


class PeerWire:
    def __init__(self, peer, members):
        if members != sorted(set(members)) or peer.rank not in members:
            raise ValueError('Require the ordered native communication group')
        self.peer, self.members = peer, tuple(members)
        self.rank, self.world = members.index(peer.rank), len(members)
        self.max_elements = (MAX_TENSOR - 24) // 4
        self.sent_tensor_bytes = 0

    def send(self, value, destination):
        if (value.dtype != torch.float32 or value.ndim != 3 or not 0 < value.numel() <= self.max_elements
                or type(destination) is not int or not 0 <= destination < self.world):
            raise ValueError('Unsupported provider boundary tensor or destination')
        payload = value.detach().to('cpu').contiguous()
        raw = struct.pack('<QQQ', *payload.shape) + payload.numpy().tobytes(order='C')
        self.peer.send(self.members, self.members[destination], 'tensor', raw)
        self.sent_tensor_bytes += len(raw)

    def receive(self, source, shape, device):
        if type(source) is not int or not 0 <= source < self.world:
            raise ValueError('Unknown provider tensor source')
        raw = self.peer.receive(self.members, self.members[source], 'tensor')
        if len(raw) < 24:
            raise ValueError('Truncated provider tensor header')
        actual = struct.unpack('<QQQ', raw[:24])
        count = math.prod(actual)
        if (actual != tuple(shape) or any(side <= 0 for side in actual)
                or count > self.max_elements or len(raw) != 24 + count*4):
            raise ValueError('Provider tensor differs from its expected shape or length')
        value = torch.frombuffer(bytearray(raw[24:]), dtype=torch.float32).reshape(actual)
        if not bool(torch.isfinite(value).all()):
            raise ValueError('Nonfinite provider tensor')
        return value.to(device)

    def exchange(self, value):
        raw = canonical(value)
        if not 0 < len(raw) <= MAX_CONTROL:
            raise ValueError('Provider control message exceeds its bound')
        for rank in self.members:
            if rank != self.peer.rank:
                self.peer.send(self.members, rank, 'control', raw)
        values = []
        for rank in self.members:
            encoded = raw if rank == self.peer.rank else self.peer.receive(self.members, rank, 'control')
            def reject_constant(_value):
                raise ValueError('Nonfinite provider JSON')
            def pairs(items):
                result = {}
                for key, item in items:
                    if key in result:
                        raise ValueError('Duplicate provider JSON key')
                    result[key] = item
                return result
            values.append(json.loads(encoded, parse_constant=reject_constant, object_pairs_hook=pairs))
        return values


class ServingMesh:
    def __init__(self, peer):
        self.peer, self.rank = peer, peer.rank
        self.members = sorted(map(int, peer.routing['providers']))
        if self.members != list(range(len(self.members))):
            raise ValueError('The complete answering graph requires every contiguous logical owner')
        self.world = len(self.members)
        self.groups = {}

    def group(self, members):
        if any(rank not in self.members for rank in members):
            raise ValueError('A communication group references an unassigned owner')
        if self.rank not in members:
            return None
        key = tuple(members)
        if key not in self.groups:
            self.groups[key] = PeerWire(self.peer, list(members))
        return self.groups[key]
