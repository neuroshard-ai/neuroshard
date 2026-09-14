"""Bounded tensor/JSON messages over a fixed operated Gloo process group."""
import json
import torch
import torch.distributed as dist


class Wire:
    def __init__(self, rank, world, max_elements=4*1024*4096):
        self.rank, self.world, self.max_elements = rank, world, max_elements
        self.sent_tensor_bytes = 0

    def send(self, value, destination):
        if value.dtype != torch.float32 or value.ndim != 3 or value.numel() > self.max_elements:
            raise ValueError('Unsupported boundary tensor')
        payload = value.detach().to('cpu').contiguous()
        header = torch.tensor(list(payload.shape), dtype=torch.int64)
        dist.send(header, dst=destination)
        dist.send(payload, dst=destination)
        self.sent_tensor_bytes += payload.numel() * 4 + 24

    def receive(self, source, shape, device):
        header = torch.empty(3, dtype=torch.int64)
        dist.recv(header, src=source)
        actual = tuple(header.tolist())
        if actual != tuple(shape) or any(n <= 0 for n in actual) or int(header.prod()) > self.max_elements:
            raise ValueError('Unexpected boundary shape')
        value = torch.empty(actual, dtype=torch.float32)
        dist.recv(value, src=source)
        if not bool(torch.isfinite(value).all()):
            raise ValueError('Nonfinite boundary tensor')
        return value.to(device)

    def sum(self, number):
        value = torch.tensor(number, dtype=torch.float64)
        dist.all_reduce(value)
        return float(value)

    def exchange(self, value):
        # Tensor encoding avoids pickle-based distributed object collectives.
        raw = json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()
        if len(raw) > 2*1024**2:
            raise ValueError('Control message exceeds bound')
        sizes = [torch.zeros(1, dtype=torch.int64) for _ in range(self.world)]
        dist.all_gather(sizes, torch.tensor([len(raw)], dtype=torch.int64))
        length = max(int(s) for s in sizes)
        if not 0 < length <= 2*1024**2:
            raise ValueError('Peer control message exceeds bound')
        payload = torch.zeros(length, dtype=torch.uint8)
        payload[:len(raw)] = torch.frombuffer(bytearray(raw), dtype=torch.uint8)
        received = [torch.empty_like(payload) for _ in sizes]
        dist.all_gather(received, payload)
        return [json.loads(bytes(v[:int(n)].tolist())) for v, n in zip(received, sizes)]
