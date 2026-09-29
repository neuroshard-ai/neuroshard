"""Granite pipeline across processes: each owner loads only its shard and keeps its own cache.

Owner 0 drives. For every step it broadcasts a two-number command, embeds the
tokens and sends the hidden state around the ring; the last owner returns only
the last position, which owner 0 scores. Boundary tensors travel as exact bf16
bytes. A failed owner breaks the group: the driver's committed tokens let a
relaunched group rebuild every cache by replaying the same steps, so the
continuation is the one an uninterrupted run would have produced.
"""
import hashlib
import json
import os
import time
from pathlib import Path

import torch
import torch.distributed as dist

from . import granite

FORWARD, RESET, STOP = 1, 2, 3


class Ring:
    def __init__(self, rank, world, hidden_size, max_tokens):
        if world < 2 or not 0 <= rank < world:
            raise ValueError('a pipeline ring needs at least two owners')
        self.rank, self.world, self.hidden, self.max_tokens = rank, world, hidden_size, max_tokens
        self.sent_bytes = self.received_bytes = 0
        self.trace = None

    def send(self, value, destination):
        if (value.dtype != torch.bfloat16 or value.ndim != 3 or value.shape[0] != 1
                or value.shape[2] != self.hidden or not 0 < value.shape[1] <= self.max_tokens):
            raise ValueError('unsupported boundary tensor')
        payload = value.detach().contiguous().view(torch.uint8)
        if self.trace is not None:
            self.trace.append(hashlib.sha256(payload.numpy().tobytes()).hexdigest())
        dist.send(payload, dst=destination)
        self.sent_bytes += payload.numel()

    def receive(self, source, length):
        if not 0 < length <= self.max_tokens:
            raise ValueError('unexpected boundary length')
        payload = torch.empty((1, length, 2 * self.hidden), dtype=torch.uint8)
        dist.recv(payload, src=source)
        self.received_bytes += payload.numel()
        value = payload.view(torch.bfloat16)
        if not bool(torch.isfinite(value).all()):
            raise ValueError('nonfinite boundary tensor')
        return value


def command(op, length=0):
    header = torch.tensor([op, length], dtype=torch.int64)
    dist.broadcast(header, src=0)
    return header.tolist()


def serve(partition, ring, fail_at=None):
    """Owner loop for ranks after 0; ``fail_at`` simulates an outage at that forward step."""
    from transformers import DynamicCache

    cache, steps = DynamicCache(), 0
    while True:
        op, length = command(0)
        if op == STOP:
            return steps
        if op == RESET:
            cache = DynamicCache()
            continue
        if op != FORWARD:
            raise ValueError('unknown pipeline command')
        steps += 1
        if fail_at is not None and steps == fail_at:
            os._exit(17)
        hidden = ring.receive(ring.rank - 1, length)
        mask = torch.ones((1, cache.get_seq_length() + length), dtype=torch.long)
        with torch.inference_mode():
            out = partition(hidden, mask, cache)
        last = ring.rank == ring.world - 1
        ring.send(out[:, -1:] if last else out, (ring.rank + 1) % ring.world)


class Driver:
    def __init__(self, partition, ring):
        from transformers import DynamicCache

        if partition.rank != 0 or ring.rank != 0:
            raise ValueError('owner 0 drives the pipeline')
        self.partition, self.ring, self.cache_type = partition, ring, DynamicCache
        self.cache = DynamicCache()

    def reset(self):
        command(RESET)
        self.cache = self.cache_type()

    def step(self, tokens, mask):
        command(FORWARD, tokens.shape[1])
        with torch.inference_mode():
            hidden = self.partition(self.partition.embed(tokens), mask, self.cache)
        self.ring.send(hidden, 1)
        back = self.ring.receive(self.ring.world - 1, 1)
        with torch.inference_mode():
            return self.partition.logits(back)

    def stop(self):
        command(STOP)


def replay(step, ids, committed):
    """Rebuild caches with the exact steps that produced ``committed``; returns the next step's input."""
    current, length = ids, ids.shape[1]
    for token in committed:
        step(current, torch.ones((1, length), dtype=torch.long))
        current, length = torch.tensor([[token]]), length + 1
    return current, length


def resume(step, ids, committed, max_new_tokens, eos_ids, on_token=None):
    """Greedy decoding that continues after ``committed`` tokens, reporting each new one."""
    tokens = list(committed)
    if tokens and (tokens[-1] in eos_ids or len(tokens) >= max_new_tokens):
        return tokens
    current, length = replay(step, ids, committed)
    while len(tokens) < max_new_tokens:
        logits = step(current, torch.ones((1, length), dtype=torch.long))
        token = int(logits[0, -1].float().argmax())
        tokens.append(token)
        if on_token:
            on_token(tokens)
        if token in eos_ids:
            break
        current, length = torch.tensor([[token]]), length + 1
    return tokens


def run_owner(config_dir, shards_dir, rank, world, address, port, job_path, result_path, *, timeout=60):
    """One owner process: load only this shard, join the ring, then drive (owner 0) or serve."""
    from datetime import timedelta

    job = json.loads(Path(job_path).read_text())
    torch.set_num_threads(job.get('threads', 1))
    config = granite.load_config(config_dir)
    started = time.monotonic()
    partition, manifest = granite.load_partition(config, shards_dir, rank)
    loaded = time.monotonic() - started
    dist.init_process_group('gloo', init_method=f'tcp://{address}:{port}', rank=rank, world_size=world,
                            timeout=timedelta(seconds=timeout))
    ring = Ring(rank, world, config.hidden_size, job['max_tokens'])
    result = {'rank': rank, 'shard_sha256': manifest['sha256'], 'resident_bytes': partition.resident_bytes(),
              'load_seconds': loaded}
    try:
        if rank == 0:
            driver = Driver(partition, ring)
            result['outputs'] = outputs = []
            for request in job['requests']:
                driver.reset()
                began = time.monotonic()
                result['inflight'] = {'id': request['id'], 'token_ids': list(request.get('committed', []))}
                tokens = resume(driver.step, torch.tensor([request['input_ids']]), request.get('committed', []),
                                request['max_new_tokens'], set(job['eos_ids']),
                                lambda tokens: result['inflight'].update(token_ids=list(tokens)))
                outputs.append({'id': request['id'], 'token_ids': tokens, 'seconds': time.monotonic() - began})
                result.pop('inflight')
            driver.stop()
        else:
            fail = job.get('fail')
            result['steps'] = serve(partition, ring, fail['at'] if fail and fail['rank'] == rank else None)
        result['completed'] = True
    except Exception as error:
        result.update(completed=False, error=f'{type(error).__name__}: {error}')
    finally:
        import resource

        result.update(sent_bytes=ring.sent_bytes, received_bytes=ring.received_bytes,
                      peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024)
        Path(result_path).write_text(json.dumps(result, indent=2) + '\n')
        if result.get('completed'):
            dist.destroy_process_group()
    return result
