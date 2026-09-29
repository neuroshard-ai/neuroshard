"""Several episodes in flight on one Granite owner ring, each computed exactly as if alone.

Every message carries a header naming its stream, so owners interleave episodes
while keeping one cache and one arm setting per stream. Each step runs the same
batch-one operations on the same shapes as sequential serving; only the owners'
idle time is filled. Owner 0 serializes its own computation, sends in order and
matches results in order: every owner processes messages in arrival order.
"""
import queue
import threading
import time

import torch
import torch.distributed as dist

FORWARD, FEATURE, RESET, CROP, STOP = 1, 2, 3, 4, 5


class Links:
    """In-band headers and exact bf16 payloads between ring neighbours."""

    def __init__(self, rank, world, hidden_size, max_tokens):
        if world < 3 or not 0 <= rank < world:
            raise ValueError('stream serving needs at least three owners')
        self.rank, self.world, self.hidden, self.max_tokens = rank, world, hidden_size, max_tokens
        self.sent_bytes = self.received_bytes = 0

    def send(self, header, payload, destination):
        dist.send(torch.tensor(header, dtype=torch.int64), dst=destination)
        if payload is not None:
            if (payload.dtype != torch.bfloat16 or payload.shape[0] != 1 or payload.shape[2] != self.hidden
                    or not 0 < payload.shape[1] <= self.max_tokens):
                raise ValueError('unsupported boundary tensor')
            raw = payload.detach().contiguous().view(torch.uint8)
            dist.send(raw, dst=destination)
            self.sent_bytes += raw.numel()

    def receive(self, source):
        header = torch.empty(4, dtype=torch.int64)
        dist.recv(header, src=source)
        stream, op, length, arg = header.tolist()
        payload = None
        if op in (FORWARD, FEATURE):
            if not 0 < length <= self.max_tokens:
                raise ValueError('unexpected boundary length')
            raw = torch.empty((1, length, 2 * self.hidden), dtype=torch.uint8)
            dist.recv(raw, src=source)
            self.received_bytes += raw.numel()
            payload = raw.view(torch.bfloat16)
            if not bool(torch.isfinite(payload).all()):
                raise ValueError('nonfinite boundary tensor')
        return [stream, op, length, arg], payload


def serve(partition, links, adapter=None):
    """Owner loop for ranks after 0: per-stream caches and arm settings, messages in arrival order."""
    from transformers import DynamicCache

    streams, steps = {}, 0
    last = links.rank == links.world - 1
    destination = (links.rank + 1) % links.world
    while True:
        header, payload = links.receive(links.rank - 1)
        stream, op, length, arg = header
        if op == STOP:
            links.send(header, None, destination)
            return steps
        if op == RESET:
            streams[stream] = {'cache': DynamicCache(), 'arm': bool(arg)}
            links.send(header, None, destination)
            continue
        if op == CROP:
            streams[stream]['cache'].crop(arg)
            links.send(header, None, destination)
            continue
        if op not in (FORWARD, FEATURE):
            raise ValueError('unknown stream command')
        # The selection feature always comes from the parent, before its episode's reset.
        state = streams[stream] if op == FORWARD else None
        if adapter is not None:
            adapter.set(bool(state and state['arm']))
        with torch.inference_mode():
            if op == FORWARD:
                steps += 1
                mask = torch.ones((1, state['cache'].get_seq_length() + length), dtype=torch.long)
                out = partition(payload, mask, state['cache'])
            else:
                out = partition(payload, None, DynamicCache())
        if last:
            out = out[:, -1:]
        links.send([stream, op, out.shape[1], arg], out, destination)


class StreamDriver:
    """Owner 0: episode threads submit steps; one receiver thread returns results in submission order."""

    def __init__(self, partition, links):
        if partition.rank != 0 or links.rank != 0:
            raise ValueError('owner 0 drives the streams')
        self.partition, self.links = partition, links
        self.compute, self.sending = threading.Lock(), threading.Lock()
        self.pending, self.failure = queue.Queue(), None
        self.in_flight, self.peak_in_flight, self.counter = 0, 0, threading.Lock()
        self.receiver = threading.Thread(target=self._receive, daemon=True)
        self.receiver.start()

    def _receive(self):
        try:
            while True:
                header, payload = self.links.receive(self.links.world - 1)
                slot = self.pending.get()
                slot['result'] = (header, payload)
                slot['done'].set()
                if header[1] == STOP:
                    return
        except Exception as error:
            self.failure = error
            while not self.pending.empty():
                self.pending.get()['done'].set()

    def submit(self, header, payload=None):
        slot = {'done': threading.Event()}
        with self.counter:
            self.in_flight += 1
            self.peak_in_flight = max(self.peak_in_flight, self.in_flight)
        with self.sending:
            if self.failure is not None:
                raise RuntimeError(f'stream ring failed: {self.failure}')
            self.pending.put(slot)
            self.links.send(header, payload, 1)
        while not slot['done'].wait(1):
            if self.failure is not None:
                break
        with self.counter:
            self.in_flight -= 1
        if 'result' not in slot:
            raise RuntimeError(f'stream ring failed: {self.failure}')
        return slot['result']

    def stream(self, index):
        return Stream(self, index)

    def stop(self):
        self.submit([0, STOP, 0, 0])


class Stream:
    """One episode's view of the ring with the interface of ``granite_serving.ServingDriver``."""

    def __init__(self, driver, index):
        from transformers import DynamicCache

        self.driver, self.index, self.cache_type = driver, index, DynamicCache
        self.cache = DynamicCache()

    def feature(self, ids):
        partition = self.driver.partition
        with self.driver.compute, torch.inference_mode():
            hidden = partition(partition.embed(torch.tensor([ids])), None, self.cache_type())
        _, back = self.driver.submit([self.index, FEATURE, len(ids), 0], hidden)
        with self.driver.compute, torch.inference_mode():
            return partition.norm(back)[0, -1].float().tolist()

    def episode(self, arm):
        self.driver.submit([self.index, RESET, 0, int(arm)])
        self.cache = self.cache_type()

    def cached_tokens(self):
        return self.cache.get_seq_length()

    def crop(self, length):
        self.driver.submit([self.index, CROP, 0, length])
        self.cache.crop(length)

    def step(self, tokens, mask):
        partition = self.driver.partition
        with self.driver.compute, torch.inference_mode():
            hidden = partition(partition.embed(tokens), mask, self.cache)
        _, back = self.driver.submit([self.index, FORWARD, tokens.shape[1], 0], hidden)
        with self.driver.compute, torch.inference_mode():
            return partition.logits(back)


def serve_episodes(driver, streams, tokenizer, policy, cases, gate, feature_prompt, eos):
    """The declared episodes on ``streams`` concurrent streams; rows come back in case order."""
    from .granite_serving import serve_episodes as sequential

    work = queue.Queue()
    for position, case in enumerate(cases):
        work.put((position, case))
    rows, errors = [None] * len(cases), []

    def worker(index):
        stream = driver.stream(index)
        try:
            while True:
                try:
                    position, case = work.get_nowait()
                except queue.Empty:
                    return
                rows[position] = sequential(stream, tokenizer, policy, [case], gate, feature_prompt, eos)[0]
        except Exception as error:
            errors.append(error)

    started = time.monotonic()
    threads = [threading.Thread(target=worker, args=(index,)) for index in range(streams)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    if errors:
        raise errors[0]
    return rows, time.monotonic() - started
