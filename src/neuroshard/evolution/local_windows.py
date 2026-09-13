"""Bounded local-training windows for operated research groups.

Local Adam states remain private to a rank. A common FP32 outer Nesterov
state changes only at the declared synchronization boundaries. This module
does not verify adversarial workers or authorize native payments.
"""
import hashlib

from . import cooperative as group
from . import reference_data as data


def validate_window(recipe, world):
    steps, batch, interval = recipe['steps'], recipe['batch_documents'], recipe['local_steps']
    if (type(world) is not int or not 1 <= world <= 8
            or any(type(value) is not int or value <= 0 for value in (steps, batch, interval))
            or batch % world or steps % interval
            or not 0 < recipe['outer_learning_rate'] <= 1
            or not 0 <= recipe['outer_momentum'] < 1):
        raise ValueError('Invalid bounded window or outer optimizer')


def gather_digests(value, world, device):
    """Exchange one SHA-256 per rank, without arbitrary object deserialization."""
    import torch
    import torch.distributed as dist
    raw = bytes.fromhex(value)
    if len(raw) != 32:
        raise ValueError('Expected SHA-256 identity')
    if world == 1:
        return [value]
    if not dist.is_initialized() or dist.get_world_size() != world:
        raise ValueError('Declared group differs from initialized membership')
    value_tensor = torch.tensor(list(raw), dtype=torch.uint8, device=device)
    received = [torch.empty_like(value_tensor) for _ in range(world)]
    dist.all_gather(received, value_tensor)
    return [bytes(item.cpu().tolist()).hex() for item in received]


def common_checkpoint(available, checkpoints, world, device):
    """Choose the latest manifest durably present, with the same hash, on every rank.

    A rank's local pointer is insufficient: a crash may leave only some
    participants with the latest group manifest. Zero means absent, never a
    checkpoint identity. The caller verifies its selected local files.
    """
    if checkpoints != sorted(set(checkpoints)) or len(checkpoints) > 16:
        raise ValueError('Checkpoint candidates must be a bounded ordered list')
    if set(available) - set(checkpoints):
        raise ValueError('Undeclared checkpoint candidate')
    selected = None
    for step in checkpoints:
        values = gather_digests(available.get(step, '00' * 32), world, device)
        if values[0] != '00' * 32 and len(set(values)) == 1:
            selected = step
    return selected


def rank_binding(prepared, runtime, arm, rank, world):
    if not 0 <= rank < world:
        raise ValueError('Invalid rank')
    return data.identity({'prepared': prepared, 'profile': group.runtime_profile(runtime),
                          'arm': arm, 'rank': rank, 'world': world})


def state_digest(named_tensors):
    digest = hashlib.sha256()
    for name, tensor in named_tensors:
        value = tensor.detach().cpu().contiguous()
        digest.update(name.encode() + b'\0' + str(tuple(value.shape)).encode() + b'\0')
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


class OuterNesterov:
    """Chunked FP32 delta reduction and CPU outer state.

    For common parent x, local endpoint y_r and k ranks:
        d = mean_r(x - y_r); v = mu*v + d; x = x - eta*(d + mu*v).

    The model receives x, while each rank keeps its own Adam moments. The
    implementation sends deltas rather than summing large parent weights.
    CPU state avoids two additional full-model GPU allocations.
    """
    def __init__(self, model, learning_rate, momentum, chunk_bytes=32 * 1024**2):
        import torch
        if (not 0 < learning_rate <= 1 or not 0 <= momentum < 1
                or type(chunk_bytes) is not int or not 4 <= chunk_bytes <= 64 * 1024**2
                or chunk_bytes % 4):
            raise ValueError('Invalid outer optimizer bounds')
        self.learning_rate, self.momentum = learning_rate, momentum
        self.chunk_elements = chunk_bytes // 4
        self.names, self.parent, self.velocity = [], [], []
        self.round = 0
        for name, parameter in model.named_parameters():
            if parameter.dtype != torch.float32 or not parameter.is_contiguous():
                raise ValueError('Outer optimizer requires contiguous FP32 parameters')
            self.names.append(name)
            self.parent.append(parameter.detach().cpu().clone())
            self.velocity.append(torch.zeros_like(self.parent[-1]))

    def synchronize(self, model, world, check=lambda: None):
        import time
        import torch
        import torch.distributed as dist
        if type(world) is not int or not 1 <= world <= 8:
            raise ValueError('Invalid outer group size')
        parameters = list(model.named_parameters())
        if [name for name, _ in parameters] != self.names:
            raise ValueError('Model parameter topology changed')
        if world > 1 and (not dist.is_initialized() or dist.get_world_size() != world):
            raise ValueError('Outer update requires the declared group')
        started = time.monotonic()
        payload = 0
        with torch.no_grad():
            for (_, parameter), parent, velocity in zip(parameters, self.parent, self.velocity):
                if parameter.shape != parent.shape or parameter.dtype != torch.float32:
                    raise ValueError('Model parameter schema changed')
                target, anchor, moment = parameter.view(-1), parent.view(-1), velocity.view(-1)
                for start in range(0, target.numel(), self.chunk_elements):
                    check()
                    stop = min(start + self.chunk_elements, target.numel())
                    delta = anchor[start:stop].to(parameter.device, copy=True)
                    delta.sub_(target[start:stop])
                    if world > 1:
                        dist.all_reduce(delta, op=dist.ReduceOp.SUM)
                        delta.div_(world)
                    delta = delta.cpu()
                    if not torch.isfinite(delta).all():
                        raise ValueError('Nonfinite outer delta; restore the last common checkpoint')
                    moment[start:stop].mul_(self.momentum).add_(delta)
                    direction = delta.add(moment[start:stop], alpha=self.momentum)
                    anchor[start:stop].add_(direction, alpha=-self.learning_rate)
                    if not torch.isfinite(anchor[start:stop]).all():
                        raise ValueError('Nonfinite outer result; restore the last common checkpoint')
                    target[start:stop].copy_(anchor[start:stop])
                    payload += (stop - start) * 4
        if next(model.parameters()).is_cuda:
            torch.cuda.synchronize()
        self.round += 1
        return {'round': self.round, 'seconds': time.monotonic() - started,
                'delta_payload_bytes_per_rank': payload,
                'scope': 'Logical FP32 payload; wire traffic is measured separately'}

    def save(self, path):
        import torch
        torch.save({'names': self.names, 'velocity': self.velocity, 'round': self.round,
                    'learning_rate': self.learning_rate, 'momentum': self.momentum,
                    'chunk_elements': self.chunk_elements}, path)

    def restore(self, path, expected_round):
        import torch
        # Caller first verifies the complete checkpoint receipt and file hashes.
        state = torch.load(path, map_location='cpu', weights_only=True, mmap=True)
        expected = {'names': self.names, 'round': expected_round,
                    'learning_rate': self.learning_rate, 'momentum': self.momentum,
                    'chunk_elements': self.chunk_elements}
        if any(state.get(key) != value for key, value in expected.items()):
            raise ValueError('Outer optimizer checkpoint identity changed')
        if len(state['velocity']) != len(self.velocity):
            raise ValueError('Incomplete outer optimizer state')
        for target, value in zip(self.velocity, state['velocity']):
            if value.shape != target.shape or value.dtype != target.dtype or not torch.isfinite(value).all():
                raise ValueError('Invalid outer optimizer tensor')
            target.copy_(value)
        self.round = expected_round

    def digest(self):
        return state_digest(zip(self.names, self.velocity))


def write_checkpoint(home, model, tokenizer, optimizer, outer, step, binding, records):
    """Publish a local receipt; only a subsequent group manifest commits it."""
    import os
    from pathlib import Path
    import uuid
    import torch
    home = Path(home)
    directory = home / f'checkpoint-{step:06d}'
    if directory.exists():
        raise ValueError('Preserve the previous checkpoint before retrying this step')
    temporary = home / ('.writing-' + uuid.uuid4().hex)
    temporary.mkdir()
    model.save_pretrained(temporary, safe_serialization=True, max_shard_size='2GB')
    tokenizer.save_pretrained(temporary)
    torch.save({'optimizer': optimizer.state_dict(), 'cpu_rng': torch.get_rng_state(),
                'cuda_rng': torch.cuda.get_rng_state_all() if next(model.parameters()).is_cuda else []},
               temporary / 'optimizer.pt')
    if outer is not None:
        outer.save(temporary / 'outer.pt')
    files = {path.name: data.sha256(path) for path in temporary.iterdir() if path.is_file()}
    receipt = {'step': step, 'binding': binding, 'files': files, 'records': records}
    data.save(temporary / 'checkpoint.json', receipt)
    for path in temporary.iterdir():
        with path.open('rb') as source:
            os.fsync(source.fileno())
    descriptor = os.open(temporary, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    os.replace(temporary, directory)
    descriptor = os.open(home, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    return {'directory': directory.name, 'receipt': data.identity(receipt)}


def verify_group_checkpoint(home, manifest, prepared, runtime, arm, rank, world, step):
    from pathlib import Path
    from . import reference as engine
    expected = {'prepared': prepared, 'arm': arm, 'world': world, 'step': step}
    if any(manifest.get(key) != value for key, value in expected.items()):
        raise ValueError('Group checkpoint identity changed')
    if len(manifest['rank_checkpoints']) != world:
        raise ValueError('Incomplete checkpoint membership')
    pointer = manifest['rank_checkpoints'][rank]
    if pointer['directory'] != f'checkpoint-{step:06d}':
        raise ValueError('Group checkpoint step changed')
    directory, receipt = engine.verify_checkpoint(
        Path(home) / f'rank-{rank}', pointer, rank_binding(prepared, runtime, arm, rank, world))
    if directory.is_symlink() or any(path.is_symlink() for path in directory.iterdir()):
        raise ValueError('Checkpoint links are not allowed')
    if receipt['step'] != step or len(receipt['records']) != step:
        raise ValueError('Checkpoint does not contain its complete local history')
    return directory, receipt
