"""Record causal training activations from the actual owned frozen paths.

Only the receiving owner stores the feature bank. Every backbone and specialist
tensor is produced by its installed owner; no pretrained model is reconstructed
on the receiver. The bank is a replayable optimization for training a small
connection, not a proof of source execution or permission to use held-out data.
"""
from pathlib import Path
import time

import torch
from safetensors.torch import save_file

from ..reference import autocast
from ..reference_data import identity, save, sha256


@torch.no_grad()
def produce(net, rows, batches, home, *, max_length, max_seconds):
    """Produce every prescribed batch once; callers bind its allowed data role."""
    graph, wire, rank = net.graph, net.all_owners, net.rank
    width = graph['parent']['config']['hidden_size']
    if (type(max_seconds) is not int or not 1 <= max_seconds <= 21600
            or type(max_length) is not int or not 2 <= max_length <= 4096
            or not isinstance(rows, list) or not 1 <= len(rows) <= 2048
            or not isinstance(batches, list) or not 1 <= len(batches) <= 2048
            or any(not isinstance(batch, list) or not 1 <= len(batch) <= 16 for batch in batches)):
        raise ValueError('Invalid bounded fusion feature prescription')
    flattened = [index for batch in batches for index in batch]
    if (any(type(index) is not int for index in flattened)
            or sorted(flattened) != list(range(len(rows)))):
        raise ValueError('Feature batches must cover every prescribed document exactly once')
    vocabulary = graph['parent']['config']['vocab_size']
    for row in rows:
        ids, labels = row['input_ids'], row['labels']
        if (not 2 <= len(ids) <= max_length or len(ids) != len(labels) or labels[0] != -100
                or any(type(token) is not int or not 0 <= token < vocabulary for token in ids)
                or any(type(label) is not int or label not in (-100, token) for token, label in zip(ids, labels))
                or not any(label != -100 for label in labels[1:])):
            raise ValueError('Invalid complete response-only fusion training tokens')
    descriptor = {'format': 'neuroshard-fusion-feature-bank-v1', 'graph': identity(graph),
        'rows': identity(rows), 'batches': batches, 'max_length': max_length,
        'source_names': ['hub', 'parent', *graph['experts']],
        # Hostnames are deployment observations, not numerical identities.
        # Keep every other field (including the actual device/runtime build)
        # in the shared commitment; GraphNetwork also checks the frozen profile.
        'runtime': {key: value for key, value in net.runtime.items() if key != 'host'}}
    if wire.exchange(identity(descriptor)) != [identity(descriptor)]*net.world_size:
        raise ValueError('Owners received different fusion source data or runtime')
    home = Path(home)
    if rank == 0:
        home.mkdir(parents=True, exist_ok=False)
    started, before = time.monotonic(), wire.sent_tensor_bytes
    captured, hook = {}, None
    if rank == 2:
        hook = net.shard.layers[str(graph['descriptor']['split']-1)].register_forward_hook(
            lambda module, args, output: captured.update(prefix=output.clone()))
    owners = {rule['id']: rule['owner'] for rule in graph['descriptor']['rules']}
    files = []
    device = net.shard.device_name
    try:
        for number, batch in enumerate(batches):
            if time.monotonic()-started > max_seconds:
                raise TimeoutError('Fusion feature production exceeded its deadline')
            selected = [rows[index] for index in batch]
            count = max(len(row['input_ids']) for row in selected)
            if len(batch)*count*width > wire.max_elements:
                raise ValueError('Fusion feature batch exceeds the owned transport bound')
            ids = torch.full((len(batch), count), net.tokenizer.eos_token_id, dtype=torch.long, device=device)
            mask = torch.zeros_like(ids)
            labels = torch.full_like(ids, -100)
            for index, row in enumerate(selected):
                length = len(row['input_ids'])
                ids[index, :length] = torch.tensor(row['input_ids'], device=device)
                labels[index, :length] = torch.tensor(row['labels'], device=device)
                mask[index, :length] = 1
            shape = (len(batch), count, width)
            tensors = {}
            if rank < 3:
                incoming = ids if rank == 0 else wire.receive(rank-1, shape, device)
                with autocast(device):
                    hidden = net.shard(incoming, mask)
                if rank < 2:
                    wire.send(hidden, rank+1)
                else:
                    wire.send(hidden, 0)
                    for owner in owners.values():
                        wire.send(captured['prefix'], owner)
                    captured.clear()
            else:
                incoming = wire.receive(2, shape, device)
                with autocast(device):
                    hidden = net.shard(incoming, mask)
                wire.send(hidden, 0)
            if rank == 0:
                tensors['parent'] = wire.receive(2, shape, device).cpu()
                for name, owner in owners.items():
                    tensors[name] = wire.receive(owner, shape, device).cpu()
            if rank < 3:
                incoming = ids if rank == 0 else wire.receive(rank-1, shape, device)
                with autocast(device):
                    hidden = net.preserved.shard(incoming, mask)
                wire.send(hidden, rank+1 if rank < 2 else 0)
            spec = None
            if rank == 0:
                tensors['hub'] = wire.receive(2, shape, device).cpu()
                tensors.update(input_ids=ids.cpu(), labels=labels.cpu(), valid=mask.bool().cpu())
                temporary = home/(str(number)+'.pending')
                save_file({key: value.contiguous() for key, value in tensors.items()}, str(temporary))
                digest = sha256(temporary)
                spec = {'sha256': digest, 'bytes': temporary.stat().st_size,
                        'shape': list(shape), 'rows': [row['id'] for row in selected]}
                temporary.replace(home/(digest+'.safetensors'))
                files.append(spec)
                save(home/'progress.json', {'complete': len(files), 'total': len(batches),
                                          'seconds': time.monotonic()-started})
            packets = wire.exchange(spec)
            if packets[0] is None or any(value is not None for value in packets[1:]):
                raise ValueError('Only the receiving owner commits the fusion feature bank')
            if rank != 0:
                files.append(packets[0])
        net.verify_unchanged()
        result = {**descriptor, 'files': files}
        if wire.exchange(identity(result)) != [identity(result)]*net.world_size:
            raise ValueError('Owners disagree on the produced fusion feature bank')
        if rank == 0:
            save(home/'manifest.json', result)
            save(home/'resources.json', {'seconds': time.monotonic()-started,
                                        'bytes': sum(spec['bytes'] for spec in files)})
        return result, {'seconds': time.monotonic()-started, 'sent_tensor_bytes': wire.sent_tensor_bytes-before,
                        'owner_runtime': net.runtime}
    finally:
        if hook is not None:
            hook.remove()
