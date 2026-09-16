"""Produce a new expert's training inputs without replacing the serving tail.

The three parent owners keep their complete partitions. A separate four-member
group sends frozen features and reference outputs to the new owner. A continued
job uses an explicitly counted, read-only copy of its accepted expert. Once
production ends, that owner can optimize locally while earlier paths serve.
The output head is an explicitly read-only replica, not another trained shard.
"""

import torch

from ..reference import autocast
from ..reference_data import identity
from . import cohort_state, feature_bank
from .branch import prefix
from .model import batch_tensors


def produce(shard, wire, records, schedule, home, binding, split, microbatch, *,
            reference_expert=None, parent=None, objects=None, resident_parameter_limit=None):
    """Cache exact padded microbatches using only the new cohort's group.

    Logical owners 0..2 hold the complete immutable parent; owner 3 is the new
    expert. The wire may map these positions to noncontiguous physical ranks.
    No forward pass or optimizer update of the new expert occurs here.
    """
    if (wire.world != 4 or not 0 < split < shard.config.num_hidden_layers
            or type(microbatch) is not int or microbatch <= 0):
        raise ValueError('Require a complete parent and one separate expert')
    if wire.rank < 3:
        if (shard.rank != wire.rank or len(shard.boundaries) != 4
                or any(parameter.requires_grad for _, parameter in shard.named_owned_parameters())):
            raise ValueError('The three serving parent partitions must be frozen')
        if wire.rank == 2 and not shard.boundaries[2] < split < shard.boundaries[3]:
            raise ValueError('The last parent owner must retain the original tail')
    elif shard.rank != 3 or shard.boundaries[-2] != split:
        raise ValueError('New learner must own exactly the separate tail')
    if (not schedule or any(not batch or any(type(i) is not int or not 0 <= i < len(records)
                                             for i in batch) for batch in schedule)):
        raise ValueError('Require a valid exact feature schedule')
    teacher = None
    declaration = {'binding': identity(binding),
                   'reference': identity(reference_expert) if reference_expert is not None else None}
    if wire.exchange(declaration) != [declaration] * 4:
        raise ValueError('Feature producers disagree on the accepted reference')
    if reference_expert is not None:
        error = None
        try:
            teacher = cohort_state.frozen_reference(shard, parent, objects, split, reference_expert,
                                                    resident_parameter_limit)
        except (OSError, ValueError, KeyError, TypeError) as failure:
            error = type(failure).__name__
        if any(wire.exchange(error)):
            raise ValueError('Accepted reference is unavailable or exceeds the declared owner limit')
    versions = tuple(p._version for p in teacher.parameters()) if teacher is not None else None
    writer = feature_bank.Writer(home, binding, shard.config, microbatch) if wire.rank == 3 else None
    for indices in schedule:
        batch = [records[index] for index in indices]
        packets = []
        for offset in range(0, len(batch), microbatch):
            rows = batch[offset:offset + microbatch]
            ids, labels, mask, weights = batch_tensors(rows, shard.device_name)
            shape = (*ids.shape, shard.config.hidden_size)
            with torch.no_grad(), autocast(shard.device_name):
                if wire.rank == 3:
                    hidden = wire.receive(2, shape, shard.device_name)
                    reference = wire.receive(2, shape, shard.device_name)
                    packets.append({'prefix': hidden, 'reference': reference, 'ids': ids,
                                    'labels': labels, 'mask': mask, 'weights': weights})
                else:
                    incoming = ids if wire.rank == 0 else wire.receive(wire.rank - 1, shape, shard.device_name)
                    if wire.rank < 2:
                        wire.send(shard(incoming, mask), wire.rank + 1)
                    else:
                        cut = prefix(shard, incoming, mask, split)
                        reference = teacher(cut, mask) if teacher is not None else shard(incoming, mask)
                        wire.send(cut, 3)
                        wire.send(reference, 3)
        if writer is not None:
            writer.batch(batch, packets)
    if teacher is not None and tuple(p._version for p in teacher.parameters()) != versions:
        raise ValueError('Feature production changed the accepted frozen reference')
    return wire.exchange(writer.finish(len(schedule)) if writer else None)[3]
