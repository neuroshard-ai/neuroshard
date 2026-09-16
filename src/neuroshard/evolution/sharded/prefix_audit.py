"""Recompute cached expert inputs one immutable parent partition at a time.

Each stage owns one partition and consumes the preceding stage's content-bound
output. All stages must be replayed to verify production. A chain of report
hashes alone is not evidence of neural execution or independent auditors.
"""
from pathlib import Path
import math
import time

import torch

from .. import expert_checkpoint, reference_data as data
from ..reference import autocast
from ..schema import root
from . import cohort_state, feature_bank, incremental_state, portable
from .branch import prefix
from .model import batch_tensors

FORMAT = 'neuroshard-expert-prefix-audit-v1'


def stage_binding(context, rank, incoming):
    return {'format': FORMAT, 'context': context, 'rank': rank, 'input_root': incoming}


def replay_stage(shard, parent, objects, records, batches, binding, split, microbatch,
                 target_root, home, incoming=None, max_seconds=900, *, reference_expert=None,
                 resident_parameter_limit=None):
    """Write one replayed stage and compare the final bank with its claimed root.

    The caller pins the numerical runtime and source, and supplies the committed
    training records and production binding. ``incoming`` is (directory, report)
    from the preceding replay. A continued job also loads a read-only copy of
    its accepted expert on the last parent owner; both sets of parameters count
    against the declared resident limit. The full backbone is never loaded.
    """
    started = time.monotonic()
    if target_root is not None:
        root(target_root)
    rank = shard.rank
    if (len(shard.boundaries) != 4 or list(shard.boundaries) != parent['boundaries']
            or rank not in (0, 1, 2) or not shard.boundaries[2] < split < shard.boundaries[3]
            or portable.configuration(shard.config) != parent['config']
            or type(microbatch) is not int or microbatch <= 0
            or not batches or any(not row or any(type(i) is not int or not 0 <= i < len(records)
                                                 for i in row) for row in batches)):
        raise ValueError('Replay exactly the committed parent partition and ordered feature batches')
    if (type(max_seconds) not in (int, float) or not math.isfinite(max_seconds)
            or not 0 < max_seconds <= 3600):
        raise ValueError('Bound the partition replay duration')
    inherited = expert_checkpoint.parent_records(parent)
    names = dict(shard.named_owned_parameters())
    if set(names) != {name for name in inherited if expert_checkpoint.owner(name, shard.boundaries) == rank}:
        raise ValueError('Incomplete owned parent parameters')
    with torch.no_grad():
        for name, parameter in names.items():
            record = inherited[name]
            values = incremental_state.tensor_values(portable.tensor_path(objects, record['sha256']), record)
            parameter.copy_(values['weight'])
            parameter.requires_grad_(False)
            del values
    shard.eval()
    versions = tuple(p._version for p in names.values())
    teacher = None
    if reference_expert is not None:
        teacher = cohort_state.frozen_reference(shard, parent, objects, split, reference_expert,
                                                resident_parameter_limit)
        if teacher is not None:
            teacher_versions = tuple(p._version for p in teacher.parameters())
    context = {'parent': data.identity(parent), 'binding': data.identity(binding),
               'records': data.identity(records), 'batches': data.identity(batches),
               'split': split, 'microbatch': microbatch, 'target': target_root}
    if reference_expert is not None:
        context['reference_expert'] = data.identity(reference_expert)
    reader = None
    if rank == 0:
        if incoming is not None:
            raise ValueError('The first partition starts from committed tokens')
        input_root = data.identity({'records': context['records'], 'batches': context['batches']})
    else:
        if incoming is None:
            raise ValueError('Require the preceding partition output')
        directory, report = incoming
        if (report['format'] != FORMAT or report['rank'] != rank - 1
                or report['context'] != context or report['completed'] is not True):
            raise ValueError('Preceding output belongs to another parent, stage or cohort')
        input_root = report['output_root']
        reader = feature_bank.Reader(directory, input_root,
            stage_binding(context, rank - 1, report['input_root']), shard.config, microbatch, len(batches))
    output_binding = binding if rank == 2 else stage_binding(context, rank, input_root)
    home = Path(home)
    writer = feature_bank.Writer(home / 'features', output_binding, shard.config, microbatch)
    count = 0
    for index, indices in enumerate(batches):
        if time.monotonic() - started > max_seconds:
            raise TimeoutError('Parent partition replay deadline expired')
        rows = [records[i] for i in indices]
        previous = reader.batch(index, rows, shard.device_name) if reader else None
        packets = []
        for number, offset in enumerate(range(0, len(rows), microbatch)):
            subset = rows[offset:offset + microbatch]
            ids, labels, mask, weights = batch_tensors(subset, shard.device_name)
            hidden = ids if previous is None else previous[number]['prefix']
            with torch.no_grad(), autocast(shard.device_name):
                if rank == 2:
                    cut = prefix(shard, hidden, mask, split)
                    reference = teacher(cut, mask) if teacher is not None else shard(hidden, mask)
                else:
                    cut = shard(hidden, mask)
                    reference = cut
            packets.append({'prefix': cut, 'reference': reference, 'ids': ids,
                            'labels': labels, 'mask': mask, 'weights': weights})
            count += 1
        writer.batch(rows, packets)
        del previous, packets
        print({'event': 'prefix-audit-batch', 'rank': rank, 'batches': index + 1}, flush=True)
    output_root = writer.finish(len(batches))
    if tuple(p._version for p in names.values()) != versions:
        raise ValueError('Prefix verification changed an immutable parent weight')
    if teacher is not None and tuple(p._version for p in teacher.parameters()) != teacher_versions:
        raise ValueError('Prefix verification changed the accepted frozen expert reference')
    result = {'format': FORMAT, 'rank': rank, 'context': context, 'input_root': input_root,
              'output_root': output_root, 'completed': True, 'microbatches': count,
              'parameters': shard.resident_parameters+(teacher.resident_parameters if teacher is not None else 0),
              'owned_names': sorted(names),
              'target_checked': rank == 2 and target_root is not None,
              'valid': output_root == target_root if rank == 2 and target_root is not None else None,
              'seconds': time.monotonic() - started}
    if reference_expert is not None:
        result['reference_parameters'] = teacher.resident_parameters if teacher is not None else 0
    data.save(home / 'result.json', result)
    if rank == 2 and target_root is not None and not result['valid']:
        raise ValueError('Recomputed prefix and reference differ from the claimed feature bank')
    return result


def complete(reports, target_root):
    """Check complete local execution receipts; never trust remote files as proof."""
    if not isinstance(reports, list) or len(reports) != 3:
        raise ValueError('The production audit requires all three parent partitions')
    context = reports[0]['context']
    if reports[0]['input_root'] != data.identity({'records': context['records'], 'batches': context['batches']}):
        raise ValueError('The first parent partition must begin at the committed input tokens')
    for rank, report in enumerate(reports):
        if (report['format'] != FORMAT or report['rank'] != rank or report['completed'] is not True
                or report['context'] != context or report['context']['target'] != target_root
                or report['target_checked'] is not (rank == 2)
                or (rank and report['input_root'] != reports[rank - 1]['output_root'])):
            raise ValueError('Missing or disconnected parent production audit')
    if reports[-1]['output_root'] != target_root or reports[-1]['valid'] is not True:
        raise ValueError('The entire computed feature bank must match')
    return {'passed': True, 'feature_root': target_root, 'stages': [data.identity(r) for r in reports],
            'microbatches_per_stage': reports[-1]['microbatches'],
            'max_owned_parameters': max(r['parameters'] for r in reports)}
