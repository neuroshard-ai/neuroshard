"""Compact fraud witnesses for asserted float32 SGD tensor updates.

This is a refutation path, not a proof of an entire training step. Publishers
assert that the indexed before/gradient/after tensors describe the signed stage.
Full-stage auditing must still check those assertions against the model files
and computed gradients. A valid Merkle opening can refute an inconsistent
assertion without sending the whole stage to every consensus validator.
"""
import base64
import hashlib
import math
import struct

from neuroshard.dataflow.store import canonical
from . import schema

FORMAT = 'neuroshard-sgd-chunks-v1'
CHUNK_ELEMENTS = 1024
MAX_WITNESS_BYTES = 32 * 1024
EMPTY = hashlib.sha256(b'neuroshard/f32-chunks/v1/empty').digest()


def _leaf(index, raw):
    value = hashlib.sha256(b'neuroshard/f32-chunks/v1/leaf\0' + struct.pack('<QI', index, len(raw)))
    value.update(raw)
    return value.digest()


def _branch(left, right):
    return hashlib.sha256(b'neuroshard/f32-chunks/v1/node\0' + left + right).digest()


def _array(value):
    from .model import torch
    if value.dtype != torch.float32 or value.ndim not in (1, 2):
        raise ValueError('Update commitments require one- or two-dimensional float32 tensors')
    schema.integer(value.numel(), 1, schema.MAX_PARAMETERS)
    # Explicit little-endian scalar representation, independent of NumPy's
    # native-endian dtype spelling. No tensor bytes are stored on the hot path.
    return value.detach().cpu().contiguous().numpy().astype('<f4', copy=False)


def _tree(value):
    array = _array(value)
    raw = memoryview(array).cast('B')
    leaves = [_leaf(index // (4 * CHUNK_ELEMENTS), raw[index:index+4*CHUNK_ELEMENTS])
              for index in range(0, len(raw), 4*CHUNK_ELEMENTS)]
    size = 1 << (len(leaves)-1).bit_length()
    leaves.extend([EMPTY] * (size-len(leaves)))
    levels = [leaves]
    while len(levels[-1]) > 1:
        row = levels[-1]
        levels.append([_branch(row[i], row[i+1]) for i in range(0, len(row), 2)])
    return array, raw, levels


def commit(value):
    array, _, levels = _tree(value)
    return {'shape': list(array.shape), 'root': levels[-1][0].hex()}


def open_chunk(value, index):
    array, raw, levels = _tree(value)
    count = (array.size + CHUNK_ELEMENTS-1) // CHUNK_ELEMENTS
    schema.integer(index, 0, count-1)
    data = raw[index*4*CHUNK_ELEMENTS:(index+1)*4*CHUNK_ELEMENTS]
    position, siblings = index, []
    for level in levels[:-1]:
        siblings.append(level[position ^ 1].hex())
        position //= 2
    return {'shape': list(array.shape), 'root': levels[-1][0].hex()}, {
        'data': base64.b64encode(data).decode(), 'siblings': siblings}


def descriptor(value, shape):
    if (not isinstance(value, dict) or set(value) != {'shape', 'root'}
            or not isinstance(value['shape'], list) or any(type(n) is not int for n in value['shape'])
            or value['shape'] != shape):
        raise ValueError('Update tensor shape differs from its model parameter')
    schema.root(value['root'])
    return value


def validate_trace(parent, trace):
    enabled = parent.get('update_witnesses') == FORMAT
    commitments = trace.get('updates')
    if not enabled:
        if commitments is not None:
            raise ValueError('Update commitments require the declared model execution profile')
        return
    if not isinstance(commitments, dict) or set(commitments) != {'format', 'tensors'} or commitments['format'] != FORMAT:
        raise ValueError('Missing declared update commitments')
    expected = [(name, tensor, shape) for name in sorted(trace['partition']['components'])
                for tensor, shape in sorted(schema.shapes(parent['config'], name).items())]
    entries = commitments['tensors']
    if not isinstance(entries, list) or len(entries) != len(expected):
        raise ValueError('Update commitments must cover every stage parameter exactly once')
    for entry, (name, tensor, shape) in zip(entries, expected):
        if (not isinstance(entry, dict) or set(entry) != {'component', 'tensor', 'before', 'gradient', 'after'}
                or (entry['component'], entry['tensor']) != (name, tensor)):
            raise ValueError('Update tensor ownership or order differs from the stage')
        for field in ('before', 'gradient', 'after'):
            descriptor(entry[field], shape)


def capture_before(shard):
    entries = []
    for name, tensor, value in shard.tensor_items():
        if value.grad is None:
            raise ValueError('Every trainable tensor requires a gradient')
        entries.append({'component': name, 'tensor': tensor,
                        'before': commit(value), 'gradient': commit(value.grad)})
    return entries


def capture_after(shard, entries):
    for entry, (name, tensor, value) in zip(entries, shard.tensor_items()):
        if (entry['component'], entry['tensor']) != (name, tensor):
            raise ValueError('Stage parameter ownership changed during the update')
        entry['after'] = commit(value)
    return {'format': FORMAT, 'tensors': entries}


def _opening(value, index, proof):
    shape = value['shape']
    elements = math.prod(shape)
    count = (elements + CHUNK_ELEMENTS-1) // CHUNK_ELEMENTS
    schema.integer(index, 0, count-1)
    depth = (count-1).bit_length()
    if not isinstance(proof, dict) or set(proof) != {'data', 'siblings'}:
        raise ValueError('Invalid tensor opening')
    siblings = proof['siblings']
    if not isinstance(siblings, list) or len(siblings) != depth:
        raise ValueError('Incorrect Merkle path depth')
    if not isinstance(proof['data'], str) or len(proof['data']) > 4*((CHUNK_ELEMENTS*4+2)//3):
        raise ValueError('Tensor opening exceeds its chunk bound')
    raw = base64.b64decode(proof['data'], validate=True)
    if base64.b64encode(raw).decode() != proof['data']:
        raise ValueError('Noncanonical tensor bytes')
    if len(raw) != min(CHUNK_ELEMENTS, elements-index*CHUNK_ELEMENTS)*4:
        raise ValueError('Incorrect tensor chunk length')
    node, position, empty = _leaf(index, raw), index, EMPTY
    for level, sibling in enumerate(siblings):
        sibling = bytes.fromhex(schema.root(sibling))
        if ((position ^ 1) << level) >= count and sibling != empty:
            raise ValueError('Noncanonical padded Merkle subtree')
        node = _branch(sibling, node) if position & 1 else _branch(node, sibling)
        position //= 2
        empty = _branch(empty, empty)
    if node.hex() != value['root']:
        raise ValueError('Tensor opening does not match the signed stage commitment')
    return raw


def check(metadata, record_root, stage, tensor_index, witness):
    """Check at most 1,024 update elements, with no artifact reads or neural replay."""
    if len(canonical(witness)) > MAX_WITNESS_BYTES:
        raise ValueError('Update witness exceeds its 32 KiB bound')
    if not isinstance(witness, dict) or set(witness) != {'chunk', 'before', 'gradient', 'after'}:
        raise ValueError('Invalid update witness')
    record = metadata.json(schema.root(record_root))
    schema.integer(stage, 0, len(record['traces'])-1)
    trace = metadata.json(record['traces'][stage])
    parent = metadata.json(record['parent'])
    if parent.get('update_witnesses') != FORMAT:
        raise ValueError('This model has no compact update execution profile')
    validate_trace(parent, trace)
    entries = trace['updates']['tensors']
    schema.integer(tensor_index, 0, len(entries)-1)
    entry = entries[tensor_index]
    values = [_opening(entry[field], witness['chunk'], witness[field])
              for field in ('before', 'gradient', 'after')]
    from .model import torch
    import numpy as np
    before, gradient, after = [torch.from_numpy(np.frombuffer(raw, dtype='<f4').copy()) for raw in values]
    if not all(torch.isfinite(value).all() for value in (before, gradient, after)):
        return {'valid': False, 'mismatch': 'nonfinite committed optimizer tensor'}
    rate, scale = float.fromhex(trace['learning_rate_hex']), float.fromhex(trace['scale_hex'])
    if (not math.isfinite(rate) or not 0 < rate <= .1 or not math.isfinite(scale) or not 0 < scale <= 1
            or rate.hex() != trace['learning_rate_hex'] or scale.hex() != trace['scale_hex']):
        raise ValueError('Invalid canonical optimizer scalars')
    before.add_(gradient, alpha=-rate*scale)
    valid = before.numpy().astype('<f4', copy=False).tobytes() == values[2]
    return {'valid': valid, 'mismatch': '' if valid else 'inconsistent committed SGD chunk',
            'elements': before.numel()}


def prepare(store, trace_root, tensor_index, chunk):
    """Reconstruct the gradient and open the publisher's three asserted tensors.

The observer still replays the stage. If commitments or earlier computations
are false, use the full-stage dispute; this witness only localizes SGD fraud.
"""
    from .worker import Session
    trace = store.json(trace_root)
    parent = store.json(trace['parent'])
    validate_trace(parent, trace)
    if 'updates' not in trace:
        raise ValueError('Trace does not carry update commitments')
    schema.integer(tensor_index, 0, len(trace['updates']['tensors'])-1)
    entry = trace['updates']['tensors'][tensor_index]
    session = Session(store, trace['parent'], trace['partition'])
    common = {'parent': trace['parent'], 'step': trace['step']}
    result = session.run({**common, 'phase': trace['phase'], 'input': trace['input']})
    if result['output'] != trace['output']:
        raise ValueError('Disagreement before optimizer update; use full-stage replay')
    if trace['phase'] == 'begin':
        result = session.run({'phase': 'loss', 'input': trace['head_input']})
        if result != {'gradient': trace['head_gradient'], 'loss_hex': trace['loss_hex']}:
            raise ValueError('Disagreement before optimizer update; use full-stage replay')
    result = session.run({'phase': 'backward', 'gradient': trace['gradient_out']})
    if result != {'gradient': trace['gradient_in'], 'norm_squared_hex': trace['norm_squared_hex']}:
        raise ValueError('Disagreement before optimizer update; use full-stage replay')
    selected = next(value for name, tensor, value in session.shard.tensor_items()
                    if (name, tensor) == (entry['component'], entry['tensor']))
    after = store.tensors(trace['components'][entry['component']]['root'],
                          schema.shapes(parent['config'], entry['component']))[entry['tensor']]
    witness = {'chunk': chunk}
    for field, value in (('before', selected), ('gradient', selected.grad), ('after', after)):
        declared, opening = open_chunk(value, chunk)
        if declared != entry[field]:
            raise ValueError('Tensor index differs from the actual stage; use full-stage replay')
        witness[field] = opening
    return witness
