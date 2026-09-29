"""Owner logs and replay audits: optimistic verification of sharded Granite work.

An owner's serving loop is a deterministic state machine: its cache and arm
setting change only by the commands it receives, and each forward pass maps a
received tensor to a sent tensor. The owner logs every command with the digests
of what it received and sent, and retains the received tensors. Anyone holding
only that owner's shard can replay the log from an empty cache and compare
digests; because execution is bit-exact, the first differing digest is a fraud
proof that any other holder of the shard can check the same way.
"""
import hashlib
import json
import time
from pathlib import Path

import torch

from .granite_pipeline import FORWARD, RESET
from .granite_serving import ADAPTER, CROP, FEATURE

LOG_FORMAT = 'neuroshard-granite-owner-log/1'


def digest(tensor):
    return hashlib.sha256(tensor.detach().contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()


class OwnerLog:
    """Commands in order, with input and output digests and the retained inputs."""

    def __init__(self, rank):
        self.rank, self.entries, self.payloads = rank, [], {}

    def command(self, op, value):
        self.entries.append({'op': op, 'value': value})

    def forward(self, op, value, received, sent):
        index = len(self.entries)
        self.entries.append({'op': op, 'value': value, 'input': digest(received), 'output': digest(sent)})
        self.payloads[index] = received.detach().contiguous().view(torch.uint8).clone()

    def save(self, directory, key=None):
        """Write the log and retained inputs; with an Ed25519 ``key`` the owner signs the log's digest."""
        from safetensors.torch import save_file

        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        save_file({f'p{index}': value for index, value in self.payloads.items()}, str(directory / 'inputs.safetensors'))
        record = {'format': LOG_FORMAT, 'rank': self.rank, 'entries': self.entries,
                  'inputs_sha256': hashlib.sha256((directory / 'inputs.safetensors').read_bytes()).hexdigest()}
        if key is not None:
            record = sign(record, key)
        (directory / 'log.json').write_text(json.dumps(record) + '\n')
        return {'entries': len(self.entries), 'forwards': len(self.payloads), 'directory': str(directory),
                'public_key': record.get('public_key')}


def statement(record):
    """The signed bytes: every field except the signature, canonically encoded."""
    body = {k: v for k, v in record.items() if k != 'signature'}
    return hashlib.sha256(json.dumps(body, sort_keys=True, separators=(',', ':')).encode()).digest()


def sign(record, key):
    from cryptography.hazmat.primitives import serialization

    public = key.public_key().public_bytes(serialization.Encoding.Raw, serialization.PublicFormat.Raw).hex()
    record = {**record, 'public_key': public}
    return {**record, 'signature': key.sign(statement(record)).hex()}


def signed_by(record, public_key):
    from cryptography.exceptions import InvalidSignature
    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey

    if record.get('public_key') != public_key or 'signature' not in record:
        return False
    try:
        Ed25519PublicKey.from_public_bytes(bytes.fromhex(public_key)).verify(bytes.fromhex(record['signature']),
                                                                             statement(record))
        return True
    except (InvalidSignature, ValueError):
        return False


def fraud_proof(record, payloads, report):
    """The owner's own signed log and the retained inputs up to its first wrong output."""
    if report['valid'] or report['first_mismatch'] is None:
        raise ValueError('no mismatch to prove')
    needed = {index: payloads[index] for index in payloads if index <= report['first_mismatch']}
    return {'record': record, 'mismatch': report['first_mismatch'], 'inputs': needed}


def check_fraud_proof(partition, proof, public_key, adapter=None):
    """True when the owner signed the log and its logged output at ``mismatch`` is not what its shard computes."""
    record = proof['record']
    if not signed_by(record, public_key) or record['rank'] != partition.rank:
        return False
    head = {**record, 'entries': record['entries'][:proof['mismatch'] + 1]}
    report = replay(partition, head, proof['inputs'], adapter)
    return report['input_mismatch'] is None and report['first_mismatch'] == proof['mismatch']


def load(directory):
    from safetensors.torch import load_file

    directory = Path(directory)
    record = json.loads((directory / 'log.json').read_text())
    if record['format'] != LOG_FORMAT:
        raise ValueError('unsupported owner log')
    if hashlib.sha256((directory / 'inputs.safetensors').read_bytes()).hexdigest() != record['inputs_sha256']:
        raise ValueError('retained inputs differ from the log')
    stored = load_file(str(directory / 'inputs.safetensors'))
    return record, {int(key[1:]): value.view(torch.bfloat16) for key, value in stored.items()}


def outputs(entries):
    return [entry['output'] for entry in entries if 'output' in entry]


def inputs(entries):
    return [entry['input'] for entry in entries if 'input' in entry]


def continuity(upstream, downstream):
    """First forward message whose received digest differs from what the upstream owner logged sending, or None."""
    sent, received = outputs(upstream), inputs(downstream)
    for index, (a, b) in enumerate(zip(sent, received)):
        if a != b:
            return index
    return None if len(sent) == len(received) else min(len(sent), len(received))


def replay(partition, record, payloads, adapter=None):
    """Re-execute one owner's log from an empty cache; returns the first entry whose output differs.

    The caller must run the auditor in the owners' pinned runtime, after the
    declared warm-up.
    """
    from transformers import DynamicCache

    last = partition.rank == len(partition.boundaries) - 2
    cache, checked, started = DynamicCache(), 0, time.monotonic()
    report = {'entries': len(record['entries']), 'input_mismatch': None, 'first_mismatch': None}
    for index, entry in enumerate(record['entries']):
        op, value = entry['op'], entry['value']
        if op == RESET:
            cache = DynamicCache()
        elif op == CROP:
            cache.crop(value)
        elif op == ADAPTER:
            if adapter is not None:
                adapter.set(bool(value))
        elif op in (FORWARD, FEATURE):
            hidden = payloads[index]
            if digest(hidden) != entry['input']:
                report['input_mismatch'] = index
                break
            with torch.inference_mode():
                if op == FORWARD:
                    mask = torch.ones((1, cache.get_seq_length() + value), dtype=torch.long)
                    out = partition(hidden, mask, cache)
                else:
                    out = partition(hidden, None, DynamicCache())
            checked += 1
            if digest(out[:, -1:] if last else out) != entry['output']:
                report['first_mismatch'] = index
                break
        else:
            raise ValueError('unknown logged command')
    return {**report, 'forwards_checked': checked, 'seconds': time.monotonic() - started,
            'valid': report['input_mismatch'] is None and report['first_mismatch'] is None}
