"""Replay a shard against a closed, cross-checked communication transcript.

Recorded boundaries are witnesses, not trusted inputs. Every sender, receiver,
collective and final optimizer partition must be replayed before accepting the
whole window. This saves auditor memory, not the total cost of computation.
"""
import copy
import json
import math
from pathlib import Path

import torch
from safetensors.torch import load_file

from ..reference_data import identity, save, sha256
from . import checkpoint, portable

FORMAT = 'neuroshard-shard-transcript-v1'


class Recorder:
    def __init__(self, wire, home):
        self.wire, self.home = wire, Path(home)
        self.home.mkdir(parents=True, exist_ok=False)
        self.rank, self.world = wire.rank, wire.world
        self.events = []

    def tensor(self, value):
        path = self.home/'tensor.pending'
        spec = checkpoint.tensor_file(path, {'value': value.detach().cpu().contiguous()})
        path.replace(portable.tensor_path(self.home, spec['sha256']))
        return {**spec, 'shape': list(value.shape)}

    def send(self, value, destination):
        self.wire.send(value, destination)
        self.events.append({'kind': 'send', 'peer': destination, 'tensor': self.tensor(value)})

    def receive(self, source, shape, device):
        value = self.wire.receive(source, shape, device)
        self.events.append({'kind': 'receive', 'peer': source, 'tensor': self.tensor(value)})
        return value

    def sum(self, value):
        result = self.wire.sum(value)
        self.events.append({'kind': 'sum', 'local': value, 'result': result})
        return result

    def exchange(self, value):
        result = self.wire.exchange(value)
        self.events.append({'kind': 'exchange', 'local': copy.deepcopy(value), 'result': copy.deepcopy(result)})
        return result

    def finish(self, binding):
        result = {'format': FORMAT, 'rank': self.rank, 'world': self.world,
                  'binding': binding, 'events': self.events}
        save(self.home/'transcript.json', result)
        return result


def validate(rows):
    """Close all communication dependencies using small manifests only."""
    if not 2 <= len(rows) <= 512:
        raise ValueError('Incomplete transcript group')
    binding = rows[0]['binding']
    sends, receives, collective = {}, {}, []
    for rank, row in enumerate(rows):
        if (row['format'] != FORMAT or row['rank'] != rank or row['world'] != len(rows)
                or row['binding'] != binding or len(row['events']) > 100000):
            raise ValueError('Transcript job, rank, count or window differs')
        calls = []
        for event in row['events']:
            kind = event['kind']
            if kind in ('send', 'receive'):
                peer = event['peer']
                if type(peer) is not int or not 0 <= peer < len(rows) or peer == rank:
                    raise ValueError('Invalid transcript endpoint')
                source, destination = (rank, peer) if kind == 'send' else (peer, rank)
                target = sends if kind == 'send' else receives
                target.setdefault((source, destination), []).append(event['tensor'])
            elif kind in ('sum', 'exchange'):
                calls.append(event)
            else:
                raise ValueError('Unknown transcript event')
        collective.append(calls)
    if sends != receives:
        raise ValueError('Boundary witnesses do not match their claimed senders')
    if any(len(c) != len(collective[0]) for c in collective):
        raise ValueError('Incomplete collective coverage')
    for calls in zip(*collective):
        if len({c['kind'] for c in calls}) != 1:
            raise ValueError('Collective order differs')
        result = [c['local'] for c in calls] if calls[0]['kind'] == 'exchange' else math.fsum(c['local'] for c in calls)
        if any(identity(c['result']) != identity(result) for c in calls):
            raise ValueError('Collective witness does not follow from all rank inputs')
    return identity(rows)


class Replay:
    def __init__(self, home, transcript):
        self.home, self.transcript = Path(home), transcript
        self.rank, self.world = transcript['rank'], transcript['world']
        self.index = 0

    def event(self, kind):
        if self.index >= len(self.transcript['events']):
            raise ValueError('Replay requested an unrecorded operation')
        row = self.transcript['events'][self.index]
        self.index += 1
        if row['kind'] != kind:
            raise ValueError('Replay communication order differs')
        return row

    def tensor(self, spec, device):
        path = portable.tensor_path(self.home, spec['sha256'])
        if path.stat().st_size != spec['bytes'] or sha256(path) != spec['sha256']:
            raise ValueError('Missing or corrupt boundary witness')
        values = load_file(path)
        if set(values) != {'value'}:
            raise ValueError('Invalid witness tensor inventory')
        value = values['value']
        if (list(value.shape) != spec['shape'] or value.dtype != torch.float32
                or value.ndim != 3 or value.numel() > 4*1024*4096 or not bool(torch.isfinite(value).all())):
            raise ValueError('Invalid witness tensor')
        return value.to(device)

    def send(self, value, destination):
        row = self.event('send')
        if row['peer'] != destination or not torch.equal(value.detach(), self.tensor(row['tensor'], value.device)):
            raise ValueError('Replayed forward value or backward gradient differs')

    def receive(self, source, shape, device):
        row = self.event('receive')
        if row['peer'] != source or row['tensor']['shape'] != list(shape):
            raise ValueError('Replayed input endpoint or shape differs')
        return self.tensor(row['tensor'], device)

    def sum(self, value):
        row = self.event('sum')
        if value != row['local']:
            raise ValueError('Replayed scalar differs')
        return row['result']

    def exchange(self, value):
        row = self.event('exchange')
        if identity(value) != identity(row['local']):
            raise ValueError('Replayed gradient inventory or declaration differs')
        return copy.deepcopy(row['result'])

    def finish(self):
        if self.index != len(self.transcript['events']):
            raise ValueError('Replay did not cover the complete communication window')
