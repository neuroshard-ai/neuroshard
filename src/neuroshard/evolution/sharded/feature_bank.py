"""Content-bound storage for an operated frozen-prefix feature production.

The bank root must come from verified production. Its hashes detect changes;
they do not certify an arbitrary producer's neural computation.
"""
import json
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file

from .. import reference_data as data
from .features import validate_packet
from .portable import configuration

FORMAT = 'neuroshard-frozen-feature-bank-v1'


class Writer:
    def __init__(self, home, binding, config, microbatch):
        if type(microbatch) is not int or microbatch <= 0:
            raise ValueError('Invalid feature microbatch size')
        self.home = Path(home)
        self.home.mkdir(parents=True, exist_ok=False)
        self.config, self.microbatch = config, microbatch
        self.manifest = {'format': FORMAT, 'binding': binding, 'config': configuration(config),
                         'microbatch': microbatch, 'batches': []}

    def batch(self, records, packets):
        if not records or len(packets) != (len(records) + self.microbatch - 1) // self.microbatch:
            raise ValueError('Incomplete feature batch')
        index = len(self.manifest['batches'])
        files = []
        for number, (offset, packet) in enumerate(zip(range(0, len(records), self.microbatch), packets)):
            subset = records[offset:offset + self.microbatch]
            validate_packet(packet, subset, self.config, packet['ids'].device)
            alias = torch.equal(packet['prefix'], packet['reference'])
            values = {key: value.detach().cpu().contiguous().clone() for key, value in packet.items()
                      if key != 'reference' or not alias}
            name = f'batch-{index:04d}-micro-{number:04d}.safetensors'
            path = self.home / name
            save_file(values, path)
            files.append({'file': name, 'sha256': data.sha256(path), 'bytes': path.stat().st_size,
                          'records': data.identity(subset), 'reference_aliases_prefix': alias})
        self.manifest['batches'].append({'records': data.identity(records), 'files': files})

    def finish(self, count):
        if len(self.manifest['batches']) != count or type(count) is not int or count <= 0:
            raise ValueError('Feature production did not cover the declared schedule')
        data.save(self.home / 'index.json', self.manifest)
        return data.identity(self.manifest)


class Reader:
    def __init__(self, home, expected, binding, config, microbatch, count):
        self.home = Path(home)
        self.manifest = json.loads((self.home / 'index.json').read_bytes())
        self.config, self.microbatch = config, microbatch
        if (data.identity(self.manifest) != expected or self.manifest['format'] != FORMAT
                or self.manifest['binding'] != binding or self.manifest['config'] != configuration(config)
                or self.manifest['microbatch'] != microbatch or len(self.manifest['batches']) != count):
            raise ValueError('Feature bank differs from its expected production commitment')

    def batch(self, index, records, device):
        if type(index) is not int or not 0 <= index < len(self.manifest['batches']):
            raise ValueError('Invalid feature batch cursor')
        batch = self.manifest['batches'][index]
        if (batch['records'] != data.identity(records)
                or len(batch['files']) != (len(records) + self.microbatch - 1) // self.microbatch):
            raise ValueError('Feature batch changed the exact ordered training records')
        packets = []
        for number, (offset, spec) in enumerate(zip(range(0, len(records), self.microbatch), batch['files'])):
            subset = records[offset:offset + self.microbatch]
            if (spec['file'] != f'batch-{index:04d}-micro-{number:04d}.safetensors'
                    or spec['records'] != data.identity(subset)
                    or type(spec['reference_aliases_prefix']) is not bool):
                raise ValueError('Feature microbatch changed its committed padding or input')
            path = self.home / spec['file']
            if path.is_symlink() or path.stat().st_size != spec['bytes'] or data.sha256(path) != spec['sha256']:
                raise ValueError('Feature tensor bytes differ from the production commitment')
            packet = load_file(path, device=str(device))
            if spec['reference_aliases_prefix']:
                if 'reference' in packet:
                    raise ValueError('Ambiguous aliased reference representation')
                packet['reference'] = packet['prefix']
            validate_packet(packet, subset, self.config, device)
            packets.append(packet)
        return packets


def produce(shard, wire, records, schedule, home, binding, frozen_layers, original, microbatch):
    """Run the frozen prefix once for every exact training microbatch."""
    from . import incremental
    from .model import batch_tensors
    last = wire.world - 1
    writer = Writer(home, binding, shard.config, microbatch) if wire.rank == last else None
    for indices in schedule:
        batch = [records[index] for index in indices]
        packets = []
        for offset in range(0, len(batch), microbatch):
            ids, labels, mask, weights = batch_tensors(batch[offset:offset + microbatch], shard.device_name)
            with torch.no_grad():
                incoming, outgoing, reference, final = incremental.forward(shard, wire, ids, mask, frozen_layers, original)
                if wire.rank == 0:
                    wire.send(reference, last)
                if wire.rank == last:
                    teacher = wire.receive(0, (*ids.shape, shard.config.hidden_size), shard.device_name)
                    packets.append({'prefix': incoming, 'reference': teacher, 'ids': ids,
                                    'labels': labels, 'mask': mask, 'weights': weights})
            del incoming, outgoing, reference, final
        if wire.rank == last:
            writer.batch(batch, packets)
        del packets
    root = wire.exchange(writer.finish(len(schedule)) if writer else None)[last]
    return root
