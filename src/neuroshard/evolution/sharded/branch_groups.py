"""Independent model paths for several separately owned transformer experts.

This transport composes existing greedy transformer execution. It does not
implement learned routing, dynamic process admission, or native settlement.
"""
import json
import re

import torch
import torch.distributed as dist

from .branch import Network, ParentWire
from .wire import Wire


class GroupWire(Wire):
    """Map logical model positions to possibly noncontiguous process ranks."""

    def __init__(self, global_rank, members, group):
        if (not isinstance(members, list) or not 2 <= len(members) <= 512
                or any(type(rank) is not int or rank < 0 for rank in members)
                or members != sorted(set(members)) or global_rank not in members
                or dist.get_process_group_ranks(group) != members):
            raise ValueError('Require the exact ordered members of the process group')
        super().__init__(members.index(global_rank), len(members))
        self.members, self.group = tuple(members), group

    def peer(self, logical_rank):
        if type(logical_rank) is not int or not 0 <= logical_rank < self.world:
            raise ValueError('Unknown logical model owner')
        return self.members[logical_rank]

    def send(self, value, destination):
        if value.dtype != torch.float32 or value.ndim != 3 or value.numel() > self.max_elements:
            raise ValueError('Unsupported branch boundary tensor')
        destination = self.peer(destination)
        payload = value.detach().to('cpu').contiguous()
        dist.send(torch.tensor(list(payload.shape), dtype=torch.int64), dst=destination, group=self.group)
        dist.send(payload, dst=destination, group=self.group)
        self.sent_tensor_bytes += payload.numel() * 4 + 24

    def receive(self, source, shape, device):
        source = self.peer(source)
        header = torch.empty(3, dtype=torch.int64)
        dist.recv(header, src=source, group=self.group)
        actual = tuple(header.tolist())
        if actual != tuple(shape) or any(n <= 0 for n in actual) or int(header.prod()) > self.max_elements:
            raise ValueError('Unexpected branch boundary shape')
        value = torch.empty(actual, dtype=torch.float32)
        dist.recv(value, src=source, group=self.group)
        if not bool(torch.isfinite(value).all()):
            raise ValueError('Nonfinite branch boundary')
        return value.to(device)

    def exchange(self, value):
        raw = json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()
        if not 0 < len(raw) <= 2 * 1024**2:
            raise ValueError('Require a bounded branch control message')
        sizes = [torch.zeros(1, dtype=torch.int64) for _ in self.members]
        dist.all_gather(sizes, torch.tensor([len(raw)], dtype=torch.int64), group=self.group)
        if any(not 0 < int(size) <= 2 * 1024**2 for size in sizes):
            raise ValueError('Branch control message exceeds its bound')
        length = max(int(size) for size in sizes)
        payload = torch.zeros(length, dtype=torch.uint8)
        payload[:len(raw)] = torch.frombuffer(bytearray(raw), dtype=torch.uint8)
        received = [torch.empty_like(payload) for _ in sizes]
        dist.all_gather(received, payload, group=self.group)
        return [json.loads(bytes(item[:int(size)].tolist())) for item, size in zip(received, sizes)]


class OrderedRoutes:
    """Existing domain choices take precedence over newly appended choices."""

    def __init__(self, rules):
        if not isinstance(rules, list) or not 1 <= len(rules) <= 64:
            raise ValueError('Require one to sixty-four explicit expert rules')
        identifiers, owners, needles = set(), set(), set()
        for rule in rules:
            if (not isinstance(rule, dict) or set(rule) != {'id', 'needle', 'owner'}
                    or not isinstance(rule['id'], str) or not re.fullmatch('[a-z][a-z0-9-]{0,63}', rule['id'])
                    or not isinstance(rule['needle'], str) or not 1 <= len(rule['needle'].encode()) <= 256
                    or rule['needle'] != rule['needle'].strip().casefold()
                    or type(rule['owner']) is not int or not 3 <= rule['owner'] <= 511
                    or rule['id'] in identifiers or rule['owner'] in owners or rule['needle'] in needles):
                raise ValueError('Invalid or repeated expert route')
            identifiers.add(rule['id'])
            owners.add(rule['owner'])
            needles.add(rule['needle'])
        self._rules = tuple((rule['id'], rule['needle'], rule['owner']) for rule in rules)

    @property
    def rules(self):
        return tuple(dict(zip(('id', 'needle', 'owner'), rule)) for rule in self._rules)

    def select(self, question):
        if not isinstance(question, str) or not question or len(question.encode()) > 32768:
            raise ValueError('Require bounded user text, without task or answer metadata')
        question = question.casefold()
        return next((identifier for identifier, needle, _ in self._rules if needle in question), None)

    def require_extension_of(self, previous):
        if len(self._rules) <= len(previous._rules) or self._rules[:len(previous._rules)] != previous._rules:
            raise ValueError('New routes must preserve every existing choice and its precedence')


class RoutedNetwork:
    """Share one parent partition across several independently connected paths."""

    def __init__(self, global_rank, shard, tokenizer, split, routes, parent_group, expert_groups):
        self.rank, self.routes = global_rank, routes
        self.parent_wire = ParentWire(global_rank, parent_group) if global_rank < 3 else None
        self.networks = {}
        for rule in routes.rules:
            members = [0, 1, 2, rule['owner']]
            if global_rank not in members:
                continue
            wire = GroupWire(global_rank, members, expert_groups[rule['id']])
            self.networks[rule['id']] = Network(shard, wire, self.parent_wire, tokenizer, split)
        if not self.networks:
            raise ValueError('This process owns no partition of the declared graph')

    def answer(self, question, max_tokens):
        selected = self.routes.select(question)
        if selected is None:
            if self.rank >= 3:
                return None
            net = next(iter(self.networks.values()))
            value = net.generate(question, max_tokens, False)
        else:
            if selected not in self.networks:
                return None
            value = self.networks[selected].generate(question, max_tokens, True)
        return {**value, 'expert': selected}
