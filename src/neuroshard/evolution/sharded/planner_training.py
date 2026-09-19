"""Fit an owned planner interface without changing the answering model.

The preserved assistant's first two owners produce frozen activations. Its last
owner fits only a separate adapter; the output owner returns hidden-state
gradients. Ordinary answering never installs this planner adapter. Training
labels describe calls and arguments, rather than factual reference answers.
"""
from contextlib import contextmanager, nullcontext
import copy
import math

import torch
from torch.nn import functional as F

from ..reference import autocast
from ..reference_data import identity
from .expert_interface import ExpertInterface
from .fused_graph import commitment
from .interface_training import AdapterState, initialize_weights
from .model import batch_tensors

FORMAT = 'neuroshard-owned-planner-training-v1'


def source(net):
    return identity(net.graph['interpreter_assets']['partitions']['2'])


@contextmanager
def installed(net, adapter):
    """Install only for planning; every owner binds the same adapter root."""
    valid = ((adapter is not None) == (net.rank == 2)
             and (adapter is None or not adapter.training and adapter.source_root == source(net)))
    if not all(net.all_owners.exchange(valid)):
        raise ValueError('A planner adapter must bind only the preserved final owner')
    roots = net.all_owners.exchange(commitment(adapter) if adapter is not None else None)
    versions = tuple(p._version for p in adapter.parameters()) if adapter is not None else ()
    with adapter.attach(net.preserved.shard) if adapter is not None else nullcontext():
        yield roots[2]
        changed = adapter is not None and versions != tuple(p._version for p in adapter.parameters())
        if any(net.all_owners.exchange(changed)):
            raise ValueError('The planner adapter changed during inference')
    net.verify_unchanged()


class PlannerTraining:
    def __init__(self, net, rows, recipe, *, adapter_rank=8, max_length=1024,
                 initial_weights=None, weights_home=None):
        fields = {'steps', 'learning_rate', 'warmup_steps', 'minimum_lr_ratio',
                  'weight_decay', 'clip_norm', 'microbatch', 'schedule'}
        if (set(recipe) != fields or type(recipe['steps']) is not int or not 1 <= recipe['steps'] <= 4096
                or type(recipe['warmup_steps']) is not int or not 0 <= recipe['warmup_steps'] < recipe['steps']
                or type(recipe['microbatch']) is not int or not 1 <= recipe['microbatch'] <= 16
                or type(max_length) is not int or not 2 <= max_length <= 1024
                or type(adapter_rank) is not int or not 1 <= adapter_rank <= 128
                or not isinstance(rows, list) or not 1 <= len(rows) <= 2048
                or not isinstance(recipe['schedule'], list) or len(recipe['schedule']) != recipe['steps']):
            raise ValueError('Require a bounded complete planner prescription')
        for key, upper in [('learning_rate', .01), ('minimum_lr_ratio', 1), ('weight_decay', 1), ('clip_norm', 100)]:
            value = recipe[key]
            if type(value) not in (int, float) or not math.isfinite(value) or not 0 <= value <= upper:
                raise ValueError('Invalid finite planner optimizer value')
        if recipe['learning_rate'] == 0 or recipe['clip_norm'] == 0:
            raise ValueError('Planner learning and clipping must be positive')
        vocabulary = net.graph['parent']['config']['vocab_size']
        for row in rows:
            ids, labels = row['input_ids'], row['labels']
            if (not isinstance(ids, list) or not 2 <= len(ids) <= max_length
                    or not isinstance(labels, list) or len(labels) != len(ids) or labels[0] != -100
                    or any(type(token) is not int or not 0 <= token < vocabulary for token in ids)
                    or any(type(label) is not int or label not in (-100, token) for token, label in zip(ids, labels))
                    or type(row['targets']) is not int or row['targets'] != sum(label != -100 for label in labels[1:])
                    or row['targets'] < 1):
                raise ValueError('Planner targets must reproduce complete response-only tokens')
        for batch in recipe['schedule']:
            if (not isinstance(batch, list) or not 1 <= len(batch) <= 64 or len(set(batch)) != len(batch)
                    or any(type(index) is not int or not 0 <= index < len(rows) for index in batch)):
                raise ValueError('Commit every complete planner batch')
        self.net, self.rows, self.recipe = net, copy.deepcopy(rows), copy.deepcopy(recipe)
        self.step, self.failed, self.adapter, self.state = 0, False, None, None
        binding = {'format': FORMAT, 'graph': identity(net.graph), 'source': source(net),
                   'rows': identity(rows), 'recipe': identity(recipe), 'max_length': max_length,
                   'adapter_rank': adapter_rank}
        if initial_weights is not None:
            if initial_weights['binding']['source'] != source(net):
                raise ValueError('Initial planner weights belong to another preserved owner')
            binding['initial_weights'] = copy.deepcopy(initial_weights)
        if net.all_owners.exchange(identity(binding)) != [identity(binding)]*net.world_size:
            raise ValueError('Owners disagree on planner training')
        if net.rank == 2:
            self.adapter = ExpertInterface(net.preserved.shard, source(net), adapter_rank)
            if initial_weights is not None:
                initialize_weights(self.adapter, weights_home, initial_weights)
            self.state = AdapterState(self.adapter, recipe, {**binding, 'layout': self.adapter.descriptor()})

    def save(self, home):
        if self.failed:
            raise ValueError('Cannot save an incomplete planner update')
        return self.net.all_owners.exchange(self.state.save(home) if self.state else None)[2]

    def restore(self, home, checkpoint):
        if self.step or self.failed:
            raise ValueError('Restore only a fresh planner training state')
        self.failed = True
        if self.state:
            self.state.restore(home, checkpoint)
        steps = self.net.all_owners.exchange(self.state.step if self.state else None)
        if type(steps[2]) is not int or any(value is not None for rank, value in enumerate(steps) if rank != 2):
            raise ValueError('Only the planner owner may restore optimizer state')
        self.step, self.failed = steps[2], False

    def advance(self):
        if self.failed or self.step >= self.recipe['steps']:
            raise ValueError('Restore failed state or stop at the prescribed planner update count')
        self.failed = True
        net, recipe = self.net, self.recipe
        net.verify_unchanged()
        wire, rank, device = net.all_owners, net.rank, net.shard.device_name
        selected = [self.rows[index] for index in recipe['schedule'][self.step]]
        warmup = recipe['warmup_steps']
        progress = (self.step-warmup)/max(1, recipe['steps']-warmup-1)
        factor = ((self.step+1)/warmup if self.step < warmup else recipe['minimum_lr_ratio']+
                  (1-recipe['minimum_lr_ratio'])*.5*(1+math.cos(math.pi*progress)))
        rate, loss_sum = recipe['learning_rate']*factor, 0.
        if self.state:
            self.state.failed = True
            self.state.optimizer.zero_grad(set_to_none=True)
            for group in self.state.optimizer.param_groups:
                group['lr'] = rate
        scope = self.adapter.attach(net.preserved.shard) if self.adapter is not None else nullcontext()
        with scope:
            for start in range(0, len(selected), recipe['microbatch']):
                rows = selected[start:start+recipe['microbatch']]
                ids, labels, mask, _ = batch_tensors(rows, device)
                shape = (*ids.shape, net.graph['parent']['config']['hidden_size'])
                if rank < 2:
                    incoming = ids if rank == 0 else wire.receive(0, shape, device)
                    with torch.no_grad(), autocast(device):
                        hidden = net.preserved.shard(incoming, mask)
                    wire.send(hidden, rank+1)
                elif rank == 2:
                    incoming = wire.receive(1, shape, device)
                    with autocast(device):
                        hidden = net.preserved.shard(incoming, mask)
                    wire.send(hidden, 0)
                if rank == 0:
                    output = wire.receive(2, shape, device).requires_grad_(True)
                    with autocast(device):
                        logits = net.preserved.shard.logits(output).float()
                    targets = labels[:, 1:]
                    per_token = F.cross_entropy(logits[:, :-1].transpose(1, 2), targets, reduction='none')
                    loss = (per_token.sum(dim=1)/(targets != -100).sum(dim=1)).sum()/len(selected)
                    if not bool(torch.isfinite(loss)):
                        raise ValueError('Nonfinite planner objective')
                    loss.backward()
                    wire.send(output.grad, 2)
                    loss_sum += float(loss.detach())
                elif rank == 2:
                    hidden.backward(wire.receive(0, shape, device))
        norm, model_root = None, None
        if self.state:
            norm = float(torch.nn.utils.clip_grad_norm_(self.adapter.parameters(), recipe['clip_norm'],
                                                       error_if_nonfinite=True, foreach=False))
            self.state.optimizer.step()
            self.state.step += 1
            self.state.failed = False
            model_root = commitment(self.adapter)
        packet = wire.exchange({'loss': loss_sum if rank == 0 else None, 'norm': norm, 'model': model_root})
        net.verify_unchanged()
        self.step += 1
        self.failed = False
        return {'step': self.step, 'loss': packet[0]['loss'], 'gradient_norm': packet[2]['norm'],
                'planner': packet[2]['model'], 'learning_rate': rate}
