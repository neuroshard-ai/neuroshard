"""Backpropagate a joint token objective to separately owned expert adapters.

The frozen common prefix runs once to form a committed feature bank. The root
owns the vocabulary head and routing gate; tail owners receive prefix tensors
and return hidden states. Only hidden-state gradients cross the return boundary.
No participant constructs the full model or receives another owner's weights.
"""
from contextlib import nullcontext
import math

import torch
from torch.nn import functional as F

from ..reference import autocast
from ..reference_data import identity
from .expert_interface import supervision
from .fused_graph import commitment
from .fusion_training import Trainer
from .mixture_training import MixtureTrainer, native_logits


class AdapterState:
    save = Trainer.save
    restore = Trainer.restore

    def __init__(self, model, recipe, binding):
        self.model, self.recipe, self.binding = model, recipe, binding
        self.step, self.failed = 0, False
        self.optimizer = torch.optim.AdamW(model.parameters(), lr=recipe['learning_rate'],
            betas=(.9, .999), eps=1e-8, weight_decay=recipe['weight_decay'], foreach=False)


class OwnedInterfaceTraining:
    def __init__(self, net, gate, adapter, rows, bank, home, recipe, objective, *, enabled=True):
        if (type(enabled) is not bool or bank.get('boundary_names') != ['prefix']
                or set(objective) != {'source_ce', 'route_ce', 'adapter_lr'}
                or any(type(v) not in (int, float) or not math.isfinite(v) or v < 0 for v in objective.values())
                or objective['source_ce'] > 10 or objective['route_ce'] > 10
                or not 0 < objective['adapter_lr'] <= .01
                or (adapter is None) != (net.rank < 3)):
            raise ValueError('Invalid owned interface training prescription')
        self.net, self.gate, self.adapter = net, gate, adapter
        self.rows, self.bank, self.recipe, self.objective = rows, bank, recipe, objective
        self.enabled, self.step, self.failed = enabled, 0, False
        self.owners = {row['id']: row['owner'] for row in net.graph['descriptor']['rules']}
        binding = {'graph': identity(net.graph), 'bank': identity(bank), 'recipe': identity(recipe),
                   'objective': objective, 'enabled': enabled}
        if net.all_owners.exchange(identity(binding)) != [identity(binding)]*net.world_size:
            raise ValueError('Owners disagree on interface training')
        self.state, self.reader = None, None
        if net.rank == 0:
            self.reader = MixtureTrainer(gate, net.preserved.shard, rows, bank, home, recipe, source_head=net.shard)
            self.reader.binding['interface_training'] = binding
            self.state = self.reader
            self.labels = [supervision(row, net.tokenizer, self.owners) for row in rows]
        elif adapter is not None and enabled:
            self.state = AdapterState(adapter, {**recipe, 'learning_rate': objective['adapter_lr']},
                {**binding, 'layout': adapter.descriptor()})

    def restore(self, home, record):
        if self.step or self.failed or len(record) != self.net.world_size:
            raise ValueError('Restore only a fresh complete owned interface state')
        local = record[self.net.rank]
        if self.state is not None:
            self.state.restore(home, local)
        elif local is not None:
            raise ValueError('Checkpoint assigned trainable weights to a frozen owner')
        steps = {value['step'] for value in record if value is not None}
        if len(steps) != 1:
            raise ValueError('Owners saved different optimizer ages')
        self.step = steps.pop()

    def save(self, home):
        if self.failed:
            raise ValueError('Cannot save an incomplete owned interface update')
        return self.net.all_owners.exchange(self.state.save(home) if self.state is not None else None)

    def advance(self):
        net, wire, rank = self.net, self.net.all_owners, self.net.rank
        if self.failed or self.step >= self.recipe['steps']:
            raise ValueError('Restore failed state or stop at the prescribed interface update count')
        self.failed = True
        if self.state:
            self.state.failed = True
            self.state.optimizer.zero_grad(set_to_none=True)
            warmup = self.recipe['warmup_steps']
            progress = (self.step-warmup)/max(1, self.recipe['steps']-warmup-1)
            factor = ((self.step+1)/warmup if self.step < warmup else self.recipe['minimum_lr_ratio']+
                      (1-self.recipe['minimum_lr_ratio'])*.5*(1+math.cos(math.pi*progress)))
            for group in self.state.optimizer.param_groups:
                group['lr'] = self.state.recipe['learning_rate']*factor
        net.verify_unchanged()
        batch = self.recipe['schedule'][self.step]
        indices = self.bank['batches'][batch]
        selected = [self.rows[index] for index in indices]
        data = self.reader.batch(batch)[0] if rank == 0 else None
        length, width = max(len(row['input_ids']) for row in selected), self.gate.hub_width
        device, losses = net.shard.device_name, {'response_ce': 0., 'general_kl': 0., 'source_ce': 0., 'route_ce': 0.}
        context = self.adapter.attach(net.shard) if self.enabled and self.adapter is not None else nullcontext()
        with context:
            for start in range(0, len(selected), self.recipe['microbatch']):
                stop = min(len(selected), start+self.recipe['microbatch'])
                shape = (stop-start, length, width)
                outgoing = None
                if rank == 0:
                    values = {key: value[start:stop].to(device) for key, value in data.items()}
                    if self.enabled:
                        for owner in self.owners.values():
                            wire.send(values['prefix'], owner)
                        hidden = {name: wire.receive(owner, shape, device).requires_grad_()
                                  for name, owner in self.owners.items()}
                    else:
                        hidden = {name: values[name] for name in self.owners}
                    logits = native_logits(net.preserved.shard, net.shard, values, self.gate.source_widths, False)
                    if self.enabled:
                        with autocast(device):
                            for name, value in hidden.items():
                                logits[name] = net.shard.logits(value).float()
                    labels = values['labels'][:, 1:]
                    mask, count = labels != -100, (labels != -100).sum(dim=1)
                    distribution = self.gate.log_probs(values['hub'], logits)[:, :-1]
                    ce = F.nll_loss(distribution.transpose(1, 2), labels, reduction='none').sum(dim=1)/count
                    general = torch.tensor([row['kind'] == 'general' for row in selected[start:stop]], device=device)
                    teacher = logits['hub'][:, :-1].log_softmax(dim=-1)
                    kl = ((teacher.exp()*(teacher-distribution)).sum(dim=-1)*mask).sum(dim=1)/count*general
                    source_ce, route_ce = torch.zeros_like(ce), torch.zeros_like(ce)
                    gate_hidden = F.silu(self.gate.query(F.layer_norm(values['hub'], (width,), eps=1e-5)))
                    scores = self.gate.output(gate_hidden)[:, :-1]
                    route_scores = torch.cat((torch.zeros_like(scores[:, :, :1]), scores), dim=-1)
                    for name in self.owners:
                        owned_labels = torch.full_like(values['labels'], -100)
                        for index, original in enumerate(indices[start:stop]):
                            local = self.labels[original][name]
                            owned_labels[index, :len(local)] = torch.tensor(local, device=device)
                        owned_labels = owned_labels[:, 1:]
                        active = owned_labels != -100
                        counts = active.sum(dim=1).clamp_min(1)
                        if self.enabled:
                            source_ce += F.cross_entropy(logits[name][:, :-1].transpose(1, 2), owned_labels,
                                reduction='none').sum(dim=1)/counts
                        route_labels = torch.full_like(owned_labels, -100)
                        route_labels[active] = list(self.gate.source_widths).index(name)+1
                        route_ce += F.cross_entropy(route_scores.transpose(1, 2), route_labels,
                            reduction='none').sum(dim=1)/counts
                    total = (ce+self.recipe['general_kl']*kl+self.objective['source_ce']*source_ce+
                             self.objective['route_ce']*route_ce).sum()/len(selected)
                    if not bool(torch.isfinite(total)):
                        raise ValueError('Nonfinite joint interface objective')
                    total.backward()
                    for key, value in zip(losses, (ce, kl, source_ce, route_ce)):
                        losses[key] += float(value.detach().sum())/len(selected)
                    if self.enabled:
                        for name, owner in self.owners.items():
                            wire.send(hidden[name].grad, owner)
                elif rank in self.owners.values() and self.enabled:
                    prefix = wire.receive(0, shape, device)
                    mask = torch.zeros(shape[:2], dtype=torch.long, device=device)
                    for index, row in enumerate(selected[start:stop]):
                        mask[index, :len(row['input_ids'])] = 1
                    with autocast(device):
                        outgoing = net.shard(prefix, mask)
                    wire.send(outgoing, 0)
                    outgoing.backward(wire.receive(0, shape, device))
        norm = None
        if self.state:
            norm = float(torch.nn.utils.clip_grad_norm_(self.state.model.parameters(), self.recipe['clip_norm'],
                                                       error_if_nonfinite=True, foreach=False))
            self.state.optimizer.step()
            self.state.step += 1
            self.state.failed = False
        net.verify_unchanged()
        self.step += 1
        record = {'step': self.step, 'root': commitment(self.state.model) if self.state else None,
                  'gradient_norm': norm, 'losses': losses if rank == 0 else None}
        observed = wire.exchange(record)
        if any(value['step'] != self.step for value in observed):
            raise ValueError('Owners completed different interface optimizer steps')
        self.failed = False
        return observed
