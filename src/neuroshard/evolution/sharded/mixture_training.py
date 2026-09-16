"""Train only the causal probability gate over verified frozen activations."""
import math

import torch
from torch.nn import functional as F

from ..reference import autocast
from .fused_graph import commitment
from .fusion_training import Trainer


@torch.no_grad()
def native_logits(hub_head, source_head, data, sources, ablation):
    with autocast(hub_head.device_name):
        result = {'hub': hub_head.logits(data['hub']).float(),
                  'parent': source_head.logits(data['parent']).float()}
        for source in sources:
            if source != 'parent':
                result[source] = result['parent'] if ablation else source_head.logits(data[source]).float()
    return result


class MixtureTrainer(Trainer):
    def __init__(self, fusion, head, rows, bank, home, recipe, *, source_head, source_ablation=False):
        super().__init__(fusion, head, rows, bank, home, recipe, source_ablation=source_ablation)
        if source_head.rank != 0 or any(parameter.requires_grad for parameter in source_head.parameters()):
            raise ValueError('Require the frozen native source output head')
        self.source_head = source_head
        self.source_head_versions = tuple(parameter._version for parameter in source_head.parameters())

    def advance(self):
        if self.failed or self.step >= self.recipe['steps']:
            raise ValueError('Discard failed or completed mixture training state')
        self.failed = True
        if (self.head_versions != tuple(p._version for p in self.head.parameters())
                or self.source_head_versions != tuple(p._version for p in self.source_head.parameters())):
            raise ValueError('A frozen native output head changed')
        data, rows = self.batch(self.recipe['schedule'][self.step])
        device = next(self.model.parameters()).device
        warmup = self.recipe['warmup_steps']
        progress = (self.step-warmup)/max(1, self.recipe['steps']-warmup-1)
        multiplier = ((self.step+1)/warmup if self.step < warmup else
            self.recipe['minimum_lr_ratio']+(1-self.recipe['minimum_lr_ratio'])*.5*(1+math.cos(math.pi*progress)))
        rate = self.recipe['learning_rate']*multiplier
        for group in self.optimizer.param_groups:
            group['lr'] = rate
        self.optimizer.zero_grad(set_to_none=True)
        total, ce_total, kl_total = 0., 0., 0.
        for start in range(0, len(rows), self.recipe['microbatch']):
            stop = start+self.recipe['microbatch']
            values = {key: value[start:stop].to(device) for key, value in data.items()}
            logits = native_logits(self.head, self.source_head, values, self.model.source_widths, self.source_ablation)
            distribution = self.model.log_probs(values['hub'], logits)[:, :-1]
            labels = values['labels'][:, 1:]
            mask, count = labels != -100, (labels != -100).sum(dim=1)
            ce = F.nll_loss(distribution.transpose(1, 2), labels, reduction='none').sum(dim=1)/count
            general = torch.tensor([row['kind'] == 'general' for row in rows[start:stop]], device=device)
            kl = torch.zeros_like(ce)
            if bool(general.any()) and self.recipe['general_kl']:
                teacher = logits['hub'][:, :-1].log_softmax(dim=-1)
                kl = ((teacher.exp()*(teacher-distribution)).sum(dim=-1)*mask).sum(dim=1)/count*general
            loss = (ce+self.recipe['general_kl']*kl).sum()/len(rows)
            if not bool(torch.isfinite(loss)):
                raise ValueError('Nonfinite mixture objective')
            loss.backward()
            total += float(loss.detach())
            ce_total += float(ce.detach().sum())/len(rows)
            kl_total += float(kl.detach().sum())/len(rows)
        norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.recipe['clip_norm'],
                                             error_if_nonfinite=True, foreach=False)
        self.optimizer.step()
        self.step += 1
        root = commitment(self.model)
        self.failed = False
        return {'step': self.step, 'fusion': root, 'loss': total, 'response_ce': ce_total,
                'general_kl': kl_total, 'gradient_norm': float(norm), 'learning_rate': rate}


@torch.no_grad()
def response_losses(models, head, rows, bank, home, recipe, *, source_head):
    reader = MixtureTrainer(models['fusion'], head, rows, bank, home,
        {**recipe, 'steps': 1, 'warmup_steps': 0, 'schedule': [0]}, source_head=source_head)
    for model in models.values():
        model.eval()
    result, device = {}, next(head.parameters()).device
    for batch in range(len(bank['batches'])):
        data, selected = reader.batch(batch)
        for start in range(0, len(selected), recipe['microbatch']):
            stop = start+recipe['microbatch']
            values = {key: value[start:stop].to(device) for key, value in data.items()}
            labels, counts = values['labels'][:, 1:], (values['labels'][:, 1:] != -100).sum(dim=1)
            logits = native_logits(head, source_head, values, models['fusion'].source_widths, False)
            distributions = {'hub': logits['hub'].log_softmax(dim=-1),
                             'fusion': models['fusion'].log_probs(values['hub'], logits),
                             'ablation': models['ablation'].log_probs(values['hub'],
                                {name: logits['hub' if name == 'hub' else 'parent'] for name in logits})}
            losses = {arm: (F.nll_loss(value[:, :-1].transpose(1, 2), labels, reduction='none').sum(dim=1)/counts).tolist()
                      for arm, value in distributions.items()}
            for index, row in enumerate(selected[start:stop]):
                result[row['id']] = {arm: value[index] for arm, value in losses.items()}
    return result
