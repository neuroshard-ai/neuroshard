"""Bounded, resumable training of the connection over committed causal features.

Frozen model weights and their output head are excluded from the optimizer.
The source-ablation arm substitutes the unchanged parent states for trained
specialist states; it is an information control, not a claim of optimal use of
a fixed total compute budget. Feature execution needs its separate verification.
"""
import copy
import math
from pathlib import Path

import torch
from torch.nn import functional as F
from safetensors.torch import load_file, save_file

from ..reference import autocast
from ..reference_data import identity, save, sha256
from .fused_graph import commitment


class Trainer:
    def __init__(self, fusion, head, rows, bank, home, recipe, *, source_ablation=False):
        fields = {'steps', 'learning_rate', 'warmup_steps', 'minimum_lr_ratio',
                  'weight_decay', 'clip_norm', 'microbatch', 'general_kl', 'schedule'}
        if (set(recipe) != fields or type(recipe['steps']) is not int or not 1 <= recipe['steps'] <= 4096
                or type(recipe['warmup_steps']) is not int or not 0 <= recipe['warmup_steps'] < recipe['steps']
                or type(recipe['microbatch']) is not int or not 1 <= recipe['microbatch'] <= 16
                or not isinstance(recipe['schedule'], list) or len(recipe['schedule']) != recipe['steps']
                or any(type(index) is not int or not 0 <= index < len(bank['files']) for index in recipe['schedule'])
                or type(source_ablation) is not bool):
            raise ValueError('Invalid frozen fusion training schedule')
        for key in ('learning_rate', 'minimum_lr_ratio', 'weight_decay', 'clip_norm', 'general_kl'):
            if type(recipe[key]) not in (int, float) or not math.isfinite(recipe[key]) or recipe[key] < 0:
                raise ValueError('Invalid finite fusion optimizer value')
        if (not 0 < recipe['learning_rate'] <= .1 or not 0 <= recipe['minimum_lr_ratio'] <= 1
                or not 0 < recipe['clip_norm'] <= 100 or recipe['general_kl'] > 100
                or recipe['weight_decay'] > 1 or bank['rows'] != identity(rows)
                or len(bank['files']) != len(bank['batches'])
                or set(bank['source_names']) != {'hub', *fusion.source_widths}
                or head.rank != 0 or any(parameter.requires_grad for parameter in head.parameters())):
            raise ValueError('Training changed its committed features, sources or frozen head')
        self.model, self.head, self.rows, self.bank = fusion, head, copy.deepcopy(rows), copy.deepcopy(bank)
        self.home, self.recipe, self.source_ablation = Path(home), copy.deepcopy(recipe), source_ablation
        self.step, self.failed = 0, False
        self.head_versions = tuple(parameter._version for parameter in head.parameters())
        self.model.train()
        self.optimizer = torch.optim.AdamW(fusion.parameters(), lr=recipe['learning_rate'],
            betas=(.9, .999), eps=1e-8, weight_decay=recipe['weight_decay'], foreach=False)
        self.binding = {'bank': identity(bank), 'recipe': identity(recipe), 'layout': fusion.descriptor(),
                        'source_ablation': source_ablation}

    def batch(self, index):
        spec = self.bank['files'][index]
        path = self.home/(spec['sha256']+'.safetensors')
        if path.stat().st_size != spec['bytes'] or sha256(path) != spec['sha256']:
            raise ValueError('Fusion feature bytes differ from their commitment')
        values = load_file(path, device='cpu')
        selected = [self.rows[index] for index in self.bank['batches'][index]]
        if (spec['rows'] != [row['id'] for row in selected]
                or set(values) != {'input_ids', 'labels', 'valid', *self.bank['source_names']}
                or values['labels'].dtype != torch.int64 or values['input_ids'].dtype != torch.int64
                or values['valid'].dtype != torch.bool):
            raise ValueError('Fusion feature inventory or target types changed')
        shape = tuple(spec['shape'])
        if (len(shape) != 3 or shape[0] != len(selected) or shape[2] != self.model.hub_width
                or any(tuple(values[key].shape) != shape or values[key].dtype != torch.float32
                       or not bool(torch.isfinite(values[key]).all()) for key in self.bank['source_names'])
                or any(tuple(values[key].shape) != shape[:2] for key in ('input_ids', 'labels', 'valid'))):
            raise ValueError('Fusion feature shapes or finite values changed')
        for index, row in enumerate(selected):
            count = len(row['input_ids'])
            if (values['input_ids'][index, :count].tolist() != row['input_ids']
                    or values['labels'][index, :count].tolist() != row['labels']
                    or not bool(values['valid'][index, :count].all())
                    or bool(values['valid'][index, count:].any())
                    or bool((values['labels'][index, count:] != -100).any())):
                raise ValueError('Feature bank targets differ from original training messages')
        return values, selected

    def advance(self):
        if self.failed:
            raise ValueError('Discard failed fusion training state and restore a complete checkpoint')
        if self.step >= self.recipe['steps']:
            raise ValueError('The prescribed fusion training has completed')
        self.failed = True
        if self.head_versions != tuple(parameter._version for parameter in self.head.parameters()):
            raise ValueError('The frozen output owner changed during fusion training')
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
        total, cross_entropy, divergence = 0., 0., 0.
        for start in range(0, len(rows), self.recipe['microbatch']):
            stop = min(start+self.recipe['microbatch'], len(rows))
            values = {key: value[start:stop].to(device) for key, value in data.items()}
            experts = {name: values['parent' if self.source_ablation else name]
                       for name in self.model.source_widths}
            hidden = self.model(values['hub'], experts, valid=values['valid'])
            with autocast(self.head.device_name):
                logits = self.head.logits(hidden).float()
            labels = values['labels'][:, 1:]
            mask = labels != -100
            counts = mask.sum(dim=1)
            losses = F.cross_entropy(logits[:, :-1].transpose(1, 2), labels, reduction='none')
            ce = losses.sum(dim=1)/counts
            general = torch.tensor([row['kind'] == 'general' for row in rows[start:stop]], device=device)
            kl = torch.zeros_like(ce)
            if bool(general.any()) and self.recipe['general_kl']:
                with torch.no_grad(), autocast(self.head.device_name):
                    teacher = self.head.logits(values['hub']).float()[:, :-1].log_softmax(dim=-1)
                student = logits[:, :-1].log_softmax(dim=-1)
                per_token = (teacher.exp()*(teacher-student)).sum(dim=-1)
                kl = (per_token*mask).sum(dim=1)/counts*general
            loss = (ce+self.recipe['general_kl']*kl).sum()/len(rows)
            if not bool(torch.isfinite(loss)):
                raise ValueError('Nonfinite fusion training objective')
            loss.backward()
            total += float(loss.detach())
            cross_entropy += float(ce.detach().sum())/len(rows)
            divergence += float(kl.detach().sum())/len(rows)
        norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.recipe['clip_norm'],
                                             error_if_nonfinite=True, foreach=False)
        self.optimizer.step()
        self.step += 1
        root = commitment(self.model)
        self.failed = False
        return {'step': self.step, 'fusion': root, 'loss': total, 'response_ce': cross_entropy,
                'general_kl': divergence, 'gradient_norm': float(norm), 'learning_rate': rate}

    def save(self, home):
        if self.failed:
            raise ValueError('Cannot persist failed fusion training state')
        home = Path(home)
        home.mkdir(parents=True, exist_ok=True)
        tensors = {}
        for name, parameter in self.model.named_parameters():
            tensors['weight/'+name] = parameter.detach().cpu().contiguous()
            state = self.optimizer.state.get(parameter, {})
            if self.step:
                for field in ('step', 'exp_avg', 'exp_avg_sq'):
                    tensors['adam/'+name+'/'+field] = state[field].detach().cpu().contiguous()
        temporary = home/'checkpoint.pending'
        save_file(tensors, str(temporary))
        digest = sha256(temporary)
        temporary.replace(home/(digest+'.safetensors'))
        value = {'format': 'neuroshard-fusion-training-checkpoint-v1', 'binding': self.binding,
                 'step': self.step, 'fusion': commitment(self.model), 'sha256': digest,
                 'bytes': (home/(digest+'.safetensors')).stat().st_size}
        save(home/('step-'+str(self.step)+'.json'), value)
        return value

    def restore(self, home, checkpoint):
        if (self.failed or self.step != 0 or checkpoint.get('format') != 'neuroshard-fusion-training-checkpoint-v1'
                or checkpoint['binding'] != self.binding
                or type(checkpoint['step']) is not int or not 0 <= checkpoint['step'] <= self.recipe['steps']):
            raise ValueError('Restore only the identical fusion training prescription')
        self.failed = True
        path = Path(home)/(checkpoint['sha256']+'.safetensors')
        if path.stat().st_size != checkpoint['bytes'] or sha256(path) != checkpoint['sha256']:
            raise ValueError('Fusion optimizer checkpoint bytes changed')
        values = load_file(path, device='cpu')
        expected = {'weight/'+name for name, _ in self.model.named_parameters()}
        if checkpoint['step']:
            expected |= {'adam/'+name+'/'+field for name, _ in self.model.named_parameters()
                         for field in ('step', 'exp_avg', 'exp_avg_sq')}
        if set(values) != expected or any(not bool(torch.isfinite(value).all()) for value in values.values()):
            raise ValueError('Fusion checkpoint tensor inventory or finite state changed')
        with torch.no_grad():
            for name, parameter in self.model.named_parameters():
                value = values['weight/'+name]
                if value.dtype != parameter.dtype or value.shape != parameter.shape:
                    raise ValueError('Fusion checkpoint parameter layout changed')
                parameter.copy_(value)
                if checkpoint['step']:
                    state = {field: values['adam/'+name+'/'+field] for field in ('step', 'exp_avg', 'exp_avg_sq')}
                    if (state['step'].numel() != 1 or float(state['step']) != checkpoint['step']
                            or any(state[field].shape != parameter.shape or state[field].dtype != parameter.dtype
                                   for field in ('exp_avg', 'exp_avg_sq'))
                            or bool((state['exp_avg_sq'] < 0).any())):
                        raise ValueError('Fusion Adam ages or moments changed')
                    self.optimizer.state[parameter] = {field: value.to(parameter.device) if field != 'step' else value
                                                      for field, value in state.items()}
        self.step = checkpoint['step']
        if commitment(self.model) != checkpoint['fusion']:
            raise ValueError('Restored fusion differs from its model commitment')
        self.failed = False
