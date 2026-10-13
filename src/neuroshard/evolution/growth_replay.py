"""The protocol trainer for growth jobs: segment training that commits a state root per step, and one-step replay.

Contributors train a growth job's segments with ``train_segment``; validators judge a disputed step with
``judge``, which re-executes exactly that step from the committed state before it. Both use the same
code, the same recipe and a pinned execution class, so an honest step reproduces bit for bit (as A4
showed across owners) and a dishonest one is provable.

A state root covers everything the next step depends on: every trainable tensor and the optimizer's
moments and step counts, as raw bytes in name order. The recipe fixes the model, arm, spec, data and
schedule; the step index fixes the microbatch.
"""

import hashlib
import io
import json

import torch

from neuroshard.evolution import assistant_experience_train as trainer


def updates(recipe):
    spec, sources = recipe['spec'], recipe['sources']
    if 'mixture' in spec:
        return trainer.mixture_schedule(sources, spec)
    return trainer.schedule(sources['experience'], sources.get('replay', []), spec)


def _optimizer(trainable, spec, arm):
    return torch.optim.AdamW(list(trainable.values()), lr=trainer.learning_rate(0, spec, arm),
                             betas=tuple(spec['betas']), eps=spec['epsilon'], weight_decay=spec['weight_decay'])


def start(model, recipe):
    """The job's start state: the arm attached from the recipe's seed, and a fresh optimizer."""
    torch.manual_seed(recipe['spec']['seed'])
    trainable = trainer.prepare(model, recipe['arm'], recipe['spec'])
    model.train()
    return trainable, _optimizer(trainable, recipe['spec'], recipe['arm'])


def _bytes(tensor):
    return tensor.detach().contiguous().cpu().reshape(-1).view(torch.uint8).numpy().tobytes()


def state_root(trainable, optimizer):
    """Digest of the training state: trainable tensors, then the optimizer's per-parameter state, by name."""
    value = hashlib.sha256()
    names = sorted(trainable)
    index = {id(trainable[name]): name for name in names}
    for name in names:
        value.update(name.encode() + b'\0' + _bytes(trainable[name]))
    for parameter, state in sorted(((index[id(p)], s) for p, s in optimizer.state.items()), key=lambda row: row[0]):
        for key in sorted(state):
            item = state[key]
            payload = _bytes(item) if torch.is_tensor(item) else json.dumps(item).encode()
            value.update(f'{parameter}/{key}'.encode() + b'\0' + payload)
    return value.hexdigest()


def step(model, trainable, optimizer, recipe, index, schedule=None):
    """Exactly one optimizer step ``index`` of the recipe's schedule; the loop body of ``trainer.train``."""
    spec, arm, sources = recipe['spec'], recipe['arm'], recipe['sources']
    batch = (schedule or updates(recipe))[index]
    for group in optimizer.param_groups:
        group['lr'] = trainer.learning_rate(index, spec, arm)
    optimizer.zero_grad(set_to_none=True)
    total = 0.0
    for kind, row in batch:
        loss = trainer.sequence_loss(model, sources[kind][row], 'cpu') / len(batch)
        loss.backward()
        total += loss.detach().item()
    norm = torch.nn.utils.clip_grad_norm_(list(trainable.values()), spec['gradient_clip'])
    if not torch.isfinite(norm):
        raise ValueError('nonfinite gradient norm')
    optimizer.step()
    return total


def train_segment(model, trainable, optimizer, recipe, first, last):
    """Steps ``first`` to ``last - 1``; returns the state root after each, for a ``growth_commit``."""
    schedule = updates(recipe)
    roots = []
    for index in range(first, last):
        step(model, trainable, optimizer, recipe, index, schedule)
        roots.append(state_root(trainable, optimizer))
    return roots


def save(trainable, optimizer):
    """The training state as bytes (a proof bundle's ``before``); ``load`` restores it exactly."""
    buffer = io.BytesIO()
    torch.save({'trainable': {name: value.detach().clone() for name, value in sorted(trainable.items())},
                'optimizer': optimizer.state_dict()}, buffer)
    return buffer.getvalue()


def load(model, recipe, payload):
    trainable, optimizer = start(model, recipe)
    saved = torch.load(io.BytesIO(payload), weights_only=True)
    if set(saved['trainable']) != set(trainable):
        raise ValueError('proof state names other tensors than the recipe attaches')
    with torch.no_grad():
        for name, value in trainable.items():
            value.copy_(saved['trainable'][name])
    optimizer.load_state_dict(saved['optimizer'])
    return trainable, optimizer


def bundle(before, step_index):
    """A growth proof bundle: the state before the disputed step, and that step's index."""
    return json.dumps({'format': 'neuroshard-growth-proof/1', 'step': step_index,
                       'before': before.hex()}, sort_keys=True).encode()


def judge(make_model, recipe, payload, request):
    """True if the bundle proves the committed root of ``request['step']`` wrong.

    The bundle's state must hash to the committed root before that step, and re-executing the step from
    it must give a different root than the one committed. A bundle whose state does not match proves
    nothing (False).
    """
    if hashlib.sha256(payload).hexdigest() != request['proof_root']:
        return False
    proof = json.loads(payload)
    if proof.get('format') != 'neuroshard-growth-proof/1' or proof.get('step') != request['step']:
        return False
    model = make_model()
    trainable, optimizer = load(model, recipe, bytes.fromhex(proof['before']))
    if state_root(trainable, optimizer) != request['before_root']:
        return False
    step(model, trainable, optimizer, recipe, request['step'])
    return state_root(trainable, optimizer) != request['claimed_root']


def recipe_root(recipe, model_root):
    """What a ``growth_open`` names: the recipe's arm, spec and data, and the model it trains."""
    return hashlib.sha256(json.dumps({'model_root': model_root, 'arm': recipe['arm'], 'spec': recipe['spec'],
                                      'sources': recipe['sources']}, sort_keys=True).encode()).hexdigest()
