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


UNIT_STORE = None  # directory of published unit tensors, ``<sha256>.safetensors``; default $NEUROSHARD_UNITS or .units


def unit_tensors(sha256):
    """A published unit's trainable tensors, accepted only if the file hashes to the pinned digest."""
    import os
    from pathlib import Path

    from safetensors.torch import load as load_tensors

    store = Path(UNIT_STORE or os.environ.get('NEUROSHARD_UNITS', '.units'))
    for path in [store / f'{sha256}.safetensors', *sorted(store.glob('*/trainable.safetensors'))]:
        if path.is_file():
            payload = path.read_bytes()
            if hashlib.sha256(payload).hexdigest() == sha256:
                return load_tensors(payload)
    raise FileNotFoundError(f'no published unit with trainable_sha256 {sha256}')


def start(model, recipe):
    """The job's start state: the arm attached from the recipe's seed, and a fresh optimizer.

    With ``continue_from`` (a published unit's ``trainable_sha256``) the arm starts from that unit's weights,
    so a growth round builds on the unit it would replace instead of starting over.
    """
    torch.manual_seed(recipe['spec']['seed'])
    trainable = trainer.prepare(model, recipe['arm'], recipe['spec'])
    if recipe.get('continue_from'):
        saved = unit_tensors(recipe['continue_from'])
        if set(saved) != set(trainable):
            raise ValueError('the continued unit names other tensors than the recipe attaches')
        with torch.no_grad():
            for name, value in trainable.items():
                value.copy_(saved[name].to(value.dtype))
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
    """The training state as bytes (a proof bundle's ``before``); ``load`` restores it exactly.

    ``order`` names the optimizer's parameters in its own order, so ``bytes_root`` can recompute the state root
    from the bytes alone, without a model: any store can check a state before accepting or serving it."""
    buffer = io.BytesIO()
    index = {id(value): name for name, value in trainable.items()}
    order = [index[id(p)] for group in optimizer.param_groups for p in group['params']]
    torch.save({'trainable': {name: value.detach().clone() for name, value in sorted(trainable.items())},
                'optimizer': optimizer.state_dict(), 'order': order}, buffer)
    return buffer.getvalue()


def bytes_root(payload):
    """The state root of saved state bytes, computed exactly as ``state_root`` does from live tensors."""
    saved = torch.load(io.BytesIO(payload), weights_only=True)
    trainable, order = saved['trainable'], saved['order']
    if sorted(order) != sorted(trainable):
        raise ValueError('state order names other tensors')
    value = hashlib.sha256()
    for name in sorted(trainable):
        value.update(name.encode() + b'\0' + _bytes(trainable[name]))
    states = saved['optimizer']['state']
    for name, state in sorted(((order[i], s) for i, s in states.items()), key=lambda row: row[0]):
        for key in sorted(state):
            item = state[key]
            payload_bytes = _bytes(item) if torch.is_tensor(item) else json.dumps(item).encode()
            value.update(f'{name}/{key}'.encode() + b'\0' + payload_bytes)
    return value.hexdigest()


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
    bound = {'model_root': model_root, 'arm': recipe['arm'], 'spec': recipe['spec'], 'sources': recipe['sources']}
    if recipe.get('continue_from'):
        bound['continue_from'] = recipe['continue_from']  # bound only when present: earlier roots are unchanged
    return hashlib.sha256(json.dumps(bound, sort_keys=True).encode()).hexdigest()
