"""The protocol trainer for growth jobs: segment training that commits a state root per step, and one-step replay.

Contributors train a growth job's segments with ``train_segment``; validators judge a disputed step with
``judge``, which re-executes exactly that step from the committed state before it. Both use the same
code, the same recipe and a pinned execution class, so an honest step reproduces bit for bit (as A4
showed across owners) and a dishonest one is provable.

A state root covers everything the next step depends on: every trainable tensor and the optimizer's
moments and step counts, as raw bytes in name order. The recipe fixes the model, arm, spec, data and
schedule; the step index fixes the microbatch.

**Thread count.** A step's rounding depends on the number of CPU threads it runs on, so the protocol pins it:
every step runs on ``STEP_THREADS`` threads, whatever the calling process or thread is set to, and the caller's
setting is restored afterwards. Without the pin, a contributor training at torch's default (four on an
eight-vCPU host) and a validator that set eight threads for its shards compute different roots for the same
honest step, and the validator would prove fraud that never happened.
"""

import contextlib
import hashlib
import io
import json

import torch

from neuroshard.evolution import assistant_experience_train as trainer

# Every committed live growth step was trained at torch's default on the pinned eight-vCPU class: four threads.
STEP_THREADS = 4


@contextlib.contextmanager
def step_threads():
    """Run the enclosed work on ``STEP_THREADS`` torch threads (a per-thread setting); restore the caller's."""
    previous = torch.get_num_threads()
    torch.set_num_threads(STEP_THREADS)
    try:
        yield
    finally:
        torch.set_num_threads(previous)


def updates(recipe):
    spec, sources = recipe['spec'], recipe['sources']
    if 'mixture' in spec:
        return trainer.mixture_schedule(sources, spec)
    return trainer.schedule(sources['experience'], sources.get('replay', []), spec)


def _optimizer(trainable, spec, arm):
    return torch.optim.AdamW(list(trainable.values()), lr=trainer.learning_rate(0, spec, arm),
                             betas=tuple(spec['betas']), eps=spec['epsilon'], weight_decay=spec['weight_decay'])


UNIT_STORE = None  # directory of published unit tensors, ``<sha256>.safetensors``; default $NEUROSHARD_UNITS or .units
# resolve(sha256) -> bytes or None: obtains a unit the store lacks (a pinned released file, or a grown unit rebuilt
# from its committed state). Set by ``growth_runtime.provide_units`` on every machine that trains or judges growth.
UNIT_RESOLVER = None


def unit_tensors(sha256):
    """A published unit's trainable tensors, accepted only if the file hashes to the pinned digest.

    Looked up by content in the unit store, then obtained through ``UNIT_RESOLVER`` and kept in the store as
    ``<sha256>.safetensors``, so every continuation a recipe names can be trained or judged on a fresh machine."""
    import os
    from pathlib import Path

    from safetensors.torch import load as load_tensors

    store = Path(UNIT_STORE or os.environ.get('NEUROSHARD_UNITS', '.units'))
    for path in [store / f'{sha256}.safetensors', *sorted(store.glob('*/trainable.safetensors'))]:
        if path.is_file():
            payload = path.read_bytes()
            if hashlib.sha256(payload).hexdigest() == sha256:
                return load_tensors(payload)
    payload = UNIT_RESOLVER(sha256) if UNIT_RESOLVER else None
    if payload is not None and hashlib.sha256(payload).hexdigest() == sha256:
        store.mkdir(parents=True, exist_ok=True)
        partial = store / f'.{sha256}.partial'
        partial.write_bytes(payload)
        partial.replace(store / f'{sha256}.safetensors')
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
    """Exactly one optimizer step ``index`` of the recipe's schedule, on the pinned ``STEP_THREADS`` threads."""
    with step_threads():
        return _step(model, trainable, optimizer, recipe, index, schedule)


def _step(model, trainable, optimizer, recipe, index, schedule=None):
    """The loop body of ``trainer.train``."""
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
    """Restore committed state bytes onto the recipe's own optimizer.

    Only what the state root commits is taken from the bytes: the trainable tensors and the optimizer's
    per-parameter state. Everything else that affects a step (learning-rate schedule, betas, epsilon, weight
    decay and the other AdamW settings, and which parameter each state belongs to) comes from the recipe,
    which ``recipe_root`` commits. Bytes that claim other settings or another parameter order are refused,
    so uncommitted metadata can never change what a step computes.
    """
    trainable, optimizer = start(model, recipe)
    saved = torch.load(io.BytesIO(payload), weights_only=True)
    if set(saved['trainable']) != set(trainable):
        raise ValueError('proof state names other tensors than the recipe attaches')
    index = {id(value): name for name, value in trainable.items()}
    order = [index[id(p)] for group in optimizer.param_groups for p in group['params']]
    if saved.get('order') != order:
        raise ValueError('state orders its optimizer parameters differently from the recipe')
    fresh = optimizer.state_dict()
    claimed = saved['optimizer'].get('param_groups', [])
    if len(claimed) != len(fresh['param_groups']):
        raise ValueError('state claims other optimizer parameter groups than the recipe')
    for mine, theirs in zip(fresh['param_groups'], claimed):
        for key in set(mine) & set(theirs) - {'params', 'lr'}:  # lr is set from the schedule before every step
            if mine[key] != theirs[key]:
                raise ValueError(f'state claims optimizer {key} {theirs[key]!r}; the recipe sets {mine[key]!r}')
        if theirs.get('params') != mine['params']:
            raise ValueError('state maps optimizer parameters differently from the recipe')
    with torch.no_grad():
        for name, value in trainable.items():
            value.copy_(saved['trainable'][name])
    optimizer.load_state_dict({'state': saved['optimizer']['state'], 'param_groups': fresh['param_groups']})
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
    try:
        proof = json.loads(payload)
        before = bytes.fromhex(proof['before'])
    except (ValueError, KeyError, TypeError):
        return False
    if proof.get('format') != 'neuroshard-growth-proof/1' or proof.get('step') != request['step']:
        return False
    recipe = execution(recipe)
    model = make_model()
    try:
        trainable, optimizer = load(model, recipe, before)
    except (ValueError, KeyError, RuntimeError, TypeError):
        return False  # a bundle that does not restore the committed state under the recipe proves nothing
    if state_root(trainable, optimizer) != request['before_root']:
        return False
    step(model, trainable, optimizer, recipe, request['step'])
    return state_root(trainable, optimizer) != request['claimed_root']


PARENT = 'granite-parent'


def execution(recipe, model_root=None):
    """Exactly the part of a recipe that ``recipe_root`` commits, and the only part execution may read.

    The arm, spec and data; the unit training continues from; the model it grows; and the base model it
    loads. The pinned Granite parent is the default base, fully determined by the verified artifact store,
    so it is bound only implicitly and earlier roots are unchanged. Any other base, and its configuration
    and seed, is bound explicitly. A parent recipe that carries a base configuration or seed is refused:
    those fields would be ignored, and a recipe must not mean two things.
    """
    base = recipe.get('base', PARENT)
    bound = {'model_root': recipe.get('model_root') if model_root is None else model_root,
             'arm': recipe['arm'], 'spec': recipe['spec'], 'sources': recipe['sources']}
    if recipe.get('continue_from'):
        bound['continue_from'] = recipe['continue_from']  # bound only when present: earlier roots are unchanged
    if base == PARENT:
        if 'base_config' in recipe or 'base_seed' in recipe:
            raise ValueError('a recipe on the pinned parent carries no base configuration or seed')
    else:
        if not isinstance(base, str) or 'base_config' not in recipe:
            raise ValueError('a recipe on another base names it and its configuration')
        bound.update(base=base, base_config=recipe['base_config'], base_seed=recipe.get('base_seed', 0))
    return bound


def recipe_root(recipe, model_root):
    """What a ``growth_open`` names: every recipe field that affects execution (``execution``)."""
    return hashlib.sha256(json.dumps(execution(recipe, model_root), sort_keys=True).encode()).hexdigest()
