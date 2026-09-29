"""Consensus of verified modules: several small arms and the parent vote on every action.

Each arm is grown from its own slice of verified experience, as independent
contributors would grow it. At every step each member proposes the next
assistant message; the committee takes the action most members agree on, where
an action is a canonical tool call, "finish" for any valid final message, or an
invalid output that agrees with nothing. A tie that includes the parent goes to
the parent, so overriding the parent needs more modules agreeing on the same
different action than a single module.
"""

import contextlib
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import threading

from neuroshard.evolution import assistant_workspace as sandbox


def action(proposal, index):
    """The canonical action a proposed message takes; invalid outputs never agree."""
    if not proposal.get('executed', True):
        return 'skip'
    if not proposal.get('terminated'):
        return f'invalid:{index}'
    try:
        calls = sandbox.parse_calls(proposal['text'])
    except (ValueError, TypeError):
        return f'invalid:{index}'
    if calls:
        return 'call:' + json.dumps(calls, sort_keys=True, separators=(',', ':'))
    return 'finish' if proposal['text'].strip() else f'invalid:{index}'


def vote(proposals, parent):
    """Index of the chosen proposal and the tally; ``parent`` is the parent's index in ``proposals``."""
    keys = [action(p, i) for i, p in enumerate(proposals)]
    counts = {}
    for key in keys:
        counts[key] = counts.get(key, 0) + 1
    best = max(counts.values())
    leaders = [key for key, count in counts.items() if count == best]
    winner = keys[parent] if keys[parent] in leaders else min(leaders, key=keys.index)
    chosen = min(i for i, key in enumerate(keys) if key == winner and i != parent) if any(
        key == winner and i != parent for i, key in enumerate(keys)) else parent
    return chosen, {'actions': keys, 'winner': winner, 'votes': best, 'members': len(keys)}


def responder(members, parent):
    """A workflow responder that asks every member (and the parent) and answers with the vote."""
    everyone = list(members) + [parent]

    def respond(messages, tools):
        with ThreadPoolExecutor(max_workers=len(everyone)) as pool:
            proposals = list(pool.map(lambda member: member(messages, tools), everyone))
        chosen, tally = vote(proposals, len(everyone) - 1)
        return {**proposals[chosen], 'committee': {**tally, 'chosen': chosen}}

    return respond


class Switch:
    """Selects which saved adapter every switchable projection applies; ``None`` is the parent."""

    def __init__(self):
        self.active, self.lock = None, threading.Lock()

    @contextlib.contextmanager
    def using(self, index):
        with self.lock:
            self.active = index
            try:
                yield
            finally:
                self.active = None


def attach(model, spec, directories, switch):
    """Wrap each declared projection with every saved adapter; the backbone is unchanged."""
    import torch
    from safetensors.torch import load_file

    from neuroshard.evolution import assistant_experience_train as trainer

    class Switched(torch.nn.Module):
        def __init__(self, base):
            super().__init__()
            self.base, self.adapters = base, []

        def forward(self, x):
            if switch.active is None:
                return self.base(x)
            a, b, scale = self.adapters[switch.active]
            delta = (x.to(torch.float32) @ a.T) @ b.T
            return self.base(x) + (delta * scale).to(x.dtype)

    saved = [load_file(str(Path(directory) / 'trainable.safetensors')) for directory in directories]
    scale = spec['alpha'] / spec['rank']
    wrapped = 0
    for name, owner, attribute in trainer.projections(model, spec['layers'], spec.get('projections', trainer.PROJECTIONS)):
        base = getattr(owner, attribute)
        module = Switched(base)
        for tensors in saved:
            a, b = tensors[name + '.lora_a'], tensors[name + '.lora_b']
            module.adapters.append((a.to(base.weight.device), b.to(base.weight.device), scale))
        setattr(owner, attribute, module)
        wrapped += 1
    return wrapped
