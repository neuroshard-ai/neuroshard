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


def cached_committee(model, switch, tokenizer, policy, members):
    """One responder per episode: members and parent decode each message as one batch over a shared prompt cache.

    Row ``i < members`` applies member ``i``'s adapter and the last row is the
    parent. Only the prompt, which every row shares, is kept between requests.
    """
    import resource
    import time

    import torch
    from transformers import DynamicCache, LogitsProcessor, LogitsProcessorList, StoppingCriteria, StoppingCriteriaList

    from neuroshard.evolution.modular_reference_execution import identity

    generation = policy['generation']
    eos = model.generation_config.eos_token_id
    eos = [eos] if isinstance(eos, int) else list(eos)
    rows = list(range(members)) + [None]
    state = {'ids': [], 'cache': DynamicCache()}

    def respond(messages, tools):
        prompt = tokenizer.apply_chat_template(messages, tools=tools, add_generation_prompt=True, tokenize=False)
        ids = tokenizer(prompt, add_special_tokens=False)['input_ids']
        base = {'id': 'workspace-turn', 'category': 'workspace', 'model': 'committee', 'prompt_sha256': identity(prompt),
                'input_token_ids': ids}
        if len(ids) > generation['max_input_tokens']:
            return {**base, 'token_ids': [], 'text': '', 'terminated': False, 'executed': False,
                    'budget_stop': 'input cap; no truncation or inference', 'seconds': 0, 'reused_prefix_tokens': 0}
        common = 0
        for cached, new in zip(state['ids'], ids):
            if cached != new:
                break
            common += 1
        common = min(common, len(ids) - 1)
        if state['cache'].get_seq_length() > common:
            state['cache'].crop(common)
        started = time.monotonic()
        first_token = []

        class Observe(StoppingCriteria):
            def __call__(self, input_ids, scores, **unused):
                if not first_token:
                    first_token.append(time.monotonic() - started)
                return torch.zeros(input_ids.shape[0], dtype=torch.bool, device=input_ids.device)

        class CheckLogits(LogitsProcessor):
            def __call__(self, input_ids, scores):
                if torch.isnan(scores).any() or torch.isposinf(scores).any() or not torch.isfinite(scores).any():
                    raise ValueError('nonfinite model logits')
                return scores

        batch = torch.tensor([ids] * len(rows))
        with torch.inference_mode(), switch.each(rows):
            output = model.generate(input_ids=batch, attention_mask=torch.ones_like(batch), past_key_values=state['cache'],
                                    do_sample=False, num_beams=1, use_cache=True, pad_token_id=eos[0],
                                    max_new_tokens=generation['max_new_tokens'],
                                    stopping_criteria=StoppingCriteriaList([Observe()]),
                                    logits_processor=LogitsProcessorList([CheckLogits()]))
        proposals = []
        for row in range(len(rows)):
            tokens = output[row, len(ids):].tolist()
            end = next((i for i, token in enumerate(tokens) if token in eos), None)
            tokens = tokens if end is None else tokens[:end + 1]
            terminated = end is not None
            proposals.append({'token_ids': tokens, 'terminated': terminated, 'executed': True,
                              'text': tokenizer.decode(tokens[:-1] if terminated else tokens, skip_special_tokens=False)})
        chosen, tally = vote(proposals, len(rows) - 1)
        state['ids'] = ids
        return {**base, **proposals[chosen], 'seconds': time.monotonic() - started,
                'first_token_seconds': first_token[0] if first_token else None, 'reused_prefix_tokens': common,
                'max_rss_bytes': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
                'committee': {**tally, 'chosen': chosen}}

    return respond


class Switch:
    """Selects which saved adapter every switchable projection applies; ``None`` is the parent.

    ``using`` applies one adapter to the whole batch; ``each`` applies one per batch row.
    """

    def __init__(self):
        self.active, self.rows, self.lock = None, None, threading.Lock()

    @contextlib.contextmanager
    def using(self, index):
        with self.lock:
            self.active = index
            try:
                yield
            finally:
                self.active = None

    @contextlib.contextmanager
    def each(self, rows):
        with self.lock:
            self.rows = list(rows)
            try:
                yield
            finally:
                self.rows = None


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
            if switch.rows is not None:
                if x.shape[0] != len(switch.rows):
                    raise ValueError('one adapter choice per batch row')
                output = self.base(x)
                for index in sorted({row for row in switch.rows if row is not None}):
                    select = torch.tensor([r for r, row in enumerate(switch.rows) if row == index], device=x.device)
                    a, b, scale = self.adapters[index]
                    delta = (x[select].to(torch.float32) @ a.T) @ b.T
                    output[select] = output[select] + (delta * scale).to(x.dtype)
                return output
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
