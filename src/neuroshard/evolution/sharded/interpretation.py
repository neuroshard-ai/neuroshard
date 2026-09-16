"""Interpret user text with preserved weights, then query a learned neural expert."""
import json
import re

import torch

from .. import incremental_facts as facts
from ..reference import autocast
from .branch import Network, route


def interpretation(text, question):
    """Validate model-produced arguments without any directory facts or labels."""
    def unique(pairs):
        if len(dict(pairs)) != len(pairs):
            raise ValueError('Duplicate interpretation key')
        return dict(pairs)
    try:
        value = json.loads(text, object_pairs_hook=unique)
    except (ValueError, TypeError):
        return None
    if (not isinstance(value, dict) or set(value) != {'name', 'field'}
            or not isinstance(value['name'], str) or not isinstance(value['field'], str)
            or not re.fullmatch(r'[A-Za-z]+(?: [A-Za-z]+){1,3}', value['name'])
            or value['name'] not in question
            or value['field'] not in ('city', 'profession', 'instrument', 'hobby')):
        return None
    return value


def example_messages(instruction, examples):
    """Preserve the published prompt independently of JSON object key order.

    Object identity ignores insertion order; model input does not. The original
    successful interpreter used name before field, including these spaces.
    """
    prefix = [{'role': 'system', 'content': instruction}]
    for question, answer in examples:
        if set(answer) != {'name', 'field'}:
            raise ValueError('Require the exact interpretation argument schema')
        ordered = {'name': answer['name'], 'field': answer['field']}
        prefix.extend([{'role': 'user', 'content': json.dumps(question)},
                       {'role': 'assistant', 'content': json.dumps(ordered)}])
    return prefix


class InterpretedNetwork:
    """Each established owner holds only its portions of the two fixed models.

    Interpretation uses the three-parent group; expert generation uses all
    four owners. Ordinary parent serving never enters an interpreter collective.
    """
    def __init__(self, trained, preserved, instruction, examples, max_tokens=40, record=None):
        if not 0 < max_tokens <= 64:
            raise ValueError('Bound interpretation generation')
        if (trained.wire.rank < 3) != (preserved is not None):
            raise ValueError('Only the three established owners hold interpreter partitions')
        self.trained, self.preserved = trained, preserved
        self.instruction, self.max_tokens, self.record = instruction, max_tokens, record
        self.prefix = example_messages(instruction, examples)
        self.additional_parameters = preserved.shard.resident_parameters if preserved else 0
        self.parameters = [p for net in (trained, preserved) if net is not None
                           for _, p in net.shard.named_owned_parameters()]
        if any(p.requires_grad for p in self.parameters):
            raise ValueError('Interpretation composition is read-only')
        self.versions = tuple(p._version for p in self.parameters)

    def verify_unchanged(self):
        if tuple(p._version for p in self.parameters) != self.versions:
            raise ValueError('A fixed model changed during interpretation or serving')

    def forward(self, ids, expert):
        return self.trained.forward(ids, expert)

    def generate(self, question, max_tokens, expert):
        return self.trained.generate(question, max_tokens, expert)

    @torch.no_grad()
    def interpret(self, question):
        if self.preserved is None:
            return None
        net = self.preserved
        messages = [*self.prefix, {'role': 'user', 'content': json.dumps(question) + '\n\n' + self.instruction}]
        ids = net.tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True)
        if len(ids) + self.max_tokens > 1024:
            raise ValueError('No silent interpretation truncation')
        output = []
        for _ in range(self.max_tokens):
            hidden = net.forward(ids, False)
            token = None
            if net.wire.rank == 0:
                with autocast(net.device):
                    token = int(net.shard.logits(hidden[:, -1:]).float().argmax(-1)[0, 0])
            token = net.parent_wire.exchange(token)[0]
            if type(token) is not int or not 0 <= token < net.config.vocab_size:
                raise ValueError('Invalid interpreter token')
            output.append(token); ids.append(token)
            if token == net.tokenizer.eos_token_id:
                break
        return {'ids': output, 'text': net.tokenizer.decode(output, skip_special_tokens=True)}

    def answer(self, question, max_tokens):
        if not route(question):
            return self.trained.answer(question, max_tokens)
        return self.answer_expert(question, max_tokens)

    def answer_expert(self, question, max_tokens):
        """Execute interpretation after an external router selected this expert."""
        value = self.interpret(question)
        values = self.trained.wire.exchange(value)
        if values[0] is None or any(row is not None and row != values[0] for row in values):
            raise ValueError('Owners disagree on the model-produced interpretation')
        value = values[0]
        parsed = interpretation(value['text'], question)
        canonical = facts.question({'name': parsed['name']}, parsed['field'], 'train', 0) if parsed else question
        result = self.trained.generate(canonical, max_tokens, True)
        if self.record:
            self.record({'raw_question': question, 'parser': value, 'interpretation': parsed,
                         'expert_question': canonical, 'answer': result})
        return result
