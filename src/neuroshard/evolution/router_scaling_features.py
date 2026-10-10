"""Turn features for the router scaling study.

Two encoders, both deterministic on one runtime:

- ``hashed``: a signed hashed bag of word unigrams and bigrams. A cheap lexical floor:
  if a frozen language model's states route worse than word counts, the feature is the
  bottleneck, not the number of routes.
- ``lm``: the accepted router's feature on a small local model, the mean final-layer
  state over the user message and the reply header after a shared, once-computed prefix
  (`assistant_routing.message_feature`). The study runs it on SmolLM2-135M on CPU, a
  stand-in for the Granite 4.1 3B parent, whose states are richer; a result here is a
  lower bound on what the same rule does with the real parent, not a measurement of it.

Each strategy's feature for turn ``t`` of a conversation is:

- ``message``: the feature of the user message alone;
- ``with-opening``: the message feature concatenated with the conversation's opening
  message feature, so a generic follow-up carries the context that opened it.
"""

import hashlib
import re

import numpy as np

WORD = re.compile(r"[a-z]+|\d+")


def hashed(text, width=512):
    """Signed hashed unigram and bigram counts, L2-normalised; digits collapse to one token."""
    words = ['<n>' if token.isdigit() else token for token in WORD.findall(text.lower())]
    grams = words + [f'{a}_{b}' for a, b in zip(words, words[1:])]
    vector = np.zeros(width)
    for gram in grams:
        digest = hashlib.sha256(gram.encode()).digest()
        index = int.from_bytes(digest[:4], 'little') % width
        vector[index] += 1.0 if digest[4] & 1 else -1.0
    norm = np.linalg.norm(vector)
    return (vector / norm if norm else vector).tolist()


SYSTEM = 'You are a workspace assistant. Use the tools you are given to complete the request.'


class LanguageModelEncoder:
    """Mean final-layer state over each user message after a cached shared prefix, as the accepted router computes it."""

    def __init__(self, directory, threads=4):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        torch.set_num_threads(threads)
        self.torch = torch
        self.tokenizer = AutoTokenizer.from_pretrained(directory)
        self.model = AutoModelForCausalLM.from_pretrained(directory, torch_dtype=torch.float32).eval()
        first, second = self._ids('Schedule'), self._ids('Create')
        common = 0
        for left, right in zip(first, second):
            if left != right:
                break
            common += 1
        if not 0 < common < min(len(first), len(second)):
            raise ValueError('turns share no prefix before the user message')
        self.prefix = first[:common]

    def _ids(self, user):
        messages = [{'role': 'system', 'content': SYSTEM}, {'role': 'user', 'content': user}]
        prompt = self.tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
        return self.tokenizer(prompt, add_special_tokens=False)['input_ids']

    def encode(self, texts, batch=16, layers=None):
        """Mean state over each message after the shared prefix; one forward pass per batch, padded right.

        With ``layers`` (hidden-state indices; 0 is the embeddings, the last is after the final
        norm, the accepted feature), returns ``{layer: features}`` instead of final-layer features.
        """
        torch = self.torch
        wanted = layers if layers is not None else [-1]
        out = {layer: [] for layer in wanted}
        for start in range(0, len(texts), batch):
            chunk = [self._ids(text) for text in texts[start:start + batch]]
            for ids in chunk:
                if ids[:len(self.prefix)] != self.prefix or len(ids) == len(self.prefix):
                    raise ValueError('a turn does not extend the shared prefix')
            longest = max(map(len, chunk))
            pad = self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else 0
            ids = torch.tensor([row + [pad] * (longest - len(row)) for row in chunk])
            mask = torch.tensor([[1] * len(row) + [0] * (longest - len(row)) for row in chunk])
            with torch.no_grad():
                hidden = self.model.model(input_ids=ids, attention_mask=mask, output_hidden_states=True).hidden_states
            for layer in wanted:
                states = hidden[layer].double()
                for row, length in zip(states, map(len, chunk)):
                    out[layer].append(row[len(self.prefix):length].mean(dim=0).tolist())
        return out if layers is not None else out[-1]


def strategy_features(cases, encode):
    """``{'message': ..., 'with-opening': ...}`` keyed by ``case#turn``; ``encode`` maps a list of texts to features."""
    texts = sorted({text for case in cases for text in case['user_turns']})
    encoded = dict(zip(texts, encode(texts)))
    message, opening = {}, {}
    for case in cases:
        first = encoded[case['user_turns'][0]]
        for turn, text in enumerate(case['user_turns']):
            key = f'{case["id"]}#{turn}'
            message[key] = encoded[text]
            opening[key] = list(encoded[text]) + list(first)
    return {'message': message, 'with-opening': opening}
