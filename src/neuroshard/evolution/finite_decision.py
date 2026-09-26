"""Exact finite text choices through native generation, including terminal EOS.

This constrains output syntax, not the semantic correctness of a choice.
"""


class FiniteDecision:
    def __init__(self, tokenizer, choices, eos_token_id):
        if not choices or len(set(choices)) != len(choices):
            raise ValueError("require distinct finite choices")
        if type(eos_token_id) is not int:
            raise ValueError("require one declared terminal EOS")
        self.paths = {}
        for choice in choices:
            tokens = tokenizer(choice, add_special_tokens=False)["input_ids"]
            if (not tokens or any(t in tokenizer.all_special_ids for t in tokens)
                    or tokenizer.decode(tokens, skip_special_tokens=False) != choice):
                raise ValueError("choice does not round-trip as ordinary tokens")
            path = tuple(tokens + [eos_token_id])
            if path in self.paths:
                raise ValueError("finite choices share a token path")
            self.paths[path] = choice
        self.max_new_tokens = max(map(len, self.paths))

    def allowed(self, generated):
        prefix = tuple(generated)
        following = sorted({path[len(prefix)] for path in self.paths
                            if len(prefix) < len(path) and path[:len(prefix)] == prefix})
        if not following:
            raise ValueError("generation escaped finite decision or continued after EOS")
        return following

    def constraint(self, prompt_tokens):
        if type(prompt_tokens) is not int or prompt_tokens <= 0:
            raise ValueError("require the actual nonempty prompt length")
        return lambda batch, prefix: self.allowed(prefix.tolist()[prompt_tokens:])

    def decode(self, generated):
        try:
            return self.paths[tuple(generated)]
        except KeyError as error:
            raise ValueError("incomplete or undeclared finite decision") from error
