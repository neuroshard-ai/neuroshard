"""Episode-scoped serving that reuses the conversation's key/value cache between generations.

Every request is still rendered and tokenized in full; only the longest token
prefix shared with the cached sequence is reused, and at least one token is
recomputed. Outputs match the uncached responder's record format.
"""

import resource
import time

from neuroshard.evolution.modular_reference_execution import identity


def cached_responder(model, tokenizer, policy):
    """One responder per episode; a new episode needs a new responder and an empty cache."""
    import torch
    from transformers import DynamicCache, LogitsProcessor, LogitsProcessorList, StoppingCriteria, StoppingCriteriaList

    generation = policy['generation']
    eos = model.generation_config.eos_token_id
    eos = [eos] if isinstance(eos, int) else list(eos)
    state = {'ids': [], 'cache': DynamicCache()}

    def respond(messages, tools):
        prompt = tokenizer.apply_chat_template(messages, tools=tools, add_generation_prompt=True, tokenize=False)
        ids = tokenizer(prompt, add_special_tokens=False)['input_ids']
        base = {'id': 'workspace-turn', 'category': 'workspace', 'model': 'cached', 'prompt_sha256': identity(prompt),
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
                return False

        class CheckLogits(LogitsProcessor):
            def __call__(self, input_ids, scores):
                if torch.isnan(scores).any() or torch.isposinf(scores).any() or not torch.isfinite(scores).any():
                    raise ValueError('nonfinite model logits')
                return scores

        with torch.inference_mode():
            output = model.generate(input_ids=torch.tensor([ids]), attention_mask=torch.ones(1, len(ids), dtype=torch.long),
                                    past_key_values=state['cache'], do_sample=False, num_beams=1, use_cache=True,
                                    max_new_tokens=generation['max_new_tokens'],
                                    stopping_criteria=StoppingCriteriaList([Observe()]),
                                    logits_processor=LogitsProcessorList([CheckLogits()]))
        tokens = output[0, len(ids):].tolist()
        terminated = bool(tokens and tokens[-1] in eos)
        text = tokenizer.decode(tokens[:-1] if terminated else tokens, skip_special_tokens=False)
        state['ids'] = (ids + tokens)[:state['cache'].get_seq_length()]
        return {**base, 'token_ids': tokens, 'text': text, 'terminated': terminated, 'executed': True,
                'seconds': time.monotonic() - started, 'first_token_seconds': first_token[0] if first_token else None,
                'reused_prefix_tokens': common, 'max_rss_bytes': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024}

    return respond
