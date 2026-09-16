"""One-pass verification of a reported token stream under fixed-shape causality.

This research check has its own numerical definition: one document padded to a
fixed context with an all-valid causal mask. It does NOT assume that a cached
decoder and this batched program are bit-identical. Current cached responses are
compared experimentally; native acceptance still requires its admitted executor.
"""
from pathlib import Path
import time

import torch
from safetensors.torch import load_file

from ..reference_data import identity, sha256
from .fused_graph import commitment
from .fusion_features import produce
from .mixture_training import native_logits


@torch.no_grad()
def verify(net, gate, prompt, reported, max_tokens, context, home, *, interface=None, adapt_interfaces=False):
    vocabulary, eos = net.graph['parent']['config']['vocab_size'], net.tokenizer.eos_token_id
    if (not isinstance(prompt, list) or not prompt or not isinstance(reported, list) or not reported
            or type(max_tokens) is not int or not 1 <= max_tokens <= 256
            or type(context) is not int or not 2 <= context <= min(1024, gate.max_context)
            or len(prompt)+max_tokens > context or len(reported) > max_tokens
            or any(type(token) is not int or not 0 <= token < vocabulary for token in prompt+reported)
            or eos in reported[:-1] or (len(reported) < max_tokens and reported[-1] != eos)):
        raise ValueError('Require a complete bounded response ending at EOS or the requested token limit')
    net.check_context(prompt, max_tokens)
    request = {'format': 'neuroshard-fixed-context-token-check-v1', 'graph': identity(net.graph),
               'gate': commitment(gate), 'prompt': prompt, 'reported': reported,
               'max_tokens': max_tokens, 'context': context, 'mask': 'all-valid-causal'}
    if net.all_owners.exchange(identity(request)) != [identity(request)]*net.world_size:
        raise ValueError('Auditors received different complete token-check requests')
    tokens = prompt+reported+[eos]*(context-len(prompt)-len(reported))
    row = {'id': identity(request), 'kind': 'general', 'input_ids': tokens,
           'labels': [-100]*len(prompt)+reported+[-100]*(context-len(prompt)-len(reported))}
    started = time.monotonic()
    bank, resources = produce(net, [row], [[0]], home, max_length=context, max_seconds=300,
                               interface=interface, adapt_interfaces=adapt_interfaces)
    result = None
    if net.rank == 0:
        spec = bank['files'][0]
        path = Path(home)/(spec['sha256']+'.safetensors')
        if path.stat().st_size != spec['bytes'] or sha256(path) != spec['sha256']:
            raise ValueError('Committed audit activations changed')
        values = {key: value.to(net.shard.device_name) for key, value in load_file(path).items()}
        distribution = gate.log_probs(values['hub'], native_logits(net.preserved.shard, net.shard,
                                      values, gate.source_widths, False))
        predicted = distribution[0, len(prompt)-1:len(prompt)+len(reported)-1].argmax(dim=-1).tolist()
        mismatches = [i for i, (actual, expected) in enumerate(zip(reported, predicted)) if actual != expected]
        result = {'request': identity(request), 'feature_bank': identity(bank),
                  'interfaces': bank.get('interfaces', {}), 'predicted': predicted,
                  'mismatches': mismatches, 'passed': not mismatches,
                  'scope': 'Fixed-context causal re-execution, not a proof of the cached numerical program'}
    resources['total_seconds'] = time.monotonic()-started
    packet = net.all_owners.exchange({'result': result, 'resources': resources})
    if any(value['result'] is not None for value in packet[1:]):
        raise ValueError('Only the installed output owner computes the checked vocabulary predictions')
    net.verify_unchanged()
    return {'result': packet[0]['result'], 'owners': [value['resources'] for value in packet]}
