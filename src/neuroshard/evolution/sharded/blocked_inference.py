"""A fixed-block causal program with owner-local, reversible attention state.

Block size and boundaries are part of this distinct numerical prescription.
They are not claimed equivalent to cached token decoding or a full-context
matrix pass. Only complete verified input blocks can remain in the cache.
"""
from contextlib import contextmanager, nullcontext
from pathlib import Path
import copy
import time

import torch
from transformers.masking_utils import create_causal_mask

from ..reference import autocast
from ..reference_data import identity, save
from .cached_inference import CachedPartition, generate_branch_cached
from .canonical_stream import parameter_versions
from .expert_interface import installed
from .fused_graph import commitment
from .mixture import ProbabilityMixture
from .mixture_training import native_logits

FORMAT = 'neuroshard-prefilled-block-inference-v1'


class BlockPartition(CachedPartition):
    def __init__(self, shard, block_size):
        super().__init__(shard)
        if type(block_size) is not int or not 1 <= block_size <= 32:
            raise ValueError('Require a bounded fixed query block')
        self.block_size = block_size

    @torch.no_grad()
    def advance(self, value, position, *, prefill=False):
        shard = self.shard
        count = value.shape[1] if value.ndim >= 2 else 0
        if self.failed:
            raise ValueError('Discard a failed block cache')
        self.failed = True
        if (any(module.training for module in shard.modules())
                or self.versions != tuple(parameter._version for parameter in shard.parameters())
                or type(prefill) is not bool or count < 1
                or (prefill and (position != 0 or count % self.block_size))
                or (not prefill and count != self.block_size)
                or type(position) is not int or position != self.length or position % self.block_size
                or value.ndim != (2 if shard.rank == 0 else 3) or value.shape[:2] != (1, count)
                or value.device != next(shard.parameters()).device
                or position+count > shard.config.max_position_embeddings
                or any(layer.get_seq_length() != position for layer in self.cache.layers)):
            raise ValueError('Invalid fixed-block continuation or changed owned weights')
        if shard.rank == 0:
            if value.dtype != torch.long or bool(((value < 0) | (value >= shard.config.vocab_size)).any()):
                raise ValueError('Invalid block input tokens')
        elif value.shape[2] != shard.config.hidden_size or value.dtype != torch.float32:
            raise ValueError('Invalid owned block boundary')
        hidden = shard.embedding(value) if shard.rank == 0 else value
        positions = torch.arange(position, position+count, device=hidden.device)
        position_ids = positions.unsqueeze(0)
        mask = torch.ones((1, position+count), dtype=torch.long, device=hidden.device)
        causal = create_causal_mask(shard.config, hidden, mask, positions, self.cache, position_ids)
        rotary = shard.rotary(hidden, position_ids)
        for layer in shard.layers.values():
            hidden = layer(hidden, attention_mask=causal, position_ids=position_ids,
                           past_key_values=self.cache, cache_position=positions,
                           position_embeddings=rotary, use_cache=True)
        self.length += count
        if any(layer.get_seq_length() != self.length for layer in self.cache.layers):
            raise ValueError('Incomplete owned block cache update')
        self.failed = False
        return hidden

    def rewind(self, position):
        if (self.failed or type(position) is not int or position < 0
                or position != self.length-self.block_size):
            raise ValueError('Only the last complete speculative block can be discarded')
        self.failed = True
        self.cache.crop(position)
        if any(layer.get_seq_length() != position for layer in self.cache.layers):
            raise ValueError('Incomplete owned block rollback')
        self.length, self.failed = position, False


class BlockNetwork:
    def __init__(self, net, gate, block_size, interface=None):
        self.net, self.gate, self.interface, self.block_size = net, gate, interface, block_size
        self.sources = {'parent': 2, **{rule['id']: rule['owner'] for rule in net.graph['descriptor']['rules']}}
        width = net.graph['parent']['config']['hidden_size']
        if (not isinstance(gate, ProbabilityMixture) or gate.hub_width != width
                or gate.source_widths != {name: width for name in sorted(self.sources)}
                or next(gate.parameters()).device != next(net.shard.parameters()).device
                or any(module.training for module in gate.modules())):
            raise ValueError('Bind the complete native-head gate to the installed block graph')
        self.trained = BlockPartition(net.shard, block_size)
        self.preserved = BlockPartition(net.preserved.shard, block_size) if net.rank < 3 else None
        self.versions = parameter_versions(gate), parameter_versions(interface)
        self.captured, self.hook = {}, None
        if net.rank == 2:
            layer = net.shard.layers[str(net.graph['descriptor']['split']-1)]
            self.hook = layer.register_forward_hook(lambda module, args, output: self.captured.update(prefix=output.clone()))

    def close(self):
        if self.hook is not None:
            self.hook.remove()

    def unchanged(self):
        changed = self.versions != (parameter_versions(self.gate), parameter_versions(self.interface))
        try:
            self.net.verify_unchanged()
        except ValueError:
            changed = True
        if any(self.net.all_owners.exchange(changed)):
            raise ValueError('The fixed-block model changed during execution')

    @torch.no_grad()
    def advance(self, tokens, position, *, prefill=False):
        net, gate, wire, rank = self.net, self.gate, self.net.all_owners, self.net.rank
        self.unchanged()
        if (type(prefill) is not bool or not isinstance(tokens, list) or not tokens
                or (prefill and (position != 0 or len(tokens) % self.block_size))
                or (not prefill and len(tokens) != self.block_size)
                or any(type(token) is not int or not 0 <= token < net.graph['parent']['config']['vocab_size'] for token in tokens)):
            raise ValueError('Every owner requires the same complete fixed token block')
        if wire.exchange(identity([position, tokens, prefill])) != [identity([position, tokens, prefill])]*net.world_size:
            raise ValueError('Owners disagree on a fixed input block')
        began, before = time.monotonic(), wire.sent_tensor_bytes
        device = net.shard.device_name
        shape = (1, len(tokens), net.graph['parent']['config']['hidden_size'])
        if rank < 3:
            incoming = torch.tensor([tokens], dtype=torch.long, device=device) if rank == 0 else wire.receive(rank-1, shape, device)
            with autocast(device):
                hidden = self.trained.advance(incoming, position, prefill=prefill)
            if rank < 2:
                wire.send(hidden, rank+1)
            else:
                if not prefill:
                    wire.send(hidden, 0)
                for name, owner in self.sources.items():
                    if name != 'parent':
                        wire.send(self.captured['prefix'], owner)
                self.captured.clear()
        else:
            incoming = wire.receive(2, shape, device)
            with autocast(device):
                hidden = self.trained.advance(incoming, position, prefill=prefill)
            if not prefill:
                wire.send(hidden, 0)
        values = ({name: wire.receive(owner, shape, device) for name, owner in self.sources.items()}
                  if rank == 0 and not prefill else None)
        if rank < 3:
            incoming = torch.tensor([tokens], dtype=torch.long, device=device) if rank == 0 else wire.receive(rank-1, shape, device)
            with autocast(device):
                hidden = self.preserved.advance(incoming, position, prefill=prefill)
            if rank < 2:
                wire.send(hidden, rank+1)
            elif not prefill:
                wire.send(hidden, 0)
        predicted = None
        if rank == 0 and not prefill:
            hub = wire.receive(2, shape, device)
            logits = gate.log_probs(hub, native_logits(net.preserved.shard, net.shard,
                {'hub': hub, **values}, gate.source_widths, False))
            predicted = logits.argmax(-1)[0].tolist()
        packet = wire.exchange({'predicted': predicted, 'seconds': time.monotonic()-began,
                                'sent_tensor_bytes': wire.sent_tensor_bytes-before})
        predicted = packet[0]['predicted']
        if (any(row['predicted'] is not None for row in packet[1:])
                or (prefill and predicted is not None)
                or (not prefill and (not isinstance(predicted, list) or len(predicted) != self.block_size
                    or any(type(token) is not int or not 0 <= token < net.graph['parent']['config']['vocab_size'] for token in predicted)))):
            raise ValueError('Only the output owner supplies bounded block predictions')
        return {'position': position, 'input': tokens, 'predicted': predicted, 'prefill': prefill,
                'owners': [{k: v for k, v in row.items() if k != 'predicted'} for row in packet]}

    def rewind(self, position):
        self.trained.rewind(position)
        if self.preserved:
            self.preserved.rewind(position)


@contextmanager
def session(net, gate, block_size, prompt, max_tokens, context, interface, adapt_interfaces):
    if (type(block_size) is not int or not 1 <= block_size <= 32
            or type(context) is not int or not block_size <= context <= min(1024, gate.max_context)
            or context % block_size or type(max_tokens) is not int or not 1 <= max_tokens <= 256
            or not isinstance(prompt, list) or not prompt or len(prompt)+max_tokens > context
            or any(type(token) is not int or not 0 <= token < net.graph['parent']['config']['vocab_size'] for token in prompt)
            or type(adapt_interfaces) is not bool or (interface is not None and not adapt_interfaces)):
        raise ValueError('Require a bounded fixed-block numerical prescription')
    net.check_context(prompt, max_tokens)
    scope = installed(net, interface) if adapt_interfaces else nullcontext({})
    with scope as roots:
        if adapt_interfaces and not roots:
            raise ValueError('Adapted block execution requires installed expert interfaces')
        request = {'format': FORMAT, 'graph': identity(net.graph), 'gate': commitment(gate),
                   'interfaces': roots, 'block_size': block_size, 'context': context,
                   'prompt': list(prompt), 'max_tokens': max_tokens,
                   'prefill': 'one-pass-through-complete-prompt-blocks-without-output-heads'}
        if net.all_owners.exchange(identity(request)) != [identity(request)]*net.world_size:
            raise ValueError('Owners disagree on the fixed-block numerical request')
        network = BlockNetwork(net, gate, block_size, interface)
        try:
            yield network, request
            network.unchanged()
        finally:
            network.close()


def proposal(net, sequence, maximum, method):
    """A drafter proposes inputs; it has no authority over accepted output."""
    observed, tokens = {}, None
    if method == 'hub' and maximum:
        if net.rank < 3:
            tokens = generate_branch_cached(net.preserved, sequence, maximum, False, observed)
    elif method == 'padding' or maximum == 0:
        tokens = [net.tokenizer.eos_token_id]*maximum if net.rank < 3 else None
        observed = {'seconds': 0., 'sent_tensor_bytes': 0}
    else:
        raise ValueError('Unknown fixed-block draft method')
    packet = net.all_owners.exchange({'tokens': tokens, 'resources': observed})
    tokens = packet[0]['tokens']
    if (not isinstance(tokens, list) or len(tokens) > maximum
            or any(row['tokens'] != tokens for row in packet[:3])
            or any(row['tokens'] is not None for row in packet[3:])):
        raise ValueError('Preserved drafter owners disagree')
    return tokens, [row['resources'] for row in packet]


def stream(net, gate, prompt, max_tokens, context, home, *, block_size=8,
           draft_method='hub', interface=None, adapt_interfaces=False):
    """Yield a checked prefix plus the target's first correction, if necessary.

    Each target call has the same query width and aligned position. Previously
    committed cache blocks contain only verified input. A rejected suffix is
    discarded locally before the next call. Every call emits at least one token,
    bounding target calls by the output allowance, in addition to prompt prefill.
    """
    if draft_method not in ('hub', 'padding'):
        raise ValueError('Unknown fixed-block draft method')
    home = Path(home)
    with session(net, gate, block_size, prompt, max_tokens, context, interface, adapt_interfaces) as (target, request):
        request = {**request, 'draft_method': draft_method}
        if net.all_owners.exchange(identity(request)) != [identity(request)]*net.world_size:
            raise ValueError('Owners disagree on the block proposal policy')
        home.mkdir(parents=True, exist_ok=False)
        save(home/'request.json', request)
        sequence, output, records, prefill = list(prompt), [], [], []
        position, eos = ((len(prompt)-1)//block_size)*block_size, net.tokenizer.eos_token_id
        if position:
            prefill.append(target.advance(sequence[:position], 0, prefill=True))
        save(home/'prefill.json', prefill)
        while len(output) < max_tokens:
            if len(records) >= max_tokens:
                raise ValueError('Fixed-block verification exceeded its per-output bound')
            known = len(sequence)-position
            if not 1 <= known <= block_size:
                raise ValueError('Verified prefix left its canonical query block')
            missing = min(block_size-known, max_tokens-len(output)-1)
            draft, resources = proposal(net, sequence, missing, draft_method)
            block = sequence[position:]+draft
            block += [eos]*(block_size-len(block))
            trace = target.advance(block, position)
            # Later speculative inputs cannot change an already emitted token
            # under the fixed-shape causal program. Fail closed if they do.
            for index in range(max(1, len(prompt)-position), known):
                if trace['predicted'][index-1] != sequence[position+index]:
                    raise ValueError('A future draft changed an already verified causal prefix')
            accepted, committed, end = [], known == block_size, False
            for index in range(known-1, block_size):
                token = trace['predicted'][index]
                accepted.append(token)
                sequence.append(token)
                output.append(token)
                end = token == eos or len(output) == max_tokens
                if end:
                    break
                if index+1 == block_size:
                    committed = True
                    break
                if token != block[index+1]:
                    break
            if committed:
                position += block_size
            else:
                target.rewind(position)
            record = {'request': identity(request), 'index': len(records),
                'offset': len(output)-len(accepted), 'tokens': accepted, 'end': end,
                'cache_committed': committed, 'target': trace, 'draft_owners': resources}
            records.append(record)
            save(home/('event-'+str(record['index'])+'.json'), record)
            if end:
                save(home/'result.json', {'request': request, 'tokens': output,
                    'events': [identity(row) for row in records], 'prefill_calls': len(prefill),
                    'prefill_positions': sum(len(row['input']) for row in prefill),
                    'complete': True})
            yield copy.deepcopy(record)
            if end:
                break


def verify(net, gate, prompt, reported, max_tokens, context, home, *, block_size=8,
           interface=None, adapt_interfaces=False):
    """Reconstruct every target cache from reported input; trust no peer cache."""
    eos = net.tokenizer.eos_token_id
    if (type(max_tokens) is not int or not 1 <= max_tokens <= 256
            or not isinstance(reported, list) or not reported or len(reported) > max_tokens
            or any(type(token) is not int or not 0 <= token < net.graph['parent']['config']['vocab_size'] for token in reported)
            or eos in reported[:-1] or (len(reported) < max_tokens and reported[-1] != eos)):
        raise ValueError('Require a complete bounded fixed-block response')
    home = Path(home)
    with session(net, gate, block_size, prompt, max_tokens, context, interface, adapt_interfaces) as (target, request):
        request = {**request, 'reported': reported}
        if net.all_owners.exchange(identity(request)) != [identity(request)]*net.world_size:
            raise ValueError('Auditors received different complete fixed-block responses')
        home.mkdir(parents=True, exist_ok=False)
        tokens = prompt+reported
        tokens += [eos]*((-len(tokens)) % block_size)
        traces, predicted = [], []
        start = ((len(prompt)-1)//block_size)*block_size
        if start:
            traces.append(target.advance(tokens[:start], 0, prefill=True))
        for position in range(start, len(tokens), block_size):
            trace = target.advance(tokens[position:position+block_size], position)
            traces.append(trace)
            predicted.extend(trace['predicted'])
        actual = predicted[len(prompt)-1-start:len(prompt)+len(reported)-1-start]
        mismatches = [i for i, (given, expected) in enumerate(zip(reported, actual)) if given != expected]
        result = {'request': request, 'predicted': actual, 'passed': not mismatches,
                  'mismatches': mismatches, 'blocks': traces}
        save(home/'result.json', result)
        return result
