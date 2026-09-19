"""Train small causal interfaces while keeping every installed expert weight.

Adapters are separate modules attached through scoped forward hooks. The frozen
partition's parameter names, values and optimizer state remain untouched. Their
commitments must be included separately in any adapted execution request.
"""
from contextlib import contextmanager

import torch
from torch import nn


class ExpertInterface(nn.Module):
    def __init__(self, partition, source_root, rank=8):
        super().__init__()
        if (type(rank) is not int or not 1 <= rank <= 128 or partition.rank == 0
                or not isinstance(source_root, str) or len(source_root) != 64
                or any(c not in '0123456789abcdef' for c in source_root)
                or any(p.requires_grad for p in partition.parameters())):
            raise ValueError('Require a frozen owned expert and bounded adapter rank')
        self.source_root, self.rank = source_root, rank
        self.inventory = {name: (module.in_features, module.out_features)
                          for name, module in partition.layers.named_modules()
                          if isinstance(module, nn.Linear)}
        if not self.inventory or len(self.inventory) > 224:
            raise ValueError('Invalid owned interface projection inventory')
        self.adapters = nn.ModuleDict()
        for name, (incoming, outgoing) in self.inventory.items():
            pair = nn.Sequential(nn.Linear(incoming, rank, bias=False), nn.Linear(rank, outgoing, bias=False))
            nn.init.zeros_(pair[1].weight)
            self.adapters[name.replace('.', '/')] = pair
        self.to(device=next(partition.parameters()).device, dtype=torch.float32)
        self.attached = False

    def descriptor(self):
        return {'format': 'neuroshard-owned-expert-interface-v1', 'source': self.source_root,
                'rank': self.rank, 'projections': {key: list(value) for key, value in self.inventory.items()}, 'scale': 1,
                'initialization': 'zero-output', 'base_weights': 'frozen'}

    @contextmanager
    def attach(self, partition):
        if self.attached or self.inventory != {name: (module.in_features, module.out_features)
                for name, module in partition.layers.named_modules() if isinstance(module, nn.Linear)}:
            raise ValueError('Interface already attached or installed on a different projection layout')
        hooks = []
        self.attached = True
        try:
            for name, module in partition.layers.named_modules():
                if name not in self.inventory:
                    continue
                adapter = self.adapters[name.replace('.', '/')]
                def add(_module, inputs, output, adapter=adapter):
                    delta = adapter(inputs[0]).to(output.dtype)
                    # Preserve exact base inference for a zero interface, even
                    # signed zero. Training retains the derivative at zero.
                    if not torch.is_grad_enabled():
                        return torch.where(delta == 0, output, output+delta)
                    return output+delta
                hooks.append(module.register_forward_hook(add))
            yield self
        finally:
            for hook in hooks:
                hook.remove()
            self.attached = False


def supervision(row, tokenizer, sources):
    """Attribute training answer spans, never inference inputs, to their sources.

    Corpus provenance establishes the source of each reference in answer order.
    Delimiters are left to the joint objective. The final source also learns EOS.
    No auxiliary target is manufactured for ordinary assistant conversations.
    """
    labels = {name: [-100]*len(row['input_ids']) for name in sources}
    if row['kind'] not in ('directory', 'protocol', 'mixed'):
        return labels
    if len(row['messages']) != 2 or len(row['groups']) != len(row['references']):
        raise ValueError('Require complete single-turn specialist provenance')
    answer = row['messages'][-1]['content']
    if answer != '; '.join(row['references']):
        raise ValueError('Reference spans differ from the actual training answer')
    text = tokenizer.apply_chat_template(row['messages'], tokenize=False, add_generation_prompt=False)
    prefix = tokenizer.apply_chat_template(row['messages'][:-1], tokenize=False, add_generation_prompt=True)
    encoded = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
    if encoded['input_ids'] != row['input_ids'] or text[len(prefix):len(prefix)+len(answer)] != answer:
        raise ValueError('Tokenizer offsets do not reproduce the committed conversation')
    at, last = len(prefix), None
    for group, reference in zip(row['groups'], row['references']):
        source = 'directory' if group.startswith('person:') else 'protocol' if group.startswith('topic:') else None
        if source not in labels:
            raise ValueError('Unknown training reference provenance')
        end = at+len(reference)
        # Byte-level tokenizers can include the leading inter-answer space.
        # Supervise only tokens overlapping this answer, never a delimiter.
        for index, (start, stop) in enumerate(encoded['offset_mapping']):
            if row['labels'][index] != -100 and start < end and stop > at:
                labels[source][index] = row['labels'][index]
        at, last = end+2, source
    for index, token in enumerate(row['labels']):
        if token == tokenizer.eos_token_id:
            labels[last][index] = token
    if any(not any(value != -100 for value in labels[name]) for name in {
            'directory' if group.startswith('person:') else 'protocol' for group in row['groups']}):
        raise ValueError('Specialist answer has no attributable tokens')
    return labels


@contextmanager
def installed(net, interface):
    """Bind all owned adapters before any adapted source execution."""
    from .fused_graph import commitment
    from ..reference_data import identity
    owners = {row['owner']: row['id'] for row in net.graph['descriptor']['rules']}
    if interface is not None and (net.rank not in owners or interface.training
            or interface.source_root != identity(net.graph['experts'][owners[net.rank]])):
        raise ValueError('An inference interface must bind its installed frozen source')
    roots = net.all_owners.exchange(commitment(interface) if interface is not None else None)
    if any(roots) and any((value is not None) != (rank in owners) for rank, value in enumerate(roots)):
        raise ValueError('Every expert owner must commit an interface, or all must use the frozen graph')
    inventory = {owners[rank]: root for rank, root in enumerate(roots) if root is not None}
    versions = tuple(parameter._version for parameter in interface.parameters()) if interface is not None else ()
    context = interface.attach(net.shard) if interface is not None else None
    try:
        if context is not None:
            context.__enter__()
        yield inventory
        if interface is not None and versions != tuple(parameter._version for parameter in interface.parameters()):
            raise ValueError('Expert interface changed during committed source execution')
    finally:
        if context is not None:
            context.__exit__(None, None, None)
