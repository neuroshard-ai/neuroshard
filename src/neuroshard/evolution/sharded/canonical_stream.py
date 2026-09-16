"""Bounded chunks checked against the fixed-context causal inference program.

Cached generation proposes tokens. Batched re-execution either accepts the
chunk or corrects its first divergent token before anything is yielded. Every
owner must consume the iterator completely; client disconnection is a delivery
concern and must not strand the other owners in a collective operation.

This is an experimental numerical execution method, not an admitted native
service. It requires a separately committed profile and measured service costs.
"""
from pathlib import Path
import copy

from ..reference_data import identity, save
from .batched_audit import verify
from .fused_graph import commitment, generate_fused


def parameter_versions(model):
    return tuple((name, id(value), value._version) for name, value in model.named_parameters()) if model else ()


def stream(net, gate, prompt, max_tokens, context, home, *, chunk_tokens=8,
           interface=None, adapt_interfaces=False):
    """Yield only fully checked, append-only chunks from one pinned model.

    At most ``chunk_tokens + 1`` batched checks are needed for each chunk when
    fixed-shape causality holds: each correction fixes a strictly longer prefix.
    The bound is enforced, and an earlier prediction changing aborts execution.
    """
    if (type(chunk_tokens) is not int or not 1 <= chunk_tokens <= 32
            or type(max_tokens) is not int or not 1 <= max_tokens <= 256
            or type(context) is not int or not 2 <= context <= min(1024, gate.max_context)
            or not isinstance(prompt, list) or not prompt or len(prompt)+max_tokens > context
            or any(type(token) is not int or not 0 <= token < net.graph['parent']['config']['vocab_size'] for token in prompt)
            or type(adapt_interfaces) is not bool or (interface is not None and not adapt_interfaces)):
        raise ValueError('Require a bounded fixed-context streaming prescription')
    net.check_context(prompt, max_tokens)
    home = Path(home)
    home.mkdir(parents=True, exist_ok=False)
    request = {'format': 'neuroshard-canonical-stream-v1', 'graph': identity(net.graph),
               'gate': commitment(gate), 'prompt': list(prompt), 'max_tokens': max_tokens,
               'context': context, 'chunk_tokens': chunk_tokens, 'adapt_interfaces': adapt_interfaces}
    roots = net.all_owners.exchange(commitment(interface) if interface is not None else None)
    request['interfaces_by_owner'] = roots
    if net.all_owners.exchange(identity(request)) != [identity(request)]*net.world_size:
        raise ValueError('Owners disagree on the versioned stream request')
    signature = parameter_versions(gate), parameter_versions(interface)

    def unchanged():
        changed = signature != (parameter_versions(gate), parameter_versions(interface))
        try:
            net.verify_unchanged()
        except ValueError:
            changed = True
        if any(net.all_owners.exchange(changed)):
            raise ValueError('A streaming request cannot switch gate or expert interface weights')

    eos, output, records = net.tokenizer.eos_token_id, [], []
    while len(output) < max_tokens:
        unchanged()
        start, number = len(output), len(records)
        count = min(chunk_tokens, max_tokens-start)
        prefix, generation = prompt+output, {}
        draft = generate_fused(net, gate, prefix, count, generation,
                               interface=interface, adapt_interfaces=adapt_interfaces)
        generation_owners = net.all_owners.exchange(generation)
        fixed, checks = 0, []
        for attempt in range(count+1):
            unchanged()
            checked = verify(net, gate, prefix, draft, count, context,
                home/('chunk-'+str(number)+'-check-'+str(attempt)),
                interface=interface, adapt_interfaces=adapt_interfaces)
            checks.append(checked)
            result = checked['result']
            if result['passed']:
                break
            position = result['mismatches'][0]
            if position < fixed:
                raise ValueError('Fixed-context execution changed an already verified causal prefix')
            draft[position] = result['predicted'][position]
            fixed = position+1
            if draft[position] == eos:
                draft = draft[:position+1]
            elif len(draft) < count and draft[-1] != eos:
                draft.append(eos)
        else:
            raise ValueError('Canonical response repair exceeded its fixed causal bound')
        unchanged()
        output.extend(draft)
        end = draft[-1] == eos or len(output) == max_tokens
        record = {'request': identity(request), 'index': number, 'offset': start,
                  'tokens': list(draft), 'end': end, 'generation_owners': generation_owners, 'checks': checks}
        records.append(record)
        save(home/('chunk-'+str(number)+'.json'), record)
        if end:
            save(home/'result.json', {'request': request, 'tokens': output,
                'chunks': [identity(value) for value in records], 'complete': True})
        # No draft or correction is exposed until the whole chunk passes.
        yield copy.deepcopy(record)
        if end:
            break
