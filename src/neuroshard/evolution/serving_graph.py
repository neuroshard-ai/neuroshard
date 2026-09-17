"""Tensor-free commitments and billing for the measured expert graph.

These checks describe execution obligations. They neither execute a model nor
turn a publisher's transcript into evidence of correct neural computation.
"""
import math
import re

from . import expert_checkpoint
from .reference_data import identity
from .schema import integer, root
from .sharded import composition

FORMAT = 'neuroshard-serving-expert-graph-v1'
FIELDS = {'format', 'descriptor', 'parent', 'experts', 'interpreter_assets',
          'interpreter_prompt', 'tokenizer', 'numerical_profile', 'executor_root'}
PRIOR = 'neuroshard-preserved-interpreter-v1/graph'
COMPOSED = 'neuroshard-composed-cohort-v1/graph'
EXTENSIBLE = 'neuroshard-extensible-expert-v1/graph'
DIRECTORY = {'id': 'directory', 'needle': 'fictional luma directory', 'owner': 3}
PROTOCOL = {'id': 'protocol', 'needle': 'neuroshard 0.4.0', 'owner': 4}


def fields(value, expected, message):
    if not isinstance(value, dict) or set(value) != set(expected):
        raise ValueError(message)


def rules(graph):
    descriptor = graph['descriptor']
    return [DIRECTORY] if descriptor['format'] == PRIOR else descriptor['rules']


def validate(graph, *, allow_untrained=False):
    fields(graph, FIELDS | ({'answering'} if isinstance(graph, dict) and 'answering' in graph else set()),
           'Invalid serving graph schema')
    if graph['format'] != FORMAT:
        raise ValueError('Unsupported serving graph format')
    parent, descriptor = graph['parent'], graph['descriptor']
    inherited = expert_checkpoint.parent_records(parent)
    shapes = {name: spec['shape'] for name, spec in inherited.items()}
    parent_size = sum(math.prod(shape) for shape in shapes.values())
    common = {'format', 'parent', 'split', 'parent_layout', 'expert_layout',
              'interpreter', 'interpretation', 'total_parameters'}
    if descriptor.get('format') == PRIOR:
        fields(descriptor, common | {'expert', 'selector', 'parent_parameters',
            'added_parameters', 'added_tensors', 'interpreter_parameters'}, 'Invalid prior graph descriptor')
        # This is the exact already evaluated rule, not an arbitrary executable
        # expression supplied in a graph descriptor.
        selector = ('Use the added expert exactly when the user question contains fictional Luma directory, '
            'case-insensitively. No entity, attribute, expected answer or evaluation ID enters routing. '
            'Otherwise execute the unchanged parent through its independent three-owner process group.')
        if descriptor['selector'] != selector:
            raise ValueError('The prior graph changed its evaluated selector')
        declared = {'directory': descriptor['expert']}
    elif descriptor.get('format') in (COMPOSED, EXTENSIBLE):
        fields(descriptor, common | {'experts', 'rules', 'previous_graph', 'composition',
            'tokenizer', 'interpreter_prompt'}, 'Invalid composed graph descriptor')
        route_rules = descriptor['rules']
        if descriptor['format'] == COMPOSED:
            if route_rules != [DIRECTORY, PROTOCOL]:
                raise ValueError('Changed measured routing, composition or prompt binding')
        else:
            if (not isinstance(route_rules, list) or not 3 <= len(route_rules) <= 64
                    or route_rules[:2] != [DIRECTORY, PROTOCOL]):
                raise ValueError('An extension must preserve the earlier expert routes')
            identifiers, needles = set(), set()
            for index, rule in enumerate(route_rules):
                fields(rule, {'id', 'needle', 'owner'}, 'Invalid appended expert route')
                if (not isinstance(rule['id'], str) or not re.fullmatch('[a-z][a-z0-9-]{0,63}', rule['id'])
                        or rule['id'] in {'parent', 'interpreter', *identifiers}
                        or not isinstance(rule['needle'], str) or not 1 <= len(rule['needle'].encode()) <= 256
                        or rule['needle'] != rule['needle'].strip().casefold() or rule['needle'] in needles
                        or type(rule['owner']) is not int or rule['owner'] != 3 + index):
                    raise ValueError('Require distinct ordered expert identities and owners')
                identifiers.add(rule['id'])
                needles.add(rule['needle'])
        if (descriptor['composition'] != composition.FORMAT
                or descriptor['interpreter_prompt'] != graph['interpreter_prompt']
                or descriptor['tokenizer'] != graph['tokenizer']['root']):
            raise ValueError('Changed measured routing, composition or prompt binding')
        root(descriptor['previous_graph'])
        if (not isinstance(descriptor['experts'], list) or len(descriptor['experts']) != len(route_rules)
                or any(set(row) != {'id', 'checkpoint'} for row in descriptor['experts'])
                or [row['id'] for row in descriptor['experts']] != [rule['id'] for rule in route_rules]):
            raise ValueError('Bind every ordered expert to its checkpoint')
        declared = {row['id']: row['checkpoint'] for row in descriptor['experts']}
    else:
        raise ValueError('Unsupported evaluated graph descriptor')
    if (descriptor['parent'] != identity(parent) or descriptor['parent_layout'] != parent['boundaries']
            or len(parent['boundaries']) != 4 or set(graph['experts']) != set(declared)):
        raise ValueError('Graph parent, partitions or experts differ from their commitments')
    extra = 0
    for name, checkpoint in graph['experts'].items():
        expert_checkpoint.unpack(parent, checkpoint)
        if (checkpoint['checkpoint'] != declared[name] or checkpoint['step'] < (0 if allow_untrained else 1)
                or checkpoint['split'] != descriptor['split']
                or checkpoint['boundaries'] != descriptor['expert_layout']
                or checkpoint['boundaries'][:-2] != parent['boundaries'][:-1]):
            raise ValueError('Serving requires the exact trained tail and its ownership')
        extra += sum(math.prod(spec['shape']) for spec in checkpoint['tensors'].values())
    assets, source = graph['interpreter_assets'], descriptor['interpreter']
    fields(assets, {'format', 'boundaries', 'partitions', 'source', 'source_weight_sha256'},
           'Invalid interpreter asset inventory')
    if (assets['format'] != 'neuroshard-preserved-interpreter-assets-v1'
            or identity(assets) != source['partitioned_assets']
            or assets['source_weight_sha256'] != source['weight_sha256']
            or assets['source'] != {k: source[k] for k in ('repo', 'revision', 'license', 'parameters')}
            or assets['boundaries'] != parent['boundaries'] or set(assets['partitions']) != {'0', '1', '2'}):
        raise ValueError('Interpreter provenance or inventory changed')
    found = {}
    for rank in range(3):
        part = assets['partitions'][str(rank)]
        expected = {name for name in shapes if expert_checkpoint.owner(name, parent['boundaries']) == rank}
        if (part['rank'] != rank or set(part['tensors']) != expected
                or part['parameters'] != sum(math.prod(shapes[name]) for name in expected)):
            raise ValueError('Incomplete interpreter partition')
        for name, spec in part['tensors'].items():
            fields(spec, {'sha256', 'bytes', 'shape', 'dtype', 'file'}, 'Invalid interpreter tensor')
            root(spec['sha256'])
            integer(spec['bytes'], 1, 16 * 1024**3)
            if (spec['shape'] != shapes[name] or spec['dtype'] != 'bfloat16'
                    or spec['file'] != spec['sha256'] + '.safetensors'):
                raise ValueError('Interpreter tensor shape, dtype or file changed')
        found.update(part['tensors'])
    if (source['parameters'] != parent_size or descriptor['total_parameters'] != 2 * parent_size + extra):
        raise ValueError('Graph parameter inventory does not add up')
    if descriptor['format'] == PRIOR and (
            descriptor['parent_parameters'] != parent_size or descriptor['interpreter_parameters'] != parent_size
            or descriptor['added_parameters'] != extra
            or descriptor['added_tensors'] != graph['experts']['directory']['tensors']):
        raise ValueError('Prior graph parameter commitments changed')
    tokenizer = graph['tokenizer']
    fields(tokenizer, {'root', 'files', 'eos_id', 'max_context'}, 'Invalid versioned tokenizer')
    root(tokenizer['root'])
    if not isinstance(tokenizer['files'], dict) or not 1 <= len(tokenizer['files']) <= 16:
        raise ValueError('Require the exact tokenizer file inventory')
    for name, digest in tokenizer['files'].items():
        if not isinstance(name, str) or '/' in name or '\\' in name or name in ('', '.', '..'):
            raise ValueError('Tokenizer files must be plain filenames')
        root(digest)
    integer(tokenizer['eos_id'], 0, parent['config']['vocab_size'] - 1)
    integer(tokenizer['max_context'], 1, min(4096, parent['config']['max_position_embeddings']))
    fields(graph['interpreter_prompt'], {'format', 'messages', 'tokens'}, 'Invalid interpretation prompt')
    if graph['interpreter_prompt']['format'] != 'name-field-json-v1':
        raise ValueError('Unsupported interpretation prompt serialization')
    for key in ('messages', 'tokens'):
        root(graph['interpreter_prompt'][key])
    interpretation = descriptor['interpretation']
    fields(interpretation, {'instruction', 'examples', 'max_tokens', 'instruction_placement', 'invalid'},
           'Invalid interpretation policy')
    integer(interpretation['max_tokens'], 1, 64)
    if interpretation['instruction_placement'] != 'after-quoted-question':
        raise ValueError('Unsupported interpretation prompt placement')
    for key in ('numerical_profile', 'executor_root'):
        root(graph[key])
    if 'answering' in graph:
        from .answering import validate_descriptor
        validate_descriptor(graph['answering'], graph)
    return graph


def calls(graph, question, max_tokens):
    """Derive bounded neural calls using only user text and committed routing."""
    if 'answering' in graph:
        raise ValueError('Complete answering requires its committed policy executor')
    selected = next((rule['id'] for rule in rules(graph) if isinstance(question, str)
                     and rule['needle'] in question.casefold()), None)
    return selected_calls(graph, selected, question, max_tokens)


def selected_calls(graph, selected, question, max_tokens):
    """Plan an already selected path; the caller must bind and audit selection."""
    if not isinstance(question, str) or not question or len(question.encode()) > 32768:
        raise ValueError('Require bounded raw user text')
    integer(max_tokens, 1, 256)
    if selected is not None and selected not in graph['experts']:
        raise ValueError('Selected expert is absent from the graph')
    if selected == 'directory':
        return [{'model': 'interpreter', 'question': question,
                 'max_tokens': graph['descriptor']['interpretation']['max_tokens']},
                {'model': 'directory', 'question': {'interpretation_of': question}, 'max_tokens': max_tokens}]
    if selected == 'protocol':
        return [{'model': 'protocol', 'question': part, 'max_tokens': max_tokens}
                for part in (composition.questions(question) or [question])]
    if selected is not None:
        return [{'model': selected, 'question': question, 'max_tokens': max_tokens}]
    return [{'model': 'parent', 'question': question, 'max_tokens': max_tokens}]


def ownership(graph, model):
    """Count actual parameters on each participating owner for one neural call."""
    parent = graph['parent']
    if model == 'interpreter':
        return {str(rank): part['parameters'] for rank, part in
                ((rank, graph['interpreter_assets']['partitions'][str(rank)]) for rank in range(3))}
    if model == 'parent':
        tensors, boundaries, ranks = parent['tensors'], parent['boundaries'], [0, 1, 2]
    else:
        checkpoint = expert_checkpoint.unpack(parent, graph['experts'][model])
        tensors, boundaries = checkpoint['tensors'], checkpoint['boundaries']
        ranks = [0, 1, 2, next(rule['owner'] for rule in rules(graph) if rule['id'] == model)]
    sizes = {str(rank): 0 for rank in ranks}
    for name, spec in tensors.items():
        sizes[str(ranks[expert_checkpoint.owner(name, boundaries)])] += math.prod(spec['shape'])
    return sizes


def maximum_price(graph, plan, unit_price):
    integer(unit_price, 1, 10**9)
    return sum(call['max_tokens'] for call in plan) * unit_price


def payments(graph, plan, outputs, unit_price):
    """Pay actual greedy tokens, including interpretation and composed calls.

    Neural auditors additionally verify prompt construction, logits, decoding
    and rendering. This arithmetic does not verify an output's correctness.
    """
    if not isinstance(outputs, list) or len(outputs) != len(plan):
        raise ValueError('Account for every planned neural call')
    integer(unit_price, 1, 10**9)
    paid = {}
    for call, output in zip(plan, outputs):
        fields(output, {'model', 'prompt_root', 'token_ids'}, 'Invalid neural call receipt')
        root(output['prompt_root'])
        tokens = output['token_ids']
        if (output['model'] != call['model'] or not isinstance(tokens, list)
                or not 1 <= len(tokens) <= call['max_tokens']):
            raise ValueError('Neural output differs from its planned call')
        for token in tokens:
            integer(token, 0, graph['parent']['config']['vocab_size'] - 1)
        eos = graph['tokenizer']['eos_id']
        if eos in tokens[:-1] or (len(tokens) < call['max_tokens'] and tokens[-1] != eos):
            raise ValueError('Each neural call must obey greedy stopping')
        sizes = ownership(graph, call['model'])
        amount, total = len(tokens) * unit_price, sum(sizes.values())
        shares = {rank: amount * size // total for rank, size in sizes.items()}
        for rank in sorted(shares, key=int)[:amount - sum(shares.values())]:
            shares[rank] += 1
        for rank, share in shares.items():
            paid[rank] = paid.get(rank, 0) + share
    return paid


def execution_identity(graph, question, max_tokens):
    """Fingerprint the deterministic computation, excluding unused experts.

    Equal fingerprints preserve a particular old answer under the committed
    executor and numerical assumptions, including an old wrong answer. They
    are not evidence of new quality, data availability or hardware agreement.
    """
    plan = calls(graph, question, max_tokens)
    models = {}
    for call in plan:
        name = call['model']
        if name == 'parent':
            models[name] = identity(graph['parent'])
        elif name == 'interpreter':
            models[name] = graph['descriptor']['interpreter']['partitioned_assets']
        else:
            models[name] = graph['experts'][name]['checkpoint']
    value = {'format': FORMAT + '/execution', 'calls': plan, 'models': models,
             'tokenizer': graph['tokenizer'], 'numerical_profile': graph['numerical_profile'],
             'executor_root': graph['executor_root']}
    if 'interpreter' in models:
        value.update(interpretation=graph['descriptor']['interpretation'],
                     interpreter_prompt=graph['interpreter_prompt'])
    return identity(value)
