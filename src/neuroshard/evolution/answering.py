"""Commit the complete answering policy without embedding router matrices.

The ledger validates bounded obligations. Policy loading and funded numerical
replay establish which conversation and neural calls actually produced a reply.
"""
import copy
import json

from neuroshard.dataflow.store import canonical
from . import planned_metering, serving_graph
from .reference_data import identity
from .schema import integer, root
from .serving_diagnosis import conversation

FORMAT = 'neuroshard-complete-answering-v1'
MAX_POLICY_BYTES = 8 * 1024 * 1024
COUNTS = {'planning': 1, 'planning_repair': 1, 'directory_arguments': 2,
          'answer': 2, 'general_answer': 2, 'composition': 1}


def policy_json(raw):
    """Decode a policy artifact under its own limit, preserving strict JSON."""
    if len(raw) > MAX_POLICY_BYTES:
        raise ValueError('The complete answering policy exceeds its object bound')

    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError('Duplicate JSON key')
            result[key] = value
        return result

    def bad_constant(_value):
        raise ValueError('Nonfinite JSON number')

    return json.loads(raw, object_pairs_hook=pairs, parse_constant=bad_constant)


def core(graph):
    return {key: value for key, value in graph.items() if key != 'answering'}


def descriptor(config, graph, policy_root):
    """Derive the small ledger obligation from the complete numerical policy."""
    graph = core(graph)
    maximum = min(256, graph['tokenizer']['max_context'] - 1)
    tariff = {'prompt_atom_price': 0, 'output_atom_price': 1,
              'context': graph['tokenizer']['max_context']}
    bounds = planned_metering.limits(config, graph, maximum, tariff)
    extra, adapter = 0, None
    if 'planner_weights' in config:
        adapter = root(config['planner_weights']['fusion'])
        before = serving_graph.ownership(graph, 'interpreter')['2']
        extra = planned_metering.ownership(graph, config,
            {'model': 'interpreter', 'purpose': 'planning'})['2'] - before
    value = {'format': FORMAT, 'policy_root': root(policy_root), 'limits': bounds,
             'planner_adapter': adapter, 'planner_parameters': extra}
    validate_descriptor(value, graph)
    return value


def validate_descriptor(value, graph):
    serving_graph.fields(value, {'format', 'policy_root', 'limits', 'planner_adapter',
                                'planner_parameters'}, 'Invalid complete answering descriptor')
    if value['format'] != FORMAT:
        raise ValueError('Unsupported complete answering policy')
    root(value['policy_root'])
    bounds = value['limits']
    if (not isinstance(bounds, dict) or not {'planning', 'directory_arguments', 'answer'} <= set(bounds)
            or not set(bounds) <= set(COUNTS)):
        raise ValueError('Bound every supported neural call class')
    for name, maximum in bounds.items():
        integer(maximum, 1, 64 if name == 'directory_arguments' else 256)
        if maximum >= graph['tokenizer']['max_context']:
            raise ValueError('Reserve context for each complete neural prompt')
    integer(value['planner_parameters'], 0, 64_000_000)
    if value['planner_adapter'] is None:
        if value['planner_parameters']:
            raise ValueError('An absent planner adapter cannot claim parameters')
    else:
        root(value['planner_adapter'])
        if not value['planner_parameters']:
            raise ValueError('A planner adapter needs its complete ownership inventory')


def attach(graph, config, store):
    """Store the policy and return a graph committing to that whole service."""
    graph = copy.deepcopy(core(graph))
    serving_graph.validate(graph, allow_untrained=True)
    if config['graph'] != identity(graph) or config['learned']['graph'] != identity(graph):
        raise ValueError('The answering policy must bind this exact model graph')
    policy = copy.deepcopy(config)
    del policy['graph']
    del policy['learned']['graph']
    payload = {'format': FORMAT + '/policy', 'configuration': policy}
    if len(canonical(payload)) > MAX_POLICY_BYTES:
        raise ValueError('The complete answering policy exceeds its object bound')
    graph['answering'] = descriptor(config, graph, store.put_json(payload))
    return serving_graph.validate(graph, allow_untrained=True)


def load(graph, store):
    """Materialize graph references and verify the committed billing bounds."""
    value = graph['answering']
    validate_descriptor(value, graph)
    raw = store.get(value['policy_root'])
    payload = policy_json(raw)
    serving_graph.fields(payload, {'format', 'configuration'}, 'Invalid answering policy object')
    if identity(payload) != value['policy_root'] or payload['format'] != FORMAT + '/policy':
        raise ValueError('The answering policy object changed')
    config = copy.deepcopy(payload['configuration'])
    if 'graph' in config or 'graph' in config['learned']:
        raise ValueError('A policy cannot substitute its own model graph')
    model = core(graph)
    config['graph'] = config['learned']['graph'] = identity(model)
    if descriptor(config, model, value['policy_root']) != value:
        raise ValueError('The policy differs from its committed neural obligations')
    return config


def quote(graph, maximum, unit_price):
    value = graph['answering']
    validate_descriptor(value, graph)
    integer(maximum, 1, value['limits']['answer'])
    integer(unit_price, 1, 10**9)
    bounds = dict(value['limits'], answer=maximum)
    if 'composition' in bounds:
        bounds['composition'] = min(bounds['composition'], maximum)
    return {'format': FORMAT + '/quote', 'graph': identity(graph), 'max_tokens': maximum,
            'limits': bounds, 'call_counts': {name: COUNTS[name] for name in bounds},
            'maximum_atoms': sum(COUNTS[name] * bound for name, bound in bounds.items()) * unit_price,
            'unit_price': unit_price}


def payments(graph, maximum, response, unit_price):
    """Check bounded receipts; only full replay can authorize these payments."""
    offer = quote(graph, maximum, unit_price)
    if not isinstance(response, dict):
        raise ValueError('Require a complete answering response')
    outputs = response.get('outputs')
    if (response.get('format') != 'neuroshard-planned-graph-service-v1/response'
            or response.get('status') not in ('completed', 'needs_clarification')
            or not isinstance(outputs, list) or not 1 <= len(outputs) <= sum(offer['call_counts'].values())):
        raise ValueError('Require a bounded complete answering response')
    counts, paid, total = dict.fromkeys(offer['limits'], 0), {}, 0
    vocabulary, eos = graph['parent']['config']['vocab_size'], graph['tokenizer']['eos_id']
    for call in outputs:
        serving_graph.fields(call, {'model', 'purpose', 'owners', 'prompt_ids', 'token_ids'}
            | ({'planner_adapter'} if 'planner_adapter' in call else set()), 'Invalid complete neural receipt')
        purpose, model = call['purpose'], call['model']
        if (not isinstance(purpose, str) or not isinstance(model, str)
                or purpose not in counts or model not in {'parent', 'interpreter', *graph['experts']}
                or purpose != 'answer' and model != 'interpreter'):
            raise ValueError('A neural call changed its declared role or model')
        counts[purpose] += 1
        if counts[purpose] > offer['call_counts'][purpose]:
            raise ValueError('Too many neural calls for this request')
        if counts['answer'] + counts.get('general_answer', 0) > 2:
            raise ValueError('Require at most two complete answers to one request')
        prompt, tokens = call['prompt_ids'], call['token_ids']
        if (not isinstance(prompt, list) or not prompt or not isinstance(tokens, list)
                or not 1 <= len(tokens) <= offer['limits'][purpose]
                or len(prompt) + offer['limits'][purpose] > graph['tokenizer']['max_context']):
            raise ValueError('A neural call exceeds its context or output allowance')
        for token in prompt + tokens:
            integer(token, 0, vocabulary - 1)
        if eos in tokens[:-1] or len(tokens) < offer['limits'][purpose] and tokens[-1] != eos:
            raise ValueError('A neural receipt changed greedy stopping')
        adapter = graph['answering']['planner_adapter'] if purpose == 'planning' else None
        if (('planner_adapter' in call) != (adapter is not None)
                or call.get('planner_adapter') != adapter):
            raise ValueError('A neural call changed the installed planner adapter')
        sizes = serving_graph.ownership(graph, model)
        if adapter:
            sizes['2'] += graph['answering']['planner_parameters']
        if (not isinstance(call['owners'], list) or any(type(rank) is not int for rank in call['owners'])
                or call['owners'] != sorted(map(int, sizes))):
            raise ValueError('A neural receipt assigned different shard owners')
        amount = len(tokens) * unit_price
        shares = {rank: amount * size // sum(sizes.values()) for rank, size in sizes.items()}
        for rank in sorted(shares, key=int)[:amount - sum(shares.values())]:
            shares[rank] += 1
        for rank, share in shares.items():
            paid[rank] = paid.get(rank, 0) + share
        total += len(tokens)
    if (type(response.get('generated_tokens')) is not int or response['generated_tokens'] != total
            or sum(paid.values()) > offer['maximum_atoms']
            or not isinstance(response.get('text'), str) or len(response['text'].encode()) > 32768):
        raise ValueError('Response text, generation count or price changed')
    return paid


def check_request(response, request):
    if not isinstance(response, dict) or not isinstance(response.get('request'), dict):
        raise ValueError('Require the response and its complete original request')
    actual = response['request']
    serving_graph.fields(actual, {'service', 'messages', 'max_tokens'}, 'Invalid answering request binding')
    integer(actual['max_tokens'], 1, 256)
    root(response.get('service'))
    if (actual['service'] != response['service'] or actual['messages'] != request['messages']
            or actual['max_tokens'] != request['max_tokens']):
        raise ValueError('The response changed its reserved conversation')
