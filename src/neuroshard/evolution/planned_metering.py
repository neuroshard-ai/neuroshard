"""Bound complete learned-service obligations without running neural code.

This accounts for a transcript, not its truth. A funded neural replay must
verify the complete response before any payment. Serialized source values are
not new neural output tokens. Prices are integer ledger atoms, never USD.
"""
from . import serving_graph
from .reference_data import identity
from .schema import integer

FORMAT = 'neuroshard-planned-metering-v1'


def limits(service, graph, maximum, tariff):
    serving_graph.fields(tariff, {'prompt_atom_price', 'output_atom_price', 'context'},
                         'Invalid complete-call tariff')
    integer(tariff['prompt_atom_price'], 0, 10**9)
    integer(tariff['output_atom_price'], 1, 10**9)
    context = integer(tariff['context'], 16, 1024)
    maximum = integer(maximum, 1, 256)
    if service['graph'] != identity(graph) or context != graph['tokenizer']['max_context']:
        raise ValueError('The price quote differs from the installed graph')
    result = {'planning': integer(service['planner']['max_tokens'], 1, 256),
              'directory_arguments': integer(graph['descriptor']['interpretation']['max_tokens'], 1, 64),
              'answer': maximum}
    if 'composer' in service:
        result['composition'] = min(maximum, integer(service['composer']['max_tokens'], 1, 256))
    if 'general_answer_policy' in service:
        from .general_answer import FORMAT as GENERAL_ANSWER, MAX_TOKENS
        if service['general_answer_policy'] != GENERAL_ANSWER:
            raise ValueError('Unknown worked general answer policy')
        result['general_answer'] = MAX_TOKENS
    if 'request_policy' in service:
        from .request_planning import FORMAT as REQUEST_POLICY, REPAIR_TOKENS, ASSISTANT_POLICIES
        if service['request_policy'] not in (REQUEST_POLICY, *ASSISTANT_POLICIES):
            raise ValueError('Unknown request-preservation policy')
        result['planning_repair'] = REPAIR_TOKENS
    if any(value >= context for value in result.values()):
        raise ValueError('Reserve room for every complete neural prompt and output')
    return result


def quote(service, graph, maximum, tariff):
    bounds = limits(service, graph, maximum, tariff)
    counts = {'planning': 1, 'directory_arguments': 2, 'answer': 2,
              **({'composition': 1} if 'composition' in bounds else {}),
              **({'general_answer': 2} if 'general_answer' in bounds else {}),
              **({'planning_repair': 1} if 'planning_repair' in bounds else {})}
    # Reserve each call's maximum prompt and output separately. This is a
    # conservative bound even when one token class costs more than the other.
    prompt_tokens = sum((tariff['context']-1)*count for count in counts.values())
    output_tokens = sum(bounds[kind]*count for kind, count in counts.items())
    return {'format': FORMAT+'/quote', 'service': identity(service), 'tariff': identity(tariff),
            'scope': 'provider_neural_execution', 'separately_funded': ['verification', 'retention'],
            'max_tokens': maximum, 'call_limits': counts, 'prompt_tokens': prompt_tokens,
            'output_tokens': output_tokens,
            'maximum_atoms': prompt_tokens*tariff['prompt_atom_price']+output_tokens*tariff['output_atom_price']}


def ownership(graph, service, call):
    counts = serving_graph.ownership(graph, call['model'])
    if call['purpose'] == 'planning' and 'planner_weights' in service:
        layout = service['planner_weights']['binding']['layout']
        rank = integer(layout['rank'], 1, 256)
        extra = 0
        if not isinstance(layout['projections'], dict) or not 1 <= len(layout['projections']) <= 512:
            raise ValueError('Bound the installed planner projection inventory')
        for shape in layout['projections'].values():
            if not isinstance(shape, list) or len(shape) != 2:
                raise ValueError('Require complete planner projection dimensions')
            extra += rank*sum(integer(side, 1, 10**6) for side in shape)
        counts['2'] += extra
    return counts


def meter(service, graph, response, tariff):
    request = response['request']
    if (response['service'] != identity(service) or request['service'] != identity(service)
            or response['format'] != 'neuroshard-planned-graph-service-v1/response'):
        raise ValueError('Response differs from its complete service identity')
    bounds = limits(service, graph, request['max_tokens'], tariff)
    offer = quote(service, graph, request['max_tokens'], tariff)
    outputs = response['outputs']
    if not isinstance(outputs, list) or not 1 <= len(outputs) <= sum(offer['call_limits'].values()):
        raise ValueError('Bound every actual neural call')
    integer(response['generated_tokens'], 1, offer['output_tokens'])
    counts = dict.fromkeys(offer['call_limits'], 0)
    payments, prompt_count, output_count, details = {}, 0, 0, []
    models = {'parent', 'interpreter', *graph['experts']}
    vocabulary, eos = graph['parent']['config']['vocab_size'], graph['tokenizer']['eos_id']
    from .request_planning import direct_question, atomic_request, explicit_questions, ASSISTANT_POLICIES, LOSSLESS_POLICY
    direct = 'request_policy' in service and direct_question(request['messages']) is not None
    if service.get('request_policy') in ASSISTANT_POLICIES:
        # This checks bounded accounting, not the selector's truth. Complete
        # neural replay still authorizes the actual general-routing decision.
        general = response.get('planning', {}).get('path') == 'general'
        if general and (len(outputs) != 1 or outputs[0].get('purpose') not in ('answer', 'general_answer')
                or outputs[0].get('model') != 'interpreter'
                or response.get('plan') != [request['messages'][-1]['content']]):
            raise ValueError('The general path must answer the intact request once')
        direct = general or atomic_request(request['messages']) is not None
        explicit = explicit_questions(request['messages']) if service['request_policy'] == LOSSLESS_POLICY else None
        if explicit is not None:
            if response.get('planning', {}).get('path') != 'explicit' or response.get('plan') != explicit:
                raise ValueError('Explicit questions must preserve the complete literal spans')
            direct = True
    for index, call in enumerate(outputs):
        serving_graph.fields(call, {'model', 'purpose', 'owners', 'prompt_ids', 'token_ids'}
                             | ({'planner_adapter'} if 'planner_adapter' in call else set()),
                             'Invalid complete neural call receipt')
        purpose, model = call['purpose'], call['model']
        if (not isinstance(purpose, str) or not isinstance(model, str)
                or purpose not in counts or model not in models
                or (not direct and (index == 0) != (purpose == 'planning'))
                or (direct and purpose in ('planning', 'planning_repair'))
                or (purpose == 'planning_repair' and (index != 1 or outputs[0]['purpose'] != 'planning'))
                or purpose != 'answer' and model != 'interpreter'
                or purpose == 'composition' and index != len(outputs)-1):
            raise ValueError('Invalid service call order or model')
        counts[purpose] += 1
        if counts[purpose] > offer['call_limits'][purpose]:
            raise ValueError('Service call inventory exceeds its reservation')
        if counts['answer'] + counts.get('general_answer', 0) > 2:
            raise ValueError('Require at most two complete answers to one request')
        expected_adapter = service.get('planner_weights', {}).get('fusion') if purpose == 'planning' else None
        if (('planner_adapter' in call) != (expected_adapter is not None)
                or call.get('planner_adapter') != expected_adapter):
            raise ValueError('A neural call changed its installed planner adapter')
        prompt, tokens = call['prompt_ids'], call['token_ids']
        if (not isinstance(prompt, list) or not prompt or not isinstance(tokens, list)
                or not 1 <= len(tokens) <= bounds[purpose]
                or len(prompt)+bounds[purpose] > tariff['context']):
            raise ValueError('Neural call exceeds its prompt or generation reservation')
        for token in prompt+tokens:
            integer(token, 0, vocabulary-1)
        if eos in tokens[:-1] or len(tokens) < bounds[purpose] and tokens[-1] != eos:
            raise ValueError('A neural call changed greedy stopping')
        sizes = ownership(graph, service, call)
        if (not isinstance(call['owners'], list) or any(type(rank) is not int for rank in call['owners'])
                or call['owners'] != sorted(map(int, sizes))):
            raise ValueError('A neural call assigned payment to different shard owners')
        amount = len(prompt)*tariff['prompt_atom_price']+len(tokens)*tariff['output_atom_price']
        shares = {rank: amount*size//sum(sizes.values()) for rank, size in sizes.items()}
        for rank in sorted(shares, key=int)[:amount-sum(shares.values())]:
            shares[rank] += 1
        for rank, share in shares.items():
            payments[rank] = payments.get(rank, 0)+share
        prompt_count += len(prompt)
        output_count += len(tokens)
        details.append({'index': index, 'purpose': purpose, 'model': model,
                        'prompt_tokens': len(prompt), 'output_tokens': len(tokens), 'payments': shares})
    if (response['generated_tokens'] != output_count or sum(payments.values()) > offer['maximum_atoms']
            or response['status'] not in ('completed', 'needs_clarification')):
        raise ValueError('Response changes complete metering or exceeds its quote')
    return {'format': FORMAT+'/receipt', 'quote': identity(offer), 'response': identity(response),
            'prompt_tokens': prompt_count, 'output_tokens': output_count, 'calls': details,
            'payments': payments, 'total_atoms': sum(payments.values()),
            'unused_reserved_atoms': offer['maximum_atoms']-sum(payments.values()),
            'requires_complete_neural_replay': True}
