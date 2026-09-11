"""Replayable forward graphs for evaluation and greedy paid inference.

Graph validation does not establish correct numerical execution. Every stage,
including the output head, must be checked by an observer under the optimistic
settlement assumption. A referee loads only one declared partition.
"""
import math

from . import schema
from .batches import unpack

FORMAT = 'neuroshard-forward-v1'
GENERATION = 'neuroshard-generation-v1'


def head_partition(model, first):
    # The head uses only the tied embedding and final norm. A dispute need not
    # publish or load unrelated transformer blocks owned by that same worker.
    return {'components':['embed','norm'], 'capacity':first['capacity'],
            'parameters':sum(model['components'][name]['parameters'] for name in ('embed','norm'))}


def record(pipeline, batch):
    store = pipeline.store
    batch_root = store.put_json(batch)
    current, traces = batch_root, []
    for endpoint, partition in zip(pipeline.endpoints, pipeline.partitions):
        request = {'phase':'forward', 'input':current}
        result = endpoint.evaluate(pipeline.session_id, request)
        traces.append(store.put_json({'model_root':pipeline.model_root,
            'partition':partition, 'request':request, 'result':result}))
        current = result['output']
    request = {'phase':'head', 'input':current, 'batch':batch_root}
    result = pipeline.endpoints[0].evaluate(pipeline.session_id, request)
    traces.append(store.put_json({'model_root':pipeline.model_root,
        'partition':head_partition(pipeline.model,pipeline.partitions[0]), 'request':request, 'result':result}))
    value = {'format':FORMAT, 'model_root':pipeline.model_root, 'batch':batch_root,
             'traces':traces, 'result':result}
    key = store.put_json(value)
    validate(store, key)
    return {'record_root':key, **value}


def finite_loss(value):
    if not isinstance(value, str):
        raise ValueError('Loss must use canonical hexadecimal float encoding')
    loss = float.fromhex(value)
    if not math.isfinite(loss) or not 0 <= loss <= 1_000_000 or loss.hex() != value:
        raise ValueError('Invalid finite loss')
    return loss


def validate(store, key):
    value = store.json(schema.root(key))
    if set(value) != {'format','model_root','batch','traces','result'} or value['format'] != FORMAT:
        raise ValueError('Invalid forward record')
    model = schema.model(store.json(value['model_root']))
    ids, _ = unpack(store.json(value['batch']), model['config']['vocab_size'])
    if not isinstance(value['traces'], list) or not 2 <= len(value['traces']) <= 65:
        raise ValueError('Forward graph requires partitions and an output head')
    traces = [store.json(schema.root(key)) for key in value['traces']]
    names = [name for trace in traces[:-1] for name in trace['partition']['components']]
    if len(names) != len(set(names)) or set(names) != set(model['components']):
        raise ValueError('Forward partitions must cover the model exactly once')
    if not {'embed','norm'} <= set(traces[0]['partition']['components']):
        raise ValueError('First forward partition must own the output head')
    order = [name for name in names if name.startswith('block_')]
    if order != sorted((name for name in model['components'] if name.startswith('block_')), key=schema.block_order):
        raise ValueError('Forward layer order mismatch')
    current = value['batch']
    for index, trace in enumerate(traces):
        if set(trace) != {'model_root','partition','request','result'} or trace['model_root'] != value['model_root']:
            raise ValueError('Mixed-model forward graph')
        schema.partition(model, trace['partition'])
        expected = {'phase':'forward', 'input':current}
        if index == len(traces)-1:
            expected.update(phase='head', batch=value['batch'])
            if trace['partition'] != head_partition(model,traces[0]['partition']) or trace['result'] != value['result']:
                raise ValueError('Output head ownership or result mismatch')
        else:
            if set(trace['result']) != {'output'}:
                raise ValueError('Invalid forward output')
            current = schema.root(trace['result']['output'])
        if trace['request'] != expected:
            raise ValueError('Forward dependency substitution')
    result = value['result']
    if set(result) != {'loss_hex','losses_hex','next_ids'}:
        raise ValueError('Invalid output-head result')
    finite_loss(result['loss_hex'])
    if not isinstance(result['losses_hex'], list) or len(result['losses_hex']) != len(ids):
        raise ValueError('One loss is required per document row')
    for loss in result['losses_hex']:
        finite_loss(loss)
    if not isinstance(result['next_ids'], list) or len(result['next_ids']) != len(ids):
        raise ValueError('One next token is required per row')
    for token in result['next_ids']:
        schema.integer(token, 0, model['config']['vocab_size']-1)
    return value


def generation(pipeline, token_ids, max_tokens=8, eos_ids=()):
    schema.integer(max_tokens, 1, 8)
    if not isinstance(token_ids, list) or not 2 <= len(token_ids) <= 192:
        raise ValueError('Prompt requires 2–192 tokens')
    unpack([token_ids], pipeline.model['config']['vocab_size'])
    ids, output, records = list(token_ids), [], []
    for _ in range(max_tokens):
        value = record(pipeline, [ids])
        records.append(value['record_root'])
        token = value['result']['next_ids'][0]
        output.append(token)
        ids.append(token)
        if token in eos_ids:
            break
    value = {'format':GENERATION, 'model_root':pipeline.model_root, 'prompt_ids':token_ids,
             'max_tokens':max_tokens, 'eos_ids':list(eos_ids), 'token_ids':output, 'records':records}
    return {'record_root':pipeline.store.put_json(value), **value}


def validate_generation(store, key):
    value = store.json(schema.root(key))
    if set(value) != {'format','model_root','prompt_ids','max_tokens','eos_ids','token_ids','records'} or value['format'] != GENERATION:
        raise ValueError('Invalid generation record')
    schema.integer(value['max_tokens'], 1, 8)
    model = schema.model(store.json(value['model_root']))
    if not isinstance(value['prompt_ids'], list) or not 2 <= len(value['prompt_ids']) <= 192:
        raise ValueError('Invalid prompt length')
    unpack([value['prompt_ids']], model['config']['vocab_size'])
    if not isinstance(value['eos_ids'], list) or len(value['eos_ids']) > 8 or len(set(value['eos_ids'])) != len(value['eos_ids']):
        raise ValueError('Invalid stop tokens')
    for token in value['eos_ids']:
        schema.integer(token, 0, model['config']['vocab_size']-1)
    if not isinstance(value['token_ids'], list) or not 1 <= len(value['token_ids']) <= value['max_tokens']:
        raise ValueError('Invalid generation length')
    if not isinstance(value['records'], list) or len(value['records']) != len(value['token_ids']):
        raise ValueError('Missing autoregressive records')
    ids = list(value['prompt_ids'])
    for index, (key, token) in enumerate(zip(value['records'], value['token_ids'])):
        schema.integer(token, 0, model['config']['vocab_size']-1)
        record = validate(store, key)
        if record['model_root'] != value['model_root'] or store.json(record['batch']) != [ids] or record['result']['next_ids'] != [token]:
            raise ValueError('Autoregressive generation dependency mismatch')
        if token in value['eos_ids'] and index != len(value['token_ids'])-1:
            raise ValueError('Generation continues after a stop token')
        ids.append(token)
    if len(value['token_ids']) < value['max_tokens'] and value['token_ids'][-1] not in value['eos_ids']:
        raise ValueError('Generation stopped before its declared bound')
    return value


def trace_roots(store, key):
    value = store.json(key)
    if value.get('format') == GENERATION:
        return [trace for key in value['records'] for trace in store.json(key)['traces']]
    return value['traces']


def bundle(store, key):
    value = store.json(key)
    keys = {key, value['model_root']}
    records = value['records'] if value.get('format') == GENERATION else [key]
    for record_root in records:
        record = store.json(record_root)
        keys.update((record_root, record['batch'], *record['traces']))
    return {key:store.json(key) for key in keys}


def dependencies(store, trace_root):
    trace = store.json(trace_root)
    model = store.json(trace['model_root'])
    needed = {model['components'][name]['root'] for name in trace['partition']['components']}
    if trace['request']['phase'] == 'head' or 'embed' not in trace['partition']['components']:
        needed.add(trace['request']['input'])
    return sorted(needed)


def audit(store, metadata, key, stage):
    from safetensors import SafetensorError
    from .worker import Session, evaluate_session
    traces = trace_roots(metadata, key)
    schema.integer(stage, 0, len(traces)-1)
    trace = metadata.json(traces[stage])
    for key, value in metadata.values.items():
        if store.put_json(value) != key:
            raise ValueError('Forward metadata changed during replay')
    try:
        session = Session(store, trace['model_root'], trace['partition'])
        result = evaluate_session(session, store, trace['request'])
        valid = result == trace['result']
        return {'valid':valid, 'mismatch':'' if valid else 'forward evaluation result'}
    except (ValueError, KeyError, TypeError, SafetensorError) as exc:
        return {'valid':False, 'mismatch':'invalid forward input: '+str(exc)}
