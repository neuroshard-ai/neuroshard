"""Compact execution-graph checks and bounded, objective replay dependencies."""
import copy
import math

from neuroshard.dataflow.store import canonical
from .objects import digest
from . import schema

def validate_record(store, root):
    """Check all cross-shard dependencies and optimizer scaling without neural replay."""
    record = store.json(root)
    parent,candidate = store.json(record['parent']),store.json(record['model_root'])
    schema.model(parent)
    schema.model(candidate)
    from .batches import unpack
    unpack(store.json(record['batch']),parent['config']['vocab_size'])
    traces = [store.json(r) for r in record['traces']]
    if not traces or len(traces)>64:
        raise ValueError('Invalid trace count')
    names = [n for t in traces for n in t['partition']['components']]
    if len(names)!=len(set(names)) or set(names)!=set(parent['components']):
        raise ValueError('Shards do not cover the parent model exactly once')
    if traces[0]['phase']!='begin' or any(t['phase']!='forward' for t in traces[1:]):
        raise ValueError('Invalid execution graph')
    if not {'embed', 'norm'} <= set(traces[0]['partition']['components']) or traces[0]['gradient_in'] is not None:
        raise ValueError('First stage must own the tied embedding and output head')
    order = [n for t in traces for n in t['partition']['components'] if n.startswith('block_')]
    if order != sorted((n for n in parent['components'] if n.startswith('block_')),key=schema.block_order):
        raise ValueError('Layer order differs from the model')
    if traces[0]['input']!=record['batch'] or traces[-1]['output']!=traces[0]['head_input']:
        raise ValueError('Batch or output-head dependency mismatch')
    expected_norms = [t['norm_squared_hex'] for t in traces]
    if expected_norms != record['norms']:
        raise ValueError('Clipping inputs differ from gradient commitments')
    norms = [float.fromhex(v) for v in expected_norms]
    if any(not math.isfinite(v) or v<0 for v in norms):
        raise ValueError('Invalid norm')
    clip = float.fromhex(record['clip_norm_hex'])
    if not 0 < clip <= 10:
        raise ValueError('Invalid clipping bound')
    scale = min(1.0,clip/(math.sqrt(math.fsum(norms))+1e-6)).hex()
    expected = copy.deepcopy(parent)
    schema.integer(record['step'], 0, 2**53-1)
    rate = float.fromhex(record['learning_rate_hex'])
    if not math.isfinite(rate) or not 0 < rate <= .1:
        raise ValueError('Invalid learning rate')
    expected['parent'] = record['parent']
    for i,t in enumerate(traces):
        schema.partition(parent, t['partition'])
        from .update_witness import validate_trace
        validate_trace(parent, t)
        if t['parent']!=record['parent'] or t['step']!=record['step']:
            raise ValueError('Mixed-parent or mixed-step trace')
        if t['scale_hex']!=scale or t['learning_rate_hex']!=record['learning_rate_hex']:
            raise ValueError('Incorrect global optimizer scaling')
        if i and t['input']!=traces[i-1]['output']:
            raise ValueError('Forward activation substitution')
        expected_gradient = traces[0]['head_gradient'] if i==len(traces)-1 else traces[i+1]['gradient_in']
        if t['gradient_out']!=expected_gradient:
            raise ValueError('Backward gradient substitution')
        if set(t['components'])!=set(t['partition']['components']):
            raise ValueError('Update changed component ownership')
        if t['partition']['parameters']!=sum(parent['components'][n]['parameters'] for n in t['partition']['components']):
            raise ValueError('Incorrect resident size')
        for name,c in t['components'].items():
            if c['parameters']!=parent['components'][name]['parameters']:
                raise ValueError('Training cannot change architecture')
        expected['components'].update(t['components'])
    if expected!=candidate or record['scale_hex']!=scale or traces[0]['loss_hex']!=record['loss_hex']:
        raise ValueError('Candidate differs from the committed execution graph')
    return {'valid':True,'stages':len(traces),'model_root':record['model_root']}


class Metadata:
    def __init__(self, values, maximum_bytes=512*1024):
        if not isinstance(values,dict) or len(canonical(values)) > maximum_bytes:
            raise ValueError('Metadata bundle exceeds bounds')
        for key,value in values.items():
            schema.root(key)
            if digest(canonical(value)) != key:
                raise ValueError('Metadata hash mismatch')
        self.values = values

    def json(self,key):
        return self.values[key]


def bundle(store,record_root):
    record = store.json(record_root)
    keys = [record_root,record['parent'],record['model_root'],record['batch'],*record['traces']]
    return {key:store.json(key) for key in keys}


def work_identity(store,record_root):
    """Identify the paid numerical task independently of ancestry or job nonces.

    Repartitioning the same weights, batch and optimizer is not another payable
    task. This also handles an exactly converged model whose weight bytes stop
    changing even as model-manifest ancestry continues to grow.
    """
    record=store.json(record_root)
    parent=store.json(record['parent'])
    config=copy.deepcopy(parent['config'])
    for name in ('rms_norm_eps','rope_theta'):
        config[name]=float(config[name]).hex()
    from .batches import unpack
    ids,labels=unpack(store.json(record['batch']),parent['config']['vocab_size'])
    targets=ids if labels is None else labels
    return digest(canonical({'domain':'neuroshard/evolution/paid-task/v1',
        'config':config,'components':{name:c['root'] for name,c in parent['components'].items()},
        'input_ids':ids,'targets':[row[1:] for row in targets],'learning_rate_hex':record['learning_rate_hex'],
        'clip_norm_hex':record['clip_norm_hex']}))


def dependencies(metadata,trace_root):
    trace = metadata.json(trace_root)
    parent = metadata.json(trace['parent'])
    schema.partition(parent,trace['partition'])
    needed = {parent['components'][name]['root'] for name in trace['partition']['components']}
    needed.add(trace['gradient_out'])
    if trace['phase']=='begin':
        needed.add(trace['head_input'])
    else:
        needed.add(trace['input'])
    return sorted(needed)


def validate_growth(metadata,parent_root,new_root):
    parent,candidate = metadata.json(parent_root),metadata.json(new_root)
    schema.model(parent)
    schema.model(candidate)
    added = candidate['config']['num_hidden_layers']-parent['config']['num_hidden_layers']
    schema.integer(added,1,16)
    expected = copy.deepcopy(parent)
    expected['parent'] = parent_root
    expected['growth'] = {'kind':'identity-residual-depth','added_layers':added}
    expected['config']['num_hidden_layers'] += added
    old_depth = parent['config']['num_hidden_layers']
    new_component = candidate['components'][f'block_{old_depth:03}']
    for i in range(old_depth,old_depth+added):
        expected['components'][f'block_{i:03}'] = new_component
    expected['parameters'] += added*new_component['parameters']
    if candidate != expected:
        raise ValueError('Growth changed existing weights or unapproved architecture fields')
    return {'valid':True,'added_layers':added,'parameters':candidate['parameters']}


def audit_growth(store,metadata,parent_root,new_root):
    checked=validate_growth(metadata,parent_root,new_root)
    from .model import grow
    for key,value in metadata.values.items():
        if store.put_json(value)!=key:
            raise ValueError('Growth metadata changed')
    expected,_=grow(parent_root,store,checked['added_layers'])
    return {'valid':expected==new_root,'expected_root':expected,'claimed_root':new_root,
            'required_parent_component':metadata.json(parent_root)['components'][f'block_{metadata.json(parent_root)["config"]["num_hidden_layers"]-1:03}']['root']}


def audit(store,metadata,record_root,stage):
    """Referee receives only artifacts already made available by the protocol.

    Off-chain observers may call this directly. An ABCI caller must first
    establish availability from finalized upload transactions on every node.
    """
    from .worker import replay_trace
    record = metadata.json(record_root)
    if record.get('format') in ('neuroshard-forward-v1','neuroshard-generation-v1'):
        from .forward import audit as audit_forward
        return audit_forward(store,metadata,record_root,stage)
    if record.get('kind')=='growth':
        schema.integer(stage,0,0)
        from safetensors import SafetensorError
        try:
            result = audit_growth(store,metadata,record['parent'],record['model_root'])
            return {**result,'mismatch':'' if result['valid'] else 'identity growth'}
        except (ValueError,KeyError,TypeError,SafetensorError) as exc:
            return {'valid':False,'mismatch':'invalid growth input: '+str(exc)}
    schema.integer(stage,0,len(record['traces'])-1)
    for key,value in metadata.values.items():
        if store.put_json(value) != key:
            raise ValueError('Metadata changed while preparing replay')
    from safetensors import SafetensorError
    try:
        return replay_trace(store,record['traces'][stage])
    except (ValueError,KeyError,TypeError,SafetensorError) as exc:
        return {'valid':False,'mismatch':'invalid committed execution input: '+str(exc)}
