#!/usr/bin/env python3
"""Compare real full-model training and compact SGD fraud witnesses.

This bounded benchmark repeats the same numerical task to measure overhead;
repetitions are not fresh training or separately payable work. It changes no
public network. --verify checks the portable witnesses without model files.
"""
import argparse
import copy
import gc
import json
import statistics
import time
from pathlib import Path

from neuroshard.dataflow.store import canonical
from neuroshard.evolution import update_witness as witness
from neuroshard.evolution.app import code_hash
from neuroshard.evolution.model import place
from neuroshard.evolution.objects import Objects, digest
from neuroshard.evolution.pipeline import Pipeline, LocalEndpoint
from neuroshard.evolution.runtime import check as runtime_check
from neuroshard.evolution.verification import Metadata, bundle, audit, dependencies, validate_record, work_identity
from neuroshard.evolution.worker import Worker


def numerical_probe():
    """Compare whole-tensor and chunk arithmetic, including tails and subnormals."""
    import numpy as np
    from neuroshard.evolution.model import torch
    rng = np.random.default_rng(20260911)
    cases = []
    for length in (31,1024,1025,2049,65537):
        bits = rng.integers(0,2**32,size=(2,length),dtype=np.uint32)
        # Bound exponents to finite magnitudes while retaining signs,
        # mantissas, signed zero and explicitly inserted subnormal values.
        bits &= np.uint32(0xf7ffffff)
        bits[:,:8] = [0,0x80000000,1,0x80000001,0x007fffff,0x807fffff,0x3f800000,0xbf800000]
        before,gradient = [torch.from_numpy(row.view(np.float32).copy()) for row in bits]
        for rate in (.1,.003,2.**-80):
            for scale in (1.,.123,2.**-40):
                whole,chunked = before.clone(),before.clone()
                whole.add_(gradient,alpha=-rate*scale)
                for start in range(0,length,witness.CHUNK_ELEMENTS):
                    chunked[start:start+1024].add_(gradient[start:start+1024],alpha=-rate*scale)
                raw = whole.numpy().astype('<f4',copy=False).tobytes()
                if raw != chunked.numpy().astype('<f4',copy=False).tobytes():
                    raise ValueError('Chunk arithmetic differs from whole-tensor SGD')
                cases.append({'elements':length,'rate':rate.hex(),'scale':scale.hex(),'sha256':digest(raw)})
    return {'cases':len(cases),'whole_equals_chunked':True,'output_digest':digest(canonical(cases)),
            'scope':'fixed finite float32 stress vectors, tails, signed zeros and subnormals; not exhaustive hardware coverage'}


def forged_update(store, record):
    """Change one embedding element and faithfully commit the wrong output."""
    value = store.json(record['record_root'])
    trace = store.json(value['traces'][0])
    index = next(i for i,item in enumerate(trace['updates']['tensors'])
                 if (item['component'],item['tensor']) == ('embed','weight'))
    changed = store.tensors(trace['components']['embed']['root'])
    changed['weight'].view(-1)[0] += 1
    trace['components']['embed']['root'] = store.put_tensors(changed)
    trace['updates']['tensors'][index]['after'] = witness.commit(changed['weight'])
    value['traces'][0] = store.put_json(trace)
    model = store.json(value['model_root'])
    model['components'].update(trace['components'])
    value['model_root'] = store.put_json(model)
    value['record_root'] = store.put_json(value)
    return value, index


def portable_case(store, record, index, expected):
    started = time.perf_counter()
    proof = witness.prepare(store, record['traces'][0], index, 0)
    preparation_seconds = time.perf_counter()-started
    metadata = bundle(store, record['record_root'])
    result = {'record_root':record['record_root'],'metadata':metadata,'stage':0,
              'tensor_index':index,'witness':proof,'expected_valid':expected,
              'preparation_seconds':preparation_seconds,'witness_bytes':len(canonical(proof))}
    verify_case(result)
    return result


def verify_case(case):
    metadata = Metadata(case['metadata'])
    validate_record(metadata, case['record_root'])
    times = []
    for _ in range(20):
        started = time.perf_counter()
        result = witness.check(metadata,case['record_root'],case['stage'],case['tensor_index'],case['witness'])
        times.append(time.perf_counter()-started)
        if result['valid'] is not case['expected_valid']:
            raise ValueError('Compact witness disagrees with the full-stage oracle')
    return {'verdict':result,'median_check_seconds':statistics.median(times),
            'maximum_check_seconds':max(times),'repetitions':len(times)}


def verify_report(path, expected_sha256, objects=None):
    raw = path.read_bytes()
    if len(raw) > 4*1024*1024 or digest(raw) != expected_sha256:
        raise ValueError('Portable report exceeds bounds or differs from its pinned hash')
    report = json.loads(raw)
    if report['source_hash'] != code_hash():
        raise ValueError('Use the exact source committed by the experiment')
    probe = numerical_probe()
    if probe != report['numerical_probe']:
        raise ValueError('Numerical stress vectors differ from the originating CPU')
    result = {'report_sha256':expected_sha256,'source_hash':code_hash(),'runtime':runtime_check(),'numerical_probe':probe,
              'cases':{name:verify_case(case) for name,case in report['portable_cases'].items()},
              'scope':'portable chunk refutations only; no full-model replay on this verifier'}
    if objects is not None:
        store = Objects(objects)
        result['full_stage_oracles'] = {}
        for name,case in report['portable_cases'].items():
            started = time.perf_counter()
            verdict = audit(store,Metadata(case['metadata']),case['record_root'],case['stage'])
            if verdict['valid'] is not case['expected_valid']:
                raise ValueError('Full-stage replay disagrees with the expected verdict')
            result['full_stage_oracles'][name] = {**verdict,'seconds':time.perf_counter()-started}
        result['scope'] = 'portable witnesses and complete replay of their selected stage; other stages are not replayed by this command'
    return result


def run(args):
    args.home = args.home.resolve()
    args.home.mkdir(parents=True,exist_ok=False)
    store = Objects(args.objects)
    baseline = store.json(args.model_root)
    if 'update_witnesses' in baseline:
        raise ValueError('Start from the unchanged model without an update-witness profile')
    indexed = copy.deepcopy(baseline)
    indexed['update_witnesses'] = witness.FORMAT
    indexed_root = store.put_json(indexed)
    capacities = [48_000_000]*len(place(baseline,[48_000_000]*64))
    if args.workers_config:
        from neuroshard.evolution.transport import Endpoint
        config = json.loads(args.workers_config.read_bytes())
        entries = config['workers'][:len(capacities)]
        if len(entries) != len(capacities):raise ValueError('Missing worker endpoints')
        endpoints = [Endpoint(item['url'],(args.workers_config.resolve().parent/Path(item['token_file']).expanduser()).read_text().strip(),store)
                     for item in entries]
    else:
        endpoints = [LocalEndpoint(Worker(args.home/f'worker{i}',store)) for i in range(len(capacities))]
    batch = json.loads(args.batch.read_bytes())
    from neuroshard.evolution.batches import unpack
    unpack(batch,baseline['config']['vocab_size'])
    roots = {'baseline':args.model_root,'indexed':indexed_root}
    measurements, records = {'baseline':[],'indexed':[]}, {}
    # The first pair includes cold artifact transfer and is reported separately.
    for repetition in range(args.repetitions+1):
        order = ('baseline','indexed') if repetition%2==0 else ('indexed','baseline')
        for name in order:
            sent = sum(getattr(endpoint,'sent',0) for endpoint in endpoints)
            received = sum(getattr(endpoint,'received',0) for endpoint in endpoints)
            started = time.perf_counter()
            pipeline = Pipeline(store,roots[name],endpoints,capacities,
                                f'{args.home.name}-{name}-{repetition}',learning_rate=args.learning_rate)
            try:
                record = pipeline.train(batch)
                metadata = Metadata(bundle(store,record['record_root']))
                validate_record(metadata,record['record_root'])
                # Include retrieving and hashing every resulting component,
                # not just receiving a worker's promise that it exists.
                for component in store.json(record['model_root'])['components'].values():
                    store.get(component['root'])
            finally:
                pipeline.close()
            measurement = {'seconds':time.perf_counter()-started,'cold_pair':repetition==0,
                'object_bytes_sent':sum(getattr(endpoint,'sent',0) for endpoint in endpoints)-sent,
                'object_bytes_received':sum(getattr(endpoint,'received',0) for endpoint in endpoints)-received,
                'metadata_bytes':len(canonical(metadata.values))}
            measurements[name].append(measurement)
            records[name] = record
            print(json.dumps({'phase':'measured_training','variant':name,'repetition':repetition,**measurement}),flush=True)
            gc.collect()
        if (store.json(records['baseline']['model_root'])['components'] != store.json(records['indexed']['model_root'])['components']
                or records['baseline']['loss_hex'] != records['indexed']['loss_hex']):
            raise ValueError('Commitment instrumentation changed numerical training')
        if work_identity(store,records['baseline']['record_root']) != work_identity(store,records['indexed']['record_root']):
            raise ValueError('Adding commitments changed the identity of the payable numerical task')
    audits = {}
    for name,record in records.items():
        values = []
        for stage in range(len(record['traces'])):
            started = time.perf_counter()
            result = audit(store,Metadata(bundle(store,record['record_root'])),record['record_root'],stage)
            if not result['valid']:raise ValueError(result)
            values.append({'stage':stage,'seconds':time.perf_counter()-started})
        audits[name] = values
    forged,index = forged_update(store,records['indexed'])
    metadata = Metadata(bundle(store,forged['record_root']))
    started = time.perf_counter()
    oracle = audit(store,metadata,forged['record_root'],0)
    oracle_seconds = time.perf_counter()-started
    if oracle['valid']:raise ValueError('Full referee accepted the deliberate optimizer corruption')
    needed = dependencies(metadata,forged['traces'][0])
    coarse_bytes = sum(len(store.get(key)) for key in needed)
    cases = {'honest':portable_case(store,records['indexed'],index,True),
             'forged':portable_case(store,forged,index,False)}
    medians = {name:statistics.median(row['seconds'] for row in values[1:]) for name,values in measurements.items()}
    result = {'format':'neuroshard-update-witness-benchmark-v1','source_hash':code_hash(),
        'runtime':runtime_check(),'numerical_probe':numerical_probe(),'parameters':baseline['parameters'],'partitions':len(capacities),
        'worker_transport':'http' if args.workers_config else 'local-single-process',
        'baseline_root':args.model_root,'indexed_root':indexed_root,
        'measurements':measurements,'warm_median_seconds':medians,
        'commitment_training_ratio':medians['indexed']/medians['baseline'],
        'parameter_bytes_and_loss_unchanged':True,'paid_work_identity_unchanged':True,
        'honest_full_audits':audits,'full_forgery_oracle':{**oracle,'seconds':oracle_seconds,'input_bytes':coarse_bytes},
        'portable_cases':cases,'compact_checks':{name:verify_case(case) for name,case in cases.items()},
        'training':records['indexed'],'forged_training':forged,
        'scope':'same-task repetitions for cost measurement, not new learning or issuance; commitments do not prove gradients',
        'traffic_scope':'raw object payloads; excludes HTTP, RPC-envelope and SSH overhead'}
    (args.home/'result.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({key:result[key] for key in ('parameters','warm_median_seconds','commitment_training_ratio','full_forgery_oracle','compact_checks')},indent=2))
    return result


if __name__=='__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--home',type=Path)
    parser.add_argument('--objects',type=Path)
    parser.add_argument('--model-root')
    parser.add_argument('--batch',type=Path)
    parser.add_argument('--workers-config',type=Path)
    parser.add_argument('--repetitions',type=int,default=3)
    parser.add_argument('--learning-rate',type=float,default=.003)
    parser.add_argument('--verify',type=Path)
    parser.add_argument('--expected-sha256')
    args = parser.parse_args()
    if args.verify:
        if not args.expected_sha256:parser.error('--verify requires --expected-sha256')
        print(json.dumps(verify_report(args.verify,args.expected_sha256,args.objects),indent=2))
    else:
        if not all((args.home,args.objects,args.model_root,args.batch)) or not 1<=args.repetitions<=5:
            parser.error('Supply --home, --objects, --model-root, --batch and 1–5 repetitions')
        run(args)
