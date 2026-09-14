"""Reconstruct paired outcomes and resource ratios without the training engine."""
import argparse
import json
import math
from pathlib import Path
import statistics


def load(path):
    return json.loads(path.read_text())

def interval(values):
    mean=statistics.mean(values)
    error=statistics.stdev(values)/math.sqrt(len(values))
    radius=PLAN['decision']['normal_multiplier']*error
    return {'n':len(values),'mean':mean,'lower':mean-radius,'upper':mean+radius}

def paired(before,after):
    assert [r['id'] for r in before]==[r['id'] for r in after]
    assert len(set(r['id'] for r in before))==len(before)
    return interval([int(b['correct'])-int(a['correct']) for a,b in zip(before,after)])

def retention(before,after):
    assert [r['id'] for r in before]==[r['id'] for r in after]
    return interval([b['loss']-a['loss'] for a,b in zip(before,after)])

def traffic(results):
    return sum(result['network_end'][name]['tx']-value['tx'] for result in results
               for name,value in result['network_start'].items())

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--home',type=Path,required=True,help='Prepared inputs, rank results and evaluations')
    parser.add_argument('--receipts',type=Path,required=True,help='Directory containing arm-processes.json receipts')
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    HOME,ROOT=args.home,args.receipts
    PLAN=load(HOME/'prepared.json')['plan']
    evaluations={arm:load(HOME/'evaluation'/f'{arm}.json') for arm in ['seed',*PLAN['arms']]}
    ranks={arm:[load(HOME/arm/f'rank-{rank}/result.json') for rank in range(world)]
           for arm,world in PLAN['arms'].items()}
    seed=evaluations['seed']
    result={'quality':{},'training':{},'scope':'One development seed, public generated task families, exposed retention; no audit cost included.'}
    for arm,evaluation in evaluations.items():
        cases=evaluation['checks']
        assert len(cases)==PLAN['test_documents']
        by_family={family:sum(r['correct'] for r in cases if r['family']==family)
                   for family in ['lookup','filter','total','sort']}
        quality={'correct':sum(r['correct'] for r in cases),'count':len(cases),'by_family':by_family}
        if arm!='seed':
            quality['against_seed']=paired(seed['checks'],cases)
            quality['retention']=retention(seed['retention'],evaluation['retention'])
            quality['against_single']=paired(evaluations['single']['checks'],cases)
            quality['against_dense']=paired(evaluations['dense-pair']['checks'],cases)
        result['quality'][arm]=quality
    for arm,values in ranks.items():
        active=max(v['active_seconds'] for v in values)
        complete=max(v['active_seconds']+v['checkpoint_seconds'] for v in values)
        process=load(ROOT/(arm+'-processes.json'))
        result['training'][arm]={'active_seconds':active,'active_gpu_seconds':active*len(values),
            'through_checkpoint_seconds':complete,'allocated_gpu_seconds_through_checkpoint':complete*len(values),
            'full_process_seconds':process['seconds'],'allocated_gpu_seconds_full_process':process['seconds']*len(values),
            'tx_bytes':traffic(values),'peak_gpu_bytes':max(v['peak_cuda_bytes'] for v in values)}
    c=result['quality']['compressed-pair']
    threshold=PLAN['decision']
    q=(c['against_seed']['mean']>=threshold['task_gain_minimum']
       and c['retention']['upper']<=threshold['retention_upper_nats']
       and c['against_single']['lower']>=-threshold['accuracy_noninferiority_margin']
       and c['against_dense']['lower']>=-threshold['accuracy_noninferiority_margin'])
    t=result['training']; speed=t['single']['active_seconds']/t['compressed-pair']['active_seconds']
    cost=t['compressed-pair']['allocated_gpu_seconds_full_process']/t['single']['allocated_gpu_seconds_full_process']
    result['decision']={'quality_screen_pass':q,'active_speedup_vs_single':speed,
        'active_speed_screen_pass':speed>=threshold['active_speedup_vs_single_minimum'],
        'allocated_gpu_process_cost_ratio_vs_single':cost,'gpu_cost_reduction_vs_single':cost<1,
        'wire_ratio_vs_dense':t['compressed-pair']['tx_bytes']/t['dense-pair']['tx_bytes'],
        'permissionless_economics_proven':False,'serving_approved':False}
    args.output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))
