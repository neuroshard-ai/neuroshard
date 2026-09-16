"""Test whether retained DDP gradient buffers permit resident PowerSGD errors.

Uses only the training partition. Compares complete numerical state after 16
updates; no evaluation cases or quality decisions are changed.
"""
import argparse
import gc
import hashlib
import json
import os
from pathlib import Path
import sys
import time
sys.path.insert(0,'scripts')
import study_learning_methods as driver
from neuroshard.evolution import cooperative as group
from neuroshard.evolution import gradient_compression as hooks
from neuroshard.evolution import local_windows as windows
from neuroshard.evolution import reference as engine
from neuroshard.evolution import reference_data as data


def run(args):
    import torch
    import torch.distributed as dist
    from torch.nn.parallel import DistributedDataParallel as DDP
    from torch.distributed.algorithms.ddp_comm_hooks.powerSGD_hook import PowerSGDState
    from datetime import timedelta
    plan=json.loads(driver.PLAN.read_bytes());prepared=driver.inputs(args,plan)
    profile=driver.runtime();rank=int(os.environ['RANK']);world=int(os.environ['WORLD_SIZE'])
    assert world==2 and rank in (0,1)
    assert driver.model_snapshot(args.model_dir,plan['model'])==prepared['model_snapshot']
    dist.init_process_group('nccl',timeout=timedelta(seconds=180))
    contract={'prepared':data.identity(prepared),'probe_source':data.sha256(Path(__file__)),
              'steps':16,'dense_graph_required':True,'modes':['offloaded-clear','resident-retain'],'runtime':group.runtime_profile(profile)}
    group.agree_digest(data.identity(contract),world,'cuda')
    output=args.home/f'buffer-probe-rank-{rank}.json'
    assert not output.exists()
    result={'contract':contract,'rank':rank,'modes':{},'scope':'Memory/transport equivalence probe. No held-out scoring or serving selection.'}
    rows=driver.records(args,prepared,'train');recipe=plan['training']
    schedule=engine.schedule(len(rows),recipe['steps'],recipe['batch_documents'],recipe['seed'])[:16]
    try:
        for mode in contract['modes']:
            torch.manual_seed(recipe['seed']);torch.cuda.reset_peak_memory_stats()
            model=engine.load_model(args.model_dir,'cuda',plan['model']['parameters'])
            model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False});model.train()
            optimizer=engine.optimizer_for(model,recipe)
            wrapped=DDP(model,device_ids=[0],broadcast_buffers=False,gradient_as_bucket_view=True,bucket_cap_mb=64)
            state=PowerSGDState(process_group=dist.group.WORLD,**plan['compression'])
            wrapped.register_comm_hook(state,hooks.offloaded_power_sgd if mode=='offloaded-clear' else hooks.resident_power_sgd)
            if mode=='resident-retain':
                original_zero_grad=optimizer.zero_grad
                optimizer.zero_grad=lambda *a,**k:original_zero_grad(set_to_none=False)
            journal=[];started=time.monotonic()
            for index,indices in enumerate(schedule):
                row=group.step(wrapped,optimizer,[rows[i] for i in indices],'cuda',recipe,index,rank,world)
                if any(parameter.grad is None for parameter in model.parameters()):
                    raise ValueError('Retained zero gradients require every parameter to participate; unused-parameter optimizer semantics differ')
                journal.append(row)
                print(json.dumps({'mode':mode,'rank':rank,'step':index+1,'seconds':row['seconds']}),flush=True)
            elapsed=time.monotonic()-started
            optimizer_state=optimizer.state_dict()
            optimizer_tensors=((str(index)+'/'+name,value) for index,entry in optimizer_state['state'].items()
                               for name,value in entry.items() if isinstance(value,torch.Tensor))
            rng=state.rng.get_state()
            digests={'parameters':group.parameter_digest(model),'optimizer':windows.state_digest(optimizer_tensors),
                     'optimizer_groups':data.identity(optimizer_state['param_groups']),
                     'errors':windows.state_digest((str(k),v) for k,v in sorted(state.error_dict.items())),
                     'p':windows.state_digest((str(k),v) for k,v in sorted(state.p_memory_dict.items())),
                     'q':windows.state_digest((str(k),v) for k,v in sorted(state.q_memory_dict.items())),
                     'projection_rng':data.identity([rng[0],rng[1].tolist(),*rng[2:]]),
                     'torch_cpu_rng':windows.state_digest([('rng',torch.get_rng_state())]),
                     'torch_cuda_rng':windows.state_digest((str(i),v) for i,v in enumerate(torch.cuda.get_rng_state_all())),
                     'trajectory':data.identity([{k:v for k,v in row.items() if k!='seconds'} for row in journal])}
            group.agree_digest(digests['parameters'],world,'cuda')
            result['modes'][mode]={'digests':digests,'journal':journal,'active_seconds':elapsed,
                                   'peak_cuda_bytes':torch.cuda.max_memory_allocated(),
                                   'final_allocated_cuda_bytes':torch.cuda.memory_allocated()}
            data.save(output,result)
            # The bound zero_grad method otherwise retains the entire optimizer.
            if mode=='resident-retain':
                del original_zero_grad
            del optimizer_state,optimizer_tensors,wrapped,optimizer,state,model
            gc.collect();torch.cuda.empty_cache();dist.barrier()
        first=result['modes']['offloaded-clear']['digests'];second=result['modes']['resident-retain']['digests']
        result['equal_fields']={key:first[key]==second[key] for key in first}
        result['complete_numerical_equality']=all(result['equal_fields'].values())
        data.save(output,result)
        print(json.dumps({'rank':rank,'complete_numerical_equality':result['complete_numerical_equality']}),flush=True)
    finally:
        dist.destroy_process_group()


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--home',type=Path,required=True)
    parser.add_argument('--model-dir',type=Path,required=True);args=parser.parse_args()
    with windows.exclusive_device('cuda'):run(args)
