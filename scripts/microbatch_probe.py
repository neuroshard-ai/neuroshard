"""Separate throughput diagnostic; never selects or scores a serving candidate."""
import argparse
import gc
import json
import statistics
import time
from pathlib import Path
import torch
import torch.nn.functional as F
from transformers import LlamaConfig, LlamaForCausalLM
from neuroshard.evolution import reference as engine
from neuroshard.evolution import reference_data as data


def summed_loss(model, rows, device):
    length=max(len(row['input_ids']) for row in rows)
    tokens=torch.zeros((len(rows),length),dtype=torch.long,device=device)
    labels=torch.full_like(tokens,-100)
    mask=torch.zeros_like(tokens)
    weights=torch.tensor([row.get('loss_weight',1) for row in rows],dtype=torch.float32,device=device)
    for index,row in enumerate(rows):
        size=len(row['input_ids'])
        tokens[index,:size]=torch.tensor(row['input_ids'],device=device)
        labels[index,:size]=torch.tensor(row['labels'],device=device)
        mask[index,:size]=1
    with engine.autocast(device):
        logits=model(input_ids=tokens,attention_mask=mask,use_cache=False).logits[:,:-1]
        loss=F.cross_entropy(logits.float().reshape(-1,logits.shape[-1]),labels[:,1:].reshape(-1),
                             ignore_index=-100,reduction='none').reshape(len(rows),-1)
    return (loss*weights[:,None]).sum()


def cpu_equivalence():
    torch.set_num_threads(1)
    torch.manual_seed(17)
    config=LlamaConfig(vocab_size=64,hidden_size=32,intermediate_size=64,num_hidden_layers=2,
                       num_attention_heads=4,num_key_value_heads=2,tie_word_embeddings=True,attention_dropout=0.)
    model=LlamaForCausalLM(config).float()
    rows=[]
    for index,length in enumerate((5,13,9,17)):
        tokens=torch.randint(1,64,(length,)).tolist();labels=[-100]*3+tokens[3:]
        rows.append({'input_ids':tokens,'labels':labels,'targets':length-3,'loss_weight':index+1})
    losses=[];gradients=[]
    for batch in (1,2,4):
        model.zero_grad(set_to_none=True);total=0.
        for start in range(0,len(rows),batch):
            subset=rows[start:start+batch]
            loss=(engine.response_loss(model,subset[0],'cpu')*subset[0]['loss_weight']
                  if batch==1 else summed_loss(model,subset,'cpu'))
            total+=float(loss.detach());loss.backward()
        losses.append(total);gradients.append(torch.cat([p.grad.flatten() for p in model.parameters()]))
    for loss,gradient in zip(losses[1:],gradients[1:]):
        torch.testing.assert_close(torch.tensor(loss),torch.tensor(losses[0]),rtol=1e-6,atol=1e-4)
        torch.testing.assert_close(gradient,gradients[0],rtol=2e-4,atol=1e-5)
    return {'weighted_response_loss_and_gradients_match':True,'unequal_lengths':True,
            'microbatches':[1,2,4],'losses':losses,'dtype':'float32','rtol':2e-4,'atol':1e-5}


def gpu_probe(args):
    runtime=engine.configure('cuda',2)
    prepared=json.loads((args.home/'prepared.json').read_bytes())
    plan=prepared['plan'];recipe=plan['training']
    rows=data.read_records(args.home/'inputs/train.jsonl',prepared['roles']['train']['sha256'])
    schedule=engine.schedule(len(rows),recipe['steps'],recipe['batch_documents'],recipe['seed'])[:16]
    output={'runtime':runtime,'prepared':data.identity(prepared),'steps':16,'warmup_steps_excluded':4,
            'scope':'Single-GPU microbatch timing diagnostic. No held-out scoring, no candidate selection. Full quality comparison still required.', 'arms':{}}
    for batch in (1,2,4):
        torch.manual_seed(recipe['seed']);torch.cuda.reset_peak_memory_stats()
        model=engine.load_model(args.model_dir,'cuda',plan['model']['parameters'])
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False});model.train()
        optimizer=engine.optimizer_for(model,recipe);journal=[]
        try:
            for index,indices in enumerate(schedule):
                examples=sorted([rows[i] for i in indices],key=lambda row:len(row['input_ids']))
                denominator=sum(row['targets']*row.get('loss_weight',1) for row in examples)
                for group in optimizer.param_groups:group['lr']=engine.learning_rate(recipe,index)
                optimizer.zero_grad(set_to_none=True);total=0.;start=time.monotonic()
                for offset in range(0,len(examples),batch):
                    subset=examples[offset:offset+batch]
                    loss=(engine.response_loss(model,subset[0],'cuda')*subset[0].get('loss_weight',1)
                          if batch==1 else summed_loss(model,subset,'cuda'))
                    total+=float(loss.detach());(loss/denominator).backward()
                norm=torch.nn.utils.clip_grad_norm_(model.parameters(),recipe['clip_norm'],error_if_nonfinite=True)
                optimizer.step();torch.cuda.synchronize()
                journal.append({'step':index+1,'seconds':time.monotonic()-start,'loss':total/denominator,'gradient_norm':float(norm)})
            output['arms'][str(batch)]={'steps':journal,'median_seconds':statistics.median(r['seconds'] for r in journal[4:]),'peak_cuda_bytes':torch.cuda.max_memory_allocated()}
        except torch.cuda.OutOfMemoryError:
            output['arms'][str(batch)]={'failed':'cuda_out_of_memory','completed_steps':journal}
        data.save(args.output,output)
        del model,optimizer,loss;gc.collect();torch.cuda.empty_cache()
    print(json.dumps({key:{k:v for k,v in arm.items() if k!='steps'} for key,arm in output['arms'].items()}),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--cpu-only',action='store_true')
    parser.add_argument('--home',type=Path)
    parser.add_argument('--model-dir',type=Path)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if args.cpu_only:
        value=cpu_equivalence();data.save(args.output,value);print(json.dumps(value))
    else:
        gpu_probe(args)
