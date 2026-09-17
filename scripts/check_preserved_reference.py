#!/usr/bin/env python3
"""Compare exposed owned-model replies with the pinned upstream full model.

This temporary numerical oracle is outside the shard network. It neither trains
nor settles work, and cannot be used as evidence of distributed ownership.
Only already executed prompt IDs are accepted as controls.
"""
import argparse
import hashlib
import json
from pathlib import Path
import time

import torch
from huggingface_hub import snapshot_download
from safetensors.torch import save as tensor_bytes
from transformers import AutoModelForCausalLM, AutoTokenizer

from neuroshard.evolution import reference
from neuroshard.evolution.reference_data import identity, save


@torch.no_grad()
def generate(model, tokenizer, ids, maximum):
    current=torch.tensor([ids],dtype=torch.long,device='cuda')
    output,past=[],None
    for step in range(maximum):
        with reference.autocast('cuda'):
            value=model(input_ids=current,past_key_values=past,use_cache=True,logits_to_keep=1)
        logits=value.logits[0,-1].float()
        if not bool(torch.isfinite(logits).all()):
            raise ValueError('Nonfinite reference output')
        token=int(logits.argmax());output.append(token)
        if token==tokenizer.eos_token_id:break
        past=value.past_key_values
        current=torch.tensor([[token]],dtype=torch.long,device='cuda')
    return output


def run(home):
    request = json.loads((home/'request.json').read_bytes())
    source = request['assets']['source']
    runtime = reference.configure('cuda', 2)
    began = time.monotonic()
    folder = home/'upstream'
    snapshot_download(source['repo'], revision=source['revision'], token=False, local_dir=folder,
        allow_patterns=['config.json', 'model.safetensors', 'model-*.safetensors', 'model.safetensors.index.json'])
    model = AutoModelForCausalLM.from_pretrained(folder, dtype=torch.float32,
        attn_implementation='sdpa', local_files_only=True, trust_remote_code=False).to('cuda').eval()
    model.requires_grad_(False)
    config = model.config.to_dict()
    differences = {key: {'owned':value,'upstream':config.get(key)}
                   for key,value in request['parent_config'].items() if config.get(key)!=value}
    tensors = {name:spec for part in request['assets']['partitions'].values() for name,spec in part['tensors'].items()}
    state = model.state_dict()
    mismatches = []
    for name,spec in tensors.items():
        raw = tensor_bytes({'weight':state[name].detach().cpu().to(torch.bfloat16).contiguous()})
        actual = hashlib.sha256(raw).hexdigest()
        if actual!=spec['sha256']:
            mismatches.append({'tensor':name,'expected':spec['sha256'],'actual':actual})
    save(home/'weights.json',{'tensor_count':len(tensors),'mismatches':mismatches,'configuration_differences':differences})
    tokenizer = AutoTokenizer.from_pretrained(request['seed'],local_files_only=True)
    if request.get('mode') == 'structured':
        from neuroshard.evolution import general_answer
        from neuroshard.evolution.ordinary_quality import correct
        results=[]
        for case in request['cases']:
            prompt=general_answer.messages(case['messages'])
            ids=tokenizer.apply_chat_template(prompt,tokenize=True,add_generation_prompt=True)
            if len(ids)+general_answer.MAX_TOKENS > 1024:
                raise ValueError('Structured answer exceeds the owned context reservation')
            tokens=generate(model,tokenizer,ids,general_answer.MAX_TOKENS)
            raw=tokenizer.decode(tokens,skip_special_tokens=True)
            try:
                answer=general_answer.visible(raw)
                error=None
            except ValueError as failure:
                answer='';error=str(failure)
            response={'text':answer,'answering':{'status':'completed' if error is None else 'needs_clarification',
                'error':error,'text':answer,'answers':[{'question':case['messages'][-1]['content'],'text':answer}]}}
            result={'id':case['id'],'raw':raw,'text':answer,'token_ids':tokens,
                'prompt_tokens':len(ids),'error':error,'correct':correct(case['scoring'],response)}
            results.append(result);save(home/'answers.json',results)
            print(json.dumps({'id':case['id'],'correct':result['correct'],'text':answer}),flush=True)
        save(home/'result.json',{'request':identity(request),'runtime':runtime,'weights_match':not mismatches,
            'configuration_differences':differences,'cases':results,'correct':sum(r['correct'] for r in results),
            'seconds':time.monotonic()-began,'scope':'Exposed development controls, forced general path only.'})
        return
    if request.get('mode') == 'reasoning':
        from neuroshard.evolution.ordinary_quality import correct
        results=[]
        for case in request['cases']:
            messages=case['messages']
            reasoning=[{'role':'system','content':request['reasoning_instruction']},*messages]
            ids=tokenizer.apply_chat_template(reasoning,tokenize=True,add_generation_prompt=True)
            notes=generate(model,tokenizer,ids,128)
            text=tokenizer.decode(notes,skip_special_tokens=True)
            final=[{'role':'system','content':request['final_instruction']},*messages,
                {'role':'assistant','content':text},{'role':'user','content':request['final_request']}]
            ids=tokenizer.apply_chat_template(final,tokenize=True,add_generation_prompt=True)
            tokens=generate(model,tokenizer,ids,64)
            answer=tokenizer.decode(tokens,skip_special_tokens=True)
            response={'text':answer,'answering':{'status':'completed','error':None,'text':answer,
                'answers':[{'question':messages[-1]['content'],'text':answer}]}}
            result={'id':case['id'],'notes':text,'notes_tokens':notes,'text':answer,'token_ids':tokens,
                'correct':correct(case['scoring'],response)}
            results.append(result);save(home/'answers.json',results)
            print(json.dumps({'id':case['id'],'correct':result['correct'],'text':answer}),flush=True)
        save(home/'result.json',{'request':identity(request),'runtime':runtime,'weights_match':not mismatches,
            'configuration_differences':differences,'cases':results,'correct':sum(r['correct'] for r in results),
            'seconds':time.monotonic()-began,'scope':'Exposed development controls, forced general path only.'})
        return
    results = []
    with torch.no_grad():
        for case in request['cases']:
            current = torch.tensor([case['prompt_ids']],dtype=torch.long,device='cuda')
            tokens, past, first = [], None, None
            for step in range(64):
                with reference.autocast('cuda'):
                    value = model(input_ids=current,past_key_values=past,use_cache=True,logits_to_keep=1)
                logits = value.logits[0,-1].float()
                if not bool(torch.isfinite(logits).all()):
                    raise ValueError('Upstream reference produced nonfinite logits')
                if first is None:
                    top = logits.topk(5)
                    first = {'ids':top.indices.cpu().tolist(),'logits':top.values.cpu().tolist()}
                token = int(logits.argmax())
                tokens.append(token)
                if token==tokenizer.eos_token_id: break
                past = value.past_key_values
                current = torch.tensor([[token]],dtype=torch.long,device='cuda')
            result = {'id':case['id'],'prompt_root':identity(case['prompt_ids']),
                'owned_tokens':case['token_ids'],'reference_tokens':tokens,
                'exact':tokens==case['token_ids'],'text':tokenizer.decode(tokens,skip_special_tokens=True),
                'first':first}
            results.append(result)
            save(home/'answers.json',results)
            print(json.dumps({key:value for key,value in result.items() if key not in ('owned_tokens','reference_tokens','first')}),flush=True)
    save(home/'result.json',{'request':identity(request),'runtime':runtime,'weights_match':not mismatches,
        'configuration_differences':differences,'cases':results,'all_tokens_match':all(row['exact'] for row in results),
        'seconds':time.monotonic()-began,'scope':'Temporary single-host numerical oracle; not a network participant.'})


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--home',type=Path,required=True)
    run(parser.parse_args().home)
