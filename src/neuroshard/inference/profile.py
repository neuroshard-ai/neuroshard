"""Freeze all model files, tokenizer, data, code, and conformance commitments."""
import hashlib
import json
import os
from importlib.metadata import version
from pathlib import Path

from neuroshard.dataflow.store import canonical,digest
from neuroshard.demo import work
from neuroshard.lab import state
from neuroshard.lab.app import native_parameters
from .engine import Engine

MODEL_REPO='HuggingFaceTB/SmolLM2-135M-Instruct'
MODEL_REVISION='12fd25f77366fa6b3b4b768ec3050bf629380bac'
MODEL_FILES=('config.json','generation_config.json','model.safetensors','tokenizer.json',
             'tokenizer_config.json','special_tokens_map.json','vocab.json','merges.txt')


def model_profile(directory):
    files={}
    for name in MODEL_FILES:
        h=hashlib.sha256()
        with (Path(directory)/name).open('rb') as f:
            for chunk in iter(lambda:f.read(1024**2),b''):h.update(chunk)
        files[name]=h.hexdigest()
    config=json.loads((Path(directory)/'config.json').read_text())
    return {'repo':MODEL_REPO,'revision':MODEL_REVISION,'license':'Apache-2.0','files':files,
        'hidden_size':config['hidden_size'],'adapter_rank':4,'learning_rate':0.02,
        'max_input_tokens':256,'max_new_tokens':64,'training_sequence_tokens':64,
        'arithmetic':'cpu-float32-single-thread-eager-SSE4_2','adapter':'post-norm-residual-v1'}


def source_hash():
    import neuroshard.lab, neuroshard.demo
    from neuroshard.core.crypto import ecdsa
    from neuroshard.dataflow import store
    paths=[*[Path(__file__).parent/name for name in ('app.py','engine.py','state.py','profile.py')],
           Path(store.__file__),*Path(neuroshard.lab.__file__).parent.glob('*.py'),
           *Path(neuroshard.demo.__file__).parent.glob('*.py'),Path(ecdsa.__file__)]
    root=Path(__file__).resolve().parents[1]
    return digest(canonical({str(p.resolve().relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest()
                             for p in sorted(paths)}))


def batch(data,step):
    sequences=data['train']
    if not sequences:raise ValueError('Training data is empty')
    index=int(hashlib.sha256(f'neuroshard/adapter-batch/v1:{step}'.encode()).hexdigest(),16)%len(sequences)
    return sequences[index]


def build(directory,data):
    model=model_profile(directory);engine=Engine(directory,model)
    weights=engine.initial();initial_root=work.digest(weights)
    validation=engine.evaluate(weights,data['validation'])
    vectors=[]
    for step in range(3):
        result=engine.train(weights,batch(data,step));weights=result['weights']
        vectors.append({k:result[k] for k in ('feature_root','gradient_root','loss_hex')})
        vectors[-1]['model_root']=work.digest(weights)
    output=engine.infer(weights,{'prompt':'What is the capital of France?','max_tokens':16})
    params={**state.PARAMS,'epoch_blocks':60,'activation_blocks':60,'lease_blocks':240,
        'evidence_blocks':172800,'evidence_seconds':172800,'max_training_tasks':10000,
        'inference_token_price':1000,'inference_blocks':240}
    spec={'version':'neuroshard-llm-v1','profile':'llm-testnet','model':model,
        'params':params,'native_consensus':native_parameters(params),'dataset_sha256':digest(canonical(data)),
        'dataset_root':data['dataset_root'],'initial_model_root':initial_root,'initial_validation_loss_hex':validation.hex(),
        'libraries':{p:version(p) for p in ('torch','numpy','transformers','tokenizers','safetensors')},
        'source_hash':source_hash(),'numerical_conformance':{'three_steps':vectors,'inference_root':work.digest(output)}}
    return spec


def check(spec,directory,data):
    if spec['source_hash'] != source_hash():raise ValueError('Execution source differs from genesis')
    actual=build(directory,data)
    if canonical(actual) != canonical(spec):raise ValueError('Model, data, libraries, or arithmetic differs from genesis')
    return actual
