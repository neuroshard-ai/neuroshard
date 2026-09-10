"""CPU replay oracle for a frozen SmolLM2 backbone and a compact trained adapter.

All files, tokenizer behavior, arithmetic, and batch selection belong to a frozen
execution profile. This is deliberately expensive full replay, not a succinct proof.
"""
import base64
import hashlib
import json
import os
from pathlib import Path

from neuroshard.dataflow.store import canonical, digest


def configure():
    os.environ.update(ATEN_CPU_CAPABILITY="default", MKL_ENABLE_INSTRUCTIONS="SSE4_2",
        OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", TOKENIZERS_PARALLELISM="false")
    import torch
    torch.set_num_threads(1)
    torch.backends.mkldnn.enabled = False
    torch.use_deterministic_algorithms(True)


def encode(tensor):
    import numpy as np
    value = tensor.detach().cpu().numpy().astype('<f4', copy=False)
    if not np.isfinite(value).all():
        raise ValueError("Nonfinite tensor")
    return {"shape": list(value.shape), "data": base64.b64encode(value.tobytes()).decode()}


def decode(value, shape):
    import numpy as np
    import torch
    if not isinstance(value, dict) or set(value) != {"shape", "data"} or value['shape'] != list(shape):
        raise ValueError("Tensor shape differs from execution profile")
    raw = base64.b64decode(value['data'], validate=True)
    if len(raw) != int(np.prod(shape))*4:
        raise ValueError("Tensor size differs from execution profile")
    array = np.frombuffer(raw, dtype='<f4').copy().reshape(shape)
    if not np.isfinite(array).all():
        raise ValueError("Nonfinite tensor")
    return torch.from_numpy(array)


class Engine:
    def __init__(self, asset_dir, profile):
        configure()
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self.profile, self.asset_dir = profile, Path(asset_dir)
        for name, expected in profile['files'].items():
            if Path(name).name != name:
                raise ValueError("Asset names must be simple filenames")
            path = self.asset_dir / name
            h = hashlib.sha256()
            with path.open('rb') as f:
                for chunk in iter(lambda:f.read(1024**2), b''):
                    h.update(chunk)
            if h.hexdigest() != expected:
                raise ValueError(f"Model asset failed checksum: {name}")
        self.tokenizer = AutoTokenizer.from_pretrained(self.asset_dir, local_files_only=True,
                                                       trust_remote_code=False)
        self.model = AutoModelForCausalLM.from_pretrained(self.asset_dir, local_files_only=True,
            trust_remote_code=False, dtype=torch.float32, attn_implementation='eager').eval()
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)
        if self.model.config.hidden_size != profile['hidden_size']:
            raise ValueError("Model shape differs from profile")
        self.rank, self.hidden = profile['adapter_rank'], profile['hidden_size']
        if not 1 <= self.rank <= 8:
            raise ValueError("Adapter rank exceeds supported bound")

    def initial(self):
        import torch
        generator = torch.Generator().manual_seed(20260910)
        a = torch.randn(self.rank, self.hidden, generator=generator) * 0.01
        b = torch.zeros(self.hidden, self.rank)
        return {'a':encode(a), 'b':encode(b)}

    def adapter(self, weights, train=False):
        if not isinstance(weights,dict) or set(weights) != {'a','b'}:
            raise ValueError("Expected exactly two adapter matrices")
        return (decode(weights['a'],(self.rank,self.hidden)).requires_grad_(train),
                decode(weights['b'],(self.hidden,self.rank)).requires_grad_(train))

    def features(self, ids):
        import torch
        if (not isinstance(ids,list) or not 2 <= len(ids) <= self.profile['max_input_tokens']
                or any(type(v) is not int or not 0 <= v < self.model.config.vocab_size for v in ids)):
            raise ValueError("Token sequence outside execution profile")
        with torch.no_grad():
            h = self.model.model(input_ids=torch.tensor([ids]), use_cache=False).last_hidden_state
        return encode(h)

    def update(self, weights, ids, features):
        import torch
        a,b = self.adapter(weights,train=True)
        h = decode(features,(1,len(ids),self.hidden))
        logits = self.model.lm_head(h + (h @ a.T) @ b.T)
        loss = torch.nn.functional.cross_entropy(logits[:,:-1].reshape(-1,logits.shape[-1]),
                                                 torch.tensor(ids[1:]))
        loss.backward()
        torch.nn.utils.clip_grad_norm_([a,b], 1.0, foreach=False)
        lr = self.profile['learning_rate']
        new = {'a':encode(a-lr*a.grad), 'b':encode(b-lr*b.grad)}
        return {'weights':new,'loss_hex':float(loss.detach()).hex(),
                'gradient_root':digest(canonical({'a':encode(a.grad),'b':encode(b.grad)}))}

    def train(self, weights, ids):
        feature = self.features(ids)
        value = self.update(weights, ids, feature)
        value['feature_root'] = digest(canonical(feature))
        return value

    def evaluate(self, weights, sequences):
        import torch
        a,b = self.adapter(weights)
        losses=[]
        with torch.no_grad():
            for ids in sequences:
                h=decode(self.features(ids),(1,len(ids),self.hidden))
                logits=self.model.lm_head(h+(h@a.T)@b.T)
                losses.append(float(torch.nn.functional.cross_entropy(
                    logits[:,:-1].reshape(-1,logits.shape[-1]),torch.tensor(ids[1:]))))
        if not losses:
            raise ValueError("Evaluation requires held-out sequences")
        return sum(losses)/len(losses)

    def infer(self, weights, request):
        import torch
        if (not isinstance(request,dict) or set(request) != {'prompt','max_tokens'}
                or not isinstance(request['prompt'],str) or not 1 <= len(request['prompt'].encode()) <= 2048
                or type(request['max_tokens']) is not int
                or not 1 <= request['max_tokens'] <= self.profile['max_new_tokens']):
            raise ValueError("Inference request outside execution profile")
        ids = self.tokenizer.apply_chat_template([{'role':'user','content':request['prompt']}],
                     add_generation_prompt=True, tokenize=True)
        if len(ids) > self.profile['max_input_tokens']:
            raise ValueError("Prompt exceeds the model token limit")
        a,b = self.adapter(weights)
        out=[];cache=None
        with torch.no_grad():
            for _ in range(request['max_tokens']):
                inputs = ids if cache is None else [out[-1]]
                result=self.model.model(input_ids=torch.tensor([inputs]),past_key_values=cache,use_cache=True)
                cache=result.past_key_values
                h=result.last_hidden_state[:,-1:]
                logits=self.model.lm_head(h+(h@a.T)@b.T)
                token=int(logits[0,-1].argmax())
                out.append(token)
                if token==self.tokenizer.eos_token_id:
                    break
        return {'text':self.tokenizer.decode(out,skip_special_tokens=True),'token_ids':out,
                'model_root':digest(canonical(weights)),'request_root':digest(canonical(request))}
