"""A fully trainable, partitionable Llama model with tied embedding/output weights.

Only a shard's components reside on a worker. The first shard owns the tied
embedding, final norm and output head, so their gradients need no dense transfer.
"""
import copy
import math
import os
from pathlib import Path
from . import schema


def configure():
    os.environ.update(ATEN_CPU_CAPABILITY='default', MKL_ENABLE_INSTRUCTIONS='SSE4_2',
                      OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', TOKENIZERS_PARALLELISM='false')
    import torch
    if torch.backends.cpu.get_cpu_capability() != 'DEFAULT':
        raise RuntimeError('PyTorch CPU dispatch was initialized before the NeuroShard numerical profile; start a fresh process')
    torch.set_num_threads(1)
    torch.set_default_dtype(torch.float32)
    torch.set_default_device('cpu')
    torch.backends.mkldnn.enabled = False
    torch.use_deterministic_algorithms(True)


configure()
import torch
from torch import nn
from torch.nn import functional as F


CONFIG_KEYS = ('hidden_size', 'intermediate_size', 'num_attention_heads', 'num_key_value_heads',
               'vocab_size', 'rms_norm_eps', 'rope_theta', 'max_position_embeddings', 'num_hidden_layers')


def rms(value, weight, epsilon):
    return value * torch.rsqrt(value.float().square().mean(-1, keepdim=True) + epsilon) * weight


class Block(nn.Module):
    def __init__(self, config, tensors):
        super().__init__()
        self.config = config
        self.weights = nn.ParameterDict({k.replace('.', '_'): nn.Parameter(v.clone()) for k, v in tensors.items()})

    def w(self, name):
        return self.weights[name.replace('.', '_')]

    def forward(self, hidden, positions=None):
        c = self.config
        batch, length, width = hidden.shape
        heads, kv_heads = c['num_attention_heads'], c['num_key_value_heads']
        dimension = width // heads
        residual = hidden
        hidden = rms(hidden, self.w('input_layernorm.weight'), c['rms_norm_eps'])
        q = F.linear(hidden, self.w('self_attn.q_proj.weight')).view(batch, length, heads, dimension).transpose(1, 2)
        k = F.linear(hidden, self.w('self_attn.k_proj.weight')).view(batch, length, kv_heads, dimension).transpose(1, 2)
        v = F.linear(hidden, self.w('self_attn.v_proj.weight')).view(batch, length, kv_heads, dimension).transpose(1, 2)
        if positions is None:
            positions = torch.arange(length).unsqueeze(0)
        inverse = 1.0 / (c['rope_theta'] ** (torch.arange(0, dimension, 2).float() / dimension))
        angles = (inverse[None, :, None] @ positions[:, None, :].float()).transpose(1, 2)
        angles = torch.cat((angles, angles), -1)
        cosine, sine = angles.cos().unsqueeze(1), angles.sin().unsqueeze(1)
        def rotate(value):
            a, b = value.chunk(2, dim=-1)
            return torch.cat((-b, a), dim=-1)
        q, k = q * cosine + rotate(q) * sine, k * cosine + rotate(k) * sine
        k = k.repeat_interleave(heads // kv_heads, dim=1)
        v = v.repeat_interleave(heads // kv_heads, dim=1)
        scores = torch.matmul(q, k.transpose(2, 3)) * dimension ** -0.5
        mask = torch.full((length, length), torch.finfo(scores.dtype).min).triu(1)
        scores = scores + mask[None, None]
        attention = F.softmax(scores, dim=-1, dtype=torch.float32)
        hidden = torch.matmul(attention, v).transpose(1, 2).contiguous().view(batch, length, width)
        hidden = residual + F.linear(hidden, self.w('self_attn.o_proj.weight'))
        residual = hidden
        hidden = rms(hidden, self.w('post_attention_layernorm.weight'), c['rms_norm_eps'])
        hidden = F.silu(F.linear(hidden, self.w('mlp.gate_proj.weight'))) * F.linear(hidden, self.w('mlp.up_proj.weight'))
        return residual + F.linear(hidden, self.w('mlp.down_proj.weight'))

BLOCK_KEYS = ('input_layernorm.weight', 'post_attention_layernorm.weight',
              'self_attn.q_proj.weight', 'self_attn.k_proj.weight', 'self_attn.v_proj.weight',
              'self_attn.o_proj.weight', 'mlp.gate_proj.weight', 'mlp.up_proj.weight', 'mlp.down_proj.weight')


def from_pretrained(directory, store):
    """Convert the pinned local seed one component at a time, without a full model allocation."""
    import json
    from safetensors import safe_open
    from neuroshard.evolution.objects import digest
    from .seed import verify
    directory = Path(directory)
    verify(directory)
    raw = json.loads((directory / 'config.json').read_text())
    config = {k: raw[k] for k in CONFIG_KEYS}
    if raw['model_type'] != 'llama' or not raw['tie_word_embeddings']:
        raise ValueError('Expected a tied-weight Llama seed')
    components = {}
    with safe_open(directory / 'model.safetensors', framework='pt', device='cpu') as source:
        for name, mapping in [('embed', {'weight':'model.embed_tokens.weight'}), ('norm', {'weight':'model.norm.weight'})]:
            values = {k:source.get_tensor(v).float() for k,v in mapping.items()}
            components[name] = {'root':store.put_tensors(values), 'parameters':sum(t.numel() for t in values.values())}
        for layer in range(config['num_hidden_layers']):
            values = {k:source.get_tensor(f'model.layers.{layer}.{k}').float() for k in BLOCK_KEYS}
            components[f'block_{layer:03}'] = {'root':store.put_tensors(values), 'parameters':sum(t.numel() for t in values.values())}
    value = {'format':'neuroshard-model-v1', 'config':config, 'components':components,
             'parameters':sum(c['parameters'] for c in components.values()), 'parent':None,
             'origin':{'model':'HuggingFaceTB/SmolLM2-135M-Instruct',
                       'weights_sha256':digest((directory/'model.safetensors').read_bytes()), 'license':'Apache-2.0'}}
    return store.put_json(value), value


def grow(parent_root, store, additional_layers=4):
    """Add identity-initialized residual blocks; all old trainable parameters are retained."""
    if type(additional_layers) is not int or not 1 <= additional_layers <= 16:
        raise ValueError('Growth must add 1–16 blocks per revision')
    value = copy.deepcopy(store.json(parent_root))
    schema.model(value)
    old_depth = value['config']['num_hidden_layers']
    last = f'block_{old_depth-1:03}'
    weights = store.tensors(value['components'][last]['root'],schema.shapes(value['config'],last))
    if any(not torch.isfinite(weight).all() for weight in weights.values()):
        raise ValueError('Growth requires finite parent weights')
    weights['self_attn.o_proj.weight'] = torch.zeros_like(weights['self_attn.o_proj.weight'])
    weights['mlp.down_proj.weight'] = torch.zeros_like(weights['mlp.down_proj.weight'])
    component = {'root':store.put_tensors(weights), 'parameters':sum(t.numel() for t in weights.values())}
    for layer in range(old_depth, old_depth + additional_layers):
        value['components'][f'block_{layer:03}'] = copy.deepcopy(component)
    value['config']['num_hidden_layers'] += additional_layers
    value['parameters'] += additional_layers * component['parameters']
    value['parent'] = parent_root
    value['growth'] = {'kind':'identity-residual-depth', 'added_layers':additional_layers}
    schema.model(value)
    return store.put_json(value), value


def place(model, capacities):
    """Contiguous block assignment under advertised resident-parameter limits."""
    schema.model(model)
    capacities = list(capacities)
    if not 1 <= len(capacities) <= 64 or any(type(c) is not int or not 0<c<=schema.MAX_PARAMETERS for c in capacities):
        raise ValueError('Positive worker parameter capacities required')
    partitions = [[] for _ in capacities]
    used = [0 for _ in capacities]
    for name in ('embed','norm'):
        partitions[0].append(name)
        used[0] += model['components'][name]['parameters']
    if used[0] > capacities[0]:
        raise ValueError('First worker cannot hold the tied embedding and output head')
    worker = 0
    for name in sorted((n for n in model['components'] if n.startswith('block_')),key=schema.block_order):
        size = model['components'][name]['parameters']
        while worker < len(capacities) and used[worker]+size > capacities[worker]:
            if not partitions[worker]:
                raise ValueError('Filter workers that cannot hold even one complete block before placement')
            worker += 1
        if worker == len(capacities):
            raise ValueError('Insufficient aggregate worker capacity for this architecture')
        partitions[worker].append(name)
        used[worker] += size
    return [{'components':names,'parameters':size,'capacity':cap} for names,size,cap in zip(partitions,used,capacities) if names]


class Shard(nn.Module):
    def __init__(self, model, partition, store):
        super().__init__()
        schema.partition(model, partition)
        self.config, self.names = model['config'], list(partition['components'])
        self.blocks = nn.ModuleDict()
        for name in self.names:
            values = store.tensors(model['components'][name]['root'], schema.shapes(self.config, name))
            if any(not torch.isfinite(value).all() for value in values.values()):
                raise ValueError('Model component contains nonfinite weights')
            if name == 'embed':
                self.embedding = nn.Embedding.from_pretrained(values['weight'], freeze=False)
            elif name == 'norm':
                self.final_norm = nn.Parameter(values['weight'])
            else:
                self.blocks[name] = Block(self.config, values)
        self.resident_parameters = sum(p.numel() for p in self.parameters())
        if self.resident_parameters != partition['parameters'] or self.resident_parameters > partition['capacity']:
            raise ValueError('Resident parameters differ from capacity commitment')

    def forward(self, value, ids=False):
        hidden = self.embedding(value) if ids else value
        for block in self.blocks.values():
            hidden = block(hidden)
        return hidden

    def logits(self, hidden):
        hidden = rms(hidden, self.final_norm, self.config['rms_norm_eps'])
        return F.linear(hidden, self.embedding.weight)

    def loss(self, hidden, ids, labels=None):
        logits = self.logits(hidden)
        targets = ids if labels is None else labels
        return F.cross_entropy(logits[:, :-1].reshape(-1, self.config['vocab_size']), targets[:, 1:].reshape(-1))

    def gradient_squared_norm(self):
        return math.fsum(float(p.grad.double().square().sum()) for _,p in sorted(self.named_parameters()) if p.grad is not None)

    def update(self, learning_rate, scale):
        with torch.no_grad():
            for p in self.parameters():
                if p.grad is None:
                    raise ValueError('A trainable component received no gradient')
                p.add_(p.grad, alpha=-learning_rate*scale)

    def save(self, store):
        values = {}
        for name in self.names:
            if name == 'embed':
                tensors = {'weight':self.embedding.weight}
            elif name == 'norm':
                tensors = {'weight':self.final_norm}
            else:
                block = self.blocks[name]
                tensors = {key:block.w(key) for key in BLOCK_KEYS}
            values[name] = {'root':store.put_tensors(tensors), 'parameters':sum(v.numel() for v in tensors.values())}
        return values
