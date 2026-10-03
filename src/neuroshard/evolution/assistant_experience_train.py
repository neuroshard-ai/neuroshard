"""Train the update and addition arms on verified experience plus parent replay.

Both arms see identical sequences in an identical order. The addition wraps the
declared q/v projections with zero-output LoRA and leaves every backbone tensor
unchanged; the update trains FP32 master copies of those same projections.
Loss covers assistant spans only, normalized per sequence.
"""

import hashlib
import json
import math
from pathlib import Path
import random

import torch

from neuroshard.evolution.modular_reference_execution import identity, save

END = '<|end_of_text|>'


def encode(tokenizer, item, tools):
    """One canonical sequence with labels on trainable assistant text and its end token."""
    messages = item['messages']
    text = tokenizer.apply_chat_template(messages, tools=tools, tokenize=False)
    spans = []
    for index, message in enumerate(messages):
        if not item['trainable'][index]:
            continue
        if message['role'] != 'assistant':
            raise ValueError('only assistant messages are trainable')
        prefix = tokenizer.apply_chat_template(messages[:index], tools=tools, add_generation_prompt=True,
                                               tokenize=False)
        end = len(prefix) + len(message['content'])
        if (not text.startswith(prefix) or text[len(prefix):end] != message['content']
                or not text.startswith(END, end)):
            raise ValueError('assistant span is not prefix-consistent under the chat template')
        spans.append((len(prefix), end + len(END)))
    encoded = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
    ids, labels = list(encoded['input_ids']), [-100] * len(encoded['input_ids'])
    for position, (start, stop) in enumerate(encoded['offset_mapping']):
        for left, right in spans:
            if start < right and stop > left:
                if start < left or stop > right:
                    raise ValueError('a token straddles an assistant span boundary')
                labels[position] = ids[position]
    if not any(label != -100 for label in labels):
        raise ValueError('sequence has no trainable tokens')
    return {'input_ids': ids, 'labels': labels, 'sha256': identity([ids, labels])}


class LoRALinear(torch.nn.Module):
    def __init__(self, base, rank, alpha, generator):
        super().__init__()
        self.base, self.scale = base, alpha / rank
        # The seeded generator lives on the CPU; drawing there keeps initialization device-independent.
        initial = torch.empty(rank, base.in_features, dtype=torch.float32)
        torch.nn.init.kaiming_uniform_(initial, a=math.sqrt(5), generator=generator)
        self.lora_a = torch.nn.Parameter(initial.to(base.weight.device))
        self.lora_b = torch.nn.Parameter(torch.zeros(base.out_features, rank, dtype=torch.float32,
                                                     device=base.weight.device))

    def forward(self, x):
        delta = (x.to(torch.float32) @ self.lora_a.T) @ self.lora_b.T
        return self.base(x) + (delta * self.scale).to(x.dtype)


class MasterLinear(torch.nn.Module):
    """Trainable FP32 master weight; the forward uses the activation dtype."""

    def __init__(self, base):
        super().__init__()
        if base.bias is not None:
            raise ValueError('Granite projections have no bias')
        self.weight = torch.nn.Parameter(base.weight.detach().to(torch.float32).clone())

    def forward(self, x):
        return torch.nn.functional.linear(x, self.weight.to(x.dtype))


PROJECTIONS = ('self_attn.q_proj', 'self_attn.v_proj')


def projections(model, layers, paths=PROJECTIONS):
    """(tensor name, owning module, attribute) for each declared projection, layer by layer."""
    for index in layers:
        layer = model.model.layers[index]
        for path in paths:
            owner, _, name = path.rpartition('.')
            yield f'model.layers.{index}.{path}', layer.get_submodule(owner), name


def prepare(model, arm, spec):
    """Freeze the backbone and attach the arm's trainable tensors; returns them by name."""
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    trainable = {}
    generator = torch.Generator(device='cpu').manual_seed(spec['seed'])
    for name, attention, projection in projections(model, spec['layers'], spec.get('projections', PROJECTIONS)):
        base = getattr(attention, projection)
        if arm == 'addition':
            wrapped = LoRALinear(base, spec['rank'], spec['alpha'], generator)
            trainable[name + '.lora_a'], trainable[name + '.lora_b'] = wrapped.lora_a, wrapped.lora_b
        elif arm == 'update':
            wrapped = MasterLinear(base)
            trainable[name + '.weight'] = wrapped.weight
        else:
            raise ValueError('unknown trained arm')
        setattr(attention, projection, wrapped)
    return trainable


def sequence_loss(model, sequence, device):
    """Mean next-token negative log-likelihood over labeled positions."""
    ids = torch.tensor([sequence['input_ids']], device=device)
    labels = torch.tensor(sequence['labels'][1:], device=device)
    hidden = model.model(input_ids=ids).last_hidden_state[0, :-1]
    positions = (labels != -100).nonzero(as_tuple=True)[0]
    logits = model.lm_head(hidden[positions]).float() / model.config.logits_scaling
    return torch.nn.functional.cross_entropy(logits, labels[positions])


@torch.no_grad()
def negative_log_likelihood(model, sequence, device='cpu'):
    return float(sequence_loss(model, sequence, device))


def sequence_logprob(model, sequence, device):
    """Summed log-probability of the labeled tokens (one assistant message and its end token)."""
    ids = torch.tensor([sequence['input_ids']], device=device)
    labels = torch.tensor(sequence['labels'][1:], device=device)
    hidden = model.model(input_ids=ids).last_hidden_state[0, :-1]
    positions = (labels != -100).nonzero(as_tuple=True)[0]
    logits = model.lm_head(hidden[positions]).float() / model.config.logits_scaling
    return torch.log_softmax(logits, dim=-1).gather(1, labels[positions].unsqueeze(1)).sum()


def encode_pair(tokenizer, pair, tools):
    """Chosen and rejected continuations of one shared prefix, each labeled on its message only."""
    def item(text):
        return {'messages': pair['messages'] + [{'role': 'assistant', 'content': text}],
                'trainable': [False] * len(pair['messages']) + [True]}

    chosen, rejected = encode(tokenizer, item(pair['chosen']), tools), encode(tokenizer, item(pair['rejected']), tools)
    return {'case_id': pair['case_id'], 'chosen': chosen, 'rejected': rejected,
            'sha256': identity([chosen['sha256'], rejected['sha256']])}


def schedule(experience, replay, spec, preferences=0):
    """Deterministic microbatch order: per update, fixed counts of experience, replay and preference rows."""
    rng = random.Random(spec['seed'])
    per_update = spec['gradient_accumulation']
    replay_count = per_update // (spec['experience_per_replay'] + 1)
    experience_count = per_update - replay_count
    preference_count = spec.get('preference_per_update', 0) if preferences else 0
    if not experience or (replay_count and not replay):
        raise ValueError('missing experience or replay sequences')

    def stream(rows):
        while True:
            order = list(range(rows))
            rng.shuffle(order)
            yield from order

    experience_stream = stream(len(experience))
    replay_stream = stream(len(replay)) if replay else iter(())
    preference_stream = stream(preferences) if preference_count else iter(())
    updates = []
    for _ in range(spec['steps']):
        batch = [('experience', next(experience_stream)) for _ in range(experience_count)]
        batch += [('replay', next(replay_stream)) for _ in range(replay_count)]
        batch += [('preference', next(preference_stream)) for _ in range(preference_count)]
        rng.shuffle(batch)
        updates.append(batch)
    return updates


def mixture_schedule(sources, spec):
    """Deterministic microbatch order for a declared mixture: per update, fixed counts from each named source."""
    rng = random.Random(spec['seed'])
    counts = spec['mixture']
    if sum(counts.values()) != spec['gradient_accumulation'] or any(count and not sources.get(name)
                                                                    for name, count in counts.items()):
        raise ValueError('mixture does not fill the update or names an empty source')

    def stream(rows):
        while True:
            order = list(range(rows))
            rng.shuffle(order)
            yield from order

    streams = {name: stream(len(sources[name])) for name in sorted(counts) if counts[name]}
    updates = []
    for _ in range(spec['steps']):
        batch = [(name, next(streams[name])) for name in sorted(streams) for _ in range(counts[name])]
        rng.shuffle(batch)
        updates.append(batch)
    return updates


def learning_rate(step, spec, arm):
    base = spec['learning_rates'][arm]
    return base * min(1.0, (step + 1) / spec['warmup_steps']) if spec['warmup_steps'] else base


def train(model, arm, experience, replay, spec, *, device='cpu', progress=None, trainable=None, pairs=None, extra=None):
    """Run the declared schedule once; returns trainable tensors and a work receipt.

    ``trainable`` continues already attached tensors. ``pairs`` adds a DPO term whose
    reference is the model as it stands before this schedule's first update. ``extra``
    names further sequence sources for a declared ``mixture`` in ``spec``.
    """
    torch.manual_seed(spec['seed'])
    trainable = trainable if trainable is not None else prepare(model, arm, spec)
    model.to(device)
    pairs = pairs or []
    references = []
    if pairs:
        model.eval()
        with torch.no_grad():
            references = [(float(sequence_logprob(model, p['chosen'], device)),
                           float(sequence_logprob(model, p['rejected'], device))) for p in pairs]
    model.train()
    optimizer = torch.optim.AdamW(list(trainable.values()), lr=learning_rate(0, spec, arm),
                                  betas=tuple(spec['betas']), eps=spec['epsilon'],
                                  weight_decay=spec['weight_decay'])
    sources = {'experience': experience, 'replay': replay, **(extra or {})}
    if 'mixture' in spec:
        if pairs:
            raise ValueError('a declared mixture takes no preference pairs')
        updates = mixture_schedule(sources, spec)
    else:
        updates = schedule(experience, replay, spec, len(pairs))
    losses, margins, tokens = [], [], 0
    for step, batch in enumerate(updates):
        for group in optimizer.param_groups:
            group['lr'] = learning_rate(step, spec, arm)
        optimizer.zero_grad(set_to_none=True)
        total = 0.0
        for kind, index in batch:
            if kind == 'preference':
                pair, (chosen_ref, rejected_ref) = pairs[index], references[index]
                margin = spec['beta'] * ((sequence_logprob(model, pair['chosen'], device) - chosen_ref)
                                         - (sequence_logprob(model, pair['rejected'], device) - rejected_ref))
                loss = -torch.nn.functional.logsigmoid(margin) * spec['preference_weight'] / len(batch)
                margins.append(margin.detach().item())
                tokens += len(pair['chosen']['input_ids']) + len(pair['rejected']['input_ids'])
            else:
                sequence = sources[kind][index]
                loss = sequence_loss(model, sequence, device) / len(batch)
                tokens += len(sequence['input_ids'])
            loss.backward()
            total += loss.detach().item()
        norm = torch.nn.utils.clip_grad_norm_(list(trainable.values()), spec['gradient_clip'])
        if not torch.isfinite(norm):
            raise ValueError('nonfinite gradient norm')
        optimizer.step()
        losses.append(total)
        if progress:
            progress(step, total)
    model.eval()
    return trainable, {'arm': arm, 'steps': len(updates), 'microbatches': sum(map(len, updates)),
                       'tokens_processed': tokens, 'losses': losses, 'preference_margins': margins,
                       'preference_pairs': len(pairs),
                       'trainable_parameters': sum(p.numel() for p in trainable.values()),
                       'schedule_sha256': identity(updates), 'optimizer_state': optimizer.state_dict()}


def checkpoint(directory, trainable, receipt, roots):
    """Terminal checkpoint: trainable tensors, optimizer state and bound roots; no selection."""
    from safetensors.torch import save_file

    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=False)
    tensors = {name: tensor.detach().cpu().contiguous() for name, tensor in trainable.items()}
    save_file(tensors, str(directory / 'trainable.safetensors'))
    torch.save({'optimizer': receipt['optimizer_state'], 'torch_rng': torch.get_rng_state()},
               directory / 'optimizer.pt')
    digest = hashlib.sha256((directory / 'trainable.safetensors').read_bytes()).hexdigest()
    manifest = {'format': 'neuroshard-assistant-experience-checkpoint/1', 'arm': receipt['arm'],
                'trainable_sha256': digest, 'trainable_parameters': receipt['trainable_parameters'],
                'tensors': {name: list(t.shape) for name, t in tensors.items()},
                'steps': receipt['steps'], 'schedule_sha256': receipt['schedule_sha256'],
                'final_loss': receipt['losses'][-1], 'roots': roots}
    save(directory / 'manifest.json', manifest, exclusive=True)
    return manifest


def serving(model, spec):
    """Store updated projections as plain Linear layers in the backbone dtype, as a deployment would.

    Adapters stay unmerged; they are the separately served module.
    """
    converted = 0
    for _, attention, projection in projections(model, spec['layers'], spec.get('projections', PROJECTIONS)):
        module = getattr(attention, projection)
        if isinstance(module, MasterLinear):
            dtype = model.get_input_embeddings().weight.dtype
            linear = torch.nn.Linear(module.weight.shape[1], module.weight.shape[0], bias=False,
                                     dtype=dtype, device=module.weight.device)
            with torch.no_grad():
                linear.weight.copy_(module.weight.to(dtype))
            linear.weight.requires_grad_(False)
            setattr(attention, projection, linear)
            converted += 1
    return converted


def load_trainable(model, arm, spec, directory):
    """Attach a saved arm to a fresh parent for evaluation or serving."""
    manifest, _ = resume(model, arm, spec, directory)
    model.eval()
    return manifest


def resume(model, arm, spec, directory):
    """Attach a saved arm and return its trainable tensors so training can continue."""
    from safetensors.torch import load_file

    directory = Path(directory)
    manifest = json.loads((directory / 'manifest.json').read_text())
    if manifest['arm'] != arm:
        raise ValueError('checkpoint belongs to another arm')
    if hashlib.sha256((directory / 'trainable.safetensors').read_bytes()).hexdigest() != manifest['trainable_sha256']:
        raise ValueError('trainable tensors changed')
    saved = load_file(str(directory / 'trainable.safetensors'))
    trainable = prepare(model, arm, spec)
    if set(saved) != set(trainable):
        raise ValueError('checkpoint tensor inventory differs')
    with torch.no_grad():
        for name, parameter in trainable.items():
            parameter.copy_(saved[name].to(parameter.device))
    return manifest, trainable
