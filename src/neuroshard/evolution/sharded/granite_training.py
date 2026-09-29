"""Train a declared arm across Granite owners with the single-host trainer's arithmetic.

Owner 0 runs the declared schedule: it embeds each sequence, receives the final
hidden state, computes the loss through its norm and tied head, and returns the
boundary gradient. The owner holding the arm's layers keeps their graphs,
backpropagates into its trainable tensors and steps its optimizer; no backbone
tensor changes and no other owner computes a backward pass. Initialization draws
the same seeded stream as the single-host trainer, so both start identically.
"""
import json
from pathlib import Path

import torch
import torch.distributed as dist

from neuroshard.evolution import assistant_experience_train as trainer

from .granite_pipeline import RESET, STOP, command

TRAIN_FORWARD, BACKWARD, STEP, FULL_FORWARD, APPLY = 4, 5, 6, 7, 8


def attach(partition, arm, spec):
    """The whole arm on this owner, drawn from the single-host initialization stream in its order."""
    if any(not partition.begin <= index < partition.end for index in spec['layers']):
        raise ValueError('the arm must live on one owner')
    generator = torch.Generator(device='cpu').manual_seed(spec['seed'])
    trainable = {}
    for index in spec['layers']:
        for name in ('q_proj', 'v_proj'):
            key = f'model.layers.{index}.self_attn.{name}'
            attention = partition.layers[index - partition.begin].self_attn
            base = getattr(attention, name)
            if arm == 'addition':
                wrapped = trainer.LoRALinear(base, spec['rank'], spec['alpha'], generator)
                trainable[key + '.lora_a'], trainable[key + '.lora_b'] = wrapped.lora_a, wrapped.lora_b
            elif arm == 'update':
                wrapped = trainer.MasterLinear(base)
                trainable[key + '.weight'] = wrapped.weight
            else:
                raise ValueError('unknown trained arm')
            setattr(attention, name, wrapped)
    return trainable


class OwnerTraining:
    """Trainable tensors on one owner, their optimizer, and graphs awaiting boundary gradients."""

    def __init__(self, partition, arm, spec, holder):
        if holder != len(partition.boundaries) - 2:
            raise ValueError('boundary gradients return to the last owner, which must hold the arm')
        self.partition, self.arm, self.spec, self.holder = partition, arm, spec, holder
        self.trainable = attach(partition, arm, spec) if partition.rank == holder else {}
        self.optimizer = (torch.optim.AdamW(list(self.trainable.values()), lr=trainer.learning_rate(0, spec, arm),
                                            betas=tuple(spec['betas']), eps=spec['epsilon'],
                                            weight_decay=spec['weight_decay']) if self.trainable else None)
        self.pending, self.gradients, self.norms = [], [], []

    def forward(self, hidden, train):
        if self.partition.rank < self.holder or not train:
            with torch.no_grad():
                return self.partition(hidden)
        with torch.enable_grad():
            output = self.partition(hidden)
        self.pending.append(output)
        return output.detach()

    def backward(self, gradient):
        self.gradients.append(gradient)

    def apply(self):
        """One backward over the microbatch's graphs, so shared tensors accumulate as in a single host."""
        if len(self.gradients) != len(self.pending):
            raise ValueError('boundary gradients do not match pending graphs')
        torch.autograd.backward(self.pending, self.gradients)
        self.pending, self.gradients = [], []

    def step(self, index):
        norm = torch.nn.utils.clip_grad_norm_(list(self.trainable.values()), self.spec['gradient_clip'])
        if not torch.isfinite(norm):
            raise ValueError('nonfinite gradient norm')
        for group in self.optimizer.param_groups:
            group['lr'] = trainer.learning_rate(index, self.spec, self.arm)
        self.optimizer.step()
        self.optimizer.zero_grad(set_to_none=True)
        self.norms.append(float(norm))

    def save(self, directory, step):
        from safetensors.torch import save_file

        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        save_file({k: v.detach().contiguous() for k, v in self.trainable.items()}, str(directory / 'trainable.safetensors'))
        torch.save(self.optimizer.state_dict(), directory / 'optimizer.pt')
        (directory / 'state.json').write_text(json.dumps({'completed_steps': step, 'norms': self.norms}))

    def load(self, directory):
        from safetensors.torch import load_file

        directory = Path(directory)
        values = load_file(directory / 'trainable.safetensors')
        if set(values) != set(self.trainable):
            raise ValueError('checkpoint differs from this owner\'s share of the arm')
        with torch.no_grad():
            for key, value in values.items():
                self.trainable[key].copy_(value)
        self.optimizer.load_state_dict(torch.load(directory / 'optimizer.pt', weights_only=True))
        state = json.loads((directory / 'state.json').read_text())
        self.norms = state['norms']
        return state['completed_steps']


def serve(partition, ring, training, checkpoints=None, fail_at_step=None):
    """Owner loop for ranks after 0 during training; the holder checkpoints after every step."""
    while True:
        op, value = command(0)
        if op == STOP:
            return len(training.norms)
        if op == RESET:
            continue
        if op in (TRAIN_FORWARD, FULL_FORWARD):
            hidden = ring.receive(ring.rank - 1, value)
            output = training.forward(hidden, op == TRAIN_FORWARD)
            ring.send(output, (ring.rank + 1) % ring.world)
        elif op == BACKWARD:
            if partition.rank == training.holder:
                gradient = torch.empty((1, value, 2 * partition.config.hidden_size), dtype=torch.uint8)
                dist.recv(gradient, src=0)
                ring.received_bytes += gradient.numel()
                training.backward(gradient.view(torch.bfloat16))
        elif op == APPLY:
            if partition.rank == training.holder:
                training.apply()
        elif op == STEP:
            if partition.rank == training.holder:
                if fail_at_step is not None and value == fail_at_step:
                    import os
                    os._exit(17)
                training.step(value)
                if checkpoints is not None:
                    training.save(checkpoints, value + 1)
        else:
            raise ValueError('unknown training command')


class TrainingDriver:
    """Owner 0: the declared schedule, losses through the tied head, and boundary gradients."""

    def __init__(self, partition, ring, holder):
        if partition.rank != 0 or holder == 0:
            raise ValueError('owner 0 drives; the arm lives on another owner')
        self.partition, self.ring, self.holder, self.leaves = partition, ring, holder, []

    def hidden(self, ids, train):
        command(TRAIN_FORWARD if train else FULL_FORWARD, ids.shape[1])
        with torch.no_grad():
            first = self.partition(self.partition.embed(ids))
        self.ring.send(first, 1)
        final = self.ring.receive(self.ring.world - 1, ids.shape[1]).clone()
        if train:
            final.requires_grad_(True)
            self.leaves.append(final)
        return final

    def head(self, final, sequence):
        """The single-host trainer's loss arithmetic on the owners' final hidden state."""
        labels = torch.tensor(sequence['labels'][1:])
        hidden = self.partition.norm(final)[0, :-1]
        positions = (labels != -100).nonzero(as_tuple=True)[0]
        logits = torch.nn.functional.linear(hidden[positions], self.partition.embedding.weight).float()
        return logits / self.partition.config.logits_scaling, labels[positions]

    def loss(self, sequence, train=True):
        logits, labels = self.head(self.hidden(torch.tensor([sequence['input_ids']]), train), sequence)
        return torch.nn.functional.cross_entropy(logits, labels)

    def logprob(self, sequence, train=True):
        logits, labels = self.head(self.hidden(torch.tensor([sequence['input_ids']]), train), sequence)
        return torch.log_softmax(logits, dim=-1).gather(1, labels.unsqueeze(1)).sum()

    def backward(self, loss):
        """Local backward to the boundary, then every boundary gradient and one remote backward."""
        loss.backward()
        for leaf in self.leaves:
            command(BACKWARD, leaf.shape[1])
            payload = leaf.grad.detach().contiguous().view(torch.uint8)
            dist.send(payload, dst=self.holder)
            self.ring.sent_bytes += payload.numel()
        command(APPLY)
        self.leaves = []

    def step(self, index):
        command(STEP, index)

    def stop(self):
        command(STOP)


def run_owner(config_dir, shards_dir, rank, world, address, port, job_path, result_path, *, timeout=60):
    """One training owner process: owner 0 drives the schedule, the last owner holds and checkpoints the arm."""
    from datetime import timedelta
    import resource
    import time

    from . import granite
    from .granite_pipeline import Ring

    job = json.loads(Path(job_path).read_text())
    torch.set_num_threads(job.get('threads', 1))
    config = granite.load_config(config_dir)
    partition, manifest = granite.load_partition(config, shards_dir, rank)
    holder = world - 1
    training = OwnerTraining(partition, job['arm'], job['spec'], holder)
    start = training.load(job['resume_from']) if rank == holder and job.get('resume_from') else job.get('start', 0)
    if start != job.get('start', 0):
        raise ValueError('the arm checkpoint and the driver disagree on completed steps')
    dist.init_process_group('gloo', init_method=f'tcp://{address}:{port}', rank=rank, world_size=world,
                            timeout=timedelta(seconds=timeout))
    ring = Ring(rank, world, config.hidden_size, job['max_tokens'])
    result = {'rank': rank, 'shard_sha256': manifest['sha256'], 'resident_bytes': partition.resident_bytes(),
              'trainable_parameters': sum(p.numel() for p in training.trainable.values()), 'start': start}
    began = time.monotonic()
    try:
        if rank == 0:
            driver = TrainingDriver(partition, ring, holder)
            progress = Path(result_path).with_name('progress.json')
            saved = Path(result_path).with_name('references.json')
            receipt = train(driver, job['arm'], job['experience'], job['replay'], job['spec'], job.get('pairs'),
                            job.get('references'), job.get('start', 0),
                            lambda done, losses: progress.write_text(json.dumps({'completed_steps': done})),
                            lambda values: saved.write_text(json.dumps(values)))
            driver.stop()
            result['receipt'] = receipt
        else:
            fail = job.get('fail')
            steps = serve(partition, ring, training, job.get('checkpoints') if rank == holder else None,
                          fail['step'] if fail and rank == holder else None)
            result['norms'] = training.norms if rank == holder else None
            result['steps'] = steps
            if rank == holder:
                result['trainable_sha256'] = digests(training.trainable)
        result['completed'] = True
    except Exception as error:
        result.update(completed=False, error=f'{type(error).__name__}: {error}')
    finally:
        result.update(seconds=time.monotonic() - began, sent_bytes=ring.sent_bytes, received_bytes=ring.received_bytes,
                      peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024)
        Path(result_path).write_text(json.dumps(result, indent=2) + '\n')
        if result.get('completed'):
            dist.destroy_process_group()
    return result


def digests(trainable):
    """SHA-256 of each tensor's raw bytes, comparable across hosts without moving the tensors."""
    import hashlib

    return {name: hashlib.sha256(value.detach().contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()
            for name, value in sorted(trainable.items())}


def train(driver, arm, experience, replay, spec, pairs=None, references=None, start=0, on_step=None,
          on_references=None):
    """The single-host ``trainer.train`` schedule and objective, driven across owners."""
    pairs = pairs or []
    if pairs and references is None:
        with torch.no_grad():
            references = [(float(driver.logprob(p['chosen'], False)), float(driver.logprob(p['rejected'], False)))
                          for p in pairs]
        if on_references:
            on_references(references)
    updates = trainer.schedule(experience, replay, spec, len(pairs))
    sources = {'experience': experience, 'replay': replay}
    losses, margins = [], []
    for step in range(start, len(updates)):
        batch, total = updates[step], 0.0
        for kind, index in batch:
            if kind == 'preference':
                pair, (chosen_ref, rejected_ref) = pairs[index], references[index]
                margin = spec['beta'] * ((driver.logprob(pair['chosen']) - chosen_ref)
                                         - (driver.logprob(pair['rejected']) - rejected_ref))
                loss = -torch.nn.functional.logsigmoid(margin) * spec['preference_weight'] / len(batch)
                margins.append(margin.detach().item())
            else:
                loss = driver.loss(sources[kind][index]) / len(batch)
            driver.backward(loss)
            total += loss.detach().item()
        driver.step(step)
        losses.append(total)
        if on_step:
            on_step(step + 1, losses)
    return {'arm': arm, 'steps': len(updates), 'losses': losses, 'preference_margins': margins,
            'references': references, 'schedule_sha256': trainer.identity(updates)}
