"""Serve the complete learned assistant across Granite owners.

Owner 0 renders every request, selects the parent or the arm once per episode
from the parent's final-layer feature, and decodes greedily; the owner holding
the arm switches it on or off for the episode. Every owner keeps one episode
cache and crops it to the longest token prefix shared with the new request, as
the single-host prefix-cache responder does, so each request runs the same
operations on the same shapes.
"""
import hashlib
import json
import resource
import time
from pathlib import Path

import torch

from neuroshard.evolution import assistant_experience_train as trainer
from neuroshard.evolution.modular_reference_execution import identity

from .granite_pipeline import FORWARD, RESET, STOP, command

CROP, ADAPTER, FEATURE = 9, 10, 11


class Adapter:
    """The saved addition arm on its owner, switchable per episode without touching the backbone."""

    def __init__(self, partition, spec, directory):
        from safetensors.torch import load_file

        from .granite_training import attach

        directory = Path(directory)
        manifest = json.loads((directory / 'manifest.json').read_text())
        if (manifest['arm'] != 'addition'
                or hashlib.sha256((directory / 'trainable.safetensors').read_bytes()).hexdigest()
                != manifest['trainable_sha256']):
            raise ValueError('arm checkpoint differs from its manifest')
        trainable = attach(partition, 'addition', spec)
        saved = load_file(str(directory / 'trainable.safetensors'))
        if set(saved) != set(trainable):
            raise ValueError('arm checkpoint tensor inventory differs')
        with torch.no_grad():
            for name, value in trainable.items():
                value.copy_(saved[name])
                value.requires_grad_(False)
        self.manifest, self.wrapped = manifest, []
        for index in spec['layers']:
            attention = partition.layers[index - partition.begin].self_attn
            for name in ('q_proj', 'v_proj'):
                self.wrapped.append((attention, name, getattr(attention, name)))
        self.enabled = True

    def set(self, enabled):
        for attention, name, module in self.wrapped:
            setattr(attention, name, module if enabled else module.base)
        self.enabled = enabled


def serve(partition, ring, adapter=None, log=None, fault=None):
    """Owner loop for ranks after 0: episode caches, prefix crops, arm switching and parent features.

    ``log`` records every command for replay audits; ``fault`` (a forward-message
    index) perturbs one sent tensor, standing in for a cheating owner in tests.
    """
    from transformers import DynamicCache

    cache, steps, busy, forwards = DynamicCache(), 0, 0.0, 0
    while True:
        op, value = command(0)
        if op == STOP:
            return {'steps': steps, 'busy_seconds': busy}
        if op in (RESET, CROP, ADAPTER) and log is not None:
            log.command(op, value)
        if op == RESET:
            cache = DynamicCache()
        elif op == CROP:
            cache.crop(value)
        elif op == ADAPTER:
            if adapter is not None:
                adapter.set(bool(value))
        elif op in (FORWARD, FEATURE):
            steps += op == FORWARD
            hidden = ring.receive(ring.rank - 1, value)
            began = time.monotonic()
            with torch.inference_mode():
                if op == FORWARD:
                    mask = torch.ones((1, cache.get_seq_length() + value), dtype=torch.long)
                    out = partition(hidden, mask, cache)
                else:
                    out = partition(hidden, None, DynamicCache())
            busy += time.monotonic() - began
            last = ring.rank == ring.world - 1
            sent = out[:, -1:] if last else out
            if fault is not None and forwards == fault:
                sent = sent.clone()
                sent.view(torch.int16).view(-1)[0] ^= 1
            forwards += 1
            if log is not None:
                log.forward(op, value, hidden, sent)
            ring.send(sent, (ring.rank + 1) % ring.world)
        else:
            raise ValueError('unknown serving command')


class ServingDriver:
    """Owner 0: episode control, the parent feature and cached decoding steps."""

    def __init__(self, partition, ring):
        from transformers import DynamicCache

        if partition.rank != 0 or ring.rank != 0:
            raise ValueError('owner 0 drives serving')
        self.partition, self.ring, self.cache_type = partition, ring, DynamicCache
        self.cache = DynamicCache()
        self.busy_seconds = 0.0

    def feature(self, ids):
        """The frozen parent's final-layer state at the last prompt position, as ``boundary_feature`` computes it."""
        command(ADAPTER, 0)
        command(FEATURE, len(ids))
        tokens = torch.tensor([ids])
        began = time.monotonic()
        with torch.inference_mode():
            hidden = self.partition(self.partition.embed(tokens), None, self.cache_type())
        self.busy_seconds += time.monotonic() - began
        self.ring.send(hidden, 1)
        back = self.ring.receive(self.ring.world - 1, 1)
        with torch.inference_mode():
            return self.partition.norm(back)[0, -1].float().tolist()

    def episode(self, arm):
        command(RESET)
        command(ADAPTER, int(arm))
        self.cache = self.cache_type()

    def cached_tokens(self):
        return self.cache.get_seq_length()

    def crop(self, length):
        command(CROP, length)
        self.cache.crop(length)

    def step(self, tokens, mask):
        command(FORWARD, tokens.shape[1])
        began = time.monotonic()
        with torch.inference_mode():
            hidden = self.partition(self.partition.embed(tokens), mask, self.cache)
        self.busy_seconds += time.monotonic() - began
        self.ring.send(hidden, 1)
        back = self.ring.receive(self.ring.world - 1, 1)
        began = time.monotonic()
        with torch.inference_mode():
            logits = self.partition.logits(back)
        self.busy_seconds += time.monotonic() - began
        return logits

    def next_token(self, tokens, mask):
        return greedy(self.step(tokens, mask))

    def stop(self):
        command(STOP)


def greedy(logits):
    """The prefix-cache responder's check and choice on the last position's logits."""
    logits = logits[0, -1].float()
    if torch.isnan(logits).any() or torch.isposinf(logits).any() or not torch.isfinite(logits).any():
        raise ValueError('nonfinite model logits')
    return int(logits.argmax())


def responder(driver, tokenizer, policy, eos):
    """One episode's responder with the record format and arithmetic of ``cached_responder``."""
    generation = policy['generation']
    state = {'ids': []}

    def respond(messages, tools):
        prompt = tokenizer.apply_chat_template(messages, tools=tools, add_generation_prompt=True, tokenize=False)
        ids = tokenizer(prompt, add_special_tokens=False)['input_ids']
        base = {'id': 'workspace-turn', 'category': 'workspace', 'model': 'cached', 'prompt_sha256': identity(prompt),
                'input_token_ids': ids}
        if len(ids) > generation['max_input_tokens']:
            return {**base, 'token_ids': [], 'text': '', 'terminated': False, 'executed': False,
                    'budget_stop': 'input cap; no truncation or inference', 'seconds': 0, 'reused_prefix_tokens': 0}
        common = 0
        for cached, new in zip(state['ids'], ids):
            if cached != new:
                break
            common += 1
        common = min(common, len(ids) - 1)
        if driver.cached_tokens() > common:
            driver.crop(common)
        started, first_token = time.monotonic(), None
        tokens, current, length = [], torch.tensor([ids[common:]]), len(ids)
        for _ in range(generation['max_new_tokens']):
            token = driver.next_token(current, torch.ones((1, length), dtype=torch.long))
            if first_token is None:
                first_token = time.monotonic() - started
            tokens.append(token)
            if token in eos:
                break
            current, length = torch.tensor([[token]]), length + 1
        terminated = bool(tokens and tokens[-1] in eos)
        text = tokenizer.decode(tokens[:-1] if terminated else tokens, skip_special_tokens=False)
        state['ids'] = (ids + tokens)[:driver.cached_tokens()]
        return {**base, 'token_ids': tokens, 'text': text, 'terminated': terminated, 'executed': True,
                'seconds': time.monotonic() - started, 'first_token_seconds': first_token,
                'reused_prefix_tokens': common, 'max_rss_bytes': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024}

    return respond


def run_owner(config_dir, shards_dir, rank, world, address, port, job_path, result_path, *, timeout=60):
    """One serving owner process: the arm's owner loads it; owner 0 serves the declared episodes."""
    from datetime import timedelta
    import torch.distributed as dist

    from neuroshard.evolution import assistant_experience_run as accelerator
    from neuroshard.evolution import assistant_workflow_data as data
    from neuroshard.evolution import granite_tokenizer

    from . import granite
    from .granite_pipeline import Ring

    job = json.loads(Path(job_path).read_text())
    torch.set_num_threads(job.get('threads', 1))
    config = granite.load_config(config_dir)
    partition, manifest = granite.load_partition(config, shards_dir, rank)
    adapter = Adapter(partition, job['spec'], job['arm']) if rank == world - 1 else None
    dist.init_process_group('gloo', init_method=f'tcp://{address}:{port}', rank=rank, world_size=world,
                            timeout=timedelta(seconds=timeout))
    streams = job.get('streams')
    if streams:
        from . import granite_streams as multi

        ring = multi.Links(rank, world, config.hidden_size, job['max_tokens'])
    else:
        ring = Ring(rank, world, config.hidden_size, job['max_tokens'])
    if job.get('warm_up'):
        for enabled in ((True, False) if adapter else (None,)):
            if adapter:
                adapter.set(enabled)
            partition.warm_up()
        if adapter:
            adapter.set(True)
    result = {'rank': rank, 'shard_sha256': manifest['sha256'], 'resident_bytes': partition.resident_bytes(),
              'arm_sha256': adapter.manifest['trainable_sha256'] if adapter else None, 'streams': streams or 1,
              'warm_up': bool(job.get('warm_up'))}
    began = time.monotonic()
    try:
        if rank == 0:
            tokenizer, report = granite_tokenizer.load(job['tokenizer'], parent_digest=job.get('parent_tokenizer_digest'))
            result['tokenizer'] = report
            driver = multi.StreamDriver(partition, ring) if streams else ServingDriver(partition, ring)
            policy = job['policy']
            if 'case_ids' in job:
                by_id = {case['id']: case for case in data.cases(job['split'])}
                cases = [by_id[key] for key in job['case_ids']]
                arguments = (tokenizer, policy, cases, job['gate'],
                             lambda case: accelerator.feature_ids(tokenizer, policy, case), set(job['eos_ids']))
                if streams:
                    result['episodes'], result['episodes_seconds'] = multi.serve_episodes(driver, streams, *arguments)
                    result['peak_in_flight'] = driver.peak_in_flight
                else:
                    started = time.monotonic()
                    result['episodes'] = serve_episodes(driver, *arguments)
                    result['episodes_seconds'] = time.monotonic() - started
            for name, conversation in job.get('conversations', {}).items():
                target = driver.stream(0) if streams else driver
                target.episode(conversation['arm'])
                respond = responder(target, tokenizer, policy, set(job['eos_ids']))
                result.setdefault('conversations', {})[name] = [respond(messages, conversation['tools'])
                                                                for messages in conversation['requests']]
            result['checked_encodes'] = tokenizer.checked_encodes
            result['busy_seconds'] = driver.busy_seconds
            driver.stop()
        elif streams:
            result.update(multi.serve(partition, ring, adapter))
        else:
            from .granite_audit import OwnerLog

            log = OwnerLog(rank) if job.get('log') else None
            fault = job.get('fault') or {}
            result.update(serve(partition, ring, adapter, log, fault.get('at') if fault.get('rank') == rank else None))
            if log is not None:
                from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

                key = (Ed25519PrivateKey.from_private_bytes(bytes.fromhex(Path(job['keys'][str(rank)]).read_text().strip()))
                       if job.get('keys') else Ed25519PrivateKey.generate())
                result['log'] = log.save(Path(result_path).with_name(f'log-{rank}'), key)
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


def serve_episodes(driver, tokenizer, policy, cases, gate, feature_prompt, eos):
    """Selected complete episodes, as ``assistant_experience_eval.evaluate_arm`` serves them on one host."""
    from neuroshard.evolution import assistant_selector as selector
    from neuroshard.evolution import assistant_workflow as workflow

    rows = []
    for case in cases:
        started = time.monotonic()
        chosen = selector.choose(gate, driver.feature(feature_prompt(case)))
        selection = time.monotonic() - started
        driver.episode(chosen)
        rows.append({**workflow.execute(case, responder(driver, tokenizer, policy, eos), policy),
                     'selected': 'arm' if chosen else 'parent', 'selection_seconds': selection})
    return rows
