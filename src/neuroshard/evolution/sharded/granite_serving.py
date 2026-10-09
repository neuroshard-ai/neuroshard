"""Serve the complete learned assistant across Granite owners.

Owner 0 renders every request, selects the parent or the arm once per episode
from the parent's final-layer feature, and decodes greedily; the owner holding
the arm, an added module or an update of declared projections, switches it on or
off for the episode. Every owner keeps one episode
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

from .granite_pipeline import FORWARD, RESET, STOP

CROP, ADAPTER, FEATURE = 9, 10, 11
ROUTER_FEATURE, ROUTER_CROP, ROUTER_RESET, ROUTE = 12, 13, 14, 15


def plain(master, dtype):
    """An updated projection as single-host serving stores it: a plain Linear in the backbone dtype."""
    linear = torch.nn.Linear(master.weight.shape[1], master.weight.shape[0], bias=False, dtype=dtype,
                             device=master.weight.device)
    with torch.no_grad():
        linear.weight.copy_(master.weight.to(dtype))
    linear.weight.requires_grad_(False)
    return linear


class Adapter:
    """A saved arm on its owner, switchable per episode without touching the backbone.

    The addition wraps the parent's projections; the update replaces them with plain projections,
    as ``trainer.serving`` stores them on one host. Switching off restores the parent's own modules.
    """

    def __init__(self, partition, spec, directory):
        from safetensors.torch import load_file

        from .granite_training import attach

        directory = Path(directory)
        manifest = json.loads((directory / 'manifest.json').read_text())
        if (manifest['arm'] not in ('addition', 'update')
                or hashlib.sha256((directory / 'trainable.safetensors').read_bytes()).hexdigest()
                != manifest['trainable_sha256']):
            raise ValueError('arm checkpoint differs from its manifest')
        sites = [(partition.layers[index - partition.begin].self_attn, name)
                 for index in spec['layers'] for name in ('q_proj', 'v_proj')]
        parents = [getattr(attention, name) for attention, name in sites]
        trainable = attach(partition, manifest['arm'], spec)
        saved = load_file(str(directory / 'trainable.safetensors'))
        if set(saved) != set(trainable):
            raise ValueError('arm checkpoint tensor inventory differs')
        with torch.no_grad():
            for name, value in trainable.items():
                value.copy_(saved[name])
                value.requires_grad_(False)
        self.manifest, self.wrapped = manifest, []
        for (attention, name), parent in zip(sites, parents):
            module = getattr(attention, name)
            if manifest['arm'] == 'update':
                module = plain(module, parent.weight.dtype)
                setattr(attention, name, module)
            self.wrapped.append((attention, name, module, parent))
        self.enabled = True

    def set(self, enabled):
        for attention, name, module, parent in self.wrapped:
            setattr(attention, name, module if enabled else parent)
        self.enabled = enabled


class AdapterBank:
    """Accepted additions on the accepted update, retaining the unchanged parent for fallback."""

    def __init__(self, partition, spec, base, additions):
        self.update = Adapter(partition, spec, base)
        self.routes = {}
        for index, directory in sorted(additions.items(), key=lambda row: int(row[0])):
            added = Adapter(partition, spec, directory)
            if added.manifest['arm'] != 'addition':
                raise ValueError('a route must be an addition on the accepted update')
            self.routes[int(index)] = added
            added.set(False)
        if not self.routes:
            raise ValueError('an adapter bank needs at least one route')
        self.route, self.enabled = min(self.routes), False
        self.set(False)

    def select(self, index):
        if index not in self.routes:
            raise ValueError('unknown accepted route')
        self.route = index
        self.set(self.enabled)

    def set(self, enabled):
        self.update.set(bool(enabled))
        if enabled:
            self.routes[self.route].set(True)
        self.enabled = bool(enabled)


def serve(partition, ring, adapter=None, log=None, fault=None, link=None):
    """Owner loop for ranks after 0: episode caches, prefix crops, arm switching and parent features.

    ``log`` records every command for replay audits; ``fault`` (a forward-message
    index) perturbs one sent tensor, standing in for a cheating owner in tests.
    With a job's ``link`` (``granite_audit.Link``) a received message is used only
    with its sender's signature over the link transcript, and every sent message
    carries this owner's.
    """
    from transformers import DynamicCache

    cache, steps, busy, forwards = DynamicCache(), 0, 0.0, 0
    caches, route, router_cache = {1: cache}, 1, DynamicCache()
    while True:
        op, value = ring.command(0)
        if op == STOP:
            return {'steps': steps, 'busy_seconds': busy}
        if op in (RESET, CROP, ADAPTER, ROUTE, ROUTER_CROP, ROUTER_RESET):
            if log is not None:
                log.command(op, value)
            if link is not None:
                link.command(op, value)
        if op == RESET:
            cache = DynamicCache()
            caches[route] = cache
        elif op == CROP:
            cache.crop(value)
        elif op == ADAPTER:
            if adapter is not None:
                adapter.set(bool(value))
        elif op == ROUTE:
            if value not in (1, 2):
                raise ValueError('unknown accepted route')
            route = value
            cache = caches.setdefault(route, DynamicCache())
            if adapter is not None:
                adapter.select(route)
        elif op == ROUTER_RESET:
            router_cache = DynamicCache()
        elif op == ROUTER_CROP:
            router_cache.crop(value)
        elif op in (FORWARD, FEATURE, ROUTER_FEATURE):
            steps += op == FORWARD
            hidden = ring.receive(ring.rank - 1, value)
            attested = link.receive(op, value, hidden, ring.receive_signature(ring.rank - 1)) if link else None
            began = time.monotonic()
            with torch.inference_mode():
                if op in (FORWARD, ROUTER_FEATURE):
                    active = router_cache if op == ROUTER_FEATURE else cache
                    mask = torch.ones((1, active.get_seq_length() + value), dtype=torch.long)
                    out = partition(hidden, mask, active)
                else:
                    out = partition(hidden, None, DynamicCache())
            busy += time.monotonic() - began
            last = ring.rank == ring.world - 1
            sent = out[:, -1:] if last and op != ROUTER_FEATURE else out
            if fault is not None and forwards == fault:
                sent = sent.clone()
                sent.view(torch.int16).view(-1)[0] ^= 1
            forwards += 1
            if log is not None:
                log.forward(op, value, hidden, sent, attested)
            ring.send(sent, (ring.rank + 1) % ring.world)
            if link is not None:
                ring.send_signature(link.send(op, value, sent), (ring.rank + 1) % ring.world)
        else:
            raise ValueError('unknown serving command')


class ServingDriver:
    """Owner 0: episode control, the parent feature and cached decoding steps.

    With a job's ``link`` the driver signs everything it sends under the job's session key
    and accepts results only with the last owner's signature; ``evidence`` is the latest one.
    """

    def __init__(self, partition, ring, link=None):
        from transformers import DynamicCache

        if partition.rank != 0 or ring.rank != 0:
            raise ValueError('owner 0 drives serving')
        self.partition, self.ring, self.cache_type, self.link = partition, ring, DynamicCache, link
        self.cache = DynamicCache()
        self.caches, self.route_index, self.router_cache = {1: self.cache}, 1, DynamicCache()
        self.router_prefix = None
        self.busy_seconds = 0.0

    def control(self, op, value=0):
        """A command that changes owner state: one step of every link transcript."""
        self.ring.command(op, value)
        if self.link is not None:
            self.link.command(op, value)

    def exchange(self, op, value, hidden):
        """Send one message around the ring; returns what the last owner sends back."""
        self.ring.send(hidden, 1)
        if self.link is not None:
            self.ring.send_signature(self.link.send(op, value, hidden), 1)
        back = self.ring.receive(self.ring.world - 1, value if op == ROUTER_FEATURE else 1)
        if self.link is not None:
            self.link.receive(op, value, back, self.ring.receive_signature(self.ring.world - 1))
        return back

    def evidence(self):
        """The last owner's latest signature over what it sent back: the user's evidence against it."""
        return self.link.received if self.link is not None else None

    def feature(self, ids):
        """The frozen parent's final-layer state at the last prompt position, as ``boundary_feature`` computes it."""
        self.control(ADAPTER, 0)
        self.ring.command(FEATURE, len(ids))
        tokens = torch.tensor([ids])
        began = time.monotonic()
        with torch.inference_mode():
            hidden = self.partition(self.partition.embed(tokens), None, self.cache_type())
        self.busy_seconds += time.monotonic() - began
        back = self.exchange(FEATURE, len(ids), hidden)
        with torch.inference_mode():
            return self.partition.norm(back)[0, -1].float().tolist()

    def episode(self, arm):
        self.control(RESET)
        self.control(ADAPTER, int(arm))
        self.cache = self.cache_type()
        self.caches[self.route_index] = self.cache

    def route(self, index, arm=True):
        self.control(ROUTE, index)
        self.route_index = index
        self.cache = self.caches.setdefault(index, self.cache_type())
        self.control(ADAPTER, int(arm))

    def message_feature(self, tokenizer, policy, user):
        """The accepted router's cached-prefix, mean-message feature, computed across owners."""
        from neuroshard.evolution import assistant_routing as routing

        self.control(ADAPTER, 0)
        ids = routing.turn_ids(tokenizer, policy, user)
        if self.router_prefix is None:
            first = routing.turn_ids(tokenizer, policy, 'Schedule')
            second = routing.turn_ids(tokenizer, policy, 'Create')
            common = 0
            for left, right in zip(first, second):
                if left != right:
                    break
                common += 1
            self.router_prefix = first[:common]
            if not self.router_prefix:
                raise ValueError('router prefix is empty')
            self.router_forward(self.router_prefix)
        length = len(self.router_prefix)
        if ids[:length] != self.router_prefix or len(ids) <= length:
            raise ValueError('router turn does not extend the accepted prefix')
        try:
            states = self.router_forward(ids[length:])
            return states[0].float().mean(dim=0).tolist()
        finally:
            self.control(ROUTER_CROP, length)
            self.router_cache.crop(length)

    def router_forward(self, ids):
        self.ring.command(ROUTER_FEATURE, len(ids))
        tokens = torch.tensor([ids])
        mask = torch.ones((1, self.router_cache.get_seq_length() + len(ids)), dtype=torch.long)
        with torch.inference_mode():
            hidden = self.partition(self.partition.embed(tokens), mask, self.router_cache)
            back = self.exchange(ROUTER_FEATURE, len(ids), hidden)
            return self.partition.norm(back)

    def cached_tokens(self):
        return self.cache.get_seq_length()

    def crop(self, length):
        self.control(CROP, length)
        self.cache.crop(length)

    def step(self, tokens, mask):
        self.ring.command(FORWARD, tokens.shape[1])
        began = time.monotonic()
        with torch.inference_mode():
            hidden = self.partition(self.partition.embed(tokens), mask, self.cache)
        self.busy_seconds += time.monotonic() - began
        back = self.exchange(FORWARD, tokens.shape[1], hidden)
        began = time.monotonic()
        with torch.inference_mode():
            logits = self.partition.logits(back)
        self.busy_seconds += time.monotonic() - began
        return logits

    def next_token(self, tokens, mask):
        return greedy(self.step(tokens, mask))

    def stop(self):
        self.ring.command(STOP)


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


def owner_key(job, rank):
    """This owner's Ed25519 key from the job's ``keys`` (rank to key file), or None if the job names none."""
    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

    path = (job.get('keys') or {}).get(str(rank))
    return Ed25519PrivateKey.from_private_bytes(bytes.fromhex(Path(path).read_text().strip())) if path else None


def job_link(job, rank, world):
    """This owner's signed links for the job's ``session``.

    The session names the chain, job and request, the user's session key (owner 0) and the
    bonded owners' log keys in shard order; the owner's own key must be the one it names.
    """
    from .granite_audit import Link, public_hex

    session, key = job['session'], owner_key(job, rank)
    senders = [session['session_key'], *session['log_keys']]
    if len(senders) != world or key is None or public_hex(key) != senders[rank]:
        raise ValueError('this owner key is not the one the job names')
    return Link(session, rank, world, key, senders[(rank - 1) % world])


def run_owner(config_dir, shards_dir, rank, world, address, port, job_path, result_path, *, timeout=60):
    """One serving owner process: the arm's owner loads it; owner 0 serves the declared episodes.

    With a ``session`` every serving link is signed (see ``serve``), and owners' logs are bound
    to that job; owner 0's result keeps the last owner's latest signature as ``attestation``.
    """
    from datetime import timedelta
    import torch.distributed as dist

    from neuroshard.evolution import assistant_experience_run as accelerator
    from neuroshard.evolution import assistant_workflow_data as data
    from neuroshard.evolution import granite_tokenizer

    from . import granite
    from .granite_pipeline import Ring

    job = json.loads(Path(job_path).read_text())
    torch.set_num_threads(job.get('threads', 1))
    streams = job.get('streams')
    if streams and job.get('session'):
        raise ValueError('signed serving links support one stream')
    link = job_link(job, rank, world) if job.get('session') else None
    config = granite.load_config(config_dir)
    partition, manifest = granite.load_partition(config, shards_dir, rank)
    adapter = Adapter(partition, job['spec'], job['arm']) if rank == world - 1 else None
    dist.init_process_group('gloo', init_method=f'tcp://{address}:{port}', rank=rank, world_size=world,
                            timeout=timedelta(seconds=timeout))
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
            driver = multi.StreamDriver(partition, ring) if streams else ServingDriver(partition, ring, link)
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
            fault, failure = job.get('fault') or {}, None
            try:
                result.update(serve(partition, ring, adapter, log, fault.get('at') if fault.get('rank') == rank else None,
                                    link))
            except Exception as error:
                failure = error
            # A bound log is kept even when serving stops early: its signed prefix is the work done.
            if log is not None and (failure is None or link is not None):
                from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

                key = owner_key(job, rank) or Ed25519PrivateKey.generate()
                result['log'] = log.save(Path(result_path).with_name(f'log-{rank}'), key, job.get('session'))
            if failure is not None:
                raise failure
        result['completed'] = True
    except Exception as error:
        result.update(completed=False, error=f'{type(error).__name__}: {error}')
    finally:
        if link is not None and rank == 0:
            result['attestation'] = link.received
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
