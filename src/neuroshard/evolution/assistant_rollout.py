"""Sampled workspace rollouts, batched across concurrent episodes on one device.

Batched sampling is not bit-reproducible across batch shapes. Acceptance does not
depend on regenerating a rollout: its transcript is replayed and scored, and its
parent likelihood is recomputed from the recorded text.
"""

from concurrent.futures import ThreadPoolExecutor
import queue
import threading
import time

from neuroshard.evolution import assistant_workflow as workflow
from neuroshard.evolution.modular_reference_execution import identity


class Batcher:
    def __init__(self, model, tokenizer, generation, *, max_batch=16, wait_seconds=0.05, device='cpu', seed=0,
                 context=None):
        self.model, self.tokenizer, self.generation = model, tokenizer, generation
        self.max_batch, self.wait_seconds, self.device = max_batch, wait_seconds, device
        # Entered around every batched generate, e.g. to select one of several adapters.
        self.context = context
        self.requests = queue.Queue()
        self.batches = []
        self._closed = threading.Event()
        self._seed = seed
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def close(self):
        self._closed.set()
        self.requests.put(None)
        self._thread.join()

    def respond(self, messages, tools):
        prompt = self.tokenizer.apply_chat_template(messages, tools=tools, add_generation_prompt=True, tokenize=False)
        ids = self.tokenizer(prompt, add_special_tokens=False)['input_ids']
        base = {'prompt_sha256': identity(prompt), 'input_token_ids': ids, 'model': 'sampled'}
        if len(ids) > self.generation['max_input_tokens']:
            return {**base, 'token_ids': [], 'text': '', 'terminated': False, 'executed': False,
                    'budget_stop': 'input cap; no truncation or inference', 'seconds': 0}
        slot = {'ids': ids, 'done': threading.Event()}
        started = time.monotonic()
        self.requests.put(slot)
        slot['done'].wait()
        if 'error' in slot:
            raise slot['error']
        return {**base, **slot['result'], 'executed': True, 'seconds': time.monotonic() - started}

    def _loop(self):
        while not self._closed.is_set():
            first = self.requests.get()
            if first is None:
                break
            batch = [first]
            deadline = time.monotonic() + self.wait_seconds
            while len(batch) < self.max_batch:
                try:
                    item = self.requests.get(timeout=max(0, deadline - time.monotonic()))
                except queue.Empty:
                    break
                if item is None:
                    self._closed.set()
                    break
                batch.append(item)
            try:
                self._generate(batch)
            except Exception as error:  # every waiting episode must observe the failure
                for slot in batch:
                    slot['error'] = error
            for slot in batch:
                slot['done'].set()

    def _generate(self, batch):
        import torch

        pad = self.tokenizer.pad_token_id
        eos = self.model.generation_config.eos_token_id
        eos = [eos] if isinstance(eos, int) else list(eos)
        width = max(len(slot['ids']) for slot in batch)
        ids = torch.full((len(batch), width), pad, dtype=torch.long)
        mask = torch.zeros((len(batch), width), dtype=torch.long)
        for row, slot in enumerate(batch):
            ids[row, width - len(slot['ids']):] = torch.tensor(slot['ids'])
            mask[row, width - len(slot['ids']):] = 1
        sampling = self.generation['temperature'] > 0
        options = ({'do_sample': True, 'temperature': self.generation['temperature'],
                    'top_p': self.generation['top_p'], 'top_k': 0} if sampling else {'do_sample': False})
        import contextlib

        with (self.context() if self.context else contextlib.nullcontext()), torch.inference_mode():
            torch.manual_seed(self._seed + len(self.batches))
            output = self.model.generate(input_ids=ids.to(self.device), attention_mask=mask.to(self.device),
                                         num_beams=1, use_cache=True, pad_token_id=pad,
                                         max_new_tokens=self.generation['max_new_tokens'], **options)
        self.batches.append(len(batch))
        for row, slot in enumerate(batch):
            tokens = output[row, width:].tolist()
            cut = next((i for i, token in enumerate(tokens) if token in eos), None)
            terminated = cut is not None
            tokens = tokens[:cut + 1] if terminated else tokens
            text = self.tokenizer.decode(tokens[:-1] if terminated else tokens, skip_special_tokens=False)
            slot['result'] = {'token_ids': tokens, 'text': text, 'terminated': terminated}


def rollouts(jobs, respond, *, workers, progress=None):
    """Execute (case, policy, sample) jobs concurrently; each episode stays sequential."""
    lock, done = threading.Lock(), [0]

    def run(job):
        case, policy, sample = job
        result = workflow.execute(case, respond, policy)
        if progress:
            with lock:
                done[0] += 1
                progress(done[0], len(jobs))
        return {'case_id': case['id'], 'sample': sample, 'policy_sha256': identity(policy), 'result': result}

    with ThreadPoolExecutor(max_workers=workers) as pool:
        return list(pool.map(run, jobs))
