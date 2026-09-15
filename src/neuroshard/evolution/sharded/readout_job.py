"""Question-only inference through three frozen owners and one learned decoder."""
from datetime import timedelta
import json
import os
from pathlib import Path
import time

import numpy as np
from safetensors.numpy import save_file
import torch
import torch.distributed as dist
from transformers import LlamaConfig

from .. import incremental_capacity as base, incremental_facts as facts
from .. import readout_experiment as contract, reference, reference_data as data
from . import incremental_state, portable, readout
from .incremental_job import tokenizer_for
from .model import Partition, batch_tensors, weighted_loss
from .wire import Wire


class Network:
    def __init__(self, shard, wire, tokenizer, config):
        self.shard, self.wire, self.tokenizer, self.config = shard, wire, tokenizer, config
        self.device = shard.device_name if shard is not None else 'cpu'

    @torch.no_grad()
    def forward(self, token_ids, feature=False):
        if (not token_ids or len(token_ids) > self.config.max_position_embeddings
                or any(type(n) is not int or not 0 <= n < self.config.vocab_size for n in token_ids)):
            raise ValueError('Invalid bounded causal input')
        rank = self.wire.rank
        shape = (1, len(token_ids), self.config.hidden_size)
        if rank < 3:
            ids = torch.tensor([token_ids], dtype=torch.long, device=self.device)
            mask = torch.ones_like(ids)
            incoming = ids if rank == 0 else self.wire.receive(rank - 1, shape, self.device)
            with reference.autocast(self.device):
                outgoing = self.shard(incoming, mask)
            if rank < 2:
                self.wire.send(outgoing, rank + 1)
            elif feature:
                # The decoder owner receives no token IDs, answer labels or logits.
                self.wire.send(outgoing[:, -1:, :], 3)
            else:
                self.wire.send(outgoing, 0)
        if feature and rank == 3:
            return self.wire.receive(2, (1, 1, self.config.hidden_size), 'cpu')[0, 0].numpy()
        if not feature and rank == 0:
            return self.wire.receive(2, shape, self.device)
        return None

    def feature(self, question):
        return self.forward(readout.prompt_tokens(self.tokenizer, question), feature=True)

    @torch.no_grad()
    def generate(self, question, max_tokens, json_prefix=False):
        ids = self.tokenizer.apply_chat_template([{'role': 'user', 'content': question}],
            tokenize=True, add_generation_prompt=True)
        output = list(readout.COMMON_PREFIX) if json_prefix else []
        ids += output
        if len(ids) + max_tokens > self.config.max_position_embeddings:
            raise ValueError('No silent truncation in parent fallback')
        for _ in range(max_tokens):
            final = self.forward(ids)
            token = None
            if self.wire.rank == 0:
                with reference.autocast(self.device):
                    token = int(self.shard.logits(final[:, -1:]).float().argmax(-1)[0, 0])
            token = self.wire.exchange(token)[0]
            if type(token) is not int or not 0 <= token < self.config.vocab_size:
                raise ValueError('Invalid parent token')
            output.append(token)
            ids.append(token)
            if token == self.tokenizer.eos_token_id:
                break
        return self.tokenizer.decode(output, skip_special_tokens=True)

    def answer(self, question, predictor, roster, max_tokens):
        if not readout.route(question, roster):
            return self.generate(question, max_tokens), 'parent'
        feature = self.feature(question)
        value = predictor.predict(feature) if self.wire.rank == 3 else None
        value = self.wire.exchange(value)[3]
        if not isinstance(value, str):
            raise ValueError('Decoder returned an invalid value')
        return json.dumps({'answer': value}, separators=(',', ':')), 'readout'

    @torch.no_grad()
    def loss(self, row):
        final = self.forward(row['input_ids'])
        value = None
        if self.wire.rank == 0:
            _, labels, _, weights = batch_tensors([row], self.device)
            with reference.autocast(self.device):
                value = float(weighted_loss(self.shard.logits(final), labels, torch.ones_like(weights))) / row['targets']
        value = self.wire.exchange(value)[0]
        return {'id': row['id'], 'targets': row['targets'], 'loss': value}


def run(args):
    plan, prepared = contract.validate()
    if any(os.environ.get(key) != str(plan['blas_threads'])
           for key in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS')):
        raise ValueError('Set the frozen BLAS thread count before process startup')
    rank, world = int(os.environ['RANK']), int(os.environ['WORLD_SIZE'])
    if world != 4 or not 0 <= rank < world:
        raise ValueError('Use three frozen model owners and one readout owner')
    final = args.command == 'final'
    if final:
        selection = json.loads(base.committed(contract.SELECTION))
        if (not selection['eligible'] or selection['prepared'] != data.identity(prepared)
                or selection['plan'] != data.identity(plan)):
            raise ValueError('Finals need a committed eligible development selection')
    elif contract.SELECTION.exists():
        raise ValueError('Development is closed after selection')
    args.home.mkdir(parents=True, exist_ok=False)
    parent = json.loads(args.parent.read_bytes())
    if data.identity(parent) != plan['parent']:
        raise ValueError('Wrong frozen parent')
    tokenizer = tokenizer_for(plan, args.seed)
    runtime = reference.configure('cuda' if rank < 3 else 'cpu', plan['threads'])
    runtime['allocator'] = os.environ['PYTORCH_CUDA_ALLOC_CONF']
    expected = plan['runtime'] if rank < 3 else plan['readout_runtime']
    if {key: runtime[key] for key in expected} != expected:
        raise ValueError('Readout runtime differs from the frozen numerical profile')
    config = LlamaConfig(**parent['config'])
    config._attn_implementation = 'sdpa'
    shard = None
    if rank < 3:
        shard = Partition(config, plan['boundaries'], rank, 'cuda', plan['parameter_limit'])
        inherited = incremental_state.records(parent)
        with torch.no_grad():
            for name, parameter in shard.named_owned_parameters():
                value = incremental_state.tensor_values(portable.tensor_path(args.objects, inherited[name]['sha256']), inherited[name])
                parameter.copy_(value['weight'])
                parameter.requires_grad_(False)
                del value
        shard.eval()
    dist.init_process_group('gloo', timeout=timedelta(seconds=600))
    wire = Wire(rank, world)
    binding = {'plan': data.identity(plan), 'prepared': data.identity(prepared), 'parent': plan['parent']}
    started = time.monotonic()
    try:
        declarations = wire.exchange({'binding': binding, 'rank': rank, 'profile': expected})
        if any(row['binding'] != binding or row['rank'] != index or row['profile'] !=
                (plan['runtime'] if index < 3 else plan['readout_runtime']) for index, row in enumerate(declarations)):
            raise ValueError('Owners disagree on the numerical experiment')
        data.save(args.home / 'started.json', {**binding, 'runtime': runtime, 'rank': rank,
            'owned_parameters': shard.resident_parameters if shard is not None else 0})
        net = Network(shard, wire, tokenizer, config)
        roster = prepared['roster']
        predictor, root = None, None
        if not final:
            records = contract.rows(prepared, args.inputs, 'train')
            records = [row for row in records if row.get('task', {}).get('family') == 'directory']
            if len(records) != 5120:
                raise ValueError('Train on all and only the admitted question-answer examples')
            features = np.empty((len(records), config.hidden_size), dtype=np.float32) if rank == 3 else None
            began = time.monotonic()
            for index, row in enumerate(records):
                if time.monotonic() - started > plan['max_seconds']:
                    raise TimeoutError('Bounded feature-production deadline')
                question = row['messages'][0]['content']
                prefix = readout.prompt_tokens(tokenizer, question)
                first = next(i for i, label in enumerate(row['labels']) if label != -100)
                if row['input_ids'][:first + 3] != prefix or first + 3 >= len(row['input_ids']):
                    raise ValueError('Training feature is not the exact question-only causal prefix')
                vector = net.feature(question)
                if rank == 3:
                    features[index] = vector
                if index % 256 == 0:
                    print(json.dumps({'event': 'features', 'rank': rank, 'count': index + 1}), flush=True)
            feature_seconds = time.monotonic() - began
            if rank == 3:
                save_file({'features': features}, args.home / 'features.safetensors')
                began = time.monotonic()
                predictor = readout.fit(features, [row['task']['expected'] for row in records], plan['ridge_alpha'])
                fit_seconds = time.monotonic() - began
                root = predictor.write(args.home / 'decoder', binding, roster)
                # Fresh deserialization, with exact binding, before any evaluation.
                predictor, reloaded_roster = readout.Predictor.read(args.home / 'decoder', root, binding)
                if roster != reloaded_roster:
                    raise ValueError('Reload changed the domain roster')
                fit_correct = sum(predictor.predict(vector) == row['task']['expected'] for vector, row in zip(features, records))
                data.save(args.home / 'training.json', {'decoder_root': root, 'fit_correct': fit_correct,
                    'count': len(records), 'feature_seconds': feature_seconds, 'fit_seconds': fit_seconds,
                    'features_sha256': data.sha256(args.home / 'features.safetensors'),
                    'parameter_count': sum(value.size for value in predictor.tensors.values()),
                    'tokens_issued': 0})
                del features
            root = wire.exchange(root)[3]
        else:
            root = selection['decoder_root']
            if rank == 3:
                predictor, roster = readout.Predictor.read(args.decoder, root, binding)
                if roster != prepared['roster']:
                    raise ValueError('Selected decoder changed its roster')
        before, after = {}, {}
        prefix = 'test' if final else 'dev'
        roles = tuple(prefix + '-' + suffix for suffix in ('knowledge', 'skills', 'conversation'))
        for role in roles:
            rows = contract.rows(prepared, args.inputs, role)
            before[role], after[role] = ({'answers': [], 'losses': []} for _ in range(2))
            for index, row in enumerate(rows):
                if time.monotonic() - started > plan['max_seconds']:
                    raise TimeoutError('Bounded readout execution deadline')
                question = row['messages'][0]['content']
                if role.endswith('conversation'):
                    if readout.route(question, roster):
                        raise ValueError('The domain selector captured a conversation retention probe')
                    before[role]['losses'].append(net.loss(row))
                    # Explicit dispatcher check before recomputing unchanged parent loss.
                    after[role]['losses'].append(net.loss(row))
                else:
                    known = role.endswith('knowledge')
                    cap = plan['generation']['knowledge' if known else 'skills']
                    began = time.monotonic()
                    text = net.generate(question, cap, json_prefix=known)
                    before[role]['answers'].append({'id': row['id'], 'text': text,
                        'route': 'parent', 'seconds': time.monotonic() - began})
                    began = time.monotonic()
                    text, selected = net.answer(question, predictor, roster, cap)
                    after[role]['answers'].append({'id': row['id'], 'text': text,
                        'route': selected, 'seconds': time.monotonic() - began})
                if (index + 1) % 32 == 0:
                    print(json.dumps({'event': 'evaluated', 'rank': rank, 'role': role, 'count': index + 1}), flush=True)
            data.save(args.home / (role + '.json'), {'before': before[role], 'after': after[role]})
        rows = {role: contract.rows(prepared, args.inputs, role) for role in roles}
        outcome = base.decision(plan, rows, before, after, not final)
        # Old correct skills are checked individually; exact fallback is a separate stronger check.
        exact = all(a['text'] == b['text'] and b['route'] == 'parent'
            for a, b in zip(before[prefix + '-skills']['answers'], after[prefix + '-skills']['answers']))
        outcome['checks']['exact_skill_fallback'] = exact
        outcome['checks']['question_only_routing'] = all(row['route'] == 'readout' for row in after[prefix + '-knowledge']['answers'])
        outcome['passed'] = all(outcome['checks'].values())
        answer_identity = data.identity({'before': {role: {kind: [{key: value for key, value in row.items() if key != 'seconds'}
            for row in values] for kind, values in result.items()} for role, result in before.items()},
            'after': {role: {kind: [{key: value for key, value in row.items() if key != 'seconds'}
            for row in values] for kind, values in result.items()} for role, result in after.items()}})
        if any(value != answer_identity for value in wire.exchange(answer_identity)):
            raise ValueError('Distributed owners disagree on generated outcomes')
        data.save(args.home / 'result.json', {**binding, 'rank': rank, 'decoder_root': root,
            'development': not final, 'decision': outcome, 'answer_identity': answer_identity,
            'seconds': time.monotonic() - started, 'sent_tensor_bytes': wire.sent_tensor_bytes,
            'tokens_issued': 0, 'native_activated': False})
    finally:
        dist.destroy_process_group()
