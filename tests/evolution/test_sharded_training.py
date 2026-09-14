"""Compare partitioned AdamW with full autograd, then restart new processes."""
from datetime import timedelta
import json
from pathlib import Path
import random
import sys

import numpy as np
import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from transformers import LlamaConfig, LlamaForCausalLM
from safetensors.torch import load_file, save_file

from neuroshard.evolution import reference
from neuroshard.evolution.reference_data import identity
from neuroshard.evolution.sharded import checkpoint as saved
from neuroshard.evolution.sharded.model import Partition, batch_tensors, weighted_loss
from neuroshard.evolution.sharded.training import train_step, generate
from neuroshard.evolution.sharded.wire import Wire

RECIPE = {'steps': 4, 'warmup_steps': 1, 'learning_rate': .0003,
          'weight_decay': .01, 'clip_norm': .1}
BOUNDARIES = [0, 2, 4]


def config():
    value = LlamaConfig(vocab_size=32, hidden_size=16, intermediate_size=32,
                        num_hidden_layers=4, num_attention_heads=2, num_key_value_heads=1,
                        max_position_embeddings=64, tie_word_embeddings=True, attention_dropout=0.)
    value._attn_implementation = 'eager'
    return value


def records():
    return [{'id': str(i), 'input_ids': [1, 3, 4] + [7+i]*(i+1) + [2],
             'labels': [-100]*3 + [7+i]*(i+1) + [2], 'targets': i+2,
             'loss_weight': 8 if i % 2 else 1} for i in range(5)]


def control_step(model, optimizer, rows, index):
    optimizer.zero_grad(set_to_none=True)
    for group in optimizer.param_groups:
        group['lr'] = reference.learning_rate(RECIPE, index)
    denominator = sum(r['targets']*r['loss_weight'] for r in rows)
    total = 0.
    for offset in range(0, len(rows), 2):
        ids, labels, mask, weights = batch_tensors(rows[offset:offset+2], 'cpu')
        loss = weighted_loss(model(input_ids=ids, attention_mask=mask, use_cache=False).logits, labels, weights)
        total += float(loss.detach())
        (loss/denominator).backward()
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), RECIPE['clip_norm'])
    optimizer.step()
    return total/denominator, float(norm)


def worker(rank, rendezvous, folder, resume):
    torch.set_num_threads(1)
    torch.manual_seed(31)
    random.seed(31+rank)
    np.random.seed(31+rank)
    reference_model = LlamaForCausalLM(config()).float()
    reference_optimizer = reference.optimizer_for(reference_model, RECIPE)
    # A shard must never instantiate a full model, even temporarily.
    def forbidden(*args, **kwargs):
        raise AssertionError('Partition attempted whole-model allocation')
    LlamaForCausalLM.__init__ = forbidden
    shard = Partition(config(), BOUNDARIES, rank, parameter_limit=10000)
    expected = dict(reference_model.named_parameters())
    with torch.no_grad():
        for name, p in shard.named_owned_parameters():
            p.copy_(expected[name])
    optimizer = reference.optimizer_for(shard, RECIPE)
    dist.init_process_group('gloo', init_method='file://'+rendezvous,
                            rank=rank, world_size=2, timeout=timedelta(seconds=45))
    wire = Wire(rank, 2)
    home = Path(folder)/f'rank-{rank}'
    binding = identity({'recipe': RECIPE, 'boundaries': BOUNDARIES, 'rows': records()})
    parent = None
    try:
        if resume:
            common = json.loads((home/'commit-000002.json').read_bytes())
            start = saved.load(home, shard, optimizer, common, binding)
            parent = identity(common)
            for i in range(start):
                control_step(reference_model, reference_optimizer, records() if i%2==0 else list(reversed(records())), i)
        else:
            start = 0
            parent = identity(saved.commit(home, shard, optimizer, wire, binding, 0, None))
        for index in range(start, 4):
            batch = records() if index%2==0 else list(reversed(records()))
            observed = train_step(shard, optimizer, wire, batch, RECIPE, index, 2)
            loss, norm = control_step(reference_model, reference_optimizer, batch, index)
            assert observed['loss'] == pytest.approx(loss, rel=2e-6)
            assert observed['gradient_norm'] == pytest.approx(norm, rel=2e-6)
            for name, parameter in shard.named_owned_parameters():
                torch.testing.assert_close(parameter, expected[name], rtol=2e-5, atol=2e-7)
                for key, value in reference_optimizer.state[expected[name]].items():
                    torch.testing.assert_close(optimizer.state[parameter][key], value, rtol=2e-5, atol=2e-7)
            if index+1 in (2, 4):
                common = saved.commit(home, shard, optimizer, wire, binding, index+1, parent)
                parent = identity(common)
        reference_model.eval()
        tokens = [1, 5, 7]
        with torch.no_grad():
            expected_tokens = []
            for _ in range(3):
                token = int(reference_model(torch.tensor([tokens+expected_tokens])).logits[0, -1].argmax())
                expected_tokens.append(token)
        generated = generate(shard, wire, tokens, 3, -1)
        assert generated == expected_tokens
        (home/('recovered.json' if resume else 'result.json')).write_text(json.dumps({
            'root': parent, 'parameters': shard.resident_parameters,
            'whole_model_parameters': sum(p.numel() for p in reference_model.parameters()),
            'generated': generated}))
    finally:
        dist.destroy_process_group()


def test_partitioned_adam_and_generation_match_reference_and_new_process_recovery(tmp_path):
    mp.spawn(worker, args=(str(tmp_path/'first-rendezvous'), str(tmp_path), False), nprocs=2, join=True)
    before = [json.loads((tmp_path/f'rank-{r}/commit-000004.json').read_bytes()) for r in range(2)]
    # Preserve the first run; restore to a common earlier checkpoint on restart.
    for rank in range(2):
        home = tmp_path/f'rank-{rank}'
        (home/'shard-000004').rename(home/'original-shard-000004')
        (home/'commit-000004.json').rename(home/'original-commit-000004.json')
    mp.spawn(worker, args=(str(tmp_path/'second-rendezvous'), str(tmp_path), True), nprocs=2, join=True)
    after = [json.loads((tmp_path/f'rank-{r}/commit-000004.json').read_bytes()) for r in range(2)]
    assert before[0] == before[1] == after[0] == after[1]
    observations = [json.loads((tmp_path/f'rank-{r}/result.json').read_bytes()) for r in range(2)]
    assert sum(r['parameters'] for r in observations) == observations[0]['whole_model_parameters']
    assert all(r['parameters'] < r['whole_model_parameters'] for r in observations)
    shard = Partition(config(), BOUNDARIES, 0)
    optimizer = reference.optimizer_for(shard, RECIPE)
    with pytest.raises(ValueError, match='different computation'):
        saved.load(tmp_path/'rank-0', shard, optimizer, after[0], 'wrong-binding')
    directory = tmp_path/'rank-0/shard-000004'
    tensor = next(directory.glob('tensor-*.safetensors'))
    body = bytearray(tensor.read_bytes()); body[-1] ^= 1; tensor.write_bytes(body)
    with pytest.raises(ValueError, match='corrupted'):
        saved.load(tmp_path/'rank-0', shard, optimizer, after[0], after[0]['binding'])

    # Even a self-consistent file/manifest must not resume the wrong Adam cursor.
    tensor.write_bytes(bytes(bytearray(body[:-1])+bytes([body[-1]^1])))
    values = load_file(tensor)
    values['step'].fill_(3)
    save_file(values, tensor)
    meta_path = directory/'manifest.json'
    meta = json.loads(meta_path.read_bytes())
    meta['files'][tensor.name]['sha256'] = saved.sha256(tensor)
    meta_path.write_text(json.dumps(meta))
    inconsistent = json.loads(json.dumps(after[0]))
    inconsistent['shards'][0]['manifest'] = identity(meta)
    with pytest.raises(ValueError, match='global cursor'):
        saved.load(tmp_path/'rank-0', shard, optimizer, inconsistent, inconsistent['binding'])
