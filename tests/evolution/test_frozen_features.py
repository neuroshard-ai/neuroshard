"""A local feature learner must reproduce the independent distributed oracle."""
from datetime import timedelta
import copy
import json
import math
from pathlib import Path

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from transformers import LlamaConfig, LlamaForCausalLM
from safetensors.torch import save_file, load_file

from neuroshard.evolution import reference
from neuroshard.evolution.reference_data import identity, sha256
from neuroshard.evolution.sharded import incremental, features
import pytest
from neuroshard.evolution.sharded.model import Partition, batch_tensors
from neuroshard.evolution.sharded.wire import Wire

ROOT = Path(__file__).resolve().parent
RECIPE = {'steps': 8, 'warmup_steps': 1, 'learning_rate': .002,
          'weight_decay': .01, 'clip_norm': .3}


def config(layers):
    value = LlamaConfig(vocab_size=47, hidden_size=24, intermediate_size=48,
        num_hidden_layers=layers, num_attention_heads=4, num_key_value_heads=2,
        max_position_embeddings=64, tie_word_embeddings=True, attention_dropout=0.)
    value._attn_implementation = 'eager'
    return value


def records():
    return [{'id': str(index), 'input_ids': [1, 4 + index, 9] + [12 + index] * (index + 1) + [2],
        'labels': [-100] * 3 + [12 + index] * (index + 1) + [2],
        'targets': index + 2, 'loss_weight': 1. / (index + 2), 'distill': index % 2 == 0}
        for index in range(5)]


def worker(rank, arm, folder, rendezvous):
    torch.set_num_threads(1)
    torch.manual_seed(39)
    parent = LlamaForCausalLM(config(4)).float().eval()
    cut, layers = (4, 6) if arm == 'append' else (2, 4)
    boundaries = [0, cut // 2, cut, layers]
    source = dict(parent.named_parameters())
    shard = Partition(config(layers), boundaries, rank)
    with torch.no_grad():
        for name, parameter in shard.named_owned_parameters():
            parts = name.split('.')
            if parts[:2] == ['model', 'layers'] and int(parts[2]) >= 4:
                parts[2] = '3'
                parameter.copy_(source['.'.join(parts)])
                if name.endswith(('self_attn.o_proj.weight', 'mlp.down_proj.weight')):
                    parameter.zero_()
            else:
                parameter.copy_(source[name])
    optimizer = incremental.configure(shard, cut, RECIPE)
    original = incremental.reference_tail(shard, cut) if arm == 'control' else None
    if rank == 2:
        cached = copy.deepcopy(shard)
        cached_optimizer = incremental.configure(cached, cut, RECIPE)
        teacher_tail = copy.deepcopy(shard).eval().requires_grad_(False) if arm == 'control' else None
        head = features.FrozenHead(parent.config, parent.model.embed_tokens.weight.detach(), parent.model.norm.weight.detach())
        full_size = sum(p.numel() for p in parent.parameters())
        resident = sum(p.numel() for p in cached.parameters()) + sum(p.numel() for p in head.parameters())
        assert resident < full_size
    # A serving/training owner need not retain this test oracle's full model.
    del parent, source
    dist.init_process_group('gloo', init_method='file://' + rendezvous, rank=rank,
        world_size=3, timeout=timedelta(seconds=60))
    wire = Wire(rank, 3)
    try:
        packets = []
        rows = records()
        for offset in range(0, len(rows), 2):
            subset = rows[offset:offset + 2]
            ids, labels, mask, weights = batch_tensors(subset, 'cpu')
            with torch.no_grad():
                incoming, outgoing, reference_hidden, final = incremental.forward(shard, wire, ids, mask, cut, original)
                if rank == 2:
                    teacher = teacher_tail(incoming, mask) if teacher_tail is not None else incoming
                    packet = {'prefix': incoming.contiguous(), 'reference': teacher.contiguous(),
                              'ids': ids, 'labels': labels, 'mask': mask, 'weights': weights}
                    path = Path(folder) / f'features-{offset}.safetensors'
                    save_file({key: value.clone() for key, value in packet.items()}, path)
                    digest = sha256(path)
                    loaded = load_file(path)
                    assert all(torch.equal(value, loaded[key]) for key, value in packet.items())
                    packets.append((loaded, subset))
                    (Path(folder) / f'features-{offset}.json').write_text(json.dumps({'sha256': digest,
                        'batch': identity(subset), 'prefix_layers': cut, 'reference_layers': 4}) + '\n')
        if rank == 2:
            del teacher_tail
        reports = []
        for step in range(RECIPE['steps']):
            observed = incremental.train_step(shard, optimizer, wire, rows, RECIPE, step, 2, cut,
                kl_strength=2., margin_strength=1., margin_min=.5, margin_max=2., reference_layers=original)
            if rank == 2:
                local = features.train_step(cached, head, cached_optimizer, [value for value, subset in packets], rows, RECIPE, step, 2)
                loss, norm = local['loss'], local['gradient_norm']
                assert loss == observed['loss'] and norm == observed['gradient_norm']
                same = True
                largest = 0.
                for (name, left), (other, right) in zip(shard.named_owned_parameters(), cached.named_owned_parameters()):
                    assert name == other
                    same &= torch.equal(left, right)
                    largest = max(largest, float((left - right).detach().abs().max()))
                    torch.testing.assert_close(left, right, rtol=2e-5, atol=2e-7)
                    for key in optimizer.state[left]:
                        same &= torch.equal(optimizer.state[left][key], cached_optimizer.state[right][key])
                        torch.testing.assert_close(optimizer.state[left][key], cached_optimizer.state[right][key], rtol=2e-5, atol=2e-7)
                reports.append({'step': step + 1, 'exact_weights_and_adam': same,
                    'maximum_parameter_difference': largest, 'distributed_loss': observed['loss'],
                    'cached_loss': loss, 'distributed_norm': observed['gradient_norm'], 'cached_norm': norm})
        if rank == 2:
            (Path(folder) / 'result.json').write_text(json.dumps({'passed': True, 'arm': arm,
                'steps': reports, 'all_states_exact': all(row['exact_weights_and_adam'] for row in reports),
                'cached_learner_parameters_including_readonly_head': resident,
                'full_parent_parameters': full_size, 'feature_production_passes': 1,
                'reused_for_updates': RECIPE['steps'], 'feature_files_bytes': sum(p.stat().st_size for p in Path(folder).glob('*.safetensors')),
                'scope': 'Small CPU numerical factorization only. No real-model quality, certified feature market, GPU speed or native activation claim.'}, indent=2) + '\n')
    finally:
        dist.destroy_process_group()


@pytest.mark.parametrize('arm', ['append', 'control'])
def test_frozen_features_reproduce_every_distributed_loss_norm_weight_and_adam_state(tmp_path, arm):
    mp.spawn(worker, args=(arm, str(tmp_path), str(tmp_path / 'rendezvous')), nprocs=3, join=True)
    result = json.loads((tmp_path / 'result.json').read_bytes())
    assert result['all_states_exact']
    assert all(row['distributed_loss'] == row['cached_loss'] and row['distributed_norm'] == row['cached_norm'] for row in result['steps'])
    assert result['cached_learner_parameters_including_readonly_head'] < result['full_parent_parameters']


def test_feature_packet_metadata_and_readonly_head_cannot_change():
    rows = records()[:2]
    ids, labels, mask, weights = batch_tensors(rows, 'cpu')
    packet = {'ids': ids, 'labels': labels, 'mask': mask, 'weights': weights,
              'prefix': torch.zeros(*ids.shape, 24), 'reference': torch.zeros(*ids.shape, 24)}
    features.validate_packet(packet, rows, config(6), 'cpu')
    changed = {**packet, 'weights': weights.double()}
    with pytest.raises(ValueError, match='tokenization, labels, padding or weights'):
        features.validate_packet(changed, rows, config(6), 'cpu')
    changed = {**packet, 'labels': labels.clone()}
    changed['labels'][0, -1] = 3
    with pytest.raises(ValueError, match='tokenization, labels, padding or weights'):
        features.validate_packet(changed, rows, config(6), 'cpu')
    head = features.FrozenHead(config(6), torch.randn(47, 24), torch.ones(24))
    with torch.no_grad():
        head.norm.weight.add_(1)
    with pytest.raises(ValueError, match='head changed'):
        head.logits(torch.zeros(1, 24))
