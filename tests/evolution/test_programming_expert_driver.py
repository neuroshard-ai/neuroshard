"""Exercise the actual new driver across four CPU processes before any GPUs."""
from datetime import timedelta
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from safetensors.torch import save_file
from transformers import LlamaConfig

from neuroshard.evolution.reference_data import sha256
from neuroshard.evolution.sharded.branch import Network, ParentWire
from neuroshard.evolution.sharded.cached_inference import generate_branch_cached
from neuroshard.evolution.sharded.model import Partition
from neuroshard.evolution.sharded.wire import Wire


def driver():
    path = Path(__file__).resolve().parents[2] / 'scripts/run_programming_expert.py'
    spec = importlib.util.spec_from_file_location('programming_expert_driver_test', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class Tokens:
    eos_token_id = 2


def initialize(shard):
    with torch.no_grad():
        for name, parameter in shard.named_owned_parameters():
            seed = int(hashlib.sha256(name.encode()).hexdigest()[:8], 16)
            parameter.copy_(torch.randn(parameter.shape, generator=torch.Generator().manual_seed(seed)) * .1)
    shard.eval()


def process(rank, folder):
    torch.set_num_threads(1)
    root = Path(folder)
    home = root / str(rank)
    home.mkdir()
    (home / 'objects').mkdir()
    config = LlamaConfig(vocab_size=32, hidden_size=16, intermediate_size=32, num_hidden_layers=6,
                         num_attention_heads=2, num_key_value_heads=2, tie_word_embeddings=True,
                         attention_dropout=0, max_position_embeddings=64)
    config._attn_implementation = 'sdpa'
    shard = Partition(config, [0, 2, 4, 6] if rank < 3 else [0, 2, 4, 5, 6], rank,
                      inference_only=rank < 3)
    initialize(shard)
    original = {name: p.detach().clone() for name, p in shard.named_owned_parameters()}
    head = {}
    if rank == 3:
        oracle = Partition(config, [0, 6], 0, inference_only=True)
        initialize(oracle)
        params = dict(oracle.named_owned_parameters())
        for name in ('model.embed_tokens.weight', 'model.norm.weight'):
            path = home / 'objects' / (name + '.safetensors')
            save_file({'weight': params[name].detach()}, path)
            head[name] = {'file': path.name, 'sha256': sha256(path)}
        del oracle, params
    dist.init_process_group('gloo', init_method='file://' + str(root / 'group'), rank=rank, world_size=4,
                            timeout=timedelta(seconds=60))
    group = dist.new_group([0, 1, 2], timeout=timedelta(seconds=60))
    wire = Wire(rank, 4)
    selection = {'head': wire.exchange(head)[3]}
    plan = {'split': 5, 'microbatch': 1, 'training': {'steps': 2, 'warmup_steps': 0,
            'learning_rate': .01, 'weight_decay': .01, 'clip_norm': 1.},
            'objective': {'kl_strength': 0., 'margin_strength': 0.}}
    rows = [{'id': 'x', 'input_ids': [3, 4, 5, 6, 2], 'labels': [-100, -100, 5, 6, 2], 'targets': 3}]
    prepared = {'batches': [[0]], 'schedule': [0, 0]}
    network = Network(shard, wire, ParentWire(rank, group) if rank < 3 else None, Tokens(), 5)
    try:
        before = generate_branch_cached(network, [3, 4], 5, False)
        wire.exchange(None)
        ablation = generate_branch_cached(network, [3, 4], 5, True)
        if rank == 0:
            assert before == ablation
        manifest = driver().train(shard, wire, plan, selection, prepared, rows, home)
        after = generate_branch_cached(network, [3, 4], 5, False)
        wire.exchange(None)
        added = generate_branch_cached(network, [3, 4], 5, True)
        if rank == 0:
            assert before == after and len(added) > 0
        if rank < 3:
            assert all(torch.equal(p, original[name]) for name, p in shard.named_owned_parameters())
        else:
            assert any(not torch.equal(p, original[name]) for name, p in shard.named_owned_parameters())
            assert (home / 'expert/adam.safetensors').is_file()
        assert manifest['step'] == 2
        (home / 'passed').touch()
    finally:
        dist.destroy_process_group()


def test_actual_driver_trains_only_added_owner_and_preserves_sharded_base(tmp_path):
    mp.spawn(process, args=(str(tmp_path),), nprocs=4, join=True)
    assert all((tmp_path / str(rank) / 'passed').exists() for rank in range(4))
