"""Exercise growth comparison generation across four CPU processes before GPUs."""
from datetime import timedelta
import hashlib
import importlib.util
import json
from pathlib import Path

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from transformers import LlamaConfig

from neuroshard.evolution.programming_expert import MBPP_SHA
from neuroshard.evolution.reference_data import identity, save
from neuroshard.evolution.sharded.branch import Network, ParentWire
from neuroshard.evolution.sharded.model import Partition
from neuroshard.evolution.sharded.wire import Wire


def driver():
    path = Path(__file__).resolve().parents[2] / 'scripts/run_programming_growth.py'
    spec = importlib.util.spec_from_file_location('programming_growth_driver_test', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class Tokens:
    eos_token_id = 2

    def apply_chat_template(self, messages, tokenize=True, add_generation_prompt=True):
        return [3, 4, len(messages)]

    def decode(self, ids, skip_special_tokens=True):
        return '```python\npass\n```'


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
    config = LlamaConfig(vocab_size=32, hidden_size=16, intermediate_size=32, num_hidden_layers=6,
                         num_attention_heads=2, num_key_value_heads=2, tie_word_embeddings=True,
                         attention_dropout=0, max_position_embeddings=64)
    config._attn_implementation = 'sdpa'
    shard = Partition(config, [0, 2, 4, 6] if rank < 3 else [0, 2, 4, 5, 6], rank, inference_only=True)
    initialize(shard)
    dist.init_process_group('gloo', init_method='file://' + str(root / 'group'), rank=rank, world_size=4,
                            timeout=timedelta(seconds=60))
    group = dist.new_group([0, 1, 2], timeout=timedelta(seconds=60))
    wire = Wire(rank, 4)
    network = Network(shard, wire, ParentWire(rank, group) if rank < 3 else None, Tokens(), 5)
    rows = [{'id': identity({'dataset': MBPP_SHA, 'task': 376}), 'task_id': 376, 'kind': 'code',
             'messages': [{'role': 'user', 'content': 'Solve.\n\nUse this callable interface and behavior:\nassert f()\n\nReturn only the complete Python code, including needed imports.'}],
             'setup': '', 'tests': ['assert f()', 'assert g()']}]
    plan = {'generation_tokens': 3}
    try:
        outputs = driver().evaluate_growth(
            network, rows, plan, home, None, None, None, None,
            check=lambda *a: {'passed': False})
        if rank == 0:
            save(home / 'report.json', {'count': len(outputs)})
            arms = {out['arm'] for out in outputs}
            assert arms == {'base', 'incumbent', 'added'}
            assert all(out['generated'] for out in outputs)
            assert all(out['path'] == 'expert' for out in outputs if out['arm'] != 'base')
        (home / 'passed').touch()
    finally:
        dist.destroy_process_group()


def test_actual_driver_generates_parent_and_both_extras_without_training(tmp_path):
    mp.spawn(process, args=(str(tmp_path),), nprocs=4, join=True)
    assert all((tmp_path / str(rank) / 'passed').exists() for rank in range(4))
    outputs = json.loads((tmp_path / '0' / 'growth-outputs.json').read_bytes())
    assert {out['arm'] for out in outputs} == {'base', 'incumbent', 'added'}
