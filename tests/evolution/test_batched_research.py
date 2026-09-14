"""Check weighted batched DDP against the established per-document objective."""
from datetime import timedelta
import json
from pathlib import Path
import sys

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel
from transformers import LlamaConfig, LlamaForCausalLM

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'scripts'))
from batch_scaling_probe import batched_step
from neuroshard.evolution import cooperative as group
from neuroshard.evolution import reference as engine

RECIPE = {'steps': 3, 'warmup_steps': 0, 'learning_rate': .0001,
          'weight_decay': .01, 'clip_norm': 1e9}


def network():
    torch.manual_seed(17)
    return LlamaForCausalLM(LlamaConfig(
        vocab_size=32, hidden_size=16, intermediate_size=32, num_hidden_layers=2,
        num_attention_heads=2, num_key_value_heads=1, max_position_embeddings=64,
        tie_word_embeddings=True, attention_dropout=0., attn_implementation='eager', use_cache=False))


def worker(rank, rendezvous, destination):
    torch.set_num_threads(1)
    dist.init_process_group('gloo', init_method='file://' + rendezvous,
                            rank=rank, world_size=2, timeout=timedelta(seconds=60))
    try:
        # Target counts differ, and one rank receives all the high-weight rows.
        rows = [{'id': str(i), 'input_ids': [1, 3, 4] + [7 + i] * (i + 1) + [2],
                 'labels': [-100] * 3 + [7 + i] * (i + 1) + [2], 'targets': i + 2,
                 'loss_weight': 8 if i % 2 == 0 else 1} for i in range(8)]
        expected, actual = network(), network()
        expected_optimizer = engine.optimizer_for(expected, RECIPE)
        actual_optimizer = engine.optimizer_for(actual, RECIPE)
        wrapped = DistributedDataParallel(actual, broadcast_buffers=False, gradient_as_bucket_view=True)
        for index in range(3):
            batch = rows if index % 2 == 0 else list(reversed(rows))
            control = group.step(expected, expected_optimizer, batch, 'cpu', RECIPE, index)
            observed = batched_step(wrapped, actual_optimizer, batch, RECIPE, index, rank, 2, 3, device='cpu')
            # Four local rows split into microbatches of three and one.
            for key in ['loss', 'gradient_norm']:
                torch.testing.assert_close(torch.tensor(observed[key]), torch.tensor(control[key]), rtol=1e-5, atol=1e-7)
            for left, right in zip(expected.parameters(), actual.parameters()):
                torch.testing.assert_close(right, left, rtol=1e-5, atol=1e-7)
                for key, value in expected_optimizer.state[left].items():
                    torch.testing.assert_close(actual_optimizer.state[right][key], value, rtol=1e-5, atol=1e-9)
        digest = group.parameter_digest(actual)
        group.agree_digest(digest, 2, 'cpu')
        (Path(destination) / f'{rank}.json').write_text(json.dumps({'rank': rank, 'passed': True}))
    finally:
        dist.destroy_process_group()


def test_weighted_batched_ddp_matches_per_document_adam_with_partial_microbatches(tmp_path):
    mp.spawn(worker, args=(str(tmp_path / 'rendezvous'), str(tmp_path)), nprocs=2, join=True)
    assert all(json.loads((tmp_path / f'{rank}.json').read_text())['passed'] for rank in range(2))
