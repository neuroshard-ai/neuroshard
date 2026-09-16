"""Actual five-owner generation through a single causal fusion stream."""
from datetime import timedelta
import json
import os
from pathlib import Path

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from neuroshard.evolution.sharded.cached_inference import generate_branch_cached
from neuroshard.evolution.sharded.fusion import CrossShardFusion
from neuroshard.evolution.sharded.fused_graph import generate_fused
from neuroshard.evolution.sharded.graph_execution import GraphNetwork
from test_graph_execution import prepare_graph, SOURCE


def owner(rank, folder):
    home = Path(folder)
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'test'
    read = lambda name: json.loads((home/name).read_bytes())
    dist.init_process_group('gloo', init_method='file://'+str(home/'fusion-meeting'),
        rank=rank, world_size=5, timeout=timedelta(seconds=90))
    try:
        graph = read('graph.json')
        net = GraphNetwork(graph, read('profile.json'), objects=home/'objects',
            interpreter=home/'interpreter', seed=home/'seed', source_home=SOURCE, rank=rank)
        messages = [{'role': 'user', 'content': 'word3 word4 word5 word6'}]
        tokens = net.tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True)
        original = generate_branch_cached(net.preserved, tokens, 4, False) if rank < 3 else None
        baseline = net.all_owners.exchange(original)[0]
        torch.manual_seed(477)
        width = graph['parent']['config']['hidden_size']
        fusion = CrossShardFusion(width, {name: width for name in ['parent', *graph['experts']]},
            rank=8, heads=2, max_context=64).eval()
        observation = {}
        assert generate_fused(net, fusion, tokens, 4, observation) == baseline
        assert observation['executed_positions'] == len(tokens)+len(baseline)-1
        assert observation['sent_tensor_bytes'] > 0
        if rank == 0:
            assert observation['fusion_cache_bytes'] == 3*2*observation['executed_positions']*8*4
        else:
            assert observation['fusion_cache_bytes'] == 0
        with torch.no_grad():
            fusion.output.weight.normal_(std=.2)
        changed = generate_fused(net, fusion, tokens, 4)
        assert generate_fused(net, fusion, tokens, 4) == changed
        (home/('fused-owner-'+str(rank)+'.json')).write_text(json.dumps(observation))
    finally:
        dist.destroy_process_group()


def test_owned_paths_share_one_token_stream_and_preserve_seed_at_initialization(tmp_path):
    prepare_graph(tmp_path)
    mp.spawn(owner, args=(str(tmp_path),), nprocs=5, join=True)
    rows = [json.loads((tmp_path/('fused-owner-'+str(rank)+'.json')).read_bytes()) for rank in range(5)]
    assert len({row['fusion'] for row in rows}) == 1
    assert all(row['tokens'] == rows[0]['tokens'] for row in rows)
