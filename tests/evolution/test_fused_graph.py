"""Actual five-owner generation through a single causal fusion stream."""
from datetime import timedelta
import copy
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
from neuroshard.evolution.sharded.fusion_features import produce
from neuroshard.evolution.sharded.fusion_training import Trainer
from neuroshard.evolution.sharded.fusion_trial import synchronize, response_losses
from neuroshard.evolution.reference_data import identity, sha256
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
        assert generate_fused(net, fusion, tokens, 4, source_ablation=True) == baseline
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
        rows = [{'id': identity(['row', index]), 'input_ids': values, 'kind': 'general',
                 'labels': [-100]*(len(values)-1)+[values[-1]]}
                for index, values in enumerate([[3, 4, 5, 6], [3, 4, 9, 8, 7, 6]])]
        net.runtime = {**net.runtime, 'host': 'distinct-owner-'+str(rank)}
        original_threads = net.runtime['threads']
        if rank == 4:
            net.runtime['threads'] += 1
        try:
            produce(net, rows, [[0, 1]], home/'rejected-bank', max_length=64, max_seconds=60)
        except ValueError as error:
            assert 'different fusion source data or runtime' in str(error)
        else:
            raise AssertionError('Owners accepted a changed numerical runtime')
        net.runtime['threads'] = original_threads
        bank, traffic = produce(net, rows, [[0, 1]], home/'fusion-bank', max_length=64, max_seconds=60)
        assert 'host' not in bank['runtime']
        assert traffic['owner_runtime']['host'] == 'distinct-owner-'+str(rank)
        assert bank['files'][0]['shape'] == [2, 6, width]
        assert traffic['sent_tensor_bytes'] > 0
        if rank == 0:
            from safetensors.torch import load_file
            path = home/'fusion-bank'/(bank['files'][0]['sha256']+'.safetensors')
            assert sha256(path) == bank['files'][0]['sha256']
            values = load_file(path)
            for name in ['hub', 'parent', *graph['experts']]:
                torch.testing.assert_close(values[name][0, :2], values[name][1, :2], rtol=1e-6, atol=1e-6)
            assert values['labels'].tolist() == [[-100, -100, -100, 6, -100, -100],
                                                [-100, -100, -100, -100, -100, 6]]
            assert not values['valid'][0, 4:].any()
            recipe = {'steps': 3, 'learning_rate': .001, 'warmup_steps': 1, 'minimum_lr_ratio': .1,
                      'weight_decay': .01, 'clip_norm': 1., 'microbatch': 1, 'general_kl': 2., 'schedule': [0]*3}
            initial = copy.deepcopy(fusion)
            trainer = Trainer(fusion, net.preserved.shard, rows, bank, home/'fusion-bank', recipe)
            first = trainer.advance()
            assert first['step'] == 1
            saved = trainer.save(home/'fusion-checkpoints')
            second = trainer.advance()
            expected = trainer.save(home/'fusion-checkpoints')
            recovered = Trainer(initial, net.preserved.shard, rows, bank, home/'fusion-bank', recipe)
            recovered.restore(home/'fusion-checkpoints', saved)
            assert recovered.advance() == second
            assert recovered.save(home/'fusion-recovered') == expected
            broken = copy.deepcopy(saved)
            broken['binding']['source_ablation'] = True
            rejected = Trainer(copy.deepcopy(initial), net.preserved.shard, rows, bank, home/'fusion-bank', recipe)
            try:
                rejected.restore(home/'fusion-checkpoints', broken)
            except ValueError:
                pass
            else:
                raise AssertionError('Changed training source policy restored')
            net.verify_unchanged()
        assert wire_exchange(net) == ['trained']*5
        synchronized = synchronize(net, fusion)
        assert net.all_owners.exchange(synchronized) == [synchronized]*5
        if rank == 0:
            values = response_losses({'fusion': fusion, 'ablation': copy.deepcopy(fusion)},
                net.preserved.shard, rows, bank, home/'fusion-bank', recipe)
            assert set(values) == {row['id'] for row in rows}
            assert all(set(value) == {'hub', 'fusion', 'ablation'} for value in values.values())
            assert all(all(loss > 0 for loss in value.values()) for value in values.values())
        assert wire_exchange(net) == ['trained']*5
        (home/('fused-owner-'+str(rank)+'.json')).write_text(json.dumps(observation))
    finally:
        dist.destroy_process_group()


def wire_exchange(net):
    return net.all_owners.exchange('trained')


def test_owned_paths_share_one_token_stream_and_preserve_seed_at_initialization(tmp_path):
    prepare_graph(tmp_path)
    mp.spawn(owner, args=(str(tmp_path),), nprocs=5, join=True)
    rows = [json.loads((tmp_path/('fused-owner-'+str(rank)+'.json')).read_bytes()) for rank in range(5)]
    assert len({row['fusion'] for row in rows}) == 1
    assert all(row['tokens'] == rows[0]['tokens'] for row in rows)
