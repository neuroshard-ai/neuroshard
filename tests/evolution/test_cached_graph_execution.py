"""Actual parent, preserved interpreter and separate expert cache execution."""
from datetime import timedelta
import json
import os
from pathlib import Path
import time
from types import SimpleNamespace

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from neuroshard.evolution.sharded.cached_inference import generate_branch_cached
from neuroshard.evolution.sharded.graph_execution import GraphNetwork
from neuroshard.evolution.sharded.planned_graph import PlannedGraphNetwork, configuration
from neuroshard.evolution.sharded.router_features import EmbeddingFeatures
from neuroshard.evolution.sharded.portable import tensor_path
from test_graph_execution import prepare_graph, SOURCE


def worker(rank, folder):
    home = Path(folder)
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'test'
    read = lambda name: json.loads((home / name).read_bytes())
    dist.init_process_group('gloo', init_method='file://' + str(home / 'rendezvous'),
                            rank=rank, world_size=5, timeout=timedelta(seconds=90))
    try:
        graph = read('graph.json')
        # Reproduce asymmetric startup: a tail finishes long before a parent.
        # Group connection gets one second, local loading gets ten. Without
        # the readiness barrier the fast expert's group times out first.
        from neuroshard.evolution.sharded import graph_execution
        original_timedelta = graph_execution.timedelta
        graph_execution.timedelta = lambda **kw: timedelta(seconds=10 if kw['seconds'] == 1200 else 1)
        tensor_values = graph_execution.incremental_state.tensor_values
        first = True
        def slow_parent(*args, **kwargs):
            nonlocal first
            if rank == 2 and first:
                first = False
                time.sleep(2.5)
            return tensor_values(*args, **kwargs)
        graph_execution.incremental_state.tensor_values = slow_parent
        net = GraphNetwork(graph, read('profile.json'), objects=home/'objects',
                           interpreter=home/'interpreter', seed=home/'seed',
                           source_home=SOURCE, rank=rank)
        graph_execution.timedelta = original_timedelta
        graph_execution.incremental_state.tensor_values = tensor_values
        records = []
        for selected in ('parent', 'directory', 'protocol', 'interpreter'):
            expert = selected in graph['experts']
            if selected == 'interpreter':
                network = net.preserved
            elif selected == 'parent':
                network = next(iter(net.net.networks.values())) if rank < 3 else None
            else:
                network = net.net.networks.get(selected)
            if network is not None:
                # EOS is disabled only in this test so every path exercises
                # multiple decode positions and a long prompt's byte savings.
                network.tokenizer = SimpleNamespace(eos_token_id=-1,
                    apply_chat_template=net.tokenizer.apply_chat_template,
                    decode=net.tokenizer.decode)
                for question in ('word3 word4 word5 word6 word7 word8', 'word9'):
                    wire = network.wire if expert else network.parent_wire
                    before = wire.sent_tensor_bytes
                    expected = network.generate(question, 8, expert)['ids']
                    uncached_bytes = wire.sent_tensor_bytes - before
                    ids = net.tokenizer.apply_chat_template([{'role': 'user', 'content': question}],
                        tokenize=True, add_generation_prompt=True)
                    observation = {}
                    actual = generate_branch_cached(network, ids, 8, expert, observation)
                    assert actual == expected
                    assert generate_branch_cached(network, ids, 8, expert) == actual
                    assert observation['sent_tensor_bytes'] < uncached_bytes
                    begin, end = network.shard.boundaries[network.shard.rank:network.shard.rank+2]
                    if expert and rank == 2:
                        end = network.split
                    assert observation['cached_layers'] == [begin, end]
                    records.append({'model': selected, 'cached': observation,
                                    'uncached_bytes': uncached_bytes, 'ids': actual})
            # Nonparticipating expert owners perform no transformer work.
            dist.barrier()
        net.verify_unchanged()
        learned = read('learned.json')
        features = None
        if rank == 0:
            digest = learned['feature_profile']['embedding_sha256']
            features = EmbeddingFeatures(tensor_path(home/'interpreter', digest), digest,
                                        net.tokenizer, graph['tokenizer']['root'])
        planner = {'instruction': 'Interpret.', 'examples': [], 'max_tokens': 1, 'repeat_instruction': True}
        planned = PlannedGraphNetwork(net, configuration(graph, learned, planner, SOURCE,
            {'protocol': {'prefix': 'word7 ', 'suffix': ' word8'}}, 'Answer briefly.'),
                                      source_home=SOURCE, features=features)
        original_messages = [{'role': 'user', 'content': 'word3'}]
        prepared_messages = planned.planning_messages(original_messages)
        assert original_messages == [{'role': 'user', 'content': 'word3'}]
        assert prepared_messages[-1]['content'] == 'word3\n\nInterpret.'
        prompt = planned.expert_question('protocol', 'word3')
        assert prompt == 'word7 word3 word8'
        assert planned.answer_messages('parent', 'word3') == [
            {'role': 'system', 'content': 'Answer briefly.'}, {'role': 'user', 'content': 'word3'}]
        assert planned.answer_messages('protocol', 'word3') == [{'role': 'user', 'content': prompt}]
        planned.call('protocol', [{'role': 'user', 'content': prompt}], 4, 'answer')
        assert planned.trace[-1]['prompt_ids'] == net.tokenizer.apply_chat_template(
            [{'role': 'user', 'content': prompt}], tokenize=True, add_generation_prompt=True)
        # The random tiny interpreter cannot emit JSON with one token. A real
        # failed neural plan must execute no experts, remain metered/replayable,
        # and never manufacture an answer from the evaluator or fallback text.
        response = planned.answer([{'role': 'user', 'content': 'word3'}], 4)
        assert response['status'] == 'needs_clarification' and response['text'] == ''
        assert response['generated_tokens'] == 1 and response['answers'] == []
        assert [row['purpose'] for row in response['outputs']] == ['planning']
        assert planned.replay(response) == (True, response)
        assert not planned.replay({**response, 'text': 'forged answer'})[0]
        (home / f'cached-rank-{rank}.json').write_text(json.dumps(records))
    finally:
        dist.destroy_process_group()


def test_cached_graph_preserves_selected_paths_and_excludes_unowned_layers(tmp_path):
    torch.set_num_threads(1)
    prepare_graph(tmp_path)
    mp.spawn(worker, args=(str(tmp_path),), nprocs=5, join=True)
    rows = [json.loads((tmp_path / f'cached-rank-{rank}.json').read_bytes()) for rank in range(5)]
    assert {row['model'] for row in rows[3]} == {'directory'}
    assert {row['model'] for row in rows[4]} == {'protocol'}
    assert {row['model'] for row in rows[0]} == {'parent', 'directory', 'protocol', 'interpreter'}
