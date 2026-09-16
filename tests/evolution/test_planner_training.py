"""Train and resume through real owned forward/backward boundaries."""
from datetime import timedelta
import json
import os
from pathlib import Path

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from neuroshard.evolution.reference_data import identity
from neuroshard.evolution.sharded.cached_inference import generate_branch_cached
from neuroshard.evolution.sharded.graph_execution import GraphNetwork
from neuroshard.evolution.sharded.planner_training import PlannerTraining, installed
from test_graph_execution import prepare_graph, SOURCE


def owner(rank, folder):
    home = Path(folder)
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'test'
    read = lambda name: json.loads((home/name).read_bytes())
    dist.init_process_group('gloo', init_method='file://'+str(home/'planner-meeting'),
        rank=rank, world_size=5, timeout=timedelta(seconds=90))
    try:
        net = GraphNetwork(read('graph.json'), read('profile.json'), objects=home/'objects',
            interpreter=home/'interpreter', seed=home/'seed', source_home=SOURCE, rank=rank)
        messages = [{'role': 'user', 'content': 'word3 word4 word5'}]
        prompt = net.tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True)
        baseline = generate_branch_cached(net.preserved, prompt, 4, False) if rank < 3 else None
        # This numerical fixture's minimal chat template does not implement
        # assistant-prefix masking; give it explicit complete token targets.
        answers = [[7, 8, net.tokenizer.eos_token_id], [9, net.tokenizer.eos_token_id]]
        rows = [{'input_ids': prompt+answer, 'labels': [-100]*len(prompt)+answer,
                 'targets': len(answer)} for answer in answers]
        recipe = {'steps': 2, 'learning_rate': .01, 'warmup_steps': 0,
            'minimum_lr_ratio': .1, 'weight_decay': .01, 'clip_norm': 1.,
            'microbatch': 1, 'schedule': [[0, 1], [1, 0]]}
        torch.manual_seed(814)
        training = PlannerTraining(net, rows, recipe, adapter_rank=4, max_length=64)
        checkpoint_home = home/('planner-'+str(rank))
        initial = training.save(checkpoint_home)
        first = training.advance()
        middle = training.save(checkpoint_home)
        second = training.advance()
        final = training.save(checkpoint_home)
        assert first['gradient_norm'] > 0 and second['gradient_norm'] > 0
        assert initial['fusion'] != middle['fusion'] != final['fusion']
        resumed = PlannerTraining(net, rows, recipe, adapter_rank=4, max_length=64)
        resumed.restore(checkpoint_home, middle)
        assert resumed.advance() == second
        assert resumed.save(checkpoint_home) == final
        continued = PlannerTraining(net, rows, recipe, adapter_rank=4, max_length=64,
            initial_weights=final, weights_home=checkpoint_home)
        continued_initial = continued.save(checkpoint_home)
        assert continued_initial['fusion'] == final['fusion']
        assert continued_initial['step'] == 0
        assert continued_initial['binding']['initial_weights'] == final
        if rank == 2:
            assert not continued.state.optimizer.state
        continued_update = continued.advance()
        continued_final = continued.save(checkpoint_home)
        continuation_replay = PlannerTraining(net, rows, recipe, adapter_rank=4, max_length=64,
            initial_weights=final, weights_home=checkpoint_home)
        continuation_replay.restore(checkpoint_home, continued_initial)
        assert continuation_replay.advance() == continued_update
        assert continuation_replay.save(checkpoint_home) == continued_final
        if rank == 2:
            resumed.adapter.eval()
        with installed(net, resumed.adapter) as root:
            assert root == final['fusion']
            adapted = generate_branch_cached(net.preserved, prompt, 4, False) if rank < 3 else None
        answers = net.all_owners.exchange(adapted)
        assert answers[:3] == [answers[0]]*3 and answers[3:] == [None, None]
        # Planning hooks leave the preserved answer path exactly as loaded.
        after = generate_branch_cached(net.preserved, prompt, 4, False) if rank < 3 else None
        assert after == baseline
        net.verify_unchanged()
        from neuroshard.evolution.sharded.planned_graph import PlannedGraphNetwork, configuration
        from neuroshard.evolution.sharded.router_features import EmbeddingFeatures
        from neuroshard.evolution.sharded.portable import tensor_path
        learned, features = read('learned.json'), None
        if rank == 0:
            digest = learned['feature_profile']['embedding_sha256']
            features = EmbeddingFeatures(tensor_path(home/'interpreter', digest), digest,
                net.tokenizer, net.graph['tokenizer']['root'])
        config = configuration(net.graph, learned, {'instruction': 'Interpret.', 'examples': [], 'max_tokens': 4},
            SOURCE, planner_weights=final, composer={'instruction': 'Answer.', 'max_tokens': 4})
        service = PlannedGraphNetwork(net, config, source_home=SOURCE, features=features,
            planner_weights_home=checkpoint_home)
        service.call('interpreter', messages, 4, 'planning')
        assert service.trace[-1]['planner_adapter'] == final['fusion']
        service.call('interpreter', messages, 4, 'answer')
        assert 'planner_adapter' not in service.trace[-1]
        expected = net.all_owners.exchange(baseline)[0]
        assert service.trace[-1]['token_ids'] == expected
        composed_messages = service.composition_messages(messages,
            [{'question': 'word3', 'expert': 'protocol', 'text': 'word7'}])
        assert composed_messages[0] == {'role': 'system', 'content': 'Answer.'}
        assert 'word7' in composed_messages[-1]['content']
        assert messages[-1]['content'] == 'word3 word4 word5'
        service.call('interpreter', composed_messages, 1, 'composition')
        assert 'planner_adapter' not in service.trace[-1]
        if rank == 2:
            with torch.no_grad():
                next(service.planner_adapter.parameters()).add_(.01)
        try:
            service.call('interpreter', messages, 4, 'planning')
        except ValueError as error:
            assert 'changed after service installation' in str(error)
        else:
            raise AssertionError('A changed planner served under its old model identity')
        summary = {'initial': initial['fusion'], 'final': final['fusion'],
                   'first': first, 'second': second, 'baseline': net.all_owners.exchange(baseline)[0]}
        assert net.all_owners.exchange(identity(summary)) == [identity(summary)]*5
        (home/('owner-'+str(rank)+'.json')).write_text(json.dumps(summary))
    finally:
        dist.destroy_process_group()


def test_planner_gradients_restore_and_scoped_answer_preservation(tmp_path):
    prepare_graph(tmp_path)
    mp.spawn(owner, args=(str(tmp_path),), nprocs=5, join=True)
    values = [json.loads((tmp_path/('owner-'+str(rank)+'.json')).read_text()) for rank in range(5)]
    assert values == [values[0]]*5
