"""Train and resume through real owned forward/backward boundaries."""
from datetime import timedelta
import json
import os
from pathlib import Path
import shutil

import pytest

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
        from neuroshard.evolution import planner_window
        from neuroshard.evolution.sharded import planner_window as numerical_window
        profile = planner_window.prescription(net.graph, rows, recipe, initial)
        expected_window = {'format': planner_window.FORMAT, 'prescription': identity(profile),
                           'checkpoints': [middle, final], 'updates': [second]}
        report, actual = numerical_window.replay(net, profile, middle, expected_window, rows,
            checkpoint_home, home/('window-replay-'+str(rank)))
        assert report['valid'] and actual == expected_window
        from neuroshard.evolution import planner_work
        from test_planner_work_settlement import network, reserve, submit, finish
        full_window = {'format': planner_window.FORMAT, 'prescription': identity(profile),
                       'checkpoints': [initial, middle, final], 'updates': [first, second]}
        native, signers = network(profile)
        native = submit(reserve(native, signers, stages=2), signers, full_window)
        verdicts = []
        for auditor in range(3):
            coverage, replayed = numerical_window.audit_report(native['candidate'], profile, net, rows,
                checkpoint_home, home/('native-audit-'+str(auditor)+'-'+str(rank)))
            assert replayed == full_window
            verdicts.append(planner_work.replay_report(native['candidate'], coverage)['valid'])
        settled = finish(native, signers, verdicts)
        assert settled['issued'] == 2*settled['manifest']['params']['reward_atoms']
        assert settled['planner_work']['checkpoint'] == final
        assert settled['serving_root'] == profile['graph']
        paid = planner_window.validate(profile, middle, expected_window)['work_ids']
        with pytest.raises(ValueError, match='already been paid'):
            planner_window.validate(profile, middle, expected_window, paid=paid)
        forged = {**expected_window, 'updates': [{**second, 'loss': second['loss']+1}]}
        report, actual = numerical_window.replay(net, profile, middle, forged, rows,
            checkpoint_home, home/('forged-window-'+str(rank)))
        assert not report['valid'] and actual == expected_window
        before_missing = net.all_owners.sent_tensor_bytes
        with pytest.raises(ValueError, match='unavailable'):
            numerical_window.restore(net, profile, middle, rows, home/'absent-window-inputs')
        assert net.all_owners.sent_tensor_bytes == before_missing
        warm_profile = planner_window.prescription(net.graph, rows, recipe, continued_initial)
        warm_second = continued.advance()
        warm_final = continued.save(checkpoint_home)
        isolated = home/('isolated-window-'+str(rank))
        isolated.mkdir()
        if rank == 2:
            shutil.copy2(checkpoint_home/(continued_final['sha256']+'.safetensors'), isolated)
            assert not (isolated/(final['sha256']+'.safetensors')).exists()
        expected_warm = {'format': planner_window.FORMAT, 'prescription': identity(warm_profile),
                         'checkpoints': [continued_final, warm_final], 'updates': [warm_second]}
        report, actual = numerical_window.replay(net, warm_profile, continued_final, expected_warm, rows,
            isolated, home/('warm-window-replay-'+str(rank)))
        assert report['valid'] and actual == expected_warm
        if rank == 2:
            from neuroshard.evolution.reference_data import sha256
            assert sha256(isolated/(warm_final['sha256']+'.safetensors')) == warm_final['sha256']
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
        from neuroshard.evolution import planned_metering
        tariff = {'prompt_atom_price': 2, 'output_atom_price': 7,
                  'context': net.graph['tokenizer']['max_context']}
        response = service.answer(messages, 4)
        bill = planned_metering.meter(service.config, net.graph, response, tariff)
        assert bill['output_tokens'] == response['generated_tokens']
        assert net.all_owners.exchange(identity(bill)) == [identity(bill)]*5
        valid, replayed = service.replay(response)
        assert valid and planned_metering.meter(service.config, net.graph, replayed, tariff) == bill
        valid, replayed = service.replay({**response, 'text': response['text']+' forged'})
        assert not valid and replayed == response
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


def test_checkpoint_retention_is_atomic_across_filesystems_and_rejects_corrupt_existing_objects(tmp_path, monkeypatch):
    import errno
    from neuroshard.evolution.reference_data import sha256
    from neuroshard.evolution.sharded.planner_window import retain
    produced, store = tmp_path/'produced', tmp_path/'store'
    produced.mkdir()
    raw = produced/'raw'
    raw.write_bytes(b'complete numerical object')
    state = {'sha256': sha256(raw), 'bytes': raw.stat().st_size}
    raw.rename(produced/(state['sha256']+'.safetensors'))
    original = os.link

    def cross_device(source, target):
        if Path(source).parent == produced:
            raise OSError(errno.EXDEV, 'fixture separate volume')
        return original(source, target)

    monkeypatch.setattr(os, 'link', cross_device)
    retain(produced, store, state)
    retain(produced, store, state)
    assert list(store.iterdir()) == [store/(state['sha256']+'.safetensors')]
    (store/(state['sha256']+'.safetensors')).write_bytes(b'corrupt')
    with pytest.raises(ValueError, match='Retained planner object'):
        retain(produced, store, state)
