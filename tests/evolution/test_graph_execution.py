"""Five real processes audit raw parent, interpreted and composed answers."""
import copy
from datetime import timedelta
import json
import math
import os
from pathlib import Path
import shutil
import socket
import time

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from tokenizers import Tokenizer, models, pre_tokenizers
from transformers import LlamaConfig, LlamaForCausalLM, PreTrainedTokenizerFast

from neuroshard.evolution import expert_checkpoint, expert_lifecycle, serving_graph, expert_router
from neuroshard.evolution.reference_data import identity, save, sha256, tokenizer_identity
from neuroshard.evolution.sharded import checkpoint, cohort_state, incremental_state, portable
from neuroshard.evolution.sharded.graph_execution import GraphNetwork, FORMAT, preflight
from neuroshard.evolution.sharded.graph_service import inference_report, inference_transcript
from neuroshard.evolution.sharded import graph_quality
from neuroshard.evolution.sharded import learned_graph
from neuroshard.evolution.sharded.router_features import EmbeddingFeatures
from neuroshard.evolution.sharded.interpretation import example_messages
from test_expert_cohort_state import prepare, update, RECIPE
from test_serving_graph import FIXTURE

SOURCE = Path(__file__).resolve().parents[2]


def prepare_graph(home):
    parent, owned, shard, optimizer = prepare(home)
    initial = cohort_state.commit_tail(home / 'initial', shard, optimizer, parent, owned, 'b'*64, 0, RECIPE, 5)
    update(shard, optimizer, 0)
    first = cohort_state.commit_tail(home / 'first', shard, optimizer, parent, owned, 'a'*64, 1, RECIPE, 5)
    update(shard, optimizer, 1)
    second = cohort_state.commit_tail(home / 'second', shard, optimizer, parent, owned, 'b'*64, 2, RECIPE, 5)
    objects, interpreter, seed = (home / name for name in ('objects', 'interpreter', 'seed'))
    for folder in (objects, interpreter, seed):
        folder.mkdir()
    torch.manual_seed(14)
    model = LlamaForCausalLM(LlamaConfig(**parent['config'])).float()
    parts = {str(rank): {'rank': rank, 'parameters': 0, 'tensors': {}} for rank in range(3)}
    for name, parameter in model.named_parameters():
        values = {'weight': parameter.detach(), 'step': torch.tensor(7.),
                  'exp_avg': torch.ones_like(parameter)*.001, 'exp_avg_sq': torch.ones_like(parameter)*.002}
        temporary = home / 'tensor.pending'
        spec = checkpoint.tensor_file(temporary, values)
        assert spec['sha256'] == parent['tensors'][name]['sha256']
        temporary.replace(portable.tensor_path(objects, spec['sha256']))
        spec = checkpoint.tensor_file(temporary, {'weight': parameter.detach().to(torch.bfloat16)})
        temporary.replace(portable.tensor_path(interpreter, spec['sha256']))
        rank = str(expert_checkpoint.owner(name, parent['boundaries']))
        parts[rank]['tensors'][name] = {**spec, 'shape': list(parameter.shape), 'dtype': 'bfloat16',
                                      'file': spec['sha256'] + '.safetensors'}
        parts[rank]['parameters'] += parameter.numel()
    for folder in (home / 'first/shard-000001', home / 'second/shard-000002'):
        for path in folder.glob('*.safetensors'):
            shutil.copyfile(path, objects / path.name)
    vocabulary = {'<unk>': 0, '<s>': 1, '</s>': 2, **{'word'+str(i): i for i in range(3, 32)}}
    backend = Tokenizer(models.WordLevel(vocabulary, unk_token='<unk>'))
    backend.pre_tokenizer = pre_tokenizers.Whitespace()
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=backend, unk_token='<unk>', bos_token='<s>', eos_token='</s>')
    # Compact text-dependent prompts keep the interpreted fixture within the
    # tiny model's context; no expected answers or routing labels enter them.
    tokenizer.chat_template = "{% for message in messages %}{{ message['content'] }} {% endfor %}{% if add_generation_prompt %}<s>{% endif %}"
    tokenizer.save_pretrained(seed)
    base = json.loads(FIXTURE.read_bytes())
    graph = base['candidate']
    graph['parent'] = parent
    graph['experts'] = {'directory': expert_checkpoint.pack(parent, first),
                        'protocol': expert_checkpoint.pack(parent, second)}
    size = sum(math.prod(spec['shape']) for spec in parent['tensors'].values())
    tail = sum(math.prod(spec['shape']) for spec in graph['experts']['directory']['tensors'].values())
    provenance = {'repo': 'local-test-model', 'revision': 'test', 'license': 'Apache-2.0', 'parameters': size}
    assets = {'format': 'neuroshard-preserved-interpreter-assets-v1', 'boundaries': parent['boundaries'],
              'source': provenance, 'source_weight_sha256': identity(provenance), 'partitions': parts}
    graph['interpreter_assets'] = assets
    graph['descriptor'].update(parent=identity(parent), split=5, parent_layout=parent['boundaries'],
        expert_layout=first['boundaries'], experts=[{'id': name, 'checkpoint': value['checkpoint']}
            for name, value in graph['experts'].items()], total_parameters=2*size+2*tail,
        interpretation={'instruction': 'Interpret.', 'examples': [], 'max_tokens': 1,
            'instruction_placement': 'after-quoted-question', 'invalid': 'Use the original question.'},
        interpreter={**provenance, 'weight_sha256': identity(provenance), 'partitioned_assets': identity(assets)})
    graph['tokenizer'] = {'root': tokenizer_identity(tokenizer), 'eos_id': 2, 'max_context': 64,
                          'files': {path.name: sha256(path) for path in seed.iterdir()}}
    prefix = example_messages('Interpret.', [])
    graph['interpreter_prompt'] = {'format': 'name-field-json-v1', 'messages': identity(prefix),
        'tokens': identity(tokenizer.apply_chat_template(prefix, tokenize=True, add_generation_prompt=False))}
    graph['descriptor'].update(tokenizer=graph['tokenizer']['root'], interpreter_prompt=graph['interpreter_prompt'])
    profile = {'format': FORMAT, 'runtime': {'device': 'cpu', 'threads': 1, 'allocator': 'test'}, 'threads': 1,
        'parameter_limit': 100000, 'resident_parameter_limit': 200000, 'numerical_profile': 'c'*64,
        'sources': {'src/neuroshard/evolution/sharded/graph_execution.py': sha256(
            SOURCE / 'src/neuroshard/evolution/sharded/graph_execution.py')}}
    graph.update(numerical_profile='c'*64, executor_root=identity(profile))
    previous = copy.deepcopy(graph)
    previous['descriptor'] = base['previous_descriptor']
    previous['descriptor'].update(parent=identity(parent), expert=identity(first), split=5,
        parent_layout=parent['boundaries'], expert_layout=first['boundaries'], parent_parameters=size,
        added_parameters=tail, added_tensors=graph['experts']['directory']['tensors'],
        interpreter=graph['descriptor']['interpreter'], interpretation=graph['descriptor']['interpretation'],
        interpreter_parameters=size, total_parameters=2*size+tail)
    del previous['experts']['protocol']
    graph['descriptor']['previous_graph'] = identity(previous['descriptor'])
    serving_graph.validate(previous)
    serving_graph.validate(graph)
    save(home / 'graph.json', graph)
    save(home / 'previous.json', previous)
    save(home / 'profile.json', profile)
    inputs = home / 'quality-inputs'
    inputs.mkdir()
    examples = []
    for index, question in enumerate((QUESTIONS[2], QUESTIONS[-1])):
        answers = ['unseen answer'] * (index + 1)
        examples.append({'id': 'new-' + str(index), 'stratum': 'single' if index == 0 else 'composed',
            'topics': ['first'] if index == 0 else ['first', 'second'], 'answers': answers,
            'messages': [{'role': 'user', 'content': question}, {'role': 'assistant', 'content': '; '.join(answers)}]})
    roles = {'test': examples}
    for role, question in zip(graph_quality.ROLES[1:], (QUESTIONS[1], QUESTIONS[0], QUESTIONS[0])):
        roles[role] = [{'id': role, 'messages': [{'role': 'user', 'content': question}],
                        'input_ids': [1, 3, 2], 'labels': [-100, 3, 2], 'targets': 2}]
    specs = {}
    for role, values in roles.items():
        path = inputs / (role + '.jsonl')
        path.write_text(''.join(json.dumps(row) + '\n' for row in values))
        specs[role] = {'file': path.name, 'sha256': sha256(path), 'count': len(values),
                       'ids': identity([row['id'] for row in values])}
    policy = {'format': graph_quality.FORMAT, 'baseline_graph': identity(previous),
        'candidate_graph': identity(graph), 'prepared': 'a'*64, 'roles': specs,
        'generation': {'new': 4, 'retained_knowledge': 4, 'retained_skills': 4},
        'gates': {'single_accuracy': .75, 'composed_accuracy': .5, 'gain_lower': .1,
                  'bootstrap_samples': 100, 'bootstrap_seed': 42, 'confidence': .95}}
    save(home / 'quality-policy.json', policy)
    template = copy.deepcopy(graph)
    template['experts']['protocol'] = expert_checkpoint.pack(parent, initial)
    template['descriptor']['experts'][1]['checkpoint'] = identity(initial)
    prospective = {key: value for key, value in policy.items() if key != 'candidate_graph'}
    prospective.update(format=graph_quality.PROSPECTIVE, candidate_template=template)
    save(home / 'prospective-policy.json', prospective)
    embedding = assets['partitions']['0']['tensors']['model.embed_tokens.weight']['sha256']
    features = EmbeddingFeatures(portable.tensor_path(interpreter, embedding), embedding,
                                 tokenizer, graph['tokenizer']['root'])
    observations = [{'id': identity([name, variant]), 'route': name, 'features': features(question)}
                    for name, question in LEARNED_QUESTIONS for variant in range(2)]
    router = expert_router.fit(observations, embedding_root=features.root,
                               tokenizer_root=graph['tokenizer']['root'], prototypes_per_route=1)
    router = expert_router.fit_classifier(observations, router)
    save(home / 'learned.json', learned_graph.configuration(graph, router, features.profile, SOURCE))
    return graph, profile


QUESTIONS = ['Hello!', 'In the fictional Luma directory, where does Robin Finch live?',
    'Regarding NeuroShard 0.4.0, which port?',
    'NeuroShard 0.4.0: First: Which port? Second: Which token? '
    'Reply with the two short answers in order. Separate the two answers with a semicolon.']
LEARNED_QUESTIONS = [('directory', 'word3'), ('protocol', 'word4'), ('parent', 'word5')]


def worker(rank, home):
    home = Path(home)
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'test'
    read = lambda name: json.loads((home / name).read_bytes())
    dist.init_process_group('gloo', init_method='file://' + str(home / 'rendezvous'),
        rank=rank, world_size=5, timeout=timedelta(seconds=90))
    try:
        graph = read('graph.json')
        net = GraphNetwork(graph, read('profile.json'), objects=home/'objects', interpreter=home/'interpreter',
                           seed=home/'seed', source_home=SOURCE, rank=rank)
        results = []
        for question in QUESTIONS:
            actual = net.answer(question, 4)
            claim = {'kind': 'expert_inference', 'model_root': identity(graph), 'graph': graph,
                'record_root': 'a'*64, 'stages': sum(len(row['token_ids']) for row in actual['outputs']),
                'executor_root': graph['executor_root'], 'job_id': 'b'*64,
                **{key: actual[key] for key in ('request', 'outputs', 'text')}}
            claim['record_root'] = identity(inference_transcript(claim, actual))
            report, replay = inference_report(claim, net)
            assert expert_lifecycle.replay_report(claim, report)['valid'] and replay == actual
            if question == QUESTIONS[-1]:
                assert len(actual['outputs']) == 2 and all(row['model'] == 'protocol' for row in actual['outputs'])
                forged = copy.deepcopy(claim)
                forged['text'] += ' fabricated'
                fake_result = {**actual, 'text': forged['text']}
                forged['record_root'] = identity(inference_transcript(forged, fake_result))
                bad, _ = inference_report(forged, net)
                assert not expert_lifecycle.replay_report(forged, bad)['valid']
            if question in QUESTIONS[:2]:
                old = net.answer(question, 4, read('previous.json'))
                assert old['outputs'] == actual['outputs'] and old['text'] == actual['text']
            results.append(actual)
        assert [row['model'] for row in results[1]['outputs']] == ['interpreter', 'directory']
        assert net.shard.resident_parameters < sum(serving_graph.ownership(graph, 'parent').values())
        with pytest.raises(ValueError, match='context truncation'):
            net.answer('word3 '*100, 4)
        with pytest.raises(ValueError, match='different inference requests'):
            net.answer(QUESTIONS[2] if rank == 4 else QUESTIONS[0], 4)
        assert net.answer(QUESTIONS[0], 4) == results[0]
        policy, previous = read('quality-policy.json'), read('previous.json')
        measured = graph_quality.evaluate(policy, home/'quality-inputs', previous, graph, net)
        assert measured['retention']['passed'] and not measured['decision']['passed']
        claim = {'kind': 'expert_quality', 'model_root': identity(graph), 'graph': graph,
            'baseline_graph': previous, 'record_root': 'a'*64, 'stages': policy['roles']['test']['count'],
            'executor_root': graph['executor_root'],
            'report': {'format': expert_lifecycle.FORMAT + '/quality', 'policy_root': identity(policy),
                'baseline_graph': identity(previous), 'candidate_graph': identity(graph),
                'prepared': policy['prepared'], 'passed': False, 'results_root': identity(measured)}}
        claim['record_root'] = identity(graph_quality.quality_transcript(claim, measured))
        report, replay = graph_quality.quality_report(claim, policy, home/'quality-inputs', net)
        assert replay == measured and expert_lifecycle.replay_report(claim, report)['valid']
        claim['report']['passed'] = True
        report, _ = graph_quality.quality_report(claim, policy, home/'quality-inputs', net)
        assert not expert_lifecycle.replay_report(claim, report)['valid']
        policy = read('prospective-policy.json')
        future = graph_quality.evaluate(policy, home/'quality-inputs', previous, graph, net)
        assert future == {**measured, 'policy': identity(policy)}
        altered = copy.deepcopy(graph)
        altered['tokenizer']['files']['tokenizer.json'] = 'f'*64
        with pytest.raises(ValueError, match='Quality policy differs'):
            graph_quality.evaluate(policy, home/'quality-inputs', previous, altered, net)
        features = None
        if rank == 0:
            embedding = graph['interpreter_assets']['partitions']['0']['tensors']['model.embed_tokens.weight']['sha256']
            features = EmbeddingFeatures(portable.tensor_path(home/'interpreter', embedding),
                                         embedding, net.tokenizer, graph['tokenizer']['root'])
        automatic = learned_graph.LearnedGraphNetwork(net, read('learned.json'), source_home=SOURCE, features=features)
        for route, question in LEARNED_QUESTIONS:
            result = automatic.answer(question, 4)
            assert result['routing']['decision']['route'] == route
            assert result['outputs'][-1]['model'] == route
            valid, replay = automatic.replay(result)
            assert valid and replay == result
            if route == 'parent':
                original = net.answer(question, 4)
                assert original['outputs'] == result['outputs'] and original['text'] == result['text']
            if route == 'protocol':
                forged = copy.deepcopy(result)
                forged['routing']['decision']['route'] = 'directory'
                assert automatic.replay(forged)[0] is False
            results.append(result)
        save(home / ('rank-' + str(rank) + '.json'), results)
    finally:
        dist.destroy_process_group()


def test_real_five_owner_inference_replay_rejects_fabricated_text(tmp_path):
    prepare_graph(tmp_path)
    context = mp.spawn(worker, args=(str(tmp_path),), nprocs=5, join=False)
    try:
        while not context.join(timeout=60):
            pass
    finally:
        for process in context.processes:
            if process.is_alive():
                process.terminate()
            process.join(timeout=10)
    results = [json.loads((tmp_path / ('rank-' + str(rank) + '.json')).read_bytes()) for rank in range(5)]
    assert all(result == results[0] for result in results)


def test_graph_executor_rejects_missing_or_changed_source_before_allocation(tmp_path, monkeypatch):
    graph, profile = prepare_graph(tmp_path)
    monkeypatch.setattr(torch, 'empty', lambda *a, **k: pytest.fail('Unexpected model allocation'))
    with pytest.raises(FileNotFoundError):
        preflight(graph, profile, tmp_path)
    wrong = copy.deepcopy(profile)
    wrong['threads'] += 1
    with pytest.raises(ValueError, match='installed executor'):
        preflight(graph, wrong, SOURCE)


def queue_worker(rank, home, port):
    import importlib.util
    home = Path(home)
    os.environ.update(RANK=str(rank), WORLD_SIZE='5', MASTER_ADDR='127.0.0.1', MASTER_PORT=str(port),
                      PYTORCH_CUDA_ALLOC_CONF='test')
    spec = importlib.util.spec_from_file_location('graph_operator', SOURCE / 'scripts/run_native_expert_service.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.run({'max_seconds': 60, 'home': str(home / ('service-' + str(rank))),
        'graph': str(home/'graph.json'), 'profile': str(home/'profile.json'),
        'baseline': str(home/'previous.json'), 'quality_policy': str(home/'quality-policy.json'),
        'objects': str(home/'objects'), 'interpreter': str(home/'interpreter'),
        'seed': str(home/'seed'), 'inputs': str(home/'quality-inputs'), 'source_home': str(SOURCE),
        'learned_service': str(home/'learned.json')})


def test_operator_queue_survives_idle_and_returns_the_same_five_owner_result(tmp_path):
    graph, _ = prepare_graph(tmp_path)
    with socket.socket() as sock:
        sock.bind(('127.0.0.1', 0))
        port = sock.getsockname()[1]
    context = mp.spawn(queue_worker, args=(str(tmp_path), port), nprocs=5, join=False)
    deadline = time.monotonic() + 90

    def wait(predicate):
        while not predicate():
            if time.monotonic() >= deadline:
                raise TimeoutError('Bounded operator queue test expired')
            if any(p.exitcode not in (None, 0) for p in context.processes):
                context.join(timeout=1)
            time.sleep(.05)

    try:
        wait(lambda: all((tmp_path / ('service-' + str(rank)) / 'ready.json').exists() for rank in range(5)))
        time.sleep(1.2)
        queue = tmp_path / 'service-0/requests'
        save(queue / ('1'*64 + '.json'), {'id': '1'*64, 'kind': 'generate',
            'graph': identity(graph), 'question': QUESTIONS[2], 'max_tokens': 4})
        paths = [tmp_path / ('service-' + str(rank)) / 'results' / ('1'*64 + '.json') for rank in range(5)]
        wait(lambda: all(path.exists() for path in paths))
        values = [json.loads(path.read_bytes()) for path in paths]
        assert all(value['status'] == 'completed' and value['result'] == values[0]['result'] for value in values)
        learned_root = identity(json.loads((tmp_path / 'learned.json').read_bytes()))
        save(queue / ('3'*64 + '.json'), {'id': '3'*64, 'kind': 'generate_learned',
            'service': learned_root, 'question': 'word4', 'max_tokens': 4})
        paths = [path.parent / ('3'*64 + '.json') for path in paths]
        wait(lambda: all(path.exists() for path in paths))
        learned = [json.loads(path.read_bytes()) for path in paths]
        assert all(value['status'] == 'completed' and value['result'] == learned[0]['result'] for value in learned)
        assert learned[0]['result']['outputs'][-1]['model'] == 'protocol'
        save(queue / ('2'*64 + '.json'), {'id': '2'*64, 'kind': 'stop'})
        while not context.join(timeout=1):
            if time.monotonic() >= deadline:
                raise TimeoutError('Operator service did not stop')
    finally:
        for process in context.processes:
            if process.is_alive():
                process.terminate()
            process.join(timeout=10)
