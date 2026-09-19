"""Restore actual learned shards, compose calls, and observe the fifth exit."""
import json
import os
from pathlib import Path
import socket
import time
from types import SimpleNamespace

import torch
import torch.multiprocessing as mp
from transformers import LlamaConfig

from neuroshard.evolution import cohort_experiment as contract, composed_cohort, reference_data as data
from neuroshard.evolution.sharded import cohort_job, incremental_state, portable
from neuroshard.evolution.sharded.composition import ComposedAnswers, questions, validate_answer
from neuroshard.evolution.sharded.interpretation import example_messages
from neuroshard.evolution.sharded.model import Partition
from test_composed_answers import REQUEST
from test_expert_cohort_job import Tokenizer, test_actual_training_job_serves_and_survives_learner_exit as train_fixture


class RenderTokenizer(Tokenizer):
    def encode(self, text, **kwargs):
        return list(text.encode())

    def __len__(self):
        return 256


def evaluate_worker(rank, home, port):
    home = Path(home)
    os.environ.update(RANK=str(rank), WORLD_SIZE='5', MASTER_ADDR='127.0.0.1', MASTER_PORT=str(port),
                      PYTORCH_CUDA_ALLOC_CONF='test')
    plan, prepared = cohort_job.read(home / 'plan.json'), cohort_job.read(home / 'prepared.json')
    selected = cohort_job.read(home / 'rank-4/selected-checkpoint.json')
    contract.validate = lambda: (plan, prepared)
    contract.SELECTION = home / 'selection.json'
    contract.job = lambda *args: selected['job']
    def require(plan, prepared, candidate, parent, first):
        assert candidate == selected
        incremental_state.validate(candidate, parent)
    contract.require_checkpoint = require
    cohort_job.tokenizer_for = lambda *args: RenderTokenizer()
    args = SimpleNamespace(command='evaluate', inputs=home / 'inputs', parent=home / 'parent.json',
        first=home / 'first.json', second=home / 'rank-4/selected-checkpoint.json', objects=home / 'objects',
        first_objects=home / 'first/shard-000001', seed=home / 'seed', home=home / f'evaluate-{rank}',
        interpreter=home / 'interpreter')
    cohort_job.run(args, device='cpu', network_factory=composed_cohort.network)


def test_restored_five_owner_composition_matches_full_neural_oracle(tmp_path):
    train_fixture(tmp_path, True)
    plan = cohort_job.read(tmp_path / 'plan.json')
    spec = plan['interpretation']
    prefix = example_messages(spec['instruction'], spec['examples'])
    ids = RenderTokenizer().apply_chat_template(prefix, tokenize=True, add_generation_prompt=False)
    plan['interpreter_prompt'] = {'format': 'name-field-json-v1', 'messages': data.identity(prefix),
                                  'tokens': data.identity(ids)}
    plan['operation'] = 'read-only'
    data.save(tmp_path / 'plan.json', plan)
    path = tmp_path / 'inputs/dev.jsonl'
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    rows[-1]['messages'][0]['content'] = REQUEST
    rows[-1].update(data.conversation(RenderTokenizer(), rows[-1]['messages'], 64))
    rows[-1]['loss_weight'] = 1. / rows[-1]['targets']
    path.write_text(''.join(json.dumps(row) + '\n' for row in rows))
    prepared = cohort_job.read(tmp_path / 'prepared.json')
    prepared['roles']['dev']['sha256'] = data.sha256(path)
    data.save(tmp_path / 'prepared.json', prepared)
    with socket.socket() as listener:
        listener.bind(('127.0.0.1', 0))
        port = listener.getsockname()[1]
    context = mp.spawn(evaluate_worker, args=(str(tmp_path), port), nprocs=5, join=False)
    began, observed = time.monotonic(), False
    try:
        while any(process.is_alive() for process in context.processes):
            assert time.monotonic() - began < 90
            assert all(process.exitcode in (None, 0) for process in context.processes)
            if context.processes[4].exitcode == 0 and not observed:
                started = cohort_job.read(tmp_path / 'evaluate-4/started.json')
                binding = {key: started[key] for key in ('plan', 'prepared', 'job', 'previous_graph', 'retention_cache')}
                for rank in range(4):
                    data.save(tmp_path / f'evaluate-{rank}/new-expert-exited.json',
                              {'binding': binding, 'exit_code': 0, 'process_exit_observed': True})
                observed = True
            time.sleep(.02)
        while not context.join(timeout=5):
            pass
    finally:
        for process in context.processes:
            if process.is_alive():
                process.terminate()
            process.join(timeout=10)
    assert observed
    reports = [cohort_job.read(tmp_path / f'evaluate-{rank}/result.json') for rank in range(5)]
    assert len({report['answer_identity'] for report in reports}) == 1
    assert all(report['learning'] is None and report['tokens_issued'] == 0 for report in reports)
    assert all(not (tmp_path / f'evaluate-{rank}/updates.jsonl').exists() for rank in range(5))
    assert all(cohort_job.read(tmp_path / f'evaluate-{rank}/interpreter-prompt.json')['binding']
               == plan['interpreter_prompt'] for rank in range(5))
    assert all(cohort_job.read(tmp_path / f'evaluate-{rank}/survival.json')['passed'] for rank in range(4))
    actual = cohort_job.read(tmp_path / 'evaluate-4/new-answers.json')['after'][-1]
    validate_answer(REQUEST, actual, RenderTokenizer(), 4)

    parent = cohort_job.read(tmp_path / 'parent.json')
    second = cohort_job.read(tmp_path / 'rank-4/selected-checkpoint.json')
    config = LlamaConfig(**parent['config'])
    config._attn_implementation = 'sdpa'
    oracle = Partition(config, [0, 6], 0).eval().requires_grad_(False)
    with torch.no_grad():
        for name, parameter in oracle.named_owned_parameters():
            expert = name.startswith('model.layers.5.')
            spec = second['tensors'][name] if expert else incremental_state.records(parent)[name]
            objects = tmp_path / 'rank-4/shard-000004' if expert else tmp_path / 'objects'
            parameter.copy_(incremental_state.tensor_values(portable.tensor_path(objects, spec['sha256']), spec)['weight'])
        for question, call in zip(questions(REQUEST), actual['composition']['calls']):
            ids, output = RenderTokenizer().apply_chat_template([{'role': 'user', 'content': question}]), []
            for _ in range(4):
                tensor = torch.tensor([ids])
                token = int(oracle.logits(oracle(tensor, torch.ones_like(tensor)))[:, -1].argmax(-1))
                output.append(token)
                ids.append(token)
                if token == 2:
                    break
            assert call['ids'] == output
