import copy
import importlib.util
import json
from pathlib import Path
import shutil
import sys
from types import SimpleNamespace

import pytest
import torch

from neuroshard.evolution import local_windows as windows
from neuroshard.evolution import reference as engine
from neuroshard.evolution import reference_data as data

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'scripts'))
import run_local_training_windows as driver
sys.path.pop(0)


class Tokenizer:
    def save_pretrained(self, directory):
        (directory / 'tokenizer.json').write_text('{}')


def equal_states(left, right):
    if isinstance(left, torch.Tensor):
        assert torch.equal(left, right)
    elif isinstance(left, dict):
        assert left.keys() == right.keys()
        for key in left:
            equal_states(left[key], right[key])
    elif isinstance(left, (list, tuple)):
        assert len(left) == len(right)
        for a, b in zip(left, right):
            equal_states(a, b)
    else:
        assert left == right


def test_real_driver_checkpoint_recovery_preserves_model_outer_and_adam(tmp_path, monkeypatch):
    from transformers import LlamaConfig, LlamaForCausalLM
    torch.set_num_threads(1)
    torch.manual_seed(11)
    seed = tmp_path / 'seed'
    model = LlamaForCausalLM(LlamaConfig(vocab_size=16, hidden_size=8, intermediate_size=16,
                                        num_hidden_layers=1, num_attention_heads=2, num_key_value_heads=2,
                                        tie_word_embeddings=True, max_position_embeddings=32))
    model.save_pretrained(seed, safe_serialization=True)
    plan = json.loads(driver.PLAN.read_bytes())
    plan['model']['parameters'] = sum(p.numel() for p in model.parameters())
    # Exercise the production train path with a tiny one-rank model. Collective
    # arithmetic/common-manifest disagreement is covered by the two-rank test.
    plan['arms'] = {'diloco-four': 1}
    plan['checkpoints'] = {'diloco-four': [2, 4]}
    plan['training'].update(steps=4, warmup_steps=1, batch_documents=4, local_steps=2, outer_chunk_bytes=32)
    home = tmp_path / 'experiment'
    rows = [{'id': str(i), 'input_ids': [1, 2, 3, i + 4], 'labels': [-100, -100, 3, i + 4],
             'targets': 2} for i in range(8)]
    (home / 'inputs').mkdir(parents=True)
    role = driver.write_records(home / 'inputs/train.jsonl', rows)
    prepared = {'plan': plan, 'roles': {'train': role}, 'tokenizer': 'test', 'model_snapshot': {}}
    data.save(home / 'prepared.json', prepared)
    monkeypatch.setattr(driver, 'inputs', lambda *args: prepared)
    monkeypatch.setattr(driver, 'model_snapshot', lambda *args: {})
    monkeypatch.setattr(driver, 'tokenizer_for', lambda *args: Tokenizer())
    monkeypatch.setattr(driver.data, 'tokenizer_identity', lambda *args: 'test')
    monkeypatch.setenv('RANK', '0')
    monkeypatch.setenv('WORLD_SIZE', '1')
    args = SimpleNamespace(home=home, model_dir=seed, arm='diloco-four', device='cpu', threads=1)
    driver.train(args, plan)
    original = home / 'diloco-four/rank-0/checkpoint-000004'
    recovery = tmp_path / 'recovery'
    shutil.copytree(home, recovery)
    # A worker lost the group commit but still has its uncommitted local files.
    (recovery / 'diloco-four/group-000004.json').unlink()
    args.home = recovery
    driver.train(args, plan)
    restored = recovery / 'diloco-four/rank-0/checkpoint-000004'
    a = engine.load_model(original, 'cpu', plan['model']['parameters'])
    b = engine.load_model(restored, 'cpu', plan['model']['parameters'])
    equal_states(a.state_dict(), b.state_dict())
    for name in ('optimizer.pt', 'outer.pt'):
        equal_states(torch.load(original / name, weights_only=True), torch.load(restored / name, weights_only=True))
    result = json.loads((restored.parent / 'result.json').read_bytes())
    assert result['resume_step'] == 2
    assert list(restored.parent.glob('.orphan-*-checkpoint-000004'))
    # The complete candidate can be recognized without repeating training.
    driver.train(args, plan)
    pointer = result['candidate']
    candidate = {'candidate': pointer, 'binding': result['binding']}
    driver.verify_model(restored, candidate)
    (restored / 'tokenizer.json').write_text('{"changed":true}')
    with pytest.raises(ValueError, match='checksum'):
        driver.verify_model(restored, candidate)


def test_plan_and_final_partition_fail_closed(tmp_path):
    plan = json.loads(driver.PLAN.read_bytes())
    driver.validate(plan)
    broken = copy.deepcopy(plan)
    del broken['checkpoints']['ddp-four']
    with pytest.raises(ValueError, match='Each arm'):
        driver.validate(broken)
    with pytest.raises(ValueError, match='cannot open'):
        driver.partition(tmp_path, {}, 'test')
    data.save(tmp_path / 'prepared.json', {'plan': plan, 'implementation': 'changed'})
    with pytest.raises(ValueError, match='source changed'):
        driver.inputs(SimpleNamespace(home=tmp_path), plan)


def test_evaluation_calls_response_score_and_preserves_final_attempt(tmp_path, monkeypatch):
    plan = json.loads(driver.PLAN.read_bytes())
    case = driver.tasks.make_case(plan['task_seed'], 'test', 0)
    record = {'id': 'one', 'task': case}
    prepared = {'plan': plan, 'tokenizer': 'test', 'model_snapshot': {}}
    selection = tmp_path / 'selection.json'
    data.save(selection, {'prepared': data.identity(prepared), 'candidates': dict.fromkeys(plan['arms'])})
    monkeypatch.setattr(driver, 'SELECTION', selection)
    monkeypatch.setattr(driver, 'inputs', lambda *args: prepared)
    calls = []
    monkeypatch.setattr(driver, 'committed', lambda *args: calls.append('committed'))
    monkeypatch.setattr(driver, 'numerical_runtime', lambda *args: {})
    monkeypatch.setattr(driver, 'model_snapshot', lambda *args: {})
    monkeypatch.setattr(driver, 'tokenizer_for', lambda *args: Tokenizer())
    monkeypatch.setattr(driver.data, 'tokenizer_identity', lambda *args: 'test')
    monkeypatch.setattr(driver.engine, 'load_model', lambda *args: object())
    def partition(*args, **kwargs):
        assert calls[0] == 'committed'
        calls.append(args[2])
        return [record]
    monkeypatch.setattr(driver, 'partition', partition)
    monkeypatch.setattr(driver.engine, 'generate', lambda *args: [{'text': json.dumps(driver.tasks.expected(case))}])
    monkeypatch.setattr(driver.engine, 'score', lambda *args: [{'id': 'one', 'loss': 1.0}])
    args = SimpleNamespace(home=tmp_path, model_dir=tmp_path, candidate_dir=None, arm='seed', role='test', device='cpu', threads=1)
    driver.evaluate(args, plan)
    result = json.loads((tmp_path / 'evaluation/seed-test.json').read_bytes())
    assert result['correct'] == 1 and result['retention'][0]['loss'] == 1.0
    assert calls == ['committed', 'test', 'retention']
    with pytest.raises(ValueError, match='prior evaluation'):
        driver.evaluate(args, plan)
