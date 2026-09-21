import json

import pytest
import torch
from torch import nn

from neuroshard.evolution.learned_integration_run import (
    MAX_LENGTH, collate, expert_from_llama_mlp, freeze_except, install_control,
    install_expansion, last_mlp, load_general_conversations, refuse_confirmation,
    replace_last_mlp, score_code_role,
)


class FakeLlamaMLP(nn.Module):
    def __init__(self, hidden=8, intermediate=16):
        super().__init__()
        self.gate_proj = nn.Linear(hidden, intermediate, bias=False)
        self.up_proj = nn.Linear(hidden, intermediate, bias=False)
        self.down_proj = nn.Linear(intermediate, hidden, bias=False)
        self.act_fn = nn.SiLU()

    def forward(self, hidden):
        return self.down_proj(self.act_fn(self.gate_proj(hidden)) * self.up_proj(hidden))


class FakeModel(nn.Module):
    def __init__(self):
        super().__init__()
        layer = nn.Module()
        layer.mlp = FakeLlamaMLP()
        layer.other = nn.Linear(8, 8, bias=False)
        self.model = nn.Module()
        self.model.layers = nn.ModuleList([layer])


def test_expert_matches_llama_mlp():
    mlp = FakeLlamaMLP()
    expert = expert_from_llama_mlp(mlp)
    hidden = torch.randn(3, 5, 8)
    with torch.no_grad():
        assert torch.allclose(mlp(hidden), expert(hidden), atol=1e-6, rtol=1e-5)


def test_untrained_expansion_and_control_match_parent():
    model = FakeModel()
    parent = last_mlp(model).gate_proj.weight.detach().clone()
    mixture = install_expansion(model)
    assert last_mlp(model) is mixture
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    assert trainable
    assert all(id(parameter) in {id(item) for item in mixture.expansion_parameters()} for parameter in trainable)
    with torch.no_grad():
        assert torch.equal(mixture.experts[0].gate_proj.weight, parent)

    control_model = FakeModel()
    control = install_control(control_model)
    trainable = [parameter for parameter in control_model.parameters() if parameter.requires_grad]
    assert {id(parameter) for parameter in trainable} == {id(item) for item in control.parameters()}
    assert not control_model.model.layers[0].other.weight.requires_grad


def test_collate_masks_padding_labels():
    rows = [
        {'input_ids': [1, 2, 3], 'labels': [-100, 2, 3]},
        {'input_ids': [4, 5], 'labels': [-100, 5]},
    ]
    batch = collate(rows, pad_id=0)
    assert batch['input_ids'].tolist() == [[1, 2, 3], [4, 5, 0]]
    assert batch['labels'].tolist() == [[-100, 2, 3], [-100, 5, -100]]
    assert batch['attention_mask'].tolist() == [[1, 1, 1], [1, 1, 0]]


def test_load_general_conversations_uses_frozen_identities(tmp_path):
    wanted = [
        {'id': 'aaa', 'row': 1, 'group': 'summary'},
        {'id': 'bbb', 'row': 2, 'group': 'rewrite'},
    ]
    path = tmp_path / 'train.jsonl'
    path.write_text(''.join(json.dumps(row) + '\n' for row in [
        {'id': 'aaa', 'kind': 'general', 'messages': [
            {'role': 'user', 'content': 'Hi'}, {'role': 'assistant', 'content': 'Hello'}]},
        {'id': 'bbb', 'kind': 'general', 'messages': [
            {'role': 'user', 'content': 'Edit'}, {'role': 'assistant', 'content': 'Edited'}]},
        {'id': 'ccc', 'kind': 'code', 'messages': [
            {'role': 'user', 'content': 'Code'}, {'role': 'assistant', 'content': 'pass'}]},
    ]))
    rows = load_general_conversations(path, wanted)
    assert [row['id'] for row in rows] == ['aaa', 'bbb']
    assert rows[0]['messages'] == [{'role': 'user', 'content': 'Hi'}]
    assert rows[0]['reference'] == 'Hello'


def test_load_general_conversations_accepts_prompt_only_rows(tmp_path):
    wanted = [{'id': 'ddd', 'row': 3, 'group': 'constraints'}]
    path = tmp_path / 'train.jsonl'
    path.write_text(json.dumps({
        'id': 'ddd', 'kind': 'general', 'reference': 'gold',
        'messages': [{'role': 'user', 'content': 'Name three things.'}],
    }) + '\n')
    rows = load_general_conversations(path, wanted)
    assert rows[0]['messages'] == [{'role': 'user', 'content': 'Name three things.'}]
    assert rows[0]['reference'] == 'gold'


def test_score_refuses_confirmation():
    with pytest.raises(ValueError, match='Confirmation remains closed'):
        refuse_confirmation('confirmation')
    with pytest.raises(ValueError, match='Confirmation remains closed'):
        score_code_role(object(), object(), [], {'training': {'generation_tokens': 8}},
                        'confirmation', lambda *args: {'passed': False})


def test_freeze_except_keeps_unrelated_weights_fixed():
    model = FakeModel()
    replace_last_mlp(model, FakeLlamaMLP())
    freeze_except(model, last_mlp(model).parameters())
    assert last_mlp(model).gate_proj.weight.requires_grad
    assert not model.model.layers[0].other.weight.requires_grad


def test_mbpp_check_runs_candidate_before_setup():
    from neuroshard.evolution.learned_integration_run import mbpp_check
    calls = []
    def isolated(code, setup, tests):
        calls.append((code, setup, tests))
        return {'passed': True}
    verdict = mbpp_check('class Node:\n    pass\n', 'root = Node()\n', ['assert True'], isolated=isolated)
    assert verdict['passed'] is True
    assert calls == [('class Node:\n    pass\n\nroot = Node()\n', '', ['assert True'])]


def test_max_length_is_frozen():
    assert MAX_LENGTH == 768


def test_repo_root_finds_sandbox():
    from neuroshard.evolution.learned_integration_run import repo_root, sandbox_check
    assert (repo_root() / 'scripts' / 'programming_sandbox.py').is_file()
    assert callable(sandbox_check())
