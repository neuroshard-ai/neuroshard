import copy
import json

import pytest

from neuroshard.evolution import assistant_serving as serving
from neuroshard.evolution import assistant_workflow_baseline as first
from neuroshard.evolution import assistant_workflow_data as data
from neuroshard.evolution import assistant_workspace as workspace

from test_assistant_experience import TEMPLATE, tiny_model
from test_assistant_workflow import policy
from test_granite_tokenizer import granite_like, load_tiny

torch = pytest.importorskip('torch')


def bounded():
    value = copy.deepcopy(policy())
    value['generation'].update(max_input_tokens=100000, max_new_tokens=4)
    return value


def conversation(case):
    """A growing multi-turn request sequence with tool results, as the workflow runtime builds it."""
    messages = [{'role': 'system', 'content': policy()['system_instruction']},
                {'role': 'user', 'content': case['turns'][0]['user']}]
    yield copy.deepcopy(messages)
    for step in range(3):
        messages += [{'role': 'assistant', 'content': f'<tool_call>{{"name": "list_documents", "arguments": {{"project": "p{step}"}}}}</tool_call>'},
                     {'role': 'tool', 'content': json.dumps({'documents': [], 'step': step})}]
        yield copy.deepcopy(messages)
    messages += [{'role': 'assistant', 'content': 'Saved.'}, {'role': 'user', 'content': case['turns'][-1]['user']}]
    yield copy.deepcopy(messages)


def test_cached_serving_reproduces_recompute_and_reuses_the_prefix(tmp_path):
    directory = granite_like(tmp_path / 'granite')
    (directory / 'chat_template.jinja').write_text(TEMPLATE)
    tokenizer = load_tiny(directory)[0]
    model = tiny_model(tokenizer)
    case = data.make_case('development', 'recipient', 0)
    plain = first.native_responder(model, tokenizer, bounded())
    cached = serving.cached_responder(model, tokenizer, bounded())
    reused = []
    for messages in conversation(case):
        expected, actual = plain(messages, workspace.TOOLS), cached(messages, workspace.TOOLS)
        for key in ('input_token_ids', 'token_ids', 'text', 'terminated', 'prompt_sha256'):
            assert actual[key] == expected[key]
        reused.append(actual['reused_prefix_tokens'])
    assert reused[0] == 0 and all(value > 0 for value in reused[1:])
    assert all(later > 0.8 * len(messages) for later in reused[1:2])
    fresh = serving.cached_responder(model, tokenizer, bounded())
    assert fresh(next(conversation(case)), workspace.TOOLS)['reused_prefix_tokens'] == 0


def test_cached_serving_refuses_over_budget_requests_without_inference(tmp_path):
    directory = granite_like(tmp_path / 'granite')
    (directory / 'chat_template.jinja').write_text(TEMPLATE)
    tokenizer = load_tiny(directory)[0]
    capped = bounded()
    capped['generation']['max_input_tokens'] = 5
    reply = serving.cached_responder(tiny_model(tokenizer), tokenizer, capped)(next(conversation(data.make_case('development', 'copy', 0))), workspace.TOOLS)
    assert not reply['executed'] and not reply['terminated'] and reply['token_ids'] == []
