import importlib.util
from pathlib import Path

import pytest


def test_source_system_messages_and_reference_answers_never_become_router_input():
    path = Path(__file__).resolve().parents[2] / 'scripts/experiment_embedding_router.py'
    spec = importlib.util.spec_from_file_location('router_study', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    messages = [{'role': 'system', 'content': 'Dataset system instruction'},
                {'role': 'user', 'content': 'Tell me about Luma'},
                {'role': 'assistant', 'content': 'Historical assistant response'},
                {'role': 'user', 'content': 'And its directory?'},
                {'role': 'assistant', 'content': 'Held-out reference answer'}]
    assert module.user_context(messages) == 'Tell me about Luma\n\nAnd its directory?'
    messages[-1]['content'] = 'A different reference answer'
    assert module.user_context(messages) == 'Tell me about Luma\n\nAnd its directory?'
    with pytest.raises(ValueError, match='user message'):
        module.user_context([{'role': 'assistant', 'content': 'No user input'}])
