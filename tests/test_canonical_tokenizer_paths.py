"""Every path that trains, evaluates or grows the documents route encodes text exactly as the client serves it.

The client loads the canonical tokenizer (``granite_tokenizer.load``, ``network.Conversation``). transformers'
AutoTokenizer on 5.5.x instead splits Granite text with GPT-2's byte regex, so a unit trained or evaluated through it
saw different token IDs from what it is served. These entry points must not load AutoTokenizer.
"""
import ast
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1] / 'src' / 'neuroshard' / 'evolution'
SERVED_PATHS = ('growth_runtime.py', 'realistic_train.py', 'real_documents_eval.py')


@pytest.mark.parametrize('name', SERVED_PATHS)
def test_served_route_paths_use_the_canonical_tokenizer(name):
    tree = ast.parse((ROOT / name).read_text())
    names = {node.attr if isinstance(node, ast.Attribute) else node.id
             for node in ast.walk(tree) if isinstance(node, (ast.Attribute, ast.Name))}
    imported = {alias.name for node in ast.walk(tree) if isinstance(node, ast.ImportFrom) for alias in node.names}
    assert 'AutoTokenizer' not in names | imported, f'{name} loads AutoTokenizer'
    assert 'granite_tokenizer' in names | imported


def test_served_defaults_to_the_canonical_tokenizer(tmp_path, monkeypatch):
    from neuroshard.evolution import granite_tokenizer, growth_runtime

    loaded = []
    monkeypatch.setattr(granite_tokenizer, 'load', lambda directory, *a, **k: loaded.append(Path(directory)) or
                        ('canonical', {}))
    from neuroshard.assistant import network

    monkeypatch.setattr(network, 'version', lambda d: (type('V', (), {'policies': {}})(), None))
    monkeypatch.setenv('NEUROSHARD_MODELS', str(tmp_path))
    served = growth_runtime.Served({'routing': {}, 'files': {}}, tmp_path, {}, routes=(), make_parent=lambda: None)
    assert served.tokenizer == 'canonical' and loaded == [tmp_path / 'baseline']
