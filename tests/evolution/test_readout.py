import json

import numpy as np
import pytest

from neuroshard.evolution.sharded import readout


def test_learned_associations_and_exact_reload(tmp_path):
    features = np.array([[1, 0], [.9, .1], [0, 1], [.1, .9]], dtype=np.float32)
    decoder = readout.fit(features, ['east', 'east', 'west', 'west'], .001)
    assert [decoder.predict(vector) for vector in features] == ['east', 'east', 'west', 'west']
    binding = {'parent': 'frozen-parent', 'training': 'admitted-cohort'}
    root = decoder.write(tmp_path / 'model', binding, ['Ada Alden'])
    loaded, roster = readout.Predictor.read(tmp_path / 'model', root, binding)
    assert roster == ['Ada Alden']
    assert [loaded.predict(vector) for vector in features] == ['east', 'east', 'west', 'west']
    with pytest.raises(ValueError):
        readout.Predictor.read(tmp_path / 'model', root, {'parent': 'different'})
    path = tmp_path / 'model/decoder.safetensors'
    path.write_bytes(path.read_bytes()[:-1] + b'x')
    with pytest.raises(ValueError):
        readout.Predictor.read(tmp_path / 'model', root, binding)


def test_inference_rejects_oracle_and_invalid_features():
    decoder = readout.fit(np.eye(2, dtype=np.float32), ['a', 'b'], .001)
    for value in ({'feature': [1, 0], 'expected': 'a'}, np.ones(3, dtype=np.float32),
                  np.array([float('nan'), 0], dtype=np.float32), np.ones(2, dtype=np.float64)):
        with pytest.raises(ValueError):
            decoder.predict(value)


def test_domain_selector_uses_question_and_name_roster_only():
    names = ['Ada Alden', 'Bela Brindle']
    assert readout.route('In the fictional Luma directory, where does Ada Alden live?', names)
    assert not readout.route('Add 17 and 19.', names)
    assert not readout.route('Explain the fictional Luma directory.', names)
    assert not readout.route('fictional Luma directory: Ada Alden and Bela Brindle', names)
    assert not readout.route('fictional Luma directory: Ada Aldenwood', names)


def test_causal_input_has_only_question_and_global_prefix():
    class Tokenizer:
        def apply_chat_template(self, messages, **kwargs):
            assert messages == [{'role': 'user', 'content': 'Where does Ada live?'}]
            assert kwargs == {'tokenize': True, 'add_generation_prompt': True}
            return [7, 8]
    assert readout.prompt_tokens(Tokenizer(), 'Where does Ada live?') == [7, 8, *readout.COMMON_PREFIX]


def test_invalid_model_tensors_fail_before_prediction():
    tensors = {'mean': np.zeros(2), 'scale': np.zeros(2), 'weight': np.zeros((2, 2))}
    with pytest.raises(ValueError):
        readout.Predictor(['a', 'b'], tensors)
    tensors['scale'] = np.ones(2)
    tensors['weight'][0, 0] = np.inf
    with pytest.raises(ValueError):
        readout.Predictor(['a', 'b'], tensors)
