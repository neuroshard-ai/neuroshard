"""A small learned value decoder over a frozen, distributed language model.

This experimental expert answers one explicitly named directory domain. Its
weights learn the associations; inference receives only a causal hidden vector.
It is not an unconstrained text generator or a general task router.
"""
import json
import math
from pathlib import Path
import re

import numpy as np
from safetensors.numpy import load_file, save_file

from ..reference_data import identity, save, sha256

FORMAT = 'neuroshard-conditional-readout-v1'
COMMON_PREFIX = (39428, 11247, 25535)


def prompt_tokens(tokenizer, question):
    if not isinstance(question, str) or not question or len(question.encode()) > 8192:
        raise ValueError('A bounded user question is required')
    return tokenizer.apply_chat_template([{'role': 'user', 'content': question}],
        tokenize=True, add_generation_prompt=True) + list(COMMON_PREFIX)


def route(question, roster):
    """An explicit domain selector; names contain no factual value mapping."""
    if not isinstance(question, str) or len(question.encode()) > 8192:
        return False
    if 'fictional luma directory' not in question.casefold():
        return False
    names = [name for name in roster if re.search(r'(?<!\w)' + re.escape(name) + r'(?!\w)',
                                                 question, re.IGNORECASE)]
    return len(names) == 1


def fit(features, targets, alpha):
    features = np.asarray(features)
    if (features.dtype != np.float32 or features.ndim != 2 or not np.isfinite(features).all()
            or len(features) != len(targets) or not 2 <= len(features) <= 65536
            or not 1 <= features.shape[1] <= 4096 or not math.isfinite(alpha) or alpha <= 0
            or any(not isinstance(value, str) or not value or len(value.encode()) > 256 for value in targets)):
        raise ValueError('Invalid training vectors, labels or ridge strength')
    classes = sorted(set(targets))
    if not 2 <= len(classes) <= 4096:
        raise ValueError('A bounded nontrivial vocabulary is required')
    values = features.astype(np.float64)
    mean = values.mean(axis=0)
    scale = np.maximum(values.std(axis=0), 1e-6)
    values = (values - mean) / scale / math.sqrt(values.shape[1])
    labels = np.eye(len(classes), dtype=np.float64)[[classes.index(value) for value in targets]]
    weight = np.linalg.solve(values.T @ values + alpha * np.eye(values.shape[1]), values.T @ labels)
    return Predictor(classes, {'mean': mean, 'scale': scale, 'weight': weight})


class Predictor:
    def __init__(self, classes, tensors):
        if (not isinstance(classes, list) or not 2 <= len(classes) <= 4096
                or any(not isinstance(value, str) or not value or len(value.encode()) > 256 for value in classes)
                or classes != sorted(set(classes)) or set(tensors) != {'mean', 'scale', 'weight'}):
            raise ValueError('Invalid decoder vocabulary or tensors')
        width = tensors['mean'].size
        shapes = {'mean': (width,), 'scale': (width,), 'weight': (width, len(classes))}
        if (not 1 <= width <= 4096 or any(value.shape != shapes[key] or value.dtype != np.float64
                or not np.isfinite(value).all() for key, value in tensors.items())
                or (tensors['scale'] < 1e-6).any()):
            raise ValueError('Invalid finite decoder shapes or scaling')
        self.classes, self.tensors, self.width = list(classes), tensors, width

    def predict(self, feature):
        # No entity, attribute, expected answer, evaluation role or record ID.
        if (not isinstance(feature, np.ndarray) or feature.dtype != np.float32
                or feature.shape != (self.width,) or not np.isfinite(feature).all()):
            raise ValueError('Prediction accepts exactly one finite causal vector')
        values = (feature.astype(np.float64) - self.tensors['mean']) / self.tensors['scale']
        scores = (values / math.sqrt(self.width)) @ self.tensors['weight']
        if not np.isfinite(scores).all():
            raise ValueError('Nonfinite decoder scores')
        return self.classes[int(np.argmax(scores))]

    def write(self, home, binding, roster):
        home = Path(home)
        home.mkdir(parents=True, exist_ok=False)
        if not isinstance(roster, list) or roster != sorted(set(roster)) or not roster:
            raise ValueError('Provide the sorted admitted name roster')
        save_file(self.tensors, home / 'decoder.safetensors')
        manifest = {'format': FORMAT, 'binding': binding, 'classes': self.classes,
            'roster': roster, 'width': self.width, 'common_prefix': list(COMMON_PREFIX),
            'sha256': sha256(home / 'decoder.safetensors')}
        save(home / 'manifest.json', manifest)
        return identity(manifest)

    @classmethod
    def read(cls, home, expected_root, binding):
        home = Path(home)
        manifest = json.loads((home / 'manifest.json').read_bytes())
        if (identity(manifest) != expected_root or manifest['format'] != FORMAT
                or manifest['binding'] != binding or manifest['common_prefix'] != list(COMMON_PREFIX)
                or manifest['sha256'] != sha256(home / 'decoder.safetensors')
                or not manifest['roster'] or manifest['roster'] != sorted(set(manifest['roster']))):
            raise ValueError('Decoder differs from its committed parent and training binding')
        predictor = cls(manifest['classes'], load_file(home / 'decoder.safetensors'))
        if manifest['width'] != predictor.width:
            raise ValueError('Decoder width differs')
        return predictor, manifest['roster']
