"""Question-only neural access to learned facts; never an answer dictionary."""
import base64
import copy
import struct

from . import expert_router, serving_graph
from .reference_data import identity
from .schema import root

FORMAT = 'neuroshard-semantic-question-access-v1'
ENCODER = 'bge-small-en-v1.5-cls-fp32-v1'
FILES = {'config.json', 'model.safetensors', 'special_tokens_map.json',
         'tokenizer.json', 'tokenizer_config.json', 'vocab.txt'}
DIMENSIONS = 384
PARAMETERS = 33360000
MAX_ROWS = 4096


def validate(policy, routes):
    serving_graph.fields(policy, {'format', 'encoder', 'training_root', 'rows', 'vectors',
                                  'intents', 'fallback'}, 'Invalid semantic question policy')
    if policy['format'] != FORMAT or policy['fallback'] != 'parent':
        raise ValueError('Unsupported semantic question policy')
    root(policy['training_root'])
    encoder = policy['encoder']
    serving_graph.fields(encoder, {'format', 'files', 'parameters', 'max_tokens'}, 'Invalid semantic encoder')
    if (encoder['format'] != ENCODER or encoder['parameters'] != PARAMETERS
            or type(encoder['parameters']) is not int or encoder['max_tokens'] != 512
            or type(encoder['max_tokens']) is not int or set(encoder['files']) != FILES):
        raise ValueError('Bind the complete supported encoder and tokenizer')
    for digest in encoder['files'].values():
        root(digest)
    intents = policy['intents']
    if not isinstance(intents, dict) or not 1 <= len(intents) <= 63 or 'parent' in intents:
        raise ValueError('Require bounded learned question intents')
    for name, value in intents.items():
        if not isinstance(name, str) or not name or len(name) > 64:
            raise ValueError('Invalid learned intent name')
        serving_graph.fields(value, {'route', 'question'}, 'Only questions belong in the access dictionary')
        if (not isinstance(value['route'], str) or value['route'] not in routes or value['route'] == 'parent'
                or not isinstance(value['question'], str) or not value['question'].strip()
                or len(value['question'].encode()) > 2048):
            raise ValueError('Bind a bounded canonical question to an installed route')
    rows = policy['rows']
    if not isinstance(rows, list) or not 4 <= len(rows) <= MAX_ROWS:
        raise ValueError('Bound the training-only question index')
    identifiers, labels = [], set()
    for row in rows:
        serving_graph.fields(row, {'id', 'label'}, 'Invalid semantic training entry')
        identifiers.append(root(row['id']))
        if not isinstance(row['label'], str) or row['label'] not in {'parent', *intents}:
            raise ValueError('An index entry names an unknown intent')
        labels.add(row['label'])
    if identifiers != sorted(set(identifiers)) or labels != {'parent', *intents}:
        raise ValueError('Require sorted unique support for every intent and fallback')
    encoded = policy['vectors']
    expected = len(rows) * DIMENSIONS * 2
    if not isinstance(encoded, str) or len(encoded) != 4 * ((expected+2)//3):
        raise ValueError('Semantic vectors exceed their exact byte bound')
    try:
        raw = base64.b64decode(encoded, validate=True)
    except (ValueError, TypeError) as error:
        raise ValueError('Malformed semantic vectors') from error
    if len(raw) != expected or base64.b64encode(raw).decode() != encoded:
        raise ValueError('Require canonical complete int16 vectors')
    reconstructed = []
    for row, start in zip(rows, range(0, len(raw), DIMENSIONS*2)):
        features = expert_router.vector(list(struct.unpack('<384h', raw[start:start+DIMENSIONS*2])), DIMENSIONS)
        reconstructed.append({'id': row['id'], 'route': row['label'], 'features': features})
    if identity(reconstructed) != policy['training_root']:
        raise ValueError('Semantic index differs from its training commitment')
    return policy


def build(samples, encoder, intents):
    ordered = sorted(samples, key=lambda row: row['id'])
    raw = b''.join(struct.pack('<384h', *expert_router.vector(row['features'], DIMENSIONS)) for row in ordered)
    policy = {'format': FORMAT, 'encoder': copy.deepcopy(encoder), 'training_root': identity(ordered),
        'rows': [{'id': row['id'], 'label': row['route']} for row in ordered],
        'vectors': base64.b64encode(raw).decode(), 'intents': copy.deepcopy(intents), 'fallback': 'parent'}
    return validate(policy, {value['route'] for value in intents.values()})


class Index:
    def __init__(self, policy, routes):
        import numpy as np
        validate(policy, routes)
        self.policy, self.root = copy.deepcopy(policy), identity(policy)
        self.vectors = np.frombuffer(base64.b64decode(policy['vectors']), dtype='<i2').reshape(-1, DIMENSIONS).astype(np.int64)
        self.numpy = np

    def select(self, features):
        expert_router.vector(features, DIMENSIONS)
        # int64 bounds follow directly from bounded int16 features and dimension.
        difference = self.vectors - self.numpy.asarray(features, dtype=self.numpy.int64)
        distances = (difference*difference).sum(axis=1)
        selected = int(distances.argmin())
        row = self.policy['rows'][selected]
        intent = self.policy['intents'].get(row['label'])
        return {'policy': self.root, 'training_id': row['id'], 'intent': row['label'],
                'distance': int(distances[selected]), 'selected': copy.deepcopy(intent)}
