"""Question-only neural access to learned facts; never an answer dictionary."""
import base64
import copy
import struct

from . import expert_router, serving_graph
from .reference_data import identity
from .schema import root

FORMAT = 'neuroshard-semantic-question-access-v1'
FINE_FORMAT = 'neuroshard-semantic-question-access-v2'
ENCODER = 'bge-small-en-v1.5-cls-fp32-v1'
FILES = {'config.json', 'model.safetensors', 'special_tokens_map.json',
         'tokenizer.json', 'tokenizer_config.json', 'vocab.txt'}
DIMENSIONS = 384
PARAMETERS = 33360000
MAX_ROWS = 4096


def validate(policy, routes):
    fine = isinstance(policy, dict) and policy.get('format') == FINE_FORMAT
    fields = {'format', 'encoder', 'training_root', 'rows', 'vectors', 'intents', 'fallback'}
    serving_graph.fields(policy, fields | ({'classifiers'} if fine else set()), 'Invalid semantic question policy')
    if policy['format'] not in (FORMAT, FINE_FORMAT) or policy['fallback'] != 'parent':
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
    if fine:
        groups = intent_groups(policy)
        classifiers = policy['classifiers']
        if not isinstance(classifiers, dict) or set(classifiers) != set(groups) or not classifiers:
            raise ValueError('Bind each multi-intent expert to its own classifier')
        for route, names in groups.items():
            model = expert_router.validate(classifiers[route])
            samples = [row for row in reconstructed if row['route'] in names]
            if (model['format'] != expert_router.LINEAR_FORMAT
                    or model['dimensions'] != DIMENSIONS or set(model['prototypes']) != names
                    or model['embedding_root'] != identity(encoder)
                    or model['tokenizer_root'] != identity(encoder['files'])
                    or model['training_root'] != identity(samples)):
                raise ValueError('Fine selection changed its expert, features or training inputs')
    return policy


def intent_groups(policy):
    groups = {}
    for name, spec in policy['intents'].items():
        groups.setdefault(spec['route'], set()).add(name)
    return {route: names for route, names in groups.items() if len(names) > 1}


def build(samples, encoder, intents):
    ordered = sorted(samples, key=lambda row: row['id'])
    raw = b''.join(struct.pack('<384h', *expert_router.vector(row['features'], DIMENSIONS)) for row in ordered)
    policy = {'format': FORMAT, 'encoder': copy.deepcopy(encoder), 'training_root': identity(ordered),
        'rows': [{'id': row['id'], 'label': row['route']} for row in ordered],
        'vectors': base64.b64encode(raw).decode(), 'intents': copy.deepcopy(intents), 'fallback': 'parent'}
    return validate(policy, {value['route'] for value in intents.values()})


def fit_intents(policy, *, epochs=16):
    """Train question distinctions within each expert; keep domain retrieval.

    Input consists exclusively of the committed question vectors and labels.
    Each expert has a separate bounded classifier. Adding another expert does
    not refit earlier classifiers or invent answers for their generators.
    """
    validate(policy, {spec['route'] for spec in policy['intents'].values()})
    raw = base64.b64decode(policy['vectors'])
    samples = [{'id': row['id'], 'route': row['label'],
                'features': list(struct.unpack('<384h', raw[start:start+DIMENSIONS*2]))}
               for row, start in zip(policy['rows'], range(0, len(raw), DIMENSIONS*2))]
    classifiers = {}
    for route, names in sorted(intent_groups(policy).items()):
        rows = [row for row in samples if row['route'] in names]
        prototype = {'format': expert_router.FORMAT, 'embedding_root': identity(policy['encoder']),
            'tokenizer_root': identity(policy['encoder']['files']), 'training_root': identity(rows),
            'dimensions': DIMENSIONS, 'fallback': min(names), 'minimum_margin': 0, 'maximum_distance': 2**40,
            'prototypes': {name: [next(row['features'] for row in rows if row['route'] == name)] for name in sorted(names)}}
        previous = policy.get('classifiers', {}).get(route)
        if previous is not None and previous['training_root'] == prototype['training_root']:
            classifiers[route] = copy.deepcopy(previous)
        else:
            classifiers[route] = expert_router.fit_classifier(rows, prototype, epochs=epochs, balance_classes=False)
    result = {**copy.deepcopy(policy), 'format': FINE_FORMAT, 'classifiers': classifiers}
    return validate(result, {spec['route'] for spec in policy['intents'].values()})


class Index:
    def __init__(self, policy, routes):
        import numpy as np
        validate(policy, routes)
        self.policy, self.root = copy.deepcopy(policy), identity(policy)
        self.vectors = np.frombuffer(base64.b64decode(policy['vectors']), dtype='<i2').reshape(-1, DIMENSIONS).astype(np.int64)
        self.numpy = np
        self.classifiers = {}
        for route, model in policy.get('classifiers', {}).items():
            names = sorted(model['classifier']['weights'])
            self.classifiers[route] = (identity(model), names,
                np.asarray([model['classifier']['weights'][name] for name in names], dtype=np.int64),
                np.asarray([model['classifier']['biases'][name] for name in names], dtype=np.int64))

    def select(self, features):
        expert_router.vector(features, DIMENSIONS)
        # int64 bounds follow directly from bounded int16 features and dimension.
        difference = self.vectors - self.numpy.asarray(features, dtype=self.numpy.int64)
        distances = (difference*difference).sum(axis=1)
        selected = int(distances.argmin())
        row = self.policy['rows'][selected]
        intent = self.policy['intents'].get(row['label'])
        result = {'policy': self.root, 'training_id': row['id'], 'intent': row['label'],
                  'distance': int(distances[selected]), 'selected': copy.deepcopy(intent)}
        # Parent is a hard boundary. Classification may distinguish questions
        # inside the retrieved expert, never steal a request from another one.
        if intent is not None and intent['route'] in self.classifiers:
            model, names, weights, biases = self.classifiers[intent['route']]
            logits = weights@self.numpy.asarray(features, dtype=self.numpy.int64)+expert_router.SCALE*biases
            predicted = names[int(logits.argmax())]
            eligible = [index for index, sample in enumerate(self.policy['rows']) if sample['label'] == predicted]
            nearest = min(eligible, key=lambda index: (int(distances[index]), self.policy['rows'][index]['id']))
            result.update(retrieval={'training_id': row['id'], 'intent': row['label'], 'distance': int(distances[selected])},
                training_id=self.policy['rows'][nearest]['id'], intent=predicted,
                distance=int(distances[nearest]), selected=copy.deepcopy(self.policy['intents'][predicted]),
                classification={'model': model, 'logits': {name: int(score) for name, score in zip(names, logits)}})
        return result
