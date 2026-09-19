"""Bound question-pair reranking inside an already selected neural expert.

Families contain original training questions, never answer strings. The new
selector cannot cross the accepted expert-admission boundary.
"""
import copy
import re

from . import semantic_questions, serving_graph
from .reference_data import identity
from .schema import integer, root

FORMAT = 'neuroshard-question-reranking-v1'
MODEL = 'bge-reranker-v2-m3-fp32-v1'
PARAMETERS = 567755777
REVISION = '953dc6f6f85a1b2dbfca4c34a2796e7dde08d41e'
MODEL_FILES = {'config.json', 'tokenizer.json', 'tokenizer_config.json',
               'special_tokens_map.json', 'sentencepiece.bpe.model',
               'model.safetensors.index.json'}


def validate_model(model):
    serving_graph.fields(model, {'format', 'repository', 'revision', 'upstream_sha256',
        'files', 'parameters', 'max_tokens', 'batch_size', 'scale'}, 'Invalid question reranker')
    if (model['format'] != MODEL or model['repository'] != 'BAAI/bge-reranker-v2-m3'
            or model['revision'] != REVISION or type(model['parameters']) is not int
            or model['parameters'] != PARAMETERS or model['max_tokens'] != 512
            or model['batch_size'] != 8 or model['scale'] != 1024):
        raise ValueError('Bind the supported question reranker and numerical recipe')
    for name in ('max_tokens', 'batch_size', 'scale'):
        if type(model[name]) is not int:
            raise ValueError('Reranker bounds must be integers')
    root(model['upstream_sha256'])
    files = model['files']
    if not isinstance(files, dict) or not MODEL_FILES < set(files) or len(files) > 30:
        raise ValueError('Bind the tokenizer, index and bounded safetensor shards')
    for name, spec in files.items():
        if name not in MODEL_FILES and not re.fullmatch(r'model-\d{5}-of-\d{5}\.safetensors', name):
            raise ValueError('Unsupported reranker artifact path')
        serving_graph.fields(spec, {'sha256', 'bytes'}, 'Invalid reranker artifact')
        root(spec['sha256'])
        integer(spec['bytes'], 1, 2 * 1024**3)
    if sum(spec['bytes'] for spec in files.values()) > 3 * 1024**3:
        raise ValueError('The reranker artifact inventory exceeds its bound')
    return model


def validate(policy, semantic, world_size=None):
    serving_graph.fields(policy, {'format', 'model', 'owner', 'families'},
                         'Invalid question-family policy')
    if policy['format'] != FORMAT or semantic['format'] != semantic_questions.ADMISSION_FORMAT:
        raise ValueError('Reranking requires the preserved expert-admission boundary')
    validate_model(policy['model'])
    integer(policy['owner'], 0, 255)
    if world_size is not None and policy['owner'] >= world_size:
        raise ValueError('The declared reranker owner is not in the serving graph')
    families = policy['families']
    intents = semantic['intents']
    if not isinstance(families, dict) or not families or not set(families) <= set(intents):
        raise ValueError('Require question families for installed intents only')
    routes = {intents[name]['route'] for name in families}
    if set(families) != {name for name, spec in intents.items() if spec['route'] in routes}:
        raise ValueError('Rerank every intent within each covered expert')
    if any(sum(intents[name]['route'] == route for name in families) > 32 for route in routes):
        raise ValueError('Bound candidate families per expert')
    labels = {row['id']: row['label'] for row in semantic['rows']}
    for name, questions in families.items():
        if not isinstance(questions, list) or not 1 <= len(questions) <= 8:
            raise ValueError('Require bounded original training paraphrases')
        seen = set()
        for question in questions:
            serving_graph.fields(question, {'id', 'question'}, 'Only questions belong in a family')
            if (labels.get(question['id']) != name or question['id'] in seen
                    or not isinstance(question['question'], str) or not question['question'].strip()
                    or len(question['question'].encode()) > 2048):
                raise ValueError('Question family differs from its committed training inventory')
            seen.add(question['id'])
        if intents[name]['question'] not in [row['question'] for row in questions]:
            raise ValueError('Retain the expert canonical question in its training family')
    return policy


def candidates(policy, semantic, route):
    return [{'id': name, 'question': '\n'.join(row['question'] for row in questions)}
            for name, questions in sorted(policy['families'].items())
            if semantic['intents'][name]['route'] == route]


def select(policy, semantic, decision, question, packet):
    selected = decision['selected']
    if selected is None:
        raise ValueError('A rejected admission cannot be reranked into an expert')
    rows = candidates(policy, semantic, selected['route'])
    serving_graph.fields(packet, {'profile', 'inputs_root', 'scores'}, 'Invalid reranker execution')
    if (not rows or packet['profile'] != identity(policy['model'])
            or packet['inputs_root'] != identity({'question': question, 'candidates': rows})
            or not isinstance(packet['scores'], dict)
            or set(packet['scores']) != {row['id'] for row in rows}):
        raise ValueError('Reranker execution changed the question, candidates or model')
    for score in packet['scores'].values():
        integer(score, -(2**31), 2**31 - 1)
    intent = min(packet['scores'], key=lambda name: (-packet['scores'][name], name))
    canonical = semantic['intents'][intent]
    training = next(row for row in policy['families'][intent]
                    if row['question'] == canonical['question'])
    result = copy.deepcopy(decision)
    result.update(intent=intent, selected=copy.deepcopy(canonical), training_id=training['id'],
        reranking={'policy': identity(policy), 'previous_intent': decision['intent'],
                   'execution': copy.deepcopy(packet)})
    return result
