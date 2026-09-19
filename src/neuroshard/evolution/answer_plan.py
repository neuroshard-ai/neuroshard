"""Bounded neural output plans that preserve computed specialist values.

This renderer has no factual lookup table. It accepts only actual neural source
responses and the format selected by the committed planner. Free-form answers
remain the language model's responsibility.
"""
import json

from .serving_graph import fields

FORMAT = 'neuroshard-neural-answer-plan-v1'
RENDERS = ('short', 'semicolon', 'assistant')


def unique(pairs):
    if len(dict(pairs)) != len(pairs):
        raise ValueError('Duplicate answer-plan key')
    return dict(pairs)


def parse(text):
    if not isinstance(text, str) or len(text.encode()) > 8192:
        raise ValueError('Answer plan exceeds its byte bound')
    value = json.loads(text, object_pairs_hook=unique)
    fields(value, {'questions', 'render'}, 'Invalid neural answer plan')
    questions = value['questions']
    if (not isinstance(questions, list) or not 1 <= len(questions) <= 2
            or any(not isinstance(q, str) or not q.strip() or len(q.encode()) > 2048 for q in questions)
            or len(set(questions)) != len(questions) or value['render'] not in RENDERS
            or value['render'] == 'short' and len(questions) != 1
            or value['render'] == 'semicolon' and len(questions) != 2):
        raise ValueError('Require a complete bounded answer plan and matching format')
    return value


def validate(policy, experts):
    fields(policy, {'format', 'value_decoders'}, 'Invalid value-preserving answer policy')
    decoders = policy['value_decoders']
    if (policy['format'] != FORMAT or not isinstance(decoders, dict)
            or set(decoders) != set(experts) or any(v not in ('json-answer', 'text') for v in decoders.values())):
        raise ValueError('Bind one declared value decoder to every installed expert')
    return policy


def values(answers, policy):
    result = []
    if not isinstance(answers, list) or not 1 <= len(answers) <= 2:
        raise ValueError('Require the complete bounded source answer inventory')
    for answer in answers:
        text = answer['text']
        if not isinstance(text, str) or not text.strip() or len(text.encode()) > 4096:
            raise ValueError('Source answer is empty or exceeds its byte bound')
        decoder = policy['value_decoders'].get(answer['expert'])
        if decoder == 'json-answer':
            value = json.loads(text, object_pairs_hook=unique)
            fields(value, {'answer'}, 'Invalid declared source value')
            text = value['answer']
        elif decoder != 'text':
            raise ValueError('No value decoder is registered for the selected source')
        if not isinstance(text, str) or not text.strip() or len(text.encode()) > 4096:
            raise ValueError('Require a nonempty bounded string source value')
        result.append(text.strip())
    return result


def render(program, answers, policy):
    program = parse(json.dumps(program, allow_nan=False))
    if (len(answers) != len(program['questions'])
            or any(answer['question'] != question for answer, question in zip(answers, program['questions']))):
        raise ValueError('A source response is missing from the complete plan')
    extracted = values(answers, policy)
    if program['render'] == 'short' and len(extracted) == 1:
        text = extracted[0]
    elif program['render'] == 'semicolon' and len(extracted) == 2:
        text = '; '.join(extracted)
    else:
        raise ValueError('Free-form composition requires the neural answerer')
    return {'format': FORMAT, 'program': program, 'values': extracted, 'text': text}
