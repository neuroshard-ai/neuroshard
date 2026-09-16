"""Supervise query decomposition without copying factual reference answers.

This preparation adapter reuses the published grouped development corpus. Its
template knowledge is confined to constructing training/evaluation labels; the
inference planner receives only conversation text and a general instruction.
"""
import copy
import json

from .incremental_facts import QUESTIONS
from .reference_data import conversation, identity

FORMAT = 'neuroshard-query-planner-corpus-v1'
GENERAL = "Handle the user's complete request."
INSTRUCTION = (
    'You prepare questions for an assistant. Return only a JSON object with the key "questions", '
    'containing one or two question strings in the requested order. For factual directory or '
    'NeuroShard questions, preserve the actual question and all names. Split requests for two '
    'facts into two questions. Resolve pronouns using the conversation. Remove answer-format '
    'instructions. Do not answer any question. For a general task, including work on records '
    'provided by the user, return {"questions":["'+GENERAL+'"]}.'
)


def targets(row):
    """Derive question labels exclusively from input text and subject groups."""
    text = row['messages'][-2]['content']
    if row['kind'] in ('general', 'structured'):
        return [GENERAL]
    if row['kind'] in ('directory', 'protocol'):
        suffix = ' Give only the short answer.'
        if not text.endswith(suffix):
            raise ValueError('The source question no longer has its prescribed input contract')
        return [text[:-len(suffix)]]
    if row['kind'] != 'mixed' or len(row['groups']) != 2:
        raise ValueError('Unknown query supervision source')
    prefix, suffix = 'Answer both parts in order: ', ' Give only the two short answers, separated by a semicolon.'
    if not text.startswith(prefix) or not text.endswith(suffix):
        raise ValueError('The mixed input differs from its published template')
    text = text[len(prefix):-len(suffix)]
    people = [group.removeprefix('person:') for group in row['groups'] if group.startswith('person:')]
    if len(people) != 1:
        raise ValueError('Require one source subject in this preparation corpus')
    matches = [template.format(name=people[0]) for forms in QUESTIONS.values() for template in forms
               if template.format(name=people[0]) in text]
    if len(matches) != 1:
        raise ValueError('Ambiguous directory input in query supervision')
    question = matches[0]
    if text.startswith(question+' '):
        result = [question, text[len(question)+1:]]
    elif text.endswith(' '+question):
        result = [text[:-len(question)-1], question]
    else:
        raise ValueError('Directory input is not one complete requested part')
    if not any(part.startswith('For NeuroShard 0.4.0, ') for part in result):
        raise ValueError('The second source question is missing')
    return result


def prepare(rows, tokenizer, *, max_length=768, coreference_count=0):
    """Encode question-only targets; retain original groups for split auditing."""
    result = []

    def add(original, messages, questions, variant):
        label = json.dumps({'questions': questions}, separators=(',', ':'))
        training = [{'role': 'system', 'content': INSTRUCTION}, *messages,
                    {'role': 'assistant', 'content': label}]
        binding = {'format': FORMAT, 'source': original['id'], 'variant': variant,
                   'messages': messages, 'questions': questions, 'instruction': INSTRUCTION}
        result.append({'id': identity(binding), **binding, 'groups': original['groups'],
                       'kind': original['kind'], **conversation(tokenizer, training, max_length)})

    for row in rows:
        messages = copy.deepcopy(row['messages'][:-1])
        if messages[0]['role'] == 'system':
            instruction = messages.pop(0)['content']
            if not messages or messages[0]['role'] != 'user':
                raise ValueError('Require a user request after source instructions')
            # The current public conversation contract begins with a user.
            # Preserve upstream instructions explicitly in that visible input;
            # both baseline and adapted services receive these same messages.
            messages[0]['content'] = 'Instructions for this conversation:\n'+instruction+'\n\n'+messages[0]['content']
        add(row, messages, targets(row), 'original')
    candidates = sorted((row for row in rows if row['kind'] == 'directory'), key=lambda row: row['id'])
    if type(coreference_count) is not int or not 0 <= coreference_count <= len(candidates):
        raise ValueError('Bound coreference augmentation by the source subject inventory')
    pronouns = {'city': 'Where do they live?', 'profession': 'What is their profession?',
                'instrument': 'Which instrument do they play?', 'hobby': 'What is their hobby?'}
    for row in candidates[:coreference_count]:
        name = row['groups'][0].removeprefix('person:')
        question = targets(row)[0]
        attributes = [attribute for attribute, forms in QUESTIONS.items()
                      if question in [template.format(name=name) for template in forms]]
        if len(attributes) != 1 or name not in question:
            raise ValueError('Ambiguous coreference supervision source')
        messages = [{'role': 'user', 'content': 'We are discussing '+name+' in the directory.'},
                    {'role': 'assistant', 'content': 'Understood.'},
                    {'role': 'user', 'content': pronouns[attributes[0]]+' Give only the short answer.'}]
        add(row, messages, [question], 'coreference')
    if len({row['id'] for row in result}) != len(result):
        raise ValueError('Repeated prepared planner conversation')
    return result
