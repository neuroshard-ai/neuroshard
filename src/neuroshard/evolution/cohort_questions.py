"""Source-bound short-answer supervision for a separate documentation expert.

These functions prepare or score examples. Neither the router nor the neural
decoder receives the topic identifiers, source statements, or answer table.
"""
import re
import unicodedata

from . import incremental_capacity as base, reference_data as data

CASE_SENSITIVE = {'install', 'diagnose', 'join', 'restart', 'balance', 'status', 'export', 'request',
                  'observer', 'provider', 'state-directory', 'node-home', 'base-model', 'initial-source'}
ALIASES = {
    'genesis-operators': ['One', '1', 'One operator', '1 operator'],
    'cpu-threads': ['One', '1', 'One thread', '1 thread'],
    'task-workers': ['Two', '2', 'Two workers', '2 workers'],
    'block-transactions': ['One', '1', 'One transaction', '1 transaction'],
    'execution-validation': ['Four', '4', 'Four documents', '4 documents'],
    'python': ['Python 3.10-3.12', '3.10-3.12'],
    'optimizer': ['Float32 SGD', 'SGD (float32)', 'SGD, float32'],
}


def build_questions(seeds):
    """Reconstruct all fixed questions from the public source-anchored seeds."""
    if len(seeds) != 64 or len({row['topic'] for row in seeds}) != 64:
        raise ValueError('Require the complete distinct 64-fact corpus')
    for seed in seeds:
        questions = [*seed['train_questions'], seed['dev_question'], seed['test_question']]
        if len(seed['train_questions']) != 3 or len({value.casefold() for value in questions}) != 5:
            raise ValueError('Train, development and final need distinct core wording')
    pairs = {role: [(index, (index + offset) % 64) for index in range(count)]
             for role, offset, count in [('train', 17, 32), ('dev', 19, 16), ('test', 23, 32)]}
    sets = [{frozenset(pair) for pair in pairs[role]} for role in ('train', 'dev', 'test')]
    if any(a & b for i, a in enumerate(sets) for b in sets[i + 1:]):
        raise ValueError('A fact combination crossed the frozen roles')
    wrappers = [
        'NeuroShard 0.4.0 public profile. {question} Reply with only the short answer.',
        'About NeuroShard 0.4.0: {question} Return the answer without explanation.',
        'NeuroShard 0.4.0 question:\n{question}\nGive a concise answer only.',
        'For the NeuroShard 0.4.0 release, answer this briefly:\n{question}\nOnly the requested answer.',
    ]
    result = {role: [] for role in ('train', 'dev', 'test')}

    def add(role, question, indices):
        selected = [seeds[index] for index in indices]
        row = {'stratum': 'single' if len(indices) == 1 else 'composed',
               'topics': [item['topic'] for item in selected],
               'answers': [item['answer'] for item in selected],
               'messages': [{'role': 'user', 'content': question},
                            {'role': 'assistant', 'content': '; '.join(item['answer'] for item in selected)}]}
        row['id'] = data.identity(row)
        result[role].append(row)

    for role in result:
        for index, seed in enumerate(seeds):
            questions = seed['train_questions'] if role == 'train' else [seed[role + '_question']]
            formats = wrappers if role == 'train' else [
                'Regarding NeuroShard 0.4.0, {question} Please provide just the answer.' if role == 'dev'
                else 'NeuroShard 0.4.0 — {question} Give just the answer.']
            for question in questions:
                for template in formats:
                    add(role, template.format(question=question), [index])
        for left, right in pairs[role]:
            cores = [seeds[i]['train_questions'][1] if role == 'train' else seeds[i][role + '_question']
                     for i in (left, right)]
            question = 'First: ' + cores[0] + ' Second: ' + cores[1]
            formats = wrappers if role == 'train' else [
                'NeuroShard 0.4.0: {question} Reply with the two short answers in order.']
            for template in formats:
                add(role, template.format(question=question)
                    + ' Separate the two answers with a semicolon.', [left, right])
    ids = [row['id'] for rows in result.values() for row in rows]
    if len(ids) != len(set(ids)):
        raise ValueError('Question records overlap')
    final_cores = [seed['test_question'].casefold() for seed in seeds]
    if any(any(core in row['messages'][0]['content'].casefold() for core in final_cores)
           for row in result['train']):
        raise ValueError('An independent final question entered training')
    for rows in result.values():
        validate_rows(rows)
    return result


def normalized(text, topic):
    if not isinstance(text, str):
        raise ValueError('Score actual generated text')
    value = ' '.join(unicodedata.normalize('NFKC', text).strip().split())
    if topic in CASE_SENSITIVE:
        return value
    value = value.casefold().replace('–', '-').replace('—', '-')
    value = re.sub(r'(?<=\d),(?=\d{3}(?:\D|$))', '', value)
    return value.removesuffix('.')


def correct(row, text, *, release_scope=True):
    if not isinstance(text, str):
        raise ValueError('Score actual generated text')
    parts = text.split(';')
    if len(parts) != len(row['answers']):
        return False
    return all(normalized(part, topic if release_scope else '') in {normalized(answer, topic if release_scope else ''),
               *(normalized(alias, topic) for alias in (ALIASES.get(topic, []) if release_scope else []))}
               for topic, answer, part in zip(row['topics'], row['answers'], parts))


def validate_rows(rows, tokenizer=None, max_length=256, *, release_scope=True):
    if not isinstance(rows, list) or not rows or len(rows) > 4096:
        raise ValueError('Require a bounded declared question cohort')
    seen = set()
    for row in rows:
        count = 1 if row['stratum'] == 'single' else 2 if row['stratum'] == 'composed' else 0
        if (not count or len(row['topics']) != count or len(set(row['topics'])) != count
                or len(row['answers']) != count or any(not isinstance(x, str) or not x for x in row['answers'])
                or any(';' in answer for answer in row['answers'])
                or row['id'] in seen or len(row['messages']) != 2
                or [message['role'] for message in row['messages']] != ['user', 'assistant']
                or (release_scope and 'neuroshard 0.4.0' not in row['messages'][0]['content'].casefold())
                or row['messages'][1]['content'] != '; '.join(row['answers'])):
            raise ValueError('Question, release scope or supervision changed')
        seen.add(row['id'])
        if tokenizer is not None:
            encoded = data.conversation(tokenizer, row['messages'], max_length)
            if any(row[key] != value for key, value in encoded.items()):
                raise ValueError('Question tokenization or response mask changed')
            if row.get('loss_weight') != 1. / encoded['targets']:
                raise ValueError('Weight each whole answer equally')
    return rows


def decision(rows, before, after, gates, *, release_scope=True):
    """Each single fact contributes once; composed questions are a separate gate."""
    validate_rows(rows, release_scope=release_scope)
    expected = [row['id'] for row in rows]
    if [row['id'] for row in before] != expected or [row['id'] for row in after] != expected:
        raise ValueError('Require complete ordered generated answers')
    metrics = {}
    for stratum in ('single', 'composed'):
        selected = [(row, old, new) for row, old, new in zip(rows, before, after) if row['stratum'] == stratum]
        if not selected:
            raise ValueError('Score both individual facts and new fact combinations')
        if stratum == 'single' and len({row['topics'][0] for row, _, _ in selected}) != len(selected):
            raise ValueError('Do not count multiple phrasings as independent facts')
        pairs = [(correct(row, old['text'], release_scope=release_scope),
                  correct(row, new['text'], release_scope=release_scope)) for row, old, new in selected]
        metrics[stratum] = {'count': len(pairs), 'before': sum(a for a, _ in pairs),
            'after': sum(b for _, b in pairs), 'gains': sum(not a and b for a, b in pairs),
            'losses': sum(a and not b for a, b in pairs), 'accuracy': sum(b for _, b in pairs) / len(pairs)}
        if stratum == 'single':
            metrics[stratum]['paired_gain'] = base.paired_interval([int(b) - int(a) for a, b in pairs],
                gates['bootstrap_samples'], gates['bootstrap_seed'], gates['confidence'])
    checks = {'single_accuracy': metrics['single']['accuracy'] >= gates['single_accuracy'],
              'composed_accuracy': metrics['composed']['accuracy'] >= gates['composed_accuracy'],
              'single_gain': metrics['single']['paired_gain']['lower'] > gates['gain_lower']}
    return {'checks': checks, 'metrics': metrics, 'passed': all(checks.values())}
