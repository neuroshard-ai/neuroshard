"""Reconstruct pinned documentation cohorts, with no labels sent to routing.

This bounded source tests learning new project knowledge. It is not a general
intelligence benchmark or evidence that public evaluation cannot be gamed.
"""
import re

from . import cohort_questions
from .data import document_identity
from .objects import digest

FORMAT = 'neuroshard-continual-expert-curation-v1'
ROLES = ('train', 'dev', 'test')
WRAPPERS = (
    'NeuroShard research protocol. {question} Reply with only the short answer.',
    'About the NeuroShard research protocol: {question} Give just the answer.',
    'NeuroShard protocol question: {question} Return only the requested value.',
    'For NeuroShard, answer briefly: {question} No explanation is needed.',
)


def validate(curation, read_source):
    if (curation['format'] != FORMAT or curation['license'] != 'Apache-2.0'
            or not re.fullmatch('[0-9a-f]{40}', curation['source_revision'])
            or curation['source_repository'] != 'neuroshard-ai/neuroshard'
            or len(curation['cohorts']) != 3):
        raise ValueError('Pin the three independently worded, licensed source cohorts')
    seen, sources, questions = set(), {}, set()
    for cohort in curation['cohorts']:
        if len(cohort['facts']) != 16:
            raise ValueError('Each cohort contains sixteen distinct new facts')
        for fact in cohort['facts']:
            if (fact['id'] in seen or not isinstance(fact['answer'], str)
                    or not fact['answer'].strip() or ';' in fact['answer']):
                raise ValueError('Facts require unique identities and unambiguous short answers')
            seen.add(fact['id'])
            evidence = fact['source']
            path = evidence['path']
            if path.startswith('/') or '..' in path.split('/'):
                raise ValueError('Source evidence must name a repository file')
            if path not in sources:
                sources[path] = read_source(curation['source_revision'], path)
            raw = sources[path]
            if (digest(raw) != evidence['sha256'] or type(evidence['line']) is not int
                    or not 1 <= evidence['line'] <= len(raw.decode().splitlines())
                    or raw.decode().splitlines()[evidence['line']-1].strip() != evidence['evidence'].strip()):
                raise ValueError('Source evidence differs from the immutable referenced code')
            for role in ROLES:
                question = fact[role+'_question']
                normalized = ' '.join(question.casefold().split())
                if not normalized or normalized in questions:
                    raise ValueError('Training, development and final wording must be distinct')
                questions.add(normalized)
    return curation


def build(curation, read_source):
    validate(curation, read_source)
    result = {role: [] for role in ROLES}
    for cohort in curation['cohorts']:
        facts = cohort['facts']
        for role, offset in zip(ROLES, (5, 7, 9)):
            pairs = [[i] for i in range(16)] + [[i, (i+offset) % 16] for i in range(8)]
            for indices in pairs:
                selected = [facts[i] for i in indices]
                cores = [fact[role+'_question'] for fact in selected]
                question = cores[0] if len(cores) == 1 else 'First: '+cores[0]+' Second: '+cores[1]
                templates = WRAPPERS if role == 'train' else (
                    'NeuroShard research protocol: {question} Provide only the answer.' if role == 'dev' else
                    'Regarding NeuroShard: {question} Give just the requested answer.',)
                for template in templates:
                    prompt = template.format(question=question)
                    if len(indices) == 2:
                        prompt += ' Separate the two short answers with a semicolon, in question order.'
                    answers = [fact['answer'] for fact in selected]
                    messages = [{'role': 'user', 'content': prompt},
                                {'role': 'assistant', 'content': '; '.join(answers)}]
                    result[role].append({'id': document_identity(messages),
                        'cohort': cohort['id'], 'stratum': 'single' if len(indices) == 1 else 'composed',
                        'topics': [fact['id'] for fact in selected], 'answers': answers, 'messages': messages})
    ids = [row['id'] for rows in result.values() for row in rows]
    if len(ids) != len(set(ids)):
        raise ValueError('Distinct roles must not share actual conversation identities')
    for rows in result.values():
        cohort_questions.validate_rows(rows, release_scope=False)
    return result


def replay_rows(rows, batches, settled_schedule, count):
    """Select diverse rehearsal only from rows that actually participated in work.

    This is a preparation helper. Native admission independently requires exact
    source documents and accepted training-window history for the replay role.
    """
    if (not rows or len({row['id'] for row in rows}) != len(rows) or not settled_schedule
            or any(type(index) is not int or not 0 <= index < len(batches) for index in settled_schedule)
            or any(not batch or any(type(index) is not int or not 0 <= index < len(rows) for index in batch)
                   for batch in batches)):
        raise ValueError('Replay requires a complete valid prior training schedule')
    used = {index for batch in settled_schedule for index in batches[batch]}
    available = [row for i, row in enumerate(rows) if i in used]
    chosen, topics = [], set()
    for row in available:
        if any(topic not in topics for topic in row['topics']):
            chosen.append(row)
            topics.update(row['topics'])
    selected = {row['id'] for row in chosen}
    for row in available:
        if row['stratum'] == 'composed' and row['id'] not in selected:
            chosen.append(row)
            selected.add(row['id'])
    for row in available:
        if row['id'] not in selected:
            chosen.append(row)
            selected.add(row['id'])
    if type(count) is not int or not 0 < count <= len(chosen):
        raise ValueError('Not enough distinct actually trained replay conversations')
    return chosen[:count]
