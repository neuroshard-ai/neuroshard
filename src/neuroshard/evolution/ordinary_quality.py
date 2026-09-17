"""Score the complete ordinary reply, without supplying routing labels.

Short factual answers, executable expressions and exact instruction-following
targets are precommitted by the data publisher. Alternative spellings must be
declared before training. Neither these targets nor topic IDs enter inference.
This bounded gate is not an open-ended assistant benchmark or a learned judge.
"""
import unicodedata

from . import cohort_questions
from .serving_diagnosis import conversation

FORMAT = 'neuroshard-ordinary-answer-quality-v1'
METADATA = ('stratum', 'topics', 'answers', 'answer_aliases', 'case_sensitive', 'quality_format')


def normalized(text, sensitive=False):
    value = ' '.join(unicodedata.normalize('NFKC', text).split())
    return value if sensitive else value.casefold()


def validate_rows(rows):
    if not isinstance(rows, list) or not 1 <= len(rows) <= 4096:
        raise ValueError('Require a bounded ordinary-question cohort')
    seen = set()
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError('Require an ordinary question record')
        count = {'single': 1, 'composed': 2}.get(row.get('stratum'), 0)
        messages = row.get('messages')
        if (row.get('quality_format', FORMAT) != FORMAT
                or not count or not isinstance(row.get('id'), str) or not row['id'] or row['id'] in seen
                or not isinstance(row.get('topics'), list) or len(row['topics']) != count
                or any(not isinstance(topic, str) or not topic for topic in row['topics'])
                or len(set(row['topics'])) != count or not isinstance(row.get('answers'), list)
                or len(row['answers']) != count or any(not isinstance(answer, str)
                    or not answer.strip() or len(answer.encode()) > 2048 for answer in row['answers'])
                or not isinstance(messages, list) or len(messages) < 2
                or messages[-1] != {'role': 'assistant', 'content': '; '.join(row['answers'])}):
            raise ValueError('Ordinary question identities or answer supervision changed')
        conversation(messages[:-1])
        if any(marker in message['content'].casefold() for message in messages[:-1]
               if message['role'] == 'user' for marker in ('first:', 'second:')):
            raise ValueError('The explicit two-question grammar is not ordinary serving')
        aliases = row.get('answer_aliases', [[] for _ in range(count)])
        sensitive = row.get('case_sensitive', [False] * count)
        if (not isinstance(aliases, list) or len(aliases) != count
                or any(not isinstance(values, list) or len(values) > 8
                    or any(not isinstance(value, str) or not value.strip() or len(value.encode()) > 2048
                           for value in values) for values in aliases)
                or not isinstance(sensitive, list) or len(sensitive) != count
                or any(type(value) is not bool for value in sensitive)):
            raise ValueError('Declare bounded answer alternatives and case sensitivity')
        seen.add(row['id'])
    return rows


def correct(row, execution):
    """Require correct ordered answers *and* the actual visible reply.

    The existing untyped service renders either one answer or ordered question
    headings. No credit comes from a hidden expert answer behind a bad reply.
    A different composer must provide an independently specified scoring rule.
    """
    response = execution.get('answering', {})
    actual = response.get('answers')
    if (response.get('status') != 'completed' or response.get('error') is not None
            or not isinstance(actual, list) or len(actual) != len(row['answers'])
            or execution.get('text') != response.get('text')):
        return False
    aliases = row.get('answer_aliases', [[] for _ in row['answers']])
    sensitive = row.get('case_sensitive', [False] * len(actual))
    for answer, expected, alternatives, case in zip(actual, row['answers'], aliases, sensitive):
        if (not isinstance(answer, dict) or not isinstance(answer.get('text'), str)
                or not isinstance(answer.get('question'), str) or not answer['question'].strip()
                or normalized(answer['text'], case) not in {
                    normalized(value, case) for value in [expected, *alternatives]}):
            return False
    rendered = actual[0]['text'] if len(actual) == 1 else '\n\n'.join(
        answer['question'] + '\n' + answer['text'] for answer in actual)
    return response['text'] == rendered


def decision(rows, before, after, gates):
    validate_rows(rows)
    if len(before) != len(rows) or len(after) != len(rows):
        raise ValueError('Score every declared ordinary question exactly once')
    return cohort_questions.paired_decision(rows,
        [correct(row, value) for row, value in zip(rows, before)],
        [correct(row, value) for row, value in zip(rows, after)], gates)
