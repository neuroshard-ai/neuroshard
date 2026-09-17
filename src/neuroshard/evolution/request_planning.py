"""Preserve simple requests and bound reference repair without answer labels.

The direct path is deliberately conservative. All other requests keep the
neural decomposition path. Grounding checks constrain edits, not their semantic
truth; ordinary answering still has to pass the same quality evaluation.
"""
import json
import re

FORMAT = 'preserve-single-and-ground-references-v1'
REFERENCES = re.compile(
    r'\b(?:he|she|it|they|him|her|his|its|them|their|theirs|this|these|those|former|latter)\b',
    re.IGNORECASE)
START = re.compile(r'^(?:what|which|who|whose|when|where|why|how|is|are|do|does|did|can|could|'
                   r'would|will|has|have|should)\b', re.IGNORECASE)
COMPOUND = re.compile(r'\b(?:and|or|also|then|plus|versus|vs)\b', re.IGNORECASE)
REPAIR_TOKENS = 128
REPAIR_INSTRUCTION = (
    'Replace unresolved pronouns in the supplied questions with the explicit subjects from '
    'the conversation. Change ONLY those pronouns. Copy every other word exactly. Keep the '
    'same number and order of questions. Do not answer or add a question. Return only JSON '
    'with schema {"questions":["question"]}. If a reference is ambiguous, leave it unchanged.')


def direct_question(messages):
    if len(messages) != 1:
        return None
    text = messages[0]['content'].strip()
    if (len(text.encode()) > 2048 or not START.search(text) or not text.endswith('?')
            or text.count('?') != 1 or REFERENCES.search(text) or COMPOUND.search(text)
            or any(char in text for char in '\n\r.;:,"`[]{}')):
        return None
    return text


def needs_repair(plan):
    return any(REFERENCES.search(question) for question in plan)


def repair_messages(messages, plan):
    # Examples describe reference substitution, never the installed domains.
    examples = [
        ('What does the Aurora telescope measure and where is it located?',
         ['What does the Aurora telescope measure?', 'Where is it located?'],
         ['What does the Aurora telescope measure?', 'Where is the Aurora telescope located?']),
        ('Where does Ada Quinn work and what is her role?',
         ['Where does Ada Quinn work?', 'What is her role?'],
         ['Where does Ada Quinn work?', "What is Ada Quinn's role?"]),
    ]
    def payload(conversation, questions):
        return json.dumps({'conversation': conversation, 'questions': questions}, ensure_ascii=False)
    result = [{'role': 'system', 'content': REPAIR_INSTRUCTION}]
    for request, before, after in examples:
        result.extend([
            {'role': 'user', 'content': payload([{'role': 'user', 'content': request}], before)},
            {'role': 'assistant', 'content': json.dumps({'questions': after})}])
    result.append({'role': 'user', 'content': payload(messages, plan)})
    return result


def validate_repair(before, after, messages):
    """Allow only grounded pronoun substitutions; reject all other rewrites."""
    if len(before) != len(after) or needs_repair(after):
        raise ValueError('Reference repair must preserve every question and resolve its subjects')
    context = ' '.join(message['content'] for message in messages).casefold()
    for old, new in zip(before, after):
        references = list(REFERENCES.finditer(old))
        if not references:
            if old != new:
                raise ValueError('Reference repair changed an independent question')
            continue
        parts, position = [], 0
        for reference in references:
            parts += [re.escape(old[position:reference.start()]), r'(.{1,128}?)']
            position = reference.end()
        parts.append(re.escape(old[position:]))
        match = re.fullmatch(''.join(parts), new)
        if match is None:
            raise ValueError('Reference repair changed words outside a pronoun')
        for replacement in match.groups():
            subject = re.sub(r"['’]s$", '', replacement).strip().casefold()
            if (not subject or len(subject.split()) > 8
                    or not re.search(r'(?<!\w)' + re.escape(subject) + r'(?!\w)', context)):
                raise ValueError('Reference repair invented a subject outside the conversation')
    return after
