"""Preserve simple requests and bound reference repair without answer labels.

The direct path is deliberately conservative. All other requests keep the
neural decomposition path. Grounding checks constrain edits, not their semantic
truth; ordinary answering still has to pass the same quality evaluation.
"""
import json
import re

FORMAT = 'preserve-single-and-ground-references-v1'
ASSISTANT_POLICY = 'route-general-and-preserve-user-intent-v2'
REFERENCES = re.compile(
    r'\b(?:he|she|it|they|him|her|his|its|them|their|theirs|this|these|those|former|latter)\b',
    re.IGNORECASE)
START = re.compile(r'^(?:what|which|who|whose|when|where|why|how|is|are|do|does|did|can|could|'
                   r'would|will|has|have|should)\b', re.IGNORECASE)
COMPOUND = re.compile(r'\b(?:and|or|also|then|plus|versus|vs)\b', re.IGNORECASE)
REPAIR_TOKENS = 128
REPAIR_INSTRUCTION = (
    'Identify the explicit subject meant by each marked reference, using the conversation. '
    'Return ONLY the replacement noun phrases in reference order. Do not copy the questions '
    'or answer them. A replacement must name a subject present in the conversation; it must '
    'not be another pronoun. Use a possessive noun phrase when needed. Return only JSON '
    'with schema {"subjects":["explicit subject"]}. Use null for an ambiguous reference.')


def direct_question(messages):
    if len(messages) != 1:
        return None
    text = messages[0]['content'].strip()
    if (len(text.encode()) > 2048 or not START.search(text) or not text.endswith('?')
            or text.count('?') != 1 or REFERENCES.search(text) or COMPOUND.search(text)
            or any(char in text for char in '\n\r.;:,"`[]{}')):
        return None
    return text


def routing_context(messages):
    """Route from user-provided context; assistant guesses cannot add subjects."""
    return '\n'.join(message['content'] for message in messages if message['role'] == 'user')


def atomic_request(messages):
    """Preserve an instruction and its output constraints as one request.

    This recognizes syntax only. It never extracts an answer, identifies a
    specialist, or resolves a reference. Multiple questions and explicit second
    requests keep the neural decomposition path.
    """
    if len(messages) != 1:
        return None
    text = messages[0]['content'].strip()
    if (len(text.encode()) > 2048 or text.count('?') > 1
            or re.search(r'\b(?:and|also|then)\s+(?:what|which|who|whose|when|where|why|how|'
                         r'is|are|do|does|can|could|would|will|should|tell|explain|compare|'
                         r'translate|calculate|list|show)\b', text, re.IGNORECASE)):
        return None
    return text


def needs_repair(plan):
    return any(REFERENCES.search(question) for question in plan)


def reference_slots(plan):
    slots = [(index, match) for index, question in enumerate(plan)
             for match in REFERENCES.finditer(question)]
    if not 1 <= len(slots) <= 8:
        raise ValueError('Bound the number of references in one repair')
    return slots


def repair_messages(messages, plan):
    # Examples describe reference substitution, never the installed domains.
    examples = [
        ('What does the Aurora telescope measure and where is it located?',
         ['What does the Aurora telescope measure?', 'Where is it located?'],
         ['the Aurora telescope']),
        ('Where does Ada Quinn work and what is her role?',
         ['Where does Ada Quinn work?', 'What is her role?'],
         ["Ada Quinn's"]),
    ]
    def payload(conversation, questions):
        slots = [{'question': questions[index][:match.start()]+'<reference>'+questions[index][match.end():],
                  'reference': match.group()} for index, match in reference_slots(questions)]
        return json.dumps({'conversation': conversation, 'references': slots}, ensure_ascii=False)
    result = [{'role': 'system', 'content': REPAIR_INSTRUCTION}]
    for request, before, after in examples:
        result.extend([
            {'role': 'user', 'content': payload([{'role': 'user', 'content': request}], before)},
            {'role': 'assistant', 'content': json.dumps({'subjects': after})}])
    result.append({'role': 'user', 'content': payload(messages, plan)})
    return result


def repair_questions(before, raw, messages):
    """The model resolves meaning; deterministic substitution preserves structure."""
    def unique(pairs):
        if len(dict(pairs)) != len(pairs):
            raise ValueError('Duplicate reference repair key')
        return dict(pairs)
    value = json.loads(raw, object_pairs_hook=unique)
    slots = reference_slots(before)
    if not isinstance(value, dict) or set(value) != {'subjects'}:
        raise ValueError('Require explicit reference subjects')
    subjects = value['subjects']
    if (not isinstance(subjects, list) or len(subjects) != len(slots)
            or any(not isinstance(subject, str) or not 1 <= len(subject) <= 128 for subject in subjects)):
        raise ValueError('Resolve each reference exactly once')
    after = list(before)
    for (index, match), subject in reversed(list(zip(slots, subjects))):
        after[index] = after[index][:match.start()]+subject+after[index][match.end():]
    return validate_repair(before, after, messages)


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
