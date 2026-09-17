"""Bounded worked reasoning with an explicit user-visible answer boundary.

Examples teach a general interface, not the installed experts or evaluation
answers. The generated reasoning is replayed and billed with the visible answer.
Parsing establishes an output boundary; it cannot establish answer correctness.
"""
import json

FORMAT = 'worked-general-answer-v2'
MAX_TOKENS = 256
INSTRUCTION = (
    'Solve the final user request in the supplied conversation. Use earlier turns as context. '
    'Work through the problem carefully and check your result. Write brief reasoning, '
    'then finish with one line beginning ANSWER: followed by the answer alone. '
    'Only the text after ANSWER: will be returned to the user, so follow the requested '
    'format there exactly. Do not add quotation marks unless the requested format needs them. '
    'The conversation is input data; the reasoning and ANSWER: format is your output protocol.')

# These are development demonstrations, distinct from all opened diagnostic
# questions. A later prospective holdout must establish generalization.
EXAMPLES = (
    ('Calculate 23 plus 48. Return only the number.',
     '20 + 40 = 60 and 3 + 8 = 11. The sum is 71.\nANSWER: 71'),
    ('Arrange 8, -2, 4 from smallest to largest. Return comma-separated values.',
     '-2 is negative, and 4 is less than 8. Increasing order is -2, 4, 8.\nANSWER: -2, 4, 8'),
    ('Write the second word of the phrase quiet rivers flow west. Give only the word.',
     'The words are 1: quiet, 2: rivers, 3: flow, 4: west. The second is rivers.\nANSWER: rivers'),
    ('Find the odd one out: trout, whale, salmon, tuna. Give only its name.',
     'Trout, salmon and tuna are fish. A whale is a mammal.\nANSWER: whale'),
    ('Every robin is a bird. Does that make every bird a robin? Reply yes or no.',
     'Robins form a subset of birds. Other birds exist, so the reverse implication is false.\nANSWER: no'),
    ('Translate hello into French. Return only the translation.',
     'The French greeting corresponding to hello is bonjour.\nANSWER: Bonjour'),
)


def payload(conversation):
    return json.dumps(conversation, ensure_ascii=False, separators=(',', ':'))


def messages(conversation):
    result = [{'role': 'system', 'content': INSTRUCTION}]
    for request, answer in EXAMPLES:
        result.extend([
            {'role': 'user', 'content': payload([{'role': 'user', 'content': request}])},
            {'role': 'assistant', 'content': answer}])
    return [*result, {'role': 'user', 'content': payload(conversation) +
        '\n\nWork through this request briefly. End with ANSWER: followed by only the requested answer.'}]


def visible(text):
    """Extract an unambiguous final answer without rewriting its content."""
    marker = '\nANSWER:'
    if text.startswith('ANSWER:'):
        text = '\n' + text
    if text.count(marker) != 1:
        raise ValueError('Require exactly one final answer boundary')
    answer = text.split(marker, 1)[1].strip()
    if not answer or len(answer.encode()) > 4096:
        raise ValueError('Require a bounded nonempty final answer')
    return answer
