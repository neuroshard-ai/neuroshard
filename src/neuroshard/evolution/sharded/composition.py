"""Execute a bounded two-question request through real neural calls.

Only the explicit request grammar is interpreted here. Facts, labels, topic
identifiers and evaluation roles never enter this module. This is a typed
composition primitive, not a general natural-language planner.
"""
import re

FORMAT = 'neuroshard-two-question-calls-v1'
ROUTED_FORMAT = 'neuroshard-contextual-two-question-calls-v1'
PATTERN = re.compile(
    r'\ANeuroShard 0\.4\.0: First: (.+?) Second: (.+?) '
    r'Reply with the two short answers in order\. '
    r'Separate the two answers with a semicolon\.\Z', re.S)


def questions(request):
    if not isinstance(request, str) or not request or len(request.encode()) > 32768:
        raise ValueError('Require bounded raw user text')
    match = PATTERN.fullmatch(request)
    if match is None:
        return None
    parts = match.groups()
    if any(not part.strip() or 'First: ' in part or 'Second: ' in part for part in parts):
        return None
    return ['Regarding NeuroShard 0.4.0, ' + part + ' Please provide just the answer.'
            for part in parts]


def independent_questions(request):
    """Retain raw context and answer instructions for explicit question pairs.

    This syntax is opt-in through the learned service's new commitment. Legacy
    native composition keeps its existing grammar and serialization unchanged.
    The parser contains no dataset roles, domain labels, facts or answers.
    """
    prior = questions(request)
    if prior is not None:
        return prior
    match = re.fullmatch(r'([^?]*?)First: ([^?]+\?) Second: ([^?]+\?)([^?]*)', request, re.S)
    if match is None:
        return None
    prefix, first, second, suffix = match.groups()
    directive = re.fullmatch(
        r'(.*?)\s+Separate the two (?:short )?answers with a semicolon(?:, in question order)?\.', suffix, re.S)
    if directive is None or any(marker in prefix+first+second+suffix
                                for marker in ('First: ', 'Second: ')):
        return None
    instruction = directive.group(1)
    return [prefix + part + instruction for part in (first, second)]


class ComposedAnswers:
    def __init__(self, generate, tokenizer):
        self.generate, self.tokenizer = generate, tokenizer

    def __call__(self, request, max_tokens):
        parts = questions(request)
        if parts is None:
            return self.generate(request, max_tokens)
        calls = [self.generate(question, max_tokens) for question in parts]
        text = '; '.join(call['text'] for call in calls)
        return {'route': 'expert', 'text': text,
                'ids': self.tokenizer.encode(text, add_special_tokens=False),
                'decoding': 'rendered-neural-calls',
                'composition': {'format': FORMAT, 'questions': parts,
                                'max_tokens_per_call': max_tokens, 'calls': calls}}


def validate_answer(request, answer, tokenizer, max_tokens):
    """Validate the transcript without mistaking rendered IDs for greedy IDs."""
    def greedy(value):
        ids = value['ids']
        if (not 0 < len(ids) <= max_tokens
                or any(type(token) is not int or not 0 <= token < len(tokenizer) for token in ids)
                or tokenizer.eos_token_id in ids[:-1]
                or not (len(ids) == max_tokens or ids[-1] == tokenizer.eos_token_id)
                or tokenizer.decode(ids, skip_special_tokens=True) != value['text']
                or set(value) - {'id', 'expert', 'ids', 'text', 'route'}):
            raise ValueError('Invalid actual greedy call transcript')
    parts = questions(request)
    if parts is None:
        greedy(answer)
        return
    composition = answer.get('composition', {})
    if (set(composition) != {'format', 'questions', 'max_tokens_per_call', 'calls'}
            or composition['format'] != FORMAT or composition['questions'] != parts
            or composition['max_tokens_per_call'] != max_tokens
            or len(composition['calls']) != 2 or answer.get('decoding') != 'rendered-neural-calls'):
        raise ValueError('Composition must execute exactly the question-derived calls')
    for call in composition['calls']:
        greedy(call)
        if call['route'] != 'expert':
            raise ValueError('Each constituent must execute the declared expert')
    text = '; '.join(call['text'] for call in composition['calls'])
    if (answer['text'] != text or answer['route'] != 'expert'
            or answer['ids'] != tokenizer.encode(text, add_special_tokens=False)):
        raise ValueError('Rendered answer differs from the actual neural outputs')
