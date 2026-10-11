"""Authored general-chat tasks with exactly checkable constraints, for the general route without tools.

Each case is one ordinary request (write, rewrite, summarise, explain, compute) with one or two
constraints a program can verify: a bullet count, a word limit, valid JSON with given keys, a
required or forbidden word, an ending phrase, lowercase, or an exact numeric answer. The
constraint is the score; prose quality beyond it is recorded, not graded. Development and
confirmation use disjoint topics and phrasings.
"""

import json
import random
import re

SPLITS = {'train': (95000, 12), 'development': (93000, 6), 'confirmation': (94000, 12)}
TOPICS = {'train': ('gardening', 'personal finance', 'commuting', 'note taking', 'exercise', 'cooking at home',
                    'focus at work', 'travel packing', 'email habits', 'learning to code', 'board games', 'volunteering'),
          'development': ('remote work', 'composting', 'learning a language', 'saving for a trip', 'team meetings',
                          'cycling to work'),
          'confirmation': ('home networking', 'meal planning', 'public speaking', 'reading habits', 'job interviews',
                           'houseplants', 'budget travel', 'sleep routines', 'code review', 'running a book club',
                           'moving house', 'online privacy')}
WORDS = ('practical', 'simple', 'weekly', 'clear', 'small', 'steady')


def bullets(text):
    return [line for line in text.splitlines() if re.match(r'^\s*([-*•]|\d+[.)])\s+\S', line)]


def word_count(text):
    return len(re.findall(r"[A-Za-z0-9']+", text))


def check(constraint, text):
    kind = constraint['kind']
    if kind == 'bullets':
        return len(bullets(text)) == constraint['n']
    if kind == 'max_words':
        return 0 < word_count(text) <= constraint['n']
    if kind == 'json_keys':
        body = text.strip()
        fenced = re.fullmatch(r'```(?:json)?\s*(.*?)\s*```', body, re.DOTALL)
        try:
            value = json.loads(fenced[1] if fenced else body)
        except ValueError:
            return False
        return isinstance(value, dict) and set(value) == set(constraint['keys'])
    if kind == 'include':
        return re.search(rf'\b{re.escape(constraint["word"])}\b', text, re.IGNORECASE) is not None
    if kind == 'exclude':
        return re.search(rf'\b{re.escape(constraint["word"])}\b', text, re.IGNORECASE) is None
    if kind == 'ends_with':
        return text.strip().rstrip('.!').lower().endswith(constraint['phrase'].lower().rstrip('.!'))
    if kind == 'lowercase':
        letters = [c for c in text if c.isalpha()]
        return bool(letters) and all(not c.isupper() for c in letters)
    if kind == 'number':
        numbers = [n.replace(',', '') for n in re.findall(r'-?\d[\d,]*(?:\.\d+)?', text)]
        return bool(numbers) and numbers[-1].rstrip('0').rstrip('.') == str(constraint['value']).rstrip('0').rstrip('.')
    raise ValueError(f'unknown constraint {kind}')


def score(expected, final_text, terminated):
    return bool(terminated and isinstance(final_text, str) and final_text.strip()
                and all(check(c, final_text) for c in expected['constraints']))


def _task(kind, rng, topic, split):
    w = rng.choice(WORDS)
    dev = split != 'confirmation'
    if kind == 'bullets':
        n = rng.randint(3, 6)
        text = (f'Give me exactly {n} bullet points of advice on {topic}.' if dev
                else f'List {n} tips about {topic}, one per bullet, and nothing else.')
        return text, [{'kind': 'bullets', 'n': n}]
    if kind == 'short':
        n = rng.choice((25, 40, 60))
        text = (f'In at most {n} words, explain why {topic} matters.' if dev
                else f'Keep it under {n + 1} words: what is the main benefit of {topic}?')
        return text, [{'kind': 'max_words', 'n': n}]
    if kind == 'json':
        keys = rng.sample(['title', 'summary', 'steps', 'difficulty', 'time_needed'], 3)
        text = (f'Reply only with a JSON object with the keys {", ".join(keys)} describing how to start {topic}.'
                if dev else f'Return JSON (no prose) with exactly these fields: {", ".join(keys)}. Subject: {topic}.')
        return text, [{'kind': 'json_keys', 'keys': keys}]
    if kind == 'include':
        text = (f'Write two sentences about {topic} that use the word "{w}".' if dev
                else f'Two sentences on {topic}, please; make sure "{w}" appears.')
        return text, [{'kind': 'include', 'word': w}]
    if kind == 'exclude':
        text = (f'Describe {topic} in three sentences without using the word "the".' if dev
                else f'Three sentences about {topic}, and avoid the word "the" entirely.')
        return text, [{'kind': 'exclude', 'word': 'the'}]
    if kind == 'ending':
        phrase = rng.choice(('Good luck', 'That is all', 'Happy planning'))
        text = (f'Give a short pep talk about {topic} and end with the words "{phrase}".' if dev
                else f'Encourage me about {topic} in a few lines; finish with "{phrase}".')
        return text, [{'kind': 'ends_with', 'phrase': phrase}]
    if kind == 'lowercase':
        text = (f'Write a haiku about {topic} using only lowercase letters.' if dev
                else f'a short poem about {topic}, all in lowercase, no capitals at all.')
        return text, [{'kind': 'lowercase'}]
    if kind == 'arithmetic':
        a, b, c = rng.randint(3, 12), rng.randint(15, 60), rng.randint(2, 9)
        text = (f'I buy {a} items at {b} each and get {c} more items free. How much do I pay in total? '
                'Give the number at the end.' if dev else
                f'A pack costs {b}. I need {a} packs and have a coupon for {c} off the whole order. '
                'What is the final price? End with the number.')
        value = a * b if dev else a * b - c
        return text, [{'kind': 'number', 'value': value}]
    raise ValueError(kind)


KINDS = ('bullets', 'short', 'json', 'include', 'exclude', 'ending', 'lowercase', 'arithmetic')


def cases(split):
    if split not in SPLITS:
        raise ValueError('unknown chat split')
    seed, count = SPLITS[split]
    out = []
    for kind in KINDS:
        for index in range(count):
            rng = random.Random(f'{seed}:{kind}:{index}')
            topic = TOPICS[split][index % len(TOPICS[split])]
            text, constraints = _task(kind, rng, topic, split)
            out.append({'id': f'chat-{split}-{kind}-{index:02d}', 'split': split, 'family': kind,
                        'turns': [{'user': text, 'expected': {'constraints': constraints}}]})
    return out
