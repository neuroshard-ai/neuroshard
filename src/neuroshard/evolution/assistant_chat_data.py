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

SPLITS = {'train': (95000, 12), 'development': (93000, 6), 'confirmation': (94000, 12),
          'confirmation2': (97000, 12), 'confirmation3': (99000, 12)}
TOPICS = {'train': ('gardening', 'personal finance', 'commuting', 'note taking', 'exercise', 'cooking at home',
                    'focus at work', 'travel packing', 'email habits', 'learning to code', 'board games', 'volunteering'),
          'development': ('remote work', 'composting', 'learning a language', 'saving for a trip', 'team meetings',
                          'cycling to work'),
          'confirmation3': ('astronomy', 'baking bread', 'cold showers', 'journaling', 'container gardening',
                            'learning to swim', 'photography', 'woodworking', 'meditation', 'home brewing coffee',
                            'sign language', 'rock climbing'),
          'confirmation2': ('bird watching', 'retirement saving', 'learning guitar', 'time zones', 'bike repair',
                            'first aid', 'podcasting', 'tea brewing', 'family history', 'recycling', 'chess openings',
                            'desk ergonomics'),
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
        if constraint.get('markup') == 'strip':
            # Emphasis around the closing words is not a different ending (confirmation 3's six failures).
            text = re.sub(r'[*_`"\'“”]+', '', text)
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


THIRD = {
    'bullets': 'Using exactly {n} bullet points and no other text, share advice on {topic}.',
    'short': 'Answer in {n} words or fewer: why might someone care about {topic}?',
    'json': 'Output a JSON object, nothing else, whose keys are exactly {keys}. Topic: {topic}.',
    'include': 'Say something brief about {topic}, and work in the word "{w}" somewhere.',
    'exclude': 'Explain {topic} in three sentences; the word "the" must not appear anywhere.',
    'ending': 'Cheer me on about {topic} in a short paragraph whose final words are "{phrase}".',
    'lowercase': 'write four short lines about {topic}. no capital letters anywhere, please.',
    'arithmetic': 'Tickets cost {b} each. I buy {a} tickets, then return {c} of them for a full refund. '
                  'How much did I spend in the end? Put the number last.',
}


def _third(kind, rng, topic):
    w = rng.choice(WORDS)
    if kind == 'bullets':
        n = rng.randint(3, 6)
        return THIRD[kind].format(n=n, topic=topic), [{'kind': 'bullets', 'n': n}]
    if kind == 'short':
        n = rng.choice((25, 40, 60))
        return THIRD[kind].format(n=n, topic=topic), [{'kind': 'max_words', 'n': n}]
    if kind == 'json':
        keys = rng.sample(['title', 'summary', 'steps', 'difficulty', 'time_needed'], 3)
        return THIRD[kind].format(keys=', '.join(keys), topic=topic), [{'kind': 'json_keys', 'keys': keys}]
    if kind == 'include':
        return THIRD[kind].format(w=w, topic=topic), [{'kind': 'include', 'word': w}]
    if kind == 'exclude':
        return THIRD[kind].format(topic=topic), [{'kind': 'exclude', 'word': 'the'}]
    if kind == 'ending':
        phrase = rng.choice(('Keep going', 'You have got this', 'Onward'))
        return THIRD[kind].format(topic=topic, phrase=phrase), [{'kind': 'ends_with', 'phrase': phrase}]
    if kind == 'lowercase':
        return THIRD[kind].format(topic=topic), [{'kind': 'lowercase'}]
    a, b = rng.randint(4, 15), rng.randint(12, 75)
    c = rng.randint(1, a - 1)
    return THIRD[kind].format(a=a, b=b, c=c), [{'kind': 'number', 'value': (a - c) * b}]


FOURTH = {
    'bullets': 'Give {n} pointers on {topic} as a bulleted list, with no intro or outro.',
    'short': 'Briefly, {n} words max: what is the point of {topic}?',
    'json': 'Describe getting started with {topic} as a JSON object containing only the keys {keys}.',
    'include': 'One or two lines on {topic}; the word "{w}" has to be in there.',
    'exclude': 'Tell me about {topic} in three sentences, never using the word "the".',
    'ending': 'Motivate me to try {topic}, and make "{phrase}" the very last words.',
    'lowercase': 'describe {topic} in a few sentences using lowercase letters only.',
    'arithmetic': 'A box of {b} pens costs {a} each box. I order {c} boxes and pay a {d} delivery fee. '
                  'What is my total? Finish with the number.',
}


def _fourth(kind, rng, topic):
    w = rng.choice(WORDS)
    if kind == 'bullets':
        n = rng.randint(3, 6)
        return FOURTH[kind].format(n=n, topic=topic), [{'kind': 'bullets', 'n': n}]
    if kind == 'short':
        n = rng.choice((25, 40, 60))
        return FOURTH[kind].format(n=n, topic=topic), [{'kind': 'max_words', 'n': n}]
    if kind == 'json':
        keys = rng.sample(['title', 'summary', 'steps', 'difficulty', 'time_needed'], 3)
        return FOURTH[kind].format(keys=', '.join(keys), topic=topic), [{'kind': 'json_keys', 'keys': keys}]
    if kind == 'include':
        return FOURTH[kind].format(w=w, topic=topic), [{'kind': 'include', 'word': w}]
    if kind == 'exclude':
        return FOURTH[kind].format(topic=topic), [{'kind': 'exclude', 'word': 'the'}]
    if kind == 'ending':
        phrase = rng.choice(('Start today', 'Go for it', 'Enjoy the ride'))
        return FOURTH[kind].format(topic=topic, phrase=phrase), [{'kind': 'ends_with', 'phrase': phrase}]
    if kind == 'lowercase':
        return FOURTH[kind].format(topic=topic), [{'kind': 'lowercase'}]
    a, b, c, d = rng.randint(6, 40), rng.choice((10, 12, 20, 24)), rng.randint(2, 9), rng.randint(3, 15)
    return FOURTH[kind].format(a=a, b=b, c=c, d=d), [{'kind': 'number', 'value': a * c + d}]


def _task(kind, rng, topic, split):
    if split == 'confirmation3':
        return _fourth(kind, rng, topic)
    if split == 'confirmation2':
        return _third(kind, rng, topic)
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
GENERATED_TOPICS = {
    'generated-train': ('kayaking', 'bookbinding', 'urban cycling', 'pottery', 'birdhouse building', 'beekeeping',
                        'stargazing apps', 'thrift shopping', 'pickling vegetables', 'trail running', 'calligraphy',
                        'origami', 'knitting', 'museum visits', 'language exchange', 'camping gear', 'sourdough',
                        'home workouts', 'speed reading', 'digital decluttering'),
    'generated-development': ('sailing', 'jigsaw puzzles', 'herb drying', 'fountain pens', 'skateboarding',
                              'mushroom foraging'),
    'confirmation4': ('ice skating', 'puppetry', 'cheese making', 'model trains', 'orienteering', 'quilting',
                      'scuba diving', 'bonsai', 'improv theatre', 'map reading', 'candle making', 'fly fishing')}
GENERATED_COUNTS = {'generated-train': 40, 'generated-development': 6, 'confirmation4': 12}
EXCLUDED = ('the', 'and', 'very', 'really')
PHRASES = ('Keep at it', 'You can do this', 'One step at a time', 'Make it happen', 'See you out there')


def _generated_task(kind, rng, topic, template_rows, index):
    """A Bedrock-written request filled with our values; the constraint and any answer are ours."""
    from neuroshard.evolution import assistant_phrasings as phrasings

    if kind == 'arithmetic':
        sub = ('arith-refund', 'arith-fee', 'arith-coupon')[index % 3]
        rows = template_rows[sub]
        a, b = rng.randint(4, 15), rng.randint(12, 75)
        values = {'a': a, 'b': b, 'c': rng.randint(1, a - 1) if sub == 'arith-refund' else rng.randint(2, 9),
                  'd': rng.randint(3, 15)}
        template = rows[(index // 3) % len(rows)]
        return template.format(**values), [{'kind': 'number', 'value': phrasings.ARITHMETIC[sub](values)}]
    rows = template_rows[kind]
    template = rows[index % len(rows)]
    if kind == 'bullets':
        n = rng.randint(3, 6)
        return template.format(n=n, topic=topic), [{'kind': 'bullets', 'n': n}]
    if kind == 'short':
        n = rng.choice((25, 40, 60))
        return template.format(n=n, topic=topic), [{'kind': 'max_words', 'n': n}]
    if kind == 'json':
        keys = rng.sample(['title', 'summary', 'steps', 'difficulty', 'time_needed'], 3)
        return template.format(keys=', '.join(keys), topic=topic), [{'kind': 'json_keys', 'keys': keys}]
    if kind == 'include':
        word = rng.choice(WORDS)
        return template.format(word=word, topic=topic), [{'kind': 'include', 'word': word}]
    if kind == 'exclude':
        word = rng.choice(EXCLUDED)
        return template.format(word=word, topic=topic), [{'kind': 'exclude', 'word': word}]
    if kind == 'ending':
        phrase = rng.choice(PHRASES)
        return template.format(phrase=phrase, topic=topic), [{'kind': 'ends_with', 'phrase': phrase, 'markup': 'strip'}]
    if kind == 'lowercase':
        return template.format(topic=topic), [{'kind': 'lowercase'}]
    raise ValueError(kind)


def generated_cases(split):
    from neuroshard.evolution import assistant_phrasings as phrasings

    rows, seed = phrasings.load('chat', split), {'generated-train': 95500, 'generated-development': 93700,
                                                 'confirmation4': 99500}[split]
    out = []
    for kind in KINDS:
        for index in range(GENERATED_COUNTS[split]):
            rng = random.Random(f'{seed}:{kind}:{index}')
            topic = GENERATED_TOPICS[split][index % len(GENERATED_TOPICS[split])]
            text, constraints = _generated_task(kind, rng, topic, rows, index)
            out.append({'id': f'chat-{split}-{kind}-{index:02d}', 'split': split, 'family': kind,
                        'turns': [{'user': text, 'expected': {'constraints': constraints}}]})
    return out


def cases(split):
    if split in GENERATED_COUNTS:
        return generated_cases(split)
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
