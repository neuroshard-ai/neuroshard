"""Varied phrasings from a larger model, with every fact, constraint and expected outcome kept in our own code.

The authored grammars phrase each question kind a few ways, and three sealed confirmations showed
that both prompt rules and a module trained on them learn the templates. This module asks Claude
on Amazon Bedrock for many natural rewordings of each request kind. It is given only the kind's
description and the placeholder names, never a workspace, a fact or an answer. Every returned
template must keep its placeholders, may not contain digits or document IDs, and is deduplicated.
Claude writes the user's side only; every assistant message we learn from comes from our verified
solver or from the parent's own replies that pass the checker.

Templates for training and for the sealed test come from separate calls with separate instructions
(casual / indirect / multi-clause for training; terse / formal / context-first for the test), and
the two sets are checked to share no template and no normalised 6-word span. Each kept template is
then filled with values and judged by a separate call that sees only the filled request and the
kind's description; arithmetic templates are filled twice and kept only if an independent solve
matches the declared formula both times, so an ambiguous problem cannot enter either set.
"""

import json
import random
import re

MODEL = 'us.anthropic.claude-sonnet-4-6'
REGION = 'us-east-1'

# Documents: (what the user asks for, placeholders). Values and answers come from the workspace.
KINDS = {
    'flight-time': ('the departure time of the user\'s current flight to {city}', ['city']),
    'hotel-code': ('the hotel booking confirmation code for the user\'s stay in {city}', ['city']),
    'leave-days': ('how many days of annual leave the user gets under the current policy', []),
    'invoice-sum': ('the total the user still owes {vendor} across all unpaid invoices', ['vendor']),
    'claim-deadline': ('the last date the user can file an expense claim for something bought on {day}', ['day']),
    'notice': ('how many days of notice the user must give to end their apartment lease', []),
    'deposit': ('the deposit amount in the user\'s apartment lease', []),
    'battery': ('the battery life in hours of the user\'s {product}', ['product']),
    'owner': ('who owns the vendor shortlist according to the team meeting notes', []),
    'not-found-insurance': ('the user\'s car insurance policy number (the documents do not contain it)', []),
    'not-found-return': ('the time of the user\'s return flight home from {city} (the documents only have the outbound '
                         'flight)', ['city']),
}
# Chat: the request must state its constraint exactly; the checker scores it.
CHAT_KINDS = {
    'bullets': ('advice on {topic} as exactly {n} bullet points and nothing else', ['n', 'topic']),
    'short': ('an answer about {topic} in at most {n} words', ['n', 'topic']),
    'json': ('a JSON object about {topic} whose keys are exactly {keys}, and no other text', ['keys', 'topic']),
    'include': ('a short text about {topic} that contains the word "{word}"', ['word', 'topic']),
    'exclude': ('a few sentences about {topic} that never use the word "{word}"', ['word', 'topic']),
    'ending': ('an encouraging short paragraph about {topic} whose very last words are "{phrase}"', ['phrase', 'topic']),
    'lowercase': ('a few sentences about {topic} written entirely in lowercase letters', ['topic']),
    'arith-refund': ('a word problem: the user buys {a} items at {b} each, then returns {c} of them for a full '
                     'refund; how much they spent in the end; the reply should end with the number', ['a', 'b', 'c']),
    'arith-fee': ('a word problem: the user orders {c} units that cost {a} each and pays a one-off delivery fee of '
                  '{d}; the total paid; the reply should end with the number', ['a', 'c', 'd']),
    'arith-coupon': ('a word problem: the user buys {a} packs at {b} each and has a coupon for {c} off the whole '
                     'order; the final price; the reply should end with the number', ['a', 'b', 'c']),
}
ARITHMETIC = {'arith-refund': lambda v: (v['a'] - v['c']) * v['b'], 'arith-fee': lambda v: v['a'] * v['c'] + v['d'],
              'arith-coupon': lambda v: v['a'] * v['b'] - v['c']}
STYLES = {'train': 'casual, indirect, chatty, multi-clause, with everyday wording, sometimes with an aside',
          'confirmation4': 'terse, formal, imperative, or starting with context before the request'}
PLACEHOLDER = re.compile(r'\{([a-z]+)\}')
FILE = 'config/experiments/assistant-general-phrasings.json'
# A second development set, generated after generated-development had judged two units (L5, L6).
FILE2 = 'config/experiments/assistant-general-phrasings-development2.json'
# Round 3: development3 (judges L8) and the sealed confirmation5, in two further styles.
FILE3 = 'config/experiments/assistant-general-phrasings-round3.json'
# Case splits built from the generated sets: split name -> (file, set name). generated-train-extra
# draws further workspaces for chosen families from the training templates.
SPLITS = {'generated-train': (FILE, 'train'), 'generated-train-extra': (FILE, 'train'),
          'generated-development': (FILE, 'development'), 'confirmation4': (FILE, 'confirmation4'),
          'generated-development2': (FILE2, 'development2'),
          'generated-development3': (FILE3, 'development3'), 'confirmation5': (FILE3, 'confirmation5'),
          'generated-train-styles': (None, None)}


# Spent sets (they judged units or were opened) become training wording for later units.
STYLES_TRAIN = ((FILE, 'development'), (FILE2, 'development2'), (FILE, 'confirmation4'))


def load(route, split):
    """The committed templates of one route and case split, by kind."""
    from pathlib import Path

    root = Path(__file__).resolve().parents[3]
    if split == 'generated-train-styles':
        merged = {}
        for path, name in STYLES_TRAIN:
            for kind, rows in json.loads((root / path).read_text())['sets'][route][name].items():
                merged.setdefault(kind, []).extend(rows)
        return merged
    path, name = SPLITS[split]
    data = json.loads((root / path).read_text())
    return data['sets'][route][name]


def table(route):
    return KINDS if route == 'documents' else CHAT_KINDS


def prompt(description, names, count, style, route='documents'):
    fields = ', '.join('{' + n + '}' for n in names) or 'none'
    who = ('an assistant that can search their own documents' if route == 'documents'
           else 'a general chat assistant (no files involved)')
    return (f'Write {count} different ways a person might ask {who}, for: {description}.\n'
            f'Style: {style}. Vary vocabulary and sentence structure a lot; do not reuse openings.\n'
            f'Placeholders to keep exactly as written: {fields}. Use each placeholder exactly once and state any '
            'requirement (counts, words, format, ending) unambiguously. '
            'Use no other braces, no digits, no dates, no names, no document IDs.\n'
            'Return only a JSON array of strings.')


def valid(template, names):
    return (isinstance(template, str) and 12 <= len(template) <= 280 and sorted(PLACEHOLDER.findall(template)) ==
            sorted(names) and not re.search(r'\d', template) and 'doc-' not in template.lower()
            and template.count('{') == template.count('}') == len(names))


def parse(text, names):
    start, end = text.find('['), text.rfind(']')
    try:
        values = json.loads(text[start:end + 1]) if 0 <= start < end else []
    except ValueError:
        values = []
    seen, out = set(), []
    for value in values:
        key = ' '.join(value.lower().split()) if isinstance(value, str) else None
        if key and key not in seen and valid(value, names):
            seen.add(key)
            out.append(' '.join(value.split()))
    return out


def client():
    import boto3
    from botocore.config import Config

    return boto3.client('bedrock-runtime', region_name=REGION,
                        config=Config(retries={'max_attempts': 8, 'mode': 'adaptive'}, read_timeout=300))


def converse(text, bedrock, max_tokens=4096, temperature=1.0):
    response = bedrock.converse(modelId=MODEL, messages=[{'role': 'user', 'content': [{'text': text}]}],
                                inferenceConfig={'maxTokens': max_tokens, 'temperature': temperature})
    content = response['output']['message'].get('content') or [{}]
    return content[0].get('text', ''), response.get('usage', {})


def generate(kind, count, style, bedrock=None, route='documents'):
    """One generation call; returns the valid, deduplicated templates and the raw response for the record."""
    description, names = table(route)[kind]
    text, usage = converse(prompt(description, names, count, style, route), bedrock or client())
    return parse(text, names), {'text': text, 'usage': usage}


def sample_values(route, kind, rng):
    """Placeholder values used only to judge a template; cases draw their own."""
    if route == 'documents':
        return {'city': rng.choice(('Bergen', 'Cusco', 'Perth')), 'vendor': 'Larkspur Supplies',
                'day': '2027-05-14', 'product': 'Orion travel kettle'}
    values = {'topic': rng.choice(('kayaking', 'bookbinding', 'urban cycling')), 'n': rng.choice((3, 4, 5, 30, 45)),
              'keys': 'name, reason, first_step', 'word': rng.choice(('steady', 'and')), 'phrase': 'Keep at it'}
    values.update(a=rng.randint(6, 14), b=rng.randint(12, 60), c=rng.randint(2, 5), d=rng.randint(4, 15))
    return values


def judge_prompt(request, description):
    return ('A user sent an assistant this request:\n<request>\n' + request + '\n</request>\n'
            f'Intended task: {description}.\n'
            'Answer NO only if the request asks for something different or conflicting, leaves out a requirement '
            'of the intended task, or could reasonably be read in a second way that changes the correct reply. '
            'Casual tone, extra chatter or a mention of files is fine. Give a one-sentence reason, then a last line '
            'that is exactly VERDICT: YES or VERDICT: NO.')


def solve_prompt(request):
    return ('Solve this exactly. Reply with only the final number, no units or words.\n<problem>\n' + request +
            '\n</problem>')


def judge(route, kind, template, bedrock, seed):
    """Semantic check on a filled template; arithmetic is also solved independently twice."""
    description, names = table(route)[kind]
    rng = random.Random(f'{seed}:{kind}:{template}')
    record = []
    for trial in range(2 if kind in ARITHMETIC else 1):
        values = sample_values(route, kind, rng)
        request = template.format(**values)
        filled = re.sub(r'\s*\([^)]*\)', '', description).format(**values)  # the judge sees the ask, not the answer
        verdict, usage = converse(judge_prompt(request, filled), bedrock, max_tokens=160, temperature=0)
        last = verdict.strip().splitlines()[-1].strip().upper() if verdict.strip() else ''
        row = {'request': request, 'verdict': verdict.strip(), 'usage': usage}
        ok = last == 'VERDICT: YES'
        if kind in ARITHMETIC:
            answer, usage2 = converse(solve_prompt(request), bedrock, max_tokens=300, temperature=0)
            numbers = re.findall(r'-?\d+(?:\.\d+)?', answer.replace(',', ''))
            row.update(answer=answer.strip(), expected=ARITHMETIC[kind](values), usage_solve=usage2)
            ok = ok and bool(numbers) and float(numbers[-1]) == ARITHMETIC[kind](values)
        record.append(row)
        if not ok:
            return False, record
    return True, record


def spans(template, n=6):
    words = re.findall(r'[a-z]+', PLACEHOLDER.sub(' ', template.lower()))
    return {' '.join(words[i:i + n]) for i in range(len(words) - n + 1)}


def disjoint(train, test):
    """No shared template and no shared normalised six-word span between the two sets."""
    a = {t for rows in train.values() for t in rows}
    b = {t for rows in test.values() for t in rows}
    shared = set().union(*(spans(t) for t in a)) & set().union(*(spans(t) for t in b))
    return not (a & b) and not shared, sorted(shared)[:10]


def holdout(rows, fraction=0.15, seed=0):
    """Split one kind's training templates into (train, development) by a fixed shuffle."""
    rows = sorted(rows)
    random.Random(seed).shuffle(rows)
    cut = max(2, round(len(rows) * fraction))
    return rows[cut:], rows[:cut]
