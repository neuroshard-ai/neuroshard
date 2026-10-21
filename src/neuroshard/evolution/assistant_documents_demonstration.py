"""Verified demonstrations of the documents route, written by a goal-directed solver for training cases only.

Each demonstration searches with the user's own words, reads what it relies on, and, where a rule in
a document decides the answer, reads that rule and applies it with the declared tool before answering.
For a question the documents cannot answer, it searches, reads the closest match, and reports not-found
with the reason (for a return flight: the itinerary only has the outbound leg). Every demonstration is
replayed through the workspace and must pass the scorer, or the job stops.

Under the evidence interface (version 3) an answer quotes, word for word, the sentence of each cited
document that states it or the rule it applies. A not-found reason quotes what the closest document
does contain and says why that is not what was asked.
"""

import json
import re

from neuroshard.evolution import assistant_documents as documents
from neuroshard.evolution import assistant_documents_data as data
from neuroshard.evolution.modular_reference_execution import identity


def call(name, **arguments):
    return '<tool_call>\n' + json.dumps({'name': name, 'arguments': arguments}) + '\n</tool_call>'


def _find(world, needle):
    return next(d for d in world['documents'] if needle(d))


TRAINING = ('train', 'generated-train', 'generated-train-extra', 'generated-train-styles', 'generated-train-styles2')


def sentences(text):
    return [part.strip() for part in re.split(r'(?<=\.)\s+', text) if part.strip()]


def evidence(document, needles):
    """The first sentence of a document containing any needle, without its full stop, as a verbatim quote."""
    for sentence in sentences(document['content']):
        if any(needle in sentence for needle in needles):
            return sentence.rstrip('.')
    raise ValueError(f"no sentence of {document['id']} states {needles}")


def keywords(case):
    """A focused search for one request: the subject it asks about and the names it gives (place, vendor,
    product), each copied from the request itself, never the surrounding chatter."""
    question, family = case['turns'][0]['user'], case['family']
    named = [name for name in data.CITIES + data.VENDORS + data.PRODUCTS if name in question]
    subject = {'flight-time': 'flight itinerary departure', 'hotel-code': 'hotel confirmation code',
               'leave-days': 'annual leave policy', 'invoice-sum': 'invoice unpaid amount due',
               'claim-deadline': 'expense claims policy', 'notice': 'apartment lease notice',
               'deposit': 'apartment lease deposit', 'battery': 'battery life specification',
               'owner': 'meeting notes vendor shortlist', 'not-found-insurance': 'car insurance policy number',
               'not-found-return': 'return flight itinerary'}[family]
    return ' '.join(named + [subject])


TRAINING = TRAINING + ('generated-train-styles3',)


def texts(case, version=2, query='message'):
    """The assistant replies, in order, that solve one training case.

    ``query='keywords'`` searches with the request's key terms (``keywords``) instead of its whole text,
    so a request wrapped in unrelated sentences is searched for what it asks.
    """
    if case['split'] not in TRAINING:
        raise ValueError('demonstrations are written for training cases only')
    if version not in (1, 2, 3):
        # /4 demonstrations need revision-selection cases and calls this solver does not write. Refuse rather than
        # emit calls the interface rejects (DESIGN_LATEST_REVISION.md; review of PR #124).
        raise ValueError(f'demonstrations are not defined for workspace-documents/{version}')
    world, turn = case['world'], case['turns'][0]
    question, expected, family = turn['user'], turn['expected'], case['family']
    out = [call('search_documents', query=keywords(case) if query == 'keywords' else question)]
    if expected.get('not_found'):
        if family == 'not-found-return':
            itinerary = _find(world, lambda d: d['title'].startswith('Itinerary ') and 'superseded' not in d['title'])
            out.append(call('read_document', document_id=itinerary['id']))
            reason = 'The current itinerary only lists the outbound flight; no return flight is in the documents.'
            if version == 3:
                leg = evidence(itinerary, ['Flight '])
                reason = (f'The current itinerary only has the outbound flight ("{leg}"); it has no return flight '
                          'home, so its departure time is not in the documents.')
        else:
            reason = 'No document mentions a car insurance policy number.'
            if version == 3:
                reason = ('No document mentions car or vehicle insurance or a policy number; the closest is the '
                          'leave and expenses policy, which is an employer policy.')
        out.append(call('report_not_found', reason=reason))
        out.append('I could not find that in your documents: ' + reason[0].lower() + reason[1:])
        return out
    sources = expected['source_ids']
    for key in sources:
        out.append(call('read_document', document_id=key))
    fact = expected['facts'][0]
    if family == 'claim-deadline':
        policy = _find(world, lambda d: d['id'] == sources[0])
        window = int(policy['content'].split('within ')[1].split(' days')[0])
        days = re.findall(r'\d{4}-\d{2}-\d{2}', question)
        if len(days) != 1:
            raise ValueError(f"expected one expense date in {case['id']}")
        day = days[0]
        out.append(call('shift_date', start_date=day, days=window))
        answer = f'File the claim by {fact}: the current policy allows {window} days from the expense date.'
    elif family == 'invoice-sum':
        amounts = [int(_find(world, lambda d, k=k: d['id'] == k)['content'].split('Amount due: ')[1].split('.')[0])
                   for k in sources]
        out.append(call('calculate', operation='add', left=amounts[0], right=amounts[1]))
        answer = f'You owe {fact} in total across the two unpaid invoices.'
    else:
        answer = f'{fact}.'
    cited = ', '.join(sources)
    if version == 3:
        needles = (['within '] if family == 'claim-deadline' else ['Amount due: '] if family == 'invoice-sum'
                   else expected['facts'])
        quotes = [evidence(_find(world, lambda d, k=k: d['id'] == k), needles) for k in sources]
        out.append(call('submit_answer', answer=answer, source_ids=sources, evidence=quotes))
    else:
        out.append(call('submit_answer', answer=answer, source_ids=sources))
    out.append(f'{answer} (Sources: {cited})')
    return out


def demonstration(case, policy):
    """Run one demonstration through the documents workspace; it must pass the scorer."""
    from neuroshard.evolution import assistant_general as general

    replies = iter(texts(case, general.version_of(policy), policy.get('search_query', 'message')))

    def respond(messages, tools):
        return {'text': next(replies), 'terminated': True, 'executed': True, 'token_ids': [], 'input_token_ids': [],
                'prompt_sha256': identity({'messages': messages, 'tools': tools}), 'seconds': 0.0,
                'model': 'demonstration'}

    row = general.documents_episode(case, respond, policy)
    if not row['passed'] or next(replies, None) is not None:
        raise ValueError(f"the demonstration does not solve {case['id']}")
    version = general.version_of(policy)
    written = sum(1 for g in row['generations'] for _ in documents.parse_calls(g['text'], version))
    if written != len(row['calls']) or any('error' in c['result'] for c in row['calls']):
        # Every tool call the solver wrote must parse and succeed; a rejected call never passes silently.
        raise ValueError(f"the demonstration of {case['id']} has a rejected tool call")
    return row


def trajectory(case, row, policy):
    """The trainer's item: the full conversation, with loss on every assistant reply."""
    messages = [{'role': 'system', 'content': policy['system_instruction']}]
    history = documents_messages(case, row, row.get('version', 2))
    messages += history
    return {'case_id': case['id'], 'family': case['family'], 'split': case['split'], 'messages': messages,
            'trainable': [m['role'] == 'assistant' for m in messages], 'complete': True, 'rounds': 1}


def documents_messages(case, row, version=2):
    """User, assistant and tool messages of a recorded episode, in order."""
    messages = [{'role': 'user', 'content': case['turns'][0]['user']}]
    calls = iter(row['calls'])
    for generated in row['generations']:
        messages.append({'role': 'assistant', 'content': generated['text']})
        for _ in documents.parse_calls(generated['text'], version):
            messages.append({'role': 'tool', 'content': json.dumps(next(calls)['result'], ensure_ascii=False,
                                                                     sort_keys=True)})
    return messages


def all_demonstrations(policy, split='train'):
    cases = [c for name in ([split] if isinstance(split, str) else split) for c in data.cases(name)]
    return [(case, demonstration(case, policy)) for case in cases]
