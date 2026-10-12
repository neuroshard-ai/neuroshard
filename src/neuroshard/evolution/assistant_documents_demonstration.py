"""Verified demonstrations of the documents route, written by a goal-directed solver for training cases only.

Each demonstration searches with the user's own words, reads what it relies on, and, where a rule in
a document decides the answer, reads that rule and applies it with the declared tool before answering.
For a question the documents cannot answer, it searches, reads the closest match, and reports not-found
with the reason (for a return flight: the itinerary only has the outbound leg). Every demonstration is
replayed through the workspace and must pass the scorer, or the job stops.
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


def texts(case):
    """The assistant replies, in order, that solve one training case."""
    if case['split'] not in ('train', 'generated-train'):
        raise ValueError('demonstrations are written for training cases only')
    world, turn = case['world'], case['turns'][0]
    question, expected, family = turn['user'], turn['expected'], case['family']
    out = [call('search_documents', query=question)]
    if expected.get('not_found'):
        if family == 'not-found-return':
            itinerary = _find(world, lambda d: d['title'].startswith('Itinerary ') and 'superseded' not in d['title'])
            out.append(call('read_document', document_id=itinerary['id']))
            reason = 'The current itinerary only lists the outbound flight; no return flight is in the documents.'
        else:
            reason = 'No document mentions a car insurance policy number.'
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
    out.append(call('submit_answer', answer=answer, source_ids=sources))
    out.append(f'{answer} (Sources: {cited})')
    return out


def demonstration(case, policy):
    """Run one demonstration through the documents workspace; it must pass the scorer."""
    from neuroshard.evolution import assistant_general as general

    replies = iter(texts(case))

    def respond(messages, tools):
        return {'text': next(replies), 'terminated': True, 'executed': True, 'token_ids': [], 'input_token_ids': [],
                'prompt_sha256': identity({'messages': messages, 'tools': tools}), 'seconds': 0.0,
                'model': 'demonstration'}

    row = general.documents_episode(case, respond, policy)
    if not row['passed'] or next(replies, None) is not None:
        raise ValueError(f"the demonstration does not solve {case['id']}")
    return row


def trajectory(case, row, policy):
    """The trainer's item: the full conversation, with loss on every assistant reply."""
    messages = [{'role': 'system', 'content': policy['system_instruction']}]
    history = documents_messages(case, row)
    messages += history
    return {'case_id': case['id'], 'family': case['family'], 'split': case['split'], 'messages': messages,
            'trainable': [m['role'] == 'assistant' for m in messages], 'complete': True, 'rounds': 1}


def documents_messages(case, row):
    """User, assistant and tool messages of a recorded episode, in order."""
    messages = [{'role': 'user', 'content': case['turns'][0]['user']}]
    calls = iter(row['calls'])
    for generated in row['generations']:
        messages.append({'role': 'assistant', 'content': generated['text']})
        for _ in documents.parse_calls(generated['text']):
            messages.append({'role': 'tool', 'content': json.dumps(next(calls)['result'], ensure_ascii=False,
                                                                     sort_keys=True)})
    return messages


def all_demonstrations(policy, split='train'):
    cases = data.cases(split)
    return [(case, demonstration(case, policy)) for case in cases]
