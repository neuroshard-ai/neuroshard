"""Realistic documents requests, written independently of the training templates, with verified gold answers.

The documents route's training, probes and sealed sets are drawn from template families with a fixed fact
schema. A unit can master those and still fail on what a person actually writes. This module builds a set
that shares nothing with them: Bedrock (Claude Sonnet 4.6) writes small personal workspaces in free prose
(emails, receipts, notes, policies, itineraries, with superseded versions and near-duplicates) and natural
questions over them, about a quarter of which the documents cannot answer.

Every item is kept only if its gold answer survives two independent checks:

- **mechanical**: each fact appears in a cited document's text (or, for a computed value, the cited
  documents contain every input and the stated operation reproduces it), and the scorer accepts the gold;
- **blind**: a second model call, given only the documents and the question, answers it; its answer must
  contain the same facts (or say not found for an unanswerable one).

The set is scored with the route's own deterministic scorer (``assistant_documents.score_round``): the
facts stated, exactly the needed sources cited, or an explicit not-found. It is an evaluation, never
training data.
"""

import hashlib
import json
from pathlib import Path
import random
import re

from neuroshard.evolution import assistant_documents as documents
from neuroshard.evolution import assistant_phrasings as phrasings

FORMAT = 'neuroshard-real-documents/1'
SETTINGS = ('a freelancer juggling client invoices and contracts', 'a parent organising school, medical and travel paperwork',
            'a small landlord with leases, repair quotes and tenant emails', 'an engineer with team notes, on-call rotas and vendor quotes',
            'a student with course syllabi, grant letters and housing documents', 'a retiree with pension letters, insurance and home repairs',
            'a nurse with shift schedules, certification renewals and payslips', 'a small shop owner with supplier orders, rent and permits',
            'someone moving abroad with visa letters, shipping quotes and bank notices',
            'a researcher with conference travel, reimbursements and reviewer emails',
            'a couple planning a wedding with venue, catering and guest emails',
            'a hobby beekeeper with association rules, equipment receipts and inspection notes')

WRITE = """You are building a test of an assistant that answers questions about a person's own documents.
Write ONE workspace for {setting}: {count} short documents (40 to 160 words each) in natural, varied prose:
emails, receipts, notes, policies, itineraries, letters. Make it realistic and a little messy:
- include at least one superseded or older version of something (an old policy, a revised quote, a moved date)
  that a careless reader would use instead of the current one;
- include near-duplicates and distracting numbers and dates;
- never use placeholder names; vary formats of dates and amounts as people really write them.

Then write {questions} questions the person might type, in their own words (casual, sometimes with context,
sometimes indirect). Mix:
- direct lookups, lookups that require choosing the current version, and a few that need one simple
  computation from stated values (a sum, a difference, a date a stated number of days later);
- exactly {missing} questions whose answer is NOT in the documents but sounds like it could be (ask about a
  detail the documents do not state).

For each answerable question give the gold answer as:
  "facts": the shortest exact strings that a correct answer must contain (copy dates and amounts exactly as
  the document writes them, or for a computed value the computed result written plainly, e.g. "2027-03-14"
  or "1,240"), and "source_ids": the IDs of exactly the documents needed; for a computation, also
  "computation": one sentence naming the inputs and the operation.
For each unanswerable question: "not_found": true.

Reply with only JSON:
{{"documents": [{{"id": "doc-xxxxxxxx", "title": "...", "content": "..."}}],
  "questions": [{{"user": "...", "facts": ["..."], "source_ids": ["doc-..."], "computation": "..."}},
                {{"user": "...", "not_found": true}}]}}
Document IDs are "doc-" followed by 8 lowercase hex characters."""

BLIND = """Here are a person's documents:
{documents}

Question: {question}

Answer using only these documents. If they do not state the answer, reply exactly NOT FOUND.
Otherwise give the short answer, then on a new line "Sources:" and the document IDs you used.
If the answer needs a computation, do it."""


def ids_ok(world):
    return all(re.fullmatch(r'doc-[0-9a-f]{8}', d['id']) for d in world['documents']) and len(
        {d['id'] for d in world['documents']}) == len(world['documents'])


def normalized(text):
    return documents.normalize(text)


def mechanical(world, item):
    """The gold answer is supported by the cited documents and accepted by the route's scorer."""
    by_id = {d['id']: d for d in world['documents']}
    if item.get('not_found'):
        return True
    if not item.get('facts') or not item.get('source_ids') or not set(item['source_ids']) <= set(by_id):
        return False
    cited = ' '.join(by_id[i]['title'] + ' ' + by_id[i]['content'] for i in item['source_ids'])
    if not item.get('computation') and not all(normalized(f) in normalized(cited) for f in item['facts']):
        return False
    snapshot = {'answer': {'answer': '; '.join(item['facts']), 'source_ids': item['source_ids']}, 'not_found': None,
                'found_ids': list(by_id)}
    return documents.score_round(snapshot, {'facts': item['facts'], 'source_ids': item['source_ids']}, True,
                                 '; '.join(item['facts']))


def blind(world, item, bedrock):
    text = '\n\n'.join(f"[{d['id']}] {d['title']}\n{d['content']}" for d in world['documents'])
    reply = phrasings.converse(BLIND.format(documents=text, question=item['user']), bedrock, 400, 0.0)[0]
    if item.get('not_found'):
        return 'NOT FOUND' in reply.upper()
    return 'NOT FOUND' not in reply.upper() and all(normalized(f) in normalized(reply) for f in item['facts'])


def parse(reply):
    start, end = reply.find('{'), reply.rfind('}')
    return json.loads(reply[start:end + 1])


def build(count=12, seed=0, cache=None, progress=print):
    """``count`` workspaces with verified questions; replies are cached so a rerun is identical and free."""
    bedrock, rng = phrasings.client(), random.Random(seed)
    cache = Path(cache) if cache else None
    store = json.loads(cache.read_text()) if cache and cache.exists() else {}
    cases, rejected = [], {'parse': 0, 'ids': 0, 'mechanical': 0, 'blind': 0}
    for index in range(count):
        setting = SETTINGS[index % len(SETTINGS)]
        prompt = WRITE.format(setting=setting, count=rng.randint(7, 12), questions=8, missing=2)
        key = hashlib.sha256(f'{seed}:{index}:{prompt}'.encode()).hexdigest()
        if key not in store:
            store[key] = phrasings.converse(prompt, bedrock, 8000, 1.0)[0]
            if cache:
                cache.write_text(json.dumps(store))
        try:
            written = parse(store[key])
            world = {'documents': [{k: d[k] for k in ('id', 'title', 'content')} for d in written['documents']]}
            documents.Workspace(world, 3)
        except (ValueError, KeyError, TypeError):
            rejected['parse'] += 1
            continue
        if not ids_ok(world):
            rejected['ids'] += 1
            continue
        for number, item in enumerate(written.get('questions', [])):
            if not mechanical(world, item):
                rejected['mechanical'] += 1
                continue
            check = f'blind:{key}:{number}'
            if check not in store:
                store[check] = blind(world, item, bedrock)
                if cache:
                    cache.write_text(json.dumps(store))
            if not store[check]:
                rejected['blind'] += 1
                continue
            expected = ({'not_found': True} if item.get('not_found') else
                        {'facts': item['facts'], 'source_ids': sorted(item['source_ids'])})
            cases.append({'id': f'real-docs-{index:02d}-{number}', 'split': 'real', 'family': (
                'real-not-found' if item.get('not_found') else 'real-computed' if item.get('computation')
                else 'real-lookup'), 'setting': setting, 'world': world,
                'turns': [{'user': item['user'], 'expected': expected}]})
        progress(f'workspace {index}: {len(cases)} verified so far, rejected {rejected}')
    return {'format': FORMAT, 'model': phrasings.MODEL, 'seed': seed, 'workspaces': count, 'rejected': rejected,
            'cases': cases, 'sha256': hashlib.sha256(json.dumps(cases, sort_keys=True).encode()).hexdigest()}


# A second set written a different way, so a unit cannot fit one author's habits: a free-form life story first, its
# documents from the story, and the questions from a separate call that sees only the documents (never the
# answers or the story). The gold is then derived by another call and must survive the same two checks.
STORY = """Write a short, specific account (150 words) of one week in the life of a real-seeming person of your
choosing: their job, family, money, travel, health or home admin. Seed: {seed}. Only the account, no title."""

DOCUMENTS_FROM_STORY = """Here is an account of someone's week:
{story}

Write the {count} documents this person would actually have on their computer or phone because of that week
(emails, receipts, bookings, letters, notes, forms), 30 to 180 words each, with real-looking details. Some
should mention the same thing differently, and one should be an earlier version of another.
Reply with only JSON: {{"documents": [{{"id": "doc-xxxxxxxx", "title": "...", "content": "..."}}]}}
Document IDs are "doc-" followed by 8 lowercase hex characters."""

ASK = """Here are a person's documents:
{documents}

Write 8 questions this person might type into an assistant about their own documents, the way people really
type (short, casual, sometimes vague or with a typo). Include 2 that the documents do NOT answer.
Reply with only JSON: {{"questions": ["...", "..."]}}"""

GOLD = """Here are a person's documents:
{documents}

Question: {question}

If the documents do not state the answer, reply with only JSON {{"not_found": true}}.
Otherwise reply with only JSON {{"facts": [the shortest exact string(s) a correct answer must contain, copied
as the document writes them, or a computed result], "source_ids": [exactly the document IDs needed],
"computation": "only if the answer needs one: the inputs and the operation"}}."""


def build_independent(count=40, seed=0, cache=None, progress=print):
    """The second evaluation set: story, documents, questions and gold each from a separate call."""
    bedrock = phrasings.client()
    cache = Path(cache) if cache else None
    store = json.loads(cache.read_text()) if cache and cache.exists() else {}

    def ask(key, text, tokens, temperature):
        if key not in store:
            store[key] = phrasings.converse(text, bedrock, tokens, temperature)[0]
            if cache:
                cache.write_text(json.dumps(store))
        return store[key]

    cases, rejected = [], {'parse': 0, 'ids': 0, 'gold': 0, 'mechanical': 0, 'blind': 0}
    for index in range(count):
        story = ask(f'story:{seed}:{index}', STORY.format(seed=f'{seed}-{index}'), 400, 1.0)
        try:
            written = parse(ask(f'docs:{seed}:{index}', DOCUMENTS_FROM_STORY.format(story=story, count=8), 6000, 1.0))
            world = {'documents': [{k: d[k] for k in ('id', 'title', 'content')} for d in written['documents']]}
            documents.Workspace(world, 3)
        except (ValueError, KeyError, TypeError):
            rejected['parse'] += 1
            continue
        if not ids_ok(world):
            rejected['ids'] += 1
            continue
        text = '\n\n'.join(f"[{d['id']}] {d['title']}\n{d['content']}" for d in world['documents'])
        try:
            questions = parse(ask(f'ask:{seed}:{index}', ASK.format(documents=text), 800, 1.0))['questions']
        except (ValueError, KeyError, TypeError):
            rejected['parse'] += 1
            continue
        for number, question in enumerate(questions[:8]):
            try:
                gold = parse(ask(f'gold:{seed}:{index}:{number}', GOLD.format(documents=text, question=question),
                                 400, 0.0))
            except (ValueError, KeyError, TypeError):
                rejected['gold'] += 1
                continue
            item = {'user': question, **gold}
            if not mechanical(world, item):
                rejected['mechanical'] += 1
                continue
            key = f'blind2:{seed}:{index}:{number}'
            if key not in store:
                store[key] = blind(world, item, bedrock)
                if cache:
                    cache.write_text(json.dumps(store))
            if not store[key]:
                rejected['blind'] += 1
                continue
            expected = ({'not_found': True} if item.get('not_found') else
                        {'facts': item['facts'], 'source_ids': sorted(item['source_ids'])})
            cases.append({'id': f'real2-docs-{index:02d}-{number}', 'split': 'real2', 'world': world,
                          'family': ('real-not-found' if item.get('not_found') else 'real-computed'
                                     if item.get('computation') else 'real-lookup'),
                          'turns': [{'user': question, 'expected': expected}]})
        progress(f'workspace {index}: {len(cases)} verified, rejected {rejected}')
    return {'format': FORMAT + '+independent', 'model': phrasings.MODEL, 'seed': seed, 'workspaces': count,
            'rejected': rejected, 'cases': cases,
            'sha256': hashlib.sha256(json.dumps(cases, sort_keys=True).encode()).hexdigest()}
