"""Training demonstrations over realistic documents, each one replayed and scored before it is kept.

The template families teach the documents route a fixed schema; on free prose (see ``real_documents``) the units
trained on them lost to the untrained parent on plain lookups. This module writes training data closer to what
people keep: Bedrock (Claude Sonnet 4.6) writes messy personal workspaces in settings disjoint from the evaluation
set's, with questions, gold answers and, for each, the search query, the answer sentence, the tool computation
or the not-found reason a careful assistant would give.

Every item becomes a scripted conversation through the real documents workspace (search, read, compute,
submit_answer or report_not_found, then the final reply), and is kept only if:

- its search finds every cited document (a second, narrower query is tried before giving up);
- a computed answer's tool calls, executed by the workspace, produce the gold value;
- the replayed episode passes the route's scorer;
- the gold survives the same blind check as the evaluation set (a separate model call with only the documents).

Anyone can recheck the first three from the published corpus alone: they are replays, not judgments.
"""

import hashlib
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import random
import threading

from neuroshard.evolution import assistant_documents as documents
from neuroshard.evolution import assistant_phrasings as phrasings
from neuroshard.evolution import real_documents as real

FORMAT = 'neuroshard-realistic-documents-training/1'
SETTINGS = ('a dentist running a two-chair practice', 'a teacher planning a school trip abroad',
            'a translator with agency purchase orders and invoices', 'a family renovating a kitchen with three quotes',
            'a dog breeder with vet records and buyer contracts', 'a choir treasurer with dues, venue hire and grants',
            'a junior lawyer with court dates and client letters', 'a farmer with feed orders, subsidies and inspections',
            'a gym owner with memberships, leases and equipment warranties', 'an IT contractor with SOWs and timesheets',
            'a pensioner with utility bills, a GP letter and a council tax notice', 'a PhD student with lab bookings and stipend letters',
            'a festival volunteer coordinator with shift lists and permits', 'a photographer with client bookings and licences',
            'a parent of a child with allergies: school plans, prescriptions, camp forms',
            'a property manager with service charges and contractor reports', 'a sailor with mooring fees and survey reports',
            'a café owner with hygiene inspections, supplier price lists and staff rotas',
            'a remote worker with a coworking contract, expense rules and travel bookings',
            'a car owner with service history, insurance renewal and a parking fine appeal',
            'a book club organiser with library loans and meeting minutes', 'a cyclist training for a sportive with entries and kit orders',
            'a homeowner with solar panel quotes, warranties and an energy tariff', 'a carer with respite bookings and benefit letters',
            'a game developer with publisher milestones and contractor invoices', 'a beekeeper selling honey at markets with stall fees',
            'an architect with planning applications and fee proposals', 'a musician with gig contracts and royalty statements',
            'a hospital doctor with rota swaps, CPD courses and expense claims', 'a nonprofit with donor pledges and grant reports')

WRITE = """You are building TRAINING data for an assistant that answers questions about a person's own documents.
Write ONE workspace for {setting}: {count} documents of 30 to 180 words, in the forms people actually keep:
{forms}. Make it realistic and untidy: an older version superseded by a newer one, near-duplicate documents,
several dates and amounts that are easy to confuse, informal notes next to formal letters.

Then write {questions} questions the person might type (casual, sometimes with context or indirect), mixing
direct lookups, lookups that must pick the current version, {computed} that need one computation, and exactly
{missing} whose answer is NOT in the documents but sounds like it might be.

For each question give:
- "search": the short keyword query a careful assistant would search with (2 to 5 words from the documents);
- for answerable ones: "facts" (usually ONE string: the exact value asked for, copied as the document writes
  it, or the computed result; add a second only if the question asks for two things; never add supporting
  details such as version numbers, dates or reference codes the question did not ask for), "source_ids"
  (exactly the documents needed), and "answer" (one short natural sentence answering the person that contains
  every fact);
- for a computed one also "steps": the tool calls that compute it, each either
  {{"tool": "calculate", "operation": "add" | "subtract" | "multiply", "left": <integer>, "right": <integer>}} or
  {{"tool": "shift_date", "start_date": "YYYY-MM-DD", "days": <integer>}}, whose final result is the fact
  (use whole numbers only: amounts in the smallest unit if needed, and say so in "answer");
- for an unanswerable one: "not_found": true and "reason" (one sentence naming what the closest document does
  contain instead).

Reply with only JSON:
{{"documents": [{{"id": "doc-xxxxxxxx", "title": "...", "content": "..."}}],
  "questions": [{{"user": "...", "search": "...", "facts": ["..."], "source_ids": ["doc-..."], "answer": "...",
                  "steps": [...]}}, {{"user": "...", "search": "...", "not_found": true, "reason": "..."}}]}}
Document IDs are "doc-" followed by 8 lowercase hex characters."""

FORMS = ('emails with quoted replies, receipts, a policy extract, a calendar note',
         'letters, an invoice table written as text, meeting minutes, a text-message summary',
         'a contract excerpt, a revised quote, a booking confirmation, a to-do note',
         'forwarded emails, a statement, a form with filled-in fields, a handwritten-style note')


def call(name, **arguments):
    return '<tool_call>\n' + json.dumps({'name': name, 'arguments': arguments}) + '\n</tool_call>'


def replies(world, item):
    """The scripted assistant replies for one item, checked against the workspace as they are built; None if the
    item cannot be demonstrated faithfully (its search misses a source, or its steps do not produce the fact)."""
    probe = documents.Workspace(world, 2)
    found = {r['id'] for r in probe.execute({'name': 'search_documents', 'arguments': {'query': item['search']}})
             .get('results', [])}
    out = [call('search_documents', query=item['search'])]
    if item.get('not_found'):
        if not found or not item.get('reason'):
            return None
        closest = probe.execute({'name': 'search_documents', 'arguments': {'query': item['search']}})['results'][0]['id']
        reason = item['reason'].strip()
        return out + [call('read_document', document_id=closest), call('report_not_found', reason=reason),
                      'I could not find that in your documents: ' + reason[0].lower() + reason[1:]]
    sources = list(item['source_ids'])
    missing = [s for s in sources if s not in found]
    if missing:
        title = next(d['title'] for d in world['documents'] if d['id'] == missing[0])
        again = {r['id'] for r in probe.execute({'name': 'search_documents', 'arguments': {'query': title}})
                 .get('results', [])}
        if set(missing) - again:
            return None
        out.append(call('search_documents', query=title))
    out += [call('read_document', document_id=key) for key in sources]
    result = None
    for step in item.get('steps') or []:
        arguments = {k: v for k, v in step.items() if k != 'tool'}
        result = probe.execute({'name': step.get('tool'), 'arguments': arguments})
        if 'error' in result:
            return None
        out.append(call(step['tool'], **arguments))
    if item.get('steps'):
        value = str(result.get('value', result.get('date', '')))
        if not any(documents.normalize(value) in documents.normalize(f) or documents.normalize(f) in
                   documents.normalize(value) for f in item['facts']):
            return None
    answer = item['answer'].strip()
    if not all(documents.normalize(f) in documents.normalize(answer) for f in item['facts']):
        return None
    out.append(call('submit_answer', answer=answer, source_ids=sources))
    out.append(f"{answer} (Sources: {', '.join(sources)})")
    return out


def episode(case, texts, policy):
    """Replay the scripted replies through the real episode loop; the row must pass the scorer."""
    from neuroshard.evolution import assistant_general as general
    from neuroshard.evolution.modular_reference_execution import identity

    stream = iter(texts)

    def respond(messages, tools):
        return {'text': next(stream), 'terminated': True, 'executed': True, 'token_ids': [], 'input_token_ids': [],
                'prompt_sha256': identity({'messages': messages, 'tools': tools}), 'seconds': 0.0, 'model': 'script'}

    row = general.documents_episode(case, respond, policy)
    if not row['passed'] or next(stream, None) is not None:
        return None
    return row


def build(count, seed, cache, policy, workers=8, progress=print):
    bedrock, lock = phrasings.client(), threading.Lock()
    cache = Path(cache)
    store = json.loads(cache.read_text()) if cache.exists() else {}
    rng = random.Random(seed)
    prompts = []
    for index in range(count):
        setting = SETTINGS[index % len(SETTINGS)]
        prompts.append(WRITE.format(setting=setting, count=rng.randint(7, 12), forms=FORMS[index % len(FORMS)],
                                    questions=10, computed='two', missing=2))

    def ask(key, text, tokens, temperature):
        with lock:
            if key in store:
                return store[key]
        value = phrasings.converse(text, bedrock, tokens, temperature)[0]
        with lock:
            store[key] = value
        return value

    def workspace(index):
        key = hashlib.sha256(f'train:{seed}:{index}:{prompts[index]}'.encode()).hexdigest()
        reply = ask(key, prompts[index], 9000, 1.0)
        try:
            written = real.parse(reply)
            world = {'documents': [{k: d[k] for k in ('id', 'title', 'content')} for d in written['documents']]}
            documents.Workspace(world, 2)
        except (ValueError, KeyError, TypeError):
            return [], {'parse': 1}
        if not real.ids_ok(world):
            # Rename IDs the writer did not make hex (doc-abc12def) to canonical ones, everywhere they appear.
            renamed = {d['id']: 'doc-' + hashlib.sha256(f"{key}:{d['id']}".encode()).hexdigest()[:8]
                       for d in world['documents']}
            if len(set(renamed)) != len(world['documents']):
                return [], {'ids': 1}
            text = json.dumps(written)
            for old in sorted(renamed, key=len, reverse=True):
                text = text.replace(f'"{old}"', f'"{renamed[old]}"').replace(old, renamed[old])
            written = json.loads(text)
            world = {'documents': [{k: d[k] for k in ('id', 'title', 'content')} for d in written['documents']]}
            if not real.ids_ok(world):
                return [], {'ids': 1}
        kept, rejected = [], {}
        for number, item in enumerate(written.get('questions', [])):
            try:
                if not real.mechanical(world, item):
                    raise ValueError('mechanical')
                texts = replies(world, item)
                if texts is None:
                    raise ValueError('replay')
                expected = ({'not_found': True} if item.get('not_found') else
                            {'facts': item['facts'], 'source_ids': sorted(item['source_ids'])})
                case = {'id': f'realtrain-{index:03d}-{number}', 'split': 'realistic-train', 'world': world,
                        'family': 'real-not-found' if item.get('not_found') else
                        'real-computed' if item.get('steps') else 'real-lookup',
                        'turns': [{'user': item['user'], 'expected': expected}]}
                if episode(case, texts, policy) is None:
                    raise ValueError('scorer')
                text = '\n\n'.join(f"[{d['id']}] {d['title']}\n{d['content']}" for d in world['documents'])
                verdict = ask(f'blind:{key}:{number}', real.BLIND.format(documents=text, question=item['user']), 400, 0.0)
                agrees = ('NOT FOUND' in verdict.upper()) if item.get('not_found') else (
                    'NOT FOUND' not in verdict.upper() and all(documents.normalize(f) in documents.normalize(verdict)
                                                               for f in item['facts']))
                if not agrees:
                    raise ValueError('blind')
                kept.append({**case, 'replies': texts})
            except (ValueError, KeyError, TypeError, AttributeError, IndexError) as error:
                reason = str(error) if str(error) in ('mechanical', 'replay', 'scorer', 'blind') else 'malformed'
                rejected[reason] = rejected.get(reason, 0) + 1
        return kept, rejected

    cases, rejected = [], {}
    with ThreadPoolExecutor(workers) as pool:
        for index, (kept, why) in enumerate(pool.map(workspace, range(count))):
            cases += kept
            for k, v in why.items():
                rejected[k] = rejected.get(k, 0) + v
            if index % 10 == 9:
                with lock:
                    cache.write_text(json.dumps(store))
                progress(f'{index + 1}/{count} workspaces: {len(cases)} kept, rejected {rejected}')
    with lock:
        cache.write_text(json.dumps(store))
    return {'format': FORMAT, 'model': phrasings.MODEL, 'seed': seed, 'workspaces': count, 'rejected': rejected,
            'cases': cases, 'sha256': hashlib.sha256(json.dumps(cases, sort_keys=True).encode()).hexdigest()}


def overlap(training, evaluation, n=8):
    """Shared word 8-grams between training and evaluation document texts and requests (should be ~0)."""
    def grams(cases):
        out = set()
        for case in cases:
            for text in [d['content'] for d in case['world']['documents']] + [case['turns'][0]['user']]:
                words = documents.words(text)
                out |= {' '.join(words[i:i + n]) for i in range(len(words) - n + 1)}
        return out
    a, b = grams(training), grams(evaluation)
    return {'training_grams': len(a), 'evaluation_grams': len(b), 'shared': len(a & b)}


def recipe(corpus, policy, spec, model_root, encode, replay, continue_from):
    """A documents growth recipe whose experience is a published realistic corpus, carried in full as evidence."""
    from neuroshard.evolution.realistic_train import trajectory

    return {'format': 'neuroshard-growth-recipe/1', 'route': 'documents', 'kind': 'realistic-corpus',
            'families': ['real-lookup', 'real-not-found', 'real-computed'], 'others': [],
            'corpus': corpus, 'corpus_sha256': corpus['sha256'],
            'arm': 'addition', 'spec': spec, 'model_root': model_root, 'base': 'granite-parent',
            'continue_from': continue_from,
            'sources': {'experience': [encode(trajectory(case, policy)) for case in corpus['cases']], 'replay': replay}}


def verify(recipe, policy, encode, replay_check):
    """An owner's check of a corpus recipe, with no model: every conversation replays through the documents
    workspace, passes the route's scorer, cites only documents its search found, and re-encodes to exactly the
    training sequence; the corpus hashes to what the recipe names; ``replay_check(sequences)`` checks the replay."""
    from neuroshard.evolution.realistic_train import trajectory

    corpus = recipe.get('corpus') or {}
    cases = corpus.get('cases', [])
    if hashlib.sha256(json.dumps(cases, sort_keys=True).encode()).hexdigest() != recipe.get('corpus_sha256'):
        return False
    if len(cases) != len(recipe['sources']['experience']) or len({c['id'] for c in cases}) != len(cases):
        return False
    for case, sequence in zip(cases, recipe['sources']['experience']):
        try:
            if case['split'] != 'realistic-train' or episode(case, case['replies'], policy) is None:
                return False
            expected = case['turns'][0]['expected']
            computed = any('"calculate"' in r or '"shift_date"' in r for r in case['replies'])
            gold = {'facts': expected.get('facts'), 'source_ids': expected.get('source_ids'),
                    'not_found': expected.get('not_found'), 'computation': computed or None}
            if not real.mechanical(case['world'], {k: v for k, v in gold.items() if v}):
                return False  # every fact is in a cited document's text, or comes from a tool computation
            if encode(trajectory(case, policy)) != sequence:
                return False
        except (ValueError, KeyError, TypeError, StopIteration):
            return False
    return bool(replay_check(recipe['sources']['replay']))
