"""Training demonstrations for ``workspace-documents/4``: revision selection over structured metadata.

The served documents unit was trained on ``/2`` transcripts. Offline on ``/4`` it almost never calls
``current_document``, misuses ``effective_document`` and over-reports not-found (documents-v4-dev-20261025).
This module writes ``/4`` conversations to train it, following ``realistic_training``: Bedrock writes messy
personal and small-business workspaces in which some documents form explicit version families (structured
``project``, ``family``, ``revision``, ``status`` and optional ``effective_from`` / ``effective_until``
fields, DESIGN_LATEST_REVISION.md) beside unrelated documents, plus the questions a person would type.

The writer proposes; our code decides. For every revision question the gold document comes from
``document_revisions`` (the same selection the tools run), never from the writer:

- latest approved (``current_document``), in effect on a date or "now" with the date given
  (``effective_document``): the writer's expected outcome must equal the selection's, and an answer's facts must
  be in the selected document's text;
- a named revision, a draft asked for explicitly, a comparison or history: exactly the named members of the
  family, read and cited;
- declines (``known_no_approved``, ``ambiguous``, ``incomplete``, ``none_effective``) are a ``report_not_found``
  whose reason this module writes from the tool's own result (outcome, statuses and document IDs), never a
  submitted answer. That is the only scorer-consistent representation: under v4 a round counts only a submitted,
  evidenced answer or an explicit ``report_not_found``, and evidence quotes cannot state metadata;
- ordinary lookups, computed answers and not-found, as in ``realistic_training``, and workspaces with no metadata
  at all, where a "latest approved" request is answered from text with that limit stated.

An item is kept only if its scripted conversation replays through the real v4 workspace and episode loop, its
tool calls produce the gold, the route scorer (v4) passes, its evidence quotes are found in the cited documents,
it fits the policy's token budget, and a blind check (a separate model call shown only the documents, their
fields and the selection rules) agrees. ``verify`` recomputes every one of these that needs no model.
"""

import datetime
import hashlib
import json
from pathlib import Path
import random
import re
import threading
from concurrent.futures import ThreadPoolExecutor

from neuroshard.evolution import assistant_documents as documents
from neuroshard.evolution import document_revisions as revisions

FORMAT = 'neuroshard-revision-documents-training/1'
# The author. Only this profile is invocable on the account (Opus 4.6, Opus 4.5 and Sonnet 4.5 were refused with
# AccessDeniedException on 2026-10-26), so the brief, settings and procedure differ from G2's instead.
AUTHOR = 'us.anthropic.claude-sonnet-4-6'
SPLIT = 'revision-train'
SETTINGS = (
    'a bakery owner with recipe cards, allergen sheets and a wholesale price list',
    'a band manager with tour riders, set lists and venue contracts',
    'a small charity with a safeguarding policy, volunteer handbook and trustee minutes',
    'a dog-walking business with a client agreement, price sheet and insurance letters',
    'a self-employed electrician with quote templates, a safety checklist and supplier invoices',
    'a primary school PTA with a fundraising plan, raffle rules and committee emails',
    'a holiday-let owner with house rules, a cleaning checklist and booking emails',
    'a community garden with plot rules, a tool-shed rota and council letters',
    'a wedding florist with order forms, a delivery schedule and supplier quotes',
    'a running club with a membership policy, race-entry rules and kit orders',
    'a two-person web studio with a statement of work, a style guide and invoices',
    'a craft brewery with a tasting-room policy, keg price list and inspection notes',
    'a tenants association with a constitution, meeting minutes and repair logs',
    'a youth football coach with a code of conduct, training plan and parent emails',
    'a freelance editor with a rate card, style sheet and publisher contracts',
    'a family with a shared-custody schedule, school letters and medical notes',
    'a food truck with a menu, a pitch licence and a supplier agreement',
    'a yoga studio with a timetable, cancellation policy and teacher contracts',
    'a small museum volunteer team with an opening-hours notice, loan agreements and a fire plan',
    'a bike repair shop with a service menu, warranty terms and parts orders',
    'a home-schooling parent with a curriculum plan, exam bookings and tutor invoices',
    'a market gardener with a veg-box price list, delivery rounds and a grant offer',
    'a choir with a rehearsal schedule, a concert budget and venue terms',
    'a startup founder with a shareholder agreement, an investor update and an office lease',
    'a hairdresser with a price list, a patch-test policy and a chair-rental agreement',
    'a residents parking scheme with permit rules, council notices and appeal letters',
    'a caravan owner with site rules, a pitch fee schedule and service records',
    'a translator collective with a fee schedule, a confidentiality policy and client briefs',
    'a pottery studio with a membership agreement, kiln rules and class bookings',
    'a scout group with a camp kit list, a risk assessment and a payment schedule',
)
FORMS = ('emails with quoted replies, notes, a price list, a policy written as numbered points',
         'letters, a schedule written as text, minutes, a text-message summary',
         'an agreement excerpt, a revised plan, a booking confirmation, a to-do note',
         'forwarded emails, a form with filled-in fields, a checklist, a handwritten-style note')
# Version-family shapes the writer is asked for; outcomes are always recomputed by ``document_revisions``.
PATTERNS = {
    'draft-newest': 'three revisions: revision 1 approved or superseded, revision 2 approved, revision 3 a draft '
                    'whose TEXT sounds final (for example it says "FINAL", "approved by the committee" or '
                    '"this replaces revision 2") while its status field is "draft"; key values differ between revisions',
    'approved-newest': 'two or three revisions where the highest revision is "approved" and the earlier ones '
                       '"superseded"; at least one key value is the same in two revisions and one differs',
    'single': 'one family with a single revision, revision 1, status "approved"',
    'same-title': 'two different families whose documents have the exact same title (either two different '
                  '"family" values in one project, or the same "family" value in two different projects), each '
                  'with one or two revisions and different values',
    'effective': 'revisions with effective dates: revision 1 "superseded" with effective_from and effective_until; '
                 'revision 2 "approved" with effective_from equal to revision 1\'s effective_until; and sometimes a '
                 'revision 3 "approved" with a LATER effective_from (then revision 2 also gets effective_until equal '
                 'to that date), so the highest approved revision is not yet in effect before that date',
    'no-approved': 'two or three revisions none of which is approved: either every one is "draft", or every one '
                   'is "superseded"',
    'ambiguous': 'two documents of one family that are BOTH "approved" with the same highest revision number',
    'incomplete': 'two or three revisions where one member has no "status" field or no "revision" field',
    'date-gap': 'two revisions with effective dates that leave a gap: revision 1 "superseded" with effective_until '
                'some weeks before revision 2 "approved" starts (effective_from)',
}
WEIGHTS = {'draft-newest': 4, 'approved-newest': 2, 'single': 2, 'same-title': 2, 'effective': 3, 'no-approved': 2,
           'ambiguous': 1, 'incomplete': 1, 'date-gap': 1}
TYPES = ('latest', 'now', 'as_of', 'named', 'compare', 'draft', 'lookup', 'computed', 'not_found', 'nometa_latest')
QUESTIONS = {
    'draft-newest': 'one "latest" for a value that differs between revisions 2 and 3, one "draft" asking what the '
                    'draft (or the newest unapproved version) says, one "named" or "compare"',
    'approved-newest': 'one "latest", one "named" about an earlier revision, one "compare" or history question',
    'single': 'one "latest"',
    'same-title': 'one "latest" for each family, phrased so the project or family is clear',
    'effective': 'one "as_of" with a date inside revision 1\'s period, one "now" where the person says today\'s '
                 'date (in a period of your choice), and one "latest"',
    'no-approved': 'one "latest" (expected outcome known_no_approved)',
    'ambiguous': 'one "latest" (expected outcome ambiguous)',
    'incomplete': 'one "latest" (expected outcome incomplete)',
    'date-gap': 'one "as_of" with a date in the gap (expected outcome none_effective) and one "as_of" inside '
                'revision 1\'s period',
}

WRITE = """You are building TRAINING data for an assistant that answers questions about a person's own documents
through tools. Write ONE workspace for {setting}: {count} documents of 30 to 170 words, in the forms people
actually keep: {forms}. Make it realistic and untidy: near-duplicates, several dates and amounts that are easy to
confuse, informal notes next to formal documents.

Some documents form VERSION FAMILIES with structured metadata fields stored beside the text (not in it):
"project" (a short name of the project or area, e.g. "Spring fair 2027"), "family" (a short id of the document
family, e.g. "price-list"), "revision" (an integer from 1), "status" ("approved", "draft" or "superseded"), and
optionally "effective_from" / "effective_until" (YYYY-MM-DD; effective_until is exclusive). Revisions of one
family usually share a title. Include these families:
{families}
All other documents have NO metadata fields at all.

How the assistant's tools choose a revision (follow this exactly when you set "outcome"):
- latest approved = the approved member with the highest revision; "known_current". If no member is approved:
  "known_no_approved". If two approved members share the highest approved revision: "ambiguous". If any member
  lacks status or an integer revision: "incomplete".
- in effect on a date d = the approved or superseded member whose [effective_from, effective_until) contains d
  (drafts never count): "known_effective"; none contains d: "none_effective"; two contain d: "ambiguous"; a
  candidate without effective_from: "incomplete".
- Metadata never comes from text: a draft whose text says "final" is still a draft.

Then write about {questions} questions the person might type (casual, sometimes indirect, sometimes with typos),
each with a "type":
- "latest": asks for something in the latest approved / current approved version of a family;
- "now": asks what is in effect now, and states today's date in the question (any natural format);
- "as_of": asks what applied on a stated date;
- "named": asks what a specific revision (e.g. "version 1", "the first draft", "rev 2") said;
- "compare": asks how something changed between two revisions, or for its history;
- "draft": explicitly asks what the draft / unapproved version says;
- "lookup": a plain lookup in a document WITHOUT metadata;
- "computed": needs one computation from stated values in documents without metadata;
- "not_found": the answer is NOT in the documents but sounds like it might be.
Per family: {per_family}. Also include two "lookup", one "computed" and one "not_found".

For each question give:
- "search": the short keyword query a careful assistant would search with (2 to 5 words from the documents);
- for latest/now/as_of: "project" and "family" exactly as in the fields, "outcome" (as defined above), and for
  now/as_of "date" (YYYY-MM-DD, the date the person means);
- for named/compare/draft: "project", "family" and "revisions" (the revision numbers it is about);
- for every question answered with a value: "facts" (usually ONE string: the exact value asked for, copied as the
  document writes it, or the computed result; one per revision for a comparison; never add details not asked
  for), "source_ids" (exactly the documents the answer relies on), "answer" (one short natural sentence that
  contains every fact, says which revision it comes from for family questions, and contains no other numbers or
  dates), and "quotes" (one to three short spans of 8 to 200 characters copied WORD FOR WORD from the cited
  documents' content that state each fact; for a computation, the spans that state its inputs);
- for "computed" also "steps": tool calls, each {{"tool": "calculate", "operation": "add" | "subtract" |
  "multiply", "left": <integer>, "right": <integer>}} or {{"tool": "shift_date", "start_date": "YYYY-MM-DD",
  "days": <integer>}}, whose final result is the fact (whole numbers only);
- for "not_found": "reason" (one sentence naming what the closest document contains instead);
- for latest/now/as_of whose outcome is not known_current/known_effective: no facts, no answer.

Reply with only JSON:
{{"documents": [{{"id": "doc-xxxxxxxx", "title": "...", "content": "...", "project": "...", "family": "...",
                 "revision": 2, "status": "approved"}}, {{"id": "doc-xxxxxxxx", "title": "...", "content": "..."}}],
  "questions": [{{"user": "...", "type": "latest", "search": "...", "project": "...", "family": "...",
                  "outcome": "known_current", "facts": ["..."], "source_ids": ["doc-..."], "answer": "...",
                  "quotes": ["..."]}}]}}
Document IDs are "doc-" followed by 8 lowercase hex characters."""

NOMETA = """You are building TRAINING data for an assistant that answers questions about a person's own documents.
Write ONE workspace for {setting}: {count} documents of 30 to 170 words, in the forms people actually keep:
{forms}. Make it realistic and untidy. The documents carry no structured metadata; some of their texts mention
versions ("v2", "final", "updated") the way people write them.

Then write about 9 questions the person might type (casual, sometimes indirect), each with a "type":
- "lookup": a plain lookup (four of these);
- "nometa_latest": asks for something in "the latest approved" or "current approved" version of a document that
  exists in ONE document only (no other version of it exists in the workspace) (two of these);
- "computed": needs one computation from stated values (one);
- "not_found": the answer is NOT in the documents but sounds like it might be (two).

For each question give "search" (2 to 5 keywords from the documents); for answered ones "facts" (usually ONE
string: the exact value asked for as the document writes it, or the computed result), "source_ids" (exactly the
documents needed), "answer" (one short sentence containing every fact and no other numbers or dates) and "quotes"
(one to three spans of 8 to 200 characters copied WORD FOR WORD from the cited documents that state each fact or
the computation's inputs); for "computed" also "steps" as tool calls {{"tool": "calculate", "operation": "add" |
"subtract" | "multiply", "left": <integer>, "right": <integer>}} or {{"tool": "shift_date", "start_date":
"YYYY-MM-DD", "days": <integer>}} whose final result is the fact; for "not_found" a "reason" (one sentence naming
what the closest document contains instead).

Reply with only JSON:
{{"documents": [{{"id": "doc-xxxxxxxx", "title": "...", "content": "..."}}],
  "questions": [{{"user": "...", "type": "lookup", "search": "...", "facts": ["..."], "source_ids": ["doc-..."],
                  "answer": "...", "quotes": ["..."]}}]}}
Document IDs are "doc-" followed by 8 lowercase hex characters."""

BLIND = """Here are a person's documents. Fields in brackets are structured metadata stored with each document;
the text is the document's content.

{documents}

Rules for versions (metadata only, never the text): a family is the documents sharing both "project" and
"family". The latest approved is the approved member with the highest revision; if none is approved, answer
DECLINE: no approved; if two approved share the highest approved revision, DECLINE: ambiguous; if a member lacks
status or revision, DECLINE: incomplete. What applied on a date d is the approved or superseded member whose
[effective_from, effective_until) contains d (effective_until exclusive; drafts never count); if none, DECLINE:
none in effect; if two, DECLINE: ambiguous; if a candidate lacks effective_from, DECLINE: incomplete.

Answer each question independently, using only these documents. Reply with only a JSON array with one object per
question, in order: {{"answer": "<short answer>", "sources": ["doc-..."]}}, or {{"answer": "NOT FOUND"}} if the
documents do not state it, or {{"answer": "DECLINE: <reason>"}} for a version question the rules cannot decide.
For a computation, do it and give the result.

Questions:
{questions}"""
FAMILY_TYPES = ('latest', 'now', 'as_of', 'named', 'compare', 'draft')
SELECTED = {'latest': ('known_current',), 'now': ('known_effective',), 'as_of': ('known_effective',)}
DECLINES = ('known_no_approved', 'ambiguous', 'incomplete', 'none_effective')
MONTHS = ('january', 'february', 'march', 'april', 'may', 'june', 'july', 'august', 'september', 'october',
          'november', 'december')
NO_METADATA_NOTE = ('These documents have no revision fields, so I cannot tell whether this is the latest approved '
                    'version.')
REASONS = ('parse', 'ids', 'type', 'family', 'selection', 'date', 'facts', 'answer', 'evidence', 'replay', 'scorer',
           'budget', 'blind', 'malformed')


class Reject(ValueError):
    """An item that cannot be demonstrated faithfully; ``str(error)`` is the reason recorded."""


def call(name, **arguments):
    return '<tool_call>\n' + json.dumps({'name': name, 'arguments': arguments}) + '\n</tool_call>'


def converse(text, bedrock, max_tokens=4096, temperature=1.0, model=AUTHOR):
    response = bedrock.converse(modelId=model, messages=[{'role': 'user', 'content': [{'text': text}]}],
                                inferenceConfig={'maxTokens': max_tokens, 'temperature': temperature})
    content = response['output']['message'].get('content') or [{}]
    return content[0].get('text', ''), response.get('usage', {})


def public_world(written):
    """The workspace as the user's device holds it: id, title, content and the metadata fields, nothing else.

    A revision the writer wrote as a digit string ("2") is stored as the integer it denotes: that is the writer's
    stated field, not something read from text. Every other value is kept as written, so a malformed field stays
    malformed and the family is ``incomplete``, exactly as the workspace treats a real user's file."""
    rows = []
    for document in written['documents']:
        row = {key: document[key] for key in ('id', 'title', 'content')}
        for key in revisions.FIELDS:
            if key in document and document[key] is not None:
                value = document[key]
                if key == 'revision' and isinstance(value, str) and value.strip().isdigit():
                    value = int(value.strip())
                row[key] = value
        rows.append(row)
    return {'documents': rows}


def mentions_date(text, iso):
    """The question states the date it is about (ISO, or day, month name and year), so no date is assumed."""
    if iso in text:
        return True
    day = datetime.date(int(iso[:4]), int(iso[5:7]), int(iso[8:]))
    lower = text.lower()
    month = MONTHS[day.month - 1]
    has_month = month in lower or re.search(rf'\b{month[:3]}\b', lower) is not None
    has_day = re.search(rf'\b0?{day.day}(st|nd|rd|th)?\b', lower) is not None
    numeric = re.search(rf'\b0?{day.day}/0?{day.month}/{day.year}\b|\b0?{day.month}/0?{day.day}/{day.year}\b', lower)
    return bool(numeric) or (has_month and has_day and str(day.year) in lower)


def sentences(text):
    return [part.strip() for part in re.split(r'(?<=[.!?;])\s+|\n+', text) if part.strip()]


def quote_for(content, fact):
    """A verbatim span of ``content`` (8 to 300 characters) stating ``fact``: its sentence, or a window around it."""
    target = documents.normalize(fact)
    for sentence in sentences(content):
        if target and target in documents.normalize(sentence):
            span = sentence.rstrip('.')
            if documents.MIN_QUOTE <= len(span) <= documents.MAX_QUOTE and documents.quoted(span, content):
                return span
            words = span.split()
            for width in range(len(target.split()), len(words) + 1):
                for start in range(0, len(words) - width + 1):
                    piece = ' '.join(words[start:start + width])
                    if (target in documents.normalize(piece) and len(piece) >= documents.MIN_QUOTE
                            and len(piece) <= 200 and documents.quoted(piece, content)):
                        return piece
    return None


def evidence(world, item, sources, computed):
    """One to four quotes, each found word for word in a cited document, together stating every fact.

    The writer's quotes are used when all of them are valid and (for a lookup) state every fact; otherwise, for a
    lookup, the sentence of a cited document that states each fact. A computed answer must have valid writer
    quotes: its result is in no document, so its inputs cannot be found mechanically."""
    by = {d['id']: d for d in world['documents']}
    facts = item.get('facts') or []
    given = [q.strip() for q in item.get('quotes') or [] if isinstance(q, str) and q.strip()]

    def valid(quote):
        return (documents.MIN_QUOTE <= len(quote) <= documents.MAX_QUOTE
                and any(documents.quoted(quote, by[s]['content']) for s in sources))
    if given and len(given) <= 4 and all(valid(q) for q in given):
        stated = all(any(documents.normalize(f) in documents.normalize(q) for q in given) for f in facts)
        covered = all(any(documents.quoted(q, by[s]['content']) for q in given) for s in sources)
        if computed or (stated and covered):
            return given
    if computed:
        return None
    out = []
    for source in sources:
        for fact in facts:
            quote = quote_for(by[source]['content'], fact)
            if quote and quote not in out:
                out.append(quote)
                break
    stated = all(any(documents.normalize(f) in documents.normalize(q) for q in out) for f in facts)
    return out if out and len(out) <= 4 and stated and all(valid(q) for q in out) else None


def family_members(probe, project, family):
    return revisions.members(probe.documents.values(), project, family)


def decline_reason(result, project, family, date=None):
    """What a decline says, written from the selection tool's own result: its outcome and the documents' IDs."""
    outcome, name = result['outcome'], f'"{family}" ({project})'
    asked = f'applied on {date}' if date else 'is the latest approved'
    if outcome == 'known_no_approved':
        rows = sorted(result['statuses'].items(), key=lambda item: (item[1][0], item[0]))
        parts = ', '.join(f'revision {rev} ({key}) is {"a draft" if status == "draft" else status}'
                          for key, (rev, status) in rows)
        if all(status == 'superseded' for _, (_, status) in rows):
            return f'No revision of {name} is approved: every version is superseded ({parts}).'
        return f'No revision of {name} is approved: {parts}.'
    if outcome == 'ambiguous':
        both = ' and '.join(result['candidates'])
        if date:
            return f'More than one revision of {name} was in effect on {date} ({both}), so I cannot tell which applied.'
        return (f'Two documents of {name} are both approved at the highest revision ({both}), so I cannot tell '
                'which one applies. Which do you mean?')
    if outcome == 'incomplete':
        if result.get('no_period'):
            return (f'The effective dates of {name} are incomplete ({", ".join(result["no_period"])} has no usable '
                    f'period), so I cannot tell which revision {asked}.')
        return (f'The revision metadata of {name} is incomplete ({", ".join(result["unorderable"])} has no usable '
                f'status or revision), so I cannot tell which revision {asked}.')
    if outcome == 'none_effective':
        return f'No recorded revision of {name} was in effect on {date}.'
    raise Reject('selection')


def gold(world, item, nometa=False):
    """The item's gold, decided by our code: (expected, gold record). Raises Reject when the writer disagrees."""
    probe = documents.Workspace(world, 4)
    kind = item.get('type')
    if kind not in TYPES or (nometa and kind in FAMILY_TYPES) or (not nometa and kind == 'nometa_latest'):
        raise Reject('type')
    record = {'type': kind}
    if kind in FAMILY_TYPES:
        project, family = item.get('project'), item.get('family')
        if not isinstance(project, str) or not isinstance(family, str):
            raise Reject('family')
        group = family_members(probe, project, family)
        if not group:
            raise Reject('family')
        record.update(project=project, family=family, members=sorted(m['id'] for m in group))
        if kind in SELECTED:
            date = item.get('date') if kind != 'latest' else None
            if kind != 'latest' and (not isinstance(date, str) or not revisions.DATE.match(date)
                                     or not mentions_date(item['user'], date)):
                raise Reject('date')
            arguments = {'project': project, 'family': family, **({'date': date} if date else {})}
            result = probe.execute({'name': 'current_document' if kind == 'latest' else 'effective_document',
                                    'arguments': arguments})
            if 'error' in result or result['outcome'] != item.get('outcome'):
                raise Reject('selection')
            record.update(date=date, outcome=result['outcome'])
            if result['outcome'] in DECLINES:
                record['reason'] = decline_reason(result, project, family, date)
                return {'not_found': True}, record
            if result['outcome'] not in SELECTED[kind]:
                raise Reject('selection')
            sources = [result['document']['id']]
            if sorted(item.get('source_ids') or []) != sources:
                raise Reject('selection')
        else:
            wanted = item.get('revisions')
            if (not isinstance(wanted, list) or not wanted or len(set(map(str, wanted))) != len(wanted)
                    or (kind == 'compare') != (len(wanted) >= 2)):
                raise Reject('family')
            sources = []
            for number in wanted:
                matches = [m for m in group if m['metadata'].get('revision') == number]
                if len(matches) != 1 or (kind == 'draft' and matches[0]['metadata'].get('status') != 'draft'):
                    raise Reject('family')
                sources.append(matches[0]['id'])
            if sorted(item.get('source_ids') or []) != sorted(sources):
                raise Reject('selection')
            record['revisions'] = list(wanted)
    elif kind == 'not_found':
        if not isinstance(item.get('reason'), str) or not 1 <= len(item['reason'].strip()) <= 400:
            raise Reject('answer')
        return {'not_found': True}, record
    else:
        sources = list(item.get('source_ids') or [])
        if not sources or len(set(sources)) != len(sources) or not set(sources) <= set(probe.documents):
            raise Reject('facts')
        if any('family' in probe.documents[s]['metadata'] for s in sources):
            raise Reject('family')  # a plain lookup must not quietly cite one revision of a family
    facts = item.get('facts')
    if not isinstance(facts, list) or not facts or not all(isinstance(f, str) and documents.normalize(f) for f in facts):
        raise Reject('facts')
    computed = kind == 'computed'
    if computed != bool(item.get('steps')):
        raise Reject('facts')
    text = ' '.join(probe.documents[s]['content'] for s in sources)
    if not computed and not all(documents.normalize(f) in documents.normalize(text) for f in facts):
        raise Reject('facts')
    if kind == 'compare' and not all(any(documents.normalize(f) in documents.normalize(probe.documents[s]['content'])
                                         for f in facts) for s in sources):
        raise Reject('facts')  # a comparison states something from every revision it compares
    answer = item.get('answer')
    if (not isinstance(answer, str) or not answer.strip()
            or not all(documents.normalize(f) in documents.normalize(answer) for f in facts)
            or (kind == 'draft' and 'draft' not in answer.lower())):
        raise Reject('answer')
    if not claims_ok(answer, [probe.documents[s] for s in sources], kind):
        raise Reject('answer')
    record.update(source_ids=sorted(sources), facts=list(facts))
    return {'facts': list(facts), 'source_ids': sorted(sources)}, record


REVISION_CLAIM = re.compile(r'\b(?:revision|rev\.?|version|v)\s*(\d+)\b', re.IGNORECASE)
CURRENCY_CLAIM = re.compile(r'\b(?:approved|latest|current|currently|final|superseded)\b', re.IGNORECASE)


def claims_ok(answer, cited, kind):
    """An answer claims only what the cited documents' fields record: every revision number it names is a cited
    document's ``revision``; and without metadata (``nometa_latest``) it makes no version or approval claim at all,
    since none can be read from text."""
    if kind == 'nometa_latest' and (REVISION_CLAIM.search(answer) or CURRENCY_CLAIM.search(answer)):
        return False
    revisions_cited = {d['metadata'].get('revision') for d in cited}
    return all(int(n) in revisions_cited for n in REVISION_CLAIM.findall(answer))


def replies(world, item, record):
    """The scripted assistant replies for one item, each tool call executed on a probe workspace as it is written.

    Raises Reject if the conversation cannot be demonstrated faithfully: the search does not show the needed
    documents (a second search by title is tried, as in ``realistic_training``), a tool refuses a call, the
    selection tool disagrees with the gold, the steps do not produce the fact, or the quotes are not found."""
    probe, out = documents.Workspace(world, 4), []

    def run(name, **arguments):
        result = probe.execute({'name': name, 'arguments': arguments})
        if 'error' in result:
            raise Reject('replay')
        out.append(call(name, **arguments))
        return result

    kind = record['type']
    query = item.get('search')
    if not isinstance(query, str) or not query.strip():
        raise Reject('replay')
    rows = run('search_documents', query=query)['results']
    found = {row['id'] for row in rows}
    if kind in SELECTED:
        satisfied = bool(found & set(record['members']))  # the search shows the family's project and family
        retry = sorted(record['members'])[0]
    elif kind == 'not_found':
        satisfied, retry = bool(rows), None
    else:
        satisfied = set(record['source_ids']) <= found
        retry = next((s for s in record['source_ids'] if s not in found), None)
    if not satisfied:
        if retry is None:
            raise Reject('replay')
        found |= {row['id'] for row in run('search_documents', query=probe.documents[retry]['title'])['results']}
        if (kind in SELECTED and not found & set(record['members'])) or (
                kind not in SELECTED and not set(record['source_ids']) <= found):
            raise Reject('replay')
    if kind == 'not_found':
        reason = item['reason'].strip()
        run('read_document', document_id=rows[0]['id'])
        run('report_not_found', reason=reason)
        return out + ['I could not find that in your documents: ' + reason[0].lower() + reason[1:]]
    if kind in SELECTED:
        arguments = {'project': record['project'], 'family': record['family']}
        if kind != 'latest':
            arguments['date'] = record['date']
        result = run('current_document' if kind == 'latest' else 'effective_document', **arguments)
        if result['outcome'] != record['outcome']:
            raise Reject('selection')
        if result['outcome'] in DECLINES:
            if decline_reason(result, record['project'], record['family'], record.get('date')) != record['reason']:
                raise Reject('selection')
            run('report_not_found', reason=record['reason'])
            return out + [f"{record['reason']} (Documents: {', '.join(record['members'])})"]
        if [result['document']['id']] != record['source_ids']:
            raise Reject('selection')
    sources = record['source_ids']
    if kind in ('named', 'compare', 'draft'):
        order = {key: probe.documents[key]['metadata']['revision'] for key in sources}
        sources = sorted(sources, key=lambda key: (order[key], key))
    for key in sources:
        run('read_document', document_id=key)
    result = None
    for step in item.get('steps') or []:
        if not isinstance(step, dict) or step.get('tool') not in ('calculate', 'shift_date'):
            raise Reject('replay')
        result = run(step['tool'], **{k: v for k, v in step.items() if k != 'tool'})
    if kind == 'computed':
        value = str(result.get('value', result.get('date', '')))
        if not any(documents.normalize(value) in documents.normalize(f) or documents.normalize(f) in
                   documents.normalize(value) for f in record['facts']):
            raise Reject('replay')
    quotes = evidence(world, item, sources, kind == 'computed')
    if not quotes:
        raise Reject('evidence')
    answer = item['answer'].strip()
    if kind == 'nometa_latest':
        answer = f'{answer} {NO_METADATA_NOTE}'
    if run('submit_answer', answer=answer, source_ids=sources, evidence=quotes) != {'submitted': True}:
        raise Reject('evidence')
    return out + [f"{answer} (Sources: {', '.join(sources)})"]


def episode(case, texts, policy):
    """Replay the scripted replies through the real v4 episode loop; the row must pass the route's scorer."""
    from neuroshard.evolution import assistant_general as general
    from neuroshard.evolution.modular_reference_execution import identity

    stream = iter(texts)

    def respond(messages, tools):
        return {'text': next(stream), 'terminated': True, 'executed': True, 'token_ids': [], 'input_token_ids': [],
                'prompt_sha256': identity({'messages': messages, 'tools': tools}), 'seconds': 0.0, 'model': 'script'}

    if general.version_of(policy) != 4:
        raise ValueError('revision demonstrations are workspace-documents/4 conversations')
    try:
        row = general.documents_episode(case, respond, policy)
    except StopIteration:
        return None
    if not row['passed'] or next(stream, None) is not None or any('error' in c['result'] for c in row['calls']):
        return None
    return row


def trajectory(case, policy):
    """System, user, then each scripted reply with the v4 workspace's tool results, as serving renders them."""
    world = documents.Workspace(case['world'], 4)
    messages = [{'role': 'system', 'content': policy['system_instruction']},
                {'role': 'user', 'content': case['turns'][0]['user']}]
    for text in case['replies']:
        messages.append({'role': 'assistant', 'content': text})
        for item in documents.parse_calls(text, 4):
            messages.append({'role': 'tool', 'content': json.dumps(world.execute(item), ensure_ascii=False,
                                                                     sort_keys=True)})
    return {'case_id': case['id'], 'family': case['family'], 'messages': messages,
            'trainable': [m['role'] == 'assistant' for m in messages]}


def fits(case, policy, tokenizer):
    """Every assistant turn's prompt is within the policy's input budget and every reply within its output budget."""
    item, tools = trajectory(case, policy), documents.tools(4)
    generation = policy['generation']
    for index, message in enumerate(item['messages']):
        if message['role'] != 'assistant':
            continue
        prompt = tokenizer.apply_chat_template(item['messages'][:index], tools=tools, add_generation_prompt=True,
                                               tokenize=False)
        if len(tokenizer(prompt, add_special_tokens=False)['input_ids']) > generation['max_input_tokens']:
            return False
        if len(tokenizer(message['content'], add_special_tokens=False)['input_ids']) >= generation['max_new_tokens']:
            return False
    return True


def blind_prompt(world, items):
    rows = []
    for d in world['documents']:
        fields = {k: d[k] for k in revisions.FIELDS if k in d}
        rows.append(f"[{d['id']}] {d['title']}" + (f' {json.dumps(fields, sort_keys=True)}' if fields else '')
                    + f"\n{d['content']}")
    questions = '\n'.join(f'{n + 1}. {item["user"]}' for n, item in enumerate(items))
    return BLIND.format(documents='\n\n'.join(rows), questions=questions)


def parse_json(text, opener='{', closer='}'):
    """The JSON value a reply ends with: its last fenced block if it has one, else from the first opener to the last
    closer, else from the latest opener that parses (a reply may reason in prose before its answer)."""
    fenced = re.findall(r'```(?:json)?\s*(.*?)```', text, re.DOTALL)
    for block in reversed(fenced):
        block = block.strip()
        if block.startswith(opener):
            try:
                return json.loads(block)
            except ValueError:
                pass
    start, end = text.find(opener), text.rfind(closer)
    try:
        return json.loads(text[start:end + 1])
    except ValueError:
        for position in range(len(text) - 1, -1, -1):
            if text[position] == opener:
                try:
                    return json.loads(text[position:end + 1])
                except ValueError:
                    continue
        raise


def agrees(verdict, expected, record):
    """The blind answer reaches the same outcome: the facts (and, for a version question, its documents), a
    decline, or not found."""
    if not isinstance(verdict, dict) or not isinstance(verdict.get('answer'), str):
        return False
    said = verdict['answer'].upper()
    if record['type'] == 'not_found':
        return 'NOT FOUND' in said
    if expected.get('not_found'):
        return said.strip().startswith('DECLINE')
    if 'NOT FOUND' in said or 'DECLINE' in said:
        return False
    if not all(documents.normalize(f) in documents.normalize(verdict['answer']) for f in expected['facts']):
        return False
    sources = verdict.get('sources') if isinstance(verdict.get('sources'), list) else []
    return record['type'] not in FAMILY_TYPES or set(expected['source_ids']) <= set(sources)


def canonical_ids(written, key):
    """Rename document IDs the writer did not make ``doc-`` + 8 hex, everywhere they appear (as G2 did)."""
    ids = [d.get('id') for d in written['documents']]
    if all(isinstance(i, str) and re.fullmatch(r'doc-[0-9a-f]{8}', i) for i in ids) and len(set(ids)) == len(ids):
        return written
    if len(set(ids)) != len(ids) or not all(isinstance(i, str) and i for i in ids):
        raise Reject('ids')
    renamed = {old: 'doc-' + hashlib.sha256(f'{key}:{old}'.encode()).hexdigest()[:8] for old in ids}
    text = json.dumps(written)
    for old in sorted(renamed, key=len, reverse=True):
        text = text.replace(f'"{old}"', f'"{renamed[old]}"')
    return json.loads(text)


def prompts(count, seed):
    """The writing prompts, in order: one in six workspaces has no metadata; the rest draw family shapes."""
    rng = random.Random(seed)
    out = []
    shapes = [name for name, weight in WEIGHTS.items() for _ in range(weight)]
    for index in range(count):
        setting, forms = SETTINGS[index % len(SETTINGS)], FORMS[(index // len(SETTINGS)) % len(FORMS)]
        if index % 6 == 5:
            out.append(('nometa', NOMETA.format(setting=setting, count=rng.randint(6, 10), forms=forms)))
            continue
        chosen = rng.sample(shapes, 2)
        while chosen[0] == chosen[1]:
            chosen = rng.sample(shapes, 2)
        families = '\n'.join(f'- family {n + 1}: {PATTERNS[name]}' for n, name in enumerate(chosen))
        if rng.random() < 0.5:
            families += '\n- put both families in the SAME project (different "family" values)'
        per_family = '; '.join(f'family {n + 1}: {QUESTIONS[name]}' for n, name in enumerate(chosen))
        out.append(('+'.join(chosen), WRITE.format(setting=setting, count=rng.randint(8, 13), forms=forms,
                                                    families=families, questions=12, per_family=per_family)))
    return out


def family_of(record, expected):
    kind = record['type']
    if kind in SELECTED and expected.get('not_found'):
        return f"rev-decline-{record['outcome']}"
    return {'latest': 'rev-latest', 'now': 'rev-now', 'as_of': 'rev-as-of', 'named': 'rev-named',
            'compare': 'rev-compare', 'draft': 'rev-draft', 'lookup': 'rev-lookup', 'computed': 'rev-computed',
            'not_found': 'rev-not-found', 'nometa_latest': 'rev-nometa-latest'}[kind]


def build(count, seed, cache, policy, tokenizer=None, workers=8, progress=print, bedrock=None):
    """``count`` workspaces of checked ``/4`` demonstrations. Every Bedrock reply is cached in ``cache`` (keyed by
    prompt), so a rebuild from the cache is identical and makes no call."""
    lock = threading.Lock()
    cache = Path(cache)
    store = json.loads(cache.read_text()) if cache.exists() else {}
    usage = {'calls': 0, 'inputTokens': 0, 'outputTokens': 0}
    state = {'bedrock': bedrock}
    written_prompts = prompts(count, seed)

    def ask(key, text, tokens, temperature):
        with lock:
            if key in store:
                return store[key]
            if state['bedrock'] is None:
                from neuroshard.evolution import assistant_phrasings as phrasings
                state['bedrock'] = phrasings.client()
        value, used = converse(text, state['bedrock'], tokens, temperature)
        with lock:
            store[key] = value
            usage['calls'] += 1
            for name in ('inputTokens', 'outputTokens'):
                usage[name] += used.get(name, 0)
        return value

    def workspace(index):
        shape, prompt = written_prompts[index]
        key = hashlib.sha256(f'revision-train:{seed}:{index}:{prompt}'.encode()).hexdigest()
        rejected = {}
        try:
            reply = ask(key, prompt, 12000, 1.0)
            written = canonical_ids(parse_json(reply), key)
            world = public_world(written)
            documents.Workspace(world, 4)
        except Reject as error:
            return [], {str(error): 1}, shape
        except (ValueError, KeyError, TypeError, AttributeError):
            return [], {'parse': 1}, shape
        nometa = shape == 'nometa'
        if nometa and any(k in d for d in world['documents'] for k in revisions.FIELDS):
            return [], {'parse': 1}, shape
        candidates = []
        for number, item in enumerate(written.get('questions') or []):
            try:
                if not isinstance(item, dict) or not isinstance(item.get('user'), str) or not item['user'].strip():
                    raise Reject('malformed')
                expected, record = gold(world, item, nometa)
                texts = replies(world, item, record)
                case = {'id': f'revtrain-{index:03d}-{number}', 'split': SPLIT, 'world': world,
                        'family': family_of(record, expected), 'gold': record,
                        'turns': [{'user': item['user'].strip(), 'expected': expected}], 'replies': texts}
                if episode(case, texts, policy) is None:
                    raise Reject('scorer')
                if tokenizer is not None and not fits(case, policy, tokenizer):
                    raise Reject('budget')
                candidates.append((item, expected, record, case))
            except Reject as error:
                reason = str(error) if str(error) in REASONS else 'malformed'
                rejected[reason] = rejected.get(reason, 0) + 1
            except (ValueError, KeyError, TypeError, AttributeError, IndexError):
                rejected['malformed'] = rejected.get('malformed', 0) + 1
        kept = []
        if candidates:
            items = [item for item, _, _, _ in candidates]
            text = ask(f'blind:{key}:{len(items)}:' + hashlib.sha256(json.dumps([i['user'] for i in items])
                                                                      .encode()).hexdigest(),
                       blind_prompt(world, items), 4000, 0.0)
            try:
                verdicts = parse_json(text, '[', ']')
            except ValueError:
                verdicts = []
            for position, (item, expected, record, case) in enumerate(candidates):
                verdict = verdicts[position] if isinstance(verdicts, list) and position < len(verdicts) else None
                if agrees(verdict, expected, record):
                    kept.append({**case, 'blind': verdict})
                else:
                    rejected['blind'] = rejected.get('blind', 0) + 1
        return kept, rejected, shape

    cases, rejected, shapes = [], {}, {}
    with ThreadPoolExecutor(workers) as pool:
        for index, (kept, why, shape) in enumerate(pool.map(workspace, range(count))):
            cases += kept
            for name in shape.split('+'):
                shapes[name] = shapes.get(name, 0) + 1
            for k, v in why.items():
                rejected[k] = rejected.get(k, 0) + v
            if index % 10 == 9:
                with lock:
                    cache.write_text(json.dumps(store))
                progress(f'{index + 1}/{count} workspaces: {len(cases)} kept, rejected {rejected}')
    with lock:
        cache.write_text(json.dumps(store))
    return {'format': FORMAT, 'model': AUTHOR, 'seed': seed, 'workspaces': count, 'shapes': shapes,
            'rejected': rejected, 'usage_new_calls': usage, 'cases': cases,
            'sha256': hashlib.sha256(json.dumps(cases, sort_keys=True).encode()).hexdigest()}


def rerender(case, policy):
    """A G2 realistic corpus conversation (``/2``) as a ``/4`` one: same world, search and reads, the answer
    submitted with mechanically chosen exact quotes from its cited documents. Returns None if no quote states
    every fact (computed answers, whose result is in no document, are not re-rendered)."""
    if case['family'] not in ('real-lookup', 'real-not-found'):
        return None
    expected = case['turns'][0]['expected']
    replies_ = list(case['replies'])
    world = case['world']
    if not expected.get('not_found'):
        submit = documents.parse_calls(replies_[-2], 2)
        if len(submit) != 1 or submit[0]['name'] != 'submit_answer':
            return None
        arguments = submit[0]['arguments']
        item = {'facts': expected['facts'], 'quotes': []}
        quotes = evidence(world, item, arguments['source_ids'], False)
        if not quotes:
            return None
        replies_[-2] = call('submit_answer', answer=arguments['answer'], source_ids=arguments['source_ids'],
                            evidence=quotes)
    out = {**case, 'id': case['id'].replace('realtrain-', 'realtrain4-'), 'split': 'realistic-train-v4',
           'replies': replies_, 'rerendered_from': case['id']}
    return out if episode(out, replies_, policy) is not None else None


def rerender_corpus(corpus, policy, counts, seed=0, tokenizer=None):
    """A seeded sample of a G2 realistic corpus re-rendered as ``/4`` (``counts``: family -> how many), with the
    counts of cases that could not be re-rendered or exceed the token budget."""
    rng = random.Random(seed)
    by_family, dropped = {}, {}
    for case in corpus['cases']:
        out = rerender(case, policy)
        if out is None:
            dropped['no-quote-or-computed'] = dropped.get('no-quote-or-computed', 0) + 1
        elif tokenizer is not None and not fits(out, policy, tokenizer):
            dropped['budget'] = dropped.get('budget', 0) + 1
        else:
            by_family.setdefault(case['family'], []).append(out)
    cases = []
    for family in sorted(counts):
        pool = by_family.get(family, [])
        cases += rng.sample(pool, min(counts[family], len(pool)))
    cases.sort(key=lambda c: c['id'])
    return {'format': FORMAT, 'source': corpus.get('sha256'), 'source_model': corpus.get('model'), 'seed': seed,
            'counts': counts, 'available': {k: len(v) for k, v in sorted(by_family.items())}, 'dropped': dropped,
            'cases': cases, 'sha256': hashlib.sha256(json.dumps(cases, sort_keys=True).encode()).hexdigest()}


def overlap(training, evaluation, n=8):
    """Shared word n-grams between training and evaluation document texts and requests, as
    ``realistic_training.overlap`` counts them; also returns a sample of the shared grams for inspection."""
    def grams(cases):
        out = set()
        for case in cases:
            texts = [d['content'] for d in case['world']['documents']] + [case['turns'][0]['user']]
            for text in texts:
                words = documents.words(text)
                out |= {' '.join(words[i:i + n]) for i in range(len(words) - n + 1)}
        return out
    a, b = grams(training), grams(evaluation)
    shared = sorted(a & b)
    return {'training_grams': len(a), 'evaluation_grams': len(b), 'shared': len(shared), 'sample': shared[:20]}


def verify(corpus, policy, encode=None, sequences=None):
    """An owner's check of a revision corpus, with no model. Returns (ok, problems).

    For every case: the corpus hashes to what it names; the conversation replays through the v4 workspace and
    passes the v4 scorer with no rejected tool call; the gold is recomputed (a latest/effective gold equals the
    selection over the world's own fields, and a decline's reason is the one the tool's result produces; a named,
    draft or comparison gold is exactly the named family members; every fact of a lookup is in a cited document's
    text, and a computed one is produced by a tool call in the conversation); and, if ``encode`` and
    ``sequences`` are given, each conversation re-encodes to exactly its training sequence."""
    problems = []
    cases = corpus.get('cases', [])
    if hashlib.sha256(json.dumps(cases, sort_keys=True).encode()).hexdigest() != corpus.get('sha256'):
        problems.append('corpus hash')
    if len({c['id'] for c in cases}) != len(cases):
        problems.append('duplicate ids')
    if sequences is not None and len(sequences) != len(cases):
        problems.append('sequence count')
    for index, case in enumerate(cases):
        try:
            if case['split'] not in (SPLIT, 'realistic-train-v4'):
                raise Reject('split')
            if episode(case, case['replies'], policy) is None:
                raise Reject('replay')
            expected = case['turns'][0]['expected']
            if case['split'] == SPLIT:
                record = case['gold']
                probe = documents.Workspace(case['world'], 4)
                if record['type'] in SELECTED:
                    arguments = {'project': record['project'], 'family': record['family']}
                    if record['type'] != 'latest':
                        arguments['date'] = record['date']
                    result = probe.execute({'name': 'current_document' if record['type'] == 'latest'
                                            else 'effective_document', 'arguments': arguments})
                    if result.get('outcome') != record['outcome']:
                        raise Reject('selection')
                    if result['outcome'] in DECLINES:
                        if not expected.get('not_found') or decline_reason(
                                result, record['project'], record['family'], record.get('date')) != record['reason']:
                            raise Reject('decline')
                    elif expected.get('source_ids') != [result['document']['id']]:
                        raise Reject('selection')
                elif record['type'] in ('named', 'compare', 'draft'):
                    group = family_members(probe, record['project'], record['family'])
                    named = sorted(m['id'] for m in group if m['metadata'].get('revision') in record['revisions'])
                    if (named != expected['source_ids'] or len(named) != len(record['revisions'])
                            or (record['type'] == 'draft'
                                and any(probe.documents[i]['metadata'].get('status') != 'draft' for i in named))):
                        raise Reject('named')
            if not expected.get('not_found'):
                by = {d['id']: d for d in case['world']['documents']}
                cited = ' '.join(by[i]['content'] for i in expected['source_ids'])
                computed = any('"calculate"' in r or '"shift_date"' in r for r in case['replies'])
                if not computed and not all(documents.normalize(f) in documents.normalize(cited)
                                            for f in expected['facts']):
                    raise Reject('facts')
                if case['split'] == SPLIT:
                    submitted = documents.parse_calls(case['replies'][-2], 4)[0]['arguments']
                    answer = submitted['answer']
                    if case['gold']['type'] == 'nometa_latest':
                        if not answer.endswith(' ' + NO_METADATA_NOTE):
                            raise Reject('claims')
                        answer = answer[:-len(NO_METADATA_NOTE) - 1]
                    if not claims_ok(answer, [probe.documents[i] for i in expected['source_ids']],
                                     case['gold']['type']):
                        raise Reject('claims')
            if encode is not None and sequences is not None and encode(trajectory(case, policy)) != sequences[index]:
                raise Reject('encoding')
        except (Reject, ValueError, KeyError, TypeError, StopIteration) as error:
            problems.append(f"{case.get('id')}: {error}")
    return not problems, problems


def template_replay(policy, per_family, randomness='f0' * 32):
    """The template families' solver demonstrations, as G2's replay drew them (same cases, ``'replay'`` purpose,
    same randomness), written with the solver's evidence replies (``/3`` texts) and replayed as ``/4``
    conversations under ``policy``. Template worlds carry no revision fields, so under ``/4`` every document is
    ``no_metadata`` and each demonstration's tool calls behave exactly as under ``/3``; only the system prompt,
    the tool list and the metadata keys in results change. Each must pass the v4 scorer with no rejected call."""
    from neuroshard.evolution import assistant_documents_demonstration as demo
    from neuroshard.evolution import growth_planner

    families = sorted(growth_planner.sealed.templates('documents'))
    out = []
    for case in growth_planner.demonstrations(families, randomness, per_family, 'replay'):
        texts = demo.texts(case, 3, policy.get('search_query', 'message'))
        row = episode(case, texts, policy)
        if row is None:
            raise ValueError(f"the template demonstration of {case['id']} does not pass under /4")
        out.append(demo.trajectory(case, row, policy))
    return out


def main(argv=None):
    """Build the corpus from Bedrock (or entirely from the cache): python -m neuroshard.evolution.revision_training
    OUT --cache CACHE [--count N] [--seed S] [--tokenizer DIR]."""
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('out', type=Path)
    parser.add_argument('--cache', type=Path, required=True)
    parser.add_argument('--count', type=int, default=240)
    parser.add_argument('--seed', type=int, default=1026)
    parser.add_argument('--workers', type=int, default=12)
    parser.add_argument('--tokenizer', type=Path, help='the parent tokenizer, to drop items over the token budget')
    parser.add_argument('--policy', type=Path, default=Path('config/experiments/assistant-general-policy-documents7.json'))
    args = parser.parse_args(argv)
    tokenizer = None
    if args.tokenizer:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, local_files_only=True)
    policy = json.loads(args.policy.read_text())
    corpus = build(args.count, args.seed, args.cache, policy, tokenizer, args.workers, lambda m: print(m, flush=True))
    args.out.write_text(json.dumps(corpus, indent=1, sort_keys=True))
    print(json.dumps({k: v for k, v in corpus.items() if k != 'cases'}), flush=True)


if __name__ == '__main__':
    main()
