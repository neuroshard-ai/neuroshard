"""The protected, human-written documents evaluation set: case format, mechanical checks and the frozen scorer.

DESIGN_LATEST_REVISION.md, acceptance test 2. People write the set, and the authors of the training corpus never
see it. This module never holds a case: the set lives outside the repository, and only ``freeze``'s manifest is
committed, before the set is first run.

**What is scored is the reply the user sees** (the final text), under every condition alike. A structured
``submit_answer``, when a condition has one, must agree with it; protocol correctness (the route's own scorer) is
reported separately, never in place of this score.

**The reported result is blinded human grading of every reply, in every condition** (``grading_sheet``,
``ingest_grades``, ``accuracy``). The machine score (``score``) is a separate, reported measure: typed values
(numbers with their sign, dates, words at word boundaries), exact citation IDs, negation and competing values. It
is not a judgment of meaning: a reply can state the gold and then contradict it in words, and only a person can
tell. Graders see only the case and the reply, never the condition or the machine verdict; the agreement between
the two scores is reported.

**Unresolved golds are excluded, by ID,** from grading and from every condition's numerator and denominator.

**The selection code never decides a gold.** ``selector_report`` compares each latest-approved or effective gold
with what the deterministic ``/4`` selection chooses. A disagreement must be adjudicated by the case's checker and
recorded in the case; it is never resolved by editing the gold to match the code, and a confirmed code error is
counted against the code.
"""
import hashlib
import json
import random
import re
import subprocess
from pathlib import Path

from neuroshard.evolution import assistant_documents as documents

FORMAT = 'neuroshard-protected-documents-set/2'
KINDS = ('answer', 'not_found', 'decline')
SELECTIONS = ('latest_approved', 'effective', 'named', 'other')
ADJUDICATIONS = ('gold_correct', 'unresolved')  # a gold found wrong is rewritten from the documents, then rechecked
CATEGORIES = ('latest_approved_with_newer_draft', 'latest_approved_newest', 'single_revision', 'unrelated_families',
              'ambiguous_versions', 'no_approved_revision', 'incomplete_metadata', 'misleading_text', 'named_revision',
              'as_of_date', 'same_value_old_and_current', 'shared_titles', 'not_found')
MINIMUM_PER_CATEGORY = 8
MINIMUM_WORKSPACES = 30
MINIMUM_WORKSPACES_PER_CATEGORY = 4
ROOT = Path(__file__).resolve().parents[3]

ID = re.compile(r'\Adoc-[0-9a-f]{8,}\Z')
CITED = re.compile(r'(?<![0-9A-Za-z_-])doc-[0-9a-f]{8,}(?![0-9A-Za-z_])')
CITED_LIKE = re.compile(r'(?<![0-9A-Za-z_-])doc-[0-9A-Za-z]+')
ISO = re.compile(r'(?<!\d)(\d{4})-(\d{2})-(\d{2})(?!\d)')
MONTHS = {m: i for i, m in enumerate('january february march april may june july august september october '
                                     'november december'.split(), 1)}
NAMED_DATE = re.compile(r'\b(\d{1,2})\s+(' + '|'.join(MONTHS) + r'),?\s+(\d{4})\b|\b(' + '|'.join(MONTHS)
                        + r')\s+(\d{1,2}),?\s+(\d{4})\b', re.I)
NUMBER = re.compile(r'(?<![\w.,\-\u2212])[-\u2212]?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?(?![\w]|[.,]\d)')
WORDS = re.compile(r"[a-z0-9]+(?:'[a-z]+)?")
NEGATIONS = {'not', 'no', 'never', "isn't", "wasn't", "aren't", "weren't", "doesn't", "didn't", "isnt", "wasnt",
             'neither', 'nor', 'instead', 'rather', 'except', 'without', 'unlike', 'formerly', 'previously'}
REFUSALS = ('could not find', "couldn't find", 'cannot find', "can't find", 'not in your documents',
            'do not contain', "don't contain", 'does not contain', "doesn't contain", 'does not say',
            "doesn't say", 'do not say', "don't say", 'no document', 'not mentioned', 'not stated')
UNCERTAIN = ('which one', 'which version', 'which of', 'not sure which', 'cannot tell', "can't tell",
             'cannot determine', "can't determine", 'unclear', 'ambiguous', 'conflict', 'two approved',
             'no approved', 'none is approved', 'none of them is approved', 'not approved', 'incomplete',
             'missing', 'cannot say which', "can't say which", 'please confirm', 'could you confirm')


class Invalid(ValueError):
    """A written case is malformed, or its gold is inconsistent with its own documents."""


def kind_of(value):
    if ISO.fullmatch(value.strip()) or NAMED_DATE.fullmatch(value.strip()):
        return 'date'
    if NUMBER.fullmatch(value.strip().replace('£', '').replace('$', '').replace('€', '').rstrip('%')):
        return 'number'
    return 'text'


def _number(token):
    """A number's value as text, sign kept: -1,200 and 1,200 differ; 1,200.00 and 1200 do not."""
    token = token.replace('\u2212', '-').replace(',', '')
    sign, digits = ('-', token[1:]) if token.startswith('-') else ('', token)
    if '.' in digits:
        digits = digits.rstrip('0').rstrip('.')
    return sign + (digits.lstrip('0') or '0')


def _iso(match):
    groups = match.groups()
    if len(groups) == 3:
        return f'{groups[0]}-{groups[1]}-{groups[2]}'
    day, month, year, month2, day2, year2 = groups
    if day:
        return f'{year}-{MONTHS[month.lower()]:02d}-{int(day):02d}'
    return f'{year2}-{MONTHS[month2.lower()]:02d}-{int(day2):02d}'


def occurrences(text, value):
    """(start, end) spans where the typed ``value`` occurs in ``text`` as a whole value, never inside another."""
    text = CITED_LIKE.sub(lambda m: ' ' * len(m.group()), text)  # IDs are not values
    kind = kind_of(value)
    if kind == 'date':
        want = _iso(ISO.fullmatch(value.strip()) or NAMED_DATE.fullmatch(value.strip()))
        spans = [(m.start(), m.end()) for m in ISO.finditer(text) if _iso(m) == want]
        return spans + [(m.start(), m.end()) for m in NAMED_DATE.finditer(text) if _iso(m) == want]
    masked = ISO.sub(lambda m: ' ' * len(m.group()), NAMED_DATE.sub(lambda m: ' ' * len(m.group()), text))
    if kind == 'number':
        want = _number(value.strip().replace('£', '').replace('$', '').replace('€', '').rstrip('%'))
        return [(m.start(), m.end()) for m in NUMBER.finditer(masked) if _number(m.group()) == want]
    words = WORDS.findall(value.lower())
    if not words:
        return []
    pattern = re.compile(r'(?<![a-z0-9])' + r'\W+'.join(map(re.escape, words)) + r'(?![a-z0-9])', re.I)
    return [(m.start(), m.end()) for m in pattern.finditer(text)]


def negated(text, start, window=4):
    """True if one of the ``window`` words before ``start`` negates what follows (within its sentence)."""
    sentence = re.split(r'[.!?;\n]', text[:start])[-1]
    return bool(NEGATIONS & set(WORDS.findall(sentence.lower())[-window:]))


def asserted(text, value):
    """True if ``value`` occurs in ``text`` at least once, as a whole value, not negated just before it."""
    return any(not negated(text, start) for start, _ in occurrences(text, value))


def mentioned(text, value):
    return bool(occurrences(text, value))


def citations(text, case_ids):
    """(cited IDs, malformed or unknown citations): complete IDs only, each checked against the case."""
    cited = set(CITED.findall(text))
    malformed = {m.group() for m in CITED_LIKE.finditer(text)} - cited
    return cited & set(case_ids), malformed | (cited - set(case_ids))


def has_phrase(text, phrases):
    low = ' '.join(WORDS.findall(text.lower().replace('\u2019', "'")))
    return any(' '.join(WORDS.findall(p)) in low for p in phrases)


def workspace_id(case):
    """Content identity of a case's workspace: its documents, order-independent, ignoring nothing else."""
    body = sorted(json.dumps(d, sort_keys=True, ensure_ascii=False) for d in case['documents'])
    return hashlib.sha256('\n'.join(body).encode()).hexdigest()


def question_id(case):
    return hashlib.sha256(' '.join(WORDS.findall(case['question'].lower())).encode()).hexdigest()


def family(case, project, name):
    return [d for d in case['documents'] if d.get('project') == project and d.get('family') == name]


def check_case(case):
    """Raise ``Invalid`` unless the case is well formed and its gold agrees with its own documents. Returns its ID.

    Mechanical only: IDs, two different people, every gold fact in a cited document's text as a whole value, and
    the structure its category claims (``structure``). It never consults the selection code (``selector_report``)."""
    key = case.get('id')
    if not isinstance(key, str) or not key:
        raise Invalid('a case needs an id')
    where = f'case {key}'
    if case.get('category') not in CATEGORIES:
        raise Invalid(f'{where}: category must be one of {", ".join(CATEGORIES)}')
    if not isinstance(case.get('question'), str) or not case['question'].strip():
        raise Invalid(f'{where}: a question is required')
    if not case.get('written_by') or not case.get('checked_by') or case['written_by'] == case['checked_by']:
        raise Invalid(f'{where}: written_by and checked_by must name two different people')
    docs = case.get('documents')
    if not isinstance(docs, list) or not docs:
        raise Invalid(f'{where}: documents are required')
    ids = [d.get('id') for d in docs]
    if any(not isinstance(i, str) or not ID.match(i) for i in ids) or len(set(ids)) != len(ids):
        raise Invalid(f'{where}: document ids must be unique and look like doc-<at least 8 lowercase hex digits>')
    by_id = {d['id']: d for d in docs}
    for d in docs:
        if not isinstance(d.get('title'), str) or not isinstance(d.get('content'), str) or not d['content'].strip():
            raise Invalid(f'{where}: {d["id"]} needs a title and content')
    expected = case.get('expected') or {}
    kind = expected.get('kind')
    if kind not in KINDS:
        raise Invalid(f'{where}: expected.kind must be one of {", ".join(KINDS)}')
    if kind == 'answer':
        facts, sources = expected.get('facts'), expected.get('source_ids')
        if not facts or not all(isinstance(f, str) and f.strip() for f in facts):
            raise Invalid(f'{where}: an answer needs facts')
        if not sources or any(s not in by_id for s in sources) or len(set(sources)) != len(sources):
            raise Invalid(f'{where}: source_ids must name documents of this case, once each')
        for fact in facts:
            if not any(mentioned(by_id[s]['content'], fact) for s in sources):
                raise Invalid(f'{where}: fact {fact!r} is not a whole value in any cited document')
        if expected.get('selection') not in SELECTIONS:
            raise Invalid(f'{where}: selection must be one of {", ".join(SELECTIONS)}')
        if expected['selection'] in ('latest_approved', 'effective'):
            if not (isinstance(expected.get('project'), str) and isinstance(expected.get('family'), str)):
                raise Invalid(f'{where}: a selected gold names its project and family')
            if expected['selection'] == 'effective' and not ISO.fullmatch(str(expected.get('date', ''))):
                raise Invalid(f'{where}: an effective gold names its date as YYYY-MM-DD')
    elif kind == 'decline':
        cite = expected.get('must_cite') or []
        if not cite or any(c not in by_id for c in cite):
            raise Invalid(f'{where}: a decline names, in must_cite, the documents the reply must point to')
    for value in expected.get('must_not_state') or []:
        if not isinstance(value, str) or not value.strip():
            raise Invalid(f'{where}: must_not_state holds values')
    problem = structure(case)
    if problem:
        raise Invalid(f'{where}: {case["category"]}: {problem}')
    return key


def structure(case):
    """Why the case does not have the structure its category claims, or None. Machine-verified coverage."""
    expected, docs, category = case['expected'], case['documents'], case['category']
    kind, selection = expected['kind'], expected.get('selection')
    families = {}
    for d in docs:
        if d.get('project') and d.get('family'):
            families.setdefault((d['project'], d['family']), []).append(d)
    gold = family(case, expected.get('project'), expected.get('family')) if expected.get('family') else []
    revisions = [d for d in gold if type(d.get('revision')) is int]
    approved = [d for d in revisions if d.get('status') == 'approved']
    if category in ('latest_approved_with_newer_draft', 'latest_approved_newest', 'single_revision'):
        if kind != 'answer' or selection != 'latest_approved':
            return 'needs an answer with latest_approved selection'
        if not approved:
            return 'its family has no approved revision'
        top = max(d['revision'] for d in revisions)
        top_approved = max(d['revision'] for d in approved)
        if category == 'latest_approved_with_newer_draft' and not any(
                d['revision'] > top_approved and d.get('status') == 'draft' for d in revisions):
            return 'needs a draft newer than the latest approved revision'
        if category == 'latest_approved_newest' and (top != top_approved or len(gold) < 2):
            return 'needs the newest of two or more revisions to be the approved one'
        if category == 'single_revision' and len(gold) != 1:
            return 'its family must have exactly one document'
    elif category == 'unrelated_families':
        projects = {}
        for project, name in families:
            projects.setdefault(project, set()).add(name)
        if not any(len(names) >= 2 for names in projects.values()):
            return 'needs two or more families in one project'
    elif category == 'ambiguous_versions':
        if kind != 'decline':
            return 'is a decline'
        if not any(len([d for d in f if d.get('status') == 'approved' and d.get('revision') == r]) >= 2
                   for f in families.values() for r in {d.get('revision') for d in f}):
            return 'needs two approved documents at the same revision of one family'
    elif category == 'no_approved_revision':
        if kind != 'decline':
            return 'is a decline'
        if not any(f and all(d.get('status') in ('draft', 'superseded') for d in f) for f in families.values()):
            return 'needs a family in which nothing is approved'
    elif category == 'incomplete_metadata':
        if kind != 'decline':
            return 'is a decline'
        if not any(any(d.get('status') is None or type(d.get('revision')) is not int for d in f)
                   for f in families.values()):
            return 'needs a family member without a status or an integer revision'
    elif category == 'misleading_text':
        if not any(d.get('status') in ('draft', 'superseded') and
                   has_phrase(d['content'], ('approved', 'final version', 'signed off', 'agreed', 'final'))
                   for d in docs):
            return "needs a draft or superseded document whose text claims approval or finality"
    elif category == 'named_revision':
        if kind != 'answer' or selection != 'named':
            return 'needs an answer with named selection'
    elif category == 'as_of_date':
        if selection != 'effective' and kind != 'decline':
            return 'needs an answer with effective selection, or a decline'
        if not any(d.get('effective_from') for d in docs):
            return 'needs effective dates'
    elif category == 'same_value_old_and_current':
        if kind != 'answer':
            return 'is an answer'
        sources = set(expected['source_ids'])
        others = [d for d in gold if d['id'] not in sources] if gold else [d for d in docs if d['id'] not in sources]
        if not any(all(mentioned(d['content'], f) for f in expected['facts']) for d in others):
            return 'needs an older or other version that holds the same value'
    elif category == 'shared_titles':
        titles = {}
        for d in docs:
            titles.setdefault(d['title'].strip().lower(), set()).add((d.get('project'), d.get('family')))
        if not any(len(keys) >= 2 for keys in titles.values()):
            return 'needs one title shared by documents of different families'
    elif category == 'not_found':
        if kind != 'not_found':
            return 'is a not_found case'
    return None


def check_set(cases):
    """Check every case, then coverage. Returns the machine-verified coverage report.

    Coverage counts distinct workspaces and questions, not labels: at least ``MINIMUM_PER_CATEGORY`` cases and
    ``MINIMUM_WORKSPACES_PER_CATEGORY`` distinct workspaces per category, ``MINIMUM_WORKSPACES`` overall, and no
    question asked twice of the same workspace. Semantic diversity is the checkers' attestation, reported as such."""
    seen, counts, spaces, per_category, pairs = set(), {}, set(), {}, set()
    for case in cases:
        key = check_case(case)
        if key in seen:
            raise Invalid(f'duplicate case id {key}')
        seen.add(key)
        space, question = workspace_id(case), question_id(case)
        if (space, question) in pairs:
            raise Invalid(f'case {key}: the same question of the same workspace appears twice')
        pairs.add((space, question))
        counts[case['category']] = counts.get(case['category'], 0) + 1
        spaces.add(space)
        per_category.setdefault(case['category'], set()).add(space)
    short = {c: counts.get(c, 0) for c in CATEGORIES if counts.get(c, 0) < MINIMUM_PER_CATEGORY}
    thin = {c: len(per_category.get(c, ())) for c in CATEGORIES
            if len(per_category.get(c, ())) < MINIMUM_WORKSPACES_PER_CATEGORY}
    if short:
        raise Invalid(f'too few cases (minimum {MINIMUM_PER_CATEGORY} per category): {short}')
    if thin:
        raise Invalid(f'too few distinct workspaces (minimum {MINIMUM_WORKSPACES_PER_CATEGORY} per category): {thin}')
    if len(spaces) < MINIMUM_WORKSPACES:
        raise Invalid(f'{len(spaces)} distinct workspaces; the set needs at least {MINIMUM_WORKSPACES}')
    return {'cases': len(cases), 'categories': counts, 'workspaces': len(spaces),
            'workspaces_per_category': {c: len(v) for c, v in sorted(per_category.items())},
            'machine_verified': ['ids', 'two people', 'gold facts in cited documents', 'category structure',
                                 'distinct workspaces', 'no repeated question per workspace'],
            'human_attested': ['realistic phrasing', 'semantic diversity', 'gold correctness (checker)']}


def selected(docs, expected):
    """(document ID or None, outcome) that the deterministic ``/4`` selection chooses for a gold."""
    from neuroshard.assistant import interfaces

    workspace = documents.Workspace(interfaces.documents_world({'documents': [dict(d) for d in docs]}, 4), 4)
    arguments = {'project': expected.get('project'), 'family': expected.get('family')}
    if expected['selection'] == 'effective':
        result = workspace.execute({'name': 'effective_document', 'arguments': {**arguments, 'date': expected['date']}})
    else:
        result = workspace.execute({'name': 'current_document', 'arguments': arguments})
    return (result.get('document') or {}).get('id'), result.get('outcome')


def selector_report(cases):
    """Where the selection code disagrees with a human gold, and how each disagreement was adjudicated.

    A disagreement never changes the gold. The case's checker, a person other than its writer, records
    ``adjudication: {"by", "decision": "gold_correct" | "unresolved", "note"}``. ``gold_correct`` counts as a
    confirmed selection-code error. A gold the checker finds wrong is rewritten from the documents and the
    specification, never from the code's output, and rechecked; its history is kept outside the set."""
    rows = []
    for case in cases:
        expected = case['expected']
        if expected['kind'] != 'answer' or expected.get('selection') not in ('latest_approved', 'effective'):
            continue
        chosen, outcome = selected(case['documents'], expected)
        if [chosen] == list(expected['source_ids']):
            continue
        decision = case.get('adjudication') or {}
        valid = decision.get('decision') in ADJUDICATIONS and decision.get('by') == case['checked_by']
        rows.append({'case': case['id'], 'gold': expected['source_ids'], 'selected': chosen, 'outcome': outcome,
                     'adjudication': decision.get('decision') if valid else None})
    return {'disagreements': rows,
            'confirmed_code_errors': sorted(r['case'] for r in rows if r['adjudication'] == 'gold_correct'),
            'unresolved': sorted(r['case'] for r in rows if r['adjudication'] == 'unresolved'),
            'not_adjudicated': sorted(r['case'] for r in rows if r['adjudication'] is None)}


def score(case, final_text, terminated=True, submitted=None, reported=None):
    """The machine score of one delivered reply: {'verdict': 'pass' | 'fail', 'reasons', 'contested'}.

    ``final_text`` is what the user sees. ``submitted`` is a structured answer ({'answer', 'source_ids'}) if the
    condition made one, ``reported`` a not-found report. A contested pass (a value negated somewhere, or a
    competing value from another version mentioned) always goes to the blinded human audit."""
    expected, ids = case['expected'], [d['id'] for d in case['documents']]
    reasons, contested = [], []
    text = final_text if isinstance(final_text, str) else ''
    if not terminated or not text.strip():
        return {'verdict': 'fail', 'reasons': ['no completed reply'], 'contested': False}
    cited, bad = citations(text, ids)
    if bad:
        reasons.append('cites IDs that are not documents of this case: ' + ', '.join(sorted(bad)))
    forbidden = [v for v in expected.get('must_not_state') or [] if asserted(text, v)]
    if forbidden:
        reasons.append('states ' + ', '.join(forbidden))
    if expected['kind'] == 'answer':
        if reported is not None:
            reasons.append('reported not found')
        for fact in expected['facts']:
            if not asserted(text, fact):
                reasons.append(f'does not state {fact}' + (' (only negated)' if mentioned(text, fact) else ''))
            elif any(negated(text, start) for start, _ in occurrences(text, fact)):
                contested.append(f'{fact} also appears negated')
        if cited != set(expected['source_ids']):
            reasons.append(f'cites {sorted(cited)}, not exactly {sorted(expected["source_ids"])}')
        sources = {d['id'] for d in case['documents'] if d['id'] in expected['source_ids']}
        rivals = set()
        for d in case['documents']:
            if d['id'] in sources:
                continue
            for value in set(NUMBER.findall(d['content'])) | {m.group() for m in ISO.finditer(d['content'])}:
                if not any(mentioned(next(x for x in case['documents'] if x['id'] == s)['content'], value)
                           for s in sources) and mentioned(text, value):
                    rivals.add(value)
        if rivals:
            contested.append('mentions values from other documents: ' + ', '.join(sorted(rivals)))
        if submitted is not None:
            if set(submitted.get('source_ids') or ()) != cited:
                reasons.append('the submitted answer cites other documents than the reply')
            if not all(asserted(submitted.get('answer', ''), f) for f in expected['facts']):
                reasons.append('the submitted answer does not state the gold')
    else:
        if submitted is not None:
            reasons.append('submitted an answer')
        values = {v for d in case['documents'] for v in NUMBER.findall(d['content'])} | \
            {m.group() for d in case['documents'] for m in ISO.finditer(d['content'])}
        stated = sorted(v for v in values if asserted(text, v))
        if stated:
            # A refusal that then states a value from the documents is an answer; graders decide its merit.
            contested.append('states values from the documents: ' + ', '.join(stated))
        if expected['kind'] == 'not_found':
            if not has_phrase(text, REFUSALS):
                reasons.append('does not say the documents do not contain it')
        else:
            missing = set(expected['must_cite']) - cited
            if missing:
                reasons.append('does not name ' + ', '.join(sorted(missing)))
            if not has_phrase(text, UNCERTAIN + REFUSALS):
                reasons.append('does not say why no document can be chosen')
    return {'verdict': 'fail' if reasons else 'pass', 'reasons': reasons, 'contested': bool(contested) and not reasons,
            'notes': contested}


GRADES = ('correct', 'incorrect')
ROLES = ('served', 'served-v4', 'candidate-v4')  # the design's three conditions, in order
ROOT_HEX = re.compile(r'\A[0-9a-f]{64}\Z')


def grading_sheet(results, seed, excluded=()):
    """The blinded human grading sheet: **every** reply of every condition, shuffled, condition hidden.

    ``results`` are dicts with 'condition', 'case', 'final_text' and 'machine' (``score``'s result). Cases in
    ``excluded`` (unresolved golds, from the manifest) are left out of grading and of every accuracy. Returns
    (sheet, key): the sheet holds only an opaque item ID, the case ID and the reply; the key, kept from graders,
    maps items back to their condition and machine verdict. A grader judges each reply ``correct`` or
    ``incorrect`` against the case's question, documents and gold; that grade is the reported result."""
    rng = random.Random(seed)
    rows = [r for r in results if r['case'] not in set(excluded)]
    order = list(range(len(rows)))
    rng.shuffle(order)
    sheet, key = [], {}
    for position, index in enumerate(order):
        row = rows[index]
        item = hashlib.sha256(f'{seed}:{position}:{row["condition"]}:{row["case"]}'.encode()).hexdigest()[:12]
        sheet.append({'item': item, 'case': row['case'], 'reply': row['final_text']})
        key[item] = {'condition': row['condition'], 'case': row['case'], 'machine': row['machine']['verdict']}
    return sheet, key


def ingest_grades(sheet, key, grades):
    """Check a returned grading is complete, then join it to the key: one row per reply with both scores.

    ``grades`` is a list of {'item', 'grade', 'grader'}. Every sheet item must be graded exactly once with one of
    ``GRADES``; no unknown item may appear. Refuses anything else, so no reply is silently left ungraded."""
    items = {row['item'] for row in sheet}
    seen = {}
    for grade in grades:
        item = grade.get('item')
        if item not in items:
            raise Invalid(f'grade for an unknown item {item!r}')
        if item in seen:
            raise Invalid(f'item {item} graded twice')
        if grade.get('grade') not in GRADES or not grade.get('grader'):
            raise Invalid(f'item {item}: a grade is correct or incorrect, with the grader named')
        seen[item] = grade
    missing = sorted(items - set(seen))
    if missing:
        raise Invalid(f'{len(missing)} replies are ungraded, for example {missing[:3]}')
    return [{**key[item], 'item': item, 'human': seen[item]['grade'], 'grader': seen[item]['grader']}
            for item in sorted(items)]


def wilson(k, n, z=1.96):
    if not n:
        return (0.0, 0.0)
    p = k / n
    centre, half = (p + z * z / (2 * n)) / (1 + z * z / n), z * ((p * (1 - p) + z * z / (4 * n)) / n) ** 0.5 / (1 + z * z / n)
    return (round(centre - half, 4), round(centre + half, 4))


def sign_test(wins, losses):
    """Two-sided exact sign test p-value."""
    from math import comb

    n, k = wins + losses, min(wins, losses)
    if not n:
        return 1.0
    return min(1.0, 2 * sum(comb(n, i) for i in range(k + 1)) / 2 ** n)


def accuracy(graded, manifest):
    """Per condition: human accuracy (the reported result), machine accuracy, their agreement, and paired tests.

    Excluded cases are already absent from ``graded``; this checks so, and that every condition graded the same
    cases, so numerators and denominators are comparable."""
    excluded = set(manifest['excluded_cases'])
    if any(row['case'] in excluded for row in graded):
        raise Invalid('graded rows include an excluded (unresolved) case')
    names = [c['name'] for c in manifest['conditions']]
    by = {name: {row['case']: row for row in graded if row['condition'] == name} for name in names}
    cases = set(by[names[0]])
    if any(set(rows) != cases for rows in by.values()):
        raise Invalid('every condition must be graded on the same cases')
    out = {'cases': len(cases), 'excluded_cases': sorted(excluded), 'conditions': {}, 'paired': {}}
    for name in names:
        rows = by[name].values()
        human = sum(r['human'] == 'correct' for r in rows)
        machine = sum(r['machine'] == 'pass' for r in rows)
        agree = sum((r['human'] == 'correct') == (r['machine'] == 'pass') for r in rows)
        out['conditions'][name] = {'human_correct': human, 'human_interval': wilson(human, len(cases)),
                                   'machine_pass': machine, 'agreement': agree, 'n': len(cases)}
    base = names[0]
    for name in names[1:]:
        wins = sum(by[name][c]['human'] == 'correct' and by[base][c]['human'] != 'correct' for c in cases)
        losses = sum(by[name][c]['human'] != 'correct' and by[base][c]['human'] == 'correct' for c in cases)
        out['paired'][f'{name} vs {base}'] = {'wins': wins, 'losses': losses, 'p': sign_test(wins, losses)}
    return out


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def source_tree():
    """(commit, {path: sha256}) of every tracked file under ``src/`` as it is on disk; refuses uncommitted changes.

    The whole source tree, not a hand-picked list: the score and the tools depend on modules (for example
    ``assistant_workspace``'s argument validation) that a list can omit."""
    def git(*args):
        return subprocess.run(['git', *args], cwd=ROOT, capture_output=True, text=True, check=True).stdout

    dirty = git('status', '--porcelain', '--', 'src').strip()
    if dirty:
        raise Invalid('commit the source before freezing or running; uncommitted: ' + dirty.replace('\n', '; '))
    files = sorted(line for line in git('ls-files', '--', 'src').splitlines() if line)
    return git('rev-parse', 'HEAD').strip(), {path: sha256(ROOT / path) for path in files}


def tree_digest(files):
    return hashlib.sha256(json.dumps(files, sort_keys=True).encode()).hexdigest()


def conditions_of(protocol):
    """Validate the three conditions against their actual policies; returns them with policy digests.

    Roles, in order: ``served`` (the served model with its current policy), ``served-v4`` (the same model with the
    ``/4`` policy) and ``candidate-v4`` (the trained candidate with that same ``/4`` policy). Each condition's
    interface must be the one its policy file declares, and each model root a real 64-hex root."""
    conditions = protocol.get('conditions') or []
    if [c.get('role') for c in conditions] != list(ROLES):
        raise Invalid('the protocol names exactly three conditions, with roles ' + ', '.join(ROLES) + ', in order')
    out = []
    for c in conditions:
        if not {'name', 'role', 'interface', 'policy', 'model_root', 'unit_sha256'} <= set(c):
            raise Invalid(f'{c.get("name")}: a condition has name, role, interface, policy, model_root, unit_sha256')
        if not ROOT_HEX.match(str(c['model_root'])) or not ROOT_HEX.match(str(c['unit_sha256'])):
            raise Invalid(f'{c["name"]}: model_root and unit_sha256 are 64-hex digests')
        policy = json.loads((ROOT / c['policy']).read_text())
        if policy.get('interface') != c['interface']:
            raise Invalid(f'{c["name"]}: its policy declares {policy.get("interface")}, not {c["interface"]}')
        out.append({**c, 'policy_sha256': sha256(ROOT / c['policy'])})
    served, served_v4, candidate = out
    if len({c['name'] for c in out}) != 3:
        raise Invalid('condition names must differ')
    if served_v4['interface'] != 'workspace-documents/4' or candidate['interface'] != 'workspace-documents/4':
        raise Invalid('served-v4 and candidate-v4 use a workspace-documents/4 policy')
    if served['interface'] == 'workspace-documents/4':
        raise Invalid('served uses the current, pre-/4 policy')
    if (served['model_root'], served['unit_sha256']) != (served_v4['model_root'], served_v4['unit_sha256']):
        raise Invalid('served and served-v4 are the same model')
    if candidate['unit_sha256'] == served['unit_sha256'] or candidate['model_root'] == served['model_root']:
        raise Invalid('candidate-v4 is a different, trained unit')
    if candidate['policy_sha256'] != served_v4['policy_sha256']:
        raise Invalid('served-v4 and candidate-v4 use the same /4 policy, so only training differs')
    return out


def freeze(set_path, protocol_path, out_path):
    """Check the set and its selector disagreements, then record every digest a run depends on. No case content.

    Unresolved golds are listed by ID in ``excluded_cases``: never graded, never in any accuracy. The manifest
    pins the set, the protocol, the whole source tree and the three conditions (policy, model root, unit)."""
    data = json.loads(Path(set_path).read_text())
    if data.get('format') != FORMAT:
        raise Invalid(f'the set must declare format {FORMAT}')
    coverage = check_set(data['cases'])
    report = selector_report(data['cases'])
    if report['not_adjudicated']:
        raise Invalid("selector disagreements need the named checker's adjudication first: "
                      + ', '.join(report['not_adjudicated']))
    protocol = json.loads(Path(protocol_path).read_text())
    conditions = conditions_of(protocol)
    if not isinstance(protocol.get('seed'), int):
        raise Invalid('the protocol fixes the grading seed')
    commit, files = source_tree()
    manifest = {'format': FORMAT + '#manifest', 'set_sha256': sha256(set_path), 'protocol_sha256': sha256(protocol_path),
                'coverage': coverage, 'selector': {k: v for k, v in report.items() if k != 'disagreements'},
                'excluded_cases': report['unresolved'], 'commit': commit, 'source_tree': tree_digest(files),
                'source_files': files, 'conditions': conditions, 'seed': protocol['seed']}
    Path(out_path).write_text(json.dumps(manifest, indent=1, sort_keys=True) + '\n')
    return manifest


def verify(manifest, set_path, protocol_path):
    """Before a run: the set, the protocol, the commit, every source file and each policy are the frozen ones."""
    problems = []
    if sha256(set_path) != manifest['set_sha256']:
        problems.append('the set changed')
    if sha256(protocol_path) != manifest['protocol_sha256']:
        problems.append('the protocol changed')
    try:
        commit, files = source_tree()
    except Invalid as error:
        problems.append(str(error))
    else:
        if commit != manifest['commit']:
            problems.append(f'the checkout is at {commit[:12]}, not the frozen {manifest["commit"][:12]}')
        changed = sorted(p for p in set(files) | set(manifest['source_files'])
                         if files.get(p) != manifest['source_files'].get(p))
        if changed:
            problems.append('source files differ: ' + ', '.join(changed[:5]) + (' …' if len(changed) > 5 else ''))
    for condition in manifest['conditions']:
        if sha256(ROOT / condition['policy']) != condition['policy_sha256']:
            problems.append(f'{condition["name"]}: its policy changed')
    if problems:
        raise Invalid('not the frozen evaluation: ' + '; '.join(problems))
    return True


def check_run(manifest, name, loaded):
    """A run binds to its condition: what the runner actually loaded must be what the manifest froze.

    ``loaded`` is computed by the runner from the loaded model and policy: {'model_root', 'unit_sha256',
    'interface', 'policy_sha256'}. Returns the condition; raises on any difference."""
    condition = next((c for c in manifest['conditions'] if c['name'] == name), None)
    if condition is None:
        raise Invalid(f'no frozen condition named {name}')
    wrong = [k for k in ('model_root', 'unit_sha256', 'interface', 'policy_sha256') if loaded.get(k) != condition[k]]
    if wrong:
        raise Invalid(f'{name}: the run loaded a different ' + ', '.join(wrong) + ' than the frozen condition')
    return condition
