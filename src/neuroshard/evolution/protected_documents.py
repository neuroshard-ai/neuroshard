"""The protected, human-written documents evaluation set: case format, gold checks and the frozen scorer.

DESIGN_LATEST_REVISION.md, acceptance test 2. People write the set, and the authors of the training corpus never
see it. This module never holds a case: the set lives outside the repository, and only its SHA-256 and this
module's are committed, before the set is first run (``freeze``).

**A case** is one workspace and one question. Documents carry the structured fields the user's device keeps
(``project``, ``family``, ``revision``, ``status``, ``effective_from``, ``effective_until``), any of which may be
absent or wrong on purpose. ``expected`` is one of:

- ``answer``: every string in ``facts`` must appear in the answer, and the cited document IDs must be exactly
  ``source_ids``. ``selection`` says how the gold source was chosen, and ``check`` verifies it against the
  deterministic code: ``latest_approved`` (``current_document``), ``effective`` on ``date``
  (``effective_document``), ``named`` (the question names a revision; no currency is claimed), or ``other``.
- ``not_found``: the documents do not contain it; the reply must report not found and submit no answer.
- ``decline``: no document can honestly be chosen (ambiguous, incomplete or no approved revision). Every ID in
  ``must_cite`` must be cited, and none of ``must_not_state`` may appear.

**The primary score is interface-neutral,** so the design's three conditions (``/3``, ``/4`` without training,
``/4`` with a trained unit) are compared on one rule. It reads the reply: a submitted answer if there is one,
otherwise the final text. The route's own scorer (``assistant_documents.score_round``), which under ``/3`` and
``/4`` also requires quoted evidence, is reported beside it, never instead of it.
"""
import hashlib
import json
import re
from pathlib import Path

from neuroshard.evolution import assistant_documents as documents

FORMAT = 'neuroshard-protected-documents-set/1'
KINDS = ('answer', 'not_found', 'decline')
SELECTIONS = ('latest_approved', 'effective', 'named', 'other')
# Coverage the set must have before it is frozen: the design's list, so no category is judged on a handful.
CATEGORIES = ('latest_approved_with_newer_draft', 'latest_approved_newest', 'single_revision', 'unrelated_families',
              'ambiguous_versions', 'no_approved_revision', 'incomplete_metadata', 'misleading_text', 'named_revision',
              'as_of_date', 'same_value_old_and_current', 'shared_titles', 'not_found')
MINIMUM_PER_CATEGORY = 8
ID = re.compile(r'\Adoc-[0-9a-f]{8,}\Z')
CITED = re.compile(r'doc-[0-9a-f]{8,}')


class Invalid(ValueError):
    """A written case is malformed, or its gold disagrees with the documents or the selection code."""


def check_case(case):
    """Raise ``Invalid`` unless the case is well formed and its gold is consistent. Returns the case ID.

    The checks are mechanical, never a judgment of the question: IDs are well formed and unique, every fact of an
    answer appears in a cited document's text, and a gold chosen by selection equals what the code selects."""
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
        raise Invalid(f'{where}: document ids must be unique and look like doc-<at least 8 hex digits>')
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
            if not any(documents.normalize(fact) in documents.normalize(by_id[s]['content']) for s in sources):
                raise Invalid(f'{where}: fact {fact!r} is in none of the cited documents')
        selection = expected.get('selection')
        if selection not in SELECTIONS:
            raise Invalid(f'{where}: selection must be one of {", ".join(SELECTIONS)}')
        if selection in ('latest_approved', 'effective'):
            chosen = select(docs, expected, where)
            if [chosen] != list(sources):
                raise Invalid(f'{where}: the selection code chooses {chosen}, not {sources}')
    elif kind == 'decline':
        cite, avoid = expected.get('must_cite') or [], expected.get('must_not_state') or []
        if not cite and not avoid:
            raise Invalid(f'{where}: a decline needs must_cite or must_not_state')
        if any(c not in by_id for c in cite):
            raise Invalid(f'{where}: must_cite names a document not in this case')
    return key


def select(docs, expected, where):
    """The document the deterministic ``/4`` code selects for this gold, or raise if it selects none."""
    from neuroshard.assistant import interfaces

    workspace = documents.Workspace(interfaces.documents_world({'documents': [dict(d) for d in docs]}, 4), 4)
    arguments = {'project': expected.get('project'), 'family': expected.get('family')}
    if expected['selection'] == 'effective':
        arguments['date'] = expected.get('date')
        result = workspace.execute({'name': 'effective_document', 'arguments': arguments})
    else:
        result = workspace.execute({'name': 'current_document', 'arguments': arguments})
    chosen = (result.get('document') or {}).get('id')
    if chosen is None:
        raise Invalid(f'{where}: the selection code chooses no document ({result.get("outcome")})')
    return chosen


def check_set(cases):
    """Check every case and the coverage minimums. Returns {category: count}."""
    seen, counts = set(), {}
    for case in cases:
        key = check_case(case)
        if key in seen:
            raise Invalid(f'duplicate case id {key}')
        seen.add(key)
        counts[case['category']] = counts.get(case['category'], 0) + 1
    short = {c: counts.get(c, 0) for c in CATEGORIES if counts.get(c, 0) < MINIMUM_PER_CATEGORY}
    if short:
        raise Invalid(f'too few cases (minimum {MINIMUM_PER_CATEGORY} per category): {short}')
    return counts


def reply_of(snapshot, final_text):
    """(text, cited IDs) the primary score reads: a submitted answer if any, otherwise the final text."""
    answer = (snapshot or {}).get('answer')
    if answer:
        return answer['answer'], set(answer['source_ids'])
    text = final_text if isinstance(final_text, str) else ''
    return text, set(CITED.findall(text))


def score(case, snapshot, final_text, terminated=True):
    """The primary, interface-neutral score: True if the conversation's outcome meets the case's gold."""
    expected = case['expected']
    if not terminated or not isinstance(final_text, str) or not final_text.strip():
        return False
    reported = (snapshot or {}).get('not_found') is not None and not (snapshot or {}).get('answer')
    text, cited = reply_of(snapshot, final_text)
    if expected['kind'] == 'not_found':
        return reported
    if expected['kind'] == 'answer':
        if reported:
            return False
        return (all(documents.normalize(f) in documents.normalize(text) for f in expected['facts'])
                and cited == set(expected['source_ids']))
    if reported:
        text = (text + ' ' + str((snapshot or {}).get('not_found', {}).get('reason', ''))).strip()
        cited |= set(CITED.findall(text))
    return (set(expected.get('must_cite') or ()) <= cited
            and not any(documents.normalize(f) in documents.normalize(text) for f in expected.get('must_not_state') or ()))


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def freeze(set_path, out_path):
    """Check the set, then record its digest and the scorer's, before it is ever run. Never copies a case."""
    data = json.loads(Path(set_path).read_text())
    if data.get('format') != FORMAT:
        raise Invalid(f'the set must declare format {FORMAT}')
    counts = check_set(data['cases'])
    manifest = {'format': FORMAT + '#manifest', 'set_sha256': sha256(set_path), 'cases': len(data['cases']),
                'categories': counts, 'scorer_sha256': sha256(__file__),
                'documents_tools_sha256': sha256(documents.__file__)}
    Path(out_path).write_text(json.dumps(manifest, indent=1, sort_keys=True) + '\n')
    return manifest
