"""Revision selection over structured document metadata: ``workspace-documents/4`` (DESIGN_LATEST_REVISION.md).

Selection is deterministic code over a workspace's structured fields; it never reads document text for
authority. A family is the set of documents sharing ``(project, family)``. Every answer is one of the outcomes
the design defines, and a document is never presented as current or effective without complete metadata.
"""

import datetime

FIELDS = ('project', 'family', 'revision', 'status', 'effective_from', 'effective_until')
STATUSES = ('approved', 'draft', 'superseded')


class Invalid(ValueError):
    """A metadata field has the wrong type or value. Its document then has no usable metadata."""


def _date(value, name):
    if value is None:
        return None
    if not isinstance(value, str):
        raise Invalid(f'{name} must be an ISO date')
    try:
        return datetime.date.fromisoformat(value)
    except ValueError:
        raise Invalid(f'{name} must be an ISO date') from None


def metadata(document):
    """``(fields, problems)``: the metadata fields present that are valid, and the names of those that are not.

    Unknown fields are never read. A malformed field is recorded as a problem rather than dropped, so a family
    containing it is ``incomplete``: an invalid field never silently becomes absent and lets the rest look
    authoritative. The workspace still opens; real documents with odd fields are not refused."""
    out, problems = {}, []
    for name in ('project', 'family'):
        if name in document:
            if isinstance(document[name], str) and document[name]:
                out[name] = document[name]
            else:
                problems.append(name)
    if 'revision' in document:
        if type(document['revision']) is int and document['revision'] >= 1:
            out['revision'] = document['revision']
        else:
            problems.append('revision')
    if 'status' in document:
        if document['status'] in STATUSES:
            out['status'] = document['status']
        else:
            problems.append('status')
    for name in ('effective_from', 'effective_until'):
        if name in document:
            try:
                _date(document[name], name)
                out[name] = document[name]
            except Invalid:
                problems.append(name)
    return out, problems


def families(documents):
    """``{(project, family): [document, ...]}`` for documents that name both; others belong to no family."""
    groups = {}
    for document in documents:
        meta = document.get('metadata', {})
        if 'project' in meta and 'family' in meta:
            groups.setdefault((meta['project'], meta['family']), []).append(document)
    return groups


def members(documents, project, family):
    """A family's members, plus any document that might belong to it but whose identity is malformed.

    A document with a valid ``family`` but malformed ``project`` (or the reverse) cannot be placed, yet could
    belong here; it is returned too, so the family is ``incomplete`` rather than silently missing a member."""
    out = []
    for document in documents:
        meta, problems = document.get('metadata', {}), document.get('problems', [])
        if meta.get('project') == project and meta.get('family') == family:
            out.append(document)
        elif ('project' in problems and meta.get('family') == family) or (
                'family' in problems and meta.get('project') == project):
            out.append(document)
    return out


def usable(member):
    return {'status', 'revision'} <= set(member.get('metadata', {})) and not member.get('problems')


def complete(members):
    return all(usable(m) for m in members)


def _unusable(members):
    return {'unorderable': sorted(m['id'] for m in members if not usable(m))}


def current(members):
    """The highest approved revision of a family, as one of the design's outcomes.

    Returns ``(outcome, chosen, detail)``. ``chosen`` is the selected document for ``known_current``, else None.
    """
    if not members:
        return 'no_metadata', None, {}
    if not complete(members):
        return 'incomplete', None, _unusable(members)
    approved = [m for m in members if m['metadata']['status'] == 'approved']
    if not approved:
        statuses = {m['id']: [m['metadata']['revision'], m['metadata']['status']] for m in members}  # JSON-shaped
        return 'known_no_approved', None, {'statuses': statuses}
    top = max(m['metadata']['revision'] for m in approved)
    leaders = [m for m in approved if m['metadata']['revision'] == top]
    if len(leaders) > 1:
        return 'ambiguous', None, {'candidates': sorted(m['id'] for m in leaders)}
    return 'known_current', leaders[0], {}


def effective(members, on):
    """The approved or superseded member whose ``[effective_from, effective_until)`` contains ``on``.

    Drafts are never candidates. A candidate without ``effective_from``, or with a reversed or empty interval,
    makes the date question unanswerable from this family (``incomplete``): its period is unknown, and it could
    be the one that applied. Overlapping intervals are ``ambiguous``; none covering the date is ``none_effective``.
    """
    day = _date(on, 'date')
    if not members:
        return 'no_metadata', None, {}
    if not complete(members):
        return 'incomplete', None, _unusable(members)
    candidates = [m for m in members if m['metadata']['status'] in ('approved', 'superseded')]
    covering, unknown = [], []
    for member in candidates:
        meta = member['metadata']
        start, end = _date(meta.get('effective_from'), 'effective_from'), _date(meta.get('effective_until'),
                                                                                 'effective_until')
        if start is None or (end is not None and end <= start):
            unknown.append(member['id'])
            continue
        if start <= day and (end is None or day < end):
            covering.append(member)
    if unknown:
        return 'incomplete', None, {'no_period': sorted(unknown)}
    if len(covering) > 1:
        return 'ambiguous', None, {'candidates': sorted(m['id'] for m in covering)}
    if not covering:
        return 'none_effective', None, {'date': on}
    return 'known_effective', covering[0], {}


def describe(document):
    """A document's reference with its metadata, as tools return it."""
    return {'id': document['id'], 'title': document['title'], **document.get('metadata', {})}
