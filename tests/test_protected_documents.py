"""The protected-set kit: its checks catch malformed or inconsistent golds, and its scorer is interface-neutral."""
import copy
import json

import pytest

from neuroshard.evolution import protected_documents as protected


def doc(key, revision, status, content, project='Lime Street lease', family='lease', **extra):
    return {'id': key, 'title': f'{project} ({revision})', 'content': content, 'project': project, 'family': family,
            'revision': revision, 'status': status, **extra}


CASE = {
    'id': 'p-001', 'category': 'latest_approved_with_newer_draft', 'written_by': 'A', 'checked_by': 'B',
    'question': 'What is the monthly rent in the lease we actually agreed?',
    'documents': [doc('doc-0000000a', 1, 'superseded', 'Rent is 1,150 a month.'),
                  doc('doc-0000000b', 2, 'approved', 'Rent is 1,200 a month.'),
                  doc('doc-0000000c', 3, 'draft', 'Rent is 1,260 a month.')],
    'expected': {'kind': 'answer', 'facts': ['1,200'], 'source_ids': ['doc-0000000b'],
                 'selection': 'latest_approved', 'project': 'Lime Street lease', 'family': 'lease'},
}


def test_a_consistent_case_passes():
    assert protected.check_case(CASE) == 'p-001'


@pytest.mark.parametrize('change, message', [
    (lambda c: c['expected'].update(source_ids=['doc-0000000c'], facts=['1,260']), 'selection code chooses doc-0000000b'),
    (lambda c: c['expected'].update(facts=['1,300']), "fact '1,300' is in none"),
    (lambda c: c.update(checked_by='A'), 'two different people'),
    (lambda c: c['documents'][1].update(id='doc-0000000a'), 'unique'),
    (lambda c: c.update(category='vibes'), 'category must be one of'),
    (lambda c: c['documents'][1].update(status='draft'), 'selection code chooses no document'),
])
def test_inconsistent_golds_are_refused(change, message):
    case = copy.deepcopy(CASE)
    change(case)
    with pytest.raises(protected.Invalid, match=message):
        protected.check_case(case)


def test_the_primary_score_reads_a_submitted_answer_or_the_final_text_alike():
    submitted = {'answer': {'answer': 'The rent is 1,200 a month.', 'source_ids': ['doc-0000000b']}, 'not_found': None}
    assert protected.score(CASE, submitted, 'The rent is 1,200 a month. (doc-0000000b)')
    prose = {'answer': None, 'not_found': None}
    assert protected.score(CASE, prose, 'The rent is 1,200 a month. (Sources: doc-0000000b)')
    assert not protected.score(CASE, prose, 'The rent is 1,200 a month. (Sources: doc-0000000b, doc-0000000c)')
    assert not protected.score(CASE, prose, 'The rent is 1,260 a month. (Sources: doc-0000000c)')
    assert not protected.score(CASE, {'answer': None, 'not_found': {'reason': 'x'}}, 'I could not find that.')
    assert not protected.score(CASE, prose, 'The rent is 1,200 a month. (Sources: doc-0000000b)', terminated=False)


def test_a_decline_must_name_the_conflict_and_state_neither_value():
    case = {**copy.deepcopy(CASE), 'category': 'ambiguous_versions'}
    case['documents'][2].update(status='approved', revision=2)
    case['expected'] = {'kind': 'decline', 'must_cite': ['doc-0000000b', 'doc-0000000c'],
                        'must_not_state': ['1,200 a month', '1,260 a month']}
    protected.check_case(case)
    empty = {'answer': None, 'not_found': None}
    assert protected.score(case, empty, 'Two approved revision 2 versions exist (doc-0000000b, doc-0000000c); '
                                        'which one applies?')
    assert not protected.score(case, empty, 'Rent is 1,200 a month (doc-0000000b).')


def test_the_writers_guide_example_passes_the_checks():
    import re
    from pathlib import Path

    text = (Path(protected.__file__).resolve().parents[3] / 'docs' / 'PROTECTED_DOCUMENTS_SET.md').read_text()
    case = json.loads(re.search(r'```json\n(.*?)```', text, re.S).group(1).replace('…', ''))
    assert protected.check_case(case) == 'p-001'


def test_freeze_needs_coverage_and_records_digests_only(tmp_path):
    path = tmp_path / 'set.json'
    path.write_text(json.dumps({'format': protected.FORMAT, 'cases': [CASE]}))
    with pytest.raises(protected.Invalid, match='too few cases'):
        protected.freeze(path, tmp_path / 'manifest.json')
    cases = []
    for category in protected.CATEGORIES:
        for n in range(protected.MINIMUM_PER_CATEGORY):
            c = copy.deepcopy(CASE)
            c.update(id=f'{category}-{n}', category=category)
            cases.append(c)
    path.write_text(json.dumps({'format': protected.FORMAT, 'cases': cases}))
    manifest = protected.freeze(path, tmp_path / 'manifest.json')
    assert manifest['cases'] == len(cases) and manifest['set_sha256'] == protected.sha256(path)
    assert 'Lime Street' not in (tmp_path / 'manifest.json').read_text()
