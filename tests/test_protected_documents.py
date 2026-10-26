"""The protected-set kit. The review's negative controls must fail; legitimate replies must pass."""
import copy
import json

import pytest

from neuroshard.evolution import document_revisions, protected_documents as protected


def doc(key, revision, status, content, project='Lime Street lease', family='lease', title=None, **extra):
    return {'id': key, 'title': title or f'{project}', 'content': content, 'project': project, 'family': family,
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


def verdict(case, text, **kwargs):
    return protected.score(case, text, **kwargs)


# The review's negative controls (pr149-150-review-20261025/probe_kit.py), each of which used to pass.

@pytest.mark.parametrize('text', [
    'The rent is 12,000 a month. (doc-0000000b)',          # a different number that contains the gold's digits
    'The rent is 11,200 a month. (doc-0000000b)',
    'The rent is not 1,200 a month. (doc-0000000b)',       # the gold, negated
    'The rent is 1,200 a month. (doc-0000000bZ)',          # an ID that is not a document of the case
    'The rent is 1,200 a month. (doc-0000000b, doc-0000000c)',  # an extra citation
    'The rent is 1,260 a month. (doc-0000000c)',           # the draft
    'Bananas.',
    'The rent is -1,200 a month. (doc-0000000b)',          # the gold's digits with the opposite sign
    'The rent is \u22121,200 a month. (doc-0000000b)',
])
def test_wrong_or_unsupported_answers_fail(text):
    assert verdict(CASE, text)['verdict'] == 'fail'


def test_a_correct_submission_cannot_hide_a_wrong_delivered_reply():
    submitted = {'answer': 'The rent is 1,200 a month.', 'source_ids': ['doc-0000000b']}
    assert verdict(CASE, 'The rent is 1,260 a month. (doc-0000000c)', submitted=submitted)['verdict'] == 'fail'
    assert verdict(CASE, 'The rent is 1,200 a month. (doc-0000000b)', submitted=submitted)['verdict'] == 'pass'


def test_a_not_found_report_followed_by_an_answer_fails():
    case = copy.deepcopy(CASE)
    case.update(category='not_found', question='What is the deposit?')
    case['expected'] = {'kind': 'not_found', 'must_not_state': ['1,200', '1,260', '1,150']}
    protected.check_case(case)
    assert verdict(case, 'The rent is 1,260 a month.', reported={'reason': 'No evidence'})['verdict'] == 'fail'
    assert verdict(case, 'I could not find the deposit in your documents.', reported={'reason': 'x'})['verdict'] == 'pass'


def test_a_decline_must_name_the_conflict_and_explain_it():
    case = copy.deepcopy(CASE)
    case.update(category='ambiguous_versions')
    case['documents'][2].update(status='approved', revision=2)
    case['expected'] = {'kind': 'decline', 'must_cite': ['doc-0000000b', 'doc-0000000c'],
                        'must_not_state': ['1,200', '1,260']}
    protected.check_case(case)
    assert verdict(case, 'Bananas.')['verdict'] == 'fail'
    assert verdict(case, 'The rent is 1,200 a month (doc-0000000b).')['verdict'] == 'fail'
    assert verdict(case, 'I see doc-0000000b and doc-0000000c.')['verdict'] == 'fail'  # names them, explains nothing
    assert verdict(case, 'Two approved revision 2 versions exist (doc-0000000b, doc-0000000c); which one applies?'
                   )['verdict'] == 'pass'


def test_legitimate_answers_pass_and_mentions_of_other_versions_go_to_review():
    for text in ('The rent is 1,200 a month. (Sources: doc-0000000b)', 'Rent: £1,200/month [doc-0000000b]',
                 'The agreed lease sets rent at 1200 a month (doc-0000000b).'):
        assert verdict(CASE, text) == {'verdict': 'pass', 'reasons': [], 'contested': False, 'notes': []}
    both = verdict(CASE, 'The approved lease says 1,200 a month, not the draft\'s 1,260. (doc-0000000b)')
    assert both['verdict'] == 'pass' and both['contested']


def test_dates_are_compared_as_dates():
    case = copy.deepcopy(CASE)
    for d, date in zip(case['documents'], ('2027-04-01', '2027-05-14', '2027-05-26')):
        d['content'] = f'The lease starts on {date}.'
    case['expected'].update(facts=['2027-05-14'])
    protected.check_case(case)
    assert verdict(case, 'It starts on 14 May 2027 (doc-0000000b).')['verdict'] == 'pass'
    assert verdict(case, 'It starts on May 14, 2027 (doc-0000000b).')['verdict'] == 'pass'
    assert verdict(case, 'It starts on 2027-05-26 (doc-0000000b).')['verdict'] == 'fail'


def test_a_gold_fact_must_be_a_whole_value_in_a_cited_document():
    case = copy.deepcopy(CASE)
    case['documents'][1]['content'] = 'Rent is 11,200 a month.'
    with pytest.raises(protected.Invalid, match='not a whole value'):
        protected.check_case(case)


def test_the_selection_code_reports_disagreements_but_never_overrules_the_gold(monkeypatch):
    protected.check_case(CASE)  # the checks never consult the selection code
    monkeypatch.setattr(document_revisions, 'current', lambda group: ('incomplete', None, {}))
    protected.check_case(CASE)
    report = protected.selector_report([CASE])
    assert report['not_adjudicated'] == ['p-001'] and report['confirmed_code_errors'] == []
    adjudicated = {**copy.deepcopy(CASE), 'adjudication': {'by': 'B', 'decision': 'gold_correct', 'note': 'rev 2'}}
    report = protected.selector_report([adjudicated])
    assert report['not_adjudicated'] == []
    assert report['confirmed_code_errors'] == ['p-001']
    for by in ('A', 'C'):  # the writer, or anyone but the named checker
        other = {**copy.deepcopy(CASE), 'adjudication': {'by': by, 'decision': 'gold_correct'}}
        assert protected.selector_report([other])['not_adjudicated'] == ['p-001']


@pytest.mark.parametrize('change, message', [
    (lambda c: c['expected'].update(facts=['1,300']), "'1,300' is not a whole value"),
    (lambda c: c.update(checked_by='A'), 'two different people'),
    (lambda c: c['documents'][1].update(id='doc-0000000a'), 'unique'),
    (lambda c: c.update(category='vibes'), 'category must be one of'),
    (lambda c: c['documents'][2].update(status='superseded'), 'needs a draft newer'),
    (lambda c: c.update(category='not_found'), 'is a not_found case'),
    (lambda c: c.update(category='ambiguous_versions'), 'is a decline'),
])
def test_malformed_cases_and_category_mismatches_are_refused(change, message):
    case = copy.deepcopy(CASE)
    change(case)
    with pytest.raises(protected.Invalid, match=message):
        protected.check_case(case)


def workspace(index, category):
    """A distinct, structurally valid case per category, for the coverage tests only."""
    project, n = f'Unit {index}', index
    h = lambda k: f'doc-{n:06x}{k:02x}'  # noqa: E731
    base = {'id': f'{category}-{index}', 'category': category, 'written_by': 'A', 'checked_by': 'B',
            'question': f'What is the rent for {project}?'}
    rent = [f'{1000 + n * 10 + r},00'.replace(',00', '') for r in range(3)]
    docs = [doc(h(1), 1, 'superseded', f'Rent is {rent[0]} a month.', project),
            doc(h(2), 2, 'approved', f'Rent is {rent[1]} a month.', project),
            doc(h(3), 3, 'draft', f'Rent is {rent[2]} a month.', project)]
    answer = {'kind': 'answer', 'facts': [rent[1]], 'source_ids': [h(2)], 'selection': 'latest_approved',
              'project': project, 'family': 'lease'}
    if category == 'latest_approved_newest':
        docs = docs[:2]
    elif category == 'single_revision':
        docs = [docs[1]]
    elif category == 'unrelated_families':
        docs.append(doc(h(4), 1, 'approved', 'Parking is 40 a month.', project, family='parking'))
    elif category == 'ambiguous_versions':
        docs[2].update(status='approved', revision=2)
        answer = {'kind': 'decline', 'must_cite': [h(2), h(3)], 'must_not_state': [rent[1], rent[2]]}
    elif category == 'no_approved_revision':
        docs[1]['status'] = 'superseded'
        answer = {'kind': 'decline', 'must_cite': [h(3)], 'must_not_state': [rent[2]]}
    elif category == 'incomplete_metadata':
        del docs[2]['status']
        answer = {'kind': 'decline', 'must_cite': [h(3)]}
    elif category == 'misleading_text':
        docs[2]['content'] += ' This is the final approved version.'
    elif category == 'named_revision':
        answer = {'kind': 'answer', 'facts': [rent[0]], 'source_ids': [h(1)], 'selection': 'named'}
    elif category == 'as_of_date':
        docs[0].update(effective_from='2027-01-01', effective_until='2027-02-01')
        docs[1].update(effective_from='2027-02-01')
        answer = {'kind': 'answer', 'facts': [rent[0]], 'source_ids': [h(1)], 'selection': 'effective',
                  'project': project, 'family': 'lease', 'date': '2027-01-15'}
    elif category == 'same_value_old_and_current':
        docs[0]['content'] = f'Rent is {rent[1]} a month.'
    elif category == 'shared_titles':
        docs.append(doc(h(4), 1, 'approved', 'Parking is 40 a month.', f'Other {index}', family='parking',
                        title=project))
    elif category == 'not_found':
        answer = {'kind': 'not_found'}
    return {**base, 'documents': docs, 'expected': answer}


def full_set():
    return [workspace(index, category) for c, category in enumerate(protected.CATEGORIES)
            for index in range(c * 100, c * 100 + protected.MINIMUM_PER_CATEGORY)]


def test_a_varied_set_passes_coverage_with_machine_and_human_parts_kept_apart():
    report = protected.check_set(full_set())
    assert report['cases'] == 13 * protected.MINIMUM_PER_CATEGORY and report['workspaces'] >= protected.MINIMUM_WORKSPACES
    assert 'category structure' in report['machine_verified'] and 'semantic diversity' in report['human_attested']


def test_one_workspace_relabelled_across_categories_is_not_coverage():
    """The review froze 104 copies of one case relabelled across all 13 categories."""
    copies = []
    for category in protected.CATEGORIES:
        for n in range(protected.MINIMUM_PER_CATEGORY):
            item = copy.deepcopy(CASE)
            item.update(id=f'{category}-{n}', category=category)
            copies.append(item)
    with pytest.raises(protected.Invalid):
        protected.check_set(copies)
    same_category = [{**copy.deepcopy(CASE), 'id': f'p-{n}'} for n in range(protected.MINIMUM_PER_CATEGORY)]
    with pytest.raises(protected.Invalid, match='same question of the same workspace'):
        protected.check_set(same_category)


def test_too_few_workspaces_per_category_is_refused():
    cases = full_set()
    first = cases[0]
    for n, case in enumerate(cases[:protected.MINIMUM_PER_CATEGORY]):
        case.update(documents=copy.deepcopy(first['documents']), expected=copy.deepcopy(first['expected']),
                    question=f'Variant {n}: what is the rent?')
    with pytest.raises(protected.Invalid, match='distinct workspaces'):
        protected.check_set(cases)


R = {k: f'{k:x}' * 64 for k in range(1, 6)}  # distinct 64-hex roots


def protocol(tmp_path, **changes):
    (tmp_path / 'v2.json').write_text(json.dumps({'interface': 'workspace-documents/2'}))
    (tmp_path / 'v4.json').write_text(json.dumps({'interface': 'workspace-documents/4'}))
    conditions = [
        {'name': 'served', 'role': 'served', 'interface': 'workspace-documents/2', 'policy': 'v2.json',
         'model_root': R[1], 'unit_sha256': R[2]},
        {'name': 'served-v4', 'role': 'served-v4', 'interface': 'workspace-documents/4', 'policy': 'v4.json',
         'model_root': R[1], 'unit_sha256': R[2]},
        {'name': 'candidate', 'role': 'candidate-v4', 'interface': 'workspace-documents/4', 'policy': 'v4.json',
         'model_root': R[3], 'unit_sha256': R[4]}]
    for index, change in changes.items():
        conditions[int(index[1:])].update(change)
    path = tmp_path / 'protocol.json'
    path.write_text(json.dumps({'seed': 7, 'conditions': conditions}))
    return path


@pytest.fixture
def tree(tmp_path, monkeypatch):
    """A temporary source tree standing in for the repository: every 'tracked' file, including the tool validator."""
    for name in ('src/neuroshard/evolution/protected_documents.py', 'src/neuroshard/evolution/document_revisions.py',
                 'src/neuroshard/evolution/assistant_workspace.py', 'src/neuroshard/assistant/interfaces.py'):
        (tmp_path / name).parent.mkdir(parents=True, exist_ok=True)
        (tmp_path / name).write_text(name)
    monkeypatch.setattr(protected, 'ROOT', tmp_path)
    state = {'commit': 'c' * 40}

    def source_tree():
        names = sorted(str(p.relative_to(tmp_path)) for p in (tmp_path / 'src').rglob('*.py'))
        return state['commit'], {n: protected.sha256(tmp_path / n) for n in names}

    monkeypatch.setattr(protected, 'source_tree', source_tree)
    return state


def frozen(tmp_path, cases=None, **changes):
    path = tmp_path / 'set.json'
    path.write_text(json.dumps({'format': protected.FORMAT, 'cases': cases or full_set()}))
    proto = protocol(tmp_path, **changes)
    return protected.freeze(path, proto, tmp_path / 'manifest.json'), path, proto


def test_freeze_pins_the_whole_source_tree_and_verify_catches_any_change(tmp_path, tree):
    manifest, path, proto = frozen(tmp_path)
    assert 'src/neuroshard/evolution/assistant_workspace.py' in manifest['source_files']
    assert [c['role'] for c in manifest['conditions']] == list(protected.ROLES)
    assert 'Lime Street' not in (tmp_path / 'manifest.json').read_text()
    assert protected.verify(manifest, path, proto)
    (tmp_path / 'src/neuroshard/evolution/assistant_workspace.py').write_text('changed validator')
    with pytest.raises(protected.Invalid, match='assistant_workspace.py'):
        protected.verify(manifest, path, proto)


def test_verify_checks_the_commit_and_new_files(tmp_path, tree):
    manifest, path, proto = frozen(tmp_path)
    tree['commit'] = 'd' * 40
    with pytest.raises(protected.Invalid, match='not the frozen'):
        protected.verify(manifest, path, proto)
    tree['commit'] = 'c' * 40
    (tmp_path / 'src/neuroshard/evolution/new_helper.py').write_text('x')
    with pytest.raises(protected.Invalid, match='new_helper.py'):
        protected.verify(manifest, path, proto)


@pytest.mark.parametrize('changes, message', [
    ({'c2': {'policy': 'v2.json'}}, 'declares workspace-documents/2, not workspace-documents/4'),  # /4 label, /2 policy
    ({'c2': {'model_root': 'r2'}}, '64-hex'),                                                     # placeholder roots
    ({'c2': {'role': 'served-v4'}}, 'exactly three conditions'),
    ({'c1': {'unit_sha256': R[5]}}, 'served and served-v4 are the same model'),
    ({'c2': {'unit_sha256': R[2], 'model_root': R[1]}}, 'different, trained unit'),
    ({'c0': {'interface': 'workspace-documents/4', 'policy': 'v4.json'}}, 'current, pre-/4 policy'),
])
def test_freeze_accepts_only_the_approved_three_way_comparison(tmp_path, tree, changes, message):
    with pytest.raises(protected.Invalid, match=message):
        frozen(tmp_path, **changes)


def test_freeze_refuses_a_two_condition_protocol(tmp_path, tree):
    path = tmp_path / 'set.json'
    path.write_text(json.dumps({'format': protected.FORMAT, 'cases': full_set()}))
    proto = protocol(tmp_path)
    data = json.loads(proto.read_text())
    data['conditions'] = data['conditions'][::2]
    proto.write_text(json.dumps(data))
    with pytest.raises(protected.Invalid, match='exactly three conditions'):
        protected.freeze(path, proto, tmp_path / 'manifest.json')


def test_a_run_must_load_exactly_its_frozen_condition(tmp_path, tree):
    manifest, _, _ = frozen(tmp_path)
    candidate = manifest['conditions'][2]
    loaded = {k: candidate[k] for k in ('model_root', 'unit_sha256', 'interface', 'policy_sha256')}
    assert protected.check_run(manifest, 'candidate', loaded) == candidate
    with pytest.raises(protected.Invalid, match='unit_sha256'):
        protected.check_run(manifest, 'candidate', {**loaded, 'unit_sha256': R[2]})


def test_freeze_refuses_unadjudicated_selector_disagreements(tmp_path, tree, monkeypatch):
    monkeypatch.setattr(document_revisions, 'current', lambda group: ('incomplete', None, {}))
    with pytest.raises(protected.Invalid, match='adjudication'):
        frozen(tmp_path)


def unresolved_set():
    """The review's control: an old revision's amount recorded as the gold, adjudicated unresolved."""
    cases = full_set()
    first = cases[0]
    first['expected'].update(source_ids=[first['documents'][0]['id']],
                             facts=[first['documents'][0]['content'].split()[2]])
    first['adjudication'] = {'by': 'B', 'decision': 'unresolved', 'note': 'Needs clarification'}
    return cases, first['id']


def test_unresolved_golds_are_excluded_from_grading_and_every_accuracy(tmp_path, tree):
    cases, disputed = unresolved_set()
    manifest, _, _ = frozen(tmp_path, cases)
    assert manifest['excluded_cases'] == [disputed]
    names = [c['name'] for c in manifest['conditions']]
    results = [{'condition': n, 'case': c['id'], 'final_text': 'reply', 'machine': {'verdict': 'pass'}}
               for n in names for c in cases]
    sheet, key = protected.grading_sheet(results, seed=7, excluded=manifest['excluded_cases'])
    assert disputed not in {row['case'] for row in sheet}
    graded = protected.ingest_grades(sheet, key, [{'item': r['item'], 'grade': 'correct', 'grader': 'G'}
                                                  for r in sheet])
    report = protected.accuracy(graded, manifest)
    assert report['cases'] == len(cases) - 1 and report['excluded_cases'] == [disputed]
    assert all(c['n'] == len(cases) - 1 for c in report['conditions'].values())
    with pytest.raises(protected.Invalid, match='excluded'):
        protected.accuracy(graded + [{**graded[0], 'case': disputed}], manifest)


def test_every_reply_is_graded_blind_and_incomplete_grading_is_refused():
    results = [{'condition': c, 'case': f'p-{i}', 'final_text': f'reply {c} {i}', 'machine': {'verdict': 'pass'}}
               for c in ('served', 'served-v4', 'candidate') for i in range(10)]
    sheet, key = protected.grading_sheet(results, seed=7)
    assert len(sheet) == 30 and all(set(row) == {'item', 'case', 'reply'} for row in sheet)
    assert protected.grading_sheet(results, seed=7) == (sheet, key)
    grades = [{'item': r['item'], 'grade': 'incorrect', 'grader': 'G'} for r in sheet]
    with pytest.raises(protected.Invalid, match='1 replies are ungraded'):
        protected.ingest_grades(sheet, key, grades[:-1])
    with pytest.raises(protected.Invalid, match='graded twice'):
        protected.ingest_grades(sheet, key, grades + grades[:1])
    with pytest.raises(protected.Invalid, match='correct or incorrect'):
        protected.ingest_grades(sheet, key, [{**grades[0], 'grade': 'maybe'}] + grades[1:])


def test_human_grades_are_the_result_even_where_the_machine_passed_a_wrong_reply(tmp_path, tree):
    """The review's control: 100 known-wrong replies the machine passes. Every one is graded, so none survives."""
    manifest, _, _ = frozen(tmp_path)
    cases = [f'p-{i}' for i in range(100)]
    results = [{'condition': c['name'], 'case': k, 'final_text': 'The rent is 1,200 a month. That amount is wrong.',
                'machine': {'verdict': 'pass'}} for c in manifest['conditions'] for k in cases]
    sheet, key = protected.grading_sheet(results, seed=7)
    graded = protected.ingest_grades(sheet, key, [{'item': r['item'], 'grade': 'incorrect', 'grader': 'G'}
                                                  for r in sheet])
    report = protected.accuracy(graded, {**manifest, 'excluded_cases': []})
    assert all(c['human_correct'] == 0 and c['machine_pass'] == 100 and c['agreement'] == 0
               for c in report['conditions'].values())


def test_the_writers_guide_example_passes_the_checks():
    import re
    from pathlib import Path

    text = (Path(protected.__file__).resolve().parents[3] / 'docs' / 'PROTECTED_DOCUMENTS_SET.md').read_text()
    case = json.loads(re.search(r'```json\n(.*?)```', text, re.S).group(1).replace('…', ''))
    assert protected.check_case(case) == 'p-001'
