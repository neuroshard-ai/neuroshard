"""workspace-documents/4: revision selection over structured metadata (DESIGN_LATEST_REVISION.md, revision 3)."""
import json
import random
from pathlib import Path

import pytest

from neuroshard.assistant import interfaces
from neuroshard.evolution import assistant_documents as documents, document_revisions as revisions
from neuroshard.evolution import revision_fixtures

GOLDEN = Path(__file__).parent / 'fixtures' / 'documents-v1-v3-golden.json'


def doc(i, project='P', family='plan', revision=None, status=None, start=None, end=None, title='Plan', **extra):
    row = {'id': f'doc-{i:08x}', 'title': title, 'content': f'Revision text {i}.'}
    for key, value in (('project', project), ('family', family), ('revision', revision), ('status', status),
                       ('effective_from', start), ('effective_until', end)):
        if value is not None:
            row[key] = value
    row.update(extra)
    return row


def ws(*docs, shuffle=None):
    rows = list(docs)
    if shuffle is not None:
        random.Random(shuffle).shuffle(rows)
    return documents.Workspace({'documents': rows}, 4)


def call(workspace, name, **arguments):
    return workspace.execute({'name': name, 'arguments': arguments})


# The design's failure tables, each with shuffled document order.

@pytest.mark.parametrize('seed', range(3))
def test_highest_approved_ignores_a_newer_draft(seed):
    w = ws(doc(1, revision=1, status='approved'), doc(2, revision=2, status='approved'),
           doc(3, revision=3, status='draft'), shuffle=seed)
    result = call(w, 'current_document', project='P', family='plan')
    assert result['outcome'] == 'known_current' and result['document']['id'] == 'doc-00000002'


def test_a_complete_family_with_nothing_approved_states_its_statuses():
    w = ws(doc(1, revision=1, status='superseded'), doc(2, revision=2, status='draft'))
    result = call(w, 'current_document', project='P', family='plan')
    assert result['outcome'] == 'known_no_approved'
    assert result['statuses'] == {'doc-00000001': [1, 'superseded'], 'doc-00000002': [2, 'draft']}


def test_two_approved_at_the_top_revision_are_ambiguous():
    w = ws(doc(1, revision=2, status='approved'), doc(2, revision=2, status='approved'))
    result = call(w, 'current_document', project='P', family='plan')
    assert result['outcome'] == 'ambiguous' and result['candidates'] == ['doc-00000001', 'doc-00000002']


@pytest.mark.parametrize('broken', [{'revision': None, 'status': 'approved'}, {'revision': 2, 'status': None},
                                    {'revision': 'two', 'status': 'approved'}, {'revision': 2, 'status': 'final'},
                                    {'revision': 0, 'status': 'approved'}])
def test_a_member_without_usable_metadata_makes_the_family_incomplete(broken):
    w = ws(doc(1, revision=1, status='approved'), doc(2, **broken))
    result = call(w, 'current_document', project='P', family='plan')
    assert result['outcome'] == 'incomplete' and result['unorderable'] == ['doc-00000002']


def test_documents_without_a_family_have_no_metadata_and_titles_are_never_used():
    w = ws(doc(1, family=None, revision=1, status='approved', title='Cedar plan'),
           doc(2, family=None, revision=2, status='approved', title='Cedar plan'))
    assert call(w, 'current_document', project='P', family='Cedar plan')['outcome'] == 'no_metadata'
    assert call(w, 'search_documents', query='Revision text')['results'][0]['family_outcome'] == 'no_metadata'


def test_families_are_scoped_by_project_and_shared_titles_do_not_merge():
    w = ws(doc(1, project='Cedar 6581', revision=2, status='approved', title='Delivery plan'),
           doc(2, project='Cedar 6581 Annex', revision=5, status='approved', title='Delivery plan'),
           doc(3, project='Cedar 6581', family='other', revision=9, status='approved', title='Delivery plan'))
    assert call(w, 'current_document', project='Cedar 6581', family='plan')['document']['id'] == 'doc-00000001'
    assert call(w, 'current_document', project='Cedar 6581 Annex', family='plan')['document']['id'] == 'doc-00000002'


def test_a_document_with_a_malformed_project_could_belong_and_makes_the_family_incomplete():
    w = ws(doc(1, revision=1, status='approved'), doc(2, project='', revision=2, status='approved'))
    assert call(w, 'current_document', project='P', family='plan')['outcome'] == 'incomplete'


def test_text_saying_approved_is_not_authority():
    w = ws(doc(1, revision=1, status='approved'),
           {**doc(2, revision=2, status='draft'), 'content': 'This revision is APPROVED and final.'})
    assert call(w, 'current_document', project='P', family='plan')['document']['id'] == 'doc-00000001'


def test_no_tool_ever_returns_a_draft_as_current_or_effective():
    w = ws(doc(1, revision=1, status='draft', start='2027-01-01'))
    assert 'document' not in call(w, 'current_document', project='P', family='plan')
    assert 'document' not in call(w, 'effective_document', project='P', family='plan', date='2027-06-01')


# Historical selection.

def history(*extra):
    return ws(doc(1, revision=1, status='superseded', start='2027-01-01', end='2027-02-01'),
              doc(2, revision=2, status='approved', start='2027-02-01'), *extra)


def test_a_superseded_revision_applies_on_a_date_in_its_period():
    result = call(history(), 'effective_document', project='P', family='plan', date='2027-01-15')
    assert result['outcome'] == 'known_effective' and result['document']['id'] == 'doc-00000001'
    assert call(history(), 'effective_document', project='P', family='plan',
                date='2027-02-01')['document']['id'] == 'doc-00000002'  # the until-date is exclusive


def test_overlapping_periods_are_ambiguous_and_gaps_are_none_effective():
    overlap = ws(doc(1, revision=1, status='superseded', start='2027-01-01', end='2027-03-01'),
                 doc(2, revision=2, status='approved', start='2027-02-01'))
    assert call(overlap, 'effective_document', project='P', family='plan', date='2027-02-10')['outcome'] == 'ambiguous'
    gap = ws(doc(1, revision=1, status='superseded', start='2027-01-01', end='2027-02-01'),
             doc(2, revision=2, status='approved', start='2027-03-01'))
    assert call(gap, 'effective_document', project='P', family='plan', date='2027-02-10')['outcome'] == 'none_effective'


@pytest.mark.parametrize('bad', [{'start': None, 'end': '2027-02-01'}, {'start': '2027-02-01', 'end': '2027-01-01'},
                                 {'start': '2027-01-01', 'end': '2027-01-01'}, {'start': 'January', 'end': None}])
def test_reversed_missing_or_malformed_periods_decline_the_date_question(bad):
    w = ws(doc(1, revision=1, status='superseded', **bad), doc(2, revision=2, status='approved', start='2027-02-01'))
    assert call(w, 'effective_document', project='P', family='plan', date='2027-03-01')['outcome'] == 'incomplete'


def test_in_effect_now_is_not_the_highest_approved_when_that_is_future_dated():
    w = ws(doc(2, revision=2, status='approved', start='2027-02-01', end='2027-06-01'),
           doc(3, revision=3, status='approved', start='2027-06-01'))
    assert call(w, 'current_document', project='P', family='plan')['document']['id'] == 'doc-00000003'
    assert call(w, 'effective_document', project='P', family='plan',
                date='2027-05-20')['document']['id'] == 'doc-00000002'


def test_an_invalid_date_argument_is_an_error_not_a_guess():
    assert 'error' in call(history(), 'effective_document', project='P', family='plan', date='May')


# The demo migration.

def demo(version, world=None):
    world = world or revision_fixtures.demo_v2()
    return documents.Workspace(interfaces.documents_world(world, version), version)


def test_the_enriched_demo_selects_cedar_revision_2_and_keeps_the_annex_separate():
    w = demo(4)
    cedar = call(w, 'current_document', project='Cedar 6581', family='delivery-plan')
    annex = call(w, 'current_document', project='Cedar 6581 Annex', family='delivery-plan')
    assert cedar['outcome'] == 'known_current' and cedar['document']['id'] == 'doc-88b36697f8'
    assert annex['document']['id'] == 'doc-35d32ed303'


def test_the_original_demo_is_no_metadata_under_v4_as_specified():
    w = demo(4, revision_fixtures.original())
    assert call(w, 'current_document', project='Cedar 6581', family='delivery-plan')['outcome'] == 'no_metadata'


def test_the_original_fixture_is_pinned_and_unchanged():
    assert revision_fixtures.roots()['original'] == revision_fixtures.ORIGINAL_ROOT


# Earlier interfaces and transcripts are unchanged.

def test_versions_1_to_3_reproduce_every_recorded_result_byte_for_byte():
    golden = json.loads(GOLDEN.read_text())
    for version in ('1', '2', '3'):
        w = documents.Workspace(golden['world'], int(version))
        results = [w.execute(c) for c in golden['calls']]
        assert results == golden['versions'][version]['results'], version
        assert w.snapshot() == golden['versions'][version]['snapshot'], version


def test_earlier_interfaces_declare_the_same_tools_and_ignore_enriched_fields():
    golden = json.loads(GOLDEN.read_text())
    for name in ('workspace-documents/2', 'workspace-documents/3'):
        assert interfaces.interface({'interface': name}).TOOLS == golden[name]['tools']
    enriched, original = revision_fixtures.demo_v2(), revision_fixtures.original()
    assert interfaces.documents_world(enriched) == interfaces.documents_world(original)  # /3 sees the same world


def test_v4_refuses_unknown_fields_and_v3_still_refuses_metadata():
    with pytest.raises(ValueError):
        documents.Workspace({'documents': [{**doc(1, revision=1, status='approved'), 'priority': 'high'}]}, 4)
    with pytest.raises(ValueError):
        documents.Workspace({'documents': [doc(1, revision=1, status='approved')]}, 3)


def test_v4_results_carry_metadata_and_the_family_outcome():
    w = demo(4)
    hit = next(r for r in call(w, 'search_documents', query='Cedar 6581 plan')['results']
               if r['id'] == 'doc-9e6d7ba44d')
    assert hit['metadata'] == {'project': 'Cedar 6581', 'family': 'delivery-plan', 'revision': 3, 'status': 'draft'}
    assert hit['family_outcome'] == 'known_current'
    read = call(w, 'read_document', document_id='doc-88b36697f8')
    assert read['metadata']['revision'] == 2 and read['content'].startswith('The plan for Cedar 6581')


def test_a_selected_document_can_be_cited_after_reading():
    w = demo(4)
    chosen = call(w, 'current_document', project='Cedar 6581', family='delivery-plan')['document']['id']
    call(w, 'read_document', document_id=chosen)
    assert call(w, 'submit_answer', answer='61 units', source_ids=[chosen],
                evidence=['The baseline quantity is 61 units']) == {'submitted': True}


def test_every_v4_result_survives_a_json_round_trip_so_transcripts_replay():
    w = demo(4)
    script = [('search_documents', {'query': 'Cedar 6581 plan'}),
              ('current_document', {'project': 'Cedar 6581', 'family': 'delivery-plan'}),
              ('effective_document', {'project': 'Cedar 6581', 'family': 'delivery-plan', 'date': '2027-05-20'}),
              ('read_document', {'document_id': 'doc-88b36697f8'})]
    calls = []
    for name, arguments in script:
        c = {'name': name, 'arguments': arguments}
        calls.append({'call': c, 'result': json.loads(json.dumps(w.execute(c)))})
    documents.replay_transcript(interfaces.documents_world(revision_fixtures.demo_v2(), 4), calls, version=4)


# Review of PR #124: one date syntax on every interpreter, and the /3 evidence gate kept for /4.

@pytest.mark.parametrize('value', ['20270101', '2027-W01-1', '2027-1-01', '２０２７-01-01', '2027-01-01T00:00',
                                   '2027-02-30', ' 2027-01-01'])
def test_only_yyyy_mm_dd_calendar_dates_are_accepted(value):
    meta = ws(doc(1, revision=1, status='approved', start=value))
    assert call(meta, 'current_document', project='P', family='plan')['outcome'] == 'incomplete'
    good = ws(doc(1, revision=1, status='approved', start='2027-01-01'))
    assert 'error' in call(good, 'effective_document', project='P', family='plan', date=value)


CROSS = '''
import json, sys
sys.path.insert(0, sys.argv[1])
from neuroshard.evolution import assistant_documents as d
docs = [{"id": "doc-00000001", "title": "Plan", "content": "x", "project": "P", "family": "plan", "revision": 1,
         "status": "approved", "effective_from": sys.argv[2]}]
w = d.Workspace({"documents": docs}, 4)
calls = [{"name": "current_document", "arguments": {"project": "P", "family": "plan"}},
         {"name": "effective_document", "arguments": {"project": "P", "family": "plan", "date": sys.argv[3]}}]
print(json.dumps([w.execute(c) for c in calls], sort_keys=True))
'''


@pytest.mark.parametrize('stored,asked', [('2027-01-01', '2027-01-15'), ('20270101', '2027-01-15'),
                                          ('2027-W01-1', '2027-01-15'), ('2027-01-01', '20270115'),
                                          ('2027-01-01', '2027-W02-5')])
def test_tool_results_are_identical_on_every_available_interpreter(stored, asked):
    import shutil
    import subprocess
    import sys

    src = str(Path(documents.__file__).resolve().parents[2])
    found = {sys.executable}
    for name in ('python3.10', 'python3.11', 'python3.12'):
        path = shutil.which(name)
        if path:
            found.add(path)
    outputs = {}
    for python in sorted(found):
        run = subprocess.run([python, '-c', CROSS, src, stored, asked], capture_output=True, text=True)
        if run.returncode == 0:
            outputs[python] = run.stdout
    assert outputs and len(set(outputs.values())) == 1, outputs


def test_v4_counts_only_a_submitted_evidenced_answer():
    w = ws(doc(1, revision=1, status='approved'))
    w.documents['doc-00000001']['content'] = 'The baseline quantity is 61 units.'
    expected = {'facts': ['61 units'], 'source_ids': ['doc-00000001']}
    call(w, 'search_documents', query='baseline quantity')
    assert documents.score_round(w.snapshot(), expected, True, '61 units (doc-00000001)', 4) is False
    call(w, 'current_document', project='P', family='plan')
    assert documents.score_round(w.snapshot(), expected, True, '61 units (doc-00000001)', 4) is False
    call(w, 'read_document', document_id='doc-00000001')
    rejected = call(w, 'submit_answer', answer='61 units', source_ids=['doc-00000001'], evidence=['not in the text'])
    assert 'error' in rejected
    assert documents.score_round(w.snapshot(), expected, True, '61 units (doc-00000001)', 4) is False
    assert call(w, 'submit_answer', answer='61 units', source_ids=['doc-00000001'],
                evidence=['The baseline quantity is 61 units']) == {'submitted': True}
    assert documents.score_round(w.snapshot(), expected, True, '61 units (doc-00000001)', 4) is True


def test_v4_demonstrations_are_refused_rather_than_silently_invalid():
    from neuroshard.evolution import assistant_documents_demonstration as demonstration

    case = {'id': 'synthetic', 'family': 'flight-time', 'split': 'train',
            'world': {'documents': [doc(1, revision=1, status='approved')]},
            'turns': [{'user': 'q', 'expected': {'facts': ['x'], 'source_ids': ['doc-00000001']}}]}
    with pytest.raises(ValueError, match='not defined for workspace-documents/4'):
        demonstration.texts(case, 4)


def test_v1_v2_v3_demonstrations_still_generate_and_only_v4_is_refused():
    """The /4 guard must not reject the earlier versions' demonstrations (review of 048f034)."""
    from neuroshard.evolution import assistant_documents_data as data
    from neuroshard.evolution import assistant_documents_demonstration as demonstration
    from neuroshard.evolution.modular_reference_execution import ROOT, read

    case = data.cases('train')[0]
    baseline = read(ROOT / 'config/experiments/assistant-general-policy-documents.json')
    assert baseline['interface'] == 'workspace-documents/1'
    assert demonstration.demonstration(case, baseline)['passed'] is True  # the repository's own /1 baseline
    for version in (1, 2, 3):
        assert demonstration.texts(case, version)
    with pytest.raises(ValueError, match='not defined for workspace-documents/4'):
        demonstration.texts(case, 4)


def label_doc(i, **extra):
    return {'id': f'doc-{i:010x}', 'project': 'P', 'family': 'plan', 'title': 'Plan', 'revision': i,
            'status': 'approved', 'content': f'The recorded amount is {i * 10}.', **extra}


def test_an_opt_in_labelled_view_keeps_every_document_and_labels_only_decided_families():
    """Soak finding (pilot-2-rc2): /2 hid revision metadata, so the model answered from revision 1 of the demo plan.
    Review of PR #143: nothing hidden; explicit identity only; earlier interfaces unchanged unless opted in."""
    original = revision_fixtures.original()
    view = interfaces.labelled_files(original, revision_fixtures.DEMO_V2_FAMILIES)
    assert [d['id'] for d in view['documents']] == [d['id'] for d in original['documents']]  # nothing hidden
    assert all(set(a) == set(b) for a, b in zip(view['documents'], original['documents']))  # same fields
    titles = {d['id']: d['title'] for d in view['documents']}
    assert titles['doc-88b36697f8'].endswith('(revision 2, approved; latest approved)')
    assert titles['doc-292283f7a9'].endswith('(revision 1, approved)')
    assert titles['doc-9e6d7ba44d'].endswith('(revision 3, draft)')
    assert titles['doc-35d32ed303'] == 'Cedar 6581 Annex delivery plan'  # a family of one: unlabelled
    ws = documents.Workspace(interfaces.documents_world(view, 2), 2)
    assert ws.search('What is the baseline quantity in the latest approved Cedar 6581 plan?')['results'][0]['id'] \
        == 'doc-88b36697f8'
    assert 'content' in ws.execute({'name': 'read_document', 'arguments': {'document_id': 'doc-292283f7a9'}})
    # Without the opt-in, earlier interfaces see exactly what they always saw.
    assert interfaces.documents_world(revision_fixtures.demo_v2(), 2) == interfaces.documents_world(original, 2)
    assert interfaces.labelled_files(original, {}) == original  # no families given: nothing changes


def test_no_label_without_explicit_identity_or_complete_metadata():
    def labels(rows):
        return interfaces.revision_labels(rows)
    no_family = [label_doc(1), label_doc(2)]
    for row in no_family:
        del row['family']
    assert labels(no_family) == {}  # same title and project, but no family: never grouped
    no_project = [label_doc(1), label_doc(2)]
    for row in no_project:
        del row['project']
    assert labels(no_project) == {}
    for field in ('status', 'revision'):
        rows = [label_doc(1), label_doc(2), label_doc(3)]
        del rows[-1][field]
        assert labels(rows) == {}, field  # an incomplete family is undecided, as /4 says
    assert labels([label_doc(1), label_doc(2), label_doc(3, project='')]) == {}  # malformed would-be member
    for field in ('project', 'family'):
        rows = [label_doc(1), label_doc(2, **{field: ['bad-type']})]
        interfaces.documents_world({'documents': rows}, 2)  # opens, as before
        assert labels(rows) == {}
    assert labels([label_doc(1, revision=2), label_doc(2)]) == {}  # a tie is ambiguous


def test_history_drafts_and_effective_dates_stay_readable():
    world = {'documents': [label_doc(1, status='superseded', effective_from='2027-01-01', effective_until='2027-06-01'),
                           label_doc(2, effective_from='2027-06-01'), label_doc(3, status='draft')]}
    view = interfaces.labelled_files(world, {})
    ws = documents.Workspace(interfaces.documents_world(view, 2), 2)
    for i in (1, 2, 3):
        assert 'content' in ws.execute({'name': 'read_document', 'arguments': {'document_id': f'doc-{i:010x}'}})
    w4 = documents.Workspace(interfaces.documents_world(world, 4), 4)
    effective = w4.execute({'name': 'effective_document',
                            'arguments': {'project': 'P', 'family': 'plan', 'date': '2027-05-20'}})
    assert effective['document']['id'] == 'doc-0000000001'


def test_explicit_families_beside_the_files_reach_only_the_documents_route():
    from neuroshard.assistant.public import Session
    from test_general_route_serving import POLICIES
    session = Session(POLICIES['drafting'], revision_fixtures.original(), name='a4-general',
                      modules=('U1', 'L2', 'L3', 'L7', 'C1'), route_policies=POLICIES,
                      document_families=revision_fixtures.DEMO_V2_FAMILIES)
    for route in POLICIES:  # drafting and scheduling accept exactly their own fields
        session.workspace(interfaces.interface(POLICIES[route]))
    titles = [d['title'] for d in session.worlds['documents'].documents.values()]
    assert any(t.endswith('latest approved)') for t in titles)
