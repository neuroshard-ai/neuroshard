import json

from neuroshard.evolution import assistant_documents as documents
from neuroshard.evolution import assistant_documents_data as docs
from neuroshard.evolution import assistant_documents_demonstration as demo
from neuroshard.evolution import assistant_general as general
from neuroshard.evolution.modular_reference_execution import ROOT, read

POLICY = read(ROOT / 'config/experiments/assistant-general-policy-documents4.json')


def world():
    return {'documents': [{'id': 'doc-00000001', 'title': 'Itinerary Lisbon',
                           'content': 'Current itinerary. Flight NA123 from Osaka to Lisbon on 2027-01-05, departing 14:10.'},
                          {'id': 'doc-00000002', 'title': 'Grocery list', 'content': 'Oats, lentils, lemons.'}]}


def submit(workspace, answer, ids, quotes):
    return workspace.execute({'name': 'submit_answer', 'arguments': {'answer': answer, 'source_ids': ids,
                                                                     'evidence': quotes}})


def test_version_three_accepts_only_quotes_found_in_cited_read_documents():
    workspace = documents.Workspace(world(), 3)
    assert 'error' in submit(workspace, '14:10', ['doc-00000001'], ['departing 14:10'])  # not read yet
    workspace.execute({'name': 'read_document', 'arguments': {'document_id': 'doc-00000001'}})
    refused = submit(workspace, '18:00', ['doc-00000001'], ['return flight departing 18:00'])
    assert refused['unquoted'] == ['return flight departing 18:00'] and workspace.answer is None
    assert submit(workspace, '14:10', ['doc-00000001'], ['Lisbon on 2027-01-05, departing 14:10.']) == {'submitted': True}
    assert workspace.answer['evidence'] == ['Lisbon on 2027-01-05, departing 14:10.']


def test_version_three_scores_only_a_submitted_answer_and_versions_one_two_are_unchanged():
    expected = {'facts': ['14:10'], 'source_ids': ['doc-00000001']}
    prose = documents.Workspace(world(), 3)
    prose.execute({'name': 'read_document', 'arguments': {'document_id': 'doc-00000001'}})
    snapshot = prose.snapshot()
    assert not documents.score_round(snapshot, expected, True, '14:10 (doc-00000001)', 3)
    assert documents.score_round(snapshot, expected, True, '14:10 (doc-00000001)', 2)
    assert documents.Workspace(world(), 2).registry is documents.REGISTRY
    assert 'evidence' not in documents.REGISTRY['submit_answer']['properties']


def test_quotes_compare_as_normalised_words():
    assert documents.quoted('departing 14:10', 'Flight X, departing 14:10.')
    assert documents.quoted('amount due 1,250', 'Amount due: 1250. Status: unpaid.')
    assert not documents.quoted('departing 14:15', 'departing 14:10')
    assert not documents.quoted('return flight departing', 'Flight X, departing 14:10.')
    assert not documents.quoted('...', 'anything')


def test_every_version_three_demonstration_quotes_and_passes():
    splits = ['generated-train', 'generated-train-extra', 'generated-train-styles']
    rows = demo.all_demonstrations(POLICY, splits)
    assert len(rows) == 1080 and all(row['passed'] for _, row in rows)
    for case, row in rows:
        snapshot = row['rounds'][-1]['snapshot']
        if case['turns'][0]['expected'].get('not_found'):
            assert snapshot['not_found'] and snapshot['answer'] is None
        else:
            assert snapshot['answer']['evidence'] and all(
                any(documents.quoted(q, d['content']) for d in case['world']['documents']
                    if d['id'] in snapshot['answer']['source_ids']) for q in snapshot['answer']['evidence'])
    item = demo.trajectory(*rows[0], POLICY)
    assert any('evidence' in m['content'] for m in item['messages'] if m['role'] == 'assistant')


def test_episode_offers_version_three_tools():
    seen = []

    def respond(messages, tools):
        seen.append(tools)
        return {'text': 'No.', 'terminated': True}

    case = docs.cases('generated-development3')[0]
    general.documents_episode(case, respond, POLICY)
    submit_tool = next(t for t in seen[0] if t['function']['name'] == 'submit_answer')
    assert 'evidence' in submit_tool['function']['parameters']['properties']


def test_round_three_sets_are_new_and_confirmation5_is_disjoint():
    from neuroshard.evolution import assistant_phrasings as phrasings

    norm = lambda t: ' '.join(t.lower().split())  # noqa: E731
    earlier = [read(ROOT / phrasings.FILE)['sets'], read(ROOT / phrasings.FILE2)['sets']]
    known = {norm(t) for sets in earlier for by in sets.values() for rows in by.values() for ts in rows.values()
             for t in ts}
    fresh = read(ROOT / phrasings.FILE3)['sets']
    for route in fresh.values():
        for rows in route.values():
            assert all(norm(t) not in known for ts in rows.values() for t in ts)
    trainable = [t for sets in earlier for by in sets.values() for rows in by.values() for ts in rows.values() for t in ts]
    trainable += [t for by in fresh.values() for ts in by['development3'].values() for t in ts]
    seen = set().union(*(phrasings.spans(t, 8) for t in trainable))
    assert all(not phrasings.spans(t, 8) & seen for by in fresh.values() for ts in by['confirmation5'].values()
               for t in ts)
    styles = {c['turns'][0]['user'] for c in docs.cases('generated-train-styles')}
    assert len(styles) == 330
