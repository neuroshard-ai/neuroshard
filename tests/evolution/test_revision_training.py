"""The /4 revision training corpus: golds agree with selection, bad items are refused, and ``verify`` rechecks.

No network and no model: every check is a replay through the real v4 workspace and episode loop."""

import copy
import hashlib
import json
from pathlib import Path

import pytest

from neuroshard.evolution import assistant_documents as documents
from neuroshard.evolution import revision_training as rt

ROOT = Path(__file__).resolve().parents[2]
POLICY = json.loads((ROOT / 'config/experiments/assistant-general-policy-documents7.json').read_text())
P, F, A = 'Bakery wholesale', 'price-list', 'allergens'


def doc(key, title, content, **fields):
    return {'id': key, 'title': title, 'content': content, **fields}


WORLD = {'documents': [
    doc('doc-a1000001', 'Wholesale price list', 'Sourdough loaf wholesale price is £2.10 per unit. Minimum order is 20 '
        'loaves.', project=P, family=F, revision=1, status='superseded'),
    doc('doc-a1000002', 'Wholesale price list', 'Sourdough loaf wholesale price is £2.40 per unit. Minimum order is 25 '
        'loaves.', project=P, family=F, revision=2, status='approved'),
    doc('doc-a1000003', 'Wholesale price list', 'FINAL, approved by Jo. Sourdough loaf wholesale price is £2.60 per '
        'unit. Minimum order is 30 loaves.', project=P, family=F, revision=3, status='draft'),
    doc('doc-b2000001', 'Allergen sheet', 'Allergen sheet for the rye bread: contains sesame seeds.', project=P,
        family=A, revision=1, status='superseded', effective_from='2027-01-01', effective_until='2027-03-01'),
    doc('doc-b2000002', 'Allergen sheet', 'Allergen sheet for the rye bread: contains walnuts, no sesame.', project=P,
        family=A, revision=2, status='approved', effective_from='2027-03-01'),
    doc('doc-c3000001', 'Café menu', 'Café menu draft one: flat white 3.20.', project='Café', family='menu',
        revision=1, status='draft'),
    doc('doc-c3000002', 'Café menu', 'Café menu draft two: flat white 3.40.', project='Café', family='menu',
        revision=2, status='draft'),
    doc('doc-d4000001', 'Delivery terms', 'Delivery terms: free delivery over 40 loaves.', project='Shop',
        family='terms', revision=2, status='approved'),
    doc('doc-d4000002', 'Delivery terms', 'Delivery terms: free delivery over 50 loaves.', project='Shop',
        family='terms', revision=2, status='approved'),
    doc('doc-e5000001', 'Staff rota', 'Staff rota, weekend shifts: Sam opens on Saturdays.', project='Shop',
        family='rota', revision=1),
    doc('doc-e5000002', 'Staff rota', 'Staff rota, weekend shifts: Priya opens on Saturdays.', project='Shop',
        family='rota', revision=2, status='approved'),
    doc('doc-f6000001', 'Oven service booking', 'Oven service booked with Hearth Repairs for 14 March 2027; the '
        'callout fee is £180.'),
    doc('doc-f6000002', 'Flour order', 'Flour order from Millstone: 12 sacks at 21 pounds each, delivered Tuesday.'),
]}

ITEMS = {
    'latest': {'user': 'whats the sourdough price in the latest approved price list?', 'type': 'latest',
               'search': 'sourdough wholesale price', 'project': P, 'family': F, 'outcome': 'known_current',
               'facts': ['£2.40'], 'source_ids': ['doc-a1000002'],
               'answer': 'Revision 2, the latest approved price list, has sourdough at £2.40 per unit.',
               'quotes': ['Sourdough loaf wholesale price is £2.40 per unit']},
    'draft': {'user': 'what does the draft price list say for sourdough?', 'type': 'draft',
              'search': 'sourdough wholesale price', 'project': P, 'family': F, 'revisions': [3],
              'facts': ['£2.60'], 'source_ids': ['doc-a1000003'],
              'answer': 'The draft (revision 3, not approved) lists sourdough at £2.60 per unit.'},
    'named': {'user': 'what was the minimum order in version 1 of the price list?', 'type': 'named',
              'search': 'price list minimum order', 'project': P, 'family': F, 'revisions': [1],
              'facts': ['20 loaves'], 'source_ids': ['doc-a1000001'],
              'answer': 'Revision 1 set the minimum order at 20 loaves.'},
    'compare': {'user': 'how did the sourdough price change from rev 1 to rev 2?', 'type': 'compare',
                'search': 'sourdough wholesale price', 'project': P, 'family': F, 'revisions': [1, 2],
                'facts': ['£2.10', '£2.40'], 'source_ids': ['doc-a1000001', 'doc-a1000002'],
                'answer': 'It went from £2.10 in revision 1 to £2.40 in revision 2.'},
    'as_of': {'user': 'what did the rye allergen sheet say on 2027-02-10?', 'type': 'as_of',
              'search': 'rye allergen sheet', 'project': P, 'family': A, 'date': '2027-02-10',
              'outcome': 'known_effective', 'facts': ['sesame seeds'], 'source_ids': ['doc-b2000001'],
              'answer': 'On 2027-02-10 revision 1 applied: the rye bread contains sesame seeds.'},
    'now': {'user': "today's 2 April 2027 - what allergen does the rye bread have now?", 'type': 'now',
            'search': 'rye allergen sheet', 'project': P, 'family': A, 'date': '2027-04-02',
            'outcome': 'known_effective', 'facts': ['walnuts'], 'source_ids': ['doc-b2000002'],
            'answer': 'Revision 2 is in effect: the rye bread contains walnuts.'},
    'no_approved': {'user': 'flat white price on the latest approved cafe menu?', 'type': 'latest',
                    'search': 'café menu flat white', 'project': 'Café', 'family': 'menu',
                    'outcome': 'known_no_approved'},
    'ambiguous': {'user': 'latest approved delivery terms - free delivery threshold?', 'type': 'latest',
                  'search': 'free delivery terms', 'project': 'Shop', 'family': 'terms', 'outcome': 'ambiguous'},
    'incomplete': {'user': 'who opens saturdays per the latest approved rota', 'type': 'latest',
                   'search': 'staff rota saturdays', 'project': 'Shop', 'family': 'rota', 'outcome': 'incomplete'},
    'none': {'user': 'what did the allergen sheet say on 2026-12-01?', 'type': 'as_of', 'search': 'rye allergen sheet',
             'project': P, 'family': A, 'date': '2026-12-01', 'outcome': 'none_effective'},
    'lookup': {'user': 'how much is the oven callout?', 'type': 'lookup', 'search': 'oven service callout fee',
               'facts': ['£180'], 'source_ids': ['doc-f6000001'], 'answer': 'The oven callout fee is £180.'},
    'computed': {'user': 'what does the whole flour order cost?', 'type': 'computed', 'search': 'flour order sacks',
                 'facts': ['252'], 'source_ids': ['doc-f6000002'], 'answer': 'The flour order costs 252 pounds.',
                 'quotes': ['12 sacks at 21 pounds each'],
                 'steps': [{'tool': 'calculate', 'operation': 'multiply', 'left': 12, 'right': 21}]},
    'not_found': {'user': 'when is the oven warranty up?', 'type': 'not_found', 'search': 'oven service warranty',
                  'reason': 'The oven document is a service booking with a callout fee; it gives no warranty date.'},
}


def make(name, item=None, world=WORLD, nometa=False):
    item = copy.deepcopy(item or ITEMS[name])
    expected, record = rt.gold(world, item, nometa)
    texts = rt.replies(world, item, record)
    case = {'id': f'revtrain-test-{name}', 'split': rt.SPLIT, 'world': world, 'family': rt.family_of(record, expected),
            'gold': record, 'turns': [{'user': item['user'], 'expected': expected}], 'replies': texts}
    return case


def corpus(cases):
    return {'format': rt.FORMAT, 'cases': cases,
            'sha256': hashlib.sha256(json.dumps(cases, sort_keys=True).encode()).hexdigest()}


@pytest.mark.parametrize('name', sorted(ITEMS))
def test_every_kind_replays_and_passes_the_v4_scorer(name):
    case = make(name)
    row = rt.episode(case, case['replies'], POLICY)
    assert row is not None and row['version'] == 4


def test_latest_gold_is_the_selection_not_the_draft():
    case = make('latest')
    names = [json.loads(r.split('\n')[1])['name'] for r in case['replies'][:-1]]
    assert names == ['search_documents', 'current_document', 'read_document', 'submit_answer']
    assert case['turns'][0]['expected']['source_ids'] == ['doc-a1000002']
    assert 'doc-a1000003' not in case['replies'][-1]
    bad = dict(ITEMS['latest'], source_ids=['doc-a1000003'], facts=['£2.60'],
               answer='Revision 3 has sourdough at £2.60 per unit.')
    with pytest.raises(rt.Reject, match='selection'):
        rt.gold(WORLD, bad)
    with pytest.raises(rt.Reject, match='selection'):
        rt.gold(WORLD, dict(ITEMS['latest'], outcome='ambiguous'))


def test_effective_dates_select_by_interval():
    assert make('as_of')['turns'][0]['expected']['source_ids'] == ['doc-b2000001']
    assert make('now')['turns'][0]['expected']['source_ids'] == ['doc-b2000002']
    with pytest.raises(rt.Reject, match='date'):  # "now" must state the date it means
        rt.gold(WORLD, dict(ITEMS['now'], user='what allergen does the rye bread have now?'))


@pytest.mark.parametrize('name,outcome', [('no_approved', 'known_no_approved'), ('ambiguous', 'ambiguous'),
                                          ('incomplete', 'incomplete'), ('none', 'none_effective')])
def test_declines_report_the_tools_outcome_without_an_answer(name, outcome):
    case = make(name)
    assert case['turns'][0]['expected'] == {'not_found': True} and case['gold']['outcome'] == outcome
    row = rt.episode(case, case['replies'], POLICY)
    snapshot = row['rounds'][0]['snapshot']
    assert snapshot['answer'] is None and snapshot['not_found']['reason'] == case['gold']['reason']
    for key in case['gold']['members']:
        assert key in case['replies'][-1]
    assert not any('submit_answer' in r for r in case['replies'])


def test_decline_reasons_state_statuses_and_ids():
    assert make('no_approved')['gold']['reason'] == (
        'No revision of "menu" (Café) is approved: revision 1 (doc-c3000001) is a draft, revision 2 (doc-c3000002) '
        'is a draft.')
    assert 'doc-d4000001 and doc-d4000002' in make('ambiguous')['gold']['reason']
    assert 'doc-e5000001' in make('incomplete')['gold']['reason']


def test_named_and_draft_read_exactly_the_named_documents():
    assert make('named')['turns'][0]['expected']['source_ids'] == ['doc-a1000001']
    assert make('compare')['turns'][0]['expected']['source_ids'] == ['doc-a1000001', 'doc-a1000002']
    with pytest.raises(rt.Reject, match='family'):  # revision 2 is approved, not a draft
        rt.gold(WORLD, dict(ITEMS['draft'], revisions=[2], source_ids=['doc-a1000002']))
    with pytest.raises(rt.Reject, match='family'):  # no revision 5
        rt.gold(WORLD, dict(ITEMS['named'], revisions=[5]))
    with pytest.raises(rt.Reject, match='answer'):  # a draft answer must say it is a draft
        rt.gold(WORLD, dict(ITEMS['draft'], answer='Sourdough is £2.60 per unit.'))


def test_bad_items_are_refused():
    with pytest.raises(rt.Reject, match='facts'):  # fact not in the cited document
        rt.gold(WORLD, dict(ITEMS['lookup'], facts=['£190'], answer='The oven callout fee is £190.'))
    with pytest.raises(rt.Reject, match='answer'):  # the answer must state the fact
        rt.gold(WORLD, dict(ITEMS['lookup'], answer='The oven callout fee is listed in the booking.'))
    with pytest.raises(rt.Reject, match='family'):  # a plain lookup may not cite one revision of a family
        rt.gold(WORLD, dict(ITEMS['lookup'], facts=['£2.40'], source_ids=['doc-a1000002'], answer='It is £2.40.'))
    item = dict(ITEMS['computed'], steps=[{'tool': 'calculate', 'operation': 'add', 'left': 12, 'right': 21}])
    expected, record = rt.gold(WORLD, item)
    with pytest.raises(rt.Reject, match='replay'):  # the steps do not produce the fact
        rt.replies(WORLD, item, record)
    item = dict(ITEMS['computed'], quotes=['twelve sacks of flour'])
    expected, record = rt.gold(WORLD, item)
    with pytest.raises(rt.Reject, match='evidence'):  # a computed answer needs quotes found in the document
        rt.replies(WORLD, item, record)
    item = dict(ITEMS['lookup'], search='zzz unknownword')
    expected, record = rt.gold(WORLD, item)
    with pytest.raises(rt.Reject, match='replay'):
        rt.replies(WORLD, item, record)


def test_invalid_writer_quotes_fall_back_to_the_stating_sentence():
    item = dict(ITEMS['lookup'], quotes=['the fee is one hundred and eighty'])
    expected, record = rt.gold(WORLD, item)
    texts = rt.replies(WORLD, item, record)
    submit = documents.parse_calls(texts[-2], 4)[0]['arguments']
    assert submit['evidence'] == ['the callout fee is £180']


def test_scorer_refuses_a_conversation_that_cites_the_draft():
    case = make('latest')
    wrong = list(case['replies'])
    wrong[2] = rt.call('read_document', document_id='doc-a1000003')
    wrong[3] = rt.call('submit_answer', answer='Revision 2, the latest approved price list, has sourdough at £2.40.',
                       source_ids=['doc-a1000003'], evidence=['Sourdough loaf wholesale price is £2.60 per unit'])
    assert rt.episode(case, wrong, POLICY) is None


def test_verify_recomputes_and_catches_tampering():
    cases = [make(name) for name in sorted(ITEMS)]
    ok, problems = rt.verify(corpus(cases), POLICY)
    assert ok, problems
    stale = copy.deepcopy(cases)
    index = next(i for i, c in enumerate(stale) if c['family'] == 'rev-latest')
    stale[index]['turns'][0]['expected'] = {'facts': ['£2.60'], 'source_ids': ['doc-a1000003']}
    assert not rt.verify(corpus(stale), POLICY)[0]
    reason = copy.deepcopy(cases)
    index = next(i for i, c in enumerate(reason) if c['family'] == 'rev-decline-ambiguous')
    reason[index]['gold']['reason'] = 'Revision 2 is approved.'
    assert not rt.verify(corpus(reason), POLICY)[0]
    claim = copy.deepcopy(cases)
    index = next(i for i, c in enumerate(claim) if c['family'] == 'rev-draft')
    call = documents.parse_calls(claim[index]['replies'][-2], 4)[0]['arguments']
    claim[index]['replies'][-2] = rt.call('submit_answer', answer=call['answer'] + ' Revision 2 is approved.',
                                          source_ids=call['source_ids'], evidence=call['evidence'])
    claim[index]['replies'][-1] = call['answer'] + ' Revision 2 is approved. (Sources: doc-a1000003)'
    assert not rt.verify(corpus(claim), POLICY)[0]
    tampered = corpus(cases)
    tampered['cases'][0]['turns'][0]['user'] += '!'
    assert not rt.verify(tampered, POLICY)[0]


def test_no_metadata_latest_answers_from_text_and_states_the_limit():
    world = {'documents': [{k: d[k] for k in ('id', 'title', 'content')} for d in WORLD['documents'][-2:]]}
    item = dict(ITEMS['lookup'], type='nometa_latest', user='callout fee in the latest approved oven booking?')
    expected, record = rt.gold(world, item, nometa=True)
    case = make('nm', item, world, nometa=True)
    assert rt.NO_METADATA_NOTE in case['replies'][-1]
    assert rt.episode(case, case['replies'], POLICY) is not None
    with pytest.raises(rt.Reject, match='type'):
        rt.gold(WORLD, ITEMS['latest'], nometa=True)


def test_g2_conversation_rerenders_with_quotes():
    world = {'documents': [{k: d[k] for k in ('id', 'title', 'content')} for d in WORLD['documents'][-2:]]}
    case = {'id': 'realtrain-000-0', 'split': 'realistic-train', 'family': 'real-lookup', 'world': world,
            'turns': [{'user': 'oven callout fee?', 'expected': {'facts': ['£180'], 'source_ids': ['doc-f6000001']}}],
            'replies': [rt.call('search_documents', query='oven callout fee'),
                        rt.call('read_document', document_id='doc-f6000001'),
                        rt.call('submit_answer', answer='It is £180.', source_ids=['doc-f6000001']),
                        'It is £180. (Sources: doc-f6000001)']}
    out = rt.rerender(case, POLICY)
    assert out['split'] == 'realistic-train-v4' and out['rerendered_from'] == case['id']
    assert documents.parse_calls(out['replies'][2], 4)[0]['arguments']['evidence'] == ['the callout fee is £180']
    assert rt.verify(corpus([out]), POLICY)[0]


def test_public_world_keeps_fields_as_written():
    written = {'documents': [{'id': 'doc-00000001', 'title': 't', 'content': 'c', 'revision': '2', 'status': 'final',
                              'author': 'x'}]}
    world = rt.public_world(written)
    assert world['documents'][0] == {'id': 'doc-00000001', 'title': 't', 'content': 'c', 'revision': 2,
                                     'status': 'final'}
    assert documents.Workspace(world, 4).documents['doc-00000001']['problems'] == ['status']


def test_answers_claim_only_what_the_fields_record():
    with pytest.raises(rt.Reject, match='answer'):  # names revision 2 while citing only the draft
        rt.gold(WORLD, dict(ITEMS['draft'], answer='The draft (revision 3) lists £2.60, up from revision 2.'))
    world = {'documents': [{k: d[k] for k in ('id', 'title', 'content')} for d in WORLD['documents'][-2:]]}
    item = dict(ITEMS['lookup'], type='nometa_latest', user='callout fee in the latest approved oven booking?')
    with pytest.raises(rt.Reject, match='answer'):  # no approval claim can come from text
        rt.gold(world, dict(item, answer='The approved booking has a callout fee of £180.'), nometa=True)
    with pytest.raises(rt.Reject, match='answer'):
        rt.gold(world, dict(item, answer='Version 2 of the booking has a callout fee of £180.'), nometa=True)
