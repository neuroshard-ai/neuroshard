"""Tool fixtures, runners, the report, rescoring, the sealed-split commitment, the judge protocol and the gate."""

import json
import secrets

import pytest

from neuroshard.evolution.general_eval import gate, items, judge, runner, served, stats, tools


def call(name, **arguments):
    return '<tool_call>\n' + json.dumps({'name': name, 'arguments': arguments}) + '\n</tool_call>'


def scripted(*replies):
    replies = iter(replies)
    return lambda messages, tool_schemas: next(replies)


def dev(item_id):
    return next(item for item in items.load_split() if item['id'] == item_id)


def test_the_development_split_is_120_fresh_items_20_per_area_and_matches_its_sources():
    split = items.load_split('development')
    assert len(split) == 120 and items.counts(split) == {area: 20 for area in items.AREAS}
    assert {item['provenance']['source'] for item in split} == {'fresh'}
    assert all('judged' in item for item in split if item['area'] == 'writing')
    tool_sets = [item['scoring']['toolset'] for item in split if item['area'] == 'tool_use']
    assert tool_sets.count('fixture/1') == 14 and tool_sets.count('documents/2') == 6
    public = items.public_view(split[0])
    assert set(public) == {'id', 'area', 'user'}


def test_fixture_tools_change_only_what_they_should():
    world = tools.FixtureWorld(dev('tool_use-dev-002')['fixture'])
    assert world.execute({'name': 'issue_refund', 'arguments': {'order_id': 'O-5002', 'amount_cents': 100,
                                                                'reason': 'x'}})['error'].startswith('order O-5002 is')
    assert world.execute({'name': 'issue_refund', 'arguments': {'order_id': 'O-5003', 'amount_cents': 10001,
                                                                'reason': 'x'}})['error']
    assert world.execute({'name': 'convert_currency', 'arguments': {'amount': 40, 'from_currency': 'GBP',
                                                                    'to_currency': 'EUR'}}) == {'amount': 46.51,
                                                                                                'currency': 'EUR'}
    assert 'overlaps' in world.execute({'name': 'create_event', 'arguments': {
        'title': 'Bank', 'date': '2026-11-04', 'start': '11:30', 'end': '12:00'}})['error']
    assert world.execute({'name': 'calculate', 'arguments': {'operation': 'divide', 'left': 1, 'right': 0}})['error']
    assert world.execute({'name': 'nope', 'arguments': {}})['error']
    assert world.changes == {'refunds': [], 'messages': [], 'events_created': []}
    assert len(world.execute({'name': 'find_customer', 'arguments': {'query': 'jordan lee'}})['customers']) == 2


def test_model_system_runs_the_tool_loop_and_scores_effects():
    item = dev('tool_use-dev-002')
    good = runner.ModelSystem(scripted(
        call('find_customer', query='jordan.lee@example.com'), call('list_orders', customer_id='C-101'),
        call('issue_refund', order_id='O-5001', amount_cents=4599, reason='arrived broken'),
        'Done: refunded 45.99 for the desk lamp.'), 'scripted')
    outcome = good.episode(item)
    assert runner.score(item, outcome)['passed']
    wrong = runner.ModelSystem(scripted(call('issue_refund', order_id='O-5004', amount_cents=899, reason='broken'),
                                        'Refunded.'), 'scripted')
    assert not runner.score(item, wrong.episode(item))['passed']  # refunded the other Jordan
    asks = runner.ModelSystem(scripted(call('find_contact', query='Priya'),
                                       'Which Priya do you mean: Priya Shah or Priya Nair?'), 'scripted')
    assert runner.score(dev('tool_use-dev-009'), asks.episode(dev('tool_use-dev-009')))['passed']
    guesses = runner.ModelSystem(scripted(call('send_message', contact_id='P-1', text='The report is ready.'),
                                          'Sent to Priya Shah (not Nair).'), 'scripted')
    assert not runner.score(dev('tool_use-dev-009'), guesses.episode(dev('tool_use-dev-009')))['passed']


def test_model_system_documents_items_use_the_served_documents_scorer():
    item = dev('tool_use-dev-018')  # the return flight does not exist
    reports = runner.ModelSystem(scripted(call('search_documents', query='Porto flight'),
                                          call('report_not_found', reason='only an outbound flight'),
                                          'There is no return flight in your documents.'), 'scripted')
    assert runner.score(item, reports.episode(item))['passed']
    guesses = runner.ModelSystem(scripted(call('read_document', document_id='doc-6f708192'),
                                          'It leaves at 07:25 (doc-6f708192).'), 'scripted')
    assert not runner.score(item, guesses.episode(item))['passed']


def test_non_terminating_or_budget_exhausting_episodes_fail():
    item = dev('reasoning-dev-001')
    system = runner.ModelSystem(scripted({'text': 'Answer: 215', 'terminated': False}), 'cut')
    assert not runner.score(item, system.episode(item))['passed']
    loop = runner.ModelSystem(lambda m, t: call('get_order', order_id='O-5002'), 'loop', max_steps=8, max_calls=5)
    outcome = loop.episode(dev('tool_use-dev-001'))
    assert outcome['failure'] == 'tool-call budget exhausted' and not runner.score(dev('tool_use-dev-001'), outcome)['passed']


def test_run_reports_counts_intervals_costs_and_rescores():
    split = items.load_split()
    chosen = [i for i in split if i['id'] in ('reasoning-dev-001', 'factual-dev-001', 'coding-dev-006',
                                              'writing-dev-011')]
    replies = {'reasoning-dev-001': 'Answer: 215', 'factual-dev-001': 'Answer: Sydney',
               'coding-dev-006': '```python\ndef second_largest(nums):\n    v = sorted(set(nums))\n'
                                 '    return v[-2] if len(v) > 1 else None\n```',
               'writing-dev-011': json.dumps({'destination': 'x', 'distance_km': 9, 'packing_list': [],
                                              'difficulty': 'easy'})}
    order = iter(i['id'] for i in chosen)
    system = runner.ModelSystem(lambda m, t: replies[next(order)], 'scripted')
    report, transcripts = runner.run(chosen, system, split='development', split_digest=items.split_digest(chosen),
                                     cost={'instance_type': 'local', 'usd_per_hour': 0.0})
    assert report['overall'] == {'passed': 3, 'cases': 4, 'macro_rate': 0.75}
    assert report['areas']['factual']['passed'] == 0 and report['areas']['coding']['passed'] == 1
    assert report['areas']['coding']['wilson95'][0] > 0.2 and report['cost']['compute_usd'] == 0.0
    assert report['baseline_claim'] is False and report['sandbox_isolation']
    assert runner.rescore(chosen, transcripts) == 3
    transcripts[0]['score']['passed'] = False
    with pytest.raises(ValueError):
        runner.rescore(chosen, transcripts)


def test_a_system_error_is_a_recorded_failure_not_a_crash():
    def broken(messages, tool_schemas):
        raise RuntimeError('model fell over')

    report, _ = runner.run([dev('reasoning-dev-002')], runner.ModelSystem(broken, 'broken'), split='development')
    assert report['areas']['reasoning']['errors'] == 1 and report['overall']['passed'] == 0


def test_served_system_routes_each_turn_through_the_a4_session():
    routes = []

    def route(text, opening):
        routes.append(text)
        return 'documents' if 'insurance' in text else 'chat'

    item = dev('tool_use-dev-015')
    replies = scripted(call('search_documents', query='car insurance excess'),
                       call('read_document', document_id='doc-2b3c4d5e'), 'The excess is 250 euros (doc-2b3c4d5e).')
    system = served.a4_general(route, lambda name: replies)
    outcome = system.episode(item)
    assert outcome['route'] == 'documents' and runner.score(item, outcome)['passed']
    seen = []

    def chat(messages, tool_schemas):
        seen.append((messages[0]['content'], [t['function']['name'] for t in tool_schemas]))
        return 'Answer: 36'

    outcome = served.a4_general(route, lambda name: chat).episode(dev('reasoning-dev-002'))
    assert runner.score(dev('reasoning-dev-002'), outcome)['passed']
    assert 'check_text' in seen[0][1] and 'calculate' in seen[0][1]  # the served chat route's own tools and prompt
    assert seen[0][0] == served.a4_policies()[0]['chat']['system_instruction']


def test_served_system_marks_caller_tools_unsupported_and_misroutes_as_failures():
    system = served.a4_general(lambda t, o: 'chat', lambda name: scripted('x'))
    outcome = system.episode(dev('tool_use-dev-001'))
    assert outcome['failure'].startswith('unsupported') and not runner.score(dev('tool_use-dev-001'), outcome)['passed']
    report, _ = runner.run([dev('tool_use-dev-001')], system, split='development')
    assert report['areas']['tool_use']['unsupported'] == 1
    misrouted = served.a4_general(lambda t, o: 'drafting', lambda name: scripted('ok'))
    outcome = misrouted.episode(dev('planning-dev-001'))
    assert outcome['failure'].startswith('routed to drafting') and not runner.score(dev('planning-dev-001'), outcome)['passed']
    on_chat = served.a4_general(lambda t, o: 'chat', lambda name: scripted('The excess is 250 (doc-2b3c4d5e).'))
    assert not runner.score(dev('tool_use-dev-015'), on_chat.episode(dev('tool_use-dev-015')))['passed']


def test_sealed_commitment_is_salted_and_detects_any_change():
    split = items.load_split()[:6]
    sealed = [{**item, 'split': 'sealed-test'} for item in split]
    salt = secrets.token_hex(32)
    published = items.commitment(sealed, salt, 'sealed-test')
    assert set(published) == {'format', 'split', 'counts', 'items', 'salted_sha256'}
    assert 'id' not in json.dumps(published['counts'])
    assert items.verify_commitment(sealed, salt, published)
    assert items.commitment(sealed, secrets.token_hex(32), 'sealed-test')['salted_sha256'] != published['salted_sha256']
    edited = json.loads(json.dumps(sealed))
    edited[0]['turns'][0]['user'] += ' '
    with pytest.raises(ValueError):
        items.verify_commitment(edited, salt, published)
    with pytest.raises(ValueError):
        items.commitment(sealed, 'short', 'sealed-test')
    with pytest.raises(ValueError):
        items.commitment(split, salt, 'sealed-test')  # items naming another split


def test_pairwise_judge_is_blind_order_balanced_and_scores_consistency():
    split = [i for i in items.load_split() if i['area'] == 'writing'][:4]
    candidate = {i['id']: 'CANDIDATE reply' for i in split}
    incumbent = {i['id']: 'incumbent reply' for i in split}
    prompts = []

    def prefers_candidate(text):
        prompts.append(text)
        first = text.split('[Response 1]\n')[1].split('\n\n[Response 2]')[0]
        return '1' if first.startswith('CANDIDATE') else '2'

    rows = judge.pairwise(split, candidate, incumbent, prefers_candidate, seed=1)
    assert [r['score'] for r in rows] == [1.0] * 4 and all(r['consistent'] for r in rows)
    assert len(prompts) == 8 and all('scripted' not in p and 'a4-general' not in p for p in prompts)
    first_always = judge.pairwise(split, candidate, incumbent, lambda text: '1', seed=1)
    assert [r['score'] for r in first_always] == [0.5] * 4 and not any(r['consistent'] for r in first_always)
    assert judge.summary(first_always)['position_consistency'] == 0.0
    with pytest.raises(ValueError):
        judge.verdict('Response 1 is better')
    cal = judge.calibration(['1', '2', 'tie', '1'], [['1', '1', '2'], ['2', '2'], ['tie', '1', 'tie'], ['2', '2', '1']])
    assert cal['agreement'] == 0.75 and round(cal['kappa'], 3) == 0.636 and cal['usable']
    assert not judge.calibration(['1', '1', '1', '1'], [['2'], ['2'], ['tie'], ['1']])['usable']


def declaration(**overrides):
    plan = json.load(open(items.DATA.parents[4] / 'config/experiments/general-capability-gate-template.json'))
    plan.update(split_commitment='c' * 64, bootstrap_samples=2000, judged=None,
                preservation={'chat': {'cases': 96, 'minimum_passed': 87}})
    plan.update(overrides)
    return plan


def rows(rates, n=60, seed=0):
    import random

    rng, out = random.Random(seed), []
    for area, rate in rates.items():
        for k in range(n):
            out.append({'id': f'{area}-{k}', 'area': area, 'family': f'f{k % 3}', 'passed': rng.random() < rate,
                        'unsupported': False, 'error': None})
    return out


def test_gate_promotes_a_clear_improvement_and_blocks_a_regression():
    base = {area: 0.6 for area in items.AREAS}
    incumbent = rows(base)
    better = [dict(r, passed=r['passed'] or int(r['id'].split('-')[-1]) % 4 == 0) for r in incumbent]
    ok = gate.decide(declaration(), better, incumbent, preservation={'chat': {'passed': 93}})
    assert ok['passed'] and ok['differences']['macro']['interval'][0] > 0
    hurt = [dict(r, passed=False) if r['area'] == 'coding' else r for r in better]
    blocked = gate.decide(declaration(), hurt, incumbent, preservation={'chat': {'passed': 93}})
    assert not blocked['passed'] and blocked['checks']['non_regression']['coding'] is False
    kept_narrow = gate.decide(declaration(), better, incumbent, preservation={'chat': {'passed': 80}})
    assert not kept_narrow['passed'] and kept_narrow['checks']['preservation']['chat'] is False
    with pytest.raises(ValueError):
        gate.decide(declaration(preservation={'a3': {'minimum_passed': None}}), better, incumbent)
    assert not gate.decide(declaration(), better[:-1], incumbent)['passed']


def test_power_numbers_quoted_in_the_design():
    table = stats.power_table()
    assert table['mcnemar_items_delta05'] == {'0.08': 249, '0.1': 312, '0.12': 375, '0.15': 469, '0.2': 626}
    assert table['unpaired_items_delta05_p05'] == 1570 and table['half_width_n400_p05'] == 0.049
    assert table['noninferiority_power_n400']['margin0.05_disc0.12'] == 0.893
    low, high = stats.wilson(10, 20)
    assert round(low, 3) == 0.299 and round(high, 3) == 0.701
    boot = stats.paired_bootstrap([(str(k), 'a') for k in range(10)], {str(k): 1.0 for k in range(10)},
                                  {str(k): 0.0 for k in range(10)}, samples=200)
    assert boot['difference'] == boot['lower'] == boot['upper'] == 1.0


def test_generic_tool_items_are_reported_beside_the_gate_not_in_it():
    plan = declaration(reported_separately={'toolsets': ['fixture/1']})
    base = {area: 0.6 for area in items.AREAS}
    incumbent = rows(base)
    for row in incumbent:
        if row['area'] == 'tool_use':
            row['toolset'] = 'fixture/1' if int(row['id'].split('-')[-1]) < 20 else 'documents/2'
    better = [dict(r, passed=r['passed'] or int(r['id'].split('-')[-1]) % 4 == 0) for r in incumbent]
    # every generic tool item lost: blocks the gate only if those items are inside it
    hurt = [dict(r, passed=False) if r.get('toolset') == 'fixture/1' else r for r in better]
    inside = gate.decide(declaration(reported_separately={}), hurt, incumbent, preservation={'chat': {'passed': 93}})
    beside = gate.decide(plan, hurt, incumbent, preservation={'chat': {'passed': 93}})
    assert not inside['checks']['non_regression']['tool_use']
    assert beside['passed'] and beside['reported_separately']['fixture/1']['candidate'] == 0
    assert beside['reported_separately']['fixture/1']['cases'] == 20
    template = json.load(open(items.DATA.parents[4] / 'config/experiments/general-capability-gate-template.json'))
    assert template['reported_separately']['toolsets'] == ['fixture/1'] and template['improvement']['rule'] == 'macro'
    assert template['sealed_reuse']['maximum_candidate_openings'] == 3 and template['custodian'] is None
    assert {a['margin'] for a in template['areas'].values()} == {0.05} and template['judged']['margin'] == 0.07
