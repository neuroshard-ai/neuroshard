import copy

from neuroshard.assistant.release_check import projection


def test_release_projection_ignores_cost_but_preserves_all_served_decisions():
    row = {'id': 'episode', 'a2_selected': 'arm', 'calls': [], 'rounds': [
        {'route': 'drafting', 'completed': True, 'final_text': 'done', 'failure': None,
         'snapshot': {'drafts': {}}, 'call_count': 0, 'generation_count': 1, 'seconds': 2}],
        'generations': [{'input_token_ids': [1, 2], 'token_ids': [3], 'text': 'done',
                         'terminated': True, 'request_sha256': 'ab', 'seconds': 1}]}
    other = copy.deepcopy(row)
    other['rounds'][0]['seconds'] = 10
    other['generations'][0]['seconds'] = 5
    assert projection(row) == projection(other)
    other['generations'][0]['token_ids'] = [4]
    assert projection(row) != projection(other)
    other = copy.deepcopy(row)
    other['rounds'][0]['route'] = 'scheduling'
    assert projection(row) != projection(other)



def test_the_attempt_two_amendment_keeps_the_workload_and_pins_every_source():
    from neuroshard.assistant import release_check
    from neuroshard.evolution.modular_reference_execution import ROOT, read, sha256

    plan = read(ROOT / release_check.CONTRACT)
    assert plan['attempt'] == 2 and plan['amendments'][-1]['attempt'] == 2
    assert plan['worker_seconds'] == 21 * 3600 and plan['budget_usd'] == 50
    assert {name: spec['cases'] for name, spec in plan['sets'].items()} == {'cross': 48, 'drafting': 192,
                                                                            'scheduling': 192}
    assert not plan['training_authorized'] and not plan['gpu_launch_authorized'] and not plan['new_final_authorized']
    for name, digest in {**plan['sources'], **plan['contracts']}.items():
        assert sha256(ROOT / name) == digest, name
    assert len(release_check.checked_workload(plan)) == 432


def test_a_stopped_client_records_a_result_instead_of_dying_silently(tmp_path, monkeypatch):
    import json
    import threading

    from neuroshard.assistant import network, release_check

    monkeypatch.setattr(release_check, 'checked_workload', lambda plan: [('drafting', {'id': 'c'}, {})])
    monkeypatch.setattr(network, 'Account', lambda path: object())
    monkeypatch.setattr(network, 'Chain', lambda rpc, chain_id: object())
    monkeypatch.setattr(network, 'fetch_stage', lambda descriptor, rank, home: (None, None))
    stop = threading.Event()
    stop.set()
    code = release_check.client({'worker_seconds': 60, 'attempt': 2}, {'rpc': '', 'chain_id': ''}, tmp_path,
                                {'commit': 'x'}, stop)
    result = json.loads((tmp_path / 'result.json').read_text())
    assert code == 1 and not result['passed'] and result['completed'] == 0 and result['attempt'] == 2
    assert result['error'].startswith('InterruptedError')
