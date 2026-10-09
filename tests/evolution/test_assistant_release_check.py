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
