import base64
import copy

import pytest

from neuroshard.dataflow.store import canonical
from neuroshard.demo import protocol
from neuroshard.evolution.provider_runtime import LocalNode
from neuroshard.evolution.provider_transport import Unavailable
from neuroshard.evolution.reference_data import identity
from neuroshard.evolution.transactions import Outbox
from test_transactions import Node


@pytest.fixture
def local_node():
    state = {'time': 0., 'chain_id': 'provider-test', 'height': 10, 'unavailable': False,
        'manifest': {'profile': 'test'}, 'epoch': 'b'*64, 'deadline': 100, 'catching_up': False}
    def rpc(url, method, params=None, **_kwargs):
        if state['unavailable']:
            raise OSError('local process unavailable')
        if method == 'status':
            return {'node_info': {'network': state['chain_id']},
                    'sync_info': {'catching_up': state['catching_up']}}
        if params['path'] == '/manifest':
            result = state['manifest']
        else:
            result = {'chain_id': state['chain_id'], 'height': state['height'], 'time_ns': 1,
                'job': {'id': 'a'*64}, 'result': None,
                'lease': {'assignment_root': state['epoch'], 'status': 'ready',
                          'work_deadline': state['deadline'], 'providers': {}}}
        return {'response': {'value': base64.b64encode(canonical(result)).decode(), 'height': state['height']}}
    node = LocalNode('http://127.0.0.1:26657', 'provider-test', identity(state['manifest']),
                     rpc=rpc, clock=lambda: state['time'])
    return state, node, rpc


def test_only_pinned_synchronized_local_validator_authorizes_frames(local_node):
    state, node, rpc = local_node
    with pytest.raises(ValueError, match='loopback'):
        LocalNode('https://remote.example', 'provider-test', identity(state['manifest']), rpc=rpc)
    with pytest.raises(ValueError, match='different genesis'):
        LocalNode(node.url, 'provider-test', 'f'*64, rpc=rpc)
    state['catching_up'] = True
    with pytest.raises(Unavailable, match='synchronizing'):
        LocalNode(node.url, 'provider-test', identity(state['manifest']), rpc=rpc)


def test_assignment_updates_stalls_and_outages_fail_closed(local_node):
    state, node, _ = local_node
    assert node.lookup('a'*64)['assignment_root'] == 'b'*64
    state.update(time=1., height=11, epoch='c'*64)
    assert node.lookup('a'*64)['assignment_root'] == 'c'*64
    state.update(time=2., unavailable=True)
    with pytest.raises(Unavailable, match='unavailable'):
        node.lookup('a'*64)
    state.update(unavailable=False, time=32.)
    with pytest.raises(Unavailable, match='stopped advancing'):
        node.lookup('a'*64)
    state.update(time=33., height=12)
    assert node.lookup('a'*64)['assignment_root'] == 'c'*64
    state.update(time=34., height=9)
    with pytest.raises(Unavailable, match='height changed'):
        node.lookup('a'*64)


def test_block_deadline_and_chain_change_revoke_transport(local_node):
    state, node, _ = local_node
    state.update(height=101)
    with pytest.raises(Unavailable, match='deadline elapsed'):
        node.lookup('a'*64)
    state.update(time=1., chain_id='other-chain')
    with pytest.raises(Unavailable, match='identity'):
        node.lookup('a'*64)


@pytest.mark.parametrize('kind', ['accept_hosted_job', 'respond_expert'])
def test_replaced_native_assignment_retires_unknown_submission_without_resigning(tmp_path, kind):
    node = Node()
    owner = protocol.Identity('provider-outbox')
    box = Outbox(tmp_path/'outbox.sqlite', 'local', 'provider-test', owner, rpc=node.rpc, query=node.query)
    fields = {'job_id': 'a'*64}
    if kind == 'accept_hosted_job':
        fields['assignment_root'] = 'b'*64
    else:
        fields['workers'] = {'0': owner.sign({'job_id': 'a'*64, 'assignment_root': 'b'*64})}
    with pytest.raises(TimeoutError):
        box.send('old-epoch', kind, timeout=.01, **fields)
    snapshot = {'chain_id': 'provider-test', 'height': 20, 'result': None,
        'job': {'id': 'a'*64, 'hosting': 'b'*64}, 'lease': {'assignment_root': 'b'*64}}
    assert box.retire_hosted(snapshot) is None
    wrong = copy.deepcopy(snapshot)
    wrong['chain_id'] = 'someone-else'
    with pytest.raises(ValueError, match='own committed chain'):
        box.retire_hosted(wrong)
    snapshot['job']['hosting'] = snapshot['lease']['assignment_root'] = 'c'*64
    evidence = box.retire_hosted(snapshot)
    assert evidence['transaction_outcome'].startswith('unknown')
    assert box.pending() is None and len(node.submissions) == 1 and node.queries == 1
    assert box.db.execute('SELECT envelope FROM operations').fetchone()[0] == node.submissions[0]
    box.close()
