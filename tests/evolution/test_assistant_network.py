import socket
import threading

import pytest

from neuroshard.assistant import network
from neuroshard.evolution import assistant_workflow_data as data
from neuroshard.evolution import granite_tokenizer
from neuroshard.inference import optimistic_network as chain_network

from test_granite_serving import BOUNDARIES, SPEC, bounded, world  # noqa: F401
from test_granite_tokenizer import granite_like  # noqa: F401

PARAMS = {**network.PARAMS, 'challenge_blocks': 3, 'proof_blocks': 3, 'job_blocks': 900}


def engine():
    try:
        return chain_network.engine_path()
    except (ValueError, OSError):
        return None


def free_ports(count):
    """``count`` consecutive free local ports."""
    for _ in range(100):
        with socket.socket() as probe:
            probe.bind(('127.0.0.1', 0))
            base = probe.getsockname()[1]
        if base + count > 65535:
            continue
        try:
            held = []
            for port in range(base, base + count):
                held.append(socket.create_server(('127.0.0.1', port)))
            return base
        except OSError:
            continue
        finally:
            for sock in held:
                sock.close()
    raise RuntimeError('no free consecutive ports')


def stage(world, rank):  # noqa: F811
    value = {'config': world['config'], 'shard': world['shards']}
    if rank == 0:
        value.update(tokenizer=world['directory'], gate=world['gate'],
                     parent_tokenizer_digest=granite_tokenizer.blob_sha1(world['directory'] / 'tokenizer.json'))
    if rank == 2:
        value['arm'] = world['arm']
    return value


def test_the_served_version_is_named_by_its_plan_and_released_files():
    descriptor = {'format': 'neuroshard-assistant-network/1', 'version': 'a2-u1', 'plan_sha256': 'aa' * 32,
                  'files': {'integration.json': {'sha256': 'bb' * 32}}}
    root = network.model_root(descriptor)
    assert root == network.model_root(dict(descriptor)) and len(root) == 64
    assert network.model_root({**descriptor, 'files': {'integration.json': {'sha256': 'cc' * 32}}}) != root
    assert network.model_root({**descriptor, 'version': 'other'}) != root


def test_a_user_picks_one_reachable_active_owner_per_shard():
    served = network.Version('tiny', BOUNDARIES, SPEC, bounded(), [0], 64, 'ab' * 32)
    state = {'owners': {'k1': {'shard': 1, 'status': 'active', 'endpoint': 'down.example:1'},
                        'k2': {'shard': 1, 'status': 'active', 'endpoint': 'up.example:1'},
                        'k3': {'shard': 1, 'status': 'leaving', 'endpoint': 'up.example:2'},
                        'k4': {'shard': 2, 'status': 'active', 'endpoint': 'up.example:3'},
                        'k5': {'shard': 2, 'status': 'active'}}}
    probe = lambda endpoint: endpoint.startswith('up.')  # noqa: E731
    assert network.choose_owners(state, served, probe=probe) == [('k2', 'up.example:1'), ('k4', 'up.example:3')]
    del state['owners']['k4']
    with pytest.raises(ValueError, match='needs a peer for it'):
        network.choose_owners(state, served, probe=probe)


def test_the_host_command_sets_the_pinned_numerical_environment_before_torch_loads(tmp_path):
    code = '''
import sys
from neuroshard.client import cli
from neuroshard.evolution import granite_shard_execution as runtime
seen, configure = [], runtime.configure

def recorded():
    seen.append("torch" in sys.modules)
    configure()
    from neuroshard.assistant import network
    network.fetch_stage = lambda *args, **kwargs: (print("ordered" if seen == [False] else "torch-before-configure"),
                                                   sys.exit(0))

runtime.configure = recorded
cli.main(["assistant", "host", "--shard", "1", "--endpoint", "127.0.0.1:1", "--home", sys.argv[1]])
'''
    import os
    import subprocess
    import sys

    from test_granite_serving import ROOT

    result = subprocess.run([sys.executable, '-c', code, str(tmp_path)], capture_output=True, text=True,
                            env={**os.environ, 'PYTHONPATH': str(ROOT / 'src')}, timeout=120)
    assert result.stdout.strip() == 'ordered', result.stdout + result.stderr


@pytest.mark.skipif(engine() is None, reason='needs CometBFT 0.38.26')
def test_owners_on_the_network_serve_a_paid_conversation_that_settles(world, tmp_path):  # noqa: F811
    served = network.Version('tiny', BOUNDARIES, SPEC, bounded(), [world['tokenizer'].eos_token_id], 100000, 'ab' * 32)
    base, faucet_port = free_ports(3), free_ports(1)
    config, chain, faucet = network.seed({'chain_id': 'neuroshard-assistant-test'}, tmp_path / 'seed', '127.0.0.1',
                                         served=served, block_seconds=0.3, params=PARAMS, p2p_port=base,
                                         faucet_port=faucet_port, progress=lambda *_: None)
    faucet_url, stop, quiet = f'http://127.0.0.1:{faucet_port}/faucet', threading.Event(), lambda *_: None
    try:
        endpoints = {}
        for rank in (1, 2):
            home = tmp_path / f'owner-{rank}'
            owner = network.Owner(served, stage(world, rank), rank, chain, network.Account(home / 'account.key'),
                                  network.signing_key(home / 'log.key'), home, threads=1)
            endpoints[rank] = f'127.0.0.1:{free_ports(1)}'
            owner.register(endpoints[rank], faucet_url, progress=quiet)
            threading.Thread(target=owner.run, args=('127.0.0.1', int(endpoints[rank].split(':')[1]), stop, quiet),
                             daemon=True).start()
        state = chain.state()
        assert sorted((o['shard'], o['endpoint']) for o in state['owners'].values()) == sorted(endpoints.items())
        case = data.make_case('development', 'copy', 0)
        user = network.Account(tmp_path / 'user' / 'account.key')
        conversation = network.Conversation(served, stage(world, 0), chain, user, case['world'], budget=32768,
                                            price=network.NEURO, faucet_url=faucet_url, progress=quiet, threads=1)
        assert [endpoint for _, endpoint in conversation.owners] == [endpoints[1], endpoints[2]]
        turns = [conversation.say(case['turns'][0]['user']), conversation.say('Thank you.')]
        conversation.close()
        assert all(turn['generations'] >= 1 for turn in turns) and turns[0]['version'] == 'tiny'
        outcome = chain.settled(conversation.job_id, timeout=120, poll=0.5)
        assert outcome['status'] == 'settled', outcome
        assert outcome['positions'][0] == outcome['positions'][1] > 0
        assert all(paid > 0 for paid in outcome['paid']) and outcome['refunded'] > 0
        assert sum(outcome['paid']) + outcome['refunded'] == network.NEURO
    finally:
        stop.set()
        faucet.shutdown()
        chain_network.stop(config)
