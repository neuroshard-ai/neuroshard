import hashlib
import json
import os
import socket
import subprocess
import sys
import threading

import pytest
import torch
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from neuroshard.evolution.sharded import granite_audit
from neuroshard.inference import relay

from test_granite_serving import BUDGET, ROOT, SPEC, bounded, session, sharded, signing_keys, single_host, world  # noqa: F401
from test_granite_tokenizer import granite_like  # noqa: F401
from test_optimistic_serving import CHAIN

RELAY = '''
import json, socket, sys, time
from pathlib import Path
import torch
from neuroshard.evolution import assistant_experience_run as accelerator, assistant_workflow_data as data, granite_tokenizer
from neuroshard.evolution.sharded import granite, granite_audit, granite_serving as serving
from neuroshard.inference import relay

config_dir, shards_dir, rank, world, home = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4]), Path(sys.argv[5])
job = json.loads((home / "job.json").read_text())
torch.set_num_threads(1)
config = granite.load_config(config_dir)
partition, _ = granite.load_partition(config, shards_dir, rank)
session, link = job["session"], serving.job_link(job, rank, world)
result = {"rank": rank}
if rank:
    adapter = serving.Adapter(partition, job["spec"], job["arm"]) if rank == world - 1 else None
    listener = socket.create_server(("127.0.0.1", 0))
    (home / f"port-{rank}").write_text(str(listener.getsockname()[1]))
    connection, _ = listener.accept()
    relay.answer(connection, rank, world, lambda chain, job_id: session["session_key"])
    ring = relay.OwnerRelay(connection, rank, world, config.hidden_size, job["max_tokens"])
    log = granite_audit.OwnerLog(rank)
    result.update(serving.serve(partition, ring, adapter, log, None, link))
    result["log"] = log.save(home / f"log-{rank}", serving.owner_key(job, rank), session)
else:
    key, sockets = serving.owner_key(job, 0), {}
    for owner in range(1, world):
        while not (home / f"port-{owner}").exists():
            time.sleep(0.05)
        body = relay.hello(session["chain_id"], session["job_id"], owner, world, config.hidden_size, job["max_tokens"])
        sockets[owner] = relay.dial("127.0.0.1:" + (home / f"port-{owner}").read_text(), body, key, timeout=120)
    ring = relay.DriverRelay(sockets, world, config.hidden_size, job["max_tokens"], signed=True)
    tokenizer, _ = granite_tokenizer.load(job["tokenizer"], parent_digest=job["parent_tokenizer_digest"])
    driver = serving.ServingDriver(partition, ring, link)
    by_id = {case["id"]: case for case in data.cases(job["split"])}
    cases = [by_id[k] for k in job["case_ids"]]
    result["episodes"] = serving.serve_episodes(driver, tokenizer, job["policy"], cases, job["gate"],
                                                lambda case: accelerator.feature_ids(tokenizer, job["policy"], case),
                                                set(job["eos_ids"]))
    driver.stop()
    result["attestation"] = link.received
result.update(sent_bytes=ring.sent_bytes, received_bytes=ring.received_bytes)
(home / f"result-{rank}.json").write_text(json.dumps(result))
'''


def relayed(world, home, keys, served):  # noqa: F811
    """The same job as ``sharded``, with owners 1 and 2 reached only through the user's device."""
    home.mkdir()
    reference = world['directory'] / 'tokenizer.json'
    from neuroshard.evolution import granite_tokenizer

    job = {'spec': SPEC, 'arm': str(world['arm']), 'gate': world['gate'], 'tokenizer': str(world['directory']),
           'parent_tokenizer_digest': granite_tokenizer.blob_sha1(reference), 'policy': bounded(),
           'eos_ids': [world['tokenizer'].eos_token_id], 'max_tokens': 100000, 'split': 'development',
           'case_ids': [c['id'] for c in world['cases']], 'keys': {str(r): str(p) for r, p in keys.items()},
           'session': served}
    (home / 'job.json').write_text(json.dumps(job))
    env = {**os.environ, 'PYTHONPATH': str(ROOT / 'src')}
    processes = [subprocess.Popen([sys.executable, '-c', RELAY, str(world['config']), str(world['shards']), str(rank),
                                   '3', str(home)], env=env) for rank in (1, 2, 0)]
    for process in processes:
        try:
            process.wait(timeout=300)
        except subprocess.TimeoutExpired:
            process.kill()
    return [json.loads((home / f'result-{rank}.json').read_text()) for rank in range(3)]


def test_owners_reached_through_the_users_relay_serve_and_log_exactly_as_on_the_ring(world, tmp_path):  # noqa: F811
    keys, public = signing_keys(tmp_path)
    served = {'chain_id': CHAIN, 'job_id': hashlib.sha256(b'relay job').hexdigest(),
              'request_root': hashlib.sha256(b'relay request').hexdigest(), 'session_key': public[0],
              'log_keys': [public[1], public[2]]}
    ring = sharded(world, tmp_path / 'ring', log=True, keys=keys, session=served)
    relay_results = relayed(world, tmp_path / 'relay', keys, served)
    assert all(r['completed'] for r in ring)
    expected = single_host(world)
    assert len(relay_results[0]['episodes']) == len(expected)
    for got, want in zip(relay_results[0]['episodes'], expected):
        assert got['score'] == want['score'] and got['calls'] == want['calls']
        assert [g['token_ids'] for g in got['generations']] == [g['token_ids'] for g in want['generations']]
    for rank in (1, 2):
        on_ring = granite_audit.load(tmp_path / 'ring' / f'log-{rank}')[0]
        on_relay = granite_audit.load(tmp_path / 'relay' / f'log-{rank}')[0]
        assert granite_audit.commitment(on_relay) == granite_audit.commitment(on_ring)
        assert not granite_audit.unattested(on_relay)
    assert relay_results[0]['attestation'] == ring[0]['attestation']
    assert relay_results[0]['attestation']['positions'] < BUDGET


def test_frames_keep_their_order_and_bounds():
    left, right = socket.socketpair()
    driver = relay.DriverRelay({1: left}, 2, 4, 8, signed=True)
    owner = relay.OwnerRelay(right, 1, 2, 4, 8)
    value = torch.arange(12, dtype=torch.float32).view(1, 3, 4).to(torch.bfloat16)
    assert driver.command(7, 3) == [7, 3] and owner.command(0) == [7, 3]
    driver.send(value, 1)
    driver.send_signature(b'\x01' * 64, 1)
    assert torch.equal(owner.receive(0, 3), value) and owner.receive_signature(0) == b'\x01' * 64
    owner.send(value[:, -1:], 0)
    assert torch.equal(driver.receive(1, 1), value[:, -1:])
    with pytest.raises(ValueError, match='boundary'):
        driver.send(torch.zeros((1, 9, 4), dtype=torch.bfloat16), 1)
    driver.send(value, 1)
    with pytest.raises(ValueError, match='length|bound'):
        owner.receive(0, 2)
    driver.command(1, 1)
    with pytest.raises(ValueError, match='frame'):
        owner.receive(0, 1)


def test_an_owner_accepts_a_link_only_from_the_jobs_session_key():
    session = Ed25519PrivateKey.generate()
    registered = session.public_key().public_bytes_raw().hex()
    listener = socket.create_server(('127.0.0.1', 0))
    endpoint = f"127.0.0.1:{listener.getsockname()[1]}"
    body = relay.hello(CHAIN, 'ab' * 32, 1, 3, 4, 8)
    outcomes = []

    def owner():
        for _ in range(2):
            connection, _ = listener.accept()
            try:
                outcomes.append(relay.answer(connection, 1, 3, lambda chain, job: registered))
            except ValueError as error:
                outcomes.append(str(error))
            finally:
                connection.close()

    thread = threading.Thread(target=owner)
    thread.start()
    with pytest.raises((ConnectionError, OSError, ValueError)):
        relay.dial(endpoint, body, Ed25519PrivateKey.generate(), timeout=10)
    relay.dial(endpoint, body, session, timeout=10).close()
    thread.join(10)
    assert outcomes == ["the hello is not signed by the job's session key", body]
    with pytest.raises(ValueError, match='host:port'):
        relay.dial('Localhost:1', body, session)
