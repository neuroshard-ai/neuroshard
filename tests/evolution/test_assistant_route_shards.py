import socket
import threading

import torch
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from neuroshard.evolution import assistant_experience_train as trainer, assistant_routing as routing
from neuroshard.evolution.sharded import granite, granite_audit, granite_serving as serving
from neuroshard.inference import relay
from test_granite_serving import SPEC, bounded, load, update_world, world  # noqa: F401


def test_route_modules_features_caches_and_audits_match_the_complete_models(update_world, tmp_path):
    torch.set_num_threads(1)
    references, additions = {}, {}
    for index in (1, 2):
        model = load(update_world['checkpoint'])
        trainer.load_trainable(model, 'update', SPEC, update_world['arm'])
        trainer.serving(model, SPEC)
        parameters = trainer.prepare(model, 'addition', SPEC)
        generator = torch.Generator().manual_seed(20 + index)
        with torch.no_grad():
            for name, value in parameters.items():
                if name.endswith('lora_b'):
                    value.copy_(torch.randn(value.shape, generator=generator))
        directory = tmp_path / f'route-{index}'
        trainer.checkpoint(directory, parameters,
                           {'arm': 'addition', 'optimizer_state': {}, 'trainable_parameters': 1,
                            'steps': 0, 'schedule_sha256': '', 'losses': [0.0]}, {})
        references[index], additions[index] = model.eval(), directory
    config = granite.load_config(update_world['config'])
    partitions = {rank: granite.load_partition(config, update_world['shards'], rank)[0] for rank in (0, 1, 2)}
    bank = serving.AdapterBank(partitions[2], SPEC, update_world['arm'], additions)
    keys = {rank: Ed25519PrivateKey.generate() for rank in (0, 1, 2)}
    public = {rank: key.public_key().public_bytes_raw().hex() for rank, key in keys.items()}
    session = {'chain_id': 'test', 'job_id': 'ab' * 32, 'request_root': 'cd' * 32,
               'session_key': public[0], 'log_keys': [public[1], public[2]]}
    sockets, logs, threads, errors = {}, {}, [], []
    def owner(rank, connection):
        try:
            ring = relay.OwnerRelay(connection, rank, 3, config.hidden_size, 100000,
                                    relay.WorkBudget(65536, config.max_position_embeddings))
            log = logs[rank] = granite_audit.OwnerLog(rank)
            link = granite_audit.Link(session, rank, 3, keys[rank], public[rank - 1])
            serving.serve(partitions[rank], ring, bank if rank == 2 else None, log, None, link)
            log.save(tmp_path / f'log-{rank}', keys[rank], session)
        except Exception as error:
            errors.append(error)
        finally:
            connection.close()
    for rank in (1, 2):
        sockets[rank], connection = socket.socketpair()
        sockets[rank].settimeout(30)
        thread = threading.Thread(target=owner, args=(rank, connection), daemon=True)
        thread.start()
        threads.append(thread)
    ring = relay.DriverRelay(sockets, 3, config.hidden_size, 100000, True,
                             relay.WorkBudget(65536, config.max_position_embeddings))
    link = granite_audit.Link(session, 0, 3, keys[0], public[2])
    driver = serving.ServingDriver(partitions[0], ring, link)
    try:
        parent, tokenizer, policy = load(update_world['checkpoint']), update_world['tokenizer'], bounded()
        prefix = routing.message_prefix(parent, tokenizer, policy, 'cpu')
        for user in ('Schedule a review.', 'Create a draft.'):
            expected = routing.message_feature(parent, tokenizer, policy, user, 'cpu', prefix)
            assert driver.message_feature(tokenizer, policy, user) == expected
        # Priming again is a no-op once the prefix is cached (the fast-start path calls it before the first message).
        driver.prime_router(tokenizer, policy)
        assert driver.message_feature(tokenizer, policy, 'Schedule a review.') == \
            routing.message_feature(parent, tokenizer, policy, 'Schedule a review.', 'cpu', prefix)
        reference_caches = {}
        for index in (1, 2, 1):
            driver.route(index)
            tokens = torch.tensor([[1, 2, 3]] if index not in reference_caches else [[7]])
            past = reference_caches.get(index)
            mask = torch.ones((1, (past.get_seq_length() if past is not None else 0) + tokens.shape[1]),
                              dtype=torch.long)
            with torch.inference_mode():
                expected = references[index](input_ids=tokens, attention_mask=mask, past_key_values=past,
                                             use_cache=True)
                reference_caches[index] = expected.past_key_values
                assert torch.equal(driver.step(tokens, mask), expected.logits[:, -1:])
        driver.stop()
        for thread in threads:
            thread.join(30)
        assert not errors and not any(thread.is_alive() for thread in threads)
        for rank in (1, 2):
            record, inputs = granite_audit.load(tmp_path / f'log-{rank}')
            assert not granite_audit.unattested(record)
            assert granite_audit.replay(partitions[rank], record, inputs, bank if rank == 2 else None)['valid']
    finally:
        for connection in sockets.values():
            connection.close()


def ring_of_three(update_world, tmp_path, label):  # noqa: F811
    """A signed three-shard ring on the tiny update world: (driver, finish) where finish() stops it and returns
    each owner's (record, inputs)."""
    config = granite.load_config(update_world['config'])
    partitions = {rank: granite.load_partition(config, update_world['shards'], rank)[0] for rank in (0, 1, 2)}
    bank = serving.Adapter(partitions[2], SPEC, update_world['arm'])
    keys = {rank: Ed25519PrivateKey.generate() for rank in (0, 1, 2)}
    public = {rank: key.public_key().public_bytes_raw().hex() for rank, key in keys.items()}
    session = {'chain_id': 'test', 'job_id': '12' * 32, 'request_root': 'cd' * 32,
               'session_key': public[0], 'log_keys': [public[1], public[2]]}
    sockets, threads, errors = {}, [], []

    def owner(rank, connection):
        try:
            ring = relay.OwnerRelay(connection, rank, 3, config.hidden_size, 100000,
                                    relay.WorkBudget(65536, config.max_position_embeddings))
            log = granite_audit.OwnerLog(rank)
            link = granite_audit.Link(session, rank, 3, keys[rank], public[rank - 1])
            serving.serve(partitions[rank], ring, bank if rank == 2 else None, log, None, link)
            log.save(tmp_path / f'{label}-{rank}', keys[rank], session)
        except Exception as error:
            errors.append(error)
        finally:
            connection.close()
    for rank in (1, 2):
        sockets[rank], connection = socket.socketpair()
        sockets[rank].settimeout(30)
        thread = threading.Thread(target=owner, args=(rank, connection), daemon=True)
        thread.start()
        threads.append(thread)
    ring = relay.DriverRelay(sockets, 3, config.hidden_size, 100000, True,
                             relay.WorkBudget(65536, config.max_position_embeddings))
    driver = serving.ServingDriver(partitions[0], ring, granite_audit.Link(session, 0, 3, keys[0], public[2]))

    def finish():
        driver.stop()
        for thread in threads:
            thread.join(30)
        for connection in sockets.values():
            connection.close()
        assert not errors and not any(thread.is_alive() for thread in threads)
        out = {}
        for rank in (1, 2):
            record, inputs = granite_audit.load(tmp_path / f'{label}-{rank}')
            assert not granite_audit.unattested(record)
            assert granite_audit.replay(partitions[rank], record, inputs, bank if rank == 2 else None)['valid']
            out[rank] = (record, inputs)
        return out

    return driver, finish


def test_priming_the_router_before_the_first_message_changes_no_feature_reply_or_log_tensor(update_world, tmp_path):  # noqa: F811
    """Fast start: the router prefix forwarded while the user types gives the same features and the same replies,
    and every owner tensor is bit-identical to priming on the first message. Owner logs replay either way."""
    from neuroshard.assistant import network

    torch.set_num_threads(1)
    tokenizer, policy = update_world['tokenizer'], bounded()
    runs = {}
    for label in ('lazy', 'early'):
        driver, finish = ring_of_three(update_world, tmp_path, label)
        try:
            if label == 'early':
                primer = network.Primer(driver, tokenizer, policy)
                primer.wait()
                assert primer.seconds is not None and driver.router_prefix
            features = [driver.message_feature(tokenizer, policy, user)
                        for user in ('Schedule a review.', 'Create a draft.')]
            driver.episode(True)
            respond = serving.responder(driver, tokenizer, policy, {tokenizer.eos_token_id})
            reply = respond([{'role': 'user', 'content': 'Create a draft.'}], None)
        finally:
            logs = finish()
        runs[label] = features, reply['token_ids'], logs
    assert runs['early'][0] == runs['lazy'][0]
    assert runs['early'][1] == runs['lazy'][1]
    for rank in (1, 2):
        (_, lazy_inputs), (_, early_inputs) = runs['lazy'][2][rank], runs['early'][2][rank]
        # Logged inputs are keyed by command position. The early path sends one extra ADAPTER 0 (a no-op: the
        # parent is already selected) before the prefix, so positions shift; the forwarded tensors, in order, do not.
        order = lambda inputs: [inputs[name] for name in sorted(inputs, key=lambda n: int(str(n).split('.')[0]))]  # noqa: E731
        lazy, early = order(lazy_inputs), order(early_inputs)
        assert len(lazy) == len(early) and all(torch.equal(a, b) for a, b in zip(lazy, early))


def test_a_failed_primer_raises_on_the_next_use_of_the_driver():
    from neuroshard.assistant import network

    class Broken:
        def prime_router(self, tokenizer, policy):
            raise ConnectionError('owner went away')

    primer = network.Primer(Broken(), None, None)
    import pytest
    with pytest.raises(ConnectionError, match='owner went away'):
        primer.wait()


def test_a_parent_based_route_serves_and_replays_beside_update_based_routes(update_world, tmp_path):
    """Route 3 is an addition on the unchanged parent (the general route); routes 1 and 2 stay on U1."""
    torch.set_num_threads(1)
    references, additions = {}, {}
    for index in (1, 3):
        model = load(update_world['checkpoint'])
        if index == 1:
            trainer.load_trainable(model, 'update', SPEC, update_world['arm'])
            trainer.serving(model, SPEC)
        parameters = trainer.prepare(model, 'addition', SPEC)
        generator = torch.Generator().manual_seed(40 + index)
        with torch.no_grad():
            for name, value in parameters.items():
                if name.endswith('lora_b'):
                    value.copy_(torch.randn(value.shape, generator=generator))
        directory = tmp_path / f'route-{index}'
        trainer.checkpoint(directory, parameters,
                           {'arm': 'addition', 'optimizer_state': {}, 'trainable_parameters': 1,
                            'steps': 0, 'schedule_sha256': '', 'losses': [0.0]}, {})
        references[index], additions[index] = model.eval(), directory
    parent = load(update_world['checkpoint']).eval()
    config = granite.load_config(update_world['config'])
    partitions = {rank: granite.load_partition(config, update_world['shards'], rank)[0] for rank in (0, 1, 2)}
    bank = serving.AdapterBank(partitions[2], SPEC, update_world['arm'], additions, parent_routes=(3,))
    import pytest
    with pytest.raises(ValueError, match='parent-based route must be one of'):
        serving.AdapterBank(partitions[1], SPEC, update_world['arm'], additions, parent_routes=(5,))
    keys = {rank: Ed25519PrivateKey.generate() for rank in (0, 1, 2)}
    public = {rank: key.public_key().public_bytes_raw().hex() for rank, key in keys.items()}
    session = {'chain_id': 'test', 'job_id': 'ef' * 32, 'request_root': 'cd' * 32,
               'session_key': public[0], 'log_keys': [public[1], public[2]]}
    sockets, logs, threads, errors = {}, {}, [], []

    def owner(rank, connection):
        try:
            ring = relay.OwnerRelay(connection, rank, 3, config.hidden_size, 100000,
                                    relay.WorkBudget(65536, config.max_position_embeddings))
            log = logs[rank] = granite_audit.OwnerLog(rank)
            link = granite_audit.Link(session, rank, 3, keys[rank], public[rank - 1])
            serving.serve(partitions[rank], ring, bank if rank == 2 else None, log, None, link)
            log.save(tmp_path / f'plog-{rank}', keys[rank], session)
        except Exception as error:
            errors.append(error)
        finally:
            connection.close()
    for rank in (1, 2):
        sockets[rank], connection = socket.socketpair()
        sockets[rank].settimeout(30)
        thread = threading.Thread(target=owner, args=(rank, connection), daemon=True)
        thread.start()
        threads.append(thread)
    ring = relay.DriverRelay(sockets, 3, config.hidden_size, 100000, True,
                             relay.WorkBudget(65536, config.max_position_embeddings))
    link = granite_audit.Link(session, 0, 3, keys[0], public[2])
    driver = serving.ServingDriver(partitions[0], ring, link)
    try:
        caches = {}
        for index in (3, 1, 3):
            driver.route(index)
            tokens = torch.tensor([[1, 2, 3]] if index not in caches else [[7]])
            past = caches.get(index)
            mask = torch.ones((1, (past.get_seq_length() if past is not None else 0) + tokens.shape[1]),
                              dtype=torch.long)
            with torch.inference_mode():
                expected = references[index](input_ids=tokens, attention_mask=mask, past_key_values=past,
                                             use_cache=True)
                caches[index] = expected.past_key_values
                assert torch.equal(driver.step(tokens, mask), expected.logits[:, -1:]), index
        driver.route(3, arm=False)  # the parent itself, on route 3's cache
        driver.episode(False)
        tokens = torch.tensor([[4, 5]])
        with torch.inference_mode():
            expected = parent(input_ids=tokens, attention_mask=torch.ones((1, 2), dtype=torch.long), use_cache=True)
            assert torch.equal(driver.step(tokens, torch.ones((1, 2), dtype=torch.long)), expected.logits[:, -1:])
        driver.stop()
        for thread in threads:
            thread.join(30)
        assert not errors and not any(thread.is_alive() for thread in threads)
        for rank in (1, 2):
            record, inputs = granite_audit.load(tmp_path / f'plog-{rank}')
            assert granite_audit.replay(partitions[rank], record, inputs, bank if rank == 2 else None)['valid']
    finally:
        for connection in sockets.values():
            connection.close()
