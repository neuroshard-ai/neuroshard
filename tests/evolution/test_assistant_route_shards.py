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
