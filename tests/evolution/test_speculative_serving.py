"""Speculative decoding over the owner ring is exact: the same tokens, logs that replay, fewer passes."""

import socket
import threading

import torch
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from neuroshard.evolution.sharded import granite, granite_audit, granite_serving as serving
from neuroshard.inference import relay
from test_granite_serving import SPEC, bounded, update_world, world  # noqa: F401


def ring_driver(world, tmp_path, name):
    """A real three-shard ring on socket pairs with signed links; returns the driver, a closer and the partitions."""
    config = granite.load_config(world['config'])
    partitions = {rank: granite.load_partition(config, world['shards'], rank)[0] for rank in (0, 1, 2)}
    adapter = serving.Adapter(partitions[2], SPEC, world['arm'])
    keys = {rank: Ed25519PrivateKey.generate() for rank in (0, 1, 2)}
    public = {rank: key.public_key().public_bytes_raw().hex() for rank, key in keys.items()}
    session = {'chain_id': 'test', 'job_id': 'ab' * 32, 'request_root': 'cd' * 32,
               'session_key': public[0], 'log_keys': [public[1], public[2]]}
    sockets, threads, errors = {}, [], []

    def owner(rank, connection):
        try:
            ring = relay.OwnerRelay(connection, rank, 3, config.hidden_size, 100000,
                                    relay.WorkBudget(65536, config.max_position_embeddings))
            log = granite_audit.OwnerLog(rank)
            link = granite_audit.Link(session, rank, 3, keys[rank], public[rank - 1])
            serving.serve(partitions[rank], ring, adapter if rank == 2 else None, log, None, link)
            log.save(tmp_path / f'{name}-log-{rank}', keys[rank], session)
        except Exception as error:  # noqa: BLE001 - surfaced by the test
            errors.append(error)
        finally:
            connection.close()

    for rank in (1, 2):
        sockets[rank], connection = socket.socketpair()
        sockets[rank].settimeout(60)
        thread = threading.Thread(target=owner, args=(rank, connection), daemon=True)
        thread.start()
        threads.append(thread)
    ring = relay.DriverRelay(sockets, 3, config.hidden_size, 100000, True,
                             relay.WorkBudget(65536, config.max_position_embeddings))
    driver = serving.ServingDriver(partitions[0], ring, granite_audit.Link(session, 0, 3, keys[0], public[2]))

    def close():
        driver.stop()
        for thread in threads:
            thread.join(60)
        for connection in sockets.values():
            connection.close()
        assert not errors, errors
        return {rank: granite_audit.load(tmp_path / f'{name}-log-{rank}') for rank in (1, 2)}

    return driver, close, partitions, adapter


def test_prompt_lookup_guesses_continue_a_repeated_ngram():
    assert serving.lookup([1, 2, 3, 9, 9, 1, 2, 3], 3) == [9, 9, 1]
    assert serving.lookup([5, 6, 7], 4) == []
    assert serving.lookup([4, 8, 4], 2, ngram=1) == [8, 4]


def test_speculative_decoding_writes_the_greedy_tokens_in_fewer_passes_and_its_logs_replay(update_world, tmp_path):  # noqa: F811
    torch.set_num_threads(1)
    tokenizer, policy = update_world['tokenizer'], {**bounded(), 'generation': {**bounded()['generation'],
                                                                                'max_new_tokens': 40}}
    eos = {tokenizer.eos_token_id}
    # A conversation whose second request repeats the first reply's wording, so guesses can land.
    turns = [[{'role': 'user', 'content': 'Repeat after me: the delivery plan for Cedar starts on 2027-05-14.'}]]
    outputs = {}
    for name, guesses in (('greedy', 0), ('speculative', 6)):
        driver, close, partitions, adapter = ring_driver(update_world, tmp_path, name)
        try:
            driver.episode(True)
            respond = serving.responder(driver, tokenizer, policy, eos, guesses=guesses)
            first = respond(turns[0], None)
            follow = turns[0] + [{'role': 'assistant', 'content': first['text']},
                                 {'role': 'user', 'content': 'Say it once more, exactly.'}]
            second = respond(follow, None)
            outputs[name] = (first, second)
        finally:
            logs = close()
        for rank in (1, 2):
            record, inputs = logs[rank]
            assert not granite_audit.unattested(record)
            assert granite_audit.replay(partitions[rank], record, inputs, adapter if rank == 2 else None)['valid']
    for got, want in zip(outputs['speculative'], outputs['greedy']):
        assert got['token_ids'] == want['token_ids'] and got['text'] == want['text']
    speculative = sum(r['passes'] for r in outputs['speculative'])
    greedy = sum(r['passes'] for r in outputs['greedy'])
    assert speculative < greedy  # guesses landed: fewer passes around the ring for the same tokens


def test_a_small_model_drafter_changes_no_token_and_its_logs_replay(update_world, tmp_path):  # noqa: F811
    """A drafter proposes; the owners verify. The reply is the greedy reply, and every owner's log replays clean."""
    from transformers import AutoModelForCausalLM

    torch.set_num_threads(1)
    tokenizer, policy = update_world['tokenizer'], {**bounded(), 'generation': {**bounded()['generation'],
                                                                                'max_new_tokens': 40}}
    eos = {tokenizer.eos_token_id}
    turns = [{'role': 'user', 'content': 'Repeat after me: the delivery plan for Cedar starts on 2027-05-14.'}]
    # The tiny checkpoint itself drafts (perfect proposals where it matches the served arm); a differently seeded
    # one would propose badly. Both must leave the reply unchanged.
    draft = AutoModelForCausalLM.from_pretrained(update_world['checkpoint'], dtype=torch.bfloat16,
                                                 attn_implementation='eager', local_files_only=True).eval()
    outputs = {}
    for name, guesses, proposer in (('greedy', 0, None), ('drafted', 6, serving.Drafter(draft, tokenizer, tokenizer))):
        driver, close, partitions, adapter = ring_driver(update_world, tmp_path, name)
        try:
            driver.episode(True)
            respond = serving.responder(driver, tokenizer, policy, eos, guesses=guesses, proposer=proposer)
            first = respond(turns, None)
            second = respond(turns + [{'role': 'assistant', 'content': first['text']},
                                      {'role': 'user', 'content': 'And once more.'}], None)
            outputs[name] = (first, second)
        finally:
            logs = close()
        for rank in (1, 2):
            record, inputs = logs[rank]
            assert not granite_audit.unattested(record)
            assert granite_audit.replay(partitions[rank], record, inputs, adapter if rank == 2 else None)['valid']
    for got, want in zip(outputs['drafted'], outputs['greedy']):
        assert got['token_ids'] == want['token_ids'] and got['text'] == want['text']
    assert sum(r['passes'] for r in outputs['drafted']) < sum(r['passes'] for r in outputs['greedy'])


def test_a_drafter_with_other_token_ids_is_refused():
    import pytest as _pytest
    from types import SimpleNamespace
    served = SimpleNamespace(get_vocab=lambda: {'a': 0, 'b': 1}, all_special_ids=[0])
    other = SimpleNamespace(get_vocab=lambda: {'a': 0, 'c': 1}, all_special_ids=[0])
    with _pytest.raises(ValueError, match='other token IDs'):
        serving.Drafter(object(), other, served)
