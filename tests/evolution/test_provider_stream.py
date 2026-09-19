import copy
import threading

import pytest

from neuroshard.demo import protocol
from neuroshard.evolution import provider_stream, provider_transport
from neuroshard.evolution.serving_stream import Observer


def test_visible_drafts_hide_worked_reasoning_and_retract_ambiguous_boundaries():
    seen = []
    tokenizer = type('Tokenizer', (), {'decode': lambda self, values, **kwargs: values})()
    observer = Observer(seen.append)
    emit = observer.tokens(tokenizer, worked=True)
    emit('Private intermediate arithmetic')
    emit('Private intermediate arithmetic\nANSWER: 4')
    emit('Private intermediate arithmetic\nANSWER: 42')
    emit('Private intermediate arithmetic\nANSWER: 42\nANSWER: wrong')
    assert seen == ['', '4', '42', '']
    assert all('Private' not in text for text in seen)
    observer.text('final correction')
    assert seen[-1] == 'final correction'


def test_disconnected_observer_cannot_change_neural_execution():
    called = []
    def disconnected(text):
        called.append(text)
        raise ConnectionError('customer left')
    observer = Observer(disconnected)
    observer.text('first')
    observer.text('second')
    assert called == ['first'] and observer.failed


def test_partial_unicode_waits_and_prefix_changes_replace_the_draft():
    seen = []
    tokenizer = type('Tokenizer', (), {'decode': lambda self, values, **kwargs: values})()
    emit = Observer(seen.append).tokens(tokenizer)
    for text in ('A\ufffd', 'Aé', 'Aé.', 'Aé .'):
        emit(text)
    assert seen == ['Aé', 'Aé.', 'Aé .']


def test_live_pinned_stream_refuses_wrong_customer_and_obsolete_assignment(tmp_path):
    owner, customer, stranger = [protocol.Identity('stream-' + name) for name in ('owner', 'customer', 'stranger')]
    job = {'id': 'a'*64, 'payer': customer.public_key, 'request': {'messages': []},
           'graph': {'tokenizer': {'root': 'c'*64}}}
    epoch, chain = 'b'*64, 'visible-stream-fixture'
    state = {'chain_id': chain, 'job': job, 'lease': {'assignment_root': epoch,
             'providers': {'0': {'owner': owner.public_key}}}}
    log = provider_stream.StreamLog(owner, lambda key: copy.deepcopy(state))
    log.begin(job, chain, epoch)
    tls, fingerprint = provider_transport.certificate(tmp_path, owner)
    box = provider_transport.Mailbox(owner.public_key, lambda key: None)
    server = provider_transport.Server(('127.0.0.1', 0), tls, box, events=log)
    thread = threading.Thread(target=server.serve_forever, kwargs={'poll_interval': .02}, daemon=True)
    thread.start()
    connection = provider_transport.PinnedConnection('https://127.0.0.1:' + str(server.server_port),
        fingerprint, timeout=2, allow_private=True)
    def poll(who=customer, after=0):
        return provider_stream.poll(connection, who, owner.public_key, chain, job['id'], epoch, after)
    try:
        first = poll()
        assert first['sequence'] == 1 and first['text'] == '' and not first['verified']
        assert poll(after=1) is None
        log.emit(job['id'], epoch, text='Visible first token')
        assert poll(after=1)['text'] == 'Visible first token'
        with pytest.raises(ValueError, match='refused'):
            poll(stranger)
        connection.close()
        # All early snapshots can be discarded; resumption gets a replacement
        # with its monotonic cursor, not an incomplete concatenated suffix.
        for i in range(100):
            log.emit(job['id'], epoch, text='draft ' + str(i))
        assert len(log.rows) == 1 and poll(after=1)['text'] == 'draft 99'
        state['lease']['assignment_root'] = 'd'*64
        with pytest.raises(ValueError, match='refused'):
            poll()
    finally:
        connection.close()
        box.close()
        server.shutdown()
        server.server_close()
        thread.join(timeout=3)
