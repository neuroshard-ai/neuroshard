import hashlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import socket
import threading

import pytest

from neuroshard.evolution.sharded.retained_objects import CorruptObject, UnavailableObject, restore, transfer


@pytest.fixture
def store():
    state = {'object': None, 'puts': 0, 'gets': 0, 'interrupt': True, 'corrupt': False}

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def do_PUT(self):
            body = self.rfile.read(int(self.headers['Content-Length']))
            assert self.headers['If-None-Match'] == '*'
            state['puts'] += 1
            if state['object'] is None:
                state['object'] = body
                # The write succeeded, but the sender cannot know that.
                self.connection.shutdown(socket.SHUT_RDWR)
                self.connection.close()
                return
            assert body == state['object']
            self.send_response(412)
            self.end_headers()

        def do_GET(self):
            state['gets'] += 1
            if self.path.startswith('/missing'):
                self.send_response(503)
                self.end_headers()
                return
            body = state['object']
            if state['corrupt'] or self.path.startswith('/bad'):
                body = bytes([body[0] ^ 1]) + body[1:]
            self.send_response(200)
            self.send_header('Content-Length', str(len(body)))
            self.end_headers()
            if state['interrupt']:
                state['interrupt'] = False
                self.wfile.write(body[:len(body)//2])
                self.wfile.flush()
                self.connection.shutdown(socket.SHUT_RDWR)
                self.connection.close()
            else:
                self.wfile.write(body)

    server = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
    worker = threading.Thread(target=server.serve_forever, daemon=True)
    worker.start()
    try:
        yield 'http://127.0.0.1:' + str(server.server_port), state
    finally:
        server.shutdown()
        server.server_close()
        worker.join()


def test_ambiguous_put_and_interrupted_readback_repeat_same_object_before_acceptance(store, tmp_path):
    url, state = store
    body = b'immutable weight and optimizer bytes' * 256
    source = tmp_path / 'checkpoint'
    source.write_bytes(body)
    key = hashlib.sha256(body).hexdigest()
    result = transfer(key, len(body), url, source=source, put_url=url, attempts=3, max_seconds=10)
    assert result == {'bytes': len(body), 'readback_verified': True, 'attempts': 3}
    assert state['puts'] == 3 and state['gets'] == 2 and state['object'] == body
    state['interrupt'] = True
    target = tmp_path / 'restored' / key
    assert transfer(key, len(body), url, destination=target, max_seconds=10)['attempts'] == 2
    assert target.read_bytes() == body and not list(target.parent.glob('.receiving-*'))


def test_corrupt_readback_is_not_published_or_retried_as_a_network_failure(store, tmp_path):
    url, state = store
    body = b'committed checkpoint'
    state.update(object=body, corrupt=True, interrupt=False)
    target = tmp_path / 'restored'
    with pytest.raises(CorruptObject, match='readback'):
        transfer(hashlib.sha256(body).hexdigest(), len(body), url, destination=target, max_seconds=10)
    assert state['gets'] == 1 and not target.exists() and list(tmp_path.iterdir()) == []


def test_restore_falls_back_from_corrupt_replica_and_resumes_without_redownload(store, tmp_path):
    url, state = store
    body = b'one immutable object with separate providers' * 128
    state.update(object=body, interrupt=False)
    key, target = hashlib.sha256(body).hexdigest(), tmp_path/'restored'
    urls = [url+'/bad?signature=do-not-log', url+'/good']
    result = restore(key, len(body), urls, target, max_seconds=5)
    assert result['replica'] == 1 and result['attempts'] == 2
    assert result['failures'] == [{'replica': 0, 'error': 'CorruptObject'}]
    assert target.read_bytes() == body and state['gets'] == 2
    assert restore(key, len(body), urls, target, max_seconds=5)['attempts'] == 0
    assert state['gets'] == 2


def test_restore_total_failure_leaves_no_accepted_partial_object_or_secret(store, tmp_path):
    url, state = store
    body = b'public committed bytes'
    state.update(object=body, interrupt=False)
    key, target = hashlib.sha256(body).hexdigest(), tmp_path/'restored'
    with pytest.raises(UnavailableObject) as failure:
        restore(key, len(body), [url+'/missing?token=do-not-log', url+'/bad?signature=do-not-log'],
                target, attempts=2, max_seconds=5)
    assert 'do-not-log' not in str(failure.value) and not target.exists()
    assert not list(tmp_path.glob('.receiving-*'))
