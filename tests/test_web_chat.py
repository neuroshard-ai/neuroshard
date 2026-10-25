"""``neuroshard assistant web``: the local browser chat, with a fake conversation (no model, no network)."""
import json
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path
from types import SimpleNamespace

import pytest

from neuroshard.assistant import admission_client, funding
from neuroshard.client import assistant_cli, web_chat

WORLD = {'documents': [{'id': 'doc-1', 'title': 'Quote', 'content': 'secret contents'},
                       {'id': 'doc-2', 'title': 'Plan', 'project': 'p', 'revision': 2, 'status': 'draft',
                        'content': 'more'}]}


class Fake:
    """A conversation that answers instantly (or after ``delay``) and records overlap."""

    def __init__(self, progress, delay=0.0, fail=None):
        self.job_id = 'ab' * 32
        self.delay, self.fail = delay, fail
        self.active, self.overlapped, self.closed, self.said = 0, False, 0, []
        self.guard = threading.Lock()
        progress('Reserving shard 1')

    def say(self, text):
        with self.guard:
            self.active += 1
            self.overlapped |= self.active > 1
        try:
            time.sleep(self.delay)
            if self.fail is not None:
                raise self.fail
            self.said.append(text)
            return {'final_text': f'echo: {text}', 'route': 'drafting', 'seconds': 1.5,
                    'snapshot': {'drafts': {'p': {'project': 'p', 'to': ['a@b'], 'subject': 'Hi'}}}, 'failure': None}
        finally:
            with self.guard:
                self.active -= 1

    def close(self):
        self.closed += 1


@pytest.fixture
def served():
    made = []

    def launch(delay=0.0, fail=None, factory=None):
        def default(progress):
            made.append(Fake(progress, delay, fail))
            return made[-1]

        service = web_chat.ChatService(WORLD, ['Try this'], factory or default, log=lambda _: None)
        server = web_chat.start(service, '127.0.0.1', 0, token='tok-123')
        launched.append((server, service))
        return server, service, made

    launched = []
    yield launch
    for server, service in launched:
        server.shutdown()
        server.server_close()
        service.shutdown(timeout=5)


def call(server, path, body=None, token='tok-123', origin=None, query_token=False):
    host, port = server.server_address[:2]
    if query_token and token:
        path += ('&' if '?' in path else '?') + f't={token}'
    request = urllib.request.Request(f'http://{host}:{port}{path}', method='POST' if body is not None else 'GET',
                                     data=None if body is None else json.dumps(body).encode())
    if body is not None:
        request.add_header('Content-Type', 'application/json')
    if token and not query_token:
        request.add_header(web_chat.TOKEN_HEADER, token)
    if origin:
        request.add_header('Origin', origin)
    try:
        with urllib.request.urlopen(request, timeout=10) as response:
            raw = response.read()
            return response.status, (json.loads(raw) if 'json' in response.headers['Content-Type'] else raw), \
                response.headers
    except urllib.error.HTTPError as error:
        return error.code, json.loads(error.read() or b'{}'), error.headers


def finish(server, request_id, timeout=10):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        _, value, _ = call(server, f'/api/poll?id={request_id}')
        if value['status'] in ('done', 'error'):
            return value
        time.sleep(0.02)
    raise AssertionError('request did not finish')


def test_the_token_is_required_on_every_route(served):
    server, _, made = served()
    for path in ('/', '/api/state', '/api/poll?id=x'):
        assert call(server, path, token=None)[0] == 403
        assert call(server, path, token='wrong')[0] == 403
    assert call(server, '/api/say', {'text': 'hi'}, token=None)[0] == 403
    assert call(server, '/api/end', {}, token=None)[0] == 403
    assert made == []  # nothing was opened
    status, page, headers = call(server, '/', query_token=True)
    assert status == 200 and b'<textarea' in page
    assert b'http://' not in page and b'https://' not in page  # self-contained: nothing fetched from elsewhere
    assert "default-src 'none'" in headers['Content-Security-Policy']
    assert 'Access-Control-Allow-Origin' not in headers
    assert call(server, '/api/state', query_token=True)[0] == 200


def test_a_foreign_origin_or_host_is_rejected(served):
    server, _, made = served()
    assert call(server, '/api/say', {'text': 'hi'}, origin='http://evil.example')[0] == 403
    assert call(server, '/api/state', origin='null')[0] == 403
    port = server.server_address[1]
    assert call(server, '/api/state', origin=f'http://127.0.0.1:{port}')[0] == 200
    assert call(server, '/api/state', origin=f'http://localhost:{port}')[0] == 403  # not this page's own origin
    request = urllib.request.Request(f'http://127.0.0.1:{port}/api/state', headers={
        web_chat.TOKEN_HEADER: 'tok-123', 'Host': f'rebound.example:{port}'})  # DNS rebinding
    with pytest.raises(urllib.error.HTTPError) as refused:
        urllib.request.urlopen(request, timeout=5)
    assert refused.value.code == 403
    assert made == []


def test_state_lists_documents_without_contents_or_paths(served):
    server, _, made = served()
    _, state, _ = call(server, '/api/state')
    assert state['documents'] == [{'id': 'doc-1', 'label': 'Quote'}, {'id': 'doc-2', 'label': 'Plan, revision 2, draft'}]
    assert state['suggestions'] == ['Try this'] and state['job'] is None
    assert 'secret contents' not in json.dumps(state)
    assert made == []  # opening the page costs nothing


def test_say_then_poll_returns_the_reply(served):
    server, _, made = served()
    status, value, _ = call(server, '/api/say', {'text': 'hello'})
    assert status == 200 and value['id']
    done = finish(server, value['id'])
    assert done['status'] == 'done'
    assert done['result']['text'] == 'echo: hello' and done['result']['route'] == 'drafting'
    assert done['result']['drafts'] == ['draft for p: subject Hi; to a@b']
    assert done['job'] == 'ab' * 8 and len(made) == 1
    second = finish(server, call(server, '/api/say', {'text': 'again'})[1]['id'])
    assert second['result']['drafts'] == []  # unchanged draft is not repeated
    assert len(made) == 1 and made[0].said == ['hello', 'again']
    assert call(server, '/api/poll?id=nope')[0] == 404
    assert call(server, '/api/say', {'text': '  '})[0] == 400


def test_progress_reaches_the_status_line(served):
    server, _, _ = served(delay=0.5)
    request_id = call(server, '/api/say', {'text': 'hello'})[1]['id']
    seen = set()
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:
        value = call(server, f'/api/poll?id={request_id}')[1]
        seen.add(value['progress'])
        if value['status'] == 'done':
            break
        time.sleep(0.02)
    assert 'Thinking' in seen and value['status'] == 'done'


def test_concurrent_says_are_serialized(served):
    server, _, made = served(delay=0.2)
    ids = []
    threads = [threading.Thread(target=lambda t=t: ids.append(call(server, '/api/say', {'text': t})[1]['id']))
               for t in ('one', 'two', 'three')]
    started = time.monotonic()
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert time.monotonic() - started < 0.3  # the handler returned without waiting for the model
    results = [finish(server, request_id) for request_id in ids]
    assert all(r['status'] == 'done' for r in results)
    assert len(made) == 1 and not made[0].overlapped and sorted(made[0].said) == ['one', 'three', 'two']


@pytest.mark.parametrize('error, expected', [
    (admission_client.NetworkBusy('the network stayed busy until the deadline; nothing was funded'), 'busy'),
    (admission_client.NotServed('cd' * 32, 900, 'shard 2 left'), 'funded but not served'),
    (admission_client.FundingLapsed('ef' * 32), 'no longer open'),
    (funding.Unresolved('funding outcome unknown for job 1234'), 'unresolved'),
    (ValueError('price must be a number of NEURO'), 'price must be a number of NEURO'),
    (RuntimeError('boom at /home/alice/secret/file.json'), 'Unexpected error (RuntimeError)'),
])
def test_errors_map_to_one_clean_line(served, error, expected):
    logged = []
    server, service, made = served(fail=error)
    service.log = logged.append
    done = finish(server, call(server, '/api/say', {'text': 'hello'})[1]['id'])
    assert done['status'] == 'error' and expected in done['error']
    assert '\n' not in done['error'] and 'Traceback' not in done['error'] and '/home/alice' not in done['error']
    assert any('Traceback' in line for line in logged)  # the terminal keeps the details
    assert made[0].closed == 1 and done['ended']  # a failed turn ends the job, as the terminal chat does


def test_paths_in_messages_are_scrubbed():
    assert '/home' not in web_chat.describe(ValueError('no such file: /home/alice/.neuroshard/account.key'))
    assert '<path>' in web_chat.scrub('open C:\\Users\\a\\key.json failed')


def test_busy_before_opening_allows_a_retry(served):
    attempts = []

    def factory(progress):
        attempts.append(1)
        if len(attempts) == 1:
            raise admission_client.NetworkBusy('the network stayed busy until the deadline; nothing was funded')
        return Fake(progress)

    server, _, _ = served(factory=factory)
    first = finish(server, call(server, '/api/say', {'text': 'hi'})[1]['id'])
    assert first['status'] == 'error' and not first['ended']
    second = finish(server, call(server, '/api/say', {'text': 'hi'})[1]['id'])
    assert second['status'] == 'done' and len(attempts) == 2


def test_end_conversation_closes_once_and_shows_the_hint(served):
    server, service, made = served()
    finish(server, call(server, '/api/say', {'text': 'hello'})[1]['id'])
    done = finish(server, call(server, '/api/end', {})[1]['id'])
    assert done['status'] == 'done' and done['ended']
    assert done['hint'] == web_chat.settlement_hint('ab' * 32)
    assert 'neuroshard assistant status --job ' + 'ab' * 32 in done['hint']
    assert call(server, '/api/end', {})[0] == 409
    assert call(server, '/api/say', {'text': 'more'})[0] == 409
    service.shutdown(timeout=5)
    assert made[0].closed == 1


def test_ending_before_any_message_pays_nothing(served):
    server, _, made = served()
    done = finish(server, call(server, '/api/end', {})[1]['id'])
    assert 'nothing was paid' in done['hint'] and made == []


def test_non_loopback_bind_is_refused_without_allow_remote():
    service = SimpleNamespace()
    for host in ('0.0.0.0', '192.168.1.5', '::', 'example.com'):
        with pytest.raises(ValueError, match='--allow-remote'):
            web_chat.start(service, host, 0)
    warnings = []
    web_chat.check_bind('0.0.0.0', allow_remote=True, warn=warnings.append)
    assert 'documents' in warnings[0]
    for host in ('127.0.0.1', 'localhost', '::1', '[::1]', '127.0.0.2'):
        web_chat.check_bind(host)


def test_the_cli_parses_web_and_dispatches_it_without_falling_into_chat(monkeypatch):
    import argparse

    parser = argparse.ArgumentParser()
    assistant_cli.register(parser.add_subparsers(dest='command'), Path('/tmp/unused'))
    args = parser.parse_args(['assistant', 'web', '--budget', '64', '--price', '0.5', '--sample', '1'])
    assert (args.port, args.host, args.allow_remote, args.budget, args.wait) == (8765, '127.0.0.1', False, 64, 600)
    assert parser.parse_args(['assistant', 'web', '--host', '0.0.0.0', '--allow-remote']).allow_remote

    from neuroshard.assistant import network
    from neuroshard.client import live_runtime

    calls = []
    monkeypatch.setattr(live_runtime, 'configure', lambda: None)
    monkeypatch.setattr(network, 'descriptor', lambda path: {})
    monkeypatch.setattr(network, 'chain', lambda *a, **k: SimpleNamespace(state=lambda: {}))
    monkeypatch.setattr(assistant_cli, 'following', lambda descriptor, *a, **k: descriptor)
    monkeypatch.setattr(assistant_cli, 'web', lambda *a: calls.append('web'))
    monkeypatch.setattr(network, 'Conversation', lambda *a, **k: calls.append('chat path'))
    monkeypatch.setattr(network, 'fetch_stage', lambda *a, **k: calls.append('chat path'))
    assistant_cli.run(args)
    assert calls == ['web']


def test_the_cli_refuses_a_remote_host_before_downloading(monkeypatch):
    from neuroshard.assistant import network

    monkeypatch.setattr(network, 'fetch_stage', lambda *a, **k: pytest.fail('downloaded before refusing'))
    args = SimpleNamespace(host='0.0.0.0', allow_remote=False, budget=64, port=8765, price='1')
    with pytest.raises(ValueError, match='--allow-remote'):
        assistant_cli.web(args, None, network, {})
