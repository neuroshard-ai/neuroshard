"""``neuroshard assistant web``: chat with the assistant in a browser on this machine.

The page is served on loopback only (unless the user explicitly allows otherwise) and every request carries a
random per-launch token. The paid conversation is opened lazily, on the first message, and runs on one worker
thread: ``network.Conversation`` is not thread-safe, so every ``say`` and the final ``close`` are serialized
there. HTTP handlers only enqueue work and read its state, so the server never blocks on the model.

Only the standard library is used. The page is self-contained (inline CSS and JS, nothing fetched from elsewhere).
Neither the account key nor any file path is ever sent to the page.
"""
import hmac
import http.server
import ipaddress
import json
import queue
import re
import secrets
import socket
import sys
import threading
import time
import traceback
import urllib.parse

TOKEN_HEADER = 'X-NeuroShard-Token'
MAX_BODY = 64 * 1024
MAX_TEXT = 8000
KEEP_REQUESTS = 200
PATH = re.compile(r"(?<![\w:/.])(?:~|[A-Za-z]:)?[\\/](?:[^\s'\"\\/]+[\\/])+[^\s'\"]*")


def loopback(host):
    """True for addresses that only this machine can reach."""
    if host == 'localhost':
        return True
    try:
        return ipaddress.ip_address(host.strip('[]')).is_loopback
    except ValueError:
        return False


def check_bind(host, allow_remote=False, warn=None):
    """Refuse a non-loopback address unless the user asked for it; warn when they did."""
    if loopback(host):
        return
    if not allow_remote:
        raise ValueError(f'refusing to serve the chat on {host}: anyone who can reach it could read your documents '
                         'and spend your conversation budget. Use the default 127.0.0.1, or pass --allow-remote.')
    (warn or (lambda text: print(text, file=sys.stderr, flush=True)))(
        f'WARNING: serving on {host}, not loopback. Anyone on your network who obtains the link can read your '
        'workspace documents and chat on your budget. Keep the link private.')


def scrub(text):
    """One line, with file paths removed: what the page may show of an error."""
    return ' '.join(PATH.sub('<path>', str(text)).split())[:500]


def describe(error):
    """A clear one-line message for the page; the traceback goes to the terminal only."""
    from neuroshard.assistant import admission_client, funding

    if isinstance(error, admission_client.NetworkBusy):
        return f'The network is busy right now and nothing was paid ({scrub(error)}). Try again in a few minutes.'
    if isinstance(error, admission_client.NotServed):
        return f'The job was funded but not served: {scrub(error)}.'
    if isinstance(error, admission_client.FundingLapsed):
        return f'Funding outcome lapsed: {scrub(error)}'
    if isinstance(error, funding.Unresolved):
        return f'A previous payment is unresolved: {scrub(error)}. Resolve it before chatting again.'
    if isinstance(error, (ValueError, OSError, KeyError)):
        return f'{scrub(error) or type(error).__name__}'
    return f'Unexpected error ({type(error).__name__}); details are in the terminal.'


def documents(world):
    """The workspace as ``show`` lists it: identifiers, titles, revisions; no contents, no paths."""
    listed = []
    for document in sorted(world.get('documents', []),
                           key=lambda d: (d.get('project', ''), d.get('revision', 0), d['id'])):
        label = str(document.get('title', ''))
        if 'revision' in document:
            label += f", revision {document['revision']}, {document.get('status', '')}"
        listed.append({'id': str(document['id']), 'label': label})
    return listed


def changed_drafts(result, before):
    """The drafts this turn changed, as ``render`` prints them."""
    lines = []
    for project, draft in (result.get('snapshot') or {}).get('drafts', {}).items():
        if before.get(project) != draft:
            fields = '; '.join(f"{key} {', '.join(map(str, value)) if isinstance(value, list) else value}"
                               for key, value in sorted(draft.items()) if key not in ('project', 'revision'))
            lines.append(f'draft for {project}: {fields}')
    return lines


def settlement_hint(job_id):
    return ('The owners commit their logs now; the job settles after its challenge window. '
            f'Check it with: neuroshard assistant status --job {job_id}')


class ChatService:
    """The conversation, its single worker thread, and the state the page polls.

    ``conversation_factory(progress)`` opens the paid conversation; it is called on the worker, on the first
    message. ``progress`` receives the lines admission and funding report, shown in the page's status line."""

    def __init__(self, world, suggestions, conversation_factory, log=None):
        self.world, self.suggestions, self.factory = world, list(suggestions), conversation_factory
        self.log = log or (lambda text: print(text, file=sys.stderr, flush=True))
        self.conversation, self.job_id, self.ended, self.failed, self.hint = None, None, False, False, None
        self.drafts, self.requests, self.order = {}, {}, []
        self.lock, self.closed_once = threading.Lock(), threading.Lock()
        self.current = None
        self.work = queue.Queue()
        self.worker = threading.Thread(target=self.loop, name='neuroshard-web-chat', daemon=True)
        self.worker.start()

    # Called from HTTP handlers: they only enqueue and read state.

    def submit(self, kind, text=None):
        with self.lock:
            if self.ended:
                raise ValueError('this conversation has ended; restart the command to start another')
            if kind == 'end':
                self.ended = True  # no new messages once ending is requested
            request_id = secrets.token_hex(8)
            self.requests[request_id] = {'kind': kind, 'status': 'queued', 'progress': '', 'started': None,
                                         'elapsed': 0.0}
            self.order.append(request_id)
            while len(self.order) > KEEP_REQUESTS:
                self.requests.pop(self.order.pop(0), None)
        self.work.put((request_id, kind, text))
        return request_id

    def poll(self, request_id):
        with self.lock:
            request = self.requests.get(request_id)
            if request is None:
                return None
            value = {k: v for k, v in request.items() if k != 'started'}
            if request['started'] is not None and request['status'] == 'running':
                value['elapsed'] = time.monotonic() - request['started']
            value.update(self.public_state())
            return value

    def state(self):
        with self.lock:
            return {'documents': documents(self.world), 'suggestions': self.suggestions, **self.public_state()}

    def public_state(self):
        return {'job': self.job_id[:16] if self.job_id else None, 'ended': self.ended, 'hint': self.hint}

    # The worker: every conversation call happens here, one at a time.

    def progress(self, text):
        self.log(str(text))
        with self.lock:
            if self.current in self.requests:
                self.requests[self.current]['progress'] = scrub(text)

    def update(self, request_id, **fields):
        with self.lock:
            if request_id in self.requests:
                self.requests[request_id].update(fields)

    def loop(self):
        while True:
            request_id, kind, text = self.work.get()
            if kind == 'stop':
                return
            with self.lock:
                self.current = request_id
            self.update(request_id, status='running', started=time.monotonic())
            started = time.monotonic()
            try:
                if kind == 'end':
                    self.close()
                    outcome = {'status': 'done', 'result': {'hint': self.hint}}
                else:
                    outcome = {'status': 'done', 'result': self.say(text)}
            except Exception as error:  # noqa: BLE001 - reported to the page as one line, traceback to the terminal
                self.log(''.join(traceback.format_exception(type(error), error, error.__traceback__)).rstrip())
                outcome = {'status': 'error', 'error': describe(error)}
                self.after_error(error)
            self.update(request_id, elapsed=time.monotonic() - started, **outcome)
            with self.lock:
                self.current = None

    def say(self, text):
        if self.conversation is None:
            self.progress('Connecting to the network')
            self.conversation = self.factory(self.progress)
            with self.lock:
                self.job_id = getattr(self.conversation, 'job_id', None)
            if self.job_id:
                self.progress(f'Admitted job {self.job_id[:16]}')
        self.progress('Thinking')
        result = self.conversation.say(text)
        reply = {'text': result.get('final_text') or '(no reply)', 'route': result.get('route'),
                 'seconds': float(result.get('seconds') or 0), 'drafts': changed_drafts(result, self.drafts),
                 'failure': scrub(result['failure']) if result.get('failure') else None}
        self.drafts = dict((result.get('snapshot') or {}).get('drafts', {}))
        if result.get('failure'):  # the terminal chat ends the job here too
            self.end_after_failure()
        return reply

    def after_error(self, error):
        """A busy network funded nothing, so the next message may try again; any other failure ends the job,
        as the terminal chat does, so a retry can never fund a second job unnoticed."""
        from neuroshard.assistant import admission_client

        if self.conversation is None and isinstance(error, admission_client.NetworkBusy):
            return
        self.end_after_failure()

    def end_after_failure(self):
        with self.lock:
            self.ended = self.failed = True
        try:
            self.close()
        except Exception as error:  # noqa: BLE001 - the original failure is what the page reports
            self.log(f'Closing after a failure: {type(error).__name__}: {error}')

    def close(self):
        """Close the conversation exactly once; the owners then commit their logs."""
        if not self.closed_once.acquire(blocking=False):
            return
        conversation, self.conversation = self.conversation, None
        with self.lock:
            self.ended = True
            self.hint = (settlement_hint(self.job_id) if self.job_id else
                         'The conversation could not open (see the message above); restart the command to try '
                         'again.' if self.failed else 'No conversation was opened from this page, so nothing was paid.')
        self.log(self.hint)
        if conversation is not None:
            conversation.close()

    def shutdown(self, timeout=30):
        """On exit: let queued work finish, close the conversation, stop the worker."""
        with self.lock:
            self.ended = True
        self.work.put((None, 'end', None))
        self.work.put((None, 'stop', None))
        self.worker.join(timeout)
        if self.worker.is_alive():  # a turn is still running; close anyway so the owners commit
            self.close()


class Server(http.server.ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = True

    def __init__(self, address, service, token):
        super().__init__(address, Handler)
        self.service, self.token, self.nonce = service, token, secrets.token_urlsafe(16)
        host, port = self.server_address[:2]
        self.local = loopback(host)
        shown = f'[{host}]' if ':' in host else host
        self.hosts = {f'{name}:{port}' for name in ('127.0.0.1', 'localhost', '[::1]', shown)}

    def url(self):
        host, port = self.server_address[:2]
        shown = f'[{host}]' if ':' in host else host
        return f'http://{shown}:{port}/?t={self.token}'


class Server6(Server):
    address_family = socket.AF_INET6


class Handler(http.server.BaseHTTPRequestHandler):
    server_version = 'NeuroShardWebChat'
    sys_version = ''

    def log_message(self, format, *args):  # noqa: A002 - the request line carries the token: never log it
        pass

    def send(self, code, body, content_type='application/json; charset=utf-8'):
        payload = body if isinstance(body, bytes) else json.dumps(body).encode()
        self.send_response(code)
        self.send_header('Content-Type', content_type)
        self.send_header('Content-Length', str(len(payload)))
        self.send_header('Cache-Control', 'no-store')
        self.send_header('X-Content-Type-Options', 'nosniff')
        self.send_header('Referrer-Policy', 'no-referrer')
        self.send_header('X-Frame-Options', 'DENY')
        self.send_header('Content-Security-Policy',
                         f"default-src 'none'; script-src 'nonce-{self.server.nonce}'; style-src 'unsafe-inline'; "
                         "connect-src 'self'; img-src 'none'; base-uri 'none'; form-action 'none'; "
                         "frame-ancestors 'none'")
        self.end_headers()
        self.wfile.write(payload)

    def allowed(self, query):
        """Same host, no foreign origin, and the launch token (header or ``t`` query parameter).

        On loopback the Host header must name loopback too (DNS rebinding); with ``--allow-remote`` any Host is
        accepted, and the token is what protects the page. An Origin must always be this page's own."""
        host = self.headers.get('Host', '')
        if self.server.local and host not in self.server.hosts:
            return False
        origin = self.headers.get('Origin')
        if origin is not None and origin != f'http://{host}':
            return False
        token = self.headers.get(TOKEN_HEADER) or (query.get('t') or [''])[0]
        return hmac.compare_digest(token.encode(), self.server.token.encode())

    def route(self, method):
        parsed = urllib.parse.urlsplit(self.path)
        query = urllib.parse.parse_qs(parsed.query)
        if not self.allowed(query):
            return self.send(403, {'error': 'forbidden'})
        service = self.server.service
        try:
            if method == 'GET' and parsed.path == '/':
                return self.send(200, PAGE.replace('__NONCE__', self.server.nonce).encode(),
                                 'text/html; charset=utf-8')
            if method == 'GET' and parsed.path == '/api/state':
                return self.send(200, service.state())
            if method == 'GET' and parsed.path == '/api/poll':
                value = service.poll((query.get('id') or [''])[0])
                return self.send(404, {'error': 'unknown request'}) if value is None else self.send(200, value)
            if method == 'POST' and parsed.path in ('/api/say', '/api/end'):
                if not self.headers.get('Content-Type', '').startswith('application/json'):
                    return self.send(415, {'error': 'expected application/json'})
                length = int(self.headers.get('Content-Length') or 0)
                if not 0 <= length <= MAX_BODY:
                    return self.send(413, {'error': 'message too long'})
                body = json.loads(self.rfile.read(length) or b'{}')
                if parsed.path == '/api/end':
                    return self.send(200, {'id': service.submit('end')})
                text = str(body.get('text', '')).strip() if isinstance(body, dict) else ''
                if not text:
                    return self.send(400, {'error': 'empty message'})
                if len(text) > MAX_TEXT:
                    return self.send(413, {'error': f'messages are limited to {MAX_TEXT} characters'})
                return self.send(200, {'id': service.submit('say', text)})
            return self.send(404, {'error': 'not found'})
        except ValueError as error:
            return self.send(409, {'error': scrub(error)})

    def do_GET(self):  # noqa: N802 - http.server's naming
        self.route('GET')

    def do_POST(self):  # noqa: N802
        self.route('POST')


def start(service, host='127.0.0.1', port=8765, allow_remote=False, token=None, warn=None):
    """Bind (refusing non-loopback unless allowed) and serve in a background thread; returns the server."""
    check_bind(host, allow_remote, warn)
    host = host.strip('[]')
    server = (Server6 if ':' in host else Server)((host, port), service, token or secrets.token_urlsafe(24))
    threading.Thread(target=server.serve_forever, name='neuroshard-web-http', daemon=True).start()
    return server


def run(world, suggestions, conversation_factory, host='127.0.0.1', port=8765, allow_remote=False):
    """Serve until Ctrl-C, then close the conversation so the owners commit their logs."""
    import signal

    service = ChatService(world, suggestions, conversation_factory)
    server = start(service, host, port, allow_remote)
    stopped = threading.Event()
    signal.signal(signal.SIGTERM, lambda *_: stopped.set())
    print(f'Open {server.url()} in your browser. Nothing is paid until you send the first message. '
          'Ctrl-C stops the server and ends the conversation.', flush=True)
    try:
        while not stopped.wait(1):
            pass
    except KeyboardInterrupt:
        pass
    finally:
        server.shutdown()
        server.server_close()
        service.shutdown()


PAGE = r'''<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<meta name="referrer" content="no-referrer">
<title>NeuroShard assistant</title>
<style>
  :root { --bg: #f6f7f9; --panel: #fff; --line: #dde1e6; --text: #1d232b; --muted: #66707c; --accent: #2457c5;
          --you: #e8effc; --err: #b3261e; }
  * { box-sizing: border-box; }
  body { margin: 0; font: 15px/1.45 system-ui, -apple-system, "Segoe UI", sans-serif; background: var(--bg);
         color: var(--text); height: 100vh; display: flex; flex-direction: column; }
  header { padding: 10px 16px; border-bottom: 1px solid var(--line); background: var(--panel); display: flex;
           align-items: center; gap: 12px; }
  header h1 { font-size: 16px; margin: 0; flex: 1; }
  main { flex: 1; display: flex; min-height: 0; }
  #chat { flex: 1; display: flex; flex-direction: column; min-width: 0; }
  #messages { flex: 1; overflow-y: auto; padding: 16px; }
  .msg { max-width: 760px; margin: 0 0 12px; padding: 10px 12px; border-radius: 8px; background: var(--panel);
         border: 1px solid var(--line); white-space: pre-wrap; word-wrap: break-word; }
  .msg.you { background: var(--you); margin-left: auto; }
  .msg.system { background: transparent; border-style: dashed; color: var(--muted); }
  .msg.error { border-color: var(--err); color: var(--err); }
  .meta { font-size: 12px; color: var(--muted); margin-top: 6px; }
  .draft { font-size: 13px; color: var(--muted); margin-top: 4px; }
  #status { padding: 6px 16px; font-size: 13px; color: var(--muted); border-top: 1px solid var(--line);
            min-height: 30px; }
  #status.error { color: var(--err); }
  form { display: flex; gap: 8px; padding: 12px 16px; border-top: 1px solid var(--line); background: var(--panel); }
  textarea { flex: 1; resize: none; font: inherit; padding: 8px; border: 1px solid var(--line); border-radius: 6px;
             min-height: 44px; }
  button { font: inherit; padding: 8px 14px; border-radius: 6px; border: 1px solid var(--accent);
           background: var(--accent); color: #fff; cursor: pointer; }
  button.secondary { background: transparent; color: var(--accent); }
  button:disabled { opacity: .5; cursor: default; }
  aside { width: 300px; border-left: 1px solid var(--line); background: var(--panel); overflow-y: auto;
          padding: 12px 16px; font-size: 13px; }
  aside h2 { font-size: 13px; text-transform: uppercase; letter-spacing: .04em; color: var(--muted); }
  aside li { margin-bottom: 6px; }
  aside code { font-size: 12px; color: var(--muted); }
  .suggestion { display: block; text-align: left; width: 100%; margin-bottom: 6px; background: transparent;
                color: var(--accent); border-color: var(--line); }
  @media (max-width: 760px) { aside { display: none; } }
</style>
</head>
<body>
<header>
  <h1>NeuroShard assistant</h1>
  <span id="job"></span>
  <button id="end" class="secondary" type="button">End conversation</button>
</header>
<main>
  <section id="chat">
    <div id="messages" aria-live="polite"></div>
    <div id="status">Ready. Nothing is paid until you send the first message.</div>
    <form id="form">
      <textarea id="text" placeholder="Type a request; Enter sends, Shift+Enter adds a line" rows="2"></textarea>
      <button id="send" type="submit">Send</button>
    </form>
  </section>
  <aside>
    <h2>Workspace</h2>
    <p id="summary"></p>
    <ul id="documents"></ul>
    <h2 id="suggest-title" hidden>Try</h2>
    <div id="suggestions"></div>
  </aside>
</main>
<script nonce="__NONCE__">
(function () {
  'use strict';
  var token = new URLSearchParams(location.search).get('t') || '';
  var $ = function (id) { return document.getElementById(id); };
  var busy = false, ended = false;

  function api(path, body) {
    var options = { headers: { 'X-NeuroShard-Token': token }, cache: 'no-store' };
    if (body !== undefined) {
      options.method = 'POST';
      options.headers['Content-Type'] = 'application/json';
      options.body = JSON.stringify(body);
    }
    return fetch(path, options).then(function (response) {
      return response.json().catch(function () { return {}; }).then(function (value) {
        if (!response.ok) { throw new Error(value.error || ('HTTP ' + response.status)); }
        return value;
      });
    });
  }
  function element(tag, cls, text) {
    var node = document.createElement(tag);
    if (cls) { node.className = cls; }
    if (text !== undefined) { node.textContent = text; }
    return node;
  }
  function add(cls, text) {
    var node = element('div', 'msg ' + cls, text);
    $('messages').appendChild(node);
    $('messages').scrollTop = $('messages').scrollHeight;
    return node;
  }
  function status(text, error) {
    $('status').textContent = text;
    $('status').className = error ? 'error' : '';
  }
  function controls() {
    $('send').disabled = busy || ended;
    $('text').disabled = ended;
    $('end').disabled = busy || ended;
  }
  function apply(state) {
    if (state.job) { $('job').textContent = 'Job ' + state.job; }
    if (state.ended && !ended) {
      ended = true;
      if (state.hint) { add('system', state.hint); }
    }
    controls();
  }
  function wait(id, onDone) {
    api('/api/poll?id=' + encodeURIComponent(id)).then(function (value) {
      if (value.status === 'queued') { status('Waiting for the previous request...'); }
      if (value.status === 'running') {
        var phase = value.progress || (value.job ? 'Thinking' : 'Connecting to the network');
        status(phase + ' (' + Math.round(value.elapsed) + ' s)');
      }
      if (value.status === 'done' || value.status === 'error') {
        busy = false;
        onDone(value);
        apply(value);
        return;
      }
      setTimeout(function () { wait(id, onDone); }, 500);
    }).catch(function (error) {
      status('Lost contact with the local server: ' + error.message + '. Retrying...', true);
      setTimeout(function () { wait(id, onDone); }, 2000);
    });
  }
  function reply(value) {
    if (value.status === 'error') {
      add('error', value.error);
      status(value.error, true);
      return;
    }
    var r = value.result, node = add('assistant', r.text);
    node.appendChild(element('div', 'meta', Math.round(r.seconds) + ' s' + (r.route ? ', ' + r.route : '')));
    r.drafts.forEach(function (line) { node.appendChild(element('div', 'draft', line)); });
    if (r.failure) {
      node.appendChild(element('div', 'draft', 'stopped: ' + r.failure));
      status('Stopped: ' + r.failure, true);
    } else {
      status('Job ' + (value.job || '') + ' open. Type your next request.');
    }
  }
  function send(text) {
    if (busy || ended || !text) { return; }
    busy = true; controls();
    add('you', text);
    $('text').value = '';
    status('Sending...');
    api('/api/say', { text: text }).then(function (value) { wait(value.id, reply); }).catch(function (error) {
      busy = false; controls();
      add('error', error.message);
      status(error.message, true);
    });
  }
  $('form').addEventListener('submit', function (event) { event.preventDefault(); send($('text').value.trim()); });
  $('text').addEventListener('keydown', function (event) {
    if (event.key === 'Enter' && !event.shiftKey) { event.preventDefault(); send($('text').value.trim()); }
  });
  $('end').addEventListener('click', function () {
    if (busy || ended || !window.confirm('End this conversation? The owners then commit their logs.')) { return; }
    busy = true; controls();
    status('Ending the conversation...');
    api('/api/end', {}).then(function (value) {
      wait(value.id, function (done) {
        if (done.status === 'error') { add('error', done.error); status(done.error, true); }
        else { status('Conversation ended.'); }
      });
    }).catch(function (error) { busy = false; controls(); status(error.message, true); });
  });
  api('/api/state').then(function (state) {
    $('summary').textContent = state.documents.length + ' documents, on this machine only.';
    state.documents.forEach(function (d) {
      var item = element('li');
      item.appendChild(element('code', '', d.id));
      item.appendChild(document.createTextNode(' ' + d.label));
      $('documents').appendChild(item);
    });
    $('suggest-title').hidden = state.suggestions.length === 0;
    state.suggestions.forEach(function (text) {
      var button = element('button', 'suggestion', text);
      button.type = 'button';
      button.addEventListener('click', function () { $('text').value = text; $('text').focus(); });
      $('suggestions').appendChild(button);
    });
    apply(state);
  }).catch(function (error) { status('Could not reach the local server: ' + error.message, true); });
  controls();
}());
</script>
</body>
</html>
'''
