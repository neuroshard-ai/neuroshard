"""Certificate-pinned HTTPS frames authorized by a native provider assignment.

No network message supplies executable code, a filesystem path or a pickle.
The local node supplies committed assignment state. Signatures authenticate a
sender; only numerical replay establishes whether its tensors are correct.
"""
from datetime import datetime, timedelta, timezone
import base64
import hashlib
import http.client
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import os
from pathlib import Path
import socket
import ssl
import threading
import time

from cryptography import x509
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import ec
from cryptography.x509.oid import NameOID

from neuroshard.dataflow.store import canonical
from neuroshard.demo import protocol
from neuroshard.client.provider_wire import PinnedConnection, Unavailable
from .schema import integer, root
from .serving_graph import fields

FORMAT = 'neuroshard-provider-frames-v1'
MAX_CONTROL = 2 * 1024**2
MAX_TENSOR = 64 * 1024**2
HEADER_LIMIT = 8192
FRAME_FIELDS = {'format', 'chain_id', 'job_id', 'assignment_root', 'source', 'destination',
                'members', 'channel', 'sequence', 'bytes', 'sha256'}


def certificate(home, identity):
    """Create a local TLS identity; publish only its fingerprint in the ledger."""
    home = Path(home)
    home.mkdir(parents=True, exist_ok=True, mode=0o700)
    private, public = home/'transport-key.pem', home/'transport-cert.pem'
    subject = hashlib.sha256(identity.public_key.encode('ascii')).hexdigest()
    if private.exists() != public.exists():
        raise ValueError('Incomplete TLS identity; preserve it for explicit recovery')
    if not private.exists():
        key = ec.generate_private_key(ec.SECP256R1())
        name = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, subject)])
        now = datetime.now(timezone.utc)
        cert = (x509.CertificateBuilder().subject_name(name).issuer_name(name)
            .public_key(key.public_key()).serial_number(x509.random_serial_number())
            .not_valid_before(now - timedelta(minutes=5)).not_valid_after(now + timedelta(days=30))
            .add_extension(x509.BasicConstraints(ca=False, path_length=None), critical=True)
            .sign(key, hashes.SHA256()))
        for path, raw in ((private, key.private_bytes(serialization.Encoding.PEM,
                serialization.PrivateFormat.PKCS8, serialization.NoEncryption())),
                (public, cert.public_bytes(serialization.Encoding.PEM))):
            descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
            with os.fdopen(descriptor, 'wb') as stream:
                stream.write(raw)
                stream.flush()
                os.fsync(stream.fileno())
    cert = x509.load_pem_x509_certificate(public.read_bytes())
    if cert.subject.get_attributes_for_oid(NameOID.COMMON_NAME)[0].value != subject:
        raise ValueError('TLS identity belongs to another provider')
    context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    context.minimum_version = ssl.TLSVersion.TLSv1_3
    context.load_cert_chain(public, private)
    return context, cert.fingerprint(hashes.SHA256()).hex()


def context(state, job_id):
    """Build an immutable routing view from the local full node's snapshot."""
    lease = state.get('hosting', {}).get('leases', {}).get(root(job_id))
    if not lease or lease['status'] not in ('preparing', 'ready'):
        raise Unavailable('No live provider execution assignment')
    deadline = lease['prepare_deadline'] if lease['status'] == 'preparing' else lease['work_deadline']
    if state['height'] > deadline:
        raise Unavailable('The provider assignment deadline elapsed')
    return {'chain_id': state['chain_id'], 'job_id': job_id,
            'assignment_root': lease['assignment_root'], 'providers': lease['providers']}


def authorize(envelope, local_owner, lookup):
    body, signer = protocol.verify(envelope)
    fields(body, FRAME_FIELDS, 'Invalid provider frame header')
    if body['format'] != FORMAT or body['channel'] not in ('control', 'tensor'):
        raise ValueError('Unsupported provider frame')
    current = lookup(root(body['job_id']))
    for name in ('chain_id', 'job_id', 'assignment_root'):
        if body[name] != current[name]:
            raise ValueError('A frame changed its chain, request or assignment epoch')
    source, target = integer(body['source'], 0, 66), integer(body['destination'], 0, 66)
    members = body['members']
    if (not isinstance(members, list) or not 2 <= len(members) <= 67
            or any(type(rank) is not int for rank in members) or members != sorted(set(members))
            or source not in members or target not in members or source == target):
        raise ValueError('Invalid communication group')
    providers = current['providers']
    allowed = [sorted(map(int, providers)), [0, 1, 2],
               *[[0, 1, 2, rank] for rank in map(int, providers) if rank >= 3]]
    if (members not in allowed or any(str(rank) not in providers for rank in members)
            or providers[str(source)]['owner'] != signer
            or providers[str(target)]['owner'] != local_owner):
        raise ValueError('Frame signer or receiver is not assigned to this group')
    integer(body['sequence'], 0, 2**31 - 1)
    integer(body['bytes'], 1, MAX_CONTROL if body['channel'] == 'control' else MAX_TENSOR)
    root(body['sha256'])
    return body


def stream_key(body):
    return (body['job_id'], body['assignment_root'], tuple(body['members']),
            body['source'], body['destination'], body['channel'])


class Mailbox:
    """One outstanding frame per stream, bounded including in-flight bodies."""
    def __init__(self, owner, lookup, *, byte_limit=256*1024**2, stream_limit=2048, timeout=30):
        self.owner, self.lookup = owner, lookup
        self.byte_limit, self.stream_limit, self.timeout = byte_limit, stream_limit, timeout
        self.condition = threading.Condition()
        self.streams = {}
        self.retired = {}
        self.allocated = 0
        self.closed = False

    def begin(self, envelope):
        body = authorize(envelope, self.owner, self.lookup)
        key, size = stream_key(body), body['bytes']
        deadline = time.monotonic() + self.timeout
        with self.condition:
            if self.closed:
                raise Unavailable('Provider mailbox is closed')
            if key[:2] in self.retired:
                last = self.retired[key[:2]].get(key)
                if last == (body['sequence'], body['sha256']):
                    return body, key, False
                raise Unavailable('This provider execution has already retired')
            if key not in self.streams:
                if len(self.streams) >= self.stream_limit:
                    raise Unavailable('Provider stream capacity is full')
                self.streams[key] = {'next': 0, 'last': None, 'pending': None}
            stream = self.streams[key]
            while stream['pending'] is not None:
                pending = stream['pending']
                if body['sequence'] == stream['next']:
                    if pending['sha256'] != body['sha256'] or pending['size'] != size:
                        raise ValueError('Conflicting duplicate provider frame')
                    if pending['payload'] is not None:
                        return body, key, False
                remaining = deadline - time.monotonic()
                if remaining <= 0 or self.closed:
                    raise Unavailable('Previous provider frame has not completed')
                self.condition.wait(min(remaining, .5))
                authorize(envelope, self.owner, self.lookup)
            sequence = body['sequence']
            if sequence < stream['next']:
                if sequence == stream['next'] - 1 and stream['last'] == body['sha256']:
                    return body, key, False
                raise ValueError('Old or conflicting provider frame')
            if sequence != stream['next']:
                raise ValueError('Provider frames must arrive in order')
            if self.allocated + size > self.byte_limit:
                raise Unavailable('Provider mailbox byte capacity is full')
            self.allocated += size
            stream['pending'] = {'sha256': body['sha256'], 'size': size, 'payload': None}
            return body, key, True

    def finish(self, body, key, payload):
        if len(payload) != body['bytes'] or hashlib.sha256(payload).hexdigest() != body['sha256']:
            self.abort(key)
            raise ValueError('Provider frame differs from its signed bytes')
        current = self.lookup(body['job_id'])
        if current['assignment_root'] != body['assignment_root']:
            self.abort(key)
            raise ValueError('Provider assignment changed while reading a frame')
        with self.condition:
            stream = self.streams.get(key)
            if self.closed or stream is None or stream['pending'] is None:
                raise Unavailable('Provider mailbox was retired during the transfer')
            stream['pending']['payload'] = payload
            self.condition.notify_all()

    def abort(self, key):
        with self.condition:
            stream = self.streams.get(key)
            if stream and stream['pending'] is not None:
                self.allocated -= stream['pending']['size']
                stream['pending'] = None
                self.condition.notify_all()

    def receive(self, routing, members, source, target, channel, sequence):
        key = (routing['job_id'], routing['assignment_root'], tuple(members), source, target, channel)
        deadline = time.monotonic() + self.timeout
        with self.condition:
            while True:
                current = self.lookup(routing['job_id'])
                if current['assignment_root'] != routing['assignment_root']:
                    raise Unavailable('The native assignment was replaced')
                stream = self.streams.get(key)
                if stream and stream['pending'] and stream['pending']['payload'] is not None:
                    if stream['next'] != sequence:
                        raise ValueError('The receiver requested a different frame sequence')
                    pending = stream['pending']
                    stream.update(next=sequence + 1, last=pending['sha256'], pending=None)
                    self.allocated -= pending['size']
                    self.condition.notify_all()
                    return pending['payload']
                remaining = deadline - time.monotonic()
                if self.closed or remaining <= 0:
                    raise Unavailable('Provider frame did not arrive within its bound')
                self.condition.wait(min(remaining, .5))

    def retire(self, job_id, assignment_root):
        with self.condition:
            recent = {}
            for key in list(self.streams):
                if key[:2] == (job_id, assignment_root):
                    stream = self.streams.pop(key)
                    pending = stream['pending']
                    self.allocated -= pending['size'] if pending else 0
                    if stream['next']:
                        recent[key] = (stream['next'] - 1, stream['last'])
            # Preserve the final acknowledgement for an ambiguous HTTP retry.
            # Retired epochs cannot start new streams or reserve body buffers.
            self.retired[(job_id, assignment_root)] = recent
            while len(self.retired) > 16:
                del self.retired[next(iter(self.retired))]
            self.condition.notify_all()

    def close(self):
        with self.condition:
            self.closed = True
            self.streams.clear()
            self.retired.clear()
            self.allocated = 0
            self.condition.notify_all()


class Server(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = True

    def __init__(self, address, tls, mailbox, *, max_connections=32, events=None):
        self.tls, self.mailbox = tls, mailbox
        self.events = events
        self.slots = threading.BoundedSemaphore(max_connections)
        super().__init__(address, Handler)

    def get_request(self):
        connection, address = super().get_request()
        connection.settimeout(10)
        connection.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        # TLS handshakes happen inside the bounded worker threads, not accept().
        return self.tls.wrap_socket(connection, server_side=True, do_handshake_on_connect=False), address

    def process_request(self, request, client_address):
        if not self.slots.acquire(blocking=False):
            request.close()
            return
        try:
            super().process_request(request, client_address)
        except BaseException:
            self.slots.release()
            raise

    def process_request_thread(self, request, client_address):
        try:
            super().process_request_thread(request, client_address)
        finally:
            self.slots.release()

    def handle_error(self, request, client_address):
        # Failed peer handshakes/frames must not log bodies or signed headers.
        pass


class Handler(BaseHTTPRequestHandler):
    protocol_version = 'HTTP/1.1'

    def log_message(self, *_args):
        pass

    def do_POST(self):
        key, reserved = None, False
        try:
            if self.path == '/v1/events':
                if (self.server.events is None or self.headers.get('Transfer-Encoding') is not None
                        or len(self.headers.get_all('Content-Length', [])) != 1):
                    raise ValueError('Require a bounded visible stream request')
                count = int(self.headers['Content-Length'])
                if not 1 <= count <= 8192:
                    raise ValueError('Stream request exceeds its bound')
                raw = self.rfile.read(count)
                if len(raw) != count:
                    raise ValueError('Truncated stream request')
                result = canonical(self.server.events.read(protocol.parse_json(raw)))
                self.send_response(200)
                self.send_header('Content-Type', 'application/json')
                self.send_header('Cache-Control', 'no-store')
                self.send_header('Content-Length', str(len(result)))
                self.end_headers()
                self.wfile.write(result)
                return
            if self.path != '/v1/frame' or self.headers.get('Transfer-Encoding') is not None:
                raise ValueError('Unsupported provider request')
            header = self.headers.get('X-NeuroShard-Frame', '')
            if (not 1 <= len(header) <= HEADER_LIMIT or len(self.headers.get_all('Content-Length', [])) != 1
                    or len(self.headers.get_all('X-NeuroShard-Frame', [])) != 1):
                raise ValueError('Require one bounded signed frame header and length')
            envelope = protocol.parse_json(base64.b64decode(header, validate=True))
            body, key, reserved = self.server.mailbox.begin(envelope)
            if int(self.headers['Content-Length']) != body['bytes']:
                raise ValueError('Provider body length differs from the signed header')
            self.connection.settimeout(self.server.mailbox.timeout)
            if reserved:
                raw = self.rfile.read(body['bytes'])
                self.server.mailbox.finish(body, key, raw)
                reserved = False
            else:
                # An ambiguous acknowledgement can cause a retransmission.
                # Check it with fixed scratch space instead of allocating a
                # second unreserved tensor-sized body.
                remaining, digest = body['bytes'], hashlib.sha256()
                while remaining:
                    chunk = self.rfile.read(min(65536, remaining))
                    if not chunk:
                        raise ValueError('Truncated repeated provider frame')
                    digest.update(chunk)
                    remaining -= len(chunk)
                if digest.hexdigest() != body['sha256']:
                    raise ValueError('Repeated provider bytes changed')
            self.send_response(200)
            self.send_header('Content-Length', '0')
            self.end_headers()
        except (ValueError, KeyError, TypeError, OSError, Unavailable, RecursionError):
            if reserved:
                self.server.mailbox.abort(key)
            self.close_connection = True
            self.send_response(400)
            self.send_header('Connection', 'close')
            self.send_header('Content-Length', '0')
            self.end_headers()


class Peer:
    def __init__(self, identity, routing, rank, mailbox, *, timeout=30, allow_private=False):
        self.identity, self.routing, self.rank, self.mailbox = identity, routing, rank, mailbox
        self.timeout, self.allow_private = timeout, allow_private
        if routing['providers'][str(rank)]['owner'] != identity.public_key:
            raise ValueError('The local key does not own this model rank')
        self.connections, self.sent, self.received = {}, {}, {}

    def send(self, members, target, channel, payload):
        stream = (tuple(members), target, channel)
        sequence = self.sent.get(stream, 0)
        body = {'format': FORMAT, 'chain_id': self.routing['chain_id'], 'job_id': self.routing['job_id'],
            'assignment_root': self.routing['assignment_root'], 'source': self.rank, 'destination': target,
            'members': list(members), 'channel': channel, 'sequence': sequence,
            'bytes': len(payload), 'sha256': hashlib.sha256(payload).hexdigest()}
        header = base64.b64encode(canonical(self.identity.sign(body))).decode('ascii')
        provider = self.routing['providers'][str(target)]
        # A retry uses the same signed content and sequence. A receiver that
        # already consumed the frame acknowledges it without enqueueing it again.
        for attempt in range(2):
            connection = self.connections.get(target)
            if connection is None:
                connection = PinnedConnection(provider['endpoint'], provider['certificate'],
                                              timeout=self.timeout, allow_private=self.allow_private)
                self.connections[target] = connection
            try:
                connection.request('POST', '/v1/frame', payload, {'X-NeuroShard-Frame': header,
                    'Content-Length': str(len(payload)), 'Content-Type': 'application/octet-stream'})
                response = connection.getresponse()
                if response.status != 200 or response.getheader('Content-Length') != '0':
                    raise ValueError('Assigned provider refused the committed frame')
                response.read()
                self.sent[stream] = sequence + 1
                return
            except (OSError, http.client.HTTPException):
                connection.close()
                self.connections.pop(target, None)
                if attempt:
                    raise Unavailable('Assigned provider transport is unavailable') from None
            except ValueError:
                connection.close()
                self.connections.pop(target, None)
                raise

    def receive(self, members, source, channel):
        stream = (tuple(members), source, channel)
        sequence = self.received.get(stream, 0)
        result = self.mailbox.receive(self.routing, members, source, self.rank, channel, sequence)
        self.received[stream] = sequence + 1
        return result

    def close(self):
        for connection in self.connections.values():
            connection.close()
        self.connections.clear()
