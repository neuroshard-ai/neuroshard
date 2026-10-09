"""Serving links across the internet: the user's device dials every owner and relays between shards.

Owners publish an endpoint on the ledger and accept connections; nobody needs to reach the
user's device. Each owner exchanges every command and message of a job with the user's
device over one TCP connection, and the device forwards each owner's signed output to the
next owner unchanged. Both sides implement the interface of the in-datacenter ring
(``granite_pipeline.Ring``), so serving, link signatures and owner logs are the same, and a
relaying device cannot alter a message without breaking its sender's signature.
"""

import hashlib
import json
import socket
import struct

import torch

from neuroshard.inference import optimistic as ledger

FORMAT = 'neuroshard-relay/1'
CONTROL, TENSOR, SIGNATURE, HELLO, READY = b'C', b'T', b'S', b'H', b'R'
HEADER = struct.Struct('>cI')
COMMAND = struct.Struct('>qq')
MAX_FRAME = 1 << 28
MAX_HELLO = 4096
HELLO_FIELDS = {'format', 'chain_id', 'job_id', 'rank', 'world', 'hidden', 'max_tokens'}
TIMEOUT = 600


def read_exactly(sock, size):
    buffer = bytearray(size)
    view, done = memoryview(buffer), 0
    while done < size:
        count = sock.recv_into(view[done:], size - done)
        if not count:
            raise ConnectionError('the peer closed the serving link')
        done += count
    return buffer


def send_frame(sock, kind, payload):
    view = memoryview(payload).cast('B')
    sock.sendall(HEADER.pack(kind, view.nbytes))
    sock.sendall(view)


def receive_frame(sock, expected, limit=MAX_FRAME):
    kind, size = HEADER.unpack(read_exactly(sock, HEADER.size))
    if kind != expected:
        raise ValueError(f'expected a {expected!r} serving frame, received {kind!r}')
    if size > limit:
        raise ValueError('serving frame exceeds its bound')
    return read_exactly(sock, size)


def tensor_payload(value, hidden, max_tokens):
    if (value.dtype != torch.bfloat16 or value.ndim != 3 or value.shape[0] != 1
            or value.shape[2] != hidden or not 0 < value.shape[1] <= max_tokens):
        raise ValueError('unsupported boundary tensor')
    return value.detach().contiguous().view(torch.uint8).numpy()


def tensor_from(payload, hidden, length, max_tokens):
    width = 2 * hidden
    if not payload or len(payload) % width or len(payload) // width != length or not 0 < length <= max_tokens:
        raise ValueError('unexpected boundary length')
    value = torch.frombuffer(payload, dtype=torch.uint8).view(1, length, width).view(torch.bfloat16)
    if not bool(torch.isfinite(value).all()):
        raise ValueError('nonfinite boundary tensor')
    return value


def signature_from(payload):
    if len(payload) != 64:
        raise ValueError('unsupported link signature')
    return bytes(payload)


class OwnerRelay:
    """An owner's side of one job: every command and message over its connection to the user's device."""

    def __init__(self, sock, rank, world, hidden_size, max_tokens):
        if world < 2 or not 0 < rank < world:
            raise ValueError('an owner relay serves a rank after the user device')
        self.sock, self.rank, self.world, self.hidden, self.max_tokens = sock, rank, world, hidden_size, max_tokens
        self.sent_bytes = self.received_bytes = 0
        self.trace = None

    def command(self, op=0, length=0):
        """The next command from the user's device; an owner never issues one."""
        payload = receive_frame(self.sock, CONTROL, COMMAND.size)
        self.received_bytes += len(payload)
        return list(COMMAND.unpack(payload))

    def receive(self, source, length):
        if source != self.rank - 1:
            raise ValueError('an owner receives from the previous shard')
        payload = receive_frame(self.sock, TENSOR)
        self.received_bytes += len(payload)
        return tensor_from(payload, self.hidden, length, self.max_tokens)

    def receive_signature(self, source):
        payload = receive_frame(self.sock, SIGNATURE, 64)
        self.received_bytes += len(payload)
        return signature_from(payload)

    def send(self, value, destination):
        if destination != (self.rank + 1) % self.world:
            raise ValueError('an owner sends to the next shard')
        payload = tensor_payload(value, self.hidden, self.max_tokens)
        if self.trace is not None:
            self.trace.append(hashlib.sha256(payload).hexdigest())
        send_frame(self.sock, TENSOR, payload)
        self.sent_bytes += payload.nbytes

    def send_signature(self, signature, destination):
        send_frame(self.sock, SIGNATURE, signature_from(signature))
        self.sent_bytes += 64


class DriverRelay:
    """The user's device: commands go to every owner, and each owner's signed output is relayed to the next.

    ``sockets`` maps each owner's rank to its connection. With ``signed`` links every
    message is followed by its sender's signature, which is relayed with it.
    """

    def __init__(self, sockets, world, hidden_size, max_tokens, signed):
        if world < 2 or sorted(sockets) != list(range(1, world)):
            raise ValueError('the user device needs one connection per owner')
        self.sockets, self.rank, self.world = sockets, 0, world
        self.hidden, self.max_tokens, self.signed = hidden_size, max_tokens, signed
        self.sent_bytes = self.received_bytes = 0
        self.trace = None

    def command(self, op, length=0):
        payload = COMMAND.pack(op, length)
        for rank in range(1, self.world):
            send_frame(self.sockets[rank], CONTROL, payload)
            self.sent_bytes += len(payload)
        return [op, length]

    def send(self, value, destination):
        if destination != 1:
            raise ValueError('the user device sends to owner 1')
        payload = tensor_payload(value, self.hidden, self.max_tokens)
        if self.trace is not None:
            self.trace.append(hashlib.sha256(payload).hexdigest())
        send_frame(self.sockets[1], TENSOR, payload)
        self.sent_bytes += payload.nbytes

    def send_signature(self, signature, destination):
        if destination != 1:
            raise ValueError('the user device sends to owner 1')
        send_frame(self.sockets[1], SIGNATURE, signature_from(signature))
        self.sent_bytes += 64

    def receive(self, source, length):
        """The last owner's message, after relaying every earlier owner's output to its successor."""
        if source != self.world - 1:
            raise ValueError('the user device receives from the last owner')
        for rank in range(1, self.world - 1):
            frames = [(TENSOR, receive_frame(self.sockets[rank], TENSOR))]
            if self.signed:
                frames.append((SIGNATURE, receive_frame(self.sockets[rank], SIGNATURE, 64)))
            for kind, payload in frames:
                send_frame(self.sockets[rank + 1], kind, payload)
                self.received_bytes += len(payload)
                self.sent_bytes += len(payload)
        payload = receive_frame(self.sockets[source], TENSOR)
        self.received_bytes += len(payload)
        return tensor_from(payload, self.hidden, length, self.max_tokens)

    def receive_signature(self, source):
        payload = receive_frame(self.sockets[source], SIGNATURE, 64)
        self.received_bytes += len(payload)
        return signature_from(payload)


def hello(chain_id, job_id, rank, world, hidden_size, max_tokens):
    return {'format': FORMAT, 'chain_id': chain_id, 'job_id': job_id, 'rank': rank, 'world': world,
            'hidden': hidden_size, 'max_tokens': max_tokens}


def dial(endpoint, body, session_key, timeout=TIMEOUT):
    """Open one owner's link of a job from the user's device, proving it holds the job's session key."""
    host, port = ledger.endpoint(endpoint).rsplit(':', 1)
    sock = socket.create_connection((host, int(port)), timeout=timeout)
    try:
        sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        send_frame(sock, HELLO, ledger.canonical({'hello': body,
                                                  'signature': session_key.sign(ledger.canonical(body)).hex()}))
        ready = json.loads(receive_frame(sock, READY, MAX_HELLO))
        if ready != {'format': FORMAT, 'rank': body['rank']}:
            raise ValueError('the owner did not accept this link')
        return sock
    except BaseException:
        sock.close()
        raise


def answer(sock, rank, world, session_key_of):
    """Accept one link as owner ``rank``: the hello must name this rank and be signed by the job's session key.

    ``session_key_of(chain_id, job_id)`` returns the session key the ledger registered for
    that job, or raises if the job does not name this owner. Returns the hello.
    """
    message = json.loads(receive_frame(sock, HELLO, MAX_HELLO))
    body = message.get('hello') if isinstance(message, dict) else None
    if (not isinstance(body, dict) or set(body) != HELLO_FIELDS or body['format'] != FORMAT
            or body['rank'] != rank or body['world'] != world or set(message) != {'hello', 'signature'}):
        raise ValueError('invalid serving hello')
    key = session_key_of(body['chain_id'], body['job_id'])
    if not isinstance(message['signature'], str) or not ledger.ed25519_valid(key, ledger.canonical(body),
                                                                             message['signature']):
        raise ValueError("the hello is not signed by the job's session key")
    send_frame(sock, READY, ledger.canonical({'format': FORMAT, 'rank': rank}))
    return body
