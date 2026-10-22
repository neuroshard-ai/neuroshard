"""Signed control messages of admission (DESIGN_ADMISSION.md §3–§5).

Before funding there is no job, so RESERVE, HOLD and RELEASE are signed by the client's *account* key, in the
ledger's envelope format with an ``admission/*`` kind that the ledger never accepts as a transaction. After funding,
RESERVE, RELEASE and HELLO are signed by the job's *session* key. Every owner reply (CAPS, OFFER, BUSY, REFUSED,
HELD, RELEASED, READY) is signed by the owner's log key. Each message carries an expiry and a random nonce, and an
owner refuses a nonce it has seen until that expiry.

Control requests travel on short connections: one ``QUERY`` frame, one ``REPLY`` frame, then close.
"""
import json
import secrets
import socket
import threading
import time

from neuroshard.inference import optimistic as ledger
from neuroshard.inference import relay

DOMAIN = 'neuroshard/admission/v1'
VERSION = 'admission/1'
QUERY, REPLY = relay.ADMISSION_QUERY, relay.ADMISSION_REPLY
MAX_CONTROL = 16384
LIFETIME = 60  # seconds a signed request stays valid
REQUESTS = ('caps', 'reserve', 'hold', 'release')
REPLIES = ('caps', 'offer', 'busy', 'refused', 'held', 'released', 'ready')
TRANSACTION_FIELDS = {'body', 'public_key'}  # a HOLD's transaction: never its signature


def unsigned(envelope):
    """What a HOLD discloses of the signed ``serve_open``: its body and signer, from which the owner derives the job
    ID (``ledger.transaction_id`` excludes the signature), but not the signature that would let it broadcast."""
    return {'body': envelope['body'], 'public_key': envelope['public_key']}


def hello_attempt(message, chain_id, now, session_key, job_id, rank, owner):
    """A reserved HELLO's admission attempt: session-signed, fresh, and bound to this job, rank and owner.

    Returns its body (``challenge``, ``reservation``); the caller passes it through its ``Replays``."""
    kind, body, (signer, _) = opened(message, chain_id, now)
    if kind != 'hello' or signer != 'key' or not signed_by(message, session_key):
        raise ValueError("the admission attempt is not signed by the job's session key")
    if body.get('job_id') != job_id or body.get('rank') != rank or body.get('owner') != owner:
        raise ValueError('the admission attempt names another job, rank or owner')
    if not isinstance(body.get('reservation'), str):
        raise ValueError('the admission attempt names no reservation')
    return body


def _body(kind, chain_id, fields, clock):
    return {'domain': DOMAIN, 'kind': kind, 'chain_id': chain_id, 'expires': int(clock()) + LIFETIME,
            'challenge': secrets.token_hex(16), **fields}


def by_account(account, chain_id, kind, clock=time.time, **fields):
    """An account-signed request: the ledger envelope format, under a kind the ledger rejects."""
    body = _body(kind, chain_id, fields, clock)
    return account.sign(chain_id, 0, 'admission/' + kind, **{k: v for k, v in body.items() if k not in ('kind', 'chain_id')})


def by_key(key, kind, chain_id, clock=time.time, **fields):
    """A request signed by an Ed25519 key (a job's session key), or a reply signed by an owner's log key."""
    body = _body(kind, chain_id, fields, clock)
    return {'body': body, 'signature': key.sign(ledger.canonical(body)).hex()}


def opened(message, chain_id, now):
    """(kind, body, signer): the request's kind, its body and who signed it ('account', key) or ('key', None)."""
    if not isinstance(message, dict) or not isinstance(message.get('body'), dict):
        raise ValueError('malformed admission message')
    body = message['body']
    if set(message) == {'body', 'public_key', 'signature'}:
        body, public = ledger.verify(message)
        kind = body.get('kind', '')
        if not kind.startswith('admission/'):
            raise ValueError('not an admission message')
        kind, signer = kind.split('/', 1)[1], ('account', public)
    elif set(message) == {'body', 'signature'} and isinstance(message['signature'], str):
        kind, signer = body.get('kind'), ('key', None)  # the caller verifies against the key it expects
    else:
        raise ValueError('malformed admission message')
    if body.get('domain') != DOMAIN or body.get('chain_id') != chain_id:
        raise ValueError('admission message for another network')
    expires = body.get('expires')
    if type(expires) is not int or not now <= expires <= now + LIFETIME + 5:
        raise ValueError('admission message expired or too far ahead')
    if not isinstance(body.get('challenge'), str) or len(body['challenge']) != 32:
        raise ValueError('admission message has no challenge')
    return kind, body, signer


def signed_by(message, public):
    return ledger.ed25519_valid(public, ledger.canonical(message['body']), message['signature'])


def request(endpoint, message, timeout=10):
    """One control round trip to an owner; returns the reply message (unverified)."""
    host, port = ledger.endpoint(endpoint).rsplit(':', 1)
    with socket.create_connection((host, int(port)), timeout=timeout) as sock:
        sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        relay.send_frame(sock, QUERY, ledger.canonical(message))
        return json.loads(relay.receive_frame(sock, REPLY, MAX_CONTROL, time.monotonic() + timeout))


def reply(sock, log_key, chain_id, kind, clock=time.time, **fields):
    relay.send_frame(sock, REPLY, ledger.canonical(by_key(log_key, kind, chain_id, clock, **fields)))


def answer(message, owner_key, chain_id, clock=time.time, kinds=REPLIES):
    """An owner reply's body, if the owner it was sent to signed it."""
    if (not isinstance(message, dict) or set(message) != {'body', 'signature'} or not isinstance(message['body'], dict)
            or not isinstance(message['signature'], str) or not signed_by(message, owner_key)):
        raise ValueError('the owner reply is not signed by its log key')
    body = message['body']
    if body.get('domain') != DOMAIN or body.get('chain_id') != chain_id or body.get('kind') not in kinds:
        raise ValueError('unexpected owner reply')
    if type(body.get('expires')) is not int or body['expires'] < clock():
        raise ValueError('the owner reply has expired')
    return body


class Replays:
    """Challenges seen until their message expires: a captured request cannot be replayed.

    Owners handle connections on separate threads; pruning, the membership check and the insertion are one atomic
    step under ``lock``, so no thread can overwrite a challenge another has just accepted."""

    def __init__(self):
        self.seen = {}
        self.lock = threading.Lock()

    def check(self, body, now):
        with self.lock:
            for challenge in [c for c, e in self.seen.items() if e < now]:
                del self.seen[challenge]
            if body['challenge'] in self.seen:
                raise ValueError('replayed admission message')
            self.seen[body['challenge']] = body['expires']
