"""Lightweight certificate-pinned delivery of provisional native chat output."""
import hashlib
import http.client
import ipaddress
import socket
import ssl
from urllib.parse import urlsplit

from . import wire
from neuroshard.evolution.schema import integer, root

FORMAT = 'neuroshard-hosted-visible-stream-v1'


class Unavailable(RuntimeError):
    pass


def fields(value, expected, message):
    if not isinstance(value, dict) or set(value) != set(expected):
        raise ValueError(message)


def endpoint(value):
    if not isinstance(value, str) or not 1 <= len(value.encode()) <= 512:
        raise ValueError('Require a bounded provider HTTPS endpoint')
    parsed = urlsplit(value)
    if (parsed.scheme != 'https' or not parsed.hostname or parsed.username is not None
            or parsed.password is not None or parsed.query or parsed.fragment
            or parsed.path not in ('', '/') or any(c.isspace() for c in value)
            or parsed.port is not None and not 1 <= parsed.port <= 65535):
        raise ValueError('Require a provider HTTPS origin without credentials or paths')
    return value.rstrip('/')



class PinnedConnection(http.client.HTTPSConnection):
    def __init__(self, address, fingerprint, *, timeout=30, allow_private=False):
        parsed = urlsplit(endpoint(address))
        tls = ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
        tls.minimum_version = ssl.TLSVersion.TLSv1_3
        tls.check_hostname = False
        tls.verify_mode = ssl.CERT_NONE  # The exact ledger certificate is checked before sending data.
        super().__init__(parsed.hostname, parsed.port or 443, timeout=timeout, context=tls)
        self.fingerprint, self.allow_private = root(fingerprint), allow_private

    def connect(self):
        addresses = socket.getaddrinfo(self.host, self.port, type=socket.SOCK_STREAM)
        for family, kind, proto, _name, address in addresses:
            ip = ipaddress.ip_address(address[0])
            if not self.allow_private and not ip.is_global:
                continue
            raw = socket.socket(family, kind, proto)
            raw.settimeout(self.timeout)
            raw.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            try:
                raw.connect(address)
                self.sock = self._context.wrap_socket(raw, server_hostname=self.host)
                if hashlib.sha256(self.sock.getpeercert(binary_form=True)).hexdigest() != self.fingerprint:
                    self.close()
                    raise ValueError('Provider TLS certificate differs from the native assignment')
                return
            except BaseException:
                raw.close()
                raise
        raise ValueError('Provider endpoint has no permitted public address')



def poll(connection, customer, coordinator, chain_id, job_id, epoch, after):
    """One bounded TLS request; the caller checks native settlement separately."""
    from neuroshard.dataflow.store import canonical
    request = {'format': FORMAT, 'chain_id': chain_id, 'job_id': job_id,
               'assignment_root': epoch, 'after': after}
    raw = canonical(customer.sign(request))
    connection.request('POST', '/v1/events', raw, {'Content-Type': 'application/json',
                                                 'Content-Length': str(len(raw))})
    response = connection.getresponse()
    count = int(response.getheader('Content-Length', '-1'))
    if response.status != 200 or not 1 <= count <= 256*1024 or response.getheader('Transfer-Encoding') is not None:
        raise ValueError('Provider refused the bounded visible stream')
    value = response.read(count)
    if len(value) != count:
        raise ValueError('Truncated visible stream reply')
    body, signer = wire.verify(wire.parse(value))
    fields(body, {'request', 'update'}, 'Invalid visible stream reply')
    if signer != coordinator or body['request'] != request:
        raise ValueError('Visible stream belongs to a different request or coordinator')
    update = body['update']
    if update is not None:
        fields(update, {'format', 'chain_id', 'job_id', 'assignment_root', 'graph', 'request_root',
                        'tokenizer', 'sequence', 'status', 'text', 'verified'}, 'Invalid visible draft')
        if (any(update[k] != request[k] for k in ('format', 'chain_id', 'job_id', 'assignment_root'))
                or update['verified'] is not False or not isinstance(update['text'], str)
                or len(update['text'].encode()) > 32768):
            raise ValueError('Provider changed the draft identity or verification status')
        integer(update['sequence'], after + 1, 2**31)
    return update
