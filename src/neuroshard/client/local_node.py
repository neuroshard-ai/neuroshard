"""Read atomic native state through a synchronized, pinned local full node."""
import base64
import copy
import threading
import time
from urllib.parse import urlsplit

from . import wire
from .provider_wire import Unavailable
from neuroshard.evolution.schema import root


class LocalNode:
    """Authority is the operator's full node, with an explicitly pinned genesis.

    These ABCI queries have no light-client proofs. Consequently remote RPC
    endpoints are deliberately unsupported for assignment authorization.
    """
    def __init__(self, url, chain_id, manifest_root, *, rpc=wire.rpc, clock=time.monotonic,
                 cache_seconds=.25, stall_seconds=30):
        parsed = urlsplit(url)
        if (parsed.scheme != 'http' or parsed.hostname not in ('127.0.0.1', '::1')
                or parsed.path not in ('', '/') or parsed.username or parsed.password
                or parsed.query or parsed.fragment):
            raise ValueError('Authorize assignments only through the local full-node loopback RPC')
        self.url, self.chain_id, self.rpc, self.clock = url, chain_id, rpc, clock
        self.cache_seconds, self.stall_seconds = cache_seconds, stall_seconds
        self.lock, self.cached = threading.RLock(), {}
        self.height, self.advanced = -1, clock()
        if wire.digest(self.query('/manifest')) != root(manifest_root):
            raise ValueError('The local full node has a different genesis manifest')
        status = self.rpc(url, 'status', timeout=5)
        if status['node_info']['network'] != chain_id or status['sync_info']['catching_up']:
            raise Unavailable('The local full node is on another chain or still synchronizing')

    def query(self, path, data=None):
        params = {'path': path, 'prove': False}
        if data is not None:
            params['data'] = wire.canonical(data).hex()
        response = self.rpc(self.url, 'abci_query', params, timeout=5)['response']
        if response.get('code', 0):
            raise Unavailable('The local full node refused the provider query')
        return wire.parse(base64.b64decode(response['value'], validate=True))

    def snapshot(self, job_id, *, refresh=False):
        job_id = root(job_id)
        with self.lock:
            now = self.clock()
            cached = self.cached.get(job_id)
            if not refresh and cached and now - cached[0] < self.cache_seconds:
                return copy.deepcopy(cached[1])
            try:
                snapshot = self.query('/hosting/job', {'job_id': job_id})
            except (OSError, ValueError) as error:
                self.cached.pop(job_id, None)
                raise Unavailable('The local validating node is unavailable') from error
            if snapshot['chain_id'] != self.chain_id or snapshot['height'] < self.height:
                raise Unavailable('Local chain identity or committed height changed')
            if snapshot['height'] > self.height:
                self.height, self.advanced = snapshot['height'], now
            if now - self.advanced >= self.stall_seconds:
                raise Unavailable('Local consensus has stopped advancing')
            if len(self.cached) >= 16 and job_id not in self.cached:
                del self.cached[min(self.cached, key=lambda key: self.cached[key][0])]
            self.cached[job_id] = (now, snapshot)
            return copy.deepcopy(snapshot)

