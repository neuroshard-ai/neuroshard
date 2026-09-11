"""Authenticated experimental worker transport, bound to loopback by default.

Use SSH forwarding for workers on another host. This transport does not claim
to implement permissionless worker discovery or public endpoint authorization.
"""
import argparse
import hmac
import json
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

import requests

from neuroshard.dataflow.store import canonical
from .objects import Objects, MAX_OBJECT_BYTES, digest


class Endpoint:
    def __init__(self, url, token, store):
        self.url, self.store = url.rstrip('/'), store
        self.http = requests.Session()
        self.http.trust_env = False
        self.http.headers['Authorization'] = 'Bearer ' + token.strip()
        self.store.fetchers.append(self.fetch)
        self.sent = self.received = 0

    def fetch(self, key):
        self.store.path(key)
        with self.http.get(self.url+'/objects/'+key, timeout=120, stream=True) as response:
            if response.status_code == 404:
                return None
            response.raise_for_status()
            raw = bytearray()
            for chunk in response.iter_content(1024*1024):
                if len(raw)+len(chunk) > MAX_OBJECT_BYTES:
                    raise ValueError('Worker returned an oversized artifact')
                raw.extend(chunk)
            self.received += len(raw)
            return bytes(raw)

    def push(self, key):
        self.store.path(key)
        response = self.http.head(self.url+'/objects/'+key, timeout=30)
        if response.status_code == 200:
            return
        if response.status_code != 404:
            response.raise_for_status()
        raw = self.store.get(key)
        response = self.http.put(self.url+'/objects/'+key, data=raw, timeout=180)
        response.raise_for_status()
        self.sent += len(raw)

    def call(self, method, *args):
        response = self.http.post(self.url+'/call', data=canonical({'method':method,'arguments':args}),
                                  headers={'Content-Type':'application/json'}, timeout=300)
        if response.status_code == 400:
            raise ValueError(response.json()['error'])
        response.raise_for_status()
        return response.json()

    def open(self, session_id, model_root, partition, start_step=0):
        self.push(model_root)
        model = self.store.json(model_root)
        for name in partition['components']:
            self.push(model['components'][name]['root'])
        return self.call('open',session_id,model_root,partition,*([start_step] if start_step else []))

    def operation(self, session_id, operation):
        for field in ('parent','input','gradient'):
            if operation.get(field) is not None:
                self.push(operation[field])
        return self.call('operation',session_id,operation)

    def evaluate(self, session_id, request):
        for field in ('input','batch'):
            if request.get(field) is not None:
                self.push(request[field])
        return self.call('evaluate',session_id,request)

    def release(self, session_id):
        return self.call('release',session_id)


def serve(home, port, token_file, capacity=48_000_000):
    from .worker import Worker
    store = Objects(Path(home)/'objects')
    worker = Worker(home,store,capacity)
    token = 'Bearer ' + Path(token_file).read_text().strip()
    if len(token) < 71:
        raise ValueError('Use a random token of at least 32 bytes, encoded as hex')

    class Handler(BaseHTTPRequestHandler):
        protocol_version = 'HTTP/1.1'

        def log_message(self, *_):
            pass

        def answer(self, status, raw=b'', head=False):
            self.send_response(status)
            self.send_header('Content-Length',str(len(raw)))
            self.send_header('Connection','close')
            self.end_headers()
            if not head:
                self.wfile.write(raw)
            self.close_connection = True

        def handle_request(self, method):
            try:
                if not hmac.compare_digest(self.headers.get('Authorization',''),token):
                    return self.answer(401)
                self.connection.settimeout(180)
                if self.path.startswith('/objects/'):
                    key = self.path.removeprefix('/objects/')
                    path = store.path(key)
                    if method in ('GET','HEAD'):
                        if not path.exists():
                            return self.answer(404)
                        if method == 'HEAD':
                            return self.answer(200,head=True)
                        return self.answer(200,store.get(key))
                    if method == 'PUT':
                        size = int(self.headers.get('Content-Length','-1'))
                        if not 0 <= size <= MAX_OBJECT_BYTES:
                            raise ValueError('Artifact exceeds transport bounds')
                        raw = self.rfile.read(size)
                        if len(raw) != size or digest(raw) != key:
                            raise ValueError('Uploaded artifact differs from requested hash')
                        store.put(raw)
                        return self.answer(200,b'{}')
                if method == 'POST' and self.path == '/call':
                    size = int(self.headers.get('Content-Length','-1'))
                    if not 0 < size <= 256*1024:
                        raise ValueError('RPC request exceeds bounds')
                    request = json.loads(self.rfile.read(size))
                    if request['method'] not in ('open','operation','evaluate','release'):
                        raise ValueError('Unknown worker method')
                    value = getattr(worker,request['method'])(*request['arguments'])
                    return self.answer(200,canonical(value))
                self.answer(404)
            except (ValueError,KeyError,TypeError,FileNotFoundError) as exc:
                self.answer(400,canonical({'error':str(exc)}))

        def do_GET(self): self.handle_request('GET')
        def do_HEAD(self): self.handle_request('HEAD')
        def do_PUT(self): self.handle_request('PUT')
        def do_POST(self): self.handle_request('POST')

    HTTPServer(('127.0.0.1',port),Handler).serve_forever()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--home',type=Path,required=True)
    parser.add_argument('--port',type=int,required=True)
    parser.add_argument('--token-file',type=Path,required=True)
    parser.add_argument('--capacity',type=int,default=48_000_000)
    args = parser.parse_args()
    serve(args.home,args.port,args.token_file,args.capacity)


if __name__ == '__main__':
    main()
