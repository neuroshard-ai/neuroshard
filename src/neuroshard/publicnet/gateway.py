"""Bounded public reads and signed-transaction relay for a local full node."""

import argparse
import base64
import datetime
from collections import OrderedDict
import json
import ipaddress
import math
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import re
import threading
import time
from urllib.parse import parse_qs, urlparse

from neuroshard.demo import client, protocol, work
from neuroshard.publicnet.bootstrap import peer as peer_address
from neuroshard.publicnet.history import TrainingHistory


RPC_METHODS = {"status", "block", "commit", "validators", "abci_query", "broadcast_tx_sync", "broadcast_tx_commit"}
QUERY_PATHS = {"/summary", "/account", "/validators", "/manifest", "/task"}


def client_ip(address, real_ip):
    """Only a loopback reverse proxy may supply its overwritten X-Real-IP."""
    if ipaddress.ip_address(address).is_loopback and real_ip:
        try:
            return str(ipaddress.ip_address(real_ip))
        except ValueError:
            pass
    return address


def parse_block_time(value):
    # Python 3.10 accepts microseconds; CometBFT emits up to nanoseconds.
    value = re.sub(r"\.(\d+)", lambda m: "." + m[1][:6].ljust(6, "0"), value)
    return datetime.datetime.fromisoformat(value.replace("Z", "+00:00"))


def validate_rpc(value):
    if not isinstance(value, dict) or set(value) - {"jsonrpc", "id", "method", "params"}:
        raise ValueError("Invalid RPC envelope")
    method, params = value.get("method"), value.get("params", {})
    if method not in RPC_METHODS or not isinstance(params, dict):
        raise ValueError("RPC method is not public")
    allowed = {"status": set(), "block": {"height"}, "commit": {"height"},
        "validators": {"height", "page", "per_page"}, "abci_query": {"path", "data", "height", "prove"},
        "broadcast_tx_sync": {"tx"}, "broadcast_tx_commit": {"tx"}}[method]
    if set(params) - allowed:
        raise ValueError("Unsupported RPC parameter")
    if method == "abci_query" and (params.get("path") not in QUERY_PATHS or params.get("prove", False)
                                  or params.get("height", 0) not in (0, "0")):
        raise ValueError("Only supported current-state queries are public")
    if method.startswith("broadcast"):
        raw = base64.b64decode(params["tx"], validate=True)
        if len(raw) > 16384:
            raise ValueError("Transaction exceeds 16 KiB")
        protocol.verify(protocol.parse_json(raw))
    if method == "validators":
        if not 1 <= int(params.get("per_page", 30)) <= 100:
            raise ValueError("At most 100 validators per page")
    return method, params


class Gateway:
    def __init__(self, home):
        self.home = Path(home)
        self.config = json.loads((self.home / "node.json").read_text())
        self.rpc = f'http://127.0.0.1:{self.config["base_port"] + 1}'
        self.cache = {}
        self.lock = threading.Lock()
        self.clients = OrderedDict()
        self.history = TrainingHistory(self.home / "explorer.sqlite", self.config["genesis_sha256"])

    def checkpoint(self):
        def load():
            response = client.rpc(self.rpc, "abci_query", {"path": "/task", "prove": False})["response"]
            if response.get("code", 0):
                raise ValueError("Model state unavailable")
            task = protocol.parse_json(base64.b64decode(response["value"]))
            if work.digest(task["weights"]) != task["model_root"]:
                raise ValueError("Checkpoint differs from its model root")
            return {"format": "neuroshard-checkpoint-v1", "chain_id": task["chain_id"],
                    "genesis_sha256": self.config["genesis_sha256"], "state_height": int(response["height"]),
                    "round": task["round"], "model_root": task["model_root"], "weights": task["weights"],
                    "encoding": "JSON tensors: shape and base64 little-endian float32; no pickle"}
        return self.cached("checkpoint", load, 3)

    def model(self):
        summary = self.summary()
        manifest = self.cached("manifest", lambda: client.query(self.rpc, "/manifest"), 60)
        weights = self.checkpoint()["weights"]
        return {"chain_id": summary["chain_id"], "height": summary["height"], "round": summary["round"],
                "model_root": summary["model_root"], "genesis_sha256": summary["genesis_sha256"],
                "parameter_count": sum(math.prod(t["shape"]) for t in weights.values()),
                "execution": manifest["execution"], "last_training_loss":
                float.fromhex(summary["last_training_loss_hex"]) if summary["last_training_loss_hex"] else None,
                "metric": "Last accepted training minibatch cross-entropy, before its update; not held-out quality"}

    def limited(self, ip):
        now = time.monotonic()
        with self.lock:
            started, count = self.clients.pop(ip, (now, 0))
            if now - started >= 60:
                started, count = now, 0
            self.clients[ip] = (started, count + 1)
            if len(self.clients) > 4096:
                self.clients.popitem(last=False)
            return count >= 240

    def cached(self, key, loader, seconds=1):
        with self.lock:
            entry = self.cache.get(key)
            if entry and time.monotonic() - entry[0] < seconds:
                return entry[1]
        value = loader()
        with self.lock:
            if len(self.cache) >= 128:
                self.cache.clear()
            self.cache[key] = (time.monotonic(), value)
        return value

    def summary(self):
        def load():
            summary = client.query(self.rpc, "/summary")
            native = client.rpc(self.rpc, "status")
            checkpoint = self.config["trusted_checkpoint"]
            block_time = parse_block_time(native["sync_info"]["latest_block_time"])
            age = max(0, (datetime.datetime.now(datetime.timezone.utc) - block_time).total_seconds())
            stalled = age > 30
            ready = not native["sync_info"]["catching_up"] and summary["height"] > 0 and not stalled
            if checkpoint:
                if summary["height"] < checkpoint["height"]:
                    ready = False
                else:
                    block = client.rpc(self.rpc, "block", {"height": str(checkpoint["height"])})
                    if block["block_id"]["hash"] != checkpoint["hash"]:
                        raise ValueError("Trusted checkpoint differs from local history")
            public_peers = []
            for peer in self.config.get("peers", []):
                try:
                    public_peers.append(peer_address(peer))
                except ValueError:
                    pass
            summary.update(ready=ready, stalled=stalled, seconds_since_block=round(age, 1), catching_up=native["sync_info"]["catching_up"],
                genesis_sha256=self.config["genesis_sha256"], node_id=self.config["node_id"],
                bootstrap_peers=public_peers,
                latest_block_height=int(native["sync_info"]["latest_block_height"]),
                latest_block_hash=native["sync_info"]["latest_block_hash"],
                latest_block_time=native["sync_info"]["latest_block_time"],
                observation="Local full-node replay; this API does not supply account inclusion proofs")
            for key in ("issued", "burned", "initial_supply"):
                summary[key] = str(summary[key])
            return summary
        return self.cached("summary", load)

    def block(self, height):
        if not 1 <= height < 2 ** 63:
            raise ValueError("Invalid block height")
        value = client.rpc(self.rpc, "block", {"height": str(height)})
        block = value["block"]
        txs = []
        encoded_txs = block["data"].get("txs") or []
        results = client.rpc(self.rpc, "block_results", {"height": str(height)}).get("txs_results") or [] if encoded_txs else []
        if len(results) != len(encoded_txs):
            raise ValueError("Block execution results unavailable")
        for encoded, result in zip(encoded_txs, results):
            raw = base64.b64decode(encoded)
            envelope = protocol.parse_json(raw)
            body = dict(envelope["body"])
            for key in ("amount", "price"):
                if key in body:
                    body[key] = str(body[key])
            import hashlib
            txs.append({"hash": hashlib.sha256(raw).hexdigest().upper(), "kind": body["kind"],
                        "sender": envelope["public_key"], "body": body, "code": int(result.get("code", 0)),
                        "log": result.get("log", "")})
        return {"height": height, "hash": value["block_id"]["hash"], "time": block["header"]["time"],
                "previous_hash": block["header"]["last_block_id"]["hash"],
                "app_hash": block["header"]["app_hash"], "app_state_height": height - 1,
                "transactions": txs}


class BoundedServer(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(self, address, handler):
        self.slots = threading.BoundedSemaphore(16)
        super().__init__(address, handler)

    def get_request(self):
        sock, address = super().get_request()
        sock.settimeout(8)
        return sock, address

    def process_request(self, request, address):
        if not self.slots.acquire(blocking=False):
            request.close()
            return
        try:
            super().process_request(request, address)
        except BaseException:
            self.slots.release()
            raise

    def process_request_thread(self, request, address):
        try:
            super().process_request_thread(request, address)
        finally:
            self.slots.release()


def handler(gateway):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_):
            pass

        def respond(self, code, value, content_type="application/json"):
            raw = value if isinstance(value, bytes) else work.canonical(value)
            self.send_response(code)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(raw)))
            self.send_header("Access-Control-Allow-Origin", "*")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(raw)

        def do_OPTIONS(self):
            self.send_response(204)
            self.send_header("Access-Control-Allow-Origin", "*")
            self.send_header("Access-Control-Allow-Headers", "Content-Type")
            self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
            self.send_header("Content-Length", "0")
            self.end_headers()

        def do_GET(self):
            try:
                if gateway.limited(client_ip(self.client_address[0], self.headers.get("X-Real-IP"))):
                    return self.respond(429, {"error": "Request budget exceeded; retry shortly"})
                parsed = urlparse(self.path)
                path, query = parsed.path, parse_qs(parsed.query)
                if path in ("/api/network", "/healthz"):
                    value = gateway.summary()
                    return self.respond(200 if path != "/healthz" or value["ready"] else 503, value)
                if path == "/network/genesis.json":
                    return self.respond(200, (gateway.home / "config/genesis.json").read_bytes())
                if path == "/network/corpus.txt":
                    return self.respond(200, (gateway.home / "corpus.txt").read_bytes(), "text/plain; charset=utf-8")
                if path == "/api/manifest":
                    return self.respond(200, gateway.cached("manifest", lambda: client.query(gateway.rpc, "/manifest"), 60))
                if path == "/api/model":
                    return self.respond(200, gateway.model())
                if path == "/api/model/checkpoint.json":
                    return self.respond(200, gateway.checkpoint())
                if path == "/api/training":
                    value = gateway.history.page(int(query.get("limit", ["50"])[0]),
                                                 int(query.get("before", [str(2 ** 63 - 1)])[0]))
                    value.update(chain_id=gateway.config["chain_id"], chain_height=gateway.summary()["height"])
                    return self.respond(200, value)
                if path == "/api/validators":
                    value = client.query(gateway.rpc, "/validators", {"after": query.get("after", [""])[0],
                                                                          "limit": int(query.get("limit", ["100"])[0])})
                    for validator in value["validators"]:
                        validator["bond"] = str(validator["bond"])
                    return self.respond(200, value)
                if path == "/api/account":
                    value = client.query(gateway.rpc, "/account", {"public_key": query["public_key"][0]})
                    value["balance"] = str(value["balance"])
                    return self.respond(200, value)
                if re.fullmatch(r"/api/block/[1-9][0-9]{0,18}", path):
                    return self.respond(200, gateway.block(int(path.rsplit("/", 1)[1])))
                if path == "/api/blocks":
                    count = int(query.get("limit", ["10"])[0])
                    if not 1 <= count <= 20:
                        raise ValueError("Block page limit must be 1–20")
                    current = gateway.summary()["height"]
                    end = min(current, int(query.get("before", [str(current + 1)])[0]) - 1)
                    if end < 0:
                        raise ValueError("Invalid block cursor")
                    value = gateway.cached(f"blocks:{end}:{count}", lambda: [gateway.block(h)
                        for h in range(end, max(0, end - count), -1)])
                    return self.respond(200, {"blocks": value, "next_before": value[-1]["height"] if value else None})
                return self.respond(404, {"error": "Unknown public endpoint"})
            except (ValueError, KeyError, TypeError, OverflowError) as exc:
                self.respond(400, {"error": str(exc)})
            except OSError:
                self.respond(503, {"error": "Full node unavailable; retry after it reconnects"})

        def do_POST(self):
            try:
                if self.path != "/rpc":
                    return self.respond(404, {"error": "Unknown public endpoint"})
                if gateway.limited(client_ip(self.client_address[0], self.headers.get("X-Real-IP"))):
                    return self.respond(429, {"error": "Request budget exceeded"})
                length = int(self.headers.get("Content-Length", "0"))
                if not 0 < length <= 32768 or self.headers.get("Transfer-Encoding"):
                    raise ValueError("RPC request size or encoding is unsupported")
                raw = self.rfile.read(length)
                if len(raw) != length:
                    raise ValueError("Truncated request")
                request = protocol.parse_json(raw)
                method, params = validate_rpc(request)
                result = client.rpc(gateway.rpc, method, params)
                return self.respond(200, {"jsonrpc": "2.0", "id": request.get("id"), "result": result})
            except (ValueError, KeyError, TypeError, OverflowError, RecursionError) as exc:
                return self.respond(400, {"error": str(exc)})
            except OSError:
                return self.respond(503, {"error": "Full node unavailable"})
    return Handler


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--home", type=Path, required=True)
    args = parser.parse_args()
    gateway = Gateway(args.home)
    server = BoundedServer((gateway.config["api_host"], gateway.config["base_port"] + 3), handler(gateway))
    stop = threading.Event()
    follower = threading.Thread(target=gateway.history.follow, args=(gateway, stop), daemon=True)
    follower.start()
    try:
        server.serve_forever()
    finally:
        stop.set()
        follower.join(timeout=35)
        server.server_close()


if __name__ == "__main__":
    main()
