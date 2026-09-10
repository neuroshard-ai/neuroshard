"""Optional task sponsorship with outbound workers; consensus remains on the chain.

The coordinator pays reservation collateral and selects workers. Anyone can run
one. It cannot approve rewards: validators replay every submitted computation.
"""

from collections import OrderedDict
import json
from pathlib import Path
import threading
import time
from urllib.parse import urlparse

from neuroshard.demo import client as wire, protocol, work
from neuroshard.lab import client
from neuroshard.publicnet.gateway import BoundedServer, Gateway, handler, client_ip


DOMAIN = "neuroshard/outbound-work/v1"


def signed(identity, chain_id, kind, **fields):
    return identity.sign({"domain": DOMAIN, "chain_id": chain_id, "kind": kind,
                          "time": int(time.time()), **fields})


def authenticate(envelope, chain_id, kind):
    body, key = protocol.verify(envelope)
    if (body.get("domain") != DOMAIN or body.get("chain_id") != chain_id
            or body.get("kind") != kind or type(body.get("time")) is not int
            or abs(time.time() - body["time"]) > 60):
        raise ValueError("Foreign or stale work message")
    return body, key


def coordinator_url(value):
    parsed = urlparse(value)
    if parsed.username or parsed.password or parsed.query or parsed.fragment:
        raise ValueError("Use a plain coordinator HTTPS URL")
    if parsed.scheme != "https" and not (parsed.scheme == "http" and parsed.hostname in ("127.0.0.1", "localhost")):
        raise ValueError("Coordinators require HTTPS, except on loopback")
    return value.rstrip("/")


def authorized_request(envelope, identity, stage, task, summary):
    body, sponsor = authenticate(envelope, task["chain_id"], "assignment")
    lease = task["lease"]
    if (not lease or lease["task_kind"] != "train" or sponsor != lease["owner"]
            or body.get("worker") != identity.public_key or body.get("stage") != stage
            or lease["workers"][stage] != identity.public_key
            or summary["chain_id"] != task["chain_id"] or summary["lease"] != lease
            or summary["height"] > lease["expires"]):
        raise ValueError("Assignment is not authorized by the local finalized lease")
    request = body["request"]
    operation = request.get("operation")
    allowed = {"task_id", "input_ids", "weights", "operation"}
    if stage == 1:
        allowed.add("activation")
    elif operation == "backward":
        allowed.update(("adjoint", "loss_hex"))
    if (set(request) != allowed or request["task_id"] != lease["task_id"]
            or operation not in (("forward", "backward") if stage == 0 else ("backward",))
            or request["input_ids"] != task["input_ids"]
            or request["weights"] != work.stage_weights(task["weights"], stage)):
        raise ValueError("Assignment differs from the local model, batch, or stage")
    return request


class Worker:
    def __init__(self, home, stage):
        self.home, self.stage_index = Path(home), stage
        self.config = json.loads((self.home / "node.json").read_text())
        self.identity = protocol.Identity.load_or_create(self.home / "account.key")
        self.rpc = f'http://127.0.0.1:{self.config["base_port"] + 1}'
        self.stage = work.Stage(stage)
        self.path = self.home / f"worker-stage-{stage}.json"
        self.journal = json.loads(self.path.read_text()) if self.path.exists() else {"task_id": None, "operations": {}}

    def save(self):
        temporary = self.path.with_suffix(".tmp")
        temporary.write_bytes(work.canonical(self.journal))
        temporary.replace(self.path)

    def compute(self, envelope):
        task, summary = wire.query(self.rpc, "/task"), wire.query(self.rpc, "/summary")
        if task["chain_id"] != self.config["chain_id"]:
            raise ValueError("Local node changed chain")
        request = authorized_request(envelope, self.identity, self.stage_index, task, summary)
        task_id, operation = request["task_id"], request["operation"]
        if self.journal["task_id"] != task_id:
            self.journal = {"task_id": task_id, "operations": {}}
        digest = work.digest(request)
        cached = self.journal["operations"].get(operation)
        if cached:
            if cached["digest"] != digest:
                raise ValueError("Sponsor changed an already assigned operation")
            if "result" not in cached:
                raise ValueError("Interrupted computation requires a new lease")
            return cached["result"]
        # Persist intent first. A crash cannot turn retries into unlimited work.
        self.journal["operations"][operation] = {"digest": digest}
        self.save()
        result = self.stage.compute(request)
        if "receipt" in result:
            result["receipt"] = self.identity.sign(result["receipt"])
        self.journal["operations"][operation]["result"] = result
        self.save()
        return result

    def run(self, url, stop=None, max_tasks=0):
        url = coordinator_url(url)
        stop = stop or threading.Event()
        delivered = set()
        while not stop.is_set():
            try:
                poll = signed(self.identity, self.config["chain_id"], "poll", stage=self.stage_index)
                assignment = wire.http(url + "/work/poll", poll, timeout=10).get("assignment")
                if assignment:
                    result = self.compute(assignment)
                    wire.http(url + "/work/result", signed(self.identity, self.config["chain_id"], "result",
                        job_id=work.digest(assignment["body"]), result=result), timeout=10)
                    if assignment["body"]["request"]["operation"] == "backward":
                        delivered.add(assignment["body"]["request"]["task_id"])
                        if max_tasks and len(delivered) >= max_tasks:
                            return
            except (OSError, ValueError, KeyError, TypeError) as error:
                print(json.dumps({"worker_error": str(error)}), flush=True)
            stop.wait(1)


class Coordinator:
    def __init__(self, home):
        self.home = Path(home)
        self.gateway = Gateway(home)
        self.config, self.rpc = self.gateway.config, self.gateway.rpc
        self.identity = protocol.Identity.load_or_create(self.home / "account.key")
        self.lock = threading.Condition()
        self.workers, self.jobs = OrderedDict(), {}
        self.remaining = None

    def poll(self, envelope):
        body, key = authenticate(envelope, self.config["chain_id"], "poll")
        stage = body.get("stage")
        if type(stage) is not int or stage not in (0, 1):
            raise ValueError("Unknown stage")
        with self.lock:
            self.workers[(key, stage)] = time.monotonic()
            if len(self.workers) > 128:
                self.workers.popitem(last=False)
            job = self.jobs.get((key, stage))
            return {"assignment": job["assignment"] if job else None}

    def result(self, envelope):
        body, key = authenticate(envelope, self.config["chain_id"], "result")
        with self.lock:
            for (owner, _), job in self.jobs.items():
                if owner == key and body.get("job_id") == work.digest(job["assignment"]["body"]):
                    if "result" not in job:
                        job["result"] = body["result"]
                    self.lock.notify_all()
                    return {"accepted_for_submission": True}
        raise ValueError("No matching pending assignment")

    def compute(self, worker, stage, request):
        assignment = signed(self.identity, self.config["chain_id"], "assignment", worker=worker, stage=stage, request=request)
        job = {"assignment": assignment}
        with self.lock:
            self.jobs[(worker, stage)] = job
            try:
                if not self.lock.wait_for(lambda: "result" in job, timeout=35):
                    raise TimeoutError("Worker did not return the assigned operation; reservation may expire")
                return job["result"]
            finally:
                self.jobs.pop((worker, stage), None)

    def mine(self, wait_seconds=180):
        deadline = time.monotonic() + wait_seconds
        selected = []
        while len(selected) != 2:
            with self.lock:
                selected = [next((key for (key, s), seen in self.workers.items()
                    if s == stage and time.monotonic() - seen < 10), None) for stage in (0, 1)]
            if all(selected):
                break
            selected = []
            if wait_seconds and time.monotonic() > deadline:
                raise TimeoutError("Waiting for one outbound worker per stage; no collateral reserved")
            time.sleep(0.5)
        client.reserve(self.rpc, self.identity, selected)
        task = wire.query(self.rpc, "/task")
        if not task["lease"] or task["lease"]["owner"] != self.identity.public_key:
            raise ValueError("Reservation no longer available")
        common = {"task_id": task["lease"]["task_id"], "input_ids": task["input_ids"]}
        first = {**common, "weights": work.stage_weights(task["weights"], 0), "operation": "forward"}
        activation = self.compute(selected[0], 0, first)["activation"]
        second = self.compute(selected[1], 1, {**common, "weights": work.stage_weights(task["weights"], 1),
            "operation": "backward", "activation": activation})
        backward = self.compute(selected[0], 0, {**first, "operation": "backward", "adjoint": second["adjoint"],
            "loss_hex": second["receipt"]["body"]["loss_hex"]})
        weights = work.apply_gradients(task["weights"], {**backward["gradients"], **second["gradients"]})
        result = wire.broadcast(self.rpc, client.transaction(self.rpc, self.identity, "submit", task_id=common["task_id"],
            result_root=work.digest(weights), receipts=[backward["receipt"], second["receipt"]]))
        with self.lock:
            for key in list(self.workers):
                if key[0] in selected:
                    # Availability for the next lease must be expressed after settlement.
                    self.workers.pop(key)
        return result

    def server(self, host, port):
        parent = handler(self.gateway)
        coordinator = self

        class Handler(parent):
            def do_GET(self):
                if self.path != "/work/status":
                    return self.respond(404, {"error": "Unknown work endpoint"})
                if coordinator.gateway.limited(client_ip(self.client_address[0], self.headers.get("X-Real-IP"))):
                    return self.respond(429, {"error": "Request budget exceeded"})
                with coordinator.lock:
                    recent = [stage for (_, stage), seen in coordinator.workers.items()
                              if time.monotonic() - seen < 10]
                    value = {"chain_id": coordinator.config["chain_id"], "sponsor": coordinator.identity.public_key,
                        "remaining_tasks": coordinator.remaining, "active_operations": len(coordinator.jobs),
                        "workers_by_stage": [recent.count(0), recent.count(1)],
                        "payment": "Conditional on completed work and finalized validator acceptance"}
                return self.respond(200, value)

            def do_POST(self):
                try:
                    if coordinator.gateway.limited(client_ip(self.client_address[0], self.headers.get("X-Real-IP"))):
                        return self.respond(429, {"error": "Request budget exceeded"})
                    if self.path not in ("/work/poll", "/work/result"):
                        return self.respond(404, {"error": "Unknown work endpoint"})
                    length = int(self.headers.get("Content-Length", "0"))
                    limit = 4096 if self.path == "/work/poll" else work.MAX_MESSAGE_BYTES
                    if not 0 < length <= limit or self.headers.get("Transfer-Encoding"):
                        raise ValueError("Invalid request size or encoding")
                    envelope = protocol.parse_json(self.rfile.read(length))
                    value = coordinator.poll(envelope) if self.path == "/work/poll" else coordinator.result(envelope)
                    return self.respond(200, value)
                except (ValueError, KeyError, TypeError, OverflowError, RecursionError) as error:
                    return self.respond(400, {"error": str(error)})
        return BoundedServer((host, port), Handler)
