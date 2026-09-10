"""Unprivileged miner and read client for the reference native chain."""

import base64
import json
import secrets
import time
from urllib.request import Request, urlopen

from neuroshard.demo import protocol, work


class Rejected(ValueError):
    pass


def http(url, body=None, timeout=30):
    request = Request(url, data=work.canonical(body) if body is not None else None,
                      headers={"Content-Type": "application/json"})
    with urlopen(request, timeout=timeout) as response:
        raw = response.read(work.MAX_MESSAGE_BYTES + 1)
    return protocol.parse_json(raw)


def rpc(url, method, params=None, timeout=30):
    response = http(url, {"jsonrpc": "2.0", "id": 1, "method": method,
                          "params": params or {}}, timeout=timeout)
    if "error" in response:
        raise Rejected(json.dumps(response["error"]))
    return response["result"]


def query(url, path="/status", data=None):
    params = {"path": path, "prove": False}
    if data is not None:
        params["data"] = work.canonical(data).hex()
    response = rpc(url, "abci_query", params)["response"]
    if response.get("code", 0):
        raise Rejected(response.get("log", "Query rejected"))
    return protocol.parse_json(base64.b64decode(response["value"]))


def broadcast(url, envelope, wait=True):
    result = rpc(url, "broadcast_tx_commit" if wait else "broadcast_tx_sync",
                 {"tx": base64.b64encode(work.canonical(envelope)).decode()})
    for receipt in (result, result.get("check_tx", {}), result.get("tx_result", {})):
        if receipt.get("code", 0):
            raise Rejected(receipt.get("log", "Transaction rejected"))
    return result


def claim_body(status, stage_keys):
    return {"kind": "claim", "chain_id": status["chain_id"], "round": status["round"],
            "parent": status["model_root"], "stages": stage_keys, "nonce": secrets.token_hex(16)}


def prepare_training(url, workers, identity):
    stage_ids = [http(worker + "/identity") for worker in workers]
    if [stage["stage"] for stage in stage_ids] != [0, 1]:
        raise ValueError("Provide the stage-zero and stage-one workers in order")
    status = query(url)
    lease = status["lease"]
    if not (lease and lease["owner"] == identity.public_key
            and status["height"] < lease["expires"]):
        broadcast(url, identity.sign(claim_body(status, [s["public_key"] for s in stage_ids])))
    task = query(url, "/task")
    if not task["lease"] or task["lease"]["owner"] != identity.public_key:
        raise Rejected("Task belongs to a different claimant")
    if task["lease"]["stages"] != [s["public_key"] for s in stage_ids]:
        raise Rejected("Lease binds different workers")
    common = {"task_id": task["lease"]["task_id"], "input_ids": task["input_ids"]}
    first = {**common, "weights": work.stage_weights(task["weights"], 0), "operation": "forward"}
    activation = http(workers[0] + "/compute", first)["activation"]
    second = http(workers[1] + "/compute", {
        **common, "weights": work.stage_weights(task["weights"], 1),
        "operation": "backward", "activation": activation})
    first.update(operation="backward", adjoint=second["adjoint"],
                 loss_hex=second["receipt"]["body"]["loss_hex"])
    backward = http(workers[0] + "/compute", first)
    weights = work.apply_gradients(task["weights"], {**backward["gradients"], **second["gradients"]})
    body = {"kind": "train", "chain_id": task["chain_id"], "round": task["round"],
            "parent": task["model_root"], "task_id": task["lease"]["task_id"],
            "result_root": work.digest(weights), "receipts": [backward["receipt"], second["receipt"]]}
    return identity.sign(body)


def mine(url, workers, identity, steps=1):
    results = []
    for _ in range(steps):
        started = time.monotonic()
        envelope = prepare_training(url, workers, identity)
        receipt = broadcast(url, envelope)
        status = query(url)
        results.append({"round": status["round"], "height": receipt["height"],
                        "model_root": status["model_root"], "total_issued": status["total_issued"],
                        "validation_loss": status["validation_loss"],
                        "elapsed_seconds": round(time.monotonic() - started, 3)})
    return results
