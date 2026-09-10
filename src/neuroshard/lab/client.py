"""Transactions for the candidate chain; no privileged admission endpoint."""

import base64
import json
from pathlib import Path

from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from neuroshard.demo import client as wire, protocol, work
from neuroshard.lab import state


def transaction(url, identity, kind, **fields):
    status = wire.query(url, "/summary")
    nonce = wire.query(url, "/account", {"public_key": identity.public_key})["nonce"]
    return identity.sign({"kind": kind, "chain_id": status["chain_id"], "nonce": nonce, **fields})


def consensus_identity(node_home):
    key = json.loads((Path(node_home) / "config/priv_validator_key.json").read_text())
    public = base64.b64decode(key["pub_key"]["value"]).hex()
    secret = Ed25519PrivateKey.from_private_bytes(base64.b64decode(key["priv_key"]["value"])[:32])
    return public, secret


def bond(url, identity, node_home, amount=state.PARAMS["bond_unit"]):
    public, secret = consensus_identity(node_home)
    status = wire.query(url, "/summary")
    nonce = wire.query(url, "/account", {"public_key": identity.public_key})["nonce"]
    proof = secret.sign(state.possession_message(status["chain_id"], identity.public_key, public, amount, nonce)).hex()
    tx = transaction(url, identity, "bond", consensus_key=public, amount=amount, possession=proof)
    return wire.broadcast(url, tx), public


def reserve(url, identity, workers, kind="train", request=None, price=0):
    status = wire.query(url, "/summary")
    tx = transaction(url, identity, "reserve", task_kind=kind, parent=status["model_root"], round=status["round"],
                     workers=workers, request=request or {}, price=price)
    return wire.broadcast(url, tx)


def training_claim(url, workers, identity):
    task = wire.query(url, "/task")
    lease = task["lease"]
    if not lease or lease["owner"] != identity.public_key or lease["task_kind"] != "train":
        raise ValueError("No training reservation for this miner")
    common = {"task_id": lease["task_id"], "input_ids": task["input_ids"]}
    first = {**common, "weights": work.stage_weights(task["weights"], 0), "operation": "forward"}
    activation = wire.http(workers[0] + "/compute", first)["activation"]
    second = wire.http(workers[1] + "/compute", {**common, "weights": work.stage_weights(task["weights"], 1),
                        "operation": "backward", "activation": activation})
    first.update(operation="backward", adjoint=second["adjoint"], loss_hex=second["receipt"]["body"]["loss_hex"])
    backward = wire.http(workers[0] + "/compute", first)
    weights = work.apply_gradients(task["weights"], {**backward["gradients"], **second["gradients"]})
    return transaction(url, identity, "submit", task_id=lease["task_id"], result_root=work.digest(weights),
                       receipts=[backward["receipt"], second["receipt"]])


def mine(url, workers, identity):
    public = [wire.http(worker + "/identity")["public_key"] for worker in workers]
    reserve(url, identity, public)
    tx = training_claim(url, workers, identity)
    wire.broadcast(url, tx)
    return tx
