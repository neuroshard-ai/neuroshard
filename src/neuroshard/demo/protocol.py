"""Pure application transitions for the development NeuroShard chain."""

import copy
import hashlib
import json
import os
import secrets
from pathlib import Path

from neuroshard.core.crypto.ecdsa import derive_keypair_from_token, ecdsa_sign, ecdsa_verify
from neuroshard.demo import work


def address(public_key):
    return hashlib.sha256(bytes.fromhex(public_key)).hexdigest()[:32]


class Identity:
    def __init__(self, token):
        self.key = derive_keypair_from_token(token)
        self.public_key = self.key.public_key_bytes.hex()
        self.address = self.key.node_id

    @classmethod
    def load_or_create(cls, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        try:
            with path.open("x") as f:
                os.chmod(path, 0o600)
                f.write(secrets.token_hex(32))
        except FileExistsError:
            pass
        return cls(path.read_text().strip())

    def sign(self, body):
        return {"body": body, "public_key": self.public_key,
                "signature": ecdsa_sign(work.canonical(body).decode(), self.key.private_key_bytes)}


def verify(envelope):
    if not isinstance(envelope, dict) or set(envelope) != {"body", "public_key", "signature"}:
        raise ValueError("Invalid signed envelope")
    public = envelope["public_key"]
    if not isinstance(public, str) or len(public) != 66:
        raise ValueError("Invalid public key")
    if not isinstance(envelope["signature"], str) or len(envelope["signature"]) > 160:
        raise ValueError("Invalid signature encoding")
    if not isinstance(envelope["body"], dict):
        raise ValueError("Invalid signed body")
    if not ecdsa_verify(work.canonical(envelope["body"]).decode(), envelope["signature"],
                        bytes.fromhex(public)):
        raise ValueError("Signature verification failed")
    return envelope["body"], public


def parse_json(raw):
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError("Duplicate JSON key")
            result[key] = value
        return result

    def bad_constant(_value):
        raise ValueError("Nonfinite JSON number")

    if len(raw) > work.MAX_MESSAGE_BYTES:
        raise ValueError("Message exceeds the demo limit")
    return json.loads(raw, object_pairs_hook=pairs, parse_constant=bad_constant)


def genesis_state(chain_id, spec):
    weights = work.encode_weights(work.make_model())
    return {"chain_id": chain_id, "manifest": spec, "height": 0, "round": 0,
            "weights": weights, "model_root": work.digest(weights), "lease": None,
            "balances": {}, "total_issued": 0, "completed": [], "seen": [],
            "training_loss_hex": None}


def transaction_id(envelope):
    # ECDSA signatures may differ for the same message; identity is the signed content.
    return work.digest({"body": envelope["body"], "public_key": envelope["public_key"]})


def transition(state, envelope, height, replay_fn):
    body, owner = verify(envelope)
    if body.get("chain_id") != state["chain_id"]:
        raise ValueError("Wrong chain")
    tx_id = transaction_id(envelope)
    if tx_id in state["seen"]:
        raise ValueError("Transaction already applied")
    if type(body.get("round")) is not int or body["round"] != state["round"]:
        raise ValueError("Wrong training round")
    if body.get("parent") != state["model_root"]:
        raise ValueError("Wrong parent checkpoint")
    if state["round"] >= state["manifest"]["max_rewarded_tasks"]:
        raise ValueError("Development emission budget exhausted")
    result = copy.deepcopy(state)
    if body.get("kind") == "claim":
        if set(body) != {"kind", "chain_id", "round", "parent", "stages", "nonce"}:
            raise ValueError("Unexpected claim fields")
        if not isinstance(body["nonce"], str) or len(body["nonce"]) != 32:
            raise ValueError("Invalid claim nonce")
        bytes.fromhex(body["nonce"])
        stages = body["stages"]
        if not isinstance(stages, list) or len(stages) != 2:
            raise ValueError("A job needs exactly two stage identities")
        for public in stages:
            if not isinstance(public, str) or len(public) != 66:
                raise ValueError("Invalid stage identity")
            # Validate the point as well as its encoding before accepting a lease.
            from cryptography.hazmat.primitives.asymmetric import ec
            ec.EllipticCurvePublicKey.from_encoded_point(ec.SECP256K1(), bytes.fromhex(public))
        lease = state["lease"]
        if lease and height <= lease["expires"]:
            raise ValueError("Current task is already assigned")
        result["lease"] = {"task_id": tx_id, "owner": owner, "stages": stages,
                           "expires": height + state["manifest"]["lease_blocks"]}
    elif body.get("kind") == "train":
        if set(body) != {"kind", "chain_id", "round", "parent", "task_id",
                         "result_root", "receipts"}:
            raise ValueError("Unexpected training claim fields")
        lease = state["lease"]
        if not lease or lease["task_id"] != body["task_id"] or lease["owner"] != owner:
            raise ValueError("Task is not assigned to this claimant")
        if height > lease["expires"]:
            raise ValueError("Task lease expired")
        if not isinstance(body["receipts"], list) or len(body["receipts"]) != 2:
            raise ValueError("Missing stage receipts")
        expected = replay_fn(state)
        if body["result_root"] != work.digest(expected["weights"]):
            raise ValueError("Training result differs from independent replay")
        for i, receipt in enumerate(body["receipts"]):
            receipt_body, public = verify(receipt)
            if (public != lease["stages"][i]
                    or work.canonical(receipt_body) != work.canonical(expected["receipts"][i])):
                raise ValueError("Stage computation differs from independent replay")
        reward = state["manifest"]["reward_atoms"]
        shares = [reward // 2, reward - reward // 2]
        for public, amount in zip(lease["stages"], shares):
            recipient = address(public)
            result["balances"][recipient] = result["balances"].get(recipient, 0) + amount
        result["total_issued"] += reward
        result["weights"] = expected["weights"]
        result["model_root"] = body["result_root"]
        result["training_loss_hex"] = float(expected["loss"]).hex()
        result["completed"].append(body["task_id"])
        result["round"] += 1
        result["lease"] = None
    else:
        raise ValueError("Unknown transaction kind")
    result["seen"].append(tx_id)
    assert result["total_issued"] == sum(result["balances"].values())
    assert result["total_issued"] == result["round"] * state["manifest"]["reward_atoms"]
    return result
