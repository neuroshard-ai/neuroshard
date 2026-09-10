"""Candidate v2: accounts, bonded membership, paid jobs, and objective expiry.

This is a deliberately bounded full-replay execution profile. It does not
enable the experimental integer verifier for floating-point training.
"""

import copy
import hashlib

from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey

from neuroshard.demo import protocol, work


PARAMS = {"atom_scale": 1_000_000, "fee": 1000, "bond_unit": 250_000,
          "epoch_blocks": 8, "activation_blocks": 4, "evidence_blocks": 24,
          "evidence_seconds": 6, "lease_blocks": 16, "reservation_bond": 2_000_000,
          "task_budget": 1_000_000, "max_training_tasks": 1000,
          "worker_budget": 800_000, "max_tx_bytes": 16_384}


def integer(value, minimum=0, maximum=2 ** 60):
    if type(value) is not int or not minimum <= value <= maximum:
        raise ValueError("Integer outside protocol bounds")
    return value


def public_key(value):
    if not isinstance(value, str) or len(value) != 66 or value != value.lower():
        raise ValueError("Noncanonical account key")
    from cryptography.hazmat.primitives.asymmetric import ec
    ec.EllipticCurvePublicKey.from_encoded_point(ec.SECP256K1(), bytes.fromhex(value))
    return value


def consensus_address(public):
    return hashlib.sha256(bytes.fromhex(public)).hexdigest()[:40].upper()


def parameters(state):
    return state["manifest"].get("params", PARAMS)


def next_epoch(height, params=None):
    params = PARAMS if params is None else params
    return ((height + params["epoch_blocks"] - 1) // params["epoch_blocks"]) * params["epoch_blocks"]


def genesis(chain_id, validators, execution_manifest):
    """Validator entries contain public account and consensus keys only."""
    params = execution_manifest.get("params", PARAMS)
    weights = work.encode_weights(work.make_model())
    state = {"chain_id": chain_id, "height": 0, "time_ns": 0, "manifest": execution_manifest,
             "accounts": {}, "validators": {}, "evidence_seen": [], "round": 0,
             "weights": weights, "model_root": work.digest(weights), "lease": None,
             "issued": 0, "burned": 0, "initial_supply": 0, "completed": [],
             "pending_verifier_reward": None, "last_training_loss_hex": None, "last_inference": None}
    for entry in validators:
        owner = public_key(entry["owner"])
        key = entry["consensus_key"]
        Ed25519PublicKey.from_public_bytes(bytes.fromhex(key))
        if len(key) != 64 or key in state["validators"]:
            raise ValueError("Invalid genesis validator identity")
        amount, liquid = integer(entry["bond"], params["bond_unit"]), integer(entry["liquid"])
        if amount % params["bond_unit"]:
            raise ValueError("Bond must be a multiple of the voting unit")
        account(state, owner)["balance"] += liquid
        state["validators"][key] = {"owner": owner, "amount": amount, "status": "active",
            "history": [[1, amount // params["bond_unit"]]], "emit_at": None,
            "removed_at": None, "release_height": None, "release_time_ns": None}
        state["initial_supply"] += amount + liquid
    if not state["validators"]:
        raise ValueError("Genesis requires nonzero bonded consensus weight")
    invariant(state)
    return state


def account(state, owner):
    return state["accounts"].setdefault(owner, {"balance": 0, "nonce": 0})


def voting_power(state, height):
    powers = {}
    for key, validator in state["validators"].items():
        value = 0
        for effective, power in validator["history"]:
            if effective <= height:
                value = power
        if value:
            powers[key] = value
    return powers


def invariant(state):
    params = parameters(state)
    liquid = sum(a["balance"] for a in state["accounts"].values())
    bonds = sum(v["amount"] for v in state["validators"].values())
    escrow = state["lease"]["escrow"] if state["lease"] else 0
    escrow += state["pending_verifier_reward"]["budget"] if state["pending_verifier_reward"] else 0
    assert all(a["balance"] >= 0 and a["nonce"] >= 0 for a in state["accounts"].values())
    assert all(v["amount"] >= 0 for v in state["validators"].values())
    assert state["initial_supply"] + state["issued"] == liquid + bonds + escrow + state["burned"]
    assert state["issued"] == state["round"] * params["task_budget"]
    assert state["round"] <= params["max_training_tasks"]


def advance(previous, height, time_ns, evidence=(), committers=None):
    """Deterministic block boundary. Evidence is supplied only by the consensus engine."""
    params = parameters(previous)
    state = copy.deepcopy(previous)
    if height != state["height"] + 1 or time_ns < state["time_ns"]:
        raise ValueError("Nonmonotonic block height or time")
    state["height"], state["time_ns"] = height, time_ns
    updates = {}
    pending = state["pending_verifier_reward"]
    if pending and committers is not None:
        powers = pending["powers"]
        total = sum(powers.values())
        signed = {key: power for key, power in powers.items() if consensus_address(key) in committers}
        if pending["height"] != height - 1 or 3 * sum(signed.values()) <= 2 * total:
            raise ValueError("Missing commit quorum for deferred verifier payment")
        paid = 0
        for key, power in signed.items():
            amount = pending["budget"] * power // total
            account(state, state["validators"][key]["owner"])["balance"] += amount
            paid += amount
        state["burned"] += pending["budget"] - paid
        state["pending_verifier_reward"] = None
    for item in evidence:
        event = f'{item["kind"]}:{item["address"]}:{item["height"]}'
        if event in state["evidence_seen"]:
            continue
        if item["kind"] not in (1, 2) or not 1 <= item["height"] < height:
            raise ValueError("Invalid consensus evidence metadata")
        for key, validator in state["validators"].items():
            if consensus_address(key) != item["address"]:
                continue
            historical = voting_power(state, item["height"]).get(key, 0)
            if not historical:
                raise ValueError("Evidence names a validator inactive at the offense height")
            if validator["status"] == "withdrawn":
                # Correct evidence-age/unbond parameters make this unreachable for valid fresh evidence.
                raise ValueError("Fresh evidence outlived reserved collateral")
            penalty = (validator["amount"] + 3) // 4
            validator["amount"] -= penalty
            state["burned"] += penalty
            if validator["status"] not in ("cooldown", "jailed"):
                validator.update(status="jailed", emit_at=None, removed_at=height + 2,
                                 release_height=None, release_time_ns=None)
                validator["history"].append([height + 2, 0])
                updates[key] = 0
            state["evidence_seen"].append(event)
            break
        else:
            raise ValueError("Evidence names an unknown validator")
    for key, validator in state["validators"].items():
        if validator["emit_at"] == height:
            if validator["status"] == "pending":
                power = validator["amount"] // params["bond_unit"]
                validator.update(status="active", emit_at=None)
                validator["history"].append([height + 2, power])
                updates[key] = power
            elif validator["status"] == "leaving":
                validator.update(status="cooldown", emit_at=None, removed_at=height + 2)
                validator["history"].append([height + 2, 0])
                updates[key] = 0
        if validator["removed_at"] == height and validator["release_height"] is None:
            # This block's time is no earlier than the last height at which it could vote.
            validator["release_height"] = height - 1 + params["evidence_blocks"] + 1
            validator["release_time_ns"] = time_ns + params["evidence_seconds"] * 1_000_000_000
    lease = state["lease"]
    if lease and height > lease["expires"]:
        # A reservation that prevents useful work has an objective, collectible cost.
        state["burned"] += params["reservation_bond"]
        account(state, lease["owner"])["balance"] += lease["escrow"] - params["reservation_bond"]
        state["lease"] = None
    if updates and not voting_power(state, height + 2):
        raise ValueError("Validator updates would remove all consensus weight")
    invariant(state)
    return state, updates


def possession_message(chain_id, owner, consensus_key, amount, nonce):
    return work.canonical({"domain": "neuroshard/validator-bond/v2", "chain_id": chain_id,
                           "owner": owner, "consensus_key": consensus_key, "amount": amount, "nonce": nonce})


def transition(previous, envelope, execute):
    params = parameters(previous)
    body, owner = protocol.verify(envelope)
    public_key(owner)
    if body.get("chain_id") != previous["chain_id"]:
        raise ValueError("Wrong chain")
    nonce = integer(body.get("nonce"), 0)
    if nonce != previous["accounts"].get(owner, {"nonce": 0})["nonce"]:
        raise ValueError("Wrong account nonce")
    kind = body.get("kind")
    extra = {"transfer": {"to", "amount"}, "bond": {"consensus_key", "amount", "possession"},
             "unbond": {"consensus_key"}, "withdraw": {"consensus_key"},
             "reserve": {"task_kind", "parent", "round", "workers", "request", "price"},
             "submit": {"task_id", "result_root", "receipts"}}
    if kind not in extra or set(body) != {"kind", "chain_id", "nonce"} | extra[kind]:
        raise ValueError("Invalid transaction schema")
    state = copy.deepcopy(previous)
    sender = account(state, owner)
    fee = params["fee"]
    if sender["balance"] < fee:
        raise ValueError("Insufficient transaction fee")
    sender["balance"] -= fee
    sender["nonce"] += 1
    state["burned"] += fee

    def debit(amount):
        if sender["balance"] < amount:
            raise ValueError("Insufficient spendable balance")
        sender["balance"] -= amount

    if kind == "transfer":
        recipient = public_key(body["to"])
        amount = integer(body["amount"], 1)
        debit(amount)
        account(state, recipient)["balance"] += amount
    elif kind == "bond":
        key, amount = body["consensus_key"], integer(body["amount"], params["bond_unit"])
        if (not isinstance(key, str) or len(key) != 64 or key != key.lower()
                or key in state["validators"] or amount % params["bond_unit"]):
            raise ValueError("Invalid or previously used consensus key/bond")
        try:
            Ed25519PublicKey.from_public_bytes(bytes.fromhex(key)).verify(bytes.fromhex(body["possession"]),
                possession_message(state["chain_id"], owner, key, amount, nonce))
        except Exception as exc:
            raise ValueError("Consensus-key possession proof failed") from exc
        debit(amount)
        state["validators"][key] = {"owner": owner, "amount": amount, "status": "pending", "history": [],
            "emit_at": next_epoch(state["height"] + params["activation_blocks"], params),
            "removed_at": None, "release_height": None, "release_time_ns": None}
    elif kind in ("unbond", "withdraw"):
        key = body["consensus_key"]
        if key not in state["validators"] or state["validators"][key]["owner"] != owner:
            raise ValueError("Consensus bond belongs to a different account")
        validator = state["validators"][key]
        if kind == "unbond":
            if validator["status"] != "active":
                raise ValueError("Only an active bond can start withdrawal")
            remaining = [v for k, v in state["validators"].items()
                         if k != key and v["status"] == "active"]
            if not remaining:
                raise ValueError("A replacement must join before the last validator exits")
            validator.update(status="leaving", emit_at=next_epoch(state["height"] + 1, params))
        else:
            if (validator["status"] not in ("cooldown", "jailed") or validator["release_height"] is None
                    or state["height"] < validator["release_height"]
                    or state["time_ns"] <= validator["release_time_ns"]):
                raise ValueError("Bond is still exposed to the evidence window")
            sender["balance"] += validator["amount"]
            validator.update(amount=0, status="withdrawn")
    elif kind == "reserve":
        if state["lease"]:
            raise ValueError("A task is already reserved")
        if body["parent"] != state["model_root"] or type(body["round"]) is not int or body["round"] != state["round"]:
            raise ValueError("Wrong task parent or round")
        task_kind = body["task_kind"]
        if task_kind not in ("train", "infer"):
            raise ValueError("Unknown task kind")
        workers = body["workers"]
        if not isinstance(workers, list) or len(workers) != (2 if task_kind == "train" else 1):
            raise ValueError("Wrong number of worker identities")
        for worker in workers:
            public_key(worker)
        price = integer(body["price"])
        if task_kind == "train":
            if body["request"] != {} or price != 0 or state["round"] >= params["max_training_tasks"]:
                raise ValueError("Invalid or exhausted training budget")
        else:
            request = body["request"]
            if (not isinstance(request, dict) or set(request) != {"prompt", "max_tokens"}
                    or not isinstance(request["prompt"], str) or not 1 <= len(request["prompt"].encode()) <= 64
                    or type(request["max_tokens"]) is not int or not 1 <= request["max_tokens"] <= 64
                    or price < 10 * fee):
                raise ValueError("Invalid inference request or budget")
        escrow = params["reservation_bond"] + price
        debit(escrow)
        state["lease"] = {"task_id": protocol.transaction_id(envelope), "owner": owner,
            "workers": workers, "task_kind": task_kind, "request": body["request"],
            "price": price, "escrow": escrow, "expires": state["height"] + params["lease_blocks"],
            "parent": state["model_root"], "round": state["round"]}
    else:
        lease = state["lease"]
        if not lease or lease["owner"] != owner or body["task_id"] != lease["task_id"]:
            raise ValueError("Task is not reserved by this account")
        if state["height"] > lease["expires"]:
            raise ValueError("Task expired")
        if not isinstance(body["receipts"], list) or len(body["receipts"]) != len(lease["workers"]):
            raise ValueError("Missing worker receipts")
        result = execute(state)
        if body["result_root"] != result["result_root"]:
            raise ValueError("Result differs from full replay")
        for worker, receipt, expected in zip(lease["workers"], body["receipts"], result["receipts"]):
            signed_body, key = protocol.verify(receipt)
            if worker != key or work.canonical(signed_body) != work.canonical(expected):
                raise ValueError("Worker receipt differs from full replay")
        sender["balance"] += params["reservation_bond"]
        if lease["task_kind"] == "train":
            state["weights"], state["model_root"] = result["weights"], result["result_root"]
            state["last_training_loss_hex"] = float(result["loss"]).hex()
            state["round"] += 1
            state["issued"] += params["task_budget"]
            for worker in lease["workers"]:
                account(state, worker)["balance"] += params["worker_budget"] // 2
            verifier_budget = params["task_budget"] - params["worker_budget"]
        else:
            # The request pins the accepted checkpoint; the output itself is stored for retrieval.
            state["last_inference"] = result["output"]
            worker_payment = lease["price"] * 4 // 5
            account(state, lease["workers"][0])["balance"] += worker_payment
            verifier_budget = lease["price"] - worker_payment
        if state["pending_verifier_reward"] is not None:
            raise ValueError("Previous verifier reward has not settled")
        state["pending_verifier_reward"] = {"height": state["height"], "budget": verifier_budget,
                                            "powers": voting_power(state, state["height"])}
        state["completed"].append(lease["task_id"])
        state["lease"] = None
    invariant(state)
    return state


def execute(state, data):
    lease = state["lease"]
    if lease["task_kind"] == "train":
        result = work.replay(state["weights"], data, state["round"], lease["task_id"])
        result["result_root"] = work.digest(result["weights"])
        return result
    output = work.infer(state["weights"], **lease["request"])
    root = work.digest(output)
    return {"result_root": root, "output": output,
            "receipts": [{"task_id": lease["task_id"], "model_root": state["model_root"],
                          "request_root": work.digest(lease["request"]), "result_root": root}]}
