"""CPU transcript-mining mechanism sketch, inspired by cuPOW (not an implementation).

Apache-2.0. Fresh low-rank encodings, tile lotteries and recovery of the useful
integer product are executable. Resource-hardness is NOT established. Matrices
and assignments are trusted verifier inputs; this is not permissionless consensus.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import time

import numpy as np

import neural_work_reference as nw


MAX_SIDE = 32
TILE = 4
NOISE_RANK = 4


@dataclass(frozen=True)
class Context:
    work_id: str
    operation: str
    chain_challenge: str
    worker: str

    def descriptor(self, left: np.ndarray, right: np.ndarray) -> dict:
        nw.product_bound(left, right)
        if any(n > MAX_SIDE or n % TILE for n in (*left.shape, *right.shape)):
            raise nw.Rejected("mining sketch requires dimensions 4..32 divisible by four")
        if self.operation not in ("forward", "weight_gradient", "input_gradient"):
            raise nw.Rejected("unknown training operation")
        for name in ("work_id", "chain_challenge", "worker"):
            value = getattr(self, name)
            if not isinstance(value, str) or len(value) != 64:
                raise nw.Rejected("expected fixed-width context hash/key")
            try:
                bytes.fromhex(value)
            except ValueError as error:
                raise nw.Rejected("invalid context encoding") from error
            if value != value.lower():
                raise nw.Rejected("noncanonical context encoding")
        return {"scheme": "neuroshard-cpu-tile-sketch-v1", "work_id": self.work_id,
                "operation": self.operation, "chain_challenge": self.chain_challenge,
                "worker": self.worker, "left": nw.matrix_root(left),
                "right": nw.matrix_root(right), "shape": [*left.shape, right.shape[1]],
                "tile": TILE, "noise_rank": NOISE_RANK, "primes": list(nw.PRIMES)}


def field_matrix(seed: str, name: str, shape: tuple[int, int], prime: int) -> np.ndarray:
    count = shape[0] * shape[1]
    values = []
    counter = 0
    ceiling = (1 << 32) - (1 << 32) % prime
    while len(values) < count:
        payload = f"neuroshard/tile-noise/v1/{seed}/{prime}/{name}/{counter}".encode()
        raw = np.frombuffer(hashlib.shake_256(payload).digest(4096), dtype="<u4")
        values.extend(int(v) % prime for v in raw if int(v) < ceiling)
        counter += 1
    return np.array(values[:count], dtype=np.int64).reshape(shape)


def encode(left: np.ndarray, right: np.ndarray, seed: str, prime: int):
    rows, inner = left.shape
    columns = right.shape[1]
    el = field_matrix(seed, "el", (rows, NOISE_RANK), prime)
    er = field_matrix(seed, "er", (NOISE_RANK, inner), prime)
    fl = field_matrix(seed, "fl", (inner, NOISE_RANK), prime)
    fr = field_matrix(seed, "fr", (NOISE_RANK, columns), prime)
    noisy_left = (left + el @ er) % prime
    noisy_right = (right + fl @ fr) % prime
    return noisy_left, noisy_right, (el, er, fl, fr)


def ticket_hash(descriptor_root: str, prime: int, row: int, inner: int,
                column: int, partial: np.ndarray) -> str:
    return nw.digest({"domain": "neuroshard/tile-ticket/v1", "descriptor": descriptor_root,
                      "prime": prime, "row": row, "inner": inner, "column": column,
                      "partial": nw.matrix_root(partial)})


def difficulty(bits: int) -> int:
    if type(bits) is not int or not 1 <= bits <= 16:
        raise nw.Rejected("research difficulty must be 1..16 bits")
    return 1 << (256 - bits)


def mine_product(left: np.ndarray, right: np.ndarray, context: Context,
                 *, bits: int = 4) -> tuple[np.ndarray, list[dict], dict]:
    """Compute ALL tiles, including losers, and recover the useful product.

    This intentionally slow CPU mechanism is capped at 32 per dimension. It is
    not the paper's asymptotic performance or security implementation.
    """
    started = time.perf_counter()
    descriptor = context.descriptor(left, right)
    root = nw.digest(descriptor)
    target = difficulty(bits)
    residues, winners = [], []
    attempts = 0
    for prime in nw.PRIMES:
        noisy_left, noisy_right, (el, er, fl, fr) = encode(left, right, root, prime)
        noised_product = np.zeros((left.shape[0], right.shape[1]), dtype=np.int64)
        for row in range(0, left.shape[0], TILE):
            for inner in range(0, left.shape[1], TILE):
                for column in range(0, right.shape[1], TILE):
                    partial = (noisy_left[row:row + TILE, inner:inner + TILE]
                               @ noisy_right[inner:inner + TILE, column:column + TILE]) % prime
                    noised_product[row:row + TILE, column:column + TILE] += partial
                    noised_product[row:row + TILE, column:column + TILE] %= prime
                    ticket = ticket_hash(root, prime, row, inner, column, partial)
                    attempts += 1
                    if int(ticket, 16) < target:
                        winners.append({"descriptor": root, "prime": prime, "row": row,
                                        "inner": inner, "column": column, "partial": partial.copy(),
                                        "ticket": ticket})
        # AB = (A+E)(B+F) - AF - E(B+F). Preserve the low-rank factorizations.
        af = (((left % prime) @ fl) % prime @ fr) % prime
        ebf = (el @ ((er @ noisy_right) % prime)) % prime
        residues.append((noised_product - af - ebf) % prime)
    p, q = nw.PRIMES
    # Signed CRT recovery is unique under the shared reference's raw bounds.
    combined = residues[0] + p * (((residues[1] - residues[0]) * pow(p, -1, q)) % q)
    recovered = np.where(combined > p * q // 2, combined - p * q, combined)
    bound = nw.product_bound(left, right)
    nw.matrix(recovered, limit=bound)
    return recovered, winners, {"descriptor": root, "tile_attempts": attempts,
                                "winning_tiles": len(winners),
                                "seconds": time.perf_counter() - started}


def verify_ticket(left: np.ndarray, right: np.ndarray, context: Context,
                  proof: dict, *, bits: int = 4) -> str:
    expected_keys = {"descriptor", "prime", "row", "inner", "column", "partial", "ticket"}
    if set(proof) != expected_keys:
        raise nw.Rejected("unexpected proof fields; no free nonce or claimed difficulty")
    root = nw.digest(context.descriptor(left, right))
    if proof["descriptor"] != root:
        raise nw.Rejected("ticket is bound to another job, operation, claimant, or challenge")
    prime = proof["prime"]
    if type(prime) is not int or prime not in nw.PRIMES:
        raise nw.Rejected("invalid field")
    for name, size in (("row", left.shape[0]), ("inner", left.shape[1]),
                       ("column", right.shape[1])):
        index = proof[name]
        if type(index) is not int or index < 0 or index % TILE or index + TILE > size:
            raise nw.Rejected("invalid tile position")
    partial = nw.matrix(proof["partial"], limit=prime - 1)
    if partial.shape != (TILE, TILE) or np.any(partial < 0):
        raise nw.Rejected("invalid partial-product encoding")
    noisy_left, noisy_right, _ = encode(left, right, root, prime)
    row, inner, column = (proof[name] for name in ("row", "inner", "column"))
    expected = (noisy_left[row:row + TILE, inner:inner + TILE]
                @ noisy_right[inner:inner + TILE, column:column + TILE]) % prime
    if not np.array_equal(partial, expected):
        raise nw.Rejected("forged intermediate product")
    ticket = ticket_hash(root, prime, row, inner, column, expected)
    if proof["ticket"] != ticket or int(ticket, 16) >= difficulty(bits):
        raise nw.Rejected("ticket does not satisfy the fixed target")
    return nw.digest({"descriptor": root, "prime": prime, "row": row,
                      "inner": inner, "column": column})


def train_with_tickets(job: nw.Job, *, challenge: str, worker: str, bits: int = 4):
    """The actual three training products run through the mining sketch."""
    work_id = job.work_id()
    certificates = []
    measurements = []

    def product(left, right, name):
        context = Context(work_id, name, challenge, worker)
        result, proofs, measured = mine_product(left, right, context, bits=bits)
        certificates.append((name, left, right, context, proofs))
        measurements.append({"operation": name, **measured})
        return result

    forward = product(job.inputs, job.weights, "forward")
    residual = nw.matrix(nw.rounded_divide(forward, job.scale) - job.targets)
    weight_gradient = product(job.inputs.T, residual, "weight_gradient")
    input_gradient = product(residual, job.weights.T, "input_gradient")
    after = nw.matrix(job.weights - nw.rounded_divide(
        weight_gradient, job.scale * len(job.inputs) * job.learning_rate_denominator))
    return {"forward": forward, "weight_gradient": weight_gradient,
            "input_gradient": input_gradient, "after": after}, certificates, measurements


def verify_bundle(job: nw.Job, trace: dict, proofs: dict, *, committed_root: str,
                  audit_seed: bytes, challenge: str, worker: str, bits: int = 4) -> list[str]:
    """Check operation origins from the authenticated linear job, not the miner.

    The caller must supply an admitted job, authenticated worker, current chain
    challenge and fixed target. Arithmetic audit randomness is separate from the
    PUBLIC mining challenge. No fork choice or cross-block receipt set exists here.
    """
    nw.verify(job, trace, committed_root=committed_root, seed=audit_seed)
    if set(proofs) != {"forward", "weight_gradient", "input_gradient"}:
        raise nw.Rejected("wrong operation set")
    residual = nw.rounded_divide(trace["forward"], job.scale) - job.targets
    operations = {
        "forward": (job.inputs, job.weights),
        "weight_gradient": (job.inputs.T, residual),
        "input_gradient": (residual, job.weights.T),
    }
    accepted = set()
    for name, (left, right) in operations.items():
        if not isinstance(proofs[name], list) or len(proofs[name]) > 1024:
            raise nw.Rejected("oversized ticket list")
        context = Context(job.work_id(), name, challenge, worker)
        for proof in proofs[name]:
            receipt_id = verify_ticket(left, right, context, proof, bits=bits)
            if receipt_id in accepted:
                raise nw.Rejected("duplicate mining ticket")
            accepted.add(receipt_id)
    return sorted(accepted)
