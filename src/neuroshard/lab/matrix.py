"""Exact, bounded integer matrix-product checks; not a complete ML proof system.

The challenge is derived from the entire statement, including the proposed
output. The 2^-rounds soundness statement assumes independent uniform binary
challenges; using this Fiat-Shamir form additionally assumes a random oracle
and accounts for the adversary's number of attempted statements.
"""

import hashlib
import json

import numpy as np


DOMAIN = b"neuroshard/integer-matmul/v1\0"
ROUNDS = 128
MAX_DIM = 2048
INPUT_BOUND = 127


def validate(a, b, c):
    for value in (a, b, c):
        if not isinstance(value, np.ndarray) or value.dtype != np.dtype("int64") or value.ndim != 2:
            raise ValueError("Require two-dimensional native int64 arrays")
        if any(not 1 <= dim <= MAX_DIM for dim in value.shape):
            raise ValueError("Matrix dimension outside the bounded profile")
    if a.shape[1] != b.shape[0] or c.shape != (a.shape[0], b.shape[1]):
        raise ValueError("Incompatible product shapes")
    for value in (a, b):
        if np.any(value < -INPUT_BOUND) or np.any(value > INPUT_BOUND):
            raise ValueError("Input exceeds signed quantized profile")
    bound = a.shape[1] * INPUT_BOUND ** 2
    if np.any(c < -bound) or np.any(c > bound):
        raise ValueError("Output exceeds the product bound")
    # Every unreduced sum in A(BR) and CR is bounded by K*N*127^2.
    assert a.shape[1] * b.shape[1] * INPUT_BOUND ** 2 < 2 ** 63


def statement(a, b, c, context):
    validate(a, b, c)
    if not isinstance(context, str) or len(context.encode()) > 1024:
        raise ValueError("Invalid operation/task context")
    hasher = hashlib.sha256(DOMAIN)
    header = json.dumps([context, list(a.shape), list(b.shape), list(c.shape)], separators=(",", ":"))
    hasher.update(header.encode() + b"\0")
    for value in (a, b, c):
        hasher.update(value.astype("<i8", copy=False).tobytes(order="C"))
    return hasher.digest()


def challenges(seed, columns, rounds=ROUNDS):
    if len(seed) != 32 or type(rounds) is not int or not 1 <= rounds <= 256:
        raise ValueError("Invalid challenge parameters")
    raw = hashlib.shake_256(DOMAIN + seed).digest((columns * rounds + 7) // 8)
    return np.unpackbits(np.frombuffer(raw, dtype=np.uint8), bitorder="little")[:columns * rounds].reshape(columns, rounds).astype(np.int64)


def check_with_challenges(a, b, c, r):
    validate(a, b, c)
    if (r.dtype != np.dtype("int64") or r.ndim != 2 or r.shape[0] != b.shape[1]
            or not 1 <= r.shape[1] <= 256 or np.any((r != 0) & (r != 1))):
        raise ValueError("Invalid binary challenges")
    return bool(np.array_equal(a @ (b @ r), c @ r))


def verify(a, b, c, context, rounds=ROUNDS):
    seed = statement(a, b, c, context)
    return check_with_challenges(a, b, c, challenges(seed, b.shape[1], rounds))
