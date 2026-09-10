"""Bounded content-addressed retrieval from interchangeable peers."""

import hashlib
from pathlib import Path
from urllib.request import urlopen


CHUNK = 16_384
MAX_OBJECT = 2_000_000


def manifest(data):
    if not data or len(data) > MAX_OBJECT:
        raise ValueError("Object outside this execution profile")
    return {"size": len(data), "sha256": hashlib.sha256(data).hexdigest(),
            "chunks": [hashlib.sha256(data[i:i + CHUNK]).hexdigest() for i in range(0, len(data), CHUNK)]}


def publish(root, data):
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    spec = manifest(data)
    for start, digest in zip(range(0, len(data), CHUNK), spec["chunks"]):
        path = root / digest
        if not path.exists():
            path.write_bytes(data[start:start + CHUNK])
    return spec


def fetch(spec, peers, output):
    if type(spec["size"]) is not int or not 0 < spec["size"] <= MAX_OBJECT:
        raise ValueError("Invalid object size")
    if len(spec["chunks"]) != (spec["size"] + CHUNK - 1) // CHUNK:
        raise ValueError("Wrong number of chunks")
    pieces, attempts, rejected = [], 0, 0
    for index, digest in enumerate(spec["chunks"]):
        if not isinstance(digest, str) or len(digest) != 64 or bytes.fromhex(digest).hex() != digest:
            raise ValueError("Invalid chunk hash")
        expected_size = min(CHUNK, spec["size"] - index * CHUNK)
        for peer in peers:
            attempts += 1
            try:
                with urlopen(peer.rstrip("/") + "/" + digest, timeout=2) as response:
                    chunk = response.read(CHUNK + 1)
                if len(chunk) != expected_size or hashlib.sha256(chunk).hexdigest() != digest:
                    raise ValueError("Corrupt chunk")
            except (OSError, ValueError):
                rejected += 1
                continue
            pieces.append(chunk)
            break
        else:
            raise ValueError("No available peer supplied the authenticated chunk")
    data = b"".join(pieces)
    if len(data) != spec["size"] or hashlib.sha256(data).hexdigest() != spec["sha256"]:
        raise ValueError("Object commitment mismatch")
    output = Path(output)
    temporary = output.with_suffix(output.suffix + ".partial")
    temporary.write_bytes(data)
    temporary.replace(output)
    return {"bytes": len(data), "attempts": attempts, "rejected_responses": rejected}
