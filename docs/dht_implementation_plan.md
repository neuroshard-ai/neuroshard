> Historical prototype document. For the supported native release, see [PUBLIC_TESTNET.md](PUBLIC_TESTNET.md) and [PROTOCOL_CANDIDATE_V2.md](PROTOCOL_CANDIDATE_V2.md).

# NeuroShard Kademlia DHT Implementation Plan

> Originally `src/neuroshard/core/network/dht_plan.py`. Moved to docs/ since this is
> a design document, not runtime code. The actual implementation is in `dht.py`,
> `dht_protocol.py`, and `dht_service.py`.

## 1. Node ID & Distance Metric
- SHA-1 or SHA-256 hash of (IP + Port + Random) -> 160-bit ID.
- Distance: XOR metric.

## 2. Routing Table (K-Buckets)
- List of buckets where index i contains nodes with distance 2^i to 2^(i+1).
- K=20 (Standard Kademlia parameter).
- Replacement cache for unresponsive nodes.

## 3. Protocol Messages (RPCs)
- PING: Check liveness.
- STORE(key, value): Store peer info or shard location.
- FIND_NODE(target_id): Returns k closest nodes to target.
- FIND_VALUE(key): Returns value if present, else k closest nodes.

## 4. Storage
- Values are Peer Metadata: {ip, port, shard_range, last_seen}.
- Keys: Hash of the shard_range (e.g., hash("0-4")) or Node ID.

## 5. Bootstrapping
- Join network by contacting any known node (can fallback to Tracker temporarily).
- Perform node lookup for self ID to populate buckets.

## 6. Integration
- P2PManager will query DHT for "Layer X" peers instead of asking Tracker.
