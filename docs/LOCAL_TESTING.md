# Local Testing

This guide describes the local test gates used before inviting outside nodes.
These gates validate protocol mechanics on cheap CPU hardware; they do not prove
useful LLM quality, GPU throughput, WAN reliability, or economic security.

## Setup

From the repository root:

```bash
python -m pip install -e ".[dev]"
```

The package distribution name is `neuroshard-ai`, but the import/module name is
`neuroshard`.

## Gate 1: Smoke

Smoke mode boots local nodes with training disabled. It validates process
startup, tracker registration, HTTP dashboards, gRPC availability, peer
discovery, ledger startup, and uptime Proof of Neural Work.

```bash
python simulate_network.py --smoke --nodes 2 --delay 5 --memory 512 \
  --duration 35 --report reports/smoke-gate.json --check
```

Expected result:

- All nodes become `READY`.
- Training steps stay at `0`.
- NEURO can increase from uptime proofs.
- The report has `"checks": {"passed": true}`.

## Gate 2: Tiny Training

Tiny mode uses a local-test model and synthetic batches:

- 2 transformer layers
- 128 hidden dimension
- 2048 vocab capacity
- no CDN tokenizer download
- no Genesis shard download
- accelerated PoNW interval

```bash
python simulate_network.py --tiny --nodes 2 --delay 10 --memory 512 \
  --duration 60 --report reports/tiny-gate.json --check \
  --min-training-nodes 2 --min-total-steps 50
```

Expected result:

- Nodes reach `TRAIN`.
- Steps increase.
- Loss is finite.
- Training PoNW rewards are generated.
- The report has `"checks": {"passed": true}`.

## Gate 3: Docker LAN

The Docker LAN gate runs nodes in separate containers on a bridge network. This
is still one physical machine, but it exercises separate network identities,
tracker discovery, container DNS, gRPC between containers, proof gossip, and
training.

```bash
python scripts/lan_gate.py --nodes 3 --duration 180 --report reports/lan-gate-3node.json --check \
  --min-training-nodes 3 --min-total-steps 300 --min-max-peers 2
```

Expected result:

- `3/3` nodes online.
- `3/3` nodes training.
- Peer count reaches at least `2`.
- Steps and NEURO increase.

## Gate 4: Docker LAN Soak

Use a longer duration to catch slower failure modes.

```bash
python scripts/lan_gate.py --nodes 3 --duration 900 --report reports/lan-gate-3node-15min.json --check \
  --min-training-nodes 3 --min-total-steps 1500 --min-max-peers 2
```

## Gate 5: Churn

Churn stops one node mid-run and expects the remaining nodes to keep training.

```bash
python scripts/lan_gate.py --nodes 3 --duration 300 --poll-interval 5 \
  --report reports/lan-gate-churn-5min.json --check \
  --churn-at 120 --churn-node 2 \
  --min-online 2 --min-training-nodes 2 \
  --min-total-steps 800 --min-max-peers 1 \
  --min-post-churn-steps-delta 300
```

## Gate 6: Restart Recovery

Restart recovery stops a node, waits, starts it again, and expects the network
to return to full participation.

```bash
python scripts/lan_gate.py --nodes 3 --duration 600 --poll-interval 5 \
  --report reports/lan-gate-restart-10min.json --check \
  --churn-at 180 --churn-node 2 --restart-after 90 \
  --min-online 3 --min-training-nodes 3 \
  --min-total-steps 1500 --min-max-peers 2 \
  --min-post-churn-steps-delta 500 \
  --min-post-restart-steps-delta 300
```

## Reports

Reports are written under `reports/` and are intentionally ignored by git. They
contain:

- mode and config
- per-node status
- snapshots over time
- pass/fail thresholds
- churn/restart events when enabled

Use these reports as promotion evidence between gates.

## Important Limits

Passing these gates means the mechanics work in a controlled environment. It
does not mean:

- the trained model is useful
- real GPU training is efficient
- real WAN/NAT behavior is solved
- economic incentives are adversary-proof
- production Genesis training is lightweight enough for small machines

Treat the gates as staged evidence, not as production guarantees.
