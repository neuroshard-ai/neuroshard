# Testnet Gates

NeuroShard should move toward public participation through explicit promotion
gates. Each gate should produce a JSON report and a short human summary. Do not
advance a gate because it looked good in a terminal; advance it because the
report passed and the logs explain any known warnings.

## Current Status

The following gates have passed in local development:

- Local smoke gate
- Local tiny training gate
- Docker LAN 2-node gate
- Docker LAN 3-node gate
- Docker LAN 3-node 15-minute soak
- Docker LAN churn gate
- Docker LAN restart recovery gate

These results validate local protocol mechanics. They are not a production
launch signal yet.

## Gate A: Local Smoke

Purpose: prove node startup and basic network services.

Pass criteria:

- all configured nodes online
- dashboards reachable
- tracker discovery works
- no unexpected crashes
- uptime PoNW can accrue

## Gate B: Local Tiny Training

Purpose: prove tiny synthetic training and training PoNW.

Pass criteria:

- all configured nodes online
- required nodes enter training
- total steps exceed threshold
- loss is finite
- training PoNW rewards accrue
- no unexpected crashes

## Gate C: Docker LAN

Purpose: prove separate container identities can discover and train over a
Docker bridge network.

Pass criteria:

- all configured nodes online
- all configured nodes training
- peer count reaches expected graph size
- total steps exceed threshold
- PoNW proofs/rewards continue
- no unexpected crashes

## Gate D: Soak

Purpose: prove the network remains stable over a longer run.

Suggested first command:

```bash
python scripts/lan_gate.py --nodes 3 --duration 900 --report reports/lan-gate-3node-15min.json --check \
  --min-training-nodes 3 --min-total-steps 1500 --min-max-peers 2
```

Pass criteria:

- no node exits unexpectedly
- steps continue increasing
- rewards continue increasing
- peer count remains stable

## Gate E: Churn

Purpose: prove remaining nodes continue after one expected node failure.

Suggested command:

```bash
python scripts/lan_gate.py --nodes 3 --duration 300 --poll-interval 5 \
  --report reports/lan-gate-churn-5min.json --check \
  --churn-at 120 --churn-node 2 \
  --min-online 2 --min-training-nodes 2 \
  --min-total-steps 800 --min-max-peers 1 \
  --min-post-churn-steps-delta 300
```

Pass criteria:

- churn event is recorded
- killed node is recorded as expected down
- unexpected crash count remains zero
- remaining nodes stay online
- remaining nodes keep training after churn
- post-churn step delta exceeds threshold

## Gate F: Restart Recovery

Purpose: prove a stopped node can rejoin.

Suggested command:

```bash
python scripts/lan_gate.py --nodes 3 --duration 600 --poll-interval 5 \
  --report reports/lan-gate-restart-10min.json --check \
  --churn-at 180 --churn-node 2 --restart-after 90 \
  --min-online 3 --min-training-nodes 3 \
  --min-total-steps 1500 --min-max-peers 2 \
  --min-post-churn-steps-delta 500 \
  --min-post-restart-steps-delta 300
```

Pass criteria:

- stopped node restarts
- final online count returns to full size
- final training count returns to full size
- peer count recovers
- total steps keep increasing after restart

## Gate G: Real LAN

Purpose: move from Docker bridge to actual machines on one LAN.

Suggested setup:

- 3 to 5 CPU-only machines or small VMs
- one tracker
- tiny local-test mode first
- test tokens only
- reports collected from the coordinator

Pass criteria:

- same as Docker LAN
- no reliance on Docker container DNS
- node addresses resolve across real hosts

## Gate H: Regional Cloud

Purpose: prove basic WAN behavior without inviting the public.

Suggested setup:

- 5 to 20 small CPU instances
- at least 2 regions
- tiny mode first
- longer soak
- churn and restart during the run

Pass criteria:

- successful peer discovery across regions
- proof gossip succeeds across regions
- nodes remain stable for hours
- bandwidth and memory are measured

## Gate I: Closed Alpha

Purpose: invite trusted operators without production claims.

Rules:

- test tokens only
- no useful-model guarantee
- no real economic value promise
- publish known limitations
- collect reports and logs from each participant

## Do Not Claim Yet

Until real-data, WAN, adversarial, and economic tests pass, do not claim:

- useful LLM quality
- production training economics
- Sybil resistance
- full trustless verification
- nationwide readiness

The correct public framing is: NeuroShard is testing decentralized training
mechanics through staged testnets.
