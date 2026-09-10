# Native testnet operator guide

This is an experimental NeuroShard chain with its own CometBFT 0.38.26 consensus, account ledger, training rewards, and bonded validator entry. The supported workload is a 34,976-parameter CPU language model with two worker stages and full verification replay. A website account, registration token, and external settlement chain are unnecessary.

The current public deployment is at https://neuroshard.com/. Its genesis allocations and validators are controlled by one operator across two machines. The network is reachable publicly, but independent ownership and production economic security have not been established. The project sponsor currently offers a bounded 100-task session with a reference stage-0 worker; a stage-1 contributor can check `/work/status` before joining. The sponsor stops on its first failure and does not restart automatically or on boot; availability is not guaranteed. The previous observer ledger and its balances are separate; no migration is defined. Test balances carry no redemption promise.

## Install and join

Use public release `v0.3.0a1`. The older PyPI package and `neuroshard --token ...` command do not join this chain.

```bash
git clone --branch v0.3.0a1 --depth 1 https://github.com/neuroshard-ai/neuroshard.git
cd neuroshard
bash scripts/install_native.sh
```

The installer uses Linux x86_64, Python 3.10–3.12, a project-local virtual environment, pinned CPU dependencies, checksum-verified Go 1.27.1, and CometBFT 0.38.26. Ubuntu may need `sudo apt install python3-venv curl`. It creates no accounts or validator keys. The recorded machines have four vCPUs and 16 GiB RAM; resource requirements for prolonged public load remain unmeasured.

Obtain `genesis.json`, its SHA-256 digest, and a public peer address from the release announcement and compare them independently. The website's Join page supplies the current values. Never substitute a digest fetched from an unrelated node. Every source/numerical change requires a compatible release; this candidate has no in-place consensus upgrade protocol.

```bash
venv_build/bin/python scripts/neuroshard_chain.py init \
  --home ~/.neuroshard/node \
  --genesis genesis.json \
  --genesis-sha256 PUBLISHED_SHA256 \
  --peer NODE_ID@PUBLIC_HOST:26656
venv_build/bin/python scripts/neuroshard_chain.py run --home ~/.neuroshard/node
```

`init` verifies the exact genesis bytes, source/data manifest, and numerical conformance before starting. It refuses to replace an initialized node. Account, P2P, and consensus keys are generated locally. A full node starts with zero balance and can replay all blocks without staking. The runtime launches and monitors consensus, the application, and a bounded ledger gateway; component failure terminates the group for supervised restart.

For a genesis older than the evidence time horizon, also supply `--trusted-height H --trusted-hash BLOCK_HASH` from a **recent independently trusted checkpoint**. The node checks that block after replay and stops on a mismatch. The operator must establish the checkpoint's authenticity and recency; the software cannot infer either from the first peer that answers. Synchronization replays from genesis; snapshots and cryptographic account inclusion proofs are not implemented.

Port defaults:

| Port | Binding and purpose |
|---|---|
| TCP 26656 | Public native P2P; enable in host/cloud firewall when accepting inbound peers |
| TCP 26657 | Loopback native RPC |
| TCP 26658 | Loopback application gRPC |
| TCP 26659 | Loopback bounded ledger gateway; expose through HTTPS if desired |
| TCP 26660 | Optional loopback sponsor service; expose `/work/` through HTTPS |

Use `--advertise YOUR_PUBLIC_HOST` when reachable from outside. PEX is enabled; initial peer addresses are connectivity hints. `--private-network` permits private addresses and multiple peers per IP for local experiments; public operators should omit it. The two-machine acceptance experiment explicitly uses that option for collocated validators while dialing the remote public peer port directly.

## Contribute computation

First run a full node so assignments can be checked against your own accepted chain. Choose an available sponsor URL. A sponsor supplies task reservation collateral; a worker needs no initial tokens and opens no inbound worker port.

```bash
venv_build/bin/python scripts/neuroshard_work.py worker \
  --home ~/.neuroshard/node --stage 0 \
  --coordinator https://SPONSOR_HOST
```

Use `--max-tasks 1` for a trial that exits after returning one stage gradient; payment still requires the complete task to finalize. A second worker supplies stage 1. Both stages can be operated by the same account; total task rewards remain capped. The worker polls over HTTPS, checks the sponsor signature and its own node's finalized lease, model, batch, assigned public key, stage, and expiry, computes, and returns signed results. Durable per-operation records reuse completed responses after retries and refuse changed requests or an interrupted computation under the same lease. A process lock prevents duplicate workers for the same home/stage. A new valid lease is required after an interrupted computation.

Anyone with sufficient liquid balance can run a sponsor:

```bash
venv_build/bin/python scripts/neuroshard_work.py sponsor \
  --home ~/.neuroshard/node --tasks 10 --port 26660
```

The listener is on loopback; terminate TLS at your reverse proxy and forward `/work/` with a 2 MB request limit. Local tests may use `http://127.0.0.1:26660`. Use `--wait-seconds 0` to wait indefinitely for workers while holding no reservation. `/work/status` reports recent workers per stage and the remaining task budget. The sponsor attempts at most the specified number of reservations and stops on the first error. Do not attach an unconditional service restart policy that silently resets this spending limit.

The coordinator has a bounded, expiring worker registry and one active task. It can select or exclude workers, and can withhold submission. It cannot mint rewards or bypass validator replay. Workers that withhold results can burn the sponsor's reservation bond; assignments provide no unconditional payment guarantee. Joining a sponsor is permissionless, but fair access, Sybil-resistant scheduling, and credit risk are unresolved. Independent sponsor operators and endpoint discovery remain necessary for a broader market.

## Balances and validation

All CLI amounts are integer atoms: **1 NEURO = 1,000,000 atoms**.

```bash
venv_build/bin/python scripts/neuroshard_chain.py account --home ~/.neuroshard/node
venv_build/bin/python scripts/neuroshard_chain.py transfer --home ~/.neuroshard/node \
  --to RECIPIENT_COMPRESSED_PUBLIC_KEY --amount 1000000
venv_build/bin/python scripts/neuroshard_chain.py bond --home ~/.neuroshard/node --amount 250000
venv_build/bin/python scripts/neuroshard_chain.py unbond --home ~/.neuroshard/node
venv_build/bin/python scripts/neuroshard_chain.py withdraw --home ~/.neuroshard/node
```

The CLI signs locally. `--rpc https://HOST/rpc` can relay a signed transaction through another gateway, with chain identity checked, but remote nonce/state reads are trusted observations. Prefer your own node. Bonding requires the node's consensus-key possession proof and liquid balance for the bond plus fee. Vote only with one running copy of a consensus key. Preserve CometBFT's signing state across restarts; restoring an old signing-state backup and signing again can create equivocation.

The `testnet` profile is committed in genesis and differs from the accelerated `lab` profile:

| Parameter | Testnet | Lab |
|---|---:|---:|
| Membership epoch | 60 blocks | 8 blocks |
| Minimum activation scheduling delay | 60 blocks | 4 blocks |
| Evidence block age | 172,800 blocks | 24 blocks |
| Evidence time age | 172,800 seconds | 6 seconds |
| Task lease | 120 blocks | 16 blocks |
| Maximum rewarded training tasks | 100,000 | 1,000 |

Common values: 1,000-atom transaction fee, 250,000-atom voting unit, 2,000,000-atom reservation bond, and 1,000,000 newly issued atoms per accepted training step. Each worker receives 400,000 atoms; 200,000 enter deferred verifier settlement. Withdrawal requires both block and time deadlines to pass after effective removal, plus the protocol's inclusion margin. The block deadline can make withdrawal substantially longer than 48 hours. The finite issuance cap and initial allocation are experiments, not a production supply policy.

## Operate and recover

A service template is in `config/neuroshard-native.service`. Adjust the checkout/home/user paths before installing it. Run under an ordinary account. Logs and state stay under the node home. Monitor `/healthz`, component exits, disk growth, peer connectivity, and observed block/round progress. The gateway reports a stalled chain after 30 seconds without a new block; this is an operational signal, not a consensus timeout. Public gateway traffic is bounded; its in-process rate limit counts proxy connections together when all clients arrive through one reverse proxy.

Keep an offline copy of the release, exact genesis, and encrypted backups of account keys. Stop the complete service before a consistent state backup. Restore the application database, CometBFT data, and signing state together, on one machine only. Replaying a nonvalidator from genesis is safer than trying to combine mismatched databases. Never delete validator signing state to force a restart. No emergency administrator can rewrite balances or approve an invalid update.

The preview uses an isolated nginx path and its own native gateway. `config/native-preview.nginx.conf` documents the route. The old homepage's Docker frontend, account database, tracker, and unrelated host services are still running. Replacing the root website and publishing the GitHub release are separate release actions after review of the candidate.

## Reproduce the checks

```bash
ATEN_CPU_CAPABILITY=default MKL_ENABLE_INSTRUCTIONS=SSE4_2 PYTHONPATH=src \
  venv_build/bin/python -m pytest -q tests/test_verified_demo.py \
  tests/test_protocol_candidate.py tests/test_public_node.py tests/test_outbound_work.py

venv_build/bin/python scripts/check_public_network.py \
  --host YOUR_SSH_ALIAS --peer-host PUBLIC_IPV4 \
  --remote-root /home/ubuntu/neuroshard-lab \
  --output docs/eval/results/public_network_candidate.json
```

The network runner requires the CPU runtime and pinned engine on both hosts and an open remote TCP 26656. SSH is used for setup and process administration; native consensus connects directly to the public peer address. Without `--keep-running`, the runner stops its own processes after testing.

The [protocol](PROTOCOL_CANDIDATE_V2.md) and [experiment report](PROTOCOL_EXPERIMENTS.md) distinguish execution verification from economic utility. Full replay, single-task scheduling, full-model storage at validators, growing account/history state, concentrated ownership, and finite CPU conformance remain limits of this candidate. An independent security review, multiple independent operators, measured sustained load, adversarial network tests, a release-signing/checkpoint policy, and a defensible public allocation are required before a production launch.
