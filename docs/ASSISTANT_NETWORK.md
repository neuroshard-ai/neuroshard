# The public assistant network

**Status: live test network `neuroshard-assistant-1`, serving A2's version (`a2-u1`).**
The grown assistant is served across machines that each hold part of it, and
every conversation is paid and settled on the native ledger. It follows the
[verifiable network design](VERIFIABLE_NETWORK_DESIGN.md) and the
[bonded serving ledger](OPTIMISTIC_SERVING.md).

Until peers join, the project runs both owners: shard 1 on the seed and a
stand-in for shard 2. Automatic auditing and A3 promotion are being qualified in
[PR #85](https://github.com/neuroshard-ai/neuroshard/pull/85), under the
[release contract](ASSISTANT_RELEASE_CHECK.md). Until deployment is recorded,
auditing is not live and users trust the seed's RPC. Test NEURO has no
value, and the tasks are the fictional drafting workspaces A2 was measured on.

## Who does what

- **Seed node.** One project-run AWS node is the ledger's first CometBFT
  validator and serves public RPC and a test-NEURO faucet. It also hosts shard 1.
  It never holds the whole model: serving needs a peer for every other shard.
- **Owners (peers).** Each hosts one shard of the served version on their own
  machine. An owner bonds NEURO and publishes its endpoint on the ledger, serves
  the jobs that name it, commits signed logs, and is paid per token position
  served. A proven fault slashes its bond.
- **Users.** Get test NEURO from the faucet and open a paid job naming one owner
  per shard. Their device runs stage 0: the embedding, layers 0–11 and the output
  head, about 2.4 GB. The client sends intermediate activations, rather than raw
  prompts or documents, to owners. Activations can leak information: this is
  not encryption or a confidentiality guarantee. Do not submit sensitive data.
  The ledger records digests and payments. Unused budget is refunded.
- **Auditors** hold one shard, replay committed logs and prove fraud for half the
  slashed bond. **Validators** order transactions and judge proofs. Independent
  operation (A5) begins when four operators run validators and none holds a
  third of the voting power.

## Join

Linux x86_64 with Python 3.12:

```bash
git clone --branch main https://github.com/neuroshard-ai/neuroshard.git
cd neuroshard
python3.12 -m venv ~/.venvs/neuroshard
source ~/.venvs/neuroshard/bin/activate
python -m pip install -r docs/granite-shard-chain-requirements.txt
python -m pip install --no-deps -e .
neuroshard assistant status
```

**Chat.** `neuroshard assistant chat` fetches stage 0, gets test NEURO from the
faucet, picks one reachable owner per shard from the ledger and opens a paid job.
It needs about 6 GB of RAM. `--sample N` picks another workspace, and `--world
FILE` loads your own documents. `/quit` ends the job; the owners then commit
their logs and the job settles after its challenge window.

**Host a shard.** `neuroshard assistant host --shard 2 --endpoint HOST:28700`
fetches only that shard's tensors, bonds 5 test NEURO and serves jobs. `HOST`
must be your public address, and inbound TCP 28700 must be open. An owner needs
about 8 GB of RAM and eight threads. To keep audits exact, it should match the
current owners' execution class: an Intel CPU with AMX, such as AWS m7i or r7i.
Owners, auditors and the seed refuse a mismatched canonical runtime. Install
Python **3.12.12** and the pinned requirements above. The client can run on other
x86_64 CPUs, but its measured quality and latency can differ.

**Audit a shard.** `neuroshard assistant audit --shard 2` holds just that shard,
downloads signed logs inside the challenge window, replays them and delivers a
funded proof when it detects a fault. `--once` checks current jobs and exits.
Clean audits currently earn no automatic reward: project funding pays their
operating cost. A slashing reward is not an honest-audit budget. See the
[optimistic protocol](OPTIMISTIC_SERVING.md).

Each owner serves one authenticated job at a time. Jobs are limited to 65,536
paid positions, bounded frames and an hour of owner time. The default chat buys
16,384 positions. Replicas add availability and concurrent capacity; they do not
automatically improve a single answer. A lost job expires and refunds unused
escrow; reconnecting a conversation is not yet supported.

## First conversation

On October 9, 2026, the client ran on a 4-core machine without AMX, separate
from both owners. It held one two-turn conversation from development sample 0.

- **Turn 1 (218 s).** The assistant saved the requested draft from the latest
  approved plan and confirmed it.
- **Turn 2 (158 s).** It saved the corrected draft from the Annex plan, but ran
  out of its six-generation budget before confirming.
- **Outcome.** Both drafts equal the case's expected outcomes. Each owner
  committed a signed log of 2,731 token positions.
- **Settlement.** Consensus settled the job at block 2,770: 0.083 NEURO to
  each owner, and 0.833 of the 1 NEURO escrow back to the user.
- **Where the time goes.** On that client, stage 0 alone costs about 50 s per
  1,000 prompt tokens and about 340 ms per generated token, for layers 0–11 and
  the output head. Running it in FP32 instead of BF16 barely changes that.

## Milestones

1. **Ledger for a public network.** `transfer`, for the faucet, and
   `owner_endpoint`, so users find owners on the ledger.
2. **Transport across the internet.** The user's device dials each owner's
   endpoint and relays signed messages between shards
   ([`neuroshard.inference.relay`](../src/neuroshard/inference/relay.py)), so it
   needs no inbound connections. On small fixtures it matches single-host serving
   token for token, with owner logs equal to the experiments' in-datacenter ring.
3. **Owner node:** `neuroshard assistant host`.
4. **User client:** `neuroshard assistant chat`.
5. **Seed and genesis.** The seed runs on AWS; the descriptor is
   [`assistant-testnet.json`](../src/neuroshard/client/networks/assistant-testnet.json),
   and owners' module files are in the
   [assistant-testnet-1 release](https://github.com/neuroshard-ai/neuroshard/releases/tag/assistant-testnet-1).
6. **Auditing live.** The seed runs an auditor, and validators hold shards and
   receive proof bundles, so a challenge can land.
7. **The current version across owners.** Cohort 3's modules (U1, L2, L3) all sit
   in layers 32–39, on the last shard. That owner will switch among them per
   turn, and every owner will keep one cache per route. The version is promoted
   once it reproduces its sealed confirmation token for token across owners.
8. **Independent validators and growth through the network.** Validators join
   for A5. Contributions train the next module, its sealed gate promotes it, and
   the network switches version, keeping the previous one for rollback.

Milestones 1–5 are implemented. The whole flow (seed, faucet, two owners, a paid
conversation and its settlement) is also tested end to end on the tiny fixture
with a local CometBFT validator.
