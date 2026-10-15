# The public assistant network

**Status: live test network `neuroshard-assistant-1`, serving `a4-general` since block 491799 on
October 15, 2026, by owner vote.** It is A3's drafting and scheduling assistant plus a general route:
questions about your own documents, with search and cited answers, and general chat with a calculator
and a requirement checker. The grown assistant is served across machines that each hold part of it, and
every conversation is paid and settled on the native ledger. It follows the
[verifiable network design](VERIFIABLE_NETWORK_DESIGN.md) and the
[bonded serving ledger](OPTIMISTIC_SERVING.md).

**Who decides the served model.** Since block 491756 the bonded owners decide, not the steward key.
The steward handed over once and irreversibly (`governance_handover`), and consensus has refused its
`model_promote` since. `a4-general` was the first owner-decided version:

1. it passed its sealed confirmation ([general route](ASSISTANT_GENERAL_ROUTE.md));
2. it reproduced all 324 accepted conversations token for token through paid, separately hosted shards
   ([release check](ASSISTANT_RELEASE_CHECK_GENERAL.md));
3. it was proposed with that evidence, both owners voted yes (10 of 10 bonded NEURO), and an
   unprivileged account enacted it ([activation record](../config/experiments/assistant-a4-activation.json)).

Owners can also vote validators in and out, and contributors can be paid to train new units on chain
([growth on the network](NETWORK_GROWTH.md)). See [decentralization status](DECENTRALIZATION.md) for
what is still run by one party.

Until peers join, the project runs both owners: shard 1 on the seed and a stand-in for shard 2. That
means the owner vote is still the project's vote in practice. A project-funded auditor replays every
owner log, and the seed validator judges fraud proofs under the served version. Users still trust the
seed's RPC unless they list more, and every host has one administrator. Test NEURO has no value. The
documents in the samples are fictional; your own documents stay on your device, and only activations
reach the owners (this is not encryption).

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

## First A3 conversations

After promotion on October 11, 2026, a client on the seed host (8 threads)
held one paid conversation per route:

- **Drafting route (87 s).** It saved the draft for Cedar 6581 to the service
  team, due 2027-05-23 with a total of 67: the case's expected outcome. Settled
  at block 206718; each owner was paid for 2,949 positions.
- **Scheduling route (72 s).** It saved a 90-minute meeting for Meadow 23280
  with the design and logistics teams on 2027-02-08 at 10:30: the expected
  outcome. Settled at block 207409; 3,288 positions per owner.
- **After an owner restart.** Owner 2 restarted with its existing key and bond,
  and a further paid job served and settled at block 208599.

Unused escrow was refunded each time, and the auditor replayed every owner
log of all three jobs without a mismatch.

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
   [assistant-a3-cohort3 release](https://github.com/neuroshard-ai/neuroshard/releases/tag/assistant-a3-cohort3).
   A2's descriptor, [`assistant-testnet-a2.json`](../src/neuroshard/client/networks/assistant-testnet-a2.json),
   and its [release](https://github.com/neuroshard-ai/neuroshard/releases/tag/assistant-testnet-1)
   are kept for rollback.
6. **Auditing live.** The seed runs an auditor, and validators hold shards and
   receive proof bundles, so a challenge can land.
7. **The current version across owners.** Cohort 3's modules (U1, L2, L3) all sit
   in layers 32–39, on the last shard. That owner will switch among them per
   turn, and every owner keeps one cache per route. It was promoted after
   reproducing its 432 confirmation conversations token for token across owners.
8. **Independent validators and growth through the network.** Validators join
   for A5. Contributions train the next module, its sealed gate promotes it, and
   the network switches version, keeping the previous one for rollback.

Milestones 1–7 are implemented. The whole flow (seed, faucet, two owners, a paid
conversation and its settlement) is also tested end to end on the tiny fixture
with a local CometBFT validator.
