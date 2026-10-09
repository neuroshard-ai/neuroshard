# The public assistant network

**Status: in development.** This is how the grown assistant reaches people:
served across peers who each hold part of it, paid and settled on the native
ledger. It follows the [verifiable network design](VERIFIABLE_NETWORK_DESIGN.md)
and the [bonded serving ledger](OPTIMISTIC_SERVING.md).

## Who does what

- **Seed node.** One project-run AWS node holds the ledger as a CometBFT
  validator and serves public RPC and a test-NEURO faucet. It may also host one
  shard and audit. It never holds the whole model: serving needs peers for the
  other shards.
- **Owners (peers).** Each hosts one shard of the served version on their own
  machine. An owner bonds NEURO and publishes its endpoint on the ledger, serves
  the jobs that name it, commits signed logs, and is paid per token position
  served. A proven fault slashes its bond.
- **Users.** Get test NEURO from the faucet and open a paid job naming one owner
  per shard. Their device runs stage 0: the embedding, first layers and output
  head, about 2.4 GB. Prompts and answers never leave it; owners see only
  intermediate activations. Unused budget is refunded.
- **Auditors.** Hold one shard, replay committed logs, and prove fraud for half
  the slashed bond. The seed runs one.
- **Validators.** The seed first. Independent operation (A5) begins when four
  operators run validators and none holds a third of the voting power.

## Milestones

1. **Ledger for a public network.** `transfer`, for the faucet, and
   `owner_endpoint`, so users find owners on the ledger.
2. **Transport across the internet.** The user's device dials each owner's
   endpoint and relays signed messages between shards, so it needs no inbound
   connections. This replaces the experiments' in-datacenter ring and keeps
   their link signatures and logs unchanged.
3. **Owner node:** `neuroshard assistant host --shard K`. It fetches only its
   shard's byte ranges, bonds, publishes its endpoint, serves jobs that name it,
   commits logs and keeps them through the challenge window.
4. **User client:** `neuroshard assistant chat`. It covers the faucet, stage 0,
   owner discovery from the ledger, the paid job, the conversation and
   settlement.
5. **Seed and genesis.** Run the seed on AWS, publish the genesis and network
   descriptor, then run end to end with peers on other machines. Join guides for
   owners and users.
6. **The current version across owners.** A2's version already reproduced its
   single-host results across three owners. Cohort 3's modules (U1, L2, L3) all
   sit in layers 32–39, on the last shard. That owner switches among them per
   turn, every owner keeps one cache per route, and the version is promoted once
   it reproduces its sealed confirmation token for token across owners.
7. **Independent validators and growth through the network.** Validators join
   for A5. Contributions train the next module, its sealed gate promotes it, and
   the network switches version with the previous one kept for rollback.

Milestone 1 is implemented and tested on the ledger's small fixtures.
