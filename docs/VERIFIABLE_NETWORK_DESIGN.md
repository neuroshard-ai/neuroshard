# Verifiable network design: from exact shards to a decentralized assistant

Status: design, September 29, 2026. It connects the evidence so far to the
[assistant plan](../TODO_ASSISTANT.md). Nothing here is a completed result
unless it links one.

## The goal

The target is an ever-growing assistant served and trained by independent
nodes. Nodes earn NEURO for useful training and inference, and users spend NEURO
for answers.

Testnet 1 already runs that token loop on a small model: rewarded training
tasks, paid inference, and validators that regenerate every answer before
paying (see the [protocol](LLM_PROTOCOL.md)). The Granite assistant track has a
far more useful model, with exact sharding and verified learning, but runs
off-chain.

The remaining gaps are:

- verifying large-model work without every node repeating it;
- privacy (every prompt is public on-chain today);
- resistance to fake or colluding participants (Sybil attacks);
- growth that no single contributor can hijack.

## The enabling fact

Sharded Granite execution is bit-exact. Generation, serving (with routing and
the conversation cache), training, recovery after a lost owner, and concurrent
serving all reproduce a single host exactly. See the results for
[execution](GRANITE_SHARD_EXECUTION_RESULTS.md), [serving](GRANITE_SHARD_SERVING_RESULTS.md),
[throughput](GRANITE_SHARD_THROUGHPUT_RESULTS.md), [training](GRANITE_SHARD_TRAINING_RESULTS.md)
and [determinism](GRANITE_SHARD_DETERMINISM.md). That makes a model owner a
deterministic state machine, which is what replicated-state-machine protocols
(Raft, Paxos, BFT) assume. We use that property without paying for full
replication.

## Layered design

1. **Ledger consensus (BFT, existing).** CometBFT orders only small facts:
   payments, bonds, commitments to owner logs, audit verdicts and slashing. No
   node replays neural work to produce a block.
2. **Owner logs (a Raft-style replicated log without replicated compute).**
   Each owner appends every command it executes (reset, crop, adapter switch,
   forward) with the digests of the tensor it received and sent. It retains the
   received tensors for a challenge window and signs the log.
   - The driver's committed tokens are the episode's log. Any replacement owner
     rebuilds its cache by replaying that log, which is what made recovery exact.
   - Failover needs no hot replicas.
3. **Optimistic verification with light auditors (a rollup-style fraud game).**
   Work is accepted unless challenged within a window.
   - An auditor needs only the shard it audits. It replays one owner's log for
     one episode from an empty cache and compares digests.
   - The first mismatch, with the owner's signed log and the inputs up to it, is
     a fraud proof. Any other holder of that shard can check it by one replay.
     The proof then slashes the owner's bond and pays the auditor.
   - Honest owners cannot be framed: bit-exact execution means their digests
     always reproduce.
   - Audit cost is one owner's share of one episode. Audit frequency and bond
     sizes set how expensive cheating is.
4. **Hop continuity.** Owner *k*'s logged inputs must equal owner *k−1*'s logged
   outputs, so transport tampering is detected and blame lands on exactly one
   owner.
5. **User-held first stage (privacy).** The user's own device can run owner 0:
   the embedding, first layers and output head, about 2.4 GB. Prompt and answer
   tokens then never leave the user. The network sees only intermediate
   activations, and the chain stores only digests. Activation inversion remains
   a risk to measure, but it is a large step from plaintext on-chain.
6. **Behavioral consensus for growth.** New capabilities arrive as small
   modules trained on verifiable experience, each submitted by a contributor
   with a bond.
   - On every action, the modules and the parent vote; overriding the parent
     needs agreement.
   - Growth is Byzantine-robust by construction: a minority of poisoned or
     low-quality modules cannot change behavior.
   - Modules could earn a share of inference fees whenever their vote is in the
     accepted majority on audited episodes, rewarding useful growth.
   - The [methodology study](ASSISTANT_EXPERIENCE_LEARNING.md#methodology-study-before-a-third-attempt)
     is measuring whether the committee reaches the quality bar.
7. **Execution classes.** Bit-exactness holds within a pinned runtime and CPU
   instruction class. Owners and auditors of a shard declare the same class.
   Integer-exact inference (deterministic integer kernels) would remove the
   class restriction and is a research direction.

## Economics sketch

- **Users.** Pay per requested token.
- **Owners.** Earn per computed hop, in proportion to the bytes their layers
  stream, and post bonds.
- **Auditors.** Earn a small fee per clean audit and the slashed bond on fraud.
- **Module contributors.** Earn the fee share described above.
- **Training rewards.** Paid only for work that replays: the exact sharded
  training already reproduces a single host.
- Sybil resistance comes from bonds and random audit selection rather than
  identity.

## Evidence so far and next steps

| Piece | State |
| --- | --- |
| Exact sharded serving, training, recovery, concurrency | Measured on the real 3B assistant |
| Signed owner logs, replay audit, fraud proofs | Tested on small checkpoints: honest owners verified; a one-bit fault caught, blamed and proven; forged accusations rejected |
| Audited serving on the real model with light auditors | Next declared execution |
| Committee-of-modules growth | Under study (A2) |
| Chain integration (Granite profile, bonded audits, fee split) | After A2 closes |
| User-held first stage | Design |
| Independent operators (A5) | Not started |
