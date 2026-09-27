# Complete decentralized assistant: architecture amendment

September 27, 2026. This prospectively amends [A1–A6](../TODO_ASSISTANT.md).
It does not revise any past experiment result, complete a milestone, or activate
a new model on the public ledger.

## The product and its learning unit

The product is a conversation that completes useful work and retains context.
The learning unit is a compatible neural module that improves those complete
conversations, including tool choice, arguments, corrections and final artifacts.
A better isolated checker or an oracle union of expert answers is insufficient.

Use a capable general backbone, initially the pinned Apache-2.0 Granite 4.1 3B,
with a bounded number of learned modules. Keep the backbone fixed for the first
addition so preservation and marginal capacity can be measured. Later shared-weight
updates and consolidation remain available, but must pass whole-assistant gates.
Do not indefinitely accumulate specialist modules that cannot work together.

A user's current documents and personal memory belong to a separate, controlled
information layer. Tools provide retrieval, calculation and explicitly authorized
actions. A candidate must learn how to use that information; filling a database
is not evidence that its neural capability improved. Private conversations do
not automatically become public training data or ledger contents. Future jobs
require opt-in data provenance and explicit access rules. This experiment uses
only authored fictional data and an in-memory workspace with no external actions.

## One serving version

A version binds backbone, modules, selector, tokenizer, tool schemas, conversation
policy, decoding and limits. Promotion evaluates that entire version. Retain the
previous version for rollback. A verifier may reject an invalid tool call; an
unqualified neural checker must not silently veto otherwise correct answers.

The first candidate routes once per bounded workspace episode, then uses the
chosen model throughout its follow-up. At most one added module is active.
Outside that explicitly selected operation, use the unchanged parent. This is a
measurable first interface, not evidence of automatic module discovery for every
possible user request. Multi-capability conversation and composition remain A3.

## Learning and comparison

The [new contract](ASSISTANT_WORKFLOW_LEARNING.md) first executes the complete
parent. Then compare the parent, an update to existing parameters, and added
parameters under the same tools, records and declared training schedule. Train
the candidate before its selector; fit selection from successful and unsuccessful
complete integration episodes, never from evaluation labels. Charge every rollout,
feature computation, retry and deployment cost. Test held-out combinations and
preserve individual earlier successes, not just an aggregate score.

A published adapter passing an unrelated benchmark is no longer a prerequisite
to this work. We already reproduced its implementation and observed real quality
failures. A1 instead requires a useful parent, reproducible serving, retained
anchors and a feasible distributed execution plan. This removes a research
bottleneck without changing the failed outcomes or relaxing A2's comparison.

## Shards and the network

A neural module is not a hardware shard. Partition the backbone and module
weights across owners; an execution owner need not hold the complete model.
Use small compute groups with measured connectivity for a training job. Additional
groups can train candidates or replicate the accepted graph. Place replicas near
users and limit the active path; making a request visit every new machine is not
a scaling strategy.

The existing partition runtime is Llama-specific. Granite needs its native
embedding, residual, attention and logit multipliers, RoPE, tied head, caches and
training semantics preserved. The [shape-derived partition plan](../config/experiments/assistant-granite-partition.json)
assigns half the layers to each of two owners, with the tied embedding/head on
owner 0. BF16 weights are approximately 3.66 GB and 3.15 GB per owner; these are
storage estimates, **not measured peak RAM**. The path returns final hidden states
to owner 0 for the tied head. It has two transfers per decode step. Activation,
optimizer, buffering and network overhead still need measurement.

Implement the port with small native-model conformance tests first. Compare
full versus sharded prefill, cached decode, gradients and a checkpoint round-trip;
then measure the real profile across hosts and recover an interrupted worker.
A learning pass cannot authorize native promotion before this compatibility work.

## Consensus and incentives

Keep NeuroShard-native BFT for ledger ordering. Sponsors fund declared work,
honest auditing and serving. Correct computation, useful quality and consensus
are separate claims. Issuance/settlement must not pay twice for a work identity;
quality admission chooses the served graph and does not turn lottery tickets
into evidence of learning. Existing recovery and supply invariants remain required.

Cheap permissionless verification and independent ownership remain open. Full
replay is an oracle for experiments, not an economical million-device solution.
A5 requires a funded audit policy with stated honesty/collusion assumptions and
four independent operators. Simulated identities on our own AWS account cannot
close it. This amendment creates no new consensus, token sale or mainnet claim.

## Research basis and limits

[Petals](https://arxiv.org/abs/2312.08361) supports investigating partitioned
transformer execution. [L2R](https://arxiv.org/abs/2408.09053) supports investigating
isolated modules followed by learned integration. [DiLoCo](https://arxiv.org/abs/2311.08105)
supports investigating less frequent synchronization inside compute groups.
These are precedents, not proofs of this architecture's quality or adversarial
security. Our proposed combination must pass its own end-to-end comparisons.
[Activated LoRA](https://arxiv.org/abs/2504.12397) has a specific activation/cache
contract; ordinary LoRA in the first experiment does not inherit that property.
