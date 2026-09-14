# Adaptive sharded learning experiment

This experiment extends the persistent-shard research path. It is not a
deployment or a change to the running native network. The frozen contract is
[`adaptive-shards.json`](../config/experiments/adaptive-shards.json); input and
source commitments are in
[`adaptive-shards-inputs.json`](../config/experiments/adaptive-shards-inputs.json).
Results must be reported separately from the acceptance criteria.

## Mechanisms

Each worker allocates only its owned Llama blocks. The embedding owner also
owns the tied output head and final norm. A frozen reference model is partitioned
across the same workers. Assistant-token cross entropy learns the new data;
KL divergence toward the reference applies only to declared replay records.
The first cohort uses the seed reference. The second uses the committed first
cohort endpoint, including its newly learned behavior. No worker needs a full
student or reference model.

Distillation follows the function-preservation idea in
[Learning without Forgetting](https://arxiv.org/abs/1606.09282). Its use here is
an experimental LLM adaptation, not a theorem that forgetting cannot occur.
Separate development and final tests remain necessary.

Portable checkpoints commit named FP32 parameters, both Adam moments, each
parameter's Adam age and optimizer-group membership. A canonical learned-state
root excludes ownership and process RNG. The full checkpoint root additionally
commits the layout, parent and RNG files. This distinction is valid only for
the pinned dropout-free profile, whose updates consume no random draws.
New owners fetch only their assigned immutable files; incomplete preparation
cannot advance HEAD. Gradient clipping sums squared norms in canonical parameter
order to avoid dependence on the number of workers.

An append transition copies one prior block's internal weights and initializes
the attention and MLP output projections to zero. The new residual blocks begin
as identity functions. Existing weights and moments remain unchanged; new
parameters start their own Adam clocks at the current global cursor. This is
inspired by [Net2Net](https://arxiv.org/abs/1511.05641), adapted to residual Llama
blocks. Finite numerical execution is required. Identity at insertion does not
demonstrate that the additional capacity improves quality.

The capacity planner minimizes estimated peak utilization over contiguous
assignments. It includes the sharded teacher, Adam and a memory reserve. Actual
GPU admission is checked again on each worker. Advertised memory is not proof of
available resources. Membership changes currently use an operated controller
and restart the process group at a committed step; there is no DHT or automatic
public admission in this experiment.

## Frozen trial

The 1.7B seed is the previously recorded `compressed-pair` checkpoint with
parameter digest `ac2153f0f3e55379f72d46f4ef4a7a9a1f6b7f30a45d471cb3611c9e15a1385b`.
It starts one new Adam phase. Adam is retained thereafter, including across
cohorts, worker redistribution and growth. Public task families are lookup,
filtering, arithmetic totals and sorting. Results on these tasks do not establish
general assistant capability.

Each cohort has 128 updates of 64 documents, with replay in every update. The
learning rate, clipping, regularizer and endpoints are fixed before training.
Phase A uses 6,144 new task examples and 2,048 previously trained conversations.
Phase B uses 4,096 new examples, 2,048 examples actually trained in phase A, and
the conversation replay. The serving model is not promoted by this driver.

Two-to-three worker migration is compared with uninterrupted execution from
the same step. Growth appends two blocks and is compared with a fixed-depth
phase-B control using identical examples and optimizer schedule. All completed
attempts, including failures and aborted candidates, remain evidence.

Final task sets are committed before training. Previously exposed conversation
retention examples are development data only; final conversation retention
comes from another pinned public source range, with prompt/document exclusion
against the prepared training records. The final test is public, not secret.
The seed model's upstream pretraining data is not fully known, so this cannot
certify absence of pretraining contamination.

The final gain requires the upper one-sided 95% paired-bootstrap confidence
bound below -0.001 nats per assistant token. Retention requires the corresponding
upper bound at most +0.02. Generated task answers must not lose correct cases in
aggregate. The phase endpoints are fixed; development checks may abort a run,
but cannot select another endpoint or change its recipe. Test outcomes cannot
rescue a failed development decision.

## Reproduction

Use the pinned GPU dependencies in `learning-reference-requirements.txt`.
`scripts/prepare_adaptive_shards.py` prepares inputs and seed ownership manifests
without loading model weights. Its arguments identify the prior tensor inventory,
previously trained replay, exposed development-retention records and upstream
cache. It writes `prepared.json` and digest-bound role files.

`scripts/run_sharded_training.py adaptive train` takes `--prepared`, `--seed` and
a fresh `--home`. `--resume` identifies a complete portable commit; cohort B
also requires `--reference` identifying its committed phase-A endpoint.
`RANK`, `WORLD_SIZE`, `MASTER_ADDR`, `MASTER_PORT` and `GLOO_SOCKET_IFNAME` configure
the operated process group. `adaptive evaluate` defaults to development roles;
opening final roles requires a selection record binding the prepared job and
candidate checkpoint identities. Source mismatches fail before training.

The CPU tests use a tiny full model solely as an independent numerical oracle.
They exercise full-response loss and distillation, partial microbatches, unequal
weights, clipping, two-to-three-to-two process groups, exact learned-state
agreement, identity insertion and preservation of old Adam moments.

This training driver issues zero NEURO. The separate
[native replay-quorum bridge](NATIVE_SHARD_REPLAY.md) settles bounded GPU update
windows in an optional genesis, using complete replay attestations weighted by
native validator bonds. Its [operated results](ADAPTIVE_SHARDS_RESULTS.md) record
GPU replay and native settlement separately from learning quality. Neither a
checkpoint hash nor a quality score proves that an untrusted worker performed
the update.

The next learning experiment is the separate [continued-learning contract](CONTINUED_LEARNING.md).
It starts from the passing phase-A checkpoint, forbids growth, and treats
generated-answer improvement as the primary gate. This adaptive plan's exposed
finals cannot be reused there.
