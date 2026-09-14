# Answer-balanced continuation

The [consolidation experiment](CONSOLIDATED_LEARNING_RESULTS.md) retains the
measured prior answers and much of the arithmetic gain, but two new sorting
regressions still reject it. The next experiment targets that interference
inside training. Its [plan](../config/experiments/balanced-continuation.json)
must be committed before preparation, and the prepared inputs must be committed
before GPU execution.

An audit of the calculation-step training corpus finds that arithmetic accounts
for 81.015% of weighted target tokens and sorting for 2.759%. These are shares
of the supervised-loss normalization mass, not measured gradient contributions.
Long calculation traces have many more target tokens and a larger task weight
than short sorting answers. This is a concrete hypothesis for the observed
interference, not a demonstrated causal explanation.

## Training the same sharded model

Start from the recorded, rejected 50% consolidated checkpoint at step 384,
including its complete Adam state. Train all 1,711,376,384 parameters for 64
additional updates. The original accepted phase-A model remains the quality
baseline. The consolidated input is the frozen reference for preserving its
learned behavior; using it as a research starting point does not approve it for
serving.

Set each document's `loss_weight` to `1.0 / targets`. The existing guarded
cross-entropy calculation therefore gives each answer equal total supervised
weight, irrespective of its response length. Float32 row weights and the
existing Python denominator are part of the frozen numerical recipe. Reference
KL and correct-token margin regularization retain their existing token-average
normalization, with strengths 2 and 1. Every training example explicitly enables
the reference anchor; fresh examples remain identified as fresh data.

| Training stratum | Documents | Per batch | Source |
| --- | ---: | ---: | --- |
| Sorting | 1,792 | 28 | Fresh generated records |
| Lookup | 512 | 8 | Fresh generated records |
| Filtering | 512 | 8 | Fresh generated records |
| Arithmetic with calculation steps | 512 | 8 | Previously trained calculation-step examples |
| Strict JSON-only arithmetic | 256 | 4 | Previously trained replay examples |
| Conversations | 512 | 8 | Previously trained replay documents |

All 4,096 documents are used exactly once in 64 batches. Each batch has the
declared composition, with a frozen shuffle. JSON object-key order cannot change
the schedule. Learning rate is 0.000001, with four warmup steps and the existing
cosine decay; microbatch size is eight and global gradient clipping remains 1.
Checkpoints at steps 416 and 448 preserve full optimizer state. Only step 448
can be selected.

New training and development examples cover all four previously public
instruction wordings. Final cases use new records under those public wordings;
this experiment does not claim an unseen-wording holdout. Training labels are
generated and checked against the executable task definition. Inference still
uses ordinary greedy generation, without output repair, constrained decoding,
model routing or a calculator.

Replay indices are selected by a frozen shuffle from the complete, committed
calculation-step training file. Each must occur in the source's completed
256-update schedule. Original messages and tokenization are preserved; only
the declared objective weights, anchor flags and provenance annotations change.
The full source file is hash-checked, so replay cannot silently substitute an
old evaluation example. Every earlier role, including the now-exposed
consolidation finals, is excluded from fresh data by identity and normalized
content. Conversation evaluation uses new complete documents from the same
SHA-256-pinned public SmolTalk parquet.

Instruction coverage and controlled replay are also studied in
[InsCL](https://arxiv.org/abs/2403.11435) and
[Maximally Interfered Retrieval](https://arxiv.org/abs/1908.04742).
This experiment uses a fixed mixture and per-answer supervision motivated by
the measured failures; its result must be established independently.

## Acceptance and execution

Before training, record the phase-A baseline on 512 prior development tasks,
128 new development tasks and 128 conversation documents. Also record the
consolidated input on the new development tasks. After step 448, require zero
lost phase-A-correct prior answers, at least eight net gains on new tasks,
no new-task family decline against either starting model, and conversation
mean loss change at most +0.01 nats against phase A. A failure preserves both
checkpoints and leaves final evaluation unopened.

Commit the actual passing endpoint and complete development comparison before
final evaluation. On 512 new tasks, require at least 16 net gains against phase
A, exact one-sided McNemar *p* < 0.05, and no per-family score decline. An added
floor requires at least 64/128 correct arithmetic answers to prevent a nominal
pass that discards most of the learned skill. Require no family decline on 256
fresh prior tasks, and a 95% bootstrap conversation-loss-change upper bound at
most +0.02 nats on 256 fresh documents. Use 10,000 samples and the frozen seed.
Loss alone cannot pass; a failed final cannot select an earlier checkpoint.

The existing runner provides `balance prepare`, `balance train`,
`balance evaluate` and `balance score`. Training and evaluation require both
`--parent` (the consolidated input) and `--baseline-checkpoint` (phase A).
Each worker loads only its owned partitions. Complete input validation and
checkpoint loading precede network-group creation, avoiding the earlier
pre-generation timeout during input loading.

The allocation remains three A10G workers, six hours and $100. Preserve both
checkpoints, full Adam state, every attempt and the complete evaluation before
removing the temporary instances and disks. This is a fixed-size learning
experiment under one operator. It issues no NEURO, changes no public serving
root, and does not establish useful model growth or permissionless operation.
