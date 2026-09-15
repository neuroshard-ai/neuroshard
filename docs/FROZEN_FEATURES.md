# Reusing the frozen part of a sharded model

An added shard can receive the established model's representations once and
train repeatedly from those representations. The established layers remain
distributed. The learner holds its own trainable tail and a read-only copy of
the tied output matrix and final normalization, so it can compute gradients
without calling every established owner on every update.

For a frozen prefix `F`, trainable tail `T`, and frozen output head `H`, the
objective is `loss(H(T(F(x))))`. Once `F(x)` has been calculated for the exact
padded microbatch, reusing it leaves the mathematical gradient with respect to
`T` unchanged. The reference representation used for retention is also fixed.
The fixed-size control caches the input to its last two blocks and the output
of the original frozen versions of those blocks.

This addresses the cost of giving a new shard enough repeated training. It
does not establish that additional updates will improve held-out answers.
Learning, retention and a matched smaller-model comparison remain required.

## What has been checked

The three-process CPU checks compare the existing distributed incremental
kernel with the feature learner over eight Adam updates, for both appended
blocks and the fixed-size tail control. Features are saved and reloaded before
reuse. Every reported loss, gradient norm, parameter and Adam state must match
exactly. The learner's tail plus read-only head is smaller than the whole test
model. Changed padding, tokenization, labels, weights and output-head state
are rejected.

```bash
python -m pytest -q tests/evolution/test_frozen_features.py
```

The [GPU probe](../config/experiments/frozen-feature-probe.json) freezes a
separate comparison against the 1.7B parent. It repeats the first four exact
training batches twice in each arm. It requires exact intermediate updates
and final serialized tail states, and reports feature production, distributed
and local update times, network bytes and GPU memory. It may use the four
existing experiment instances only after the complete frozen capacity
comparison, replacement check and any selected final evaluations finish.
The probe's updates and costs are recorded separately. The allocation's
original deadline and spending limit still apply.

This probe does not select a learning candidate or open another final set.
Its source, plan, input identity and parent must be committed before execution.
GPU equivalence and performance are pending until results are published.

## What must be preserved for reuse

Representations depend on the parent weights, tokenizer, exact ordered
microbatch, padding, numerical profile and cut layer. Regrouping examples can
change numerical results and requires a new feature computation. Each feature
packet also binds the labels and objective weights. The current learner
requires a complete trainable tail on one owner; larger tails need their own
distributed backward path.

A tensor hash establishes which bytes were used. It does not establish that
the correct frozen prefix produced them. A permissionless implementation must
audit feature production against the committed parent and input, then bind
each subsequent update to that certified feature object. The numerical kernel
does not yet implement that certification or a native settlement profile.

For the existing 128-batch schedule, one FP32 representation stream occupies
13.82 GB before metadata. The fixed-size control needs both prefix and
reference streams. The experiment must account for producing, storing,
transferring and auditing those bytes, and for the read-only output-head
replica. Reuse saves repeated prefix work; it introduces persistent feature
storage.
