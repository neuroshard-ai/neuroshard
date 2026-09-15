# Learning in added model capacity

The [completed GPU comparison](INCREMENTAL_CAPACITY_RESULTS.md) failed the
useful-learning gate in all four candidates. Exact recovery after replacing
the final owner passed. No final set was opened or model promoted.

This numerical candidate trains new transformer blocks over a frozen, sharded
parent model. It targets the outstanding useful-growth problem: additional
peers should support additional learned capabilities while retaining the
established model's capabilities. CPU execution and checkpoint recovery are
implemented; a real-model learning comparison is still required.

The earlier [growth experiment](ADAPTIVE_SHARDS_RESULTS.md) used the same small,
decaying learning rate for inherited and added parameters. Its larger model
failed the useful-growth gate. Here the added blocks have their own optimizer
and learning schedule. The established parameters and their Adam history stay
immutable. This follows the frozen-parent/new-block direction studied in
[LLaMA Pro](https://arxiv.org/abs/2401.02415), with a distributed execution path
and a shared computation for retention supervision.

## Distributed computation

Let the frozen parent's final representation be `h = P(x)`, its frozen output
head be `H`, and the added blocks be `T`. The reference answer distribution is
`H(h)` and the candidate distribution is `H(T(h))`. Both use the same parent
forward pass. The new blocks start as an identity, then learn through the
frozen head. Only new-block owners hold gradients and Adam state. Backward
propagation stops at the frozen parent boundary.

Owners hold their assigned layers; the tied embedding and output head belong
to the first owner. Added blocks may span multiple owners. The implementation
also supports a partition containing the end of the frozen parent followed by
the beginning of the trainable tail.

Freezing the parent's tensors preserves their bytes, not every answer produced
by the expanded model. A trained tail can still cause forgetting. Learning and
retention must therefore be measured on generated responses under a committed
evaluation rule.

## A meaningful smaller-model control

The comparison control fine-tunes the original final blocks and keeps a frozen
copy of just those blocks for its reference. For a 24-layer parent and two
trainable blocks:

| Candidate | Parent/reference forward | Trainable forward | Trainable backward |
| --- | --- | --- | --- |
| Grow to 26 layers | 24 layers | 2 new layers | 2 new layers |
| Keep 24 layers | 22 shared layers plus 2 frozen original layers | 2 original layers | 2 original layers |

Both have the same number of trainable parameters and the same layer-operation
counts. Actual runtime, memory, communication and serving cost still need
measurement. The control currently places its complete trainable tail on the
last owner. It starts fresh Adam on that tail; the original parent objects,
including its earlier moments, remain preserved. This optimizer reset is an
explicit comparison method, not an implicit modification of the parent.

## Checkpoints and verification

The research checkpoint records each parameter's actual optimizer update count.
Frozen objects retain their original hashes and counts; new parameters advance
only when trained. Local checkpoints reuse verified immutable parent objects
and write the changed tail. Whole-checkpoint commitments bind the parent,
architecture, trainable boundary, recipe, cohort cursor and complete ownership.
Tensor storage uses safetensors. Existing native portable-work rules reject
this separate research format.

`tests/evolution/test_incremental_shards.py` compares three- and four-process
Gloo execution with full-model autograd, including a trainable tail spanning
two owners. It checks exact frozen parameters, active weights and Adam,
identity initialization, checkpoint reload, and fresh-process recovery from
update two to the exact update-four checkpoint. Rehashed claims that change
frozen parent objects or parameter update counts are rejected. The control's
reference is constructed from the original parent before candidate recovery.

Run from the repository's pinned development environment:

```bash
python -m pytest -q tests/evolution/test_incremental_shards.py
```

These checks establish the numerical and recovery mechanics. A useful-growth
claim requires a frozen real-model comparison with fresh evaluation, retention,
and complete resource accounting before any native growth integration.

## The real-model comparison

The [experiment plan](../config/experiments/incremental-capacity.json) starts
from the established 1.7B R4 checkpoint and compares 26 layers with a 24-layer
control. Four workers own the complete model; none holds the full model.
Both arms train two blocks for 128 updates, with exactly two declared learning
rates. Only the terminal checkpoint is eligible for selection.

The learning task introduces 640 fictional directory facts about 160 people.
Training includes questions and narrative descriptions. Evaluation asks about
those learned facts through held-out question forms: 32 people for development
and the other 128 for final scoring. Facts are training material, not held-out
knowledge. Eight questions about one person count as one statistical cluster.

Replay uses records from completed R4 training updates. Each batch includes
40 fresh questions, five narrative descriptions, twelve trained skill records
and seven trained conversations. All fresh records are used once. The 512
trained conversations cycle to provide 896 replay appearances. Previously
exposed skill and conversation evaluations measure retention; they are not
described as new independent quality evidence.

A candidate must answer at least 75% of the final knowledge questions, show a
positive gain exceeding the declared confidence margin, lose none of the
previously correct skill answers, and meet the conversation-retention bound.
The added-capacity decision additionally requires beating the fixed-size
control by at least 32 answers with a positive entity-cluster confidence bound.
Learning in added blocks and an advantage over the control are reported
separately.

`scripts/run_incremental_capacity.py` implements preparation, training,
baseline evaluation, candidate selection and final scoring. Preparation binds
the Git-committed plan and numerical sources; GPU execution requires a second
commit containing prepared identities and the exact batch schedule. Selection
requires reports for every declared candidate and closes development. Final
evaluation accepts only the committed selected checkpoints. Scoring rechecks
the actual generated tokens, complete text, stopping rule and correctness.

Training restores only owned, hash-verified tensor objects and copies the
parent's last block when initializing added layers. A fresh GPU identity probe
runs before the first update. Checkpoints preserve the frozen tensor objects
and support restart at update 64. Evaluation restores model weights without
allocating an optimizer. Metrics record per-owner memory, elapsed time and
network counters. The numerical candidate issues no tokens or changes to the
public network.
