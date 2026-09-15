# Repeated exposure for learning in an added shard

The short capacity experiment gives each fictional fact twelve training
targets: eight questions and four narrative descriptions. Its first completed
candidates learn the JSON format but recall few of the facts. This experiment
isolates training duration as a possible cause. It changes neither the target
answers nor the required learning and retention margins.

The [plan](../config/experiments/knowledge-rehearsal.json) repeats the original
128 batches eight times at the original lower peak learning rate, for 1,024
updates and 96 target exposures per fact. The cosine schedule spans those
1,024 updates. Both appended and existing-tail arms keep the same optimizer,
microbatches, padding, reference KL, margin penalty and proven-trained replay
examples. Only the terminal checkpoint can qualify.

[Knowledge Capacity Scaling Laws](https://arxiv.org/html/2404.05405v1) reports
that repeated exposure substantially affects fact learning in its controlled
pretraining experiments. That motivates measuring duration here; its exposure
counts are not a proven threshold for our frozen-parent setting.
[Knowledge Storage and Extraction](https://arxiv.org/html/2309.14316v3) also
distinguishes memorizing text from answering different questions about it.
The existing generated-answer gate therefore remains necessary.

## Training and memory ownership

The distributed prefix produces a content-bound bank for the exact 128
training batches. Every stored microbatch binds its parent, source/profile,
input order, labels, loss weights, padding and tensor hashes. Identical prefix
and reference representations share storage. Subsequent updates load only the
current batch and train the owned tail with a read-only output head.

Established owners keep their assigned layers. Their weights and previous Adam
history remain unchanged. The new owner holds its tail, its own Adam state,
one batch of representations and the frozen output-head replica. No owner
needs the whole model. This is an operated experiment; permissionless feature
certification and payment are separate work.

The midpoint at update 512 preserves the complete tail and optimizer cursor
for recovery. A resumed worker recreates the immutable bank from the same
parent and inputs, then continues the bound schedule without resetting Adam.
All owners commit the terminal checkpoint before any development scoring.

## Freeze and selection

Preparation refuses to proceed until two prerequisite records are committed:

- Exact real-model feature factorization for both arms, including all eight
  intermediate updates under the declared source commit.
- Failure of all four original short-run candidates, with neither arm selected
  to open the original final set.

The prepared record binds those receipts, every numerical dependency, the
existing tokenized inputs and the repeated batch cursor. Training refuses
uncommitted or changed preparation. If the original experiment succeeds or
feature factorization fails, this plan cannot run as written.

Both rehearsal arms must finish before selection. Eligible terminal candidates
must pass every existing development gate; finals use the still-unused
knowledge questions and the previously exposed retention probes. A failed arm
cannot open finals. Learning in added blocks and an advantage over the
fixed-size control are reported separately. No numerical result in this
experiment issues tokens or changes serving.

## Validation

```bash
python -m pytest -q tests/evolution/test_feature_bank.py tests/evolution/test_rehearsal.py
```

The three-process integration checks exercise feature production, durable
storage, repeated local updates, checkpoint writing and answer generation for
both arm layouts. Their final parameters and Adam states match eight ordinary
distributed updates exactly. Additional checks reject altered feature bytes,
rehashed replacement banks, changed input order, incomplete controls and
premature checkpoint selection. Failed development does not read final inputs.

[Real-model feature equivalence](FROZEN_FEATURE_RESULTS.md) passed all eight
updates in both layouts. The [prepared record](../config/experiments/knowledge-rehearsal-prepared.json)
and [bounded execution plan](../config/experiments/knowledge-rehearsal-execution.json)
bind the longer trial before any worker allocation. Its learning outcome remains pending.
