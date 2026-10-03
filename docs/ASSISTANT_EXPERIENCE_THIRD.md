# Third attempt: a bounded update as the learning unit

Declared on October 2, 2026, after both confirmations failed and four studies
found no low-rank candidate, and before any look at the third confirmation split.
Authorized by the project owner, who chose this amendment. The
[contract](../config/experiments/assistant-experience-third.json) pins every rule
below.

## The amendment

A2 asked for a new module that matches the equal-data update while training 60
times fewer parameters. Twice, the 1M-parameter low-rank module came within 4–8
episodes of the 63M-parameter update without meeting the margin
([first](ASSISTANT_EXPERIENCE_CONFIRMATION_RESULTS.md),
[second](ASSISTANT_EXPERIENCE_CONFIRMATION2_RESULTS.md)). The
[methodology](ASSISTANT_EXPERIENCE_STUDY_RESULTS.md),
[growth](ASSISTANT_EXPERIENCE_GROWTH_RESULTS.md),
[compositional](ASSISTANT_EXPERIENCE_COMPOSE_RESULTS.md) and
[repair](ASSISTANT_EXPERIENCE_REPAIRS_RESULTS.md) studies then found no low-rank
candidate that kept every parent success.

A2's learning unit is now any bounded, separable learned update. Its tensors are
stored apart from the frozen backbone and applied only when the selector chooses
them; serving the parent rolls it back. It trains a small fraction of the
backbone's parameters. A low-rank module qualifies, and so does an update of
declared projections. The rest of A2 is unchanged: beat the unchanged parent on
frozen development and fresh confirmation gates, lose no parent success, report
training and serving costs and bound latency.

Whether low-rank modules can match updates of the same projections moves to A3.
A3 already compares separate modules with repeated shared-weight updates, by
cumulative retention across cohorts under a declared budget.

## What was seen first

On the second confirmation, the round-4 update control solved 182 of 192
episodes, lost no parent success and had a p95 of 106.1 s. Under this amendment
it would have passed every check that applies to it. That split is spent and
earns nothing: its results motivated the amendment. The decision therefore rests
only on the sealed `confirmation3` split, which no system has seen. Choosing a
candidate on one split and confirming it on a fresh one is what the sealed
splits are for.

## Candidate

The round-4 update system, unchanged:

- **Weights.** Updated `q_proj` and `v_proj` matrices in layers 32–39: 62,914,560
  trained parameters, 1.85% of the backbone's 3,402,836,480. About 126 MB in
  BF16, stored apart from the 6.8 GB parent. Trainable digest `58eafbfc…`.
- **Selection.** Its round-4 gate (integration digest `28bfa5fa…`) chooses once
  per episode from the parent's feature. Anchors are served by the parent.
- **Serving.** Prefix-cache serving on the canonical CPU runtime.

There is no new rollout, training or gate refit.

## Development and A1 (opened data)

One CPU host serves the update's version on the 24 opened development cases, as
round 4 did: each episode with its selection pass, the forced-arm anchors, then
a fresh-process replay of the two declared episodes.

- **Development gate.** At least 18/24, at least 4 more than the parent, no lost
  parent or anchor success, and p95 at most 180 s including selection.
- **A1 served-system check.** At least 6 of the 8 primitive workflows with one
  per primitive family, the original anchor gate with every protected anchor
  routed to the parent, p95 at most 180 s, and both fresh-process replays
  matching exactly.

Comparisons with a separate update control do not apply.

## Confirmation (sealed, once)

Only if both checks pass, `confirmation3` opens once. It has 192 episodes, 24
per family, from its own generator seed, disjoint in ID and project from every
earlier split. The parent control and the update system each run on their own
host under prefix-cache serving. The gate is the second confirmation's, without
the update comparisons:

- at least 154 successes, and at least 16 in each family;
- at least 20 more successes than the parent, with no parent success lost;
- a family-cluster bootstrap lower bound (10,000 resamples, seed 27092039) above
  zero against the parent;
- p95 at most 180 s, including selection.

## Outcomes

- **Pass.** A2 is established for this capability. With the A1 served-system
  check, A1's usable-foundation condition is also met on fresh data; A1 closes
  once its remaining items are recorded. A3 can then begin.
- **Fail.** A2 stays open and `confirmation3` is spent. Another attempt needs a
  split from a new generator seed.

## Limits

- **The amendment follows results.** It was chosen after the update did well on
  spent data. Only the fresh split can show that was not luck.
- **Size.** The update is larger to distribute than the module: about 126 MB
  against 2 MiB. Rollback is still a switch to the parent.
- **Selection.** Its gate has chosen the update for every workspace episode so
  far. Selection therefore separates workspace episodes from anchors; it has not
  shown when to fall back within the workspace.
- **One capability.** Repeated growth (A3) remains unproved.
- **Budget.** Three r7i.4xlarge host runs: development (at most $8), then the
  parent and update confirmation hosts (at most $12 each). The ceiling is $40;
  about $10 is expected.
