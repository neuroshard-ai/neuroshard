# Third attempt results

Declaration: [third attempt](ASSISTANT_EXPERIENCE_THIRD.md).

**Passed. A2 is established for the workspace capability.** On the 192 sealed
`confirmation3` episodes, opened once, the learned update solved 183 (95.3%)
against the unchanged parent's 119 (62.0%). It lost none of the parent's
successes, and every declared check passed.

## Confirmation: passed

Evidence: [report](../config/experiments/assistant-experience-confirmation3-report.json) and
per-system results ([parent](../config/experiments/assistant-experience-confirmation3-parent-result.json),
[update](../config/experiments/assistant-experience-confirmation3-update-result.json)), commit `fa18eef`.
Both systems ran under prefix-cache serving, one host each.

| Check | Outcome |
| --- | --- |
| Total ≥154 | Pass (183) |
| Each family ≥16 | Pass (lowest 21: difference) |
| Net vs parent ≥+20 | Pass (+64: 64 gained, none lost) |
| Lost parent successes = 0 | Pass |
| Protected successes | Pass |
| Lower 95% gain vs parent >0 | Pass (+22.4 points) |
| p95 ≤180 s, including selection | Pass (103.0 s; parent 88.8 s) |

| Family | Parent | Update |
| --- | --- | --- |
| Copy | 16 | 24 |
| Date | 15 | 23 |
| Sum | 19 | 24 |
| Difference | 1 | 21 |
| Latest | 18 | 22 |
| Recipient | 17 | 24 |
| Reschedule | 19 | 23 |
| Scope | 14 | 22 |

- **Where it fails.** Nine episodes: three differences, two latest, two scope,
  one date and one reschedule.
- **Selection.** The gate chose the update for all 192 episodes. Anchors are
  served by the parent.
- **Consistency.** The same system solved 182 of the 192 spent second-confirmation
  episodes, so the fresh result matches the one that motivated the amendment.

Two r7i.4xlarge hosts cost $7.10 (parent $3.23, update $3.86), and everything is
retired. With development, the third attempt cost $7.68 of its $40 ceiling.

## What this establishes

Learning from verified experience gives the complete assistant a useful new
capability: it is about 1.5 times as reliable on fresh workspace conversations,
with every parent success kept, at a p95 of 103 s. The learned unit is 62.9M
parameters (1.85% of the backbone), stored apart from the parent and rolled
back by serving the parent.

It does not establish:

- that a low-rank module can match the update; that question is now part of A3;
- repeated growth across cohorts with retention (A3);
- selection that falls back within the workspace, since the gate always chose
  the update;
- serving by independent operators (A5).

## Development and A1 served-system check: passed

**Both checks passed, so the sealed `confirmation3` split opens once.** The
unchanged round-4 update, gated alone and served under the prefix cache, solved
19 of the 24 opened development cases against the parent's 9. It lost no parent
success.

Evidence: [result](../config/experiments/assistant-experience-development-third-result.json) and
[report](../config/experiments/assistant-experience-development-third-report.json), commit `c85965c`.

| Check | Outcome |
| --- | --- |
| Total ≥18/24 | Pass (19) |
| Net vs parent ≥+4 | Pass (+10: 10 gained, none lost) |
| Lost parent successes = 0 | Pass |
| Protected successes | Pass |
| p95 ≤180 s, including selection | Pass (94.8 s) |
| A1: primitive ≥6/8, one per family | Pass (8/8: copy 2, date 2, sum 2, difference 2) |
| A1: anchor gate, anchors routed to the parent | Pass (19/24 canonical anchors) |
| A1: fresh-process replays | Pass (both declared episodes matched exactly) |

- **Reproduction.** All 24 episodes matched the round-4 run of the same system
  exactly, on a different host four days later.
- **Forced anchors.** Forced onto the anchors, the update loses one protected
  anchor (`granite-instruction-counts`), as in round 4. The served system routes
  anchors to the parent, so that loss is never served.
- **Selection.** The gate chose the update for all 24 episodes.

One r7i.4xlarge host cost $0.58, and everything is retired.
