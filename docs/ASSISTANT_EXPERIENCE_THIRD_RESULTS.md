# Third attempt results

Declaration: [third attempt](ASSISTANT_EXPERIENCE_THIRD.md).

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

## Confirmation

Opens once, under the declared gate.
