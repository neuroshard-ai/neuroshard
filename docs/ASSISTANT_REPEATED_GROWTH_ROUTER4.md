# A3 stage 1: the message-feature router, shifted toward scheduling

Declared on October 5, 2026, after the [message-feature router's results](ASSISTANT_REPEATED_GROWTH_ROUTER3_RESULTS.md)
and before any calibration or development with it. Contract:
[assistant-growth-router4.json](../config/experiments/assistant-growth-router4.json).
The project owner chose this change.

## Why

On what the user wrote, the router is 92.6% accurate on the integration turns, and its
errors all go one way. It sent 15 of the 104 turns that need the calendar to the drafting
route, and none of the other 100 to scheduling. The two errors cost differently. A turn that
needs the calendar always fails on the drafting route, which has none. A drafting turn on
the scheduling route can still succeed: in the centroid router's development, 12 went there
and no accepted success was lost. Round 4's candidate met the scheduling and cross levels
exactly when every turn went to scheduling, so misrouted scheduling turns decide the gate.

## The change: a shift toward scheduling

A turn's margin is its cosine to the scheduling class mean minus its cosine to the drafting
class mean. The router sends a turn to scheduling when the margin exceeds minus the shift;
a shift of zero is the router as fitted.

The shift is chosen on training turns only. Each of four folds of training cases, the folds
of the reported held-out accuracy, is scored by the rule fitted without it. At most 1% of
the 384 drafting training turns, rounded down to 3, may cross to scheduling. The shift is
the largest that allows this, and never negative. Labels, data, feature and the centroid
rule are unchanged. The router file records the shift, so development and confirmation
serve it unchanged.

## When development runs

Only if the shift sends fewer of the integration turns that need the calendar to the
drafting route than the unshifted router does in the same fit. Otherwise the result returns
to the project owner. Development then serves round 4's units with every gate unchanged,
and the sealed confirmation opens once only if the candidate passes. Nothing is trained.

## Budget

A3 has spent $39.21. The calibrated refit allows at most $5.50 on one CPU host, development
$18 on three and confirmation $47 on five. That is $109.71 in total, within the $110
ceiling.
