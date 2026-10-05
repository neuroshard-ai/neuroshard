# A3 router refit results: what the user wrote separates the turns

**On the parent's mean state over the user's own message, the turn-need router is
92.6% accurate on the integration turns, up from 73.0%, and every error is
one-sided.** It was refitted on the same 2,112 labelled training turns. On held-out
training cases it chose correctly 88.6% of the time. On the 204 integration turns it
misrouted 15. All 15 were scheduling or cross turns sent to the drafting route; no
drafting turn was sent to the scheduling route.

Declaration: [router refit](ASSISTANT_REPEATED_GROWTH_ROUTER3.md). Evidence:
[result](../config/experiments/assistant-growth-router3-result.json) and
[report](../config/experiments/assistant-growth-router3-report.json), commit `3c6af57`.

## Accuracy

| | Turns | Previous feature | Message feature |
| --- | --- | --- | --- |
| Held out, four folds of training cases | 2,112 | 74.3% | 88.6% |
| Integration turns | 204 | 73.0% | 92.6% |

The errors cost unequally. A scheduling turn sent to the drafting route always
fails, because that route has no calendar. A drafting turn sent to the scheduling
route can still succeed: in the centroid router's development, 12 drafting turns
went there and no accepted success was lost.

## Cost

The shared prefix made each feature a fraction of a second: 2,316 features took
under ten minutes. One r7i.4xlarge host cost $0.23, and the instance is terminated
with its security group retired. The first launch stopped before allocation when
GitHub could not assign runners to its CI jobs; it cost nothing. A3 has spent $39.21
of its $110 ceiling, plus stage 0's $2.15.
