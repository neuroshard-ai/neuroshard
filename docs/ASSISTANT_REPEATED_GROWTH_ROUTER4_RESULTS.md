# A3 calibrated router results: the shift recovers every misrouted scheduling turn

**Shifted toward scheduling by 0.747, an amount chosen on held-out training turns only,
the router sends none of the 104 integration turns that need the calendar to the drafting
route, down from 15. It still sends none of the other 100 to scheduling.** Held out, it
sends 2 of the 384 drafting training turns to scheduling, within the declared 3. As the
declaration requires, development runs with this router pinned.

Declaration: [calibrated router](ASSISTANT_REPEATED_GROWTH_ROUTER4.md). Evidence:
[result](../config/experiments/assistant-growth-router4-result.json) and
[report](../config/experiments/assistant-growth-router4-report.json), commit `becdace`.

## Routes

| | Unshifted | Shifted |
| --- | --- | --- |
| Held out: turns that need the calendar sent to drafting | 240 of 1,664 | 0 |
| Held out: drafting training turns sent to scheduling | 0 of 384 | 2 (3 allowed) |
| Held-out accuracy | 88.6% | 99.9% |
| Integration: turns that need the calendar sent to drafting | 15 of 104 | 0 |
| Integration: other turns sent to scheduling | 0 of 100 | 0 |
| Integration accuracy | 92.6% | 100% |

The two kinds of turn were already separated. The unshifted rule misrouted turns in one
direction only, so its boundary sat inside the scheduling turns. The shift moves it to the
edge of the drafting turns. Integration accuracy is reported, not gated. Development on the
opened cases decides whether the candidate reaches the sealed confirmation.

The refit recomputed all 2,316 features on a fresh host. They match the message-feature
router's bit for bit, with the same digest, and so do its 15 unshifted integration errors.

## Cost

One r7i.4xlarge host cost $0.22; the instance is terminated and its security group retired.
A3 has spent $39.43 of its $110 ceiling, plus stage 0's $2.15. The router file is staged
with round 4's units for the three development hosts.
