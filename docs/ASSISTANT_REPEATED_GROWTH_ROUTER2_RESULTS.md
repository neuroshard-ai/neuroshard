# A3 turn-need router results: clean labels, uninformative features

**The router trained on what each turn needs is only about 74% accurate, so the
features, not the labels, now limit routing.** It was fitted on 2,112 training turns,
1,664 of them needing the calendar. On held-out training cases it chose correctly
74% of the time, and on the 204 integration turns 73%. Routing every turn to
scheduling would score 79% on the training turns. The parent's final-layer state at
the start of its reply encodes little of what the user asked for. Development has
not been run with this router.

Declaration: [turn-need router](ASSISTANT_REPEATED_GROWTH_ROUTER2.md). Evidence:
[result](../config/experiments/assistant-growth-router2-result.json) and
[report](../config/experiments/assistant-growth-router2-report.json), commit `daf6692`.

## Accuracy

| | Turns | Correct |
| --- | --- | --- |
| Held out, four folds of training cases | 2,112 | 74.3% |
| Integration turns | 204 | 73.0% |

On the integration turns every drafting first turn went to the drafting route.
The errors were 31 scheduling and cross turns sent to the drafting route and 24
drafting follow-ups sent to the scheduling route. A turn's feature is the parent's
final-layer state at the first token of its reply, after the instruction, the tools
and that message. It appears to separate a first request from a short follow-up
better than it separates scheduling from drafting.

## Cost

One r7i.4xlarge host computed 2,316 features in 1.7 hours, $1.82 in total. AWS no
longer listed the terminated instance when the report was written, and its security
group is retired. A3 has spent $38.98 of its raised $110 ceiling, plus stage 0's $2.15.
