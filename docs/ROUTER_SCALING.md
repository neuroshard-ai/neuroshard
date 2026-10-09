# Router scaling study

October 9, 2026. A local measurement, not a gate. It trains no unit, opens no sealed or
development split, spends no cloud money and grants no checklist credit.

## Question

The accepted assistant routes each user turn to one of two units with a centroid rule on
the parent's mean message state ([router 3](ASSISTANT_REPEATED_GROWTH_ROUTER3.md),
calibrated by [router 4](ASSISTANT_REPEATED_GROWTH_ROUTER4.md)). An ever-growing assistant
adds a unit per cohort, and a misrouted turn always fails. **Does per-turn routing keep
working as units are added, and does adding a unit break turns that routed correctly
before?**

## Method

- **Routes.** The two real capabilities, drafting and scheduling (anchors from their
  `train`/`cross-train` and `integration`/`cross-integration` splits only), then twelve
  authored fictional capabilities added one at a time in a fixed order
  ([`router_scaling_data`](../src/neuroshard/evolution/router_scaling_data.py)). Several were
  chosen to be confusable with the real ones and with each other: invoices vs drafting,
  rooms vs scheduling, approvals vs plan revisions, expenses vs invoices, reminders vs
  meetings. Follow-ups include generic turns shared across capabilities ("Move that 3
  calendar days later"), and every pair of synthetic capabilities has a cross conversation.
- **Splits.** `fit` fits routers; `test` uses the same phrasings with new values;
  `unseen` uses phrasings no router was fitted on, to measure generalisation to wording.
- **Rules** ([`router_scaling`](../src/neuroshard/evolution/router_scaling.py)). K-route
  centroids (the accepted rule; it agrees with `assistant_selector.fit_centroid` at two
  routes, tested), with refitted or frozen centring, and a multinomial logistic router on
  the same features.
- **Strategies.** `message`: each turn from its own message state, as served today.
  `with-opening`: the message state concatenated with the conversation's opening message
  state. `episode`: route the whole conversation from its first turn, as the A2 gate does.
- **Features** ([`router_scaling_features`](../src/neuroshard/evolution/router_scaling_features.py)):
  the accepted feature (mean final-layer state over the message after a shared prefix) on
  **SmolLM2-135M on CPU**, a stand-in for Granite 4.1 3B, plus a hashed word-bigram floor.
- **Retention.** At each size: of the turns the previous router sent correctly, how many
  the router with one more route now sends elsewhere.

836 fit conversations, 556 test, 420 unseen. Features took 21 minutes on two threads.
Full tables for every rule, strategy and split: [SmolLM2](router-scaling/smollm2.md),
[hashed](router-scaling/hashed.md).

```bash
PYTHONPATH=src python scripts/run_router_scaling.py --encoder hashed --out .neuroshard/router-scaling/hashed
PYTHONPATH=src python scripts/run_router_scaling.py --encoder lm --model .neuroshard/seed-smollm2-135m \
    --out .neuroshard/router-scaling/smollm2
```

## Results (SmolLM2-135M features)

Turn accuracy on `test` (familiar phrasing, new values), and the lowest per-route recall:

| routes | centroid, message (today's rule) | logistic, message | logistic, with-opening | episode (logistic) |
| --- | --- | --- | --- | --- |
| 2 | 0.926 / 0.920 | 1.000 / 1.000 | 1.000 / 1.000 | 0.980 / 0.962 |
| 4 | 0.913 / 0.833 | 0.966 / 0.875 | 0.993 / 0.978 | 0.980 / 0.962 |
| 8 | 0.857 / 0.722 | 0.919 / 0.768 | 0.983 / 0.949 | 0.938 / 0.907 |
| 10 | 0.803 / 0.562 | 0.914 / 0.783 | 0.987 / 0.952 | 0.913 / 0.870 |
| 14 | **0.769 / 0.479** | 0.903 / 0.809 | **0.987 / 0.955** | 0.869 / 0.823 |

On `unseen` phrasing, at 14 routes:

| rule, strategy | turn acc | episode acc | min recall |
| --- | --- | --- | --- |
| centroid, message | 0.441 | 0.252 | 0.044 |
| centroid, with-opening | 0.589 | 0.426 | 0.090 |
| logistic, message | 0.733 | 0.550 | 0.471 |
| logistic, with-opening | **0.817** | **0.686** | 0.627 |

Retention: adding one route displaced up to 17 of 189 previously correct `test` turns for
the centroid message rule (at the third route), and 0–1 per step for logistic
with-opening. On `unseen`, adding a route displaces up to 47 previously correct turns per
step (episode routing at the expenses step), and even the best rule loses 18–30 at some steps.

## Findings

1. **Today's rule does not scale.** The centroid message router falls from 92.6% at two
   routes (the same figure the real Granite router measured on integration) to 76.9% at 14,
   with one route recalled less than half the time. Class means of a shared, frozen state
   crowd together as routes are added. Freezing the centring makes it worse, not better.
2. **Context fixes most of it.** Generic follow-ups ("make it 2 days earlier") cannot be
   routed from the message alone. Adding the opening message's state and fitting a
   discriminative router held 98.7% turn and 97.5% episode accuracy at 14 routes, with
   near-zero displacement of earlier decisions. Routing the whole episode from its first
   turn is worse: it cannot follow a conversation that changes capability.
3. **Wording generalisation is the real bottleneck.** On phrasings no router saw, the best
   router reaches 81.7% of turns and 68.6% of conversations at 14 routes, and adding a route
   regularly breaks turns that routed correctly before. A cohort's sealed confirmation
   re-checks earlier cohorts on their own grammar's phrasing, so it would not detect this.
4. **The frozen state is a weak feature.** A hashed word-bigram vector matched or beat
   SmolLM2's message state under the centroid rule (88.5% vs 76.9% at 14 routes). Granite's
   states are richer, so this is a lower bound, but it says the routing feature, not the
   number of routes, sets the ceiling.

## What this means for growth

- Replace the two-route centroid with a K-route discriminative router over **conversation
  context**, refitted on every earlier cohort's routing data whenever a unit is added.
- Treat routing as part of every cohort's acceptance: re-check earlier cohorts' routing with
  **paraphrased** turns, not just their own grammar, and report per-route recall, not only
  the aggregate.
- Make routing data a contributed asset: each cohort ships its labelled turns, so the
  router can be refitted without access to private conversations.
- Before cohort 4, re-run this study with Granite 4.1 3B features on one CPU host (the
  script takes `--model`); the authored capabilities are already in place.

## Limits

Authored templates, not real user language; the `test` split shares templates with `fit`,
so `unseen` is the meaningful generalisation check. SmolLM2-135M stands in for the parent.
Labels say which unit a turn needs; no unit was served, so this measures routing alone, not
end-to-end success. The order of added capabilities is one declared order.
