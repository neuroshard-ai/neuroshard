# Router scaling study

October 9, 2026. A local measurement, not a gate. It trains no unit, opens no sealed or
development split, spends no cloud money and grants no checklist credit. Every number below
is reproducible from the commands at the end; intervals are 95% conversation-level bootstrap
intervals, and a difference is called real only when its paired interval excludes zero.

## Question

The accepted assistant routes each user turn to one of two units with a centroid rule on
the parent's mean message state ([router 3](ASSISTANT_REPEATED_GROWTH_ROUTER3.md),
calibrated by [router 4](ASSISTANT_REPEATED_GROWTH_ROUTER4.md)). An ever-growing assistant
adds a unit per cohort, and a misrouted turn always fails. **Does per-turn routing keep
working as units are added, does adding a unit break turns that routed correctly before,
and does routing survive wording the router was not fitted on?**

## Method

- **Routes.** The two real capabilities, then twelve authored fictional capabilities added
  one at a time ([`router_scaling_data`](../src/neuroshard/evolution/router_scaling_data.py)).
  Several are deliberately confusable: invoices vs drafting, rooms vs scheduling, approvals
  vs plan revisions, expenses vs invoices, reminders vs meetings. Follow-ups include generic
  turns shared by every capability ("Move that 3 calendar days later"), and every ordered
  pair of synthetic capabilities has one cross conversation.
- **Real anchors.** Drafting and scheduling turns from the real grammars' `train` and
  `cross-train` splits (48 sampled cases each) for fitting, and their `integration` and
  `cross-integration` splits for `test`, labelled by the accepted router's own rule
  (a turn needs scheduling if its expected outcome includes a meeting). The code refuses
  any sealed or development split.
- **Splits.** `fit`: 836 conversations, 1,615 turns. `test`: 556 conversations, 1,040 turns,
  the same phrasings with new values. `unseen`: 468 conversations, 909 turns, written in
  phrasings no router was fitted on, **including 48 reworded drafting and scheduling
  requests**, so earlier cohorts are re-checked on new wording too.
- **Rules** ([`router_scaling`](../src/neuroshard/evolution/router_scaling.py)). K-route
  centroids, the accepted rule (tested to agree with `assistant_selector.fit_centroid` at
  two routes), with refitted or frozen centring; and a multinomial logistic router on the
  same centred, normalised features.
- **Strategies.** `message`: each turn from its own message state, as served today.
  `with-opening`: the message state concatenated with the conversation's opening message
  state. `episode`: the whole conversation routed from its first turn, as the A2 gate does.
- **Features** ([`router_scaling_features`](../src/neuroshard/evolution/router_scaling_features.py)):
  the accepted feature, the mean state over the message after a shared cached prefix, on
  **SmolLM2-135M on CPU** as a stand-in for Granite 4.1 3B. Captured at hidden layers 8, 15,
  22 and the final normed layer (the accepted one; verified identical to `last_hidden_state`).
  Batched and single features differ by at most 2×10⁻⁶. A hashed word-bigram vector is a
  lexical floor.
- **Retention.** At each size, of the turns the previous router sent correctly, how many the
  router with one more route sends elsewhere.
- **Orders.** The declared order and four random orders of the synthetic capabilities.

## Check against the real router

With the two real routes, the centroid message rule on SmolLM2 misroutes 15 of the same 204
integration turns the [Granite router](ASSISTANT_REPEATED_GROWTH_ROUTER3_RESULTS.md) was
checked on: 92.6%, as Granite measured. The equal figure is a coincidence, not a calibration:
Granite's 15 errors were all scheduling turns sent to drafting, while 8 of SmolLM2's go the
other way, and this study fits on a 48-case sample of each training split where Granite's
router used all 2,112 training turns. It does show the stand-in
is not an easier feature than the parent on the real task.

## Results (SmolLM2-135M, final layer, declared order)

**Turn accuracy at 14 routes**, with 95% intervals:

| rule, strategy | `test` (familiar wording) | `unseen` (new wording) | `unseen` episodes |
| --- | --- | --- | --- |
| centroid, message (today's rule) | 0.769 [0.743, 0.794] | 0.441 [0.410, 0.471] | 0.256 |
| centroid, with-opening | 0.867 [0.847, 0.885] | 0.586 [0.550, 0.621] | 0.423 |
| logistic, message | 0.903 [0.883, 0.921] | 0.694 [0.664, 0.728] | 0.511 |
| **logistic, with-opening** | **0.987 [0.979, 0.992]** | **0.772 [0.742, 0.802]** | 0.635 |
| logistic, episode | 0.869 [0.850, 0.889] | 0.628 [0.590, 0.671] | 0.524 |

**As routes are added** (turn accuracy on `test`; range over five orders at 8 routes):

| routes | centroid, message | logistic, with-opening |
| --- | --- | --- |
| 2 | 0.926 | 1.000 |
| 3 | 0.872 | 1.000 |
| 8 | 0.857 (0.741–0.857 over orders) | 0.983 (0.982–0.993) |
| 14 | 0.769 | 0.987 |

**Paired differences at 14 routes** (second minus first, turn accuracy):

| comparison | `test` | `unseen` |
| --- | --- | --- |
| logistic vs centroid (message) | +0.134 [+0.111, +0.158] | +0.253 [+0.219, +0.290] |
| with-opening vs message (logistic) | +0.084 [+0.063, +0.104] | +0.078 [+0.048, +0.108] |
| logistic with-opening vs today's rule | +0.217 [+0.191, +0.245] | +0.331 [+0.293, +0.372] |
| layer 15 vs final (centroid, message) | +0.053 [+0.031, +0.075] | +0.222 [+0.193, +0.253] |
| layer 15 vs final (logistic, with-opening) | +0.002 [−0.005, +0.009] | +0.012 [−0.016, +0.038] |
| two fit phrasings vs one (logistic, with-opening, synthetic turns) | +0.033 [+0.019, +0.049] | +0.080 [+0.055, +0.107] |

**Retention**, summed over all twelve additions, per order (turns displaced that the previous
router had right): logistic with-opening loses 1–6 on `test` but 101–160 on `unseen`;
centroid message loses 71–136 on `test` and 126–217 on `unseen`.

**Reworded earlier cohorts.** On the 48 reworded drafting and scheduling conversations, with
only the two real routes, drafting recall is 0.26–0.34 depending on rule (scheduling 1.00).
At 14 routes, logistic with-opening recalls drafting 0.18 and scheduling 0.38. Of 38
reworded drafting turns, 7 stay on drafting and the rest spread over seven routes, most
often approvals (7) and summaries (6); of 39 reworded scheduling turns, as many go to rooms
(15) as stay on scheduling (15). The hashed floor recalls 0.82 of reworded drafting at two
routes and collapses as routes are added (0.41 of all `unseen` turns at 14).

**Abstention** (logistic with-opening, 14 routes; abstain on the least confident share):

| abstain | `test` kept accuracy | `unseen` kept accuracy | `unseen` errors kept |
| --- | --- | --- | --- |
| 0% | 0.987 | 0.772 | 207 |
| 5% | 0.999 | 0.796 | 176 |
| 10% | 1.000 | 0.814 | 152 |
| 30% | 1.000 | 0.909 | 58 |

**Unit descriptions.** A router fitted only on three written descriptions per unit routes
0.648 of `test` and 0.397 of `unseen` turns (centroid, message); blending them into the
fitted means (weights 0.25 and 0.5) changes accuracy by at most 0.03 either way. Written
cards, under this encoder, do not substitute for routed examples.

## Findings

1. **Today's rule does not scale with units.** The centroid message router falls from
   0.926 at two routes to 0.769 at 14, and adding a unit displaces 71–136 earlier-correct
   familiar turns over the twelve additions, depending on order. Freezing the centring did
   not help (0.745 at 14; not tested for significance).
2. **A discriminative router over conversation context scales on familiar wording.**
   Logistic with-opening stays at 0.987 at 14 routes, displaces at most 6 familiar turns
   over twelve additions, and is +0.217 better than today's rule there and +0.331 on new
   wording. Each ingredient helps independently: logistic over centroid, and context over
   the message alone. Routing the whole episode from its first turn cannot follow
   conversations that change capability.
3. **New wording is the bottleneck, and earlier cohorts suffer most.** On phrasings no
   router saw, the best router routes 0.772 of turns and 0.635 of conversations, and each
   addition displaces new-wording turns. Reworded drafting requests, the first accepted
   capability, are recalled 0.18–0.34 of the time, and poorly even before synthetic routes
   exist: the real
   grammars phrase each capability one way, so a router fitted on them learns templates,
   not needs. A cohort's sealed confirmation re-checks earlier cohorts only on their own
   grammar, so it would not see this.
4. **What helps with new wording.** Varied fit phrasing (+0.080 from a second phrasing per
   capability, significant) and abstention (30% abstained leaves 0.909 accuracy). Middle
   layers rescue the centroid rule (+0.222 at layer 15) but add nothing significant to the
   logistic router, which already weights the useful directions. Descriptions did not help.

## Consequences for growth

- **Router.** Replace the two-route centroid with a K-route discriminative router over the
  message and the conversation's opening, refitted on every cohort's routing turns whenever
  a unit is added.
- **Routing data is a contributed asset.** Each cohort should ship labelled routing turns in
  many phrasings, including paraphrases of every earlier capability; this is the cheapest
  measured lever on new wording. Contributor demonstrations can supply them.
- **Acceptance.** Re-check every earlier cohort's routing on paraphrased turns, and report
  per-route recall, not only the aggregate. A unit whose addition lowers an earlier route's
  recall beyond a declared margin should not be accepted.
- **Fallback.** Route low-confidence turns to the parent or ask a clarifying question; at 5%
  abstention familiar-wording errors fall from 14 to 1. Calibrate the threshold on
  paraphrased turns (see the packaged router below).
- **Next measurement.** Re-run with Granite 4.1 3B features on one CPU host before cohort 4
  (the scripts take `--model`), and fit the router on the real accepted units' routing turns
  plus paraphrases, measuring end-to-end episode success rather than labels alone.

## Packaged router

[`assistant_turn_router`](../src/neuroshard/evolution/assistant_turn_router.py) packages the
findings as a router file any party can verify and recompute: the logistic router over
`message ⊕ opening`, a fallback route for low-confidence turns, and an admission rule for
adding a unit. It is not wired into serving. On the study's SmolLM2 features it reproduces
the study exactly (0.987 `test`, 0.772 `unseen` at 14 routes).

**Calibration on fit folds does not transfer to new wording.** The threshold is chosen on
held-out folds of the fit conversations as the largest coverage meeting a target kept
accuracy, with at least half the turns kept:

| target | `test` coverage / kept accuracy / misrouted | `unseen` coverage / kept accuracy / misrouted |
| --- | --- | --- |
| none | 1.000 / 0.987 / 14 | 1.000 / 0.772 / 207 |
| 0.98 | 0.999 / 0.987 / 13 | 0.955 / 0.796 / 177 |
| 0.99 | 0.988 / 0.990 / 10 | 0.733 / 0.892 / 72 |

At target 0.99 the router meets it on familiar wording and falls to 0.892 on new wording,
though misroutes fall from 207 to 72. A deployed threshold must be calibrated on paraphrased
turns, not on the router's own fit data.

**Admission.** A candidate with one more route is refused when an earlier route's recall on
the paraphrase check set falls by more than 0.05 *and* the turns it lost significantly
outnumber those it gained (one-sided exact sign test, α = 0.05), or when an earlier route has
fewer than 30 check turns. With 38–76 `unseen` turns per route, one turn is worth 1.3–2.6
points, so a plain margin refuses almost every addition on noise; the paired test does not.
Replaying the twelve additions (each compared with the router before it, whether or not
that one was admitted), the rule refuses five, each a large one-way displacement:

| added | displaced route | check turns | recall before → after | lost / gained | p |
| --- | --- | --- | --- | --- | --- |
| invoices | scheduling | 39 | 1.00 → 0.77 | 9 / 0 | 0.002 |
| rooms | scheduling | 39 | 0.62 → 0.31 | 12 / 0 | 0.0002 |
| expenses | invoices | 54 | 0.98 → 0.63 | 19 / 0 | <10⁻⁵ |
| approvals | drafting | 38 | 0.34 → 0.03 | 12 / 0 | 0.0002 |
| timesheets | reminders | 68 | 0.96 → 0.74 | 15 / 0 | 3×10⁻⁵ |

These are the confusable pairs the grammars were written to contain; the other seven
additions pass. A refused unit is not useless: its cohort should add paraphrased routing
turns for the route it displaced and refit, which the diversity result above suggests will help.

## Limits

Authored templates, not real user language; `test` shares templates with `fit`, so `unseen`
is the meaningful check, and its phrasings are also authored. SmolLM2-135M stands in for the
parent. Labels say which unit a turn needs; no unit was served, so this measures routing,
not conversation success. Abstention thresholds are quantiles on the evaluated turns, a
description of the trade-off rather than a calibrated policy. One fit sample of the real
grammars; five orders of the synthetic capabilities.

## Reproduce

```bash
PYTHONPATH=src python scripts/run_router_scaling.py --encoder lm \
    --model .neuroshard/seed-smollm2-135m --out .neuroshard/router-scaling/smollm2-v2   # ~25 min on 2 threads
PYTHONPATH=src python scripts/analyse_router_scaling.py --out .neuroshard/router-scaling/smollm2-v2
PYTHONPATH=src python scripts/run_router_scaling.py --encoder hashed --out .neuroshard/router-scaling/hashed2
```

Full per-size tables for every rule, strategy and split: [SmolLM2](router-scaling/smollm2.md),
[hashed](router-scaling/hashed.md); paired comparisons: [analysis](router-scaling/smollm2-analysis.json).
