# Router scaling study

October 10, 2026. A measurement, not a gate. It trains no unit, opens no sealed or development
split and grants no checklist credit. Features come from the pinned Granite 4.1 3B parent on one
canonical CPU host ($0.23), with SmolLM2-135M on a local CPU as a comparison. Every number
below is reproducible from the commands at the end. Intervals are 95% bootstrap intervals that
resample whole conversations; a difference counts only when its paired interval excludes zero.

## Question

The accepted assistant routes each user turn to one of two units with a centroid rule on the
parent's mean message state ([router 3](ASSISTANT_REPEATED_GROWTH_ROUTER3.md), calibrated by
[router 4](ASSISTANT_REPEATED_GROWTH_ROUTER4.md)). An ever-growing assistant adds a unit per
cohort, and a misrouted turn always fails. **Does per-turn routing keep working as units are
added? Does adding a unit break turns that routed correctly before? Does routing survive
wording the router was not fitted on?**

## Method

- **Routes.** The two real capabilities, then twelve authored fictional capabilities, added one
  at a time ([`router_scaling_data`](../src/neuroshard/evolution/router_scaling_data.py)).
  Several are deliberately confusable: invoices vs drafting, rooms vs scheduling, approvals vs
  plan revisions, expenses vs invoices, reminders vs meetings. Follow-ups include generic turns
  shared by every capability ("Move that 3 calendar days later"). Every ordered pair of
  synthetic capabilities has one cross conversation.
- **Real anchors.** Turns from the real grammars' training splits for fitting (48 sampled cases
  from each `train` split and all 32 `cross-train` cases), and their integration splits for
  `test`. Labels follow the accepted router's rule: a turn needs scheduling if its expected
  outcome includes a meeting. The code refuses any sealed or development split.
- **Splits.** `fit`: 836 conversations, 1,615 turns. `test`: 556 conversations, 1,040 turns, the
  same phrasings with new values. `unseen`: 468 conversations, 909 turns, in phrasings no router
  was fitted on. It includes 48 reworded drafting and scheduling conversations, so earlier
  cohorts are also re-checked on new wording.
- **Rules** ([`router_scaling`](../src/neuroshard/evolution/router_scaling.py)). K-route
  centroids, the accepted rule (tested to agree with `assistant_selector.fit_centroid` at two
  routes), with refitted or frozen centring. Also a multinomial logistic router on the same
  centred, normalised features.
- **Strategies.** `message`: each turn routed from its own message state, as served today.
  `with-opening`: the message state concatenated with the conversation's opening message state.
  `episode`: the whole conversation routed from its first turn, as the A2 gate does.
- **Features.** The accepted router feature: the mean final-layer state over the user message
  after the drafting policy's shared prefix
  ([`router_scaling_granite`](../src/neuroshard/evolution/router_scaling_granite.py)). The
  worker also checked its first 8 texts against `assistant_routing.message_feature` itself: they
  matched bit for bit. It also captured hidden layers 20 and 30 from the same forward pass. All
  2,910 texts took 604 s at 8.4 GB peak memory. A hashed word-bigram vector is a lexical floor.
- **Retention.** At each size: of the turns the previous router sent correctly, how many the
  router with one more route sends elsewhere.
- **Orders.** The declared order and four random orders of the synthetic capabilities.

## Check against the real router

With the two real routes, the centroid message rule on Granite routes all 204 integration turns
correctly. The [real router 3 fit](ASSISTANT_REPEATED_GROWTH_ROUTER3_RESULTS.md) misrouted 15
of them (92.6%), so this study's two-route setup is *easier* than the real one. The fit data
differs: this study sampled 128 real conversations and labelled turns by expected outcome,
while router 3 used all 2,112 training turns labelled by the solver's calendar calls. This
study did not isolate which difference matters. The comparisons below are between rules on
the same data, not absolute predictions for the served router. SmolLM2, by contrast, misrouted
15 of the same 204 turns.

## Results (Granite 4.1 3B, final layer, declared order)

**Turn accuracy at 14 routes**, with 95% intervals:

| rule, strategy | `test` (familiar wording) | `unseen` (new wording) | `unseen` conversations |
| --- | --- | --- | --- |
| centroid, message (today's rule) | 0.878 [0.855, 0.896] | 0.685 [0.656, 0.719] | 0.513 |
| centroid, with-opening | 0.895 [0.877, 0.913] | 0.734 [0.704, 0.765] | 0.590 |
| logistic, message | 0.900 [0.879, 0.918] | 0.781 [0.755, 0.809] | 0.622 |
| **logistic, with-opening** | **0.995 [0.990, 0.999]** | **0.931 [0.913, 0.947]** | **0.880** |
| logistic, episode | 0.869 [0.850, 0.889] | 0.778 [0.743, 0.807] | 0.660 |

**As routes are added** (turn accuracy; range over five orders at 8 routes):

| routes | centroid, message: `test` / `unseen` | logistic, with-opening: `test` / `unseen` |
| --- | --- | --- |
| 2 | 1.000 / 0.844 | 1.000 / 1.000 |
| 3 | 0.936 / 0.696 | 1.000 / 0.664 |
| 8 | 0.892 / 0.622 (orders 0.884–0.912 / 0.603–0.816) | 0.993 / 0.896 (0.993–0.996 / 0.896–0.932) |
| 14 | 0.878 / 0.685 | 0.995 / 0.931 |

**Paired differences at 14 routes** (second minus first, turn accuracy):

| comparison | `test` | `unseen` |
| --- | --- | --- |
| logistic with-opening vs today's rule | +0.117 [+0.097, +0.140] | +0.245 [+0.209, +0.279] |
| logistic vs centroid (message) | +0.022 [+0.010, +0.035] | +0.096 [+0.065, +0.128] |
| with-opening vs message (logistic) | +0.095 [+0.076, +0.116] | +0.150 [+0.119, +0.178] |
| with-opening vs message (centroid) | +0.017 [−0.013, +0.049] | +0.048 [+0.017, +0.079] |
| layer 30 vs final (centroid, message) | −0.014 [−0.030, +0.000] | +0.102 [+0.074, +0.129] |
| layer 30 vs final (logistic, with-opening) | +0.001 [−0.003, +0.006] | −0.017 [−0.031, −0.002] |
| layer 20 vs final (logistic, with-opening) | −0.001 [−0.006, +0.003] | −0.052 [−0.071, −0.034] |
| two fit phrasings vs one (logistic, with-opening, synthetic turns) | +0.029 [+0.016, +0.043] | +0.036 [+0.020, +0.053] |

**Retention.** Summed over all twelve additions, per order, these are the turns displaced that
the previous router had right. Logistic with-opening loses 0–3 on `test` and 66–86 on `unseen`.
Centroid message loses 21–76 on `test` and 67–126 on `unseen`.

**Reworded earlier cohorts.** Logistic with-opening routes every one of the 48 reworded real
conversations correctly while only the two real routes exist. The first look-alike capability
breaks that. Adding invoices drops reworded drafting recall from 1.00 to 0.37 and scheduling
from 1.00 to 0.54, displacing 42 turns at once. At 14 routes, drafting recall is 0.45 and
scheduling 0.82. Of 38 reworded drafting turns, 17 reach drafting; the rest go to summaries
(10), invoices (6) and approvals (4). Of 39 reworded scheduling turns, 7 go to rooms. Every
other route has `unseen` recall of at least 0.90 at 14 routes (lowest: invoices 0.90, travel and
tickets 0.91). The earliest cohort is the weakest route.

**Abstention** (logistic with-opening, 14 routes; abstain on the least confident share):

| abstain | `test` kept accuracy | `unseen` kept accuracy | `unseen` errors kept |
| --- | --- | --- | --- |
| 0% | 0.995 | 0.931 | 63 |
| 5% | 1.000 | 0.951 | 42 |
| 10% | 1.000 | 0.968 | 26 |
| 20% | 1.000 | 0.990 | 7 |

**Unit descriptions.** A router fitted only on three written descriptions per unit routes 0.577
of `test` and 0.618 of `unseen` turns (centroid, message). Blending them into the fitted means
at weight 0.5 raises `unseen` from 0.685 to 0.713 (message) and 0.734 to 0.790 (with-opening),
and changes `test` by −0.035 to 0. The blend was not compared with paired intervals; the
logistic router, which cannot use them this way, is better than either.

## Findings

1. **Today's rule loses accuracy as units are added.** The centroid message router falls from
   1.000 at two routes to 0.878 at 14 on familiar wording, and to 0.685 on new wording. Over
   the twelve additions it displaces 21–76 earlier-correct familiar turns, depending on order.
   Freezing the centring is worse (0.830).
2. **A discriminative router over conversation context scales.** Logistic with-opening holds
   0.995 on familiar wording and 0.931 on new wording at 14 routes, with 0.880 of new-wording
   conversations entirely correct. Over twelve additions it displaces at most 3 familiar turns.
   Both ingredients are needed: with Granite, context helps the logistic router (+0.150 on new
   wording) far more than the centroid (+0.048), and logistic alone adds +0.096.
3. **Earlier cohorts are the weak spot, and one look-alike unit is enough to expose it.** The
   real grammars phrase each capability one way. A router fitted on them separates drafting
   from scheduling on any wording, but when a capability with overlapping vocabulary arrives
   (invoices: "draft", "team", "due"), reworded drafting requests move to it. A cohort's sealed
   confirmation re-checks earlier cohorts only on their own grammar, so it would not see this.
4. **The parent's final layer is the right feature for the logistic router.** Middle layers
   rescue the centroid rule (+0.102 at layer 30 on new wording) but make the logistic router
   significantly worse (−0.017 and −0.052). The served feature needs no change.
5. **Abstention is cheap on Granite.** Abstaining on the least confident 10% of new-wording turns
   leaves 0.968 accuracy on the rest, and 20% leaves 0.990.
6. **SmolLM2 understated the gains and overstated the problem.** On SmolLM2 the best router
   reached 0.772 on new wording and reworded drafting recall was 0.18. Directions agree between
   encoders; magnitudes do not. Conclusions about the served assistant should come from Granite.

## Paraphrases of the real requests

[`router_scaling_paraphrase`](../src/neuroshard/evolution/router_scaling_paraphrase.py) rewrites
only the opening frame and closing instruction of each real request. Every meaning-carrying
clause is kept verbatim, and each paraphrase is checked to contain them: project, recipient,
revision and date/total rules, attendees, duration, and date and time constraints. The expected
outcomes are therefore unchanged, and the same turns can later be served end to end. The frames
share no opening with the held-out rewordings. Results from adding one paraphrase of each of the
128 real fit conversations:

| encoder, check (logistic with-opening, 14 routes) | without | with | paired difference |
| --- | --- | --- | --- |
| Granite, `unseen` | 0.931 | 0.926 | −0.005 [−0.014, +0.004] |
| Granite, reworded drafting recall | 0.45 | 0.66 | |
| Granite, reworded scheduling recall | 0.82 | 0.64 | |
| Granite, `test` | 0.995 | 0.991 | −0.004 [−0.008, −0.001] |
| SmolLM2, `unseen` | 0.772 | 0.787 | +0.014 [+0.003, +0.026] |

On Granite, frame paraphrases trade one earlier cohort for the other. Reworded drafting turns
that reached drafting rose from 17 to 25 of 38, but reworded scheduling turns sent to rooms rose
from 7 to 14 of 39. The overall effect is not significant. Authored frames are not the lever: the
held-out rewordings change content words ("handoff written up", "first slot", "hold it"), and
routing data for earlier cohorts needs that kind of variety, which contributor demonstrations
can supply and templates cannot.

## Consequences for growth

- **Router.** Replace the two-route centroid with a K-route logistic router over the message
  and the conversation's opening, on the parent's final layer as served today. Refit it on
  every cohort's routing turns whenever a unit is added.
- **Acceptance.** Re-check every earlier cohort's routing on reworded turns, and report per-route
  recall, not only the aggregate. Refuse a unit whose addition significantly displaces an
  earlier route (the rule below).
- **Routing data is a contributed asset.** Each cohort should ship labelled routing turns written
  in varied words, including rewordings of every earlier capability's requests. A refused unit's
  cohort should add such turns for the route it displaced and refit.
- **Fallback.** Send low-confidence turns to the parent or a clarifying question. Calibrate the
  threshold on reworded turns, not the router's own fit data (below).
- **Next measurement.** Put the router in front of the real accepted units and measure
  whole-conversation success, not labels. The paraphrased real requests keep their expected
  outcomes, so they can be served and scored unchanged.

## Packaged router

[`assistant_turn_router`](../src/neuroshard/evolution/assistant_turn_router.py) packages the
findings as a router file any party can verify and recompute. It holds the logistic router over
`message ⊕ opening`, a fallback route for low-confidence turns, and an admission rule for adding
a unit. It is not wired into serving. On the study's Granite features it reproduces the study
exactly (0.995 `test`, 0.931 `unseen` at 14 routes).

**Calibration on fit folds.** The threshold is chosen on held-out folds of the fit
conversations: the largest coverage that meets a target kept accuracy, with at least half the
turns kept.

| target | `test` coverage / kept accuracy / misrouted | `unseen` coverage / kept accuracy / misrouted |
| --- | --- | --- |
| none, 0.95, 0.98 | 1.000 / 0.995 / 5 | 1.000 / 0.931 / 63 |
| 0.99 | 0.991 / 0.999 / 1 | 0.931 / 0.959 / 35 |

Held-out fit accuracy is 0.983, so targets up to 0.98 abstain on nothing. At 0.99 the router
meets the target on familiar wording and reaches 0.959 on new wording, while abstaining on 6.9%
of new-wording turns; misroutes fall from 63 to 35. Fit folds understate how uncertain new
wording is; a deployed threshold should be calibrated on reworded turns.

**Admission.** A candidate with one more route is refused in two cases. The first is when an
earlier route's recall on the reworded check set falls by more than 0.05 *and* the turns it lost
significantly outnumber those it gained (one-sided exact sign test, α = 0.05). The second is when
an earlier route has fewer than 30 check turns. With 38–76 `unseen` turns per route, one turn is
worth 1.3–2.6 points, so a plain margin would refuse almost every addition on noise; the paired
test does not. Each addition was compared with the router before it, whether or not that one was
admitted. Replaying the twelve additions, the rule refuses three, each a one-way displacement:

| added | displaced route | check turns | recall before → after | lost / gained | p |
| --- | --- | --- | --- | --- | --- |
| invoices | drafting | 38 | 1.00 → 0.37 | 24 / 0 | 6×10⁻⁸ |
| invoices | scheduling | 39 | 1.00 → 0.54 | 18 / 0 | 4×10⁻⁶ |
| rooms | scheduling | 39 | 0.64 → 0.41 | 9 / 0 | 0.002 |
| approvals | drafting | 38 | 0.68 → 0.53 | 6 / 0 | 0.016 |

These are the look-alikes of the earliest cohorts: invoices and approvals for drafting, rooms for
scheduling. The other nine additions pass. On SmolLM2 the same rule refused five, including
expenses (displacing invoices) and timesheets (displacing reminders), which Granite separates.

## Limits

- **Templates.** The phrasings are authored templates, not real user language. `test` shares
  templates with `fit`, so `unseen` is the meaningful check, and its phrasings are also
  authored.
- **Labels, not success.** Labels say which unit a turn needs; no unit was served, so this
  measures routing, not conversation success.
- **Abstention.** The study's abstention table abstains on quantiles of the evaluated turns,
  which describes the trade-off. The packaged router's threshold is calibrated on fit folds and
  then judged on `test` and `unseen`.
- **Sampling.** The real grammars were sampled once for fitting. The synthetic capabilities were
  added in five orders.
- **Admission replay.** Admission was replayed on the same `unseen` turns the study reports, so
  it demonstrates the rule; it is not a fresh confirmation.
- **Easier than the served router.** The two-route setup is easier than the served router's (see
  above).

## Reproduce

```bash
# Granite features on one canonical CPU host (r7i.4xlarge, 2-hour expiry, retired automatically)
PYTHONPATH=src python scripts/modular_reference_cloud.py run --profile router-scaling-granite --home <home>
F=<home>/evidence/.study/attempts/baseline-features
PYTHONPATH=src python scripts/run_router_scaling.py --encoder lm --features $F --out <out>
PYTHONPATH=src python scripts/analyse_router_scaling.py --out <out> --features $F
# SmolLM2 comparison and the hashed floor, local CPU
PYTHONPATH=src python scripts/run_router_scaling.py --encoder lm --model .neuroshard/seed-smollm2-135m --out <out2>
PYTHONPATH=src python scripts/analyse_router_scaling.py --out <out2>
PYTHONPATH=src python scripts/run_router_scaling.py --encoder hashed --out <out3>
```

Evidence: [run report](../config/experiments/router-scaling-granite-report.json) (features
digest, resources, CI run) and [result](../config/experiments/router-scaling-granite-result.json).
Full per-size tables: [Granite](router-scaling/granite.md), [SmolLM2](router-scaling/smollm2.md),
[hashed](router-scaling/hashed.md). Comparisons, layers, diversity, paraphrases and admission:
[Granite analysis](router-scaling/granite-analysis.json),
[SmolLM2 analysis](router-scaling/smollm2-analysis.json). The 86 MB feature file is not
committed; its digest is in the run report.
