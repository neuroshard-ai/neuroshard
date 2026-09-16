# Learning to query owned experts

The jointly trained token-level interface reached low training loss but still
failed combined held-out questions. The next method preserves each expert's
successful question interface and trains the assistant to request the relevant
facts. This follows the general learned-call direction of
[Toolformer](https://arxiv.org/abs/2302.04761), with separately owned neural
experts supplying the results. That paper does not establish this implementation's
accuracy or decentralization.

`PlannerTraining` backpropagates through the preserved assistant's final owned
partition. Its first two owners execute their frozen layers; the output owner
computes response-only cross-entropy and returns hidden-state gradients. Only a
separate adapter on the last owner enters Adam. Planning temporarily installs
that adapter. Ordinary answering and specialist execution retain their loaded
weights. The service binds the complete adapter checkpoint and rejects a changed
adapter before generation. Every planning, argument, expert and optional answer
composition call remains in the replayable, metered response.

Supervision consists of input questions and resolved arguments, without factual
reference answers. The grouped corpus supplies existing single and combined
questions, ordinary assistant/structured requests and explicit pronoun examples.
Original subject/topic groups remain disjoint between fitting, development and
final roles. Initial upstream system instructions are preserved in visible user
input because this conversation interface starts with a user; both measured
arms receive the same normalized conversation. Preparation templates construct
labels only and are not inference-time routing rules.

The numerical check trains through actual process boundaries, restores a saved
Adam state and reproduces the next update exactly. It checks that the adapter is
used only for planning, that ordinary answering remains unchanged after removal,
and that another adapter cannot serve under the old service identity. These are
execution checks; useful held-out decomposition and answers require the frozen
GPU comparison. This fits an interface to existing experts and does not count
as a new admitted expert cohort or a resource-matched growth comparison.

The [frozen trial](../config/experiments/owned-planner-trial.json) fits 1,152
question-only records for 288 updates, with an exact replay of the final 16
optimizer steps. It measures 56 development conversations before and after
training; the 144 final conversations remain unopened unless every development
gate passes. Single-expert answers, structured tasks and unchanged ordinary
answers each require at least 87.5% accuracy; combined and pronoun requests
require at least 75%. Combined accuracy must improve by at least 12.5 percentage
points over the untrained-planner baseline, with no category regression. Separate
question-plan thresholds also apply. Both arms use the same frozen experts,
router and optional neural answer composer. All composer calls are recorded.

The budget is five A10G owners for at most two hours, with a $20 planning cap
and automatic termination. The run stops on infrastructure or numerical failure
and never adds epochs or selects another checkpoint after scoring. Five focused
data, distributed training and conversation checks passed in 28.60 seconds
before the prescription was frozen. This is an interface-learning experiment;
the six-item live-LLM checklist remains unchanged.

The initial allocation was stopped during bootstrap, before numerical execution:
the shared conversation encoder labeled earlier assistant turns as well as the
final plan. The [corrected prescription](../config/experiments/owned-planner-trial-retry.json)
masks all preceding turns. A full check of all 1,352 prepared records reproduced
the exact inference prefix and decoded only the intended final JSON plan from
the supervised tokens. Four focused data checks passed in 0.05 seconds. The
retry retains the original records, optimizer schedule, quality gates, combined
$20 cap and absolute deadline of 2026-09-16 23:24:03 UTC. Its earlier frozen
source and stopped allocation remain recorded.

The corrected run completed. Correct development answers improved from 13/56
to 48/56, but the frozen gates **failed** and the 144 final cases remained
unopened. All 48 original question plans were exact. Eight pronoun plans used
valid paraphrases instead of the required literal wording; all eight actual
answers were correct. The experts returned both correct values for all 16 mixed
requests, and correct values for all 24 single factual requests. The final
composer mishandled six mixed responses and one single response. Ordinary
answers were unchanged in 8/8 cases; structured answers reached 7/8.
The final 16 optimizer updates replayed exactly. Complete checkpoints, outputs,
timings and source identities are in the [published result](../config/experiments/owned-planner-results.json).
Both allocations are retired; their combined compute upper bound was $2.70,
with storage and transfer separate.

The [next prescription](../config/experiments/owned-answer-plan-trial.json)
starts from that retained planner, with fresh Adam and one fixed 144-update
epoch. It teaches a bounded output program: the neural planner chooses the
questions and either `short`, `semicolon` or `assistant` rendering. Short forms
extract actual neural source values using each expert's declared response schema
and preserve their order. They cannot supply an answer from labels or a lookup
table. Open-ended composition still uses the assistant. Invalid source values,
ambiguous JSON and output overruns fail explicitly.

The comparison reuses the complete published 56-case development result from
the **trained** planner. It verifies the original record root and every actual
conversation before reuse. Final baseline inference executes afresh only after
all development gates pass. Correct-answer thresholds and the prohibition on
category regression remain unchanged; mixed accuracy must improve by another
12.5 percentage points. Prospectively, pronoun planning is judged by the actual
resolved name and field, together with the selected output operation, instead
of exact synonymous question wording. Other question-plan gates remain exact.
This change does not turn the previous failed trial into a pass.

This bounded trial allows five A10G hosts for at most one hour and a $10 planning
cap. It tests composition of existing learned facts. Three admitted cohorts,
broader assistant quality and matched-resource growth remain separate unmet
criteria in the fixed checklist.
