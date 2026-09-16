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
