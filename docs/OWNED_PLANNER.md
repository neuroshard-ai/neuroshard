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

The answer-plan run completed its development and previously unopened final
comparison. Development improved from 48/56 to 53/56 correct answers. Final
answers improved from 129/144 to 136/144: mixed requests rose from 23/32 to
29/32, pronoun requests from 15/16 to 16/16, directory remained 32/32, protocol
16/16, structured 29/32 and ordinary-answer retention 14/16. All final answer
and category-retention thresholds passed. The **complete frozen gate failed**:
only 25/32 mixed plans were exact, below 28/32. Four further correct replies
omitted a terminal full stop in a directory question; three mixed plans were
invalid. These observations do not retroactively change the failed gate.

The final 16 optimizer updates replayed exactly. Median final response time was
3.04 seconds versus 2.89 seconds for the baseline; this is a sequential finite
workload, not a public load result. Inference bandwidth is unavailable: the
driver observed the all-owner transport counter while cached inference used
separate branch transports. Its recorded zeros must not be presented as zero
traffic. Complete evidence and limitations are in the
[result](../config/experiments/owned-answer-plan-results.json).
All five GPU hosts, disks and the temporary security group are retired. The
compute upper bound was $3.26, with storage and transfer separate.

This establishes an improvement in using existing distributed expert knowledge.
It does not establish repeated new-data learning, resource-matched growth or
authorization to promote this service on the native chain. Future question-plan
criteria should assess resolved requests instead of incidental punctuation,
with fresh acceptance data and rules committed before evaluation.

The operator service also accepts an optional `planned_tariff` file containing
integer `prompt_atom_price`, `output_atom_price` and the installed tokenizer's
`context` limit. `quote_planned` reserves the worst case across planning, two
argument calls, two answers and optional composition. Generation returns a
metering receipt; successful complete replay reproduces that receipt. Actual
prompt processing and neural output are charged separately. Planning includes
the additional adapter parameters in its owner shares; deterministic rendering
adds no neural token charge. Every receipt conserves its integer total and fits
the quoted bound. These receipts do not themselves authorize a ledger payment,
establish economically sufficient prices or replace funded neural verification.
The quote explicitly covers provider execution; verification and retained
evidence require separate funded obligations before native acceptance.
