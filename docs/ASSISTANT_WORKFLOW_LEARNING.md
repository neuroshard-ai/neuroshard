# Complete workspace-assistant learning contract

September 27, 2026. New experiment; no prior study is resumed. The deliverable is
a conversational assistant that reads a document, chooses its approved version,
calculates a date or quantity, saves a draft, and correctly applies a follow-up.
The authoritative output is the structured draft shown to the user. Free-text
acknowledgements are recorded but not semantically graded. This finite authored
workflow grammar is not a benchmark of arbitrary personal-assistant intelligence.

## Frozen method

The [learning plan](../config/experiments/assistant-workflow-learning.json) declares
three arms against the pinned Granite 4.1 3B parent: unchanged parent; q/v updates
to the existing last eight layers; and a frozen backbone with rank-16 q/v LoRA in
those layers. The addition contains 1,048,576 new parameters. The update trains
62,914,560 existing parameters; it needs the prior parent for routing/rollback,
which is included in its storage bill. Neither arm is implemented by merging two
independently trained tail vectors.

Both trained arms receive the same assistant-decision windows, 128 optimizer
steps and a declared time allowance. Actual compute is reported: equal schedules
are not equal FLOPs. After freezing each arm, fit the declared binary selector
using only the separate integration partition. The selector sees a frozen parent
feature at the first user request, never scoring labels or future corrections.
It selects once per episode; tools and arguments remain automatically generated.
Parent feature extraction and all integration rollouts are charged. There is no
candidate KV-cache reuse from an adapted prefix without recomputation.

The training driver, prepared trajectories and exact paid execution inventory
are still to be implemented and frozen. This contract does not launch training.
No 135M fallback or hidden-label selection substitutes for this candidate.

## Data and success

The [data manifest](../config/experiments/assistant-workflow-data.json) commits
256 training, 64 integration, 24 development and 96 confirmation episodes.
Projects and document IDs are disjoint. Development and confirmation use different
conjunctions of corrections; they still share the declared workflow grammar.
Scoring goals are public/reconstructable commitments, not secret security inputs.
Only tools receive public workspace records; the neural responder receives the
conversation, tool definitions and results of tools it actually calls.

Development requires at least 18/24 complete episodes, including 12 compound
conversations, net +4 against each control, no lost control successes, and every
protected anchor. Confirmation opens only after the complete development pass:
at least 77/96, eight successes per family, net +10 against each control, no lost
control successes, and a strictly positive paired lower bound bootstrapped over
eight operation families. This small family count limits what the uncertainty
estimate says. Latency must be at most 180 seconds p95 and twice the trained
no-growth control. The [JSON plan](../config/experiments/assistant-workflow-learning.json)
is authoritative for numerical settings. A failure rejects the candidate.

## First execution: parent baseline only

The implemented [runner](../src/neuroshard/evolution/assistant_workflow_baseline.py)
accepts **development only**. It first repeats the original 24 assistant anchors,
requiring the prior 18 successes. It then executes all 24 new workflows. Baseline
qualification requires at least 6/8 primitive successes, at least one per primitive
family, p95 within 180 seconds and peak RSS within 64 GiB. Save every successful
workflow as protected before training. If the parent exceeds 20/24, the declared
+4 development comparison has insufficient room; publish that fact instead of
weakening the parent or silently changing the gate.

The common [tool environment](../src/neuroshard/evolution/assistant_workspace.py)
executes only five declared in-memory operations. It has no network or filesystem
actions and does not receive expected results. Wrong but valid drafts can be
saved and must score wrong. Read-source citations, strict JSON, budgets, actual
tool events and intermediate draft states are checked independently from saved
model responses. A round passes only after both the draft and a completed reply.
A later correction cannot hide failure on the first turn.

One disposable CPU host, no GPU: r7i.4xlarge, two-hour expiry, $6 planning allowance
including storage/transfer headroom. Exact-commit CI must pass before allocation.
Primary worker: 70 minutes; conditional two-case process replay: 10 minutes.
No automatic retry. The controller copies bounded evidence and retires the host,
volume and temporary security group on completion or failure. The protected
network hosts remain outside this allocation.

```bash
PYTHONPATH=src venv_build/bin/python scripts/modular_reference_cloud.py run \
  --profile assistant-workflow-baseline \
  --home .neuroshard/assistant-workflow-20260927
```

Use the recorded background service for the scheduled attempt; do not start a
second copy of this command. The source must already be committed and pushed.
The [execution inventory](../config/experiments/assistant-workflow-execution.json)
pins contracts, source and runtime. The [preflight](../config/experiments/assistant-workflow-preflight.json)
checks template/execution compatibility without pretrained model generations.
Neither baseline qualification nor this harness completes A2 or proves sharding.
