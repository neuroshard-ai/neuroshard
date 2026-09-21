# Staged answering: corrected CPU mechanism study

This is a separate candidate after [the predecessor stopped before training](STAGED_INTEGRATION_RESULTS.md).
It retains staged expert-then-gate training, automatic top-1 serving, the pinned
135M seed, and every quality, preservation, cost, and time gate. No GPU, larger
model, promotion, or checklist completion is authorized.

## What changes

The serving interface uses an explicit arithmetic system instruction and a
32-token generation cap. Scoring requires EOS and accepts either a single
unsigned integer or a complete equation matching the question's operands and
operation. A terminal full stop is permitted. Only the model supplies the
answer value: the parser neither calculates the answer nor searches prose for
the correct number. Training targets remain the numeric answers. Every arm
uses the same interface, scoring, and generation budget.

Two calibration variants scored 6/16 and 7/16 addition cases; both missed their
12/16 screen. Those failures are retained. This contract proceeds only to a
fresh baseline first, using the original 8/32 requirement for training. It does
not assume the new baseline will qualify, or infer learning from calibration.

All 160 successor pairs are distinct from the predecessor's 160 pairs and the
20 calibration pairs. Unordered pairs remain unique across all six roles.
To supply that unused pool, operands range from 0 through 31 instead of 19.
This is a new task sample, not a directly comparable repeat of the old score.
Roles retain 32 new and 16 practice examples in each training phase, and 32
development and 32 retention cases. Hash ordering fixes assignments before
model generation. No examples are selected by whether the parent gets them right.
No MBPP data, earlier confirmation, original 128-task final, or opened 64-case
programming diagnostic is loaded.

## What stays fixed

Train the added last-layer MLP for 64 steps with guaranteed training access.
Freeze it; train the gate for 64 steps through a soft mixture of the frozen
MLPs. Serving executes one hard-selected MLP per token. Forced routing is
training-only. Checkpoints, optimizer/RNG states, training costs, and automatic
route traces are saved by the existing implementation without modifying it.

The no-expansion control follows the same ordered training stream until its
optimization CPU time reaches the expansion's optimization plus probe CPU
time. Record overshoot, tokens, steps, and full process costs. Separate fresh
processes measure each arm's memory and latency.

Before training, save all parent-correct retention identities. If fewer than
eight of 32 are correct, stop without training. Passing later requires:

- At least two more correct development answers than both parent and control.
- Every protected retention answer remains correct; gains do not offset losses.
- The added MLP is used on an automatic gain over the parent.
- p95 latency is at most 1.5 times control and 10 seconds.
- Peak RSS is at most 1.5 times control and 4 GiB.
- The control matches measured training CPU; incumbent weights stay frozen;
  the added expert stays frozen during gate training.

The complete study is capped at 1,800 wall seconds; each worker at 600 wall/CPU
seconds. Outcomes distinguish absent training signal, ineffective automatic
integration, and a complete system evaluated against all gates. A pass remains
synthetic CPU mechanism evidence, not assistant admission or decentralized growth.
Independent operators remain separate.

## Execution

The [plan](../config/experiments/staged-answering.json),
[data](../config/experiments/staged-answering-data.json), and
[source inventory](../config/experiments/staged-answering-freeze.json) must match
committed HEAD. The inventory includes the reused predecessor code and data.
Future outcomes belong in a separate results file; this document is frozen.

```bash
PYTHONPATH=src venv_build/bin/python scripts/run_staged_answering.py
PYTHONPATH=src venv_build/bin/python scripts/run_staged_answering.py \
  --run --seed .neuroshard/seed-smollm2-135m --home NEW_STUDY_DIRECTORY
```

The worker downloads nothing and creates no AWS resources. Existing evidence
is never overwritten. A failed baseline or candidate remains failed.
