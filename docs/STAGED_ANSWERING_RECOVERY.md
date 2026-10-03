# Staged answering: execution recovery amendment

This amendment recovers the [timed-out CPU study](STAGED_ANSWERING_RESULTS.md).
It does not resume that directory under the old freeze. The original method,
data, optimizer schedule, serving rule, protected answers, and quality gates
remain fixed. No GPU, 1.7B model, admission, or checklist credit is authorized.
Item 4 remains the separate trial requiring four independent operators.

## Bound inputs and recovery

The [amendment](../config/experiments/staged-answering-recovery.json) pins the
original method contract `ac66cb34…`, source freeze `66e3a9f4…`, timeout record,
all input receipts, all 15 protected answers, and both saved expert files.
The expert boundary is step 64, tensor root `c39608ac…`; the weights hash is
`04249508…` and optimizer/RNG file hash is `722f085d…`.

Execution requires committed sources matching the [new inventory](../config/experiments/staged-answering-recovery-freeze.json).
It creates a new study directory and copies/hash-verifies the recovery inputs.
The old study is read-only. Its verified baseline is reused, with every
protected identity unchanged. No new examples are selected or scored first.

Restore the saved expert and verify the incumbent is identical to the seed.
Require the saved router weights and bias to be exactly zero. Reconstruct the
fixed training-only before/after probes for diagnosis, then restore the saved
Torch and Python RNG states so these probes cannot alter gate training.
Use a fresh gate AdamW optimizer with the original learning rate and schedule.
Do not load the expert optimizer into the gate.

Repeat all 64 gate steps in the original batch order. Before evaluation, require
the first 63 steps to reproduce the interrupted run's recorded losses, gradient
norms, token counts, and routing statistics exactly. Timing is excluded from
that comparison. A discrepancy stops the attempt. Save the completed gate
checkpoint, then run the control and both automatic serving arms in fresh
processes. Only the existing complete-system scorer determines the result.

## Costs and limits

The control's optimization budget is the **entire failed expansion process CPU
bill (598.805385 seconds) plus the entire gate recovery process CPU bill**.
The latter is measured by the parent after the child exits, including imports,
restoration, probes, serialization, and exit. The control uses the original
ordered stream, optimizer, 512-step maximum, and first-completed-step-at-budget
stop rule. Its extra training compensates for retry expenditure; there is no
quality-dependent stopping.

Report the useful 64 expert updates, discarded 63 recorded gate updates, repeated
64 gate updates, reconstruction probes, and remaining process CPU separately.
The interrupted final update cannot be separated from the old process remainder;
that entire remainder is charged. Baseline CPU is counted once in total study
expenditure, outside the training comparison. Control/evaluation child costs and
recovery coordinator CPU are also included in total expenditure. Earlier
calibration and development/CI costs are outside this study's accounting scope.

This comparison bills more than the original optimization-plus-probe rule;
that is the explicit execution-cost amendment. No loss, answer, retention,
latency, memory, or model-selection threshold is relaxed.

| Arm | Wall/CPU limit |
| --- | ---: |
| Restart gate, including restoration/probes/checkpoint | 600 seconds |
| Matched-cost control, including checkpoint | 1,500 seconds |
| Automatic expansion evaluation | 600 seconds |
| Automatic control evaluation | 600 seconds |

The new execution has a 3,600-second overall wall cap, one CPU thread, and the
original 4 GiB acceptance bound. The old run's 768.82 seconds remain recorded
separately; its costs are not erased by the new clock. The expected gate workload
is approximately 300 seconds based on the interrupted trace. The control needs
extra allowance because it must also cover the discarded work. There is one
attempt, no automatic retry, and evidence is retained after failure.

## Validation and execution

A tiny CPU model with dropout tests restored RNG state and fresh optimizer
semantics: recovered gate tensors must exactly match an uninterrupted run.
Other checks reject changed sources, changed prefix values, omitted retry costs,
and invalid recovery identities. These fixtures are not model-quality evidence.

```bash
PYTHONPATH=src venv_build/bin/python scripts/run_staged_recovery.py
PYTHONPATH=src venv_build/bin/python scripts/run_staged_recovery.py \
  --run --seed .neuroshard/seed-smollm2-135m \
  --previous .neuroshard/staged-answering-cpu-20260921T192003Z \
  --home NEW_RECOVERY_DIRECTORY
```

Commit the amendment and its complete execution source before the real run.
Record the outcome separately; this freeze is not edited after generation.
