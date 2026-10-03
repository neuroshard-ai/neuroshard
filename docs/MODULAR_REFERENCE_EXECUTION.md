# A1 execution amendment — September 26, 2026

This amends execution of the BAR reference frozen at `25b3976`. It does not
change the nine questions, model revisions, training policy, decoder, token
limit, scoring or quality gates in
[`modular-reference-a1.json`](../config/experiments/modular-reference-a1.json).
No GPU, instance deployment, training, promotion or checklist credit is authorized
by this amendment. A1 remains open.

The original runner accepted old replies after prompt changes, checked memory
only after generation, allowed twenty additional minutes outside the task budget,
and did not implement its required independent replay. These are execution
defects. The original outputs remain diagnostics under their original commit;
they cannot supply primary or replay evidence for this amended run.

## Frozen inputs and receipts

- The execution amendment pins the unchanged plan and a complete artifact
  inventory. The inventory comes from the exact published Hugging Face commits.
  It includes all weight shards, the weight index, tokenizer vocabulary/merges,
  tokenizer configuration, special tokens, generation configuration and template.
  LFS objects use SHA-256; ordinary files use their Git blob SHA-1.
- Before execution, the runner compares every declared source file against its
  committed Git bytes. It records the commit, source hashes, dependency versions,
  Python version and CPU features. Dirty or incompatible execution is rejected.
  Package initialization is included: the existing ATen-default/MKL-SSE4.2
  numerical profile is installed before importing PyTorch in each fresh worker.
- Preparation verifies every artifact's size and content digest. File identity,
  size and nanosecond modification/change times must remain unchanged around
  every generation. This is protection against accidental mutation in a trusted
  experiment environment, not a Byzantine verification scheme.
- Each reply binds that execution identity, model, phase and complete task hash.
  A matching task ID alone cannot reuse an answer. Reply and launch receipts are
  written once; a completed run can be inspected again without generating text.
  A partial, failed or interrupted run requires an explicit execution amendment
  before repeating work. It is never automatically restarted.

## Execution and resource accounting

The runner uses Linux systemd user services with `MemoryMax=12 GiB`,
`MemorySwapMax=0`, `RuntimeMaxSec`, `TimeoutStopSec=0` and `KillMode=control-group`.
The kernel memory cap includes memory charged to the worker's cgroup, including
charged file cache. Shared pages can be charged elsewhere, so the recorded peak
RSS must also satisfy the original limit. Timer expiry terminates the
worker and descendants even if the controller has disappeared. An external kill,
timeout, budget exhaustion or incomplete receipt stops the study; it is never
scored as an ordinary wrong answer.

Limits remain 40 minutes per reply, 8 hours per checkpoint evaluation and 3 hours
per preparation. The 8-hour allowance now explicitly includes **both primary and
replay generations plus the original baseline's recorded evaluation time**.
Preparation includes download and full artifact hashing. Completed and failed
launches are charged their wall time including startup and cleanup. If the
controller dies without recording an outcome, the entire worker allowance is
reserved as spent. No missing outcome is charged zero. CPU time on an externally
killed worker and the original unrecorded download time are reported as unknown.

The original baseline must have finished and written its summary before the new
run starts. That summary is copied and hashed for historical cost accounting,
not reused for answer scoring. The old study directory is preserved. Cached
checkpoint files can be reused only after passing the new full hash checks.

## Required independent replay

For each checkpoint, the runner starts a fresh process for each primary question,
then a separate fresh process for each replay question. Both phases use the same
fixed inputs and runtime. Decoded text, termination and scoring must match exactly;
token equality is also recorded. Replay does not copy or rescore the primary
reply. Missing or disagreeing replay prevents readiness.

A failed baseline usability gate stops before downloading the modular checkpoint.
Otherwise the complete modular primary/replay comparison follows. A completed
comparison still requires review of deviations and a feasible placement estimate
before A1 can be checked off. The runner never marks the milestone complete.

## Running and checking status

Run from an isolated checkout of the committed amendment with the pinned Python
environment. Supply absolute paths; the new study home must not be the old one.
The Linux user systemd manager must be available. The command performs preparation,
primary inference, independent replay and scoring in sequence:

```bash
PYTHONPATH=src python scripts/run_modular_reference.py run \
  --home /path/to/new-a1-execution \
  --models /path/to/original/.neuroshard/modular-reference \
  --legacy /path/to/original/.neuroshard/modular-reference/baseline-result.json
```

Use a persistent service for the controller when disconnecting. The maximum
worker allocation is bounded by two 3-hour preparations and two 8-hour evaluation
allowances, with the historical baseline time deducted. Actual use may be much
lower. This is a CPU reference reproduction, not a serving-latency measurement.

```bash
PYTHONPATH=src python scripts/run_modular_reference.py status \
  --home /path/to/new-a1-execution
```

`status.json` identifies the active checkpoint, phase, task and worker deadline.
`attempts/` contains immutable requests, logs, replies and outcomes.
`progress.json` contains completed comparisons; `result.json` contains the final
decision and accounting. A stopped run has an explicit error and no admission
credit. There is no need for continuous interactive polling.
