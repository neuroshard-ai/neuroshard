# Learning milestone — frozen experiment contract

**Status: selection committed, 12 September 2026.** The plan was frozen on 11 September. The [sealed manifest](../config/experiments/learning-milestone-selection.json) contains all 448 documents, 128 training batches and 20 generation probes. Training requires those exact Git-committed bytes and the matching implementation. No learning outcome is claimed by this status.

The intended product remains a collectively trained, openly retrievable language model whose usable capacity can grow as reliable compute joins. The [scaling design](SCALING_DESIGN.md) still requires economical verification, funded honest audits and independent operators. Those are separate claims. This milestone answers three narrower questions, in order:

1. **Useful learning.** Can one pre-registered recipe produce a 135M checkpoint that improves on untouched evaluation data inside a declared budget?
2. **Continual learning.** If so, can a second data cohort add a further gain without giving back the first?
3. **Useful scaling.** What does a second physical machine actually buy—at this size, reliability with measured overhead—not implied capacity or throughput?

Machine-readable constants live in [`config/experiments/learning-milestone.json`](../config/experiments/learning-milestone.json). Changing a number is a new plan revision. It cannot be done on a run that already saw the sealed set.

## Why this is public

NeuroShard is Apache-2.0. The seed ([SmolLM2-135M-Instruct](https://huggingface.co/HuggingFaceTB/SmolLM2-135M-Instruct), revision `12fd25f77366fa6b3b4b768ec3050bf629380bac`) and the corpus ([Smol-SmolTalk](https://huggingface.co/datasets/HuggingFaceTB/smol-smoltalk/tree/f73fe857d519ff6ac5af2ea67c4d3834da7b8bcc), revision `f73fe857d519ff6ac5af2ea67c4d3834da7b8bcc`) are Apache-2.0. There is no private evaluation set.

**Sealed** means the document identities are chosen and committed to git *before* the first training step. Anyone with this repository and the pinned sources can reconstruct the same windows. It does not mean a hidden holdout. After the decision, the scored losses, generation transcripts and checkpoint roots are published; large tensors stay out of git, as with other experiment artifacts.

Untouched means unused by this NeuroShard experiment. The seed's published model card describes prior Smol-SmolTalk fine-tuning; this experiment cannot establish that its documents were unseen during the seed's training. Its paired comparison measures additional learning relative to that same seed.

Independent reproduction and refutation are in scope. A passing local run is not a protocol security result.

## What earlier runs already showed

| Run | Recipe | Fresh change (UCB) | Outcome |
| --- | --- | --- | --- |
| Response-from-seed, 32 steps, lr 0.003 | 135M, no growth | about −0.0014 nats (UCB above −0.001) | reject |
| Exploratory 32 steps, lr 0.03 | 135M, no growth | loss increased | reject |
| Native lifecycle, 4 steps, then 4 identity blocks | 148.7M candidate | about −0.0012 nats (UCB −0.0003) | reject |

Those are execution successes and quality failures. This plan does not relax `fresh_min_gain = 0.001` or `retention_margin = 0.02`. It does not grow the model. It uses the learning rate that at least moved in the right direction, with four times the 32-step budget and 64 documents per role so the interval can actually close.

If that still fails, the milestone fails. A follow-up needs a **new** sealed set. Reusing this test set after looking at it is forbidden.

## Phase 0 — freeze, then select

Do not train in this phase.

1. Keep the public 0.4.0 chain on its released profile.
2. Import the pinned seed and bind the [text protocol](TEXT_PROTOCOL.md). The imported tensor root must be `978db595a79e1bb381fd9723702b21682af10c68ce872d5e3f4d9e4429d0e22e` before tokenizer metadata is attached.
3. Collect from unused source rows:

| Pool | Split | Cursor start | Scan limit | Keep |
| --- | --- | --- | --- | --- |
| Train | `train` | 8192 | 2048 | 256 complete training documents |
| Held-out | `test` | 3072 | 2048 | 64 documents each in `retention`, `fresh`, `test` |

Held-out roles follow the existing `document_id mod 3` partition. Incomplete evaluation documents are rejected, not truncated into extra observations. If the scan limit cannot fill 64 documents per role, **prepare fails**; do not reduce `n`.

4. Commit `config/experiments/learning-milestone-selection.json` with document ids, source rows, window roots, the 20 generation prompts (the last user turn of the first 20 sealed `test` documents in sorted id order), and a digest of this plan file. Set plan status to `selection-committed` only after that file is in git.
5. `python -c "from neuroshard.evolution.milestone import load, training_allowed; p=load(); assert training_allowed(p)"` must then succeed.

Training against an uncommitted or altered selection is a protocol violation of this milestone, not a near miss.

## Phase 1 — useful learning (blocker)

One host, three worker processes, 48M declared capacity each, **no second machine**, **no growth**, **one recipe**.

| Item | Frozen value |
| --- | --- |
| Steps | 128 SGD steps, 2 response windows per step |
| Replay | none (no prior trained cohort) |
| Optimizer | SGD, learning rate 0.003, global clip 1.0 |
| Wall clock | 72 hours including evaluation, retries and failed steps |
| Disk | 256 GiB experiment home under `.neuroshard/` |
| Pass rule | sealed `test` UCB below −0.001 nats **and** `retention` UCB below +0.02 nats |

`fresh` is scored and published. It cannot rescue a failing sealed test. Generations (greedy, 32 tokens, 20 sealed prompts) are published next to the parent; they are not a substitute for the loss gate. A loss win with clearly worse answers is reported as such.

The parent is the bound 135M seed. An equal-cost comparison against that parent is required. An optional single-process companion run may be recorded after the pass/fail; it must not be used to pick hyperparameters.

Budget exhaustion, disk exhaustion, or a crash that cannot resume from the coordinator journal is a **fail**, not an invitation to change the recipe.

## Phase 2 — continual learning

Blocked on a phase-1 pass. The promoted checkpoint is the only allowed parent.

The phase-1 sealed `test` documents remain frozen and are **not** trained on. Cohort 2 collects the next unused train and held-out ranges recorded in the selection/result files. Replay is 25% of the schedule, taken from **windows that phase 1 actually trained**, not from unused admitted documents.

Pass:

- New sealed `test` vs the phase-1 parent: UCB below −0.001 nats.
- Phase-1 sealed `test` vs the phase-1 parent: UCB below +0.02 nats.

This is stability across two unused SmolTalk slices with rehearsal of trained windows. It is not domain-incremental learning and not a lifelong-memory claim. A second licensed source with the same tokenizer would be a stronger follow-up, not this plan.

## Phase 3 — useful scaling

Blocked on a phase-1 pass. This is not a reason to run two hosts during phase 1.

At 135M on ~16 GiB hosts, a second machine is **not** assumed necessary to hold the model. The primary question is reliability.

| Condition | Layout |
| --- | --- |
| A | Three workers on host 1, 16 sequential steps, same numerical profile and the same pre-committed batches |
| B | The same 16 steps with workers split across host 1 and host 2 |

During B, stop one worker after step 8. Recover from the last accepted journal. The recovered step must reuse the same work identity; a second payment or a second numerical claim for that step is a fail.

Success: B finishes, its model root **matches** A, the kill is recovered, and the log reports wall-clock, object bytes moved, idle time and retries for both conditions. If B is slower, that still counts, provided recovery works: the second machine bought a failure domain.

This plan does not treat a matching two-host root as a throughput win or as capacity growth. Throughput would need a different schedule. Capacity would need a model that cannot reside on one host under a declared memory envelope.

## What this milestone will not do

- Replace or migrate `neuroshard-llm-testnet-1`.
- Pay public NEURO or change issuance.
- Lower the quality margins because a run looks close.
- Treat validator replay, funded audits, compact SGD witnesses, or a second operator as done.
- Count unused-row replay as preserved skill.
- Publish only passing generations.

## Execution and evidence

The [prepare/train/score driver](../scripts/run_learning_milestone.py) enforces the commitment and the fixed recipe. Experiment homes, traces and tensors stay under `.neuroshard/`, not in the source wheel. Compact results (roots, losses, UCBs, decision, generation texts, recovery hashes) belong in a later evidence note, in the same style as [native lifecycle results](NATIVE_LIFECYCLE_RESULTS.md).

Use a full Git clone and a Python environment with `docs/evolution-requirements.txt` plus the pinned collector dependencies in `pyproject.toml`. The driver runs from the checkout. It does not use the installed 0.4.0 client's released runtime. Obtain the seed's pinned files using `python -m neuroshard.evolution download-seed --model-dir ./seed`, as described in the [evolution protocol](EVOLUTION_PROTOCOL.md).

```bash
export PYTHONPATH=src
export ATEN_CPU_CAPABILITY=default
export MKL_ENABLE_INSTRUCTIONS=SSE4_2
python scripts/run_learning_milestone.py prepare \
  --home .neuroshard/learning-milestone \
  --model-dir /path/to/pinned-smollm2-135m
```

Preparation requires a committed implementation and a `plan-frozen` plan. It scans the entire declared held-out range before the training range, uses the existing exact/near-duplicate registry, and selects complete documents in document-id order. Identical input/target windows cannot cross selected roles. Selection priority is `test`, `retention`, `fresh`, then `train`; no loss or generation is computed during preparation.

The 256 training documents each contribute one window, chosen by the lowest SHA-256 of the canonical JSON `[plan_digest, "window", window_root]`. Documents are ordered by the hash of `[plan_digest, "document", document_id]`, then paired into 128 batches. All selected evaluation windows are scored. Document losses are weighted by their response-target counts, giving each document one observation in the confidence interval. Unused training windows are retained as artifacts and are excluded from subsequent trained-window replay.

The 20 generation probes retain the original last user turn. The existing executor accepts at most 192 prompt tokens, so the rendering policy is explicit: apply the pinned chat template, retain its last 192 tokens, and record the original token count and truncation flag. No prompt is replaced because it is long or produces an unfavorable answer. Both models receive identical rendered inputs and generate up to 32 greedy tokens, stopping at EOS.

Commit the generated selection first. Its `plan_digest` hashes the original plan file at the recorded `plan_commit`; its `implementation_digest` binds the Python package, driver and numerical dependency lock. Then change only the plan's status to `selection-committed` and commit that status separately. The guard checks both current files against Git, checks the original plan bytes, and allows only that status transition. A staged file is insufficient. Do not modify the bound implementation during the experiment.

```bash
python -c "from neuroshard.evolution.milestone import load, training_allowed; assert training_allowed(load())"
python scripts/run_learning_milestone.py run --home .neuroshard/learning-milestone
```

For independent reconstruction, use a repository revision containing both the committed selection and status, with the matching implementation, and run `reconstruct` with the same arguments as `prepare` in a new experiment home. The result note records that executable revision; `plan_commit` identifies the earlier frozen plan and does not itself contain the selection. Preserve those commits when merging this work. Reconstruction checks that all document identities, windows, batches and prompt renderings match the committed manifest. An optional `--upstream-cache` reuses the pinned, hash-verified Parquet files; it cannot substitute their bytes.

The three workers have separate durable databases and share one local content-addressed object directory. Training retains every checkpoint. The driver recovers accepted steps from worker receipts even if the coordinator died before writing its outer progress file. Each completed scoring window and generation is persisted. Retryable transport failures reopen those same journals, with at most three recorded transport failures; integrity errors fail the run. Restarting the command preserves the original 72-hour deadline, including downtime. A failed run cannot be reset through this driver.

`run.json` holds progress, `coordinator.json` holds the accepted training position, and `result.json` contains the complete comparison and both generation transcripts. The wall-clock and 256 GiB limits apply to phase 1, including scoring, generation, retries and failed work. A successful command means the experiment completed; read `decision.pass` to determine whether useful learning passed. Phases 2 and 3 remain blocked unless it did.

Reproduce the contract without training:

```bash
python -m pytest -q tests/evolution/test_learning_milestone.py tests/evolution/test_learning_driver.py
```

The public product goal does not change if phase 1 fails. It means the current recipe, at this budget, did not produce a better serving model. That is a publishable result.
