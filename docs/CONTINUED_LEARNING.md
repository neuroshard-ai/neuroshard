# Continued learning from the passing phase-A checkpoint

**Status: plan frozen, 14 September 2026.** This is a development experiment. It does not change the public 0.4.0 chain, mint NEURO, or promote a serving model. Machine-readable constants live in [`continued-learning.json`](../config/experiments/continued-learning.json). Changing a number is a new plan revision.

The [adaptive shard trial](ADAPTIVE_SHARDS_RESULTS.md) left one passing 1.7B candidate: the three-worker phase-A checkpoint, with Adam, at step 128. Both later continuation recipes lowered teacher-forced response loss and then **lost greedy invoice totals**. Those exposed finals cannot be reused. This plan is the next learning objective. It does not add model growth or a new architecture.

## What this plan freezes

1. **Actual artifacts, not only evaluation IDs.** Preparation commits content hashes for the new datasets, the parent portable checkpoint (weights **and** optimizer state), the frozen reference model, replay window identities, tokenizer files and identity, architecture config, and the recorded CUDA 12.8 / A10G runtime. Fresh evaluation examples must not overlap this run's training data or the previously exposed adaptive-shard finals. The training driver refuses to start if those hashes disagree.

2. **Generated-answer improvement is the primary acceptance criterion.** The minimum net gain, exact one-sided McNemar test and per-family retention floors are predeclared. A gain in sorting cannot hide lost arithmetic. Model size stays 1,711,376,384 parameters and 24 layers. Develop on training and development data, then lock the predetermined step-224 candidate before opening finals. A failed final stays a reported failure.

3. **Settlement after a pass is development order, not payment for this file.** Once the method passes, implement job activation, then reproduce the accepted updates through an activated, funded job with `reserve_shards`, reserved windows and signed worker receipts. Current reservations already bind those receipts to the training windows ([native shard replay](NATIVE_SHARD_REPLAY.md#ledger-behavior)). An existing checkpoint is not sufficient for payment.

Verified computation earns the agreed training reward. Verified quality authorizes serving. A later quality transaction must bind an **already-settled** checkpoint to this frozen evaluator, the frozen evaluation data and the validated results before changing `serving_root`. Promotion mints nothing. Correctly executed work must not lose its promised payment merely because the quality gate fails.

## Why this recipe

Phase B used the same 3e-6 learning rate and 128-step budget as phase A, with KL toward the phase-A teacher. Loss and conversation retention passed; greedy `{"total": ...}` answers did not. This continuation therefore:

- starts from the passing portable checkpoint and **keeps its Adam moments**;
- does not grow depth or change shard boundaries `[0, 6, 15, 24]`;
- uses a lower peak learning rate (`1e-6`) for 96 additional updates (global cursor 128→224);
- upweights the invoice-total family in new training data;
- scores **all** 256 new-task cases for generation, not a 64-example prefix;
- treats generated-answer gain as the pass/fail rule, with conversation retention as a hard floor and response loss as a published measurement only.

This is one pre-registered hypothesis. It is not a claim that the previous failure was only a learning-rate bug.

## Quality gate

Compare the locked candidate to the frozen parent on unused data.

| Check | Rule | Role |
| --- | --- | --- |
| Primary | Net generated wins − losses ≥ 8 **and** exact one-sided McNemar *p* < 0.05 | `test-new` (256 cases, 64 per family) |
| Per-family floor | Candidate correct ≥ parent correct for lookup, filter, total and sort | `test-new` and `test-prior` |
| Conversation retention | Paired bootstrap 95% upper bound on loss change ≤ +0.02 nats | `retention` (128 SmolTalk documents from a new cursor) |
| Response loss | Published, not an acceptance criterion | all scored roles |
| Development abort | Mean retention-loss change > +0.05 nats | `dev-retention` only; cannot retune or pick another endpoint |

`test-prior` is a new 128-example probe of the same four families. It is not the old `test-a` / `test-b` sets. Those identities, and the previous retention documents from SmolTalk test rows 14000–18999, are excluded.

If the primary generation test fails, the run fails even when loss improves. If totals drop while sorting rises, the run fails.

## Freeze, then train

Do not train in the plan-frozen status.

1. Keep public 0.4.0 unchanged.
2. Run `scripts/prepare_continued_learning.py` with the pinned tokenizer files, `--prior-train` pointing at the recorded `train-a.jsonl`, all eight prior role files beside it, and `--plan-commit` identifying the committed frozen plan. This preprocessing needs no GPU or model weights. It excludes normalized prompt/document content from all previously exposed roles. Only the explicitly sampled, previously trained replay may reuse prior training identities.
3. Copy the generated `prepared.json` to `config/experiments/continued-learning-prepared.json` and commit it without changing this plan's constants. Advance status to `prepared-committed`. Keep the prepared JSON and its role files together outside Git for worker execution; the supplied JSON must match the committed freeze.
4. Train only through `scripts/run_sharded_training.py continued`. Fresh and resumed runs check the same parent/reference, tokenizer, config, runtime, committed sources and prepared bytes before allocating model shards. The driver evaluates development retention at every declared checkpoint and before continuing a resumed checkpoint. A failed development gate stops execution and writes `aborted.json`.
5. After step 224 passes development retention, commit `config/experiments/continued-learning-selection.json` binding the actual candidate manifest identity, frozen plan/evaluator and its `development-000224.json` decision. Set status to `selection-committed`. The selection's `development` field contains that complete decision. Final evaluation requires the supplied selection to match the committed file and the actual candidate to be step 224 of the frozen job.
6. Score using `scripts/score_continued_learning.py` with the committed plan/prepared/selection, `--baseline` and `--candidate` evaluation reports, the actual candidate `--checkpoint`, and the pinned tokenizer `--seed`. Scoring validates the endpoint and decoded output tokens before applying the frozen gates. It refuses an existing output path so previous decisions remain intact.

Intermediate checkpoints at 160 and 192 exist for abort and recovery. They are not selectable finals.

Development evaluation also generates answers for `dev-new` and `dev-prior`, so diagnostics can inspect the primary behavior without opening the final roles. These diagnostics cannot change the frozen recipe or select another endpoint.

## Settlement order, after a pass

The current portable-work genesis pins one prepared job, one parent checkpoint, an optional `reference_root` and a step ceiling. `claim_shards` rejects a changed job, config, layout or any non-null transition. A new dataset therefore needs an **explicit activation** that rebinds prepared inputs, the phase-A reference and the starting checkpoint to a new job identity. Copying `three128` onto a ledger is not that activation.

After a quality pass, and not before:

1. Specify activation: new isolated genesis or a dedicated transaction that installs the accepted prepared hash, parent checkpoint and reference.
2. Reproduce the 96 updates as reserved windows (`max_window_steps` remains 1–4) with signed per-rank receipts.
3. Settle those windows. That payment is for verified computation. It does not set `serving_root`.
4. A separate quality transaction may then bind the settled checkpoint to this evaluator and, if the gate still holds, set `serving_root` with mint 0.

Until that protocol exists, a passing research checkpoint is evidence about the method, not a payable native artifact.

## Hardware

Three `g5.2xlarge` A10G workers in one availability zone, matching the parent numerical profile. Tear the instances down after the run. Do not keep GPUs idle. Each worker downloads its own share of the 20.54 GB parent inventory; keep full model/optimizer artifacts off the primary CPU disk.

## Out of scope

Growth, a second recipe on these finals, public-network changes, promotion without a settled checkpoint, and treating lower response loss as success. Independent operators, economical verification and a useful general assistant remain separate claims.
