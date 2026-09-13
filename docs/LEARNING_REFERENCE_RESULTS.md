# 1.7B GPU learning reference: September 13, 2026

**Full-model training and checkpoint recovery worked. This run does not establish a better general assistant.** A single NVIDIA L40S trained all 1,711,376,384 parameters of SmolLM2-1.7B-Instruct. The final-test mean response loss fell slightly, but its approximate paired 99% interval includes zero. Generated answers show improvements and regressions. The candidate is not promoted to serving.

The [complete evidence](https://neuroshard.com/experiments/learning-reference-gpu-20260913/README.md) includes [every measured answer](https://neuroshard.com/experiments/learning-reference-gpu-20260913/generations.html), raw scores, the step journal, recovery checks, runtime details, checksums and downloadable model weights. This is an operated development reference. It issued no NEURO and changed no native ledger state. The [earlier 135M contract](LEARNING_MILESTONE_RESULTS.md) remains rejected.

## Frozen inputs and execution

The [plan](../config/experiments/learning-reference.json) and [prepared selection](../config/experiments/learning-reference-data-selection.json) specify 2,048 complete training conversations, 256 AdamW updates and 655,372 assistant/EOS targets. Contexts contain at most 1,024 tokens. Parameters, gradients and Adam moments are FP32; CUDA forward/backward uses BF16 autocast. The [reference guide](LEARNING_REFERENCE.md) provides the commands and pinned dependencies.

Training ran at `c1499edf1c57ebe65ed7395aba0c765c748b77a6`; the bound numerical source is unchanged from preparation at `153cec2`. The [exact final candidate](../config/experiments/learning-reference-selection.json) was committed at `2dd09bc` before opening the final test. Candidate selection was the predetermined step 256, with no selection among intermediate checkpoints. Its receipt is `002e27fb415e74b048c39ff5babd226cfd355e58ff6a446e609cbba8c252934a`.

The seed already had related SmolTalk training. These partitions are unused by this experiment, not demonstrated unseen during upstream training. Short, complete conversations are overrepresented by the length filter. This run does not demonstrate fresh factual learning, continual learning, distributed speedup or model growth. Moving to a larger upstream seed is not growth produced by NeuroShard.

## Loss and generated answers

Each document contributes one mean assistant-response loss, in nats per target token. Changes are candidate minus seed. Intervals use the paired document standard error and multiplier 2.576; they are approximate descriptive intervals for this one run, not uncertainty across training seeds or a general capability certificate.

| Partition | Documents | Seed | Candidate | Change | Approximate 99% interval |
| --- | ---: | ---: | ---: | ---: | --- |
| Development | 128 | 0.580172 | 0.572142 | −0.008030 | [−0.015449, −0.000612] |
| Retention | 128 | 0.544322 | 0.540342 | −0.003980 | [−0.012793, +0.004833] |
| Final test | 256 | 0.536945 | 0.533697 | −0.003248 | [−0.010635, +0.004139] |

The report independently recomputes these statistics from individual scores, checks target counts and document identities, and retains token-weighted means separately. The original `result.json` records the development phase with `test_scored=false`; the later test is recorded separately in `test-result.json`. Neither file is rewritten to conceal sequencing.

All 20 development and 20 test generation pairs were reviewed under the [rubric committed before training](../config/experiments/learning-reference-answer-review.json). This is an unblinded development-agent review with no independent human validation. Preferences are qualitative judgments, not benchmark pass rates.

| Partition | Candidate preferred | Seed preferred | Tie | Neither | Unresolved |
| --- | ---: | ---: | ---: | ---: | ---: |
| Development | 1 | 5 | 11 | 1 | 2 |
| Final test | 4 | 2 | 9 | 5 | 0 |

On the final test, the candidate improves the deductive-reasoning example but regresses on a Pandas conversion: it returns date objects when the request asks for strings. Both versions produce faulty feature-extraction functions. Simple executable counterexamples and relevant primary documentation accompany the judgments. Token-budget exhaustion falls from 8 to 5 test answers, while remaining 2 versus 2 on development. Finishing more often does not establish correctness.

## Recovery, memory and cost

| Measurement | Observed value |
| --- | --- |
| Host | One g6e.2xlarge: NVIDIA L40S, 8 vCPU, 64 GiB RAM, 200 GiB encrypted gp3 |
| Training/development driver | 1,257.38 seconds; complete process 21m 15.14s |
| Sum of the 256 training-step timers | 437.94 seconds; median 1.69 seconds |
| Separate final-test process | 7m 35.95s, including cold seed-file verification |
| Peak allocated Torch CUDA memory | 31,598,901,248 bytes (29.43 GiB) |
| Peak retained training artifacts | 82,184,074,411 bytes (76.54 GiB) |
| Same-host replay | Restore step 192, repeat 64 updates; 170.19 seconds |
| Replay result | Identical numerical journal fields and all 218 final parameter tensors |
| Public candidate model archive | 6,376,971,282 bytes; exact FP32 weights and tokenizer |

The replay restores the full Adam/RNG state and compares every final parameter bit for bit against the original candidate. A separate small CUDA preflight also reproduces the next loss and weights after restoration. These are same-host recovery checks, not cross-GPU reproducibility or an independent operator trial. Replaying 64 updates is not a cheap proof of a malicious worker's computation.

The GPU rate was $2.24208/hour. The experiment used a $100 ceiling and a 12-hour automatic stop deadline; the instance was terminated after copying and hash-verifying its checkpoint and evidence. AWS confirmed termination and root-volume deletion. The observed lifetime gives a compute upper estimate of $1.94; the public resource receipt retains the exact calculation. Storage, transfer and taxes are separate; the estimate is not an AWS invoice. Training-step time alone omits setup, evaluation, checkpoint I/O and idle review time.

The model archive omits optimizer/RNG state and intermediate checkpoints. The complete final checkpoint is retained by the operator; reproducing training state requires that checkpoint or rerunning the pinned recipe. The public weights are research artifacts, not an approved serving release.

## Data audit and next experiment

Review found a development reference answer that uses “happy” once despite requiring it three times. Loss against a flawed reference can reward the wrong behavior. The new [instruction-target audit](../scripts/audit_instruction_targets.py) checks a narrow set of explicit English templates without changing records or admitting them to the ledger:

```bash
PYTHONPATH=src python3 scripts/audit_instruction_targets.py \
  .neuroshard/reference-gpu/inputs/train.jsonl --output .neuroshard/train-target-audit.json
```

After correcting placeholder handling, it checks 91 constraints in 81 of the 2,048 training records and flags four records: a bullet-count mismatch, a strict quoted-ending mismatch and two missing exact keywords. It checks 14 constraints in 11 development records and flags the repeated-word failure. The remaining records are unassessed by these templates. Four initial false flags came from interpreting `[keywords]` as a literal; the corrected code, regression test and original audit are retained in the evidence. These counts are parser findings, not four proven causes of model regression or a general corpus-quality estimate. Inflections, quoted text, instruction conflicts and exact-ending conventions still need contextual review.

The next learning experiment should use verified targets and independently specified executable tasks, while retaining broad conversation and reasoning probes. Freeze a new data cohort and quality contract before training; this test is now exposed and cannot support another confirmatory claim. Compare training on reviewed targets against the unchanged seed and unreviewed-data baseline at a fixed budget. Only a useful recipe should proceed to multi-peer execution, funded verification and repeated fresh-data cohorts. Additional GPUs alone cannot resolve incorrect targets or establish useful learning.
