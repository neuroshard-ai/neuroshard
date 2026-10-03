# Block expert: competence before selection

This is a separate 135M CPU learning experiment. The staged-answering recovery
remains [completed and failed](STAGED_ANSWERING_RECOVERY_RESULTS.md). No result,
method or gate from that study is changed. No router is trained here.

The question is whether newly added Transformer blocks learn additional unseen
answers before attempting to select between experts. This is a competence
experiment on generated arithmetic, not evidence of a useful public assistant.

## Method and comparison

The expert appends two copies of the parent's final two Transformer blocks.
Each copy's attention output and MLP down-projection matrices start at zero,
making the added residual blocks identity functions. All parent parameters,
including embeddings and the output head, stay frozen. Both attention and MLP
parameters of the added blocks train. Initial parent logits must be reproduced
exactly before optimization.

This uses the identity-block principle from
[LLaMA Pro](https://arxiv.org/abs/2401.02415) and its
[authors' implementation](https://github.com/TencentARC/LLaMA-Pro/blob/main/scripts/block_expansion.py).
The experiment appends two blocks instead of reproducing that paper's
interleaved large-model recipe. It makes no claim to invent block expansion.
Identity initialization protects initial outputs; it does not guarantee that
the trained expert preserves answers.

The unchanged-capacity control trains the original final two blocks on the
same ordered examples, optimizer, learning rate and answer-token objective.
It stops after the first complete optimization step meeting the **entire expert
training child process CPU bill**, including setup, cache loading and saving.
The measured extra prefix computation needed only by the expert is added to
that control budget as well.
There is no quality-based checkpoint selection or stopping.

## Data and execution

The [plan](../config/experiments/block-expert.json) and
[data](../config/experiments/block-expert-data.json) define 768 modular-addition
training questions, 384 ordinary-addition practice questions, 64 new-answer
development questions and 64 retention questions. Unordered operand pairs are
unique across all roles and exclude earlier staged-integration, staged-answering
and calibration pairs. Operands range from 0 to 63. Four prompt wordings per
family appear in both training and evaluation; held-out values are the split,
not unseen wording. No MBPP data or original programming final is read.

Training uses float32, one CPU thread, batches of eight, 576 updates (four
passes), AdamW at 0.0003 with 16 warmup steps, weight decay 0.01 and gradient
clipping at 1.0. Only assistant targets including EOS contribute to loss.
Evaluation uses complete greedy answers, up to 32 generated tokens, and the
unchanged strict arithmetic-answer parser. No arithmetic tools or answer repair
are available to the model.

Both training arms cache their frozen prefix activations, computed from training
examples only. Cached and full forward logits are compared before optimization;
unit tests also compare losses, gradients and updates. Output-head projection
is restricted to supervised causal positions, with the same masked loss. The
shared cache preparation is charged once to the complete lifecycle and is not
hidden in either arm's cost. The final two frozen prefix blocks needed only by
the expert are timed separately and credited to the control's optimization
budget; only the common preparation cost cancels. Evaluation runs the complete model without this
training cache. Each evaluation has an isolated peak-memory receipt.

The overall wall cap is two hours. Individual CPU/wall caps are 15 minutes for
baseline and each evaluation, 20 minutes for cache preparation, 30 minutes for
expert training, and 40 minutes for control training. The overall cap takes
precedence. Peak RSS must stay under 4 GiB. There is one attempt and no automatic
retry; progress, checkpoints and interrupted process costs are retained.

## Decision rule

The baseline must have at least eight correct retention answers before any
training. The expert must answer at least 24/64 new questions correctly and
beat **both** parent and control by at least four answers, with positive lower
95% paired-bootstrap gain bounds. Expert p95 latency must be within 1.5 times
control and ten seconds; isolated evaluation RSS must be within 1.5 times
control and 4 GiB. Frozen parent bytes and matched training spend must verify.

Every lost and gained retention answer is reported individually. Retention
losses do not get repaired by an oracle or hidden behind total accuracy. This
expert-only competence gate does **not** require preservation by the forced
expert: a pass permits review of a separate selector contract, not serving or
promotion. Automatic serving would still have to preserve every protected answer
and satisfy its own frozen gates. Selector training is not authorized here.

A miss stops this candidate. A timeout records an incomplete execution, without
a quality conclusion. No GPU, 1.7B run, new final, native promotion, NEURO
issuance or checklist credit follows. Independent hosting remains item 4.

The [execution freeze](../config/experiments/block-expert-freeze.json) includes
the plan, generated data, numerical sources, scorer, tests and this document.
All must be committed before the runner accepts `--run`:

```bash
PYTHONPATH=src venv_build/bin/python scripts/run_block_expert.py \
  --run --seed .neuroshard/seed-smollm2-135m --home .neuroshard/block-expert-NEW
```
