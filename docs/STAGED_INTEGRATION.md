# Staged integration: 135M CPU mechanism candidate

Status: a new candidate, not executed. No GPU, 1.7B trial, promotion, or checklist
completion is authorized. The previous learned-integration **development failed;
confirmation was never opened**. Its source, contract, and result remain intact.

This document and the candidate source are immutable parts of the new freeze.
Record future outcomes separately rather than editing this freeze after answers.

## Question and scope

Can a new last-layer MLP learn with guaranteed training access, then become
useful under an independently trained gate and deterministic automatic serving?

This is a CPU mechanism study on pinned SmolLM2-135M-Instruct. It can diagnose
the learning/selection obstruction. It is not evidence of broad coding ability,
general assistant improvement, decentralized execution, or continual admission.
A pass does not authorize a larger model or GPU run.

The method is motivated by expert-then-gate training in
[PHATGOOSE](https://arxiv.org/html/2402.05859v2#A1) and expert-then-routing
integration in [Branch-Train-MiX](https://arxiv.org/abs/2403.07816). This candidate
uses a full MLP copy and a two-output soft gate, not either paper's exact recipe.
Those papers do not establish NeuroShard's retention or networking properties.

## Frozen method

1. Copy the seed's final decoder MLP. Keep the seed, attention, original MLP,
   embeddings, and output head frozen. Initialize a separate linear gate to
   zero; inference ties choose the original MLP.
2. Train only the added MLP for 64 steps, always selecting it on these training
   examples. This guaranteed access is training-only, never a serving result.
3. Freeze both MLPs. Train only the gate for 64 steps. During this phase use a
   differentiable softmax-weighted combination of both outputs. Bill both MLP
   evaluations. Serving selects exactly one MLP, without sampling or a forced
   choice. The difference between soft training and hard serving is an explicit
   possible failure, measured by routing traces and generated answers.
4. Train a no-expansion control on the same ordered source batches. Stop after
   the first complete step whose cumulative optimization CPU time reaches the
   expansion's optimization **plus training-probe** CPU time. Cap at 512 steps;
   inability to reach the budget rejects the comparison. Report step overshoot,
   tokens, wall time, and CPU time. There is no quality-based stopping.

The control follows the same stream, cycling if necessary; timing can result in
a different number of batches. This is an actual CPU-budget comparison, not a
claim of identical token counts, identical steps, or theoretical equal FLOPs.
Timed control stopping is a research procedure, not a consensus transition.

## New data, protected before training

The deterministic arithmetic data were created for this mechanism study.
Ordinary addition supplies baseline practice and retention; addition modulo
seven supplies the candidate skill. Prompts contain the task and operands,
without router labels. Answers are checked by exact numeric text after
stripping surrounding whitespace. No output repair is applied.

All six roles use distinct unordered operand pairs from 0 through 19, including
across task families. Expert and gate training each have 32 new and 16 baseline
examples. Development and retention each have 32 separate examples. Both the
fixed generator and rendered data are hashed. This tests new combinations
inside a small synthetic family, not new natural-language task families.

No MBPP loader is imported. The original 128-task final, opened 64-case
diagnostic, and previous learned-integration confirmation are not read or scored.
There is no new confirmation set in this mechanism study.

Before training, generate the parent's answers and save their complete receipt
and the identities of every correctly answered retention case. At least eight
of the 32 retention cases must be correct. Otherwise stop without training;
do not replace examples or call an empty retention result preservation.
Every protected answer must remain correct. Gains cannot cancel losses.
The baseline practice examples are not earlier admitted training windows and
do not prove NeuroShard's native replay lifecycle.

## Measurements and three-way diagnosis

- Save added/accepted weights and router tensors, optimizer state, RNG states,
  phase/step, seed and tokenizer binding, and source/data hashes after each
  training phase. Never overwrite an earlier study or checkpoint.
- Record per-step CPU/wall time, input/answer/padded tokens, gradients and
  training routing statistics. Gate traces distinguish soft probabilities from
  the corresponding hard choices.
- Save every automatic generated answer and every routing call's choices and
  margins. Count the last token of each generation call as an answer decision;
  selecting an expert somewhere in a prompt is not an answer-use result.
- Run baseline scoring, expansion training, control training, and each serving
  arm in fresh processes. Measure Linux process RSS high-water marks, including
  loading and transient serving allocations. Training memory cannot contaminate
  serving memory. Record complete child CPU/wall expenditure including setup
  and failure, separately from the comparison's training budget.

When automatic gains are absent, the first diagnosis is **expert learning not
observed** if added weights do not change and/or the fixed training-only probe
fails its 5% relative loss reduction. This is a training signal, not proof that
the expert cannot generalize. The second is **automatic integration not
effective** when that signal exists but automatic answer gain or actual added
use does not. The third is **complete system evaluated** when automatic gain
and added use both appear; retention or cost can still reject it. Actual
generated-answer gains take precedence over the training-probe diagnostic.
All gate details and raw replies remain visible.

Only automatic generated answers can pass: at least two more correct development
answers than both parent and control, zero protected losses, and added-module
use on a gain. Expansion p95 latency must be at most 1.5 times control and 10
seconds; fresh-process peak RSS must be at most 1.5 times control and 4 GiB.
Single-thread training is charged as above. The complete study is bounded at
1,800 wall seconds, each child at 600 wall/CPU seconds. The memory number is an
acceptance bound, not an OS memory-enforcement promise.

## Run boundary

The contract is [staged-integration.json](../config/experiments/staged-integration.json),
the data are [staged-integration-data.json](../config/experiments/staged-integration-data.json),
and the source inventory is
[staged-integration-freeze.json](../config/experiments/staged-integration-freeze.json).

Verification does not load a model or train:

```bash
PYTHONPATH=src venv_build/bin/python scripts/run_staged_integration.py
```

An explicit `--run --seed LOCAL_135M_DIRECTORY --home NEW_STUDY_DIRECTORY`
requires every frozen file to match committed HEAD. The runner downloads
nothing, has no GPU/1.7B option, refuses altered seed files, and keeps evidence
after failure. No real-model run is part of preparing this candidate.

[Independent hosting](INDEPENDENT_HOSTING.md) remains separate. Four genuinely
independent validator administrators are still required; this candidate neither
supplies operators nor completes item 4.
