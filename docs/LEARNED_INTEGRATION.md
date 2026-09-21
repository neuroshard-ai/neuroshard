# Learned integration of new capacity

**Status: specified, not executed. No GPU. Not admission.**
The [programming-growth campaign](PROGRAMMING_GROWTH.md) trained two last-four-layer
tails independently and then tried inexpensive selectors or weight arithmetic.
Those challengers failed their declared gates. Complementary coverage still
exists in the unchanged tails. Four heuristic selectors failing does not
establish that learned routing cannot work. This experiment changes how the
model learns to use added capacity. It is not another independently trained
tail followed by a selector search. A fresh dataset by itself would not address
the identified failure. The original 128-task final stays closed. The opened 64
leftover cases remain development history.

Machine-readable contract:
[`learned-integration.json`](../config/experiments/learned-integration.json)
(`6801d1a2…`). `gpu_launch_authorized` is false. `train` is false. This document
does not spend GPUs.

## Closed campaign, recorded combination

TIES preserved all 29 incumbent successes and reached 31/64, including leftover
task 54, which neither original tail produced. It failed the declared gate
(unique added 1/3; required 32/64) and remains rejected. The leftover incumbent
extra remains the research baseline because challengers failed acceptance, not
because it had the highest observed score. Record:
[PROGRAMMING_GROWTH_TIES.md](PROGRAMMING_GROWTH_TIES.md).

The conclusion that one extra decode cannot harvest complementary coverage
exceeds that evidence. Training useful pieces and making them useful together
are separate learning problems.

## Question

> Does adding trainable capacity improve the complete answering system more
> than spending the same training budget on its existing capacity, while
> preserving earlier capabilities and meeting the same serving budget?

Published research demonstrates learned expert routing (Switch Transformers,
GShard) and expansion of expert capacity while holding the active expert count
fixed (sparse upcycling and later MoE expansion studies). Those results support
investigating this approach. They do not prove continual retention on generated
answers, permissionless operation, or NeuroShard serving latency.

## Three frozen decisions

1. **Learn integration during training.** Freeze accepted modules. Add one
   expert, initialized as a copy of the incumbent expert in the last decoder
   layer. Train that expert and a linear token-to-expert gate on new examples
   plus replay of earlier training data. Still evaluate retention: changing
   routing can break answers without changing their weights. Active experts per
   token stay at one.

2. **Use a meaningful control.** Continue training the existing last-layer MLP
   without expansion, on the same new and replay examples, matched steps,
   optimizer, and generation cap. Measure wall-clock latency, peak memory, and
   active MLP FLOPs per token. Sparse activation alone does not guarantee low
   network latency.

3. **Evaluate the complete system on fresh questions.** Freeze the method before
   confirmation outcomes. Success is generated executable answers, not lower
   loss. Development (32 unused MBPP tasks) must pass before confirmation (32
   unused MBPP tasks) opens. The opened 64 leftover cases and the original
   128-task final stay out of evaluation.

## Stages

Stage 1 establishes the mechanism on SmolLM2-135M-Instruct, last decoder layer
only, single process. That is a mechanism study. It is not a 0.4.0 protocol
upgrade and not the 1.7B research assistant. `src/neuroshard/core/model/moe.py`
is not this experiment's runtime.

Stage 2, only after stage 1 passes its generated-answer confirmation gate,
exercises the same method through the existing four-owner 1.7B runtime. Sparse
activation is not a latency result.

## Frozen splits

Seed `20260921`. Remaining MBPP identities after the burned programming-growth
union (789 ineligible tasks, including the parent 128-task final and the opened
64). New train 64, replay train 32, code retention 16, development 32,
confirmation 32. Eight general conversations from the existing programming-expert
general corpus are reserved for exact parent-response preservation; their
document identities are recorded in a later execution freeze, not here.

Gate, frozen before confirmation: expansion must beat the matched control by at
least one full-test success, keep at least the parent's code-retention successes,
match all eight parent general responses, keep one active expert per token, and
stay within 1.5× control p95 latency and peak memory, with p95 ≤ 90 s.

## What this specification does not do

It does not train. It does not authorize a GPU launch. It does not promote a
model. It does not reopen the original final. It does not reuse the opened 64
as evaluation. It does not treat the leftover tail as accepted. It does not
replace the 0.4.0 genesis. Item 4 remains an independent operator.
