# Observable reasoning: completed, rejected

The CPU experiment on commit `f0515520fb2198822f562dd9eb400caa7fc3858f`
finished on September 25, 2026 at 19:52:43 UTC. Execution completed normally;
the quality gates failed. The service is stopped. No retry, admission, public
model upgrade, NEURO issuance, GPU launch, or checklist credit follows.

The [execution contract](OBSERVABLE_REASONING.md) and its historical scorer
remain unchanged. The [complete recorded result](../config/experiments/observable-reasoning-result.json)
contains all raw replies, routes, per-answer scores, gate decisions, receipt
hashes, and CPU accounting. An offline rescore reproduced every result field;
the committed source freeze, six receipt hashes, selector fit, and protected
list hash were verified. No model was trained or decoded during that review.

## Frozen comparison

| System | New answers / 96 | Retention answers / 128 |
| --- | ---: | ---: |
| Parent | 0 | 0 |
| Added blocks, automatic serving | 7 | 0 |
| In-place trained control, same routing | 10 | 0 |
| Constant training-majority answer `6` | 16 | — |

The expert needed at least 32/96 and net gains of at least 6 over every control,
with positive paired bootstrap lower bounds. Its gain over the trained control
was **−3/96**, lower bound **−0.114583**; versus the constant it was **−9/96**,
lower bound **−0.1875**. Beating the zero-scoring parent alone does not pass.
Forced-expert scores matched automatic scores on all 96 new questions.

The question-only selector routed all 96 modular questions to the candidate
and all 128 ordinary additions to the parent, for both trained arms. This
demonstrates task routing on these two synthetic families. It does not
demonstrate a general correctness predictor or automatic multi-skill reasoning.

## What failed

All 96 new replies from both trained models terminated and followed the
declared trace format. The added blocks did not reliably compute the answers.
An additional **post-run diagnostic**, not an admission gate, checked the
generated trace components against the committed arithmetic inputs:

| Trace property / 96 | Added blocks | Trained control |
| --- | ---: | ---: |
| First operand remainder correct | 14 | 18 |
| Second operand remainder correct | 18 | 12 |
| Generated sum equals the two generated remainders | 52 | 79 |
| Generated final answer equals generated sum modulo 7 | 88 | 79 |
| All intermediate values and final answer correct | 0 | 2 |

Trace formatting and mostly consistent final reduction did not establish the
required composition of operand remainders and addition. Selection did not
cause the failed new-answer gate. These opened cases cannot become a fresh
success set for later method selection.

The retention measurement also had a material limitation: **no parent reply
qualified under the frozen whole-reply format rule**. Some prose replies did
contain the correct arithmetic answer; for example, the parent correctly wrote
that 82 + 184 is 266, surrounded by explanatory sentences. The strict parser
rejected that reply. Consequently, 0/128 is the protocol score, not evidence
that the parent cannot add or that all its semantic answers were wrong.

The protected list was empty. `all_protected_kept: true` is therefore vacuous;
`protected_set_nonempty: false` correctly prevents a pass. All 128 automatically
served retention texts and termination flags did exactly match the baseline,
but this does not repair the missing nonempty protected-answer gate. Future
retention work needs an established usable baseline, not a post-hoc change to
this experiment's parser or pass conditions.

## Execution and cost

The expert completed 864 updates; the control completed 903. Each trained
7,080,192 parameters. Both frozen-parameter checks passed. The control spent
2,476.966 optimization CPU seconds against a required 2,474.886 seconds, which
included the full expert training process and expert-specific prefix cost.

Total wall time was **2 hours 40 minutes**. Billed child CPU was 9,555.884
seconds, plus 9.997 orchestrator CPU seconds. There were no retries and no new
cloud instances. Complete-response p95 was 6.434 seconds for the expert policy
and 6.564 for the control policy, across the combined evaluation. Both serving
workers peaked below 1.14 GB RSS. Cost, latency, frozen-weight and memory gates
passed; competence, comparative gain and nonempty protection did not.

This candidate is closed. The measured obstruction is useful neural learning
on the declared held-out recombinations, with an additional retention-design
limitation. More routing tweaks or more peers do not address that result.
The next method, if pursued, needs its own contract; no follow-up execution is
authorized by this report. Item 4 still requires four independent operators.
