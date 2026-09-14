# Consolidating the learned update before accepting it

The [calculation-step experiment](REASONED_LEARNING_RESULTS.md) learned useful
arithmetic but changed five previously correct final answers. This experiment
tests whether a smaller version of that same learned weight update retains the
new skill while preserving earlier answers. Its [plan](../config/experiments/consolidated-learning.json)
is frozen before preparation and execution. It adds no new gradient updates,
model layers, installed client command or native transaction.

## One resulting model

For each owned FP32 parameter, compute

`weight = alpha * fast_weight + (1 - alpha) * parent_weight`.

The parent is the accepted phase-A step-128 checkpoint. The fast weights are the
recorded, rejected calculation-step checkpoint at step 384. Execute the two
declared PyTorch CUDA operations on each shard. The output retains the fast
checkpoint's complete Adam moments, step counters, parameter ages and groups.
The consolidation changes weights without counting another optimizer update or
issuing tokens. Its checkpoint records both input roots and the coefficient.

This produces one 1.7B model. Generation uses that model for every task; it does
not route requests between models, repair outputs or invoke a calculator.
Intermediate calculation prompts and the 256-token greedy generation limit are
the same for parent and candidate. Earlier JSON-only tasks keep their original
prompt and scoring rules.

Weight interpolation has a research precedent in [Wortsman et al., WiSE-FT](https://arxiv.org/abs/2109.01903),
which improved fine-tuning robustness on image classification distributions.
That motivates this experiment; it does not establish the result for this LLM.
The retained fast optimizer state is an explicit continuation choice. This is
one terminal consolidation, not an evaluation of the repeated updates in the
[Lookahead optimizer](https://arxiv.org/abs/1907.08610). A later learning cohort
would still need to test continuation from the consolidated state.

## Development selection

The only coefficients are `0.75`, `0.5`, `0.25`, `0.125`, in that order. For each:

1. Generate all 512 fresh, original-style development task answers. Reject the
   coefficient if it loses any answer the parent got right. A gain elsewhere
   cannot hide that loss. For this rejected coefficient, skip the other roles.
2. Otherwise, generate 128 new-task development answers and score 128 fresh
   conversation documents. Require at least eight net new correct answers, no
   new-task family score loss, and mean conversation loss change at most +0.01.
3. Select the first coefficient that passes. Record every attempted coefficient,
   its actual checkpoint, complete evaluated answers and decision. If none
   passes, preserve the failure and leave final evaluation unopened.

The prior-task probe is eight times the preceding experiment's size. Even zero
losses on this larger probe is not a guarantee about all unseen prompts. The
separate final evaluation is therefore mandatory.

The actual selection and checkpoint must be Git-committed before any final
evaluation. Changing a source, tokenizer, runtime, prompt, target mask, input
checkpoint or prepared artifact invalidates the frozen execution.

## Final gate

On 512 new-task cases, require at least 16 net correct-answer gains and exact
one-sided McNemar *p* < 0.05. Require no per-family score loss on those cases or
256 fresh original-style prior tasks. On 256 fresh conversation documents,
require the paired bootstrap 95% loss-change upper bound to stay at most +0.02
nats. Use 10,000 bootstrap samples and the frozen seed. Loss alone cannot pass.

These counts double the preceding final sets. The minimum primary gain fraction
and retention bound are unchanged. A failed final is reported as a failure;
another coefficient cannot replace the selected one afterward.

Preparation excludes every adaptive, direct-continuation and calculation-step
role, including their now-exposed finals, by identity and normalized content.
Conversation inputs come from the exact SHA-256-pinned SmolTalk test parquet.
A frozen permutation of its 24,229 row indices selects unused complete
documents for development and then final evaluation, with exclusions recorded.
The generated tasks use new frozen seeds. No secret evaluation is assumed.

## Execution

The existing `scripts/run_sharded_training.py` accepts
`consolidate prepare`, `consolidate screen`, `consolidate evaluate` and
`consolidate score`. Preparation runs without GPU allocation. Screening and
evaluation use the recorded three-owner A10G runtime and boundaries
`[0, 6, 15, 24]`; each worker loads only its owned input weights and optimizer.

Keep every attempted checkpoint and screen. Back up complete optimizer state
and evidence with verified readback before deleting the temporary resources.
The allocation is bounded to three workers, six hours and $100. This is a
learning-method experiment; it changes neither the public 0.4.0 network nor its
serving model. A future native consolidation would need an explicit non-issuing
transition rather than pretending this interpolation was another trained step.
