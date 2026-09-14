# Calculation-step training with protection of correct token margins

This is a separate, fixed-size learning experiment following the
[failed direct-answer continuation](CONTINUED_LEARNING_RESULTS.md). Its
[machine-readable plan](../config/experiments/reasoned-learning.json) is frozen
before preparation. Training requires committed prepared artifacts. It uses the
existing sharded trainer and does not change the public chain or issue NEURO.

The [completed run](REASONED_LEARNING_RESULTS.md) gains 50 net new-task answers
but fails its answer-retention floors. The original contract remains unchanged.

## Hypothesis

The preceding recipe acquired no new correct final answers and lost two old
answers, despite improving response loss. This experiment targets both observed
failures with a joint intervention:

1. For new invoice-total examples, supervise each product and running sum before
   the final JSON. At inference, the model generates those intermediate tokens
   itself. Both parent and candidate receive the same prompt and 256-token
   generation limit. A parser extracts the final JSON; it never runs arithmetic,
   repairs a wrong total or supplies intermediate answers. Original prior-task
   prompts still require a single JSON object without a work block.
2. On trained replay, add a penalty for reducing correct reference token
   margins. Let the label's reference margin be its logit minus the largest
   competing logit. If that margin is positive, the student is penalized when
   its margin falls below the reference margin clipped to [0.5, 2.0]. Positions
   where the teacher favors an incorrect label receive no margin protection.
   Ordinary weighted response loss and replay KL remain in the objective.

The margin penalty is averaged over replay target tokens and has coefficient
1.0; replay KL has coefficient 2.0. The frozen reference never receives
gradients. The margin penalty is zero, with zero gradient, when its target is
already met. This is a regularizer, not a guarantee about unseen greedy answers.

Intermediate computation has an existing research basis: [Nye et al., *Show
Your Work*](https://arxiv.org/abs/2112.00114) demonstrated improvements from
scratchpad supervision on arithmetic and program execution. That evidence
motivates this test; it does not establish that this model or recipe will pass.
This combined experiment cannot attribute a result separately to the scratchpad,
margin penalty or changed training budget.

## Frozen execution

Start again from the passing phase-A checkpoint at global step 128, including
its Adam state. Keep 1,711,376,384 parameters, 24 layers and boundaries
`[0, 6, 15, 24]`. No worker needs the whole student, optimizer or reference.

The prepared dataset contains 4,096 new generated tasks, 2,048 previously
trained phase-A task windows and all 2,048 previously trained conversation
windows. Two declared epochs use every record once per epoch. Each of the 256
updates contains 32 new examples and 32 replay examples. Repeated data is
explicit in the frozen schedule; it is not represented as distinct new data.
Peak learning rate is 3e-6, with eight warmup updates and the existing decay
rule. Other optimizer and numerical runtime settings are pinned in the plan.

The only selectable endpoint is step 384. Checkpoints at 192, 256, 320 and 384
support retention checks, abort and recovery. They cannot supply alternative
final candidates. The driver measures generated development answers at each
checkpoint as well as conversation retention.

## Abort and acceptance

Before training, abort if the parent's development score leaves too few errors
for the declared development gain to be possible. At every checkpoint, abort
if mean development retention loss rises above +0.05 nats. From step 256 onward,
require at least two correct development invoice totals. At the final endpoint,
also require four net new correct development answers and no per-family count
loss on either development task role. These are development gates, not evidence
of final acceptance.

Only an actual step-384 checkpoint with its passing development decision may be
committed for final evaluation. The final gate is unchanged in strength:

- At least eight net correct-answer gains on 256 fresh primary tasks and exact
  one-sided McNemar *p* < 0.05.
- No per-family accuracy loss on either the primary set or 128 fresh original
  JSON-only prior tasks.
- Conversation retention loss-change 95% bootstrap upper bound at most +0.02
  nats on 128 fresh documents. Loss improvement alone cannot pass.

## Data and reproduction

Preparation excludes all prior adaptive and continued-learning roles by
identity and normalized rendered content. Only the explicitly selected,
previously trained phase-A replay may overlap training history. New task seeds
and the exact prepared role hashes are published before training. Conversation
development starts at SmolTalk test row 21,000; final retention starts at row
23,000. Previously used documents are also excluded by content. The source's
test file has 24,229 rows, so both bounded scans fit that file.

Use `scripts/prepare_continued_learning.py --plan
config/experiments/reasoned-learning.json`, supplying the pinned tokenizer,
phase-A `--prior-train`, the frozen `--plan-commit`, and `--exclude-continued`
pointing at the previous continuation's role directory. The existing
`scripts/run_sharded_training.py continued` and
`scripts/score_continued_learning.py` consume the new plan and committed
prepared/selection files. The new plan has separate artifact paths; it cannot
silently use the previous plan's candidate lock.

The allocation remains three A10G workers, a six-hour deadline and a $100
budget cap. Preserve evidence and complete optimizer checkpoints before deleting
the temporary instances and disks. A pass would establish a bounded learning
result. Native payment still requires activated, reserved and verified training
windows, followed by a separate quality decision for serving.
