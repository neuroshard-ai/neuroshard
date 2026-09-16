# Learned mixing of native token predictions

The hidden-state fusion trial failed its development gate. A subsequent frozen
source diagnostic recovered all 16 single-fact answers under the experts'
original input contracts, using either frozen output head. On raw questions,
protocol stayed at 8/8 and directory dropped to 3/8. This shows that the hidden
connection lost some knowledge already available from its sources; output-head
incompatibility does not explain the main observed failure.

This trial keeps the native model predictions in their shared token vocabulary.
For a causal hub state, a small neural gate produces nonnegative source weights.
The next-token distribution is the normalized weighted sum of the preserved
assistant and the source distributions. An inactive gate reproduces the hub's
inference distribution exactly. Its prescribed boundary derivative allows the
initially inactive sources to learn. A fixed probability floor bounds numerical
gradients during training.

The gate uses the preserved assistant's hidden state. It receives no task ID,
expected answer, directory lookup or generated subquestion. Every frozen model
path sees the same original conversation and shared generated tokens. Each
expert sends its last hidden state to the owner of its native output head.
The two backbones remain partitioned and the learned tails remain separately
owned. All source paths still execute; this is not a sparse compute claim.

[`probability-mixture-trial.json`](../config/experiments/probability-mixture-trial.json)
freezes the 262,659-parameter gate, both 512-update arms, exact optimizer restart
check and the unchanged training/development/final data and acceptance gates.
The comparison substitutes unchanged parent predictions for learned expert
predictions. It isolates source information and does not establish the best use
of an equal total resource budget without growth.

The development set has informed method selection. The original final set has
never been scored and remains unopened unless development passes. No directory
interface fix is claimed: the diagnostic's 3/8 raw directory result is a known
limitation. A failing candidate is rejected with all outputs retained. The
allocation is capped at 90 minutes and $20; neither success nor native serving
admission is assumed.
# Recorded outcome

The terminal development gate failed: directory 3/8, protocol 6/8, combined
questions 0/16; structured tasks improved from 2/8 to 7/8. General response-loss
UCB was +0.03775 against the +0.02 limit. Both optimizer restart checks passed,
and all five owners saved identical results. The final was not evaluated.
See the [machine-readable result](../config/experiments/probability-mixture-results.json)
and the next [owned-interface prescription](EXPERT_INTERFACE_TRIAL.md).
