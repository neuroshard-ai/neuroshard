# Continued interface learning and batched audit measurement

The first owned-interface trial improved single-fact answers to 13/16 and
combined answers to 2/16. General retention passed (upper bound +0.01412 against
+0.02), and structured answers stayed at 7/8. It failed the combined-answer and
control-gain gates; the final remains unopened. Both three-owner optimizer
restart checks passed. The [complete result](../config/experiments/expert-interface-results.json)
includes the public evidence and retired-resource accounting.

The [continuation](../config/experiments/expert-interface-continuation.json)
tests a specific remaining possibility: insufficient optimization of the mixed
conversation interface. Mixed training response loss fell from 1.8643 to 0.3072
over four epochs; the auxiliary source loss ended at 0.8829. These training
measurements do not guarantee generalization. They motivate one bounded run
from the exact preceding gate and adapter roots.

Both arms receive 1,536 further updates on the unchanged training documents.
Each starts from its own committed terminal weights with explicitly fresh Adam
state, a lower learning rate and the same quality gates. This is a newly
prescribed optimization trajectory, not a restart of the former optimizer.
Checkpoint spacing is 64 updates, with an additional boundary for replaying the
last 16. No intermediate selection is permitted. The two-hour/$25 limit and
automatic retirement remain in force.

After quality scoring, eight frozen development cases also measure a distinct
inference verification program. A reported response is included as input to one
causal forward pass across all source owners. At each response position, the
checker compares its predicted next token with the reported token. Two cases
also change the first output token to check rejection and causal independence
from future input. All original and tampered observations are retained.

The one-pass idea is described in the [Verde production discussion](https://www.gensyn.ai/research/verde-verification-system-in-production).
NeuroShard's diagnostic uses one document padded to a fixed context, with every
position marked valid and a causal attention mask. Under a deterministic causal
implementation with this fixed execution shape, earlier predictions cannot
depend on later reported tokens. This gives a well-defined batched check of the
greedy response. It is our proposed numerical contract, not an assertion that
ordinary cached decoding follows the same floating-point program.

The trial therefore measures cached-response mismatches as well as time and
transport cost. Passing examples do not establish universal equivalence. The
existing native chain still uses its admitted replay method. A deployment of
this alternative would need its own execution profile, provider behavior on
numerical disagreements, funded audits and native acceptance integration.
Cross-hardware reproducibility remains a separate constraint; see the
[Verde paper](https://arxiv.org/abs/2502.19405).
