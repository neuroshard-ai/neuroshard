# Continued interface learning and batched audit measurement

The continuation completed and was rejected. Single answers stayed at 13/16,
combined answers stayed at 2/16, and structured answers stayed at 7/8. General
retention failed: the upper confidence bound was +0.02127 against +0.02. Both
1,536-update arms and both exact optimizer replay checks completed. The final
remained unopened. Mixed training response loss fell from 0.27065 to 0.01163;
this did not transfer to better held-out combined answers. More steps on this
interface are not the next experiment. See the
[complete result](../config/experiments/expert-interface-continuation-results.json).

Eight GPU token checks took 0.640–0.661 seconds each. The three 128-token cached
responses took 15.44–15.68 seconds to generate. Each fixed-context check sent
83,886,320 tensor bytes, versus 3,968,048–19,314,688 for cached generation across
the eight cases. Six cached responses matched; two differed (one and two token
positions). Both deliberately forged responses were rejected. These measured
disagreements confirm that native acceptance cannot silently substitute one
numerical method for the other.

The next [frozen streaming trial](../config/experiments/checked-streaming-trial.json)
uses those same weights and eight exposed cases. It checks the implemented
canonical correction procedure, complete-response re-execution, two different
chunk-boundary comparisons, and two first-token forgeries. It measures first
checked-chunk latency, complete time and traffic under a one-hour/$10 bound.
There is no training or quality search in this run. The candidate remains
rejected regardless of an execution-method pass.

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
