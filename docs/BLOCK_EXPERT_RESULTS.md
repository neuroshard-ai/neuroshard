# Block expert: stopped at the parent baseline

The CPU experiment on commit `5354557948d3ab7e727f94d3b2bf040b13164698`
ran on September 22, 2026. Its [frozen contract](BLOCK_EXPERT.md) required at
least eight correct parent retention answers before training. The parent
supplied six, so execution stopped at that prerequisite.

**No expert or control was trained. The added blocks have no learning-quality
result.** The initial-identity and cached-training tests passed on tiny models;
they do not establish useful learning on the 135M model.

| Parent baseline | Correct | Complete replies | Parseable replies |
| --- | ---: | ---: | ---: |
| New modular-addition questions | 0/64 | 0/64 | 0/64 |
| Ordinary-addition retention questions | 6/64 | 19/64 | 16/64 |

The strict scorer required EOS within 32 generated tokens. The new questions
all exhausted that allowance; many retention replies also failed to finish.
This is a baseline suitability result, not evidence for or against the proposed
block-learning method. The eight-answer entry threshold and the output rule
remain unchanged. These evaluation cases are now opened.

Execution consumed 423.75 wall seconds and 422.55 child CPU seconds, plus
0.42 coordinator CPU seconds. It wrote the baseline, six protected identities
and its stop receipt. No training cache, optimizer history or checkpoint exists.
No GPU or external allocation was used.

The [public evidence record](../config/experiments/block-expert-baseline-result.json)
includes every baseline reply, the complete process receipt, source/data
bindings, raw-file hashes and independent score reconstruction. The execution
freeze still verifies against the committed sources. Nothing is admitted,
promoted or marked complete; item 4 remains separate.

The requested expert-learning comparison remains unfinished. A later design
must address that this expert-only study's retention prerequisite prevented it
from measuring competence. This result does not authorize relaxing the recorded
threshold, resuming under this freeze or calling the method a learning failure.
