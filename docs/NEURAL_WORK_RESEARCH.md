# Neural work: arithmetic binding before mining

Status: experimental CPU reference, September 20, 2026. Apache-2.0.
This research does not activate a consensus change, issue NEURO, restart the
GPU alpha, or change the six bounded [live-LLM criteria](../TODO.md).

## Objective and decision

The intended network allows strangers to contribute model shards, earn rewards
for accepted computation and use a collectively improved assistant. The current
question is whether a prescribed shard computation can also support a cheap,
hard-to-forge mining proof. Three independent obligations are involved:

1. **Arithmetic:** the prescribed computation produced the asserted result.
2. **Resource security:** producing a fresh mining opportunity cannot be reduced
   to cheap manipulation of an already known result.
3. **Utility:** the work belongs to an admitted task; a separately evaluated
   answering system improves before it replaces the accepted system.

The first implementation checks a complete, deliberately small linear training
transition using randomized matrix-product checks. It includes two executable
negative controls: receipt hashing allows fresh nonce search without fresh
training, and final-output-only low-rank noising admits a factorization shortcut.
Neither candidate qualifies as neural-work consensus. A cheaper arithmetic
verifier would still be useful for the existing funded audit path.

The executable reference is [neural_work_reference.py](../scripts/neural_work_reference.py),
with an [experiment driver](../scripts/study_neural_work.py),
[adversarial tests](../tests/test_neural_work_reference.py), and a
[frozen measurement plan](../config/experiments/neural-work-reference.json).
Results must be compared against the faster of two exact dense replays, including
an optimized binary64 BLAS baseline within the declared integer bounds. A slow
integer baseline alone is not evidence of economical verification.

## Prior work and prospective contribution

[Komargodski and Weinstein, version 4](https://arxiv.org/html/2504.09971v4)
already study permissionless useful-work mining from arbitrary matrix products.
Their construction uses randomized encodings and intermediate computation
transcripts. Its security depends on a conjectured hardness assumption. It
explicitly explains why low-rank noising of the final output alone fails on easy
inputs. Our negative control reproduces that failure; it does **not** break their
full construction, implement cuPOW, or inherit its security claims.

[Slalom](https://arxiv.org/html/1806.03287v2) applies Freivalds-style checks to
outsourced neural linear operations: check `C r = A (B r)` instead of recomputing
`C = A B`. For a fixed incorrect product over a field and an independent uniform
field vector, false acceptance is at most `1/p` per round. Its trusted-hardware
execution model is not a permissionless consensus model.

[Verde](https://arxiv.org/abs/2502.19405) addresses disputes across ML computation
graphs with reproducible operators, under an at-least-one-honest-provider
assumption. Neither an inexpensive referee nor a matching signature funds or
establishes independent observation.

The prospective NeuroShard contribution is a binding between productive shard
execution, permissionless resource accounting and an evolving accepted graph.
This document and toy verifier establish no novelty claim. We must account for
dependency correctness, data availability, funded verification, numerical cost,
and mining shortcuts together before making a stronger claim.

## Numerical statement actually implemented

A job contains signed integer matrices `X` (batch by input width), `W` (input by
output width), and `T` (batch by output width), together with scale `S` and inverse
learning rate `L`. All dimensions are at most 512 and operand entries have
absolute value at most 32,768. The prototype verifies these operations:

```
F  = X W
Y  = round(F / S)
E  = Y - T
GW = X^T E
GX = E W^T
W' = W - round(GW / (S * batch * L))
```

Rounding is nearest integer with ties away from zero. This is a declared
fixed-point approximation to linear least-squares SGD: it uses a straight-through
gradient convention for rounding. It is not the mathematical derivative of a
discontinuous rounding function. `GX` is the raw upstream-gradient numerator;
its conversion to a preceding shard is outside this local job.

The worker supplies `F`, `GW`, `GX`, and `W'`. The verifier reconstructs `E` from
the committed forward result and targets, checks **all three** products, and
checks the complete elementwise update. A fabricated gradient with a consistent
optimizer update is rejected. This closes that attack for this local profile;
it does not extend the existing [compact SGD witness](COMPACT_UPDATE_DISPUTES.md)
to the live Transformer automatically.

This is one linear training shard, not a complete Transformer, trained language
model, cross-device GPU numerical profile, or proof of its upstream activations.
Softmax, attention, normalization, SwiGLU, AdamW, tokenization, whole-graph
dependencies and privacy are not covered. Inputs and targets come from the
admitted job; their external provenance and desirability are stewarded assumptions.

### Integer soundness and bounded arithmetic

For each product, the verifier derives an absolute bound
`B = inner_dimension * max_abs(A) * max_abs(B_matrix)`. Both the true product
and claimed product must lie in `[-B, B]`, with
`2 B < 65521 * 65519`. It checks the product over **both** prime fields.
Consequently a nonzero bounded integer error cannot vanish in both fields.
The tests explicitly attempt to add one modulus to a product: a one-field check
would miss it, while this two-field check rejects it.

Five independent projection columns per field give a per-fixed-invalid-transcript
arithmetic error bound of at most `65519^-5` (approximately `2^-80`) in the ideal
random-vector model. This is **not** a complete protocol security level. SHAKE
expands a 256-bit verifier seed using domain-separated contexts and rejection
sampling. The concrete construction additionally assumes cryptographic hash/XOF
security, an immutable pre-challenge commitment and an unpredictable honest
challenge. Across many attempts the error bound accumulates. No adversary may
choose the seed, reroll challenges or edit its transcript after learning one.

Every modular dot product is bounded by `512 * 65520^2`, safely below int64
overflow. Range checks precede absolute values, including for `INT64_MIN`.
The dense binary64 baseline is exact here because every integer product and
partial sum has absolute bound below `2^32`, well within binary64's exact-integer
range. This argument does not authorize ordinary floating-point PyTorch replay.

### Interactive audit, not a succinct public proof

The full transcript is committed before the verifier samples its challenge.
The auditor then reads the witness and checks it. This is an interactive
verification protocol with a local challenge issuer, not a SNARK, fraud proof,
Fiat-Shamir consensus certificate or proof that the worker performed particular
physical FLOPs. Anyone replaying a saved public seed also needs evidence that it
was unpredictable when the commitment became immutable. The toy process does
not establish that through a blockchain.

The auditor needs all job tensors and witness tensors. A small seed is not the
proof's total communication cost. The report records witness serialization,
commitment time, audit time, cached/uncached tensor bytes and an explicitly ideal
bandwidth floor. There is no network transport benchmark. The producer may be
able to reuse retained training buffers; otherwise witness retention adds memory.

## Admission and payment model

`AdmissionBook` is an in-memory test model with one trusted steward and honest
verifier. It is not native settlement, a service endpoint, persistent accounting
or a new ledger. It records at most one accepted result per numerical work ID.
Worker strings stand for already authenticated callers; this harness does not
implement authentication, identity checks, leases, collateral, budget economics
or Sybil-resistant assignment.

The work ID binds the profile, learning parameters and all input tensor bytes.
It deliberately excludes the worker, sponsor, human job name, block header and
challenge. Relabeling the same computation or creating another account cannot
make it payable twice in this model. A changed checkpoint, data tensor or recipe
requires separately admitted work. Commitments cannot be replaced and repeated
challenge requests return the same seed. Missing evidence never pays. Correct
results are recorded independently of any mining-lottery outcome.

This rule does not prove that admitted data are useful, prevent a malicious
steward from inventing jobs, solve competing-fork accounting, establish an
independent operator, or make a signup form Sybil resistance. A public rollout
needs explicit assignment, challenge-generation, censorship, expiry, availability
and reward-budget mechanisms.

## Attacks and what they decide

| Attempt | Expected observation |
| --- | --- |
| Corrupt forward, either gradient, or updated weights | Both dense replay and projected verification reject. |
| Invent zero gradients and matching unchanged weights | The gradient relation fails, even though the optimizer relation matches. |
| Change witness after challenge | Commitment mismatch; no second commitment or seed draw. |
| Copy result into another worker, data or checkpoint claim | Assignment/work identity check rejects. |
| Repeat or relabel settled numerical work | One accepted result only. |
| Withhold an intermediate tensor | No acceptance. |
| Exploit integer overflow or a one-prime alias | Profile bounds/two-field verification reject. |
| Reuse a correct cached transcript under a new challenge | **Accepted as arithmetic evidence:** does not establish fresh effort. |
| Hash a cached receipt plus freely chosen nonce | **Mining shortcut succeeds:** reject this PoW design. |
| Prove only `(A+E)(B+F)` with low-rank noise and `A=B=0` | **Low-rank shortcut succeeds:** reject final-output-only noising. |

The last shortcut computes `EL (ER FL) FR` with `n^2 r + 2 n r^2`
scalar multiplications, versus `n^3` for the already-noised dense product.
These are arithmetic counts, not an optimized hardware timing comparison.
The cited cuPOW transcript construction is specifically intended to address this
class of shortcut; these attacks do not settle its security or practical cost.

## Reproduction and research gate

Install the pinned NumPy and pytest versions in an isolated research environment
(`numpy==2.2.6`, `pytest==9.1.1`). No cloud credentials, model downloads or network
services are used. Run from a checkout containing the committed experiment:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  python -m pytest -q tests/test_neural_work_reference.py

OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  python scripts/study_neural_work.py \
  --output .neuroshard/neural-work-reference/result.json
```

The driver refuses uncommitted changes to its numerical source or frozen plan.
The synthetic 16-step linear fit only checks that the declared arithmetic can
learn a toy relationship; it is not a fresh LLM holdout, retention test or model
promotion. Timing samples repeat cached inputs, include one warmup, and are
reported individually. Lower verification time must survive the optimized exact
baseline and all commitment/transport costs before motivating a GPU trial.

The next construction must bind full execution dependencies to mining-relevant
intermediate work, specify new-header/fork behavior and task scarcity, and fund
honest checks. Matrix correctness alone cannot authorize block-production weight.
An independently reviewed resource-hardness argument is a separate prerequisite
from passing the software tests. Native BFT remains the research reference.
