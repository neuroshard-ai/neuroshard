# Neural-work binding: first CPU result

September 20, 2026. Research source
[`1452bf57dd6454c9a904be1283c4ca77966fe04f`](https://github.com/neuroshard-ai/neuroshard/tree/1452bf57dd6454c9a904be1283c4ca77966fe04f).
The [specification](NEURAL_WORK_RESEARCH.md) defines arithmetic, trust assumptions
and reproduction. The [complete final record](../config/experiments/neural-work-boundary-results.json)
includes all samples, source hashes, verifier seeds and attack commitments.

## Finding

**Complete arithmetic verification works for the declared linear shard, but this
implementation is not cheaper than the strongest measured replay control and is
not a proof-of-work consensus construction.**

The verifier checks forward computation, both gradients and the optimizer. It
rejects a coherent invented gradient even when the optimizer equation matches.
The admission model rejects changed inputs, withheld witnesses, post-challenge
edits and duplicate payments. This establishes no independent operator, native
issuance, language-model training or live-network change.

Two deliberately weak mining designs fail executable attacks:

- Hashing a cached training receipt plus a nonce produces new winning tickets
  with **zero additional training products**.
- Proving only the final low-rank-noised product for zero useful inputs permits
  a factorization shortcut: 69,632 scalar multiplications instead of 2,097,152
  for the dense product at size 128 and rank 4. These are operation counts, not
  hardware timings. The cuPOW paper discusses this failure and uses intermediate
  transcripts to address it; these tests neither implement nor refute cuPOW.

## Honest-path cost

Existing Xeon Platinum 8259CL host; Python 3.10.12, NumPy 2.2.6, OpenBLAS 0.3.29;
one BLAS/OpenMP thread requested. Five cached repetitions after one warmup per
operation, one operator, no new AWS resources. These local CPU measurements are
not GPU/WAN predictions or performance confidence intervals.

Every audit/replay timing includes validation and commitment checking. The
projected verifier reads four witness matrices. Minimal dense replay receives
only the resulting weights and upstream gradient, reconstructing the other
intermediates itself. Each replay control uses the faster of bounded-exact
binary64 BLAS and int64 execution.

| Batch / input / output | Projected audit | Minimal replay | Audit / replay | Witness / minimal output bytes |
| --- | ---: | ---: | ---: | ---: |
| 32 / 64 / 64 | 1.486 ms | 1.204 ms | 1.234x | 98,304 / 49,152 |
| 64 / 256 / 256 | 11.321 ms | 9.639 ms | 1.175x | 1,310,720 / 655,360 |
| 128 / 512 / 512 | 42.535 ms | 38.771 ms | 1.097x | 5,242,880 / 2,621,440 |

In the largest case, exact training took 14.698 ms. Producing the full commitment
took 33.423 ms versus 25.508 ms for minimal boundary outputs. The 5 MiB witness
has an ideal 100 Mbit/s payload-only transfer floor of 419 ms; minimal replay
needs half those output bytes. Framing, cold inputs, latency, retries and verifier
compensation add costs. Timing the final matrix check alone hides these costs.

The modular control took 111.515 ms in that case. Exact-integer projections are
substantially faster, but minimal ordinary replay remains faster for all three
shapes. This rejects an economy claim for this implementation and these shapes,
not every possible size or commitment/transport design.

## Earlier measurement and the fixed-point stop

The preserved [full-witness comparison](../config/experiments/neural-work-reference-results.json),
source `9162a4541e571b3ab224964d55219a107a3f276d`, measured 42.888 ms projected
verification versus 51.072 ms full-witness replay on the largest case. That 16%
improvement applies to the matching witness interface. It does not survive the
stronger control: replay need not receive the proof's internal witness. The
architecture decision uses minimal replay, and both records remain available.

The first modular-only driver, source `fa777a3`, stopped when the toy learner
repeated settled numerical work. Its original reporter did not save the partially
assembled timings. A diagnostic identified a quantized fixed point after ten
accepted steps. The revised driver saves partial results and stops there without
weakening duplicate-payment rules. This is a disclosed driver repair, not a
changed LLM acceptance threshold.

The synthetic training mean-square error fell from 522.133 to 273.799 over ten
accepted transitions, then rounding made the updates zero. There is no held-out,
retention, conversational or Transformer-training result. This illustrates why
payable computation and continued learning are different claims.

The earlier saved record's three honest and fifteen forged numerical transcripts
were reconstructed from fixtures and saved seeds; both verifiers agreed. Tests
also cover hand-calculated gradients, near-limit exactness, overflow, one-prime
aliasing, immutable challenges and cached-work reuse.

## Decision and the next construction

Retain the reference as an adversarial correctness oracle. This evidence does
not authorize a native upgrade or another GPU allocation. The next construction
must address **witness/commitment cost** and **fresh effort binding** together.
The proposed research contract is:

1. A task fixes recipe, graph parent, actual data and shard dependencies. Its
   numerical identity excludes worker names, mining headers and challenges,
   preventing relabeling from multiplying useful-work payments.
2. A separate mining statement binds a fresh chain challenge, claimant and an
   operation inside that task. Its witness must concern expensive intermediate
   execution, under an explicit hardness assumption. A cached correct output
   is insufficient. The mechanism below tests bindings but does not establish
   this resource-security statement.
3. Verification must bind operation inputs to the full graph, including nonlinear
   operations, backward computation and optimizer state. Tensor roots alone do
   not establish their origin. Useful results should survive a lost block lottery
   when their parent remains valid; competing-fork accounting needs explicit rules.
4. Quality and retention separately determine model promotion. A mining winner
   cannot choose the served checkpoint using a resource proof alone.

If only approved jobs confer block-making opportunities, their admission
authority can restrict miners. Public work availability, assignment and finite
funding therefore belong in the security analysis. Allowing arbitrary invented
jobs avoids that gate but no longer ensures contribution to the shared model.
Chain liveness when meaningful work is scarce also needs a rule; assuming endless
useful jobs does not solve it.

Native consensus can order and fund this research without equating audit receipts
with consensus weight. Neural-work block production requires a separate
resource-security argument, complete dependency binding and independent review.
Account registration supplies none of these properties.

## Intermediate mining mechanism

The next CPU step is implemented in
[neural_work_mining.py](../scripts/neural_work_mining.py), under the
[mining sketch plan](../config/experiments/neural-work-mining-sketch.json).
Source `1954dd3da5840256e3b0224eb91c6bc9edcc73f7` produced the
[complete mechanism record](../config/experiments/neural-work-mining-results.json),
including every winning witness, operation roots, measurements and verifier seed.

For each actual forward/backward matrix product, the context binds its numerical
job, operation, input tensor roots, worker identifier and chain challenge. Two
prime fields encode signed integers. Public challenge-dependent low-rank factors
produce `A' = A+E` and `B' = B+F`. The worker computes 4-by-4 intermediate products
`A'[I,K] B'[K,J]`; their hashes supply lottery attempts with no free receipt nonce.
Summing **all** tiles, including losers, recovers `A'B'`. Subtracting `AF + E(B+F)`
through the low-rank factors and applying signed CRT recovers the original `AB`.

The 16-by-16 toy training step executes all three real products through this
path: it is not a separate random matrix job appended to ordinary training.
Its forward result, both gradients and final weights exactly match unmodified
training. At the four-bit test target, 384 tile attempts yield **27 tickets**.
At a sixteen-bit target the same fixture yields **zero** tickets and still
produces the identical useful result. A changed test header changes the encoded
transcript without changing the recovered training transition.

The bundle verifier reconstructs the expected operation operands from the
admitted job and audited training trace, rather than trusting matrices attached
by the miner. Tested attempts to reuse tickets under another header, claimant,
operation or job fail. Rehashing an old partial product under a new header fails;
so do changed partials, free nonce/target fields, duplicated tickets within a
bundle, and coherent fabricated gradients. These are specific tested attacks,
not a proof that all shortcuts fail.

**The CPU sketch is about 103x slower to produce than ordinary training on this
tiny fixture.** It is a mechanism demonstration with Python tile/hash loops and
toy parameters, not an optimized cuPOW implementation or evidence for the paper's
asymptotic efficiency. The complete timing samples are retained. There was no
GPU allocation or NEURO issuance. The combined targeted suite has 47 tests.

Trust and security boundaries remain explicit:

- The test headers/worker identifiers are predetermined fixture bytes, not a
  demonstrated unpredictable chain beacon, authenticated public network or key
  admission system. Arithmetic-audit randomness is sampled separately after the
  committed training result.
- Ticket verification uses the actual input matrices, currently available to
  the verifier; it is not a succinct commitment-opening proof. The full linear
  arithmetic audit is still required. Graph-wide nonlinear dependencies remain
  outside this toy stage.
- The sketch adapts the intermediate-work idea with two fields and toy constants.
  It does not claim to implement the paper's full construction or inherit its
  hardness conjecture/theorem. No fair resource meter, ASIC/shortcut resistance,
  fork choice, cross-block spent set, public assignment or economic equilibrium
  is established by passing these tests.
- Zero matrices also yield resource tickets. An explicit test preserves that
  limitation: mining evidence cannot decide whether a job teaches the model
  anything. Admission and quality evaluation remain separate obligations.
- Cached useful results can coexist with recomputed mining transcripts. A proof
  of challenge-dependent work must not be advertised as proof that learning was
  new. Numerical deduplication and protocol-level eligibility are still required.
- Each fixed context offers only a finite set of tile attempts. The no-winner
  case proves output preservation, not continued block production. A consensus
  design still needs rules for fresh encoding attempts, admissible challenges,
  difficulty and progress when no available job yields a winning ticket. An
  encoding nonce would have to change the expensive computation, rather than
  merely rehash a cached receipt.

Reproduce from the committed checkout with the same NumPy environment:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  python -m pytest -q tests/test_neural_work_reference.py tests/test_neural_work_mining.py

OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  python scripts/study_neural_mining.py \
  --output .neuroshard/neural-work-mining/result.json
```

This gives the next research step an executable object to analyze: bind useful
training to intermediate-work tickets, attempt shortcut attacks, and reduce
complete proof cost before proposing a consensus change.
