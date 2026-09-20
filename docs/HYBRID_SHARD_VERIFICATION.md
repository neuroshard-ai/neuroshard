# Cheaper shard verification: one bounded attempt

**Result, September 20, 2026: cost target failed; this candidate is stopped.**
Exact arithmetic and the adversarial checks pass. At the primary shape the
median paired audit-cost ratio is **1.265**, against a required maximum of 0.5;
modeled latency is **1.270**. No shape or accounting scenario passes. See the
[raw record](../config/experiments/hybrid-shard-verification-results.json) and
[measured results](#measured-results). Economical verification remains open.

## Goal and scope

The product goal remains a shared assistant supported by contributed hardware.
Extra peers first supply replicas, serving, audits and funded training. A request
uses a bounded set of backbone shards and experts. Growth requires measured
quality and retention within a resource budget; it is not triggered by peer count.

This experiment addresses one enabling problem: reduce the **complete additional
cost of verification** to at most half of minimal replay. Ordinary useful work is
common to both methods and is reported separately, including work-plus-audit
totals. Halving audit overhead does not mean halving the total training bill.

The [frozen contract](../config/experiments/hybrid-shard-verification.json) fixes
one candidate, its control, shapes, assumptions, accounting and stop rule before
timing. This is CPU research on the existing exact linear training profile. It
does not change native BFT, issue tokens, deploy GPUs, change the TODO completion
criteria or establish cheaper verification for the floating-point LLM.

## Why this candidate

The [previous result](NEURAL_WORK_RESULTS.md) loses to minimal replay: the
projected verifier consumes intermediate matrices that replay can reconstruct.
Faster inner checks alone do not repair that communication bill.

The candidate deliberately recomputes the forward product. This removes its
witness entirely and determines the residual from authenticated inputs. It then
checks the two backward products with randomized projections. With the existing
notation:

```
F = X W                       # recomputed by the auditor
E = round(F / S) - T
G = X^T E                     # checked by projections
H = E W^T                     # checked by projections
Q = W - W' = round(G / d)     # d = S * batch * learning_rate_denominator
R = G - d Q
```

The mandatory outputs are `W'` and `H`. Instead of sending `G`, send a compact
rounding remainder. Define `sign_Q = -1` when `Q < 0`, otherwise `+1`, and encode
`U = sign_Q * R + floor(d/2)`. Correct nearest-integer rounding with ties away
from zero gives `0 <= U < d`. The receiver reconstructs `G` and checks both its
range and `round(G/d) == Q`, including the special zero-quotient tie boundary.
An invented gradient consistent with the optimizer must still pass `G = X^T E`.

`U` needs `ceil(bit_length(d-1)/8)` bytes per entry. All boundary outputs use
canonical little-endian signed 32-bit integers, **for both candidate and replay**;
their declared ranges fit. The candidate has less extra traffic than the old
full witness, but still more than minimal replay. Packing is a representation
change, not a new cryptographic theorem or a claim of novelty.

Both backward identities use ten independent byte-vector projections with
bounded exact binary64 intermediates, as in the existing reference. For a fixed
invalid product the ideal-vector false-acceptance bound is `256^-10`; conservatively
union-bounding two product checks gives at most `2 * 256^-10`. This assumes the
entire payload was fixed before the auditor's fresh seed. Hash/XOF security and
honest independent challenge generation remain assumptions. This is not a
publicly verifiable proof or evidence of fresh physical computation.

## Cost contract and stop rule

The primary workload is the maximum batch/width already admitted by the current
profile: `512 x 512 x 512`. The three previous shapes remain mandatory controls.
The asymptotic hypothesis is one dense forward product plus quadratic checks
instead of three dense products. It needs enough arithmetic relative to hashing,
conversion and communication to be useful; small cases may lose.

The primary deployment assumption is an honest auditor in a local compute group,
with cold inputs, a modeled 10 Gbit/s link, 1 ms challenge round trip and no
per-byte charge. This is compatible with separately administered nearby machines
but does not demonstrate them. A cold 100 Mbit/s charged-WAN scenario is mandatory
negative coverage. Prices are hypothetical fixed accounting inputs, not AWS quotes.

For each of seven paired samples, record:

- Producer encoding and commitment CPU/wall time for the actual payload.
- Auditor validation, input hashing, decoding, authentication and all arithmetic.
- The faster of optimized minimal replay and the existing reference replay.
- Complete protocol bytes, including cold inputs and challenge/commitment messages.
- Accounted cost: CPU seconds at the frozen rate plus charged traffic.
- Serialized latency model: producer and auditor wall time, payload transmission
  and the additional challenge round trip. This is not a WAN measurement.
- Ordinary training and combined useful-work-plus-audit cost, so the common
  training cost is visible rather than mistaken for a saving.

No precomputed admission hash or input conversion is free. Cached-input cases
still pay their per-job validation and hashing; only previously delivered input
traffic is excluded. Both paths use the same boundary representation and prices.
There is no slower integer-only baseline used to manufacture a speedup.

A pass requires every primary paired sample to achieve **both** at most 0.5
accounted cost and at most 0.5 modeled latency, with all correctness checks passing.
Preserve failures and all other scenarios. A miss stops this candidate. Even a
pass establishes only conditional arithmetic savings, not economical public
verification. A full market bill also needs measured availability, storage,
dispute funding, settlement and independent auditor incentives.

## Security and numerical blockers kept separate

The experiment assumes one honest auditor, immutable job inputs, an immutable
pre-challenge payload and evidence availability. Missing or malformed evidence
rejects. Correct receipts are not proof that an auditor did the work. The existing
[funded-audit design](FUNDED_AUDITING.md) supplies obligations, but independent
selection, collusion resistance and fully financed disputes remain unresolved.

The exact integer profile is essential. Floating-point matrix products do not
obey the reassociation identity used by Freivalds checks. For example, in float32,
`A = [2^24, 1]`, `B = [[1, -1], [1, 0]]` and `r = [1, 1]^T` can give
`(A B) r = 0` but `A (B r) = 1` for an honest result. Widening a tolerance would
change the correctness statement and give an attacker an error budget.

Consequently no result here certifies the current Transformer, its attention,
normalization, nonlinearities, clipping, optimizer state or inter-shard
dependencies. A later design must cover the actual numerical recipe or propose
a separately evaluated exact recipe. Full replay remains the oracle.

[Slalom](https://arxiv.org/abs/1806.03287) studies specialized linear verification
in a trusted-hardware inference setting. [Verde](https://arxiv.org/abs/2502.19405)
provides graph dispute resolution under an honest-provider assumption.
[zkDL](https://arxiv.org/abs/2307.16273) studies proofs that include nonlinear
training. These are prior methods with different cost and trust models; none of
their end-to-end guarantees transfers to this experiment.

## Execution sequence

1. Commit this contract before implementation measurements.
2. Implement the candidate and strongest matching replay control; check them
   against the independent int64 reference and explicit forgeries.
3. Commit the exact tested driver and source before running the frozen study.
4. Publish the raw record and verdict. Do not relax a failed target or authorize
   additional infrastructure from this result.

## Measured results

The contract was committed as `fce733b` and tested implementation as
`f32db5cfab84fdd8c5fdbc78084c1fbb182a06a2`, before measurement. The driver refuses
uncommitted source, preserves partial failures and refuses to overwrite evidence.
The [complete record](../config/experiments/hybrid-shard-verification-results.json)
has SHA-256 `6233c6275052ae0326fd6eab4b7fce1e5a5af7578bdc8b7e72cce5649c08d140`.

Existing Intel Xeon Platinum 8259CL host; Python 3.10.12, NumPy 2.2.6, OpenBLAS
0.3.29, one requested thread. Seven paired samples after one warmup, with reversed
measurement order on alternate samples. This is one shared CPU host, not an
isolated hardware population or a measured network. Prices and link conditions
are frozen hypothetical scenarios, not actual AWS charges or a market price.

| Batch / input / output | Hybrid audit CPU, ms | Minimal replay audit CPU, ms | Local audit cost ratio | Local modeled latency ratio | Charged WAN cost ratio |
| --- | ---: | ---: | ---: | ---: | ---: |
| 32 / 64 / 64 | 2.343 | 1.700 | 1.374 | 1.946 | 1.152 |
| 64 / 256 / 256 | 21.298 | 13.914 | 1.548 | 1.602 | 1.191 |
| 128 / 512 / 512 | 82.438 | 52.478 | 1.545 | 1.548 | 1.191 |
| **512 / 512 / 512** | **117.876** | **92.899** | **1.265** | **1.270** | **1.153** |

Columns report separate medians; ratios are medians of paired ratios, not ratios
of column medians. CPU totals include producer input preparation, output encoding
and commitment, and the complete auditor operation. Both displayed scenarios use
cold inputs. The fastest control is selected per sample and metric. Cached-input
results and every raw phase timing are also published.

In the primary case the candidate sends 2,883,592 output-witness bytes versus
2,097,160 for minimal replay: **37.5% more**, despite removing the forward witness
and compressing the weight-gradient remainder. Both receive the same 3,145,756
input bytes in the cold case. Commitments, identities and the random challenge
are counted separately in the record. Median producer output preparation is
19.663 ms for the candidate versus 7.761 ms for the boundary-only control.
The candidate's auditor also remains slower. Packing and checking overhead
outweigh the saved dense products in this implementation and these shapes.

Adding ordinary useful training gives a primary combined cost ratio of **1.202**,
also a loss. Even free transfer does not meet the goal; switching the hypothetical
tariff cannot turn the measured local CPU result into a pass. Larger matrices,
GPUs, different encodings or another numerical profile were not tried afterward.

All 28 recorded honest executions agree with the independent int64 oracle.
All 32 frozen malformed/forged claims are rejected, including optimizer-consistent
invented gradients. The combined targeted suite passes **76 tests** (29 for this
candidate, 47 for the existing arithmetic and mining references). The float32
counterexample produces 0 versus 1 exactly as stated above. These tested attacks
do not prove resistance to all attacks or solve auditor collusion.

No new EC2 resources, model training, quality holdouts, token payments or native
protocol changes occurred. The six-item TODO is unchanged. The code is retained
as a rejected, reproducible research candidate; it is not an activated verifier.
A future proposal needs an explanation for beating these complete costs and a
prospective contract. A better inner matrix benchmark alone is insufficient.

Reproduce at the frozen source commit:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  python -m pytest -q tests/test_hybrid_shard_verifier.py \
  tests/test_neural_work_reference.py tests/test_neural_work_mining.py

OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  python scripts/study_hybrid_verification.py \
  --output .neuroshard/hybrid-verification-reproduction/result.json
```
