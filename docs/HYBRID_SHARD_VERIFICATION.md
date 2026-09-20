# Cheaper shard verification: one bounded attempt

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
