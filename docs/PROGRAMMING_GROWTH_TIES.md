# TIES composition of two programming tails

**Status: composition frozen; extras not yet decoded.**
Unit task-vector addition failed because the merged extra was often not even
valid Python (18/38 extractable versus 38/38 for each frozen tail). On the
frozen last-four-layer tensors, **33.6%** of jointly nonzero signed parameters
have opposite signs. This experiment replaces that sum with TIES: keep the top
20% magnitude of each delta, elect a consensus sign, and average only the
values that agree. It does not train. It does not open the original 128-task
final. It is not a selector.

Serving stays one extra decode. Parent public-example pass still returns the
parent. Failure spends that extra on the TIES mix instead of the unit mix.

## Frozen rule

`θ = θ_parent + mean({trim_0.2(θ_inc − θ_parent), trim_0.2(θ_add − θ_parent)}`
restricted to the elected sign). Keep `0.2` and scale `1.0` are taken from the
TIES paper defaults and are not tuned on the opened 64 cases.

CPU merge of the frozen checkpoints is pinned in
`config/experiments/programming-growth-ties-expert.json`
(`9c06f100…`). A GPU worker must reproduce those hashes before decoding.

## Screen (opened 64, not admission)

Generate the 38 extras whose parent public example already failed. Join the
saved parent, incumbent and added traces. Required:

| Gate | Required |
| --- | ---: |
| TIES extras extractable | 38/38 |
| unique added recovered | 3 |
| incumbent successes preserved | 29 |
| old successes preserved | 12 |
| selected full-test | 32/64 |

A pass is only eligibility for a separate 511–600 confirmation freeze. Failure
stops this composition. No native issuance.

## What this is for

The vision is a served skill that grows when another shard joins. Routing the
two tails from the question/parent text failed three times. This tries the other
honest door: compose the shards in weight space without the interference that
broke code syntax.
