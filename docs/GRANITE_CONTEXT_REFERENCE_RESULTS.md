# Contextual assistant reference result

**Completed September 26, 2026; failed. Nothing admitted.** Source freeze
`386e07cd2257b85dd106686f3053ba0cc4126505`; exact-commit CI passed. The
[contract](GRANITE_CONTEXT_REFERENCE.md) is unchanged. All 64 cases are now opened.

| Complete answering path | Correct / 64 | p95 seconds |
| --- | ---: | ---: |
| Full-history retrieval | 29 | 4.63 |
| Parent rewrite | 35 | 7.11 |
| Published module rewrite | 35 | 8.52 |

The module gains six over history (paired block-bootstrap lower bound 0.03125),
but gains **zero** over parent rewriting (lower bound zero). It gets 7/32
contextual, 12/16 standalone and 16/16 unsupported requests correct. It misses
the required total, contextual and incremental-gain gates. There are no lost
control successes, and both models retain all 18 protected assistant answers.
Latency passes. Conditional reload replays were not run after quality failed.

## What the failure identifies

Full-history retrieval already contains the correct source for **48/48**
answerable cases. Parent rewriting supplies it for 40/48, and module rewriting
for 46/48. The module produces valid rewrite JSON on 64/64, versus 57/64 for the
parent; this does not create missing evidence or improve the final score.
This set cannot justify more work trying to gain evidence through rewriting.

Among the module's 29 failed answers, 19 are not JSON and ten are parseable
JSON with an incorrect structure, value or citation. Twenty-three contain the
expected literal somewhere in the reply. These are diagnostic counts, **not a
relaxed score**: copying the right value into prose is not the declared answer
with a valid source. Examples include extra spaces inside email addresses and
missing characters in citation IDs. Some replies are simply JSON strings.

The next implementation should make copying and serialization deterministic,
leaving semantic evidence selection to the model. That is an interface change,
not proof of useful neural growth. Evaluate it separately; preserve this failure.
No new adapter training is authorized by this result. A1 remains open.

## Evidence and resources

The [raw result](../config/experiments/granite-context-reference-result.json)
has SHA-256 `a2310db30fffea66983d1bd58318c650fffa34ce3db6f195de029951f4f660fd`.
An independent CPU rescore matches. The
[report](../config/experiments/granite-context-reference-report.json) records
individual failures, inputs/outputs, aggregate work and cleanup verification.

The worker ran 1,230.61 wall seconds. Conservative instance time was 1,327.85
seconds, **$0.39039 compute**, with storage/transfers separate. Instance
`i-0846312eff33f5485` retired at 21:28 UTC. Independent AWS checks confirmed it
terminated, its tagged volumes absent and its security group deleted. No GPU,
training, native-network change or automatic retry occurred.
