# Append-only execution: oracle gates passed, assistant failed

The CPU run on commit `a817a76` finished on September 24, 2026. Its own gates
passed. **That pass is not a deployable assistant.** Nothing is admitted, no
NEURO is issued, and public 0.4.0 is unchanged.

| Arm | New answers | Retention |
| --- | ---: | ---: |
| Parent | 0/64 | 4/64 |
| Added blocks | 9/64 | 1/64 |
| In-place control | 9/64 | 3/64 |
| Constant training label `6` | 9/64 | — |

The selector in `append_only_growth.py` chooses a reply by reading
`parent["passed"]` and `added["passed"]`. Those flags are comparisons with the
hidden answer. A conversation does not have them, so preservation was applied
by the evaluator. The added blocks did not beat the most common training
label: answering `6` for every new question is also 9/64. The expert's actual
texts were mostly `5` (28) and `1` (20), not a single constant, and still
landed on that trivial score.

Replies and counts are in
[`append-only-execution-result.json`](../config/experiments/append-only-execution-result.json).
The execution freeze and its oracle scorer stay as the historical method. The
next contract is [observable selection](OBSERVABLE_SELECTION.md). It cannot
read the hidden answer.
