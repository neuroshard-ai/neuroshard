# Staged integration: baseline stop and output calibration

The study on commit `42d316d` stopped before training. Parent development and
retention both scored **0/32** under the frozen exact-string, eight-token rule.
The contract required at least eight correct retention answers. No expert or
gate was trained, so this is not a result about staged learning or integration.

The isolated baseline took 90.47 wall seconds and 89.92 process CPU seconds,
with 1.06 GiB peak RSS. The study used the existing local CPU host, no GPUs.
Raw replies and identities are published in
[the baseline record](../config/experiments/staged-integration-baseline-result.json).
The original source, plan, data, and freeze remain unchanged.

The output contract was unsuitable. For example, `1 + 5 = 6` was rejected
because it was not exactly `6`; `9 + 16 = 2` was cut off at eight tokens.
Neither example licenses rescoring this frozen study as a pass.

## Calibration of a successor interface

Twenty separate operand pairs were fixed before generation: sixteen ordinary
additions and four modular additions. All are excluded from the successor.
A strict parser accepts one integer or one complete equation whose operands
and operation match the prompt. EOS is required; incomplete outputs, prose,
contradictory answers, and arithmetic repair are rejected. The generation cap
is 32 tokens. The parser does not consult the answer label.

| Calibration wording | Addition | Modular addition | 12/16 addition screen |
| --- | ---: | ---: | --- |
| Original instruction | 6/16 | 0/4 | Failed |
| Explicit system instruction | 7/16 | 0/4 | Failed |

The second arm used the same calibration cases, so the one-answer difference
is interface development, not held-out improvement. Both failed screens remain
failed. The longer outputs also exposed genuine arithmetic mistakes, including
`19 + 30 = 59`. Formatting alone does not explain the parent's weaknesses.
Total measured calibration process CPU was 88.75 seconds. There was no training.

[Calibration plans and raw replies](../config/experiments/staged-answer-calibration-results.json)
record both arms. The first calibration launcher had the same body without the
later `--plan` argument; its pre-generation hash is retained. Neural loading,
generation, and parsing sources were identical across the two arms.

The [separate staged-answering contract](STAGED_ANSWERING.md) chooses the explicit
system instruction and authorizes a fresh baseline. It does **not** count these
calibrations as a pass or permit training without the original 8/32 protected
baseline requirement. None of the old or calibration cases can be selected as
favorable retention examples for the next study.
