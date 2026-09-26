# BAR dense baseline diagnostic — September 26, 2026

The original A1 baseline finished at **03:12:54 UTC** under source freeze
`25b3976`. It scored **4/9** and failed the declared usability gate because
tool use had no successful task. A1 remains open. The amended execution and
BAR-5x7B comparison were not started.

| Category | Passed |
| --- | --- |
| Conversation | 3/3 |
| Instruction following | 1/3 |
| Tool use | 0/3 |
| Total | 4/9 |

The original plan requires at least one success in every category. The nine
scores were recomputed from the saved replies with the unchanged scorer and
matched exactly. This was CPU inference only; no model was trained or promoted.

## What failed

- `instruction-prime`: answered `no` when asked whether 17 is prime.
- `instruction-two-lines`: generated both requested words but added two spaces
  before the newline. That fails the frozen exact-format rule.
- `tool-weather`: generated prose about lacking real-time capabilities and
  reached the 32-token limit without a valid function call.
- `tool-add`: generated `call add(12, 30)`. It selected the requested function
  and values, but did not emit the required tagged call with keyword arguments.
- `tool-balance`: described using the lookup function in prose and reached the
  32-token limit without a valid call.

The immediate blocker is valid tool calling through this model/interface and
generation policy. These observations do not identify whether a documented
upstream interface, a different foundation, or further adaptation is needed.
The next decision must audit that interface before another execution contract.
The questions are now opened diagnostic cases; changing their scoring or prompts
cannot turn this result into a passing frozen run.

## Evidence and cost

The [original result](../config/experiments/modular-reference-a1-legacy-baseline-result.json)
is preserved byte for byte. Its SHA-256 is
`a087a02063ad3b727b423238fab90fe1d25d523d5d994ff1724890931f6c4322`.
The unchanged plan hash is
`773d5d59403554e3c9713e8636d235d3e39287d59c6e075fed30c26c88aab7e4`.

Evaluation took **5,203.386 seconds (86.7 minutes)** on the existing CPU host.
The sum of recorded generation times is 5,130.053 seconds. Highest recorded
worker peak RSS was 6,576,115,712 bytes (6.12 GiB). Historical download time was
not recorded. No additional instance or GPU was launched.

The original runner lacked complete reply/input binding and independent replay.
This result therefore remains a **legacy diagnostic**, not completed A1 evidence.
The [execution amendment](MODULAR_REFERENCE_EXECUTION.md) at `adafb19` passed CI
and its 24 focused tests, but was not run against these models. Repeating the
known-failing baseline unchanged or downloading the larger checkpoint would
not resolve its failed starting condition.
