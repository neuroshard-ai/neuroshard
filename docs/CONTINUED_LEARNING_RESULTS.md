# Continued-learning result: generated answers did not improve

The fixed 1.7B continuation completed all 96 updates on 14 September 2026 and
**failed** its frozen quality gate. Lower response loss did not produce better
answers. This candidate is not promoted, earns no research-run NEURO and does
not authorize settlement implementation.

## Result

The [frozen contract](CONTINUED_LEARNING.md) required at least eight net new
correct answers on the 256-case primary set, exact one-sided McNemar
*p* < 0.05, no per-family accuracy loss on either task set, and a conversation
retention upper bound of at most +0.02 nats.

| Measurement | Parent | Candidate | Decision |
| --- | ---: | ---: | --- |
| Correct primary answers, 256 cases | 192 | 191 | Fail: zero wins, one loss |
| Correct prior-task answers, 128 cases | 96 | 95 | Fail: zero wins, one loss |
| Primary response loss | 0.144141 | 0.136006 | Measurement only |
| Prior-task response loss | 0.145954 | 0.138349 | Measurement only |
| Conversation retention loss | 0.542989 | 0.543930 | Pass: change UCB +0.001764 nats |

Primary lookup and filtering stayed at 64/64 each. Sorting fell from 64/64 to
63/64. Invoice totals stayed at 0/64. On the prior-task set, invoice totals fell
from 2/32 to 1/32; the other family counts were unchanged. The primary paired
test has *p* = 1.0. These are complete final sets, not selected examples.

All three development retention checks passed: mean changes were +0.002260,
+0.001841 and +0.002201 nats at steps 160, 192 and 224. Those checks neither
established generated-answer gain nor overrode the final failure.

## Executed computation

Three `g5.2xlarge` A10G workers jointly trained one model with boundaries
`[0, 6, 15, 24]`. Each held its own student partition, Adam state and frozen
reference partition. Owned parameter counts were 503,343,104 / 604,016,640 /
604,016,640; peak allocated GPU memory was 12.65 / 12.97 / 12.97 GiB. No worker
held the complete training state.

The training phase took 1,991.00 seconds, including its checkpoint and retention
checks. Parent and candidate final evaluations took 365.48 and 357.45 seconds.
Every worker completed successfully and produced the same common checkpoint
and evaluation measurements. These are cooperating workers under one operator,
not independently owned replications.

The implementation repairs made before execution enforce trained replay
membership, exclusion by rendered content, the committed freeze on resume,
the actual final checkpoint lock and the development abort. Validation passed
412 tests, repository checks and wheel/sdist content checks. Both GitHub CI
runs passed for the final selection commit.

## Frozen artifacts

- Plan and numerical source reference: `af16131`.
- Prepared inputs committed at `d8cf1d9`, identity
  `1599f1ad67e37fb04e2908b32a544529a76d017d9d0f39c2969a9dc8e8ac9251`.
- Actual endpoint selected at `887bb059e21d197311705cbdfee75ccc19245bc9`,
  before final evaluation started.
- Parent checkpoint:
  `094138fb6e3e2a8962f8455b0bf81de3f2fbe82029222df76aa71e2e22e14d49`.
- Candidate checkpoint:
  `0af8dd9e03917a44257cb76282515bcf6ee67082bb7817b77cb0eead3785f523`.
- Candidate learned-state root:
  `b1fa7229b23708a8a8e72c08f02fd26be611cae0768858085e8bfe78af04dcd6`.

Each of the three new checkpoints, including its complete Adam state, was
uploaded directly from its owners to S3 and verified by full readback. The
pre-cleanup evidence archive has SHA-256
`d7e064e765a88e0f7d96c538c8cea1aafd21721331bee9a3f12f32516da20536`.
It contains prepared data, logs, evaluations, numerical sources and checkpoint
backup receipts. Large tensor objects are stored separately. This is
operator-controlled storage, not a demonstration of public artifact availability.

Reproduce the frozen run from its source commits; later numerical changes must
not be substituted for those bytes. Keep the plan in `selection-committed`
status for verification of the recorded final reports. This document records
the failed outcome without changing the original contract.

## Implication

The distributed computation works for this numerical profile. This learning
recipe does not solve useful continual learning. Its final examples are now
exposed and cannot serve as fresh finals for another recipe. Any subsequent
method needs its own frozen inputs and acceptance decision. No serving root or
public 0.4.0 state changed.
