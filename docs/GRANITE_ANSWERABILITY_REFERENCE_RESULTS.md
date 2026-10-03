# Fresh answerability reference result

**Finished September 26, 2026 at 23:23 UTC; failed. Independently verified
September 27. Nothing admitted; A1 remains open.** Freeze
`a327e7e2d37aa42fac44f530cce7eef3fc6c8ed6` passed exact-commit CI. The
[contract](GRANITE_ANSWERABILITY_REFERENCE.md) and its gates remain unchanged.
All 80 prospective cases are now opened.

| Complete path | Fresh correct / 80 | p95 seconds |
| --- | ---: | ---: |
| Selection alone | 41 | 1.027 |
| Selection plus parent checking | 69 | 2.254 |
| Selection plus published module checking | 74 | 2.580 |

The module gains five over parent checking, with no lost fresh control successes.
It gets 39/40 supported and 35/40 unsupported requests correct. The gain and
category thresholds pass. However, **all five additional successes belong to
the same `metric` template family**. The predeclared bootstrap across eight
families gives a lower 95% gain bound of **zero**, which fails the strictly
positive threshold. Five variants of one pattern do not establish a broadly
repeated advantage.

## Preservation failure

The module keeps **62/64** previously correct lookup answers. The original
assistant anchors remain **18/18** correct in both checkpoints. The failures are:

| Opened task | Relevant conversation | What happened |
| --- | --- | --- |
| `context-03-first` | Ask for the entry code at the first of two named stops | Selector finds the correct source/value; checker rejects it as unanswerable |
| `context-10-first` | Ask for the entry code needed before lunch | Selector finds the correct morning-stop source/value; checker rejects it as unanswerable |

The base selectors match exactly on all 144 source tasks. Requested adapter
activation is recorded, and the generated finite decisions match their receipts.
The observed failure occurs when the added checker vetoes an already correct
selection on a contextual request. It is not evidence that the base weights
forgot those answers or that source copying failed. The run does not identify
the internal reason for the checker's mistake.

Both checked paths execute 158 model calls on the fresh cases, with identical
53,959 input tokens. Parent checking emits 499 tokens and module checking 504.
Selection alone reuses the parent first stage in the laboratory; its serving
cost is 80 calls and 80 output tokens, not additional executed work. Latency
passes: the module's p95 is about 1.14 times parent checking.

## Decision and evidence

Reject this candidate under the frozen uncertainty and preservation gates.
Conditional replays did not run. No training, promotion, native-network change
or new allocation follows this result. The observed gain is a limited signal
for the module, not a passing growth milestone. The earlier 64/64 diagnostic
proved its opened interface cases; the new selector score of 41/80 exposes its
limits on scoped and conditional documents.

The next design review should address preservation of conversation context when
modules are composed. Any new serving policy requires its own prospective
contract and fresh evaluation. Do not bypass the checker for these two task IDs,
relax the lower bound, or tune on these 80 cases and call them new evidence.

An independent CPU rescore matches the saved report and verifies the execution
binding against CI. The [raw result](../config/experiments/granite-answerability-reference-result.json)
has SHA-256 `81099c5f68fb37c3fc961f1c6a74a3b2efd389e0e7d4cacea2049cce10016d93`.
The [report](../config/experiments/granite-answerability-reference-report.json)
records per-case gains/losses, work and resource verification.

Parent and modular worker peaks were 7.92 and 9.01 GiB respectively; each ran in
its own process. Full checkpoint inventories were 6.82 and 8.31 GB. Controller
wall time was 840.21 seconds. Conservative instance time was 934.84 seconds,
**$0.27484 compute**, with storage/transfers separate. The retirement receipt
reports instance `i-0bdde1639416c76a3` terminated. Subsequent AWS verification
finds no returned instance record, no tagged volumes and no security group.
