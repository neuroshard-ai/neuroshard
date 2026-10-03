# Fresh reference: deterministic rejection and execution recovery

**Result:** the [recovered baseline](MODULAR_REFERENCE_FRESH_RESULTS.md) finished
11/24 and failed the unchanged quality gate. This study is closed and its
temporary resources are retired. The contract below is retained as frozen.

The CPU study on `f1e384b58c8b8bc5dbfeba2c7afa029a0a05880f` stopped
after nine baseline generations, eight of which entered the progress record.
CI had passed. No replay or modular checkpoint preparation began. The stopped
study remains closed; this amendment authorizes one new study after exact-commit
CI. It grants no assistant checklist credit.

## Failure and repair

The ninth reply used `include_archived=false` in a tagged Python function call.
The frozen wire format requires a Python literal, so this answer was wrong.
Both processes rejected it, but `ast.literal_eval` included an AST object's
memory address in the error. The controller compared the different error
strings and aborted with `reply rescore mismatch`.

The parser now emits `argument must be a Python literal` for this error. It
still rejects lowercase `false`, expressions and executable calls. Expected
answers, prompts, tool definitions and acceptance rules are unchanged. Tests
score the actual reply in three independent processes, preserve the pass/fail
status of all nine generated replies, reject forged validation receipts, and
exercise complete replay of wrong answers without stopping the comparison.

The [unaltered failed result](../config/experiments/modular-reference-fresh-interrupted-result.json)
and [raw worker/resource receipts](../config/experiments/modular-reference-fresh-interrupted-receipts.json)
preserve the original error strings and accounting. The previous allocation's
instance, volume and security group were retired.

## Frozen recovery

The [new execution amendment](../config/experiments/modular-reference-fresh-recovery-execution.json)
uses profile `fresh-reference-recovery` and the **byte-identical 24-case plan**,
artifact inventory, runtime, decoding, task order and quality gates. Start from
an empty study directory and regenerate every primary answer. No reply is
imported from the stopped binding. Replay every answer independently if the
baseline usability gate passes, then run the modular comparison and its replay.

Nine cases have now been opened. This is a declared harness recovery with a
fixed method, not a new holdout or nine additional independent observations.
The original 128-task programming final and opened 64-case programming study
are not involved. There is no training, GPU, network activation or promotion.

The original gates remain: baseline at least 16/24 and 4/8 in every category;
then exact replay and paired comparison. Growth additionally requires at least
two gained answers, zero lost baseline successes, and both latency gates. A
failed baseline stops before downloading the modular weights. Any further
execution failure stops this allocation without an automatic retry.

## Carrying the bill forward

| Recorded work | Charged amount |
| --- | ---: |
| Evaluation before the interrupted fresh study | 11,328.797205 seconds |
| Nine interrupted-study primary workers | 259.160461 seconds |
| Cumulative baseline evaluation before recovery | 11,587.957667 seconds |
| Interrupted-study preparation | 143.488572 seconds |
| Retired instance time | 526.651463 seconds |
| Retired instance compute | $0.154836 |

The baseline has **6,940.839539 seconds** left in the original 7,200-second new
evaluation envelope; the modular checkpoint retains 7,200 seconds. Baseline
preparation has **3,456.511428 seconds** left from 3,600. Current and prior work
appear separately and cumulatively in the result. Earlier diagnostic preparation
and the original unknown download remain disclosed in their existing records.

The [recovery resource contract](../config/experiments/modular-reference-fresh-recovery-resources.json)
allows one `r7i.4xlarge`, 160 GiB gp3, for at most **7.8 hours**. The allocator
verifies the pinned retirement receipt and rejects a still-live old allocation
or combined time exceeding eight instance-hours. Prior compute, new maximum
compute and $3 storage/network headroom must fit the original **$15 planning
cap**. Storage and transfer charges remain separate from the compute estimate.
Both expiry guards, evidence copy and confirmed retirement remain enabled.

Commit and push this amendment, then start `scripts/modular_reference_cloud.py`
from a detached checkout at that commit in a persistent user service. It waits
for successful CI before allocating. The latest local handoff is
`.neuroshard/modular-reference-fresh-latest.json`; its study home contains
`status.json`, `result.json`, `evidence/` and `resources-finished.json`.

This run answers whether the published modular reference improves the same
assistant tasks without regressions at bounded latency. A usable reference is
the prerequisite to training our own contributed capability under A2; completion
of this execution alone does not prove a growing decentralized assistant.
