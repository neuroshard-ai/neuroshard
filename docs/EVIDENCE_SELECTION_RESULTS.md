# Evidence-selection diagnostic result

**Completed September 26, 2026; diagnostic passed. A1 remains open.**
Freeze `01feddc41e5bc6204268017796932a0a1ba64690` passed exact-commit CI.
The [contract](EVIDENCE_SELECTION.md) used the unchanged parent on 64 already
opened cases. No module was trained or evaluated for new capability.

| Path | Correct / 64 | p95 seconds | Output tokens |
| --- | ---: | ---: | ---: |
| Direct answer and citation generation | 59 | 4.711 | 1,445 |
| One evidence choice, then exact copying | 64 | 0.945 | 64 |

Selection scored 32/32 contextual, 16/16 standalone and 16/16 unsupported
requests. It lost no correct direct answer. All 18 previously successful
assistant anchors remained correct, and all six independent reload replays
matched. An independent rescore verified the receipts, answers and frozen gates.

Both arms received explicit value options and the instruction after the quoted
conversation. Direct generation improved from the earlier study's 29/64 to
59/64 under that revised interface. The controlled gain attributable to selection
in this diagnostic is **five answers**, not 35. Exact copying prevents altered
values and citations; semantic evidence selection can still fail on other cases.

This is development evidence for the executor. It cannot establish neural
growth, fresh generalization, A1 completion, permissionless verification or a
public assistant. A valid receipt proves copying from supplied bytes, not source
truth or correct neural execution. The next comparison needs fresh cases and an
actual published neural module against both the selector and a matched parent
control. Previous failed references remain failed.

The [raw result](../config/experiments/granite-evidence-diagnostic-result.json)
has SHA-256 `8d1af0cb6423c4d349fb605ad1cd51cea642c2a41f23548fad871a31f4fed713`.
The [report](../config/experiments/granite-evidence-diagnostic-report.json)
records work, CI, replay verification and cleanup. Worker time was 415.52 seconds;
conservative instance time was 512.76 seconds, **$0.15075 compute**, with
storage/transfers separate. Independent AWS checks confirmed instance
`i-097b0c541eff9b587` terminated and its volumes/security group absent.
