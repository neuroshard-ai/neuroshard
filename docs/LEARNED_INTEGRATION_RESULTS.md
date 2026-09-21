# Stage-1 learned-integration development result

**Status: failed development. Stop. Confirmation closed. No GPU.**
SmolLM2-135M last-layer expansion (one new expert plus a trained top-1 gate)
did not beat the matched no-expansion control on generated development answers.
The control produced 5/32 full-test successes. Expansion produced 0/32, matching
the untrained parent. All eight parent general responses were preserved.
Confirmation was not opened. This is not admission and not a 0.4.0 upgrade.

Machine-readable score:
[`learned-integration-stage1-result.json`](../config/experiments/learned-integration-stage1-result.json)
(`07d4ac47…`). Compact per-task record:
[`learned-integration-stage1-record.json`](../config/experiments/learned-integration-stage1-record.json)
(`bf17899c…`). Execution freeze `e87456a6…`. Method freeze `dc76e6ad…`.
Research contract `6801d1a2…`.

## Frozen question

> Does adding trainable capacity improve the complete answering system more
> than spending the same training budget on its existing capacity, while
> preserving earlier capabilities and meeting the same serving budget?

Success was generated executable answers, not lower loss. Development had to
pass before confirmation could open.

## Measured development gate

| Gate | Required | Measured |
| --- | --- | ---: |
| expansion vs control | control + 1 | **0 vs 5** |
| code retention vs parent | ≥ parent | **0 = 0** |
| general exact parent match | 8/8 | **8/8** |
| active experts per token | 1 | **1** |
| expansion p95 latency vs control | ≤ 1.5× and ≤ 90 s | **25.12 s vs 25.43 s** |
| expansion p95 memory vs control | ≤ 1.5× | **1.59 GiB vs 2.13 GiB** |

`passed` is false. `next` is `stop`. `confirmation_opened` is false.
`admission_evidence` is false. `gpu_launch_authorized` is false.

Parent 0/32. Expansion 0/32. Control 5/32 on tasks 517, 733, 807, 896, and 924.
Code-retention holdout: parent 0/16, expansion 0/16. Active last-layer MLP FLOPs
per token stayed 5,308,416.

## What this does not show

The control's five generated successes show that the same 128 teacher-forced
steps, data, and last-layer budget can change answers on this 135M setup.
Expansion staying at the parent's 0/32 and matching all eight parent general
texts is the measured expansion outcome. It does not prove that learned routing
cannot work. It fails this frozen method on this host.

Lower training loss is not success. Confirmation stays closed. The opened 64
leftover cases and the original 128-task final stay closed. Serving for the
1.7B research assistant remains the leftover incumbent extra. Stage 2 is not
authorized.

## Host

One CPU process, Intel Xeon Platinum 8259CL, four virtual CPUs. Wall-clock
6,187.781 seconds. Weights stayed local and are not in git.
