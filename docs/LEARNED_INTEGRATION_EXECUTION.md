# Stage-1 learned-integration execution freeze

**Status: CPU execution freeze recorded. Not run. No GPU. Confirmation closed.**
This freeze binds the smallest learned-integration run: SmolLM2-135M-Instruct,
last decoder layer, single process on CPU. It records seed file hashes and eight
general-retention identities before any generated answers exist. It does not
open confirmation. It does not authorize a GPU launch. It is not a 0.4.0
upgrade and not the 1.7B four-owner runtime.

Machine-readable freeze:
[`learned-integration-execution.json`](../config/experiments/learned-integration-execution.json).
Method freeze remains
[`learned-integration-method.json`](../config/experiments/learned-integration-method.json).
Research contract remains
[`learned-integration.json`](../config/experiments/learned-integration.json)
(`6801d1a2…`), with `train` false and `gpu_launch_authorized` false.

## Host

CPU, one process. `gpu_launch_authorized` is false. Sparse activation is not a
latency result. Stage 2 stays unauthorized.

## Seed

`HuggingFaceTB/SmolLM2-135M-Instruct` revision `12fd25f77366fa6b3b4b768ec3050bf629380bac`.
File hashes are the published seed pin in `src/neuroshard/evolution/seed.py`,
including `model.safetensors` `5af571cb…`. Weights are not in git.

## General retention

Eight conversations from the programming-expert Smol-SmolTalk test parquet
(`be6773dc…`), two from each group, selected with seed `20260921` from
programming-expert training-general rows and excluding that trial's development
and final general IDs. Identities are `identity({dataset, row})`. Parent 135M
responses have not been generated. Scoring remains exact match to those future
parent texts.

## What this freeze authorizes

After this freeze is committed, a CPU run may train the new last-layer expert
and gate versus the matched no-expansion control, then score development and
code retention. Confirmation stays closed until development passes. Missing
local 135M files block the run; they do not license a GPU.

`scripts/run_learned_integration.py` still refuses. Use
`scripts/run_learned_integration_stage1.py` to bind this freeze.
