# BAR decoder agreement audit

**Result:** [decoder agreement passed](MODULAR_DECODER_PARITY_RESULTS.md):
65/65 logit checks and 24/24 generated answers matched. The 11/24 quality failure
stands. The audit is closed and its temporary resources are retired.

The [fresh reference baseline](MODULAR_REFERENCE_FRESH_RESULTS.md) failed at
11/24. That result remains closed. This audit distinguishes an execution
discrepancy from the model/interface's observed capability before another
foundation decision or training run. It grants no quality or checklist credit.

## Local check

The small Olmo2 and FlexOlmo tests now serialize their weights into sharded
safetensors and independently reload them through Transformers and our layer
streamer. Float32 and bfloat16 prefill/cached decoding produce equal logits.
This exercises the real weight index and loader rather than borrowing weight
tensors and the final norm from the oracle object.

An initial exploratory test rounded rotary-position buffers by calling
`.to(bfloat16)` on an already constructed reference model. That produced a
difference not present when using `from_pretrained(dtype=bfloat16)`, the actual
upstream loading path. The corrected test retains FP32 rotary buffers. No
production decoder correction has been justified by the local check.

## Committed checkpoint experiment

The [execution contract](../config/experiments/modular-decoder-parity-execution.json)
pins the original 24 questions, failed result, BAR-7B revision and artifact
hashes, runtime packages and six upstream Transformers source files. Model
weights, prompts, tokenizer/template, 128-token cap, greedy rule and scoring
stay fixed. All cases are already opened; this is implementation diagnosis.
BAR-5x7B is not downloaded or evaluated.

On one host, load the baseline independently through
`AutoModelForCausalLM.from_pretrained` and our `Checkpoint` loader:

1. Compare the complete logit tensors at prefill and every saved continuation
   position for `fresh-inventory`, `fresh-tool-documents` and
   `fresh-tool-reschedule`. These cover an incorrect ordinary answer, an invalid
   tool call and a correct longer call. Each implementation has its own KV cache.
   Require finite, elementwise-equal logits and matching argmax; record numerical
   differences and hashes. Stop on the first disagreement. Saved tokens provide
   identical prefixes solely for this diagnostic.
2. If logits agree, run standard `model.generate` on all 24 original prompts.
   Require the recorded token IDs, text, termination, prompt hash and score to
   match. Expected answers and saved replies never enter these generation inputs.
   `logits_to_keep=0` keeps the same projection shape as the streamed path;
   sampling is disabled, beam count is one and KV caching is enabled.
3. On the first generation disagreement only, perform one additional streamed
   generation of that same prompt on the current host, inside the same worker
   budget. Record whether it matches upstream, the old reply, or neither, then
   stop. There is no prompt change, parameter search or additional attempt.

A measured disagreement is a completed diagnostic with `decoder_agreement=false`.
An interrupted worker, changed source/artifact or exceeded resource bound is an
execution failure. An empty or partial report cannot claim agreement.

| Outcome | Next decision |
| --- | --- |
| Full agreement | The failed baseline is reproduced through the standard implementation on these cases. Reconsider the foundation/interface before growth training. |
| Logits differ on the same host | Locate the numerical/execution difference before another quality study. |
| Standard generation differs from the prior run | Use the single same-host streamed control to separate a current decoder difference from a difference with the prior machine/run. |
| Execution fails | Preserve receipts and charges; no automatic retry. |

## Resources and handoff

The [resource contract](../config/experiments/modular-decoder-parity-resources.json)
allows one temporary `r7i.4xlarge`, **two hours maximum, $6 incremental planning
cap**, 80 GiB gp3 and no GPU. At the already pinned $1.0584/hour rate, maximum
compute is $2.1168 with $3 reserved for storage/network headroom. Earlier costs
remain published separately. The allocator checks the prior retirement receipt.

Setup is limited to 1,800 seconds; artifact preparation to 900 seconds; the
entire parity worker to 1,800 seconds and 112 GiB, including all comparisons and
the optional same-host control. Evidence-copy allowance is 600 seconds, within
the allocation deadline. Worker wall time, failures and interruptions are charged
using the existing supervisor. Independent expiry guards and automatic instance,
volume and security-group retirement remain enabled.

Commit first. From an isolated checkout at the committed revision, start the
existing controller as a persistent user service:

```bash
PYTHONPATH=src python scripts/modular_reference_cloud.py run \
  --profile decoder-parity --home /absolute/path/to/new-parity-study
```

It waits for successful exact-commit CI before allocating. Local handoff:
`.neuroshard/modular-decoder-parity-latest.json`. The study home contains
`status.json`, `result.json`, `evidence/.study/attempts/`, and
`resources-finished.json`. Per-step progress survives inside the parity worker's
attempt directory even if its final receipt is interrupted.

The experiment ends with collected evidence and retirement. No model training,
new final, blockchain change, admission, automatic relaunch or assistant
milestone completion follows from this contract.
