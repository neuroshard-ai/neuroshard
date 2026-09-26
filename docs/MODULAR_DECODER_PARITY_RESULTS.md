# BAR decoder audit result — September 26, 2026

The audit on freeze `d788955` **passed decoder agreement** and retired its
temporary resources at **17:06:40 UTC**. Standard Transformers reproduced all
24 recorded baseline answers, including the errors. The baseline remains
**11/24**; no quality result or assistant milestone has changed.

| Check | Result |
| --- | --- |
| Full logit comparisons, prefill and cached positions | 65/65 exactly equal |
| Maximum absolute logit difference | 0 |
| Standard greedy generation versus saved replies | 24/24 matched |
| Token IDs, text, termination, prompt hashes and scores | All matched |
| Baseline correct answers | 11/24: conversation 4/8, instructions 4/8, tools 3/8 |
| Conditional same-host streamed control | Not needed |

The numerical cases were `fresh-inventory` (2 prediction steps),
`fresh-tool-documents` (25) and `fresh-tool-reschedule` (38). Both implementations
used independently loaded weights and independent KV caches on the same host.
All 24 generation comparisons used standard `model.generate` with the original
prompts; recorded replies were comparison targets, not generation inputs.

After collection, the controller rechecked the source inventory, completion
criteria, logit summaries/hashes, all generated replies and all frozen scores.
AWS separately confirmed the instance was terminated and its volume and security
group removed. No training, GPU or larger modular model was used.

## Implication

The declared tests found no discrepancy between the custom layer-streamed
decoder and the standard implementation. The 11/24 result is reproducible
through the standard path; retrying this decoder is not the next learning step.
This does not establish equality for every input, numerical profile or model,
nor does it prove the modular architecture cannot work.

The foundation/interface decision now needs reconsideration under A1. Choose
a starting assistant that meets the usability requirements before testing our
own contributed capability. Keep these 24 cases as opened diagnostics; changing
their prompts or scoring cannot turn the failed frozen study into a pass. Any
new capability experiment needs its own prospective contract. No new allocation
or automatic retry follows from this result.

## Evidence and accounting

- [Unaltered worker/controller result](../config/experiments/modular-decoder-parity-result.json),
  SHA-256 `0f752c9a53e42b72f077176743ee185bbb8b2834bbae7a59bcbc4ab54c38f25a`.
- [Summary, CI, worker outcomes and resource receipts](../config/experiments/modular-decoder-parity-report.json).
- [Frozen audit contract](MODULAR_DECODER_PARITY.md).
- [Exact-source CI](https://github.com/neuroshard-ai/neuroshard/actions/runs/36256122570)
  passed before allocation.

The parity worker used **231.458 seconds**, including supervised launch overhead;
preparation used **151.168 seconds**. Its internal comparison timer was 228.603
seconds, process CPU time 1,132.325 seconds, and peak recorded RSS 29,747,589,120
bytes. These figures include both implementations and are not single-assistant
serving latency or memory measurements.

Conservative allocation time was **495.382 seconds**, costing **$0.145642** in
instance compute. Combined compute for the two fresh-study allocations and this
audit is **$0.590785**. Storage and transfer charges remain separate. Historical
baseline evaluation of 12,306.292 seconds remains recorded; this audit is an
additional cost, not new independent quality evidence.
