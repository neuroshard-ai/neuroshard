# Granite standalone adapter integration audit

September 26, 2026. **Opened-case diagnosis; no new learning or admission.**
The [Granite reference](GRANITE_REFERENCE_RESULTS.md) remains failed. Its parent
qualified and the composed model preserved 18/18 assistant successes, but the
requirement checker lost two checks where at most one was allowed.

The next question is whether those mistakes also occur with the published
standalone adapter, or arise when embedding it into the composed model. Resolve
that before investing in training. This supports A1's implementation diagnosis;
it cannot close A1, demonstrate growth, or substitute for automatic tool use.

## Finding before inference

The [local tokenizer audit](../config/experiments/granite-adapter-audit-invocation.json)
found identical effective inputs and the intended recorded Switch activation
boundary in all 16 cases. It also found a standalone-interface trap in all 16:

- The published adapter config requires invocation tokens `[27, 71226, 29]`.
- The published instruction starts with `<requirements>:`. Its tokenization is
  `[27, 71226, 27916]`; `>:` occupies one token.
- PEFT 0.19.0's exact-sequence detector therefore returns no active offset.
- Switch uses a control token and did activate the adapter at the intended
  position in the completed run. This mismatch does **not** explain away that
  run's two false rejections.

For this audit, standalone generation explicitly supplies PEFT's `alora_offsets`
at the recorded Switch activation boundary. The offset comes solely from the
input/control position, never from an expected answer. Prompts and model weights
remain unchanged. This compares equivalent activation; it does not demonstrate
working default PEFT invocation. Changing punctuation to activate the adapter
would alter the inputs and is outside this contract.

The raw adapter is F32; the composed checkpoint is BF16. Disable PEFT's adapter
autocasting, load the original adapter, convert it to BF16, and verify the loaded
tensors. This compares the published composed precision, not F32 accuracy.

Primary artifacts: [published adapter and interface](https://huggingface.co/ibm-granite/granitelib-core-r1.0/tree/d0a2a96a4cd07e96f0fe7ca29a42bfe088299d43/requirement-check),
[pinned composition implementation](https://github.com/generative-computing/granite-switch/tree/60d546d211907bf42934113a3e69e9a302af9868),
and [PEFT aLoRA implementation](https://github.com/huggingface/peft/blob/v0.19.0/src/peft/tuners/lora/variants.py).
The tokenizer finding is our local measurement, not a claim made by those sources.

## Frozen measurement

The [plan](../config/experiments/granite-adapter-audit.json) pins the raw adapter,
interface files and published composition map, alongside the unchanged parent,
modular checkpoint and original 16 reference cases. Reuse the original scorer,
greedy decoder, native prompts and 32-token generation cap.

1. Verify artifact hashes and all effective prompt tokens. Report default PEFT
   detection separately from explicitly aligned activation.
2. Check all 243 backbone tensor mappings exactly, including fused projections.
   Check all 320 adapter tensors against BF16 A and alpha/rank-scaled B, including
   zero padding and zero weights in non-target MLP projections.
3. Generate 16 standalone answers and 16 composed answers on the same CPU host.
   Save actual PEFT offsets, Switch routes, input and output token IDs, text,
   termination, executable scores, timing and cumulative process peak memory.
4. Compare all 16 composed outputs to the old record, including routes. Compare
   both implementations' text, tokens, termination and verdicts exactly.

The comparison deliberately makes no full-logit equality claim: fused versus
separate projections and adapter scaling can have different BF16 rounding.
An output discrepancy is evidence to explain, not permission to select whichever
answer passes. Both successful and unsuccessful answers are compared.
Agreement would localize the observed behavior to the shared adapter/input/
precision combination; it would not prove that every possible invocation has
the same quality or identify the training cause of an error.

## Decision and stop rule

| Observation | Conclusion |
| --- | --- |
| Artifact, tensor, input or activation mismatch | Stop with an implementation discrepancy. No repair or retry under this freeze. |
| Composed output differs from the old record | Reproduction remains unresolved; do not infer a quality improvement. |
| Composed replay matches but standalone answers differ | Publish the differences. Their cause remains unresolved. |
| All paired answers and composed replays match | The same published-adapter errors are reproduced. Stop loader/prompt diagnosis; future capability work needs its own prospective contract. |

All outcomes leave the old failed study and checklist unchanged. No training,
new examples, threshold calibration, quality promotion, or native-network change.
The requirement checker is a learned component, not the authority for admitting
model updates. Deterministic checks remain the judge of these formal constraints.

## Execution and accounting

Commit the [execution freeze](../config/experiments/granite-adapter-audit-execution.json)
before any full-checkpoint generation. The controller requires successful Protocol
CI for that exact commit. One disposable `r7i.4xlarge`, two-hour expiry, **$6 total
planning cap**, 80 GiB disk, 60-minute worker, 64 GiB worker memory. No GPUs.
Prior resources must be retired. The controller copies evidence and terminates
the instance and its volumes after success, error or timeout. No automatic retry.
Compute, storage and transfer accounting remain distinguishable.

The isolated runtime adds PEFT 0.19.0, Accelerate 1.15.0 and psutil 7.2.2 to the
pinned Granite stack; production/ledger dependencies are unchanged. Local
preflight checks the real PEFT save/load/generation path on a tiny random model,
the full published topology on the meta device, and the recorded token inputs.
It generates no answers with the real checkpoint before the committed run.

Run the frozen controller using:

```sh
PYTHONPATH=src venv_build/bin/python scripts/modular_reference_cloud.py run \
  --profile granite-adapter-audit --home /absolute/path/to/new-audit-home
```

Inspect `status.json`, `result.json`, and `resources-finished.json` in that home.
The current controller path and unit will be recorded in the ignored
`.neuroshard/modular-assistant-latest.json` when queued.
