# A1 foundation decision after the BAR failure

September 26, 2026. **Qualify Granite 4.1 3B and its published Granite Switch
checkpoint.** This is a reference candidate, not the accepted assistant.
[A1](../TODO_ASSISTANT.md) remains open; do not train a new expert yet.

The [BAR baseline](MODULAR_REFERENCE_FRESH_RESULTS.md) stays failed at 11/24.
The [decoder audit](MODULAR_DECODER_PARITY_RESULTS.md) reproduced all 24 answers
with the upstream implementation. More work on that decoder is not the next
learning experiment.

## Research conclusion

The next question is whether a capable assistant can retain ordinary behavior
while gaining an independently developed neural function. Start with a
published pair and its actual invocation interface. Do not restart the failed
tail merging or lexical selection experiments.

Granite Switch embeds adapters in one backbone's attention and feed-forward
projections. Control tokens select a function; there is no learned semantic
router. The libraries include response requirement checking. This is a better
reference for incremental assistant capabilities than BAR's domain-pretraining
experts. That choice is our inference from the artifacts, not a measured result.
[Model card](https://huggingface.co/ibm-granite/granite-switch-4.1-3b-preview),
[IBM's design description](https://research.ibm.com/blog/granite-libraries-project-switch).

Activated LoRA trains an adapter to consume an unchanged base-model prefix
before activation. This defines a cache boundary for independent modules.
It does **not** make arbitrary adapted caches interchangeable or prove automatic
composition, preservation, or permissionless verification.
[Activated LoRA](https://arxiv.org/abs/2504.12397),
[serving study](https://arxiv.org/abs/2512.17910).

The path remains A1 → A2 → A3: qualify the assistant/reference; learn a tool-use
extension against the no-growth control; then test repeated growth and an upgrade.
A2 must measure the complete automatic path. Explicit calls in A1 cannot replace
that. An assistant that invokes a module but cannot use its output still fails.

## Alternatives examined

Availability refers to the inspected author repositories on the date above.

| Candidate | Finding and decision |
| --- | --- |
| [BAR](https://allenai.org/blog/bar) | Released pair and supported decoder; our baseline quality failed. Stop this candidate. |
| [Qwen3-4B-Instruct-2507](https://huggingface.co/Qwen/Qwen3-4B-Instruct-2507) | Apache-2.0, dense 4B, non-thinking assistant with tool support. Useful alternative foundation; the model card does not provide an independently grown modular checkpoint. Qualifying it alone leaves that A1 criterion open. |
| [Qwen3.5-4B](https://huggingface.co/Qwen/Qwen3.5-4B) | Available, but hybrid linear/full-attention and vision add a different cache/partitioning target. Newer scores do not establish a modular recipe. |
| [MeteoRA](https://github.com/NJUDeepEngine/meteora) | Released adapters and token routing; older Llama base models and task-specific evaluation settings. A port to a current assistant is a new method. |
| [GoD-MoE](https://ojs.aaai.org/index.php/AAAI/article/view/40077) | Relevant adaptive expansion. The inspected [code snapshot](https://github.com/xyguo1229/GoD-MoE_code/tree/a746729d4f03331cd2de9f4c2a168be38a5031fe) did not supply a ready baseline/modular artifact pair for this study. |
| [CoMoL](https://github.com/DCDmllm/CoMoL/tree/011306a0401cf371c855ce7a1582f0ae21b2f238) | Compact token-routed adapter cores and training code; no downloadable trained comparison in the inspected release. Joint fine-tuning is not a continual-admission rule. |
| [LoRA-Mixer](https://github.com/hustcselwb/LoRA-Mixer/tree/076f673c5ecd317e50d89b4af45c7145cef6b9fa) | Interesting composition; inspected tree contains README/license, with implementation still promised. Not an executable reference yet. |
| [Macaron-V1](https://github.com/MindLab-Research/Macaron-V1) | Relevant assistant design, but smallest released profile is much larger and selects one specialist per turn. It does not establish the composition claim we need. |
| [Granite Switch](https://github.com/generative-computing/granite-switch) | Published small assistant, modular weights, provenance and runnable implementation. Selected for this reference comparison. |

## Artifacts and recipe

| Component | Pinned identity |
| --- | --- |
| Parent | `ibm-granite/granite-4.1-3b` at `c0650403e44e78ec0262dab1c90914c65b196c4e` |
| Modular | `ibm-granite/granite-switch-4.1-3b-preview` at `7a3ac02e07868411424ac89440397b475da66fa7` |
| Code | `generative-computing/granite-switch` at `60d546d211907bf42934113a3e69e9a302af9868` |
| Parameters | Parent 3,402,836,480; modular 4,149,977,600 BF16 parameters; 12 embedded adapters, ranks 16/32 |
| Terms | Model cards and upstream code declare Apache-2.0; repository license unchanged |
| Runtime | Isolated Python 3.12.12, PyTorch 2.10.0 CPU, Transformers 5.5.4; full dependency inventory committed |

The [published composition record](https://huggingface.co/ibm-granite/granite-switch-4.1-3b-preview/blob/7a3ac02e07868411424ac89440397b475da66fa7/BUILD.md)
identifies the three source adapter libraries and revisions. Published adapter
weights initialize this reference; nothing is trained. The composer adds control
vocabulary and a switch. No attention, embedding, head, adapter or switch
parameter is updated here. A later module must bind its exact base revision,
target projections, rank/scaling and activation interface. A backbone upgrade
requires renewed compatibility and quality evaluation.

Latest inspected code `6013c7f…` uses a two-slot multi-switch layout. This
checkpoint has one switch slot plus 40 decoder layers. The pinned earlier
implementation matches it. Metadata-only construction confirmed 40 physical
layers and the exact parameter count; a tiny synthetic CPU forward activated
only its requested adapter. These checks are not quality results.

The selected single-switch implementation cannot return to the parent inside
the same cached sequence. Each reference call starts with a fresh cache.
Future multi-step serving must pay for rebuilding context or separately validate
another cache implementation. No published speedup is credited to this run.

## Frozen comparison

The [plan](../config/experiments/granite-reference.json),
[execution inventory](../config/experiments/granite-reference-execution.json),
[artifacts](../config/experiments/granite-reference-artifacts.json) and
[resources](../config/experiments/granite-reference-resources.json) define one
inference-only CPU allocation after commit and passing CI.

1. Run 24 new public qualification cases: eight conversation, eight instruction,
   eight native JSON tool calls. Require parent ≥18/24 and ≥5/8 per category.
   Otherwise stop before downloading the modular checkpoint.
2. Evaluate the modular checkpoint without adapters on the same cases. Preserve
   every correct parent answer. This tests behavior, not elementwise identity
   between the two upstream architectures.
3. Compare 16 balanced requirement checks. Both arms receive the same instruction
   and response to judge; the modular arm also receives its published activation
   token. Require ≥12/16, ≥6/8 per label and at most one regression against the
   prompted parent. Publish all outputs and the comparison, including a tie.
4. Require p95 generation time ≤120 seconds across primary calls. If quality
   passes, reload each model and repeat three declared cases. Tokens, prompt
   identities, scores and routing traces must match exactly.

Use upstream `model.generate`, native templates, greedy BF16 CPU eager attention,
256 new tokens for assistant cases, 32 for reference checks and at most 2,048
input tokens. No constrained decoding, answer cleanup, tuning or quantization.
Tool operations and typed arguments must match; extra prose, duplicate keys and
invalid JSON fail. Remove only terminal EOS. Truncated replies fail.

This is a **small functional reference reproduction**, not reproduction of IBM's
full benchmark scores or proof of general assistant readiness. Explicit checking
is a published neural function, not newly learned capability or automatic
selection. A passing report still needs review against every A1 criterion;
the runner cannot close the checklist. Earlier studies/finals remain untouched.

## Route to A4

BF16 parameter storage is about 6.34 GiB for the parent and 7.73 GiB for the
modular checkpoint, before integer buffers, caches, allocator overhead and
temporary tensors. The single-host reference records cumulative process peak
RSS, not isolated per-model memory or distributed execution.

Hidden width 2,560 requires 5,120 BF16 activation bytes per boundary token.
Three sequential stages send at least 10,240 bytes per decoded token across
their two cuts; a 4,096-token prefill sends about 40 MiB across both cuts.
Training additionally sends backward activations and stores optimizer state.
These are payload estimates, not throughput or internet-latency measurements.

Forty layers, eight KV heads, head dimension 64 and BF16 caches require about
320 MiB for a 4,096-token sequence, plus switch state. Layer placement divides
weights and caches. Embedding/head ownership, checkpoints and adapter optimizer
state still need a concrete A4 partition. Replicas can increase concurrency;
more serial stages do not promise a faster answer.

## Cost and handoff

One `r7i.4xlarge`, no GPU, two-hour instance expiry, one attempt, $6 planning cap
including a storage/transfer reserve. Worker limit: 80 minutes and 64 GiB.
Setup is bounded at 20 minutes; evidence copying reserves 10 minutes. Sources,
weights and runtime are checked. Loading/routing errors, exhausted budgets and
failed gates stop the study. Evidence is copied before the instance, volume and
security group retire. Interrupted work and historical costs remain charged.

Progress is linked from `.neuroshard/modular-assistant-latest.json` after launch.
The background job requires no active model session. This freeze activates no
network, training, deployment or checklist completion.
