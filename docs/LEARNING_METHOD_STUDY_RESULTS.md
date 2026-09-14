# Shared-gradient learning study: September 14, 2026

**Two workers trained a useful task improvement with 91.4% less transmitted data than dense synchronization, while passing the declared quality screen. The initial compressed implementation was slower than one GPU.** This separates a promising learning method from an unresolved execution bottleneck.

The [frozen method](LEARNING_METHOD_STUDY.md) trains all 1,711,376,384 parameters of SmolLM2-1.7B-Instruct. All three arms use 128 updates, a global batch of 32 documents, the same initial model, shared AdamW update rule, global target normalization and training schedule. The training set contains 3,072 generated task conversations and 1,024 public conversation-replay documents. This is a research experiment; it issues zero NEURO and promotes no native serving model.

The input and numerical-source commitment is `019cdb4d04a17a5819e5399f9ed96c3bc6fcb880`. Its [prepared identity](../config/experiments/learning-method-study-inputs.json) is `0f1e2af42ce199d08cb7844e794429b0712f92a15f23832f23794b7aedd87e1e`. All three predetermined step-128 candidates were [committed together](../config/experiments/learning-method-study-selection.json) at `35aef1b` before any final scoring.

## Learning

| Fresh task cases | Seed | One GPU | Two GPUs, dense | Two GPUs, compressed |
| --- | ---: | ---: | ---: | ---: |
| Correct / 1,024 | 108 | 730 | 723 | 727 |
| Accuracy | 10.55% | 71.29% | 70.61% | 71.00% |
| Lookup / 256 | 105 | 254 | 254 | 254 |
| Filtering / 256 | 1 | 250 | 250 | 251 |
| Sorting / 256 | 2 | 221 | 215 | 221 |
| Invoice totals / 256 | 0 | 5 | 4 | 1 |
| Mean retention-loss change, nats | — | −0.008112 | −0.008612 | −0.008686 |
| Approximate 99% upper retention bound | — | −0.000613 | −0.001245 | −0.001301 |

The compressed candidate passes all four declared quality conditions:

- Task accuracy gains 60.45 percentage points over the seed, exceeding the ten-point minimum.
- Its retention upper bound stays below the +0.02-nat limit.
- The paired accuracy lower bound against one GPU is −1.74 points, above the −3-point noninferiority margin.
- The paired lower bound against dense synchronization is −0.94 points, also above that margin.

There is no evidence here that compression improves quality over either control. The result supports preserving the measured learning gain while reducing communication. Arithmetic remains poor across all candidates. These are new generated cases from four public task families, one development seed, and 128 previously exposed retention probes. They do not certify general assistant quality, repeated continual learning, or increasing model capacity.

## Primary execution measurements

| Measurement | One GPU | Two GPUs, dense | Two GPUs, compressed |
| --- | ---: | ---: | ---: |
| Maximum rank active time | 793.44 s | 764.00 s | 882.84 s |
| Active time plus final checkpoint | 808.64 s | 779.18 s | 898.09 s |
| Complete process group | 831.10 s | 803.04 s | 931.63 s |
| Allocated GPU seconds, complete process | 831.10 | 1,606.08 | 1,863.27 |
| All hosts, transmitted GB | 0.000095 | 1,774.846 | 152.860 |

The initial rank-32 PowerSGD implementation keeps full-precision error feedback on CPU between completed bucket reductions. It performs eight dense warmup updates and 120 compressed updates. It transmits 8.61% of the dense arm's observed bytes, including transport overhead and warmup, but takes 11.27% longer than one GPU in active time. Its complete process consumes 2.24 times the allocated GPU seconds. It fails the declared 1.10-times speed screen and establishes no cost advantage.

Both hosts are g6e.2xlarge instances with one L40S each, placed in different availability zones because a same-zone pair was unavailable. All primary arms run sequentially without bulk artifact transfers or competing training. Whole-host counters include management traffic; received bytes are retained separately and are not counted again as independently transmitted data. Arm order, network credits and hardware placement limit causal throughput claims. Cross-zone transfer charges must be included in allocation accounting.

## Interpretation

The learning result justifies investigating shared-optimizer compression further. It does not justify building a public training economy around the initial implementation. The current experiment holds model size fixed and uses operated, cooperating workers. It does not establish economical malicious-work verification, elastic participation, recovery of compression state, useful model growth or competitive general inference.

The prior local-window failure remains a failure. Its data size, batch size, local optimization histories and loss normalization all differ from this study; the new comparison cannot identify which change explains the different quality outcome.

## Memory and batching diagnostics after candidate selection

A [16-update paired probe](../scripts/probe_gradient_buffers.py) identifies a concrete improvement. Retaining and zeroing DDP gradient buffers allows the full-precision compression errors to remain on GPU. On both hosts, the offloaded and resident variants match exactly in parameters, Adam tensors and parameter-group metadata, compression errors, P/Q projections, the complete projection RNG, PyTorch CPU/CUDA RNG states and the numerical trajectory. Every parameter participates in every update; the zero-gradient and absent-gradient semantics are not interchangeable for unused parameters.

After eight dense warmup updates, the median compressed update changes from **6.847 to 3.896 seconds** on the slower rank. Peak allocated memory remains approximately **36.94 GiB**. This is a memory/transport equivalence result over 16 updates. It does not replace the original 128-update runtime with an estimated faster result.

A separate [microbatch probe](../scripts/microbatch_probe.py) also exposes a weak single-GPU implementation. Processing one, two and four documents per forward/backward pass gives median update times of **5.979, 3.141 and 1.966 seconds** respectively over updates 5–16. The probe sorts documents by length within each global batch. A CPU check with unequal lengths and weights verifies matching response losses and gradients within FP32 tolerances, including masking of prompts and padding. GPU batching changes the floating-point computation, so these probes neither inherit an earlier final model's quality score nor establish a matched full-length speed advantage.

Together these results motivate a fresh comparison with efficient batching on both sides and enough computation per synchronization. The shared-optimizer method has passed its first quality screen; sustained useful scaling still requires the stronger comparison.
