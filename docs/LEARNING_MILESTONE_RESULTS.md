# Frozen useful-learning experiment — September 12, 2026

**Result: reject.** The 128-step full-model run completed, but its sealed-test upper confidence bound did not meet the [frozen contract](LEARNING_MILESTONE.md). Retention passed. The reported fresh set also improved; it cannot rescue the sealed test. Continual learning and two-host scaling remain blocked by this experiment.

The [complete evidence](https://neuroshard.com/experiments/learning-milestone-20260912/README.md) includes all losses, all [20 generation pairs](https://neuroshard.com/experiments/learning-milestone-20260912/generations.html), the [candidate checkpoint](https://neuroshard.com/experiments/learning-milestone-20260912/candidate-checkpoint.tar.gz), source identities and [SHA-256 checksums](https://neuroshard.com/experiments/learning-milestone-20260912/SHA256SUMS). Rejected weights remain downloadable for inspection and reproduction. They are not activated on the public network.

## Measured result

Changes are candidate minus seed assistant-response loss, in nats. Each group has 64 documents; every selected evaluation window is scored. A document contributes one observation after weighting its windows by their actual response-target counts. The upper bound is the paired mean plus 2.576 standard errors, following the committed approximate normal-interval rule.

| Role | Seed loss | Candidate loss | Mean change | Upper bound | Required bound | Result |
| --- | --- | --- | --- | --- | --- | --- |
| Sealed test | 1.42084935 | 1.41077176 | −0.01007759 | +0.00400660 | below −0.001 | Fail |
| Retention | 1.49754274 | 1.48750696 | −0.01003578 | +0.00226162 | below +0.020 | Pass |
| Fresh | 1.55173863 | 1.54101069 | −0.01072795 | −0.00494882 | below −0.001 | Pass, reported only |

The [result JSON](https://neuroshard.com/experiments/learning-milestone-20260912/result.json) retains full-precision values. A separate evidence checker reconstructs all document averages from window losses and target counts, then recomputes the decision without importing the training implementation.

![Paired loss changes and frozen limits](https://neuroshard.com/experiments/learning-milestone-20260912/learning-comparison.svg)

## What the failure shows

The test-set mean improved, but changes were heterogeneous: 41 documents improved and 23 worsened. The median change was −0.00163362 nats. One document with two response targets improved by −0.33345723 nats, producing a large contribution to both the mean and its uncertainty. It remains in the evaluation. The [document audit](https://neuroshard.com/experiments/learning-milestone-20260912/document-loss-audit.json) publishes every change, including regressions. This is descriptive analysis after scoring; it does not supply a replacement significance test.

A separate [post-run token diagnostic](https://neuroshard.com/experiments/learning-milestone-20260912/token-loss-diagnostic.json) rescored the largest test gain and largest retention regression one target at a time. In the first, loss on `False` fell by 0.60956 nats and on the end-of-response token by 0.05735. In the second, loss on `three` increased by 0.84699 despite an end-token improvement of 0.16558. Both gains and regressions therefore affect answer content. The diagnostic used 12 read-only forwards in 65.28 seconds after the frozen run stopped; its reconstructed means match the original window losses within 0.000001. These two documents were chosen after scoring and are not a new evaluation set.

Eleven of the 20 response-token sequences changed; nine were identical. Fifteen responses from each model used the full 32-token allowance. Some changes reduce repetition, but the simple earthworm question still receives the incorrect answer “Loose,” and several code examples remain unfinished. These probes do not establish a better general assistant. All original prompts, rendered token inputs and outputs remain published; seven inputs used the frozen 192-token truncation rule.

Neither a larger test set chosen after inspecting these results nor a different aggregation rule can retroactively pass this run. A follow-up must have a new committed plan and unused evaluation documents, with its selection and stopping rules fixed before scoring. This result does not authorize phase 2 or phase 3 of the current plan.

## Execution and resource cost

One Intel Xeon Platinum 8259CL host, with four virtual CPUs, ran three worker processes. All 134,515,008 model parameters were trainable. The recipe used 128 SGD steps, two response windows per step, learning rate 0.003 and global clipping at 1.0. There was no growth or replay. The 256 selected training documents contained 604 windows; exactly one per document was trained. Unused admitted windows are not counted as trained rehearsal data.

The accepted history contains 128 distinct work identities and 384 worker receipts. Training-step time totalled 2,309.85 seconds; the median step took 18.27 seconds. The complete phase, including both sets of scores and generations, took 7,625.03 seconds (127.08 minutes), with zero retries. Peak retained experiment size was 66.92 GiB, within the 256 GiB limit. The service's memory cap was 6 GiB. A second local preparation reconstructed the committed selection byte for byte. It shared the operator and host and overlapped part of training. Other public services remained active; the [environment record](https://neuroshard.com/experiments/learning-milestone-20260912/execution-environment.json) also records the separate read-only research probes. These timings are not an isolated hardware benchmark.

The experiment issued zero tokens and did not upgrade the public 0.4.0 chain, which continues to use its released adapter profile. Successful execution and a rejected quality decision are both recorded.

## Reproduce the exact experiment

Use a full clone at execution revision [`4f18f903b738d5f2e5ba6d31434b7e0ae5e06fd7`](https://github.com/neuroshard-ai/neuroshard/tree/4f18f903b738d5f2e5ba6d31434b7e0ae5e06fd7), with its numerical and collector dependencies. Follow the [driver instructions](LEARNING_MILESTONE.md#execution-and-evidence) to download the pinned seed, reconstruct the selected windows into a new home, and run the identical recipe. The earlier `plan_commit` identifies the original frozen plan; it does not contain the later selection. Preserve this ancestry when merging.

| Artifact | SHA-256 / root |
| --- | --- |
| Committed selection | `c583aba89d2c6c18f05efb6d3c63519b5c21b442a961727ddc76f331a98e58ac` |
| Bound implementation | `a0fd7942b91ec84720a96051147b6fe8cddb6519142810919ae40c2435ead763` |
| Seed with tokenizer metadata | `7ca58fefa5977056edfbca3241b174d1376f4be3042aafb3fd768cb2d21dbc02` |
| Rejected candidate | `ecb4262c2b7d63a0bfb0d17a9dc9023af33c351645d0e6ba12f087ba6d474677` |
| Checkpoint archive | `d8cfa118c271c89b45f8d6d523189682e1c655a10dbc9012b23562226f4e22ae` |

The 423,060,091-byte archive contains 35 native model/tokenizer objects, all checked against their content hashes, plus provenance and license files. These are float32 NeuroShard objects, not a Hugging Face `save_pretrained` directory. Its included README explains inspection with the existing tools. Intermediate training history remains reconstructable from the public recipe and pinned sources.

The seed and corpus are Apache-2.0. The seed's upstream model card describes prior Smol-SmolTalk fine-tuning. “Untouched” here means unused by this experiment, not proven absent from the seed's original training. Public precommitment and one local reproduction do not establish adversarial evaluation security or independent ownership.

## Separate engineering probes

A [lossless delta probe](https://neuroshard.com/experiments/learning-milestone-20260912/checkpoint-delta-note.md) used the final real transition. XOR against the exact parent and zlib compression reduced 538,085,392 tensor bytes to 153,503,190 bytes, compared with 422,376,714 bytes for directly compressed candidate tensors. All target bytes and hashes reconstructed exactly. Encoding and restoration/hash verification took about 9.83 and 4.80 seconds, respectively, in one shared-host measurement. This suggests a transfer or archival option when the exact parent is already available. It does not prove training, justify deleting dispute inputs, or measure a network speedup.

[Verification research](https://neuroshard.com/experiments/learning-milestone-20260912/verification-note.md) tested two exact bounded-integer matrix checks. The first was substantially slower than optimized multiplication. A revised formulation was approximately tied with optimized replay after including matching input validation. Neither is a verifier for the current rounded FP32 training graph. Source, failed comparisons and timing samples are published; no probe was adopted into the protocol or used to change the learning decision.
