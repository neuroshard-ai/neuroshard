# Batched cooperative learning: September 14, 2026

**Two L40S workers complete the matched 1.7B training experiment 1.45× faster than an efficiently batched single GPU and pass the declared task-quality and retention screen. They use 37.8% more allocated GPU seconds.** This is evidence for a cooperative training method on these machines, with fixed model size and narrow tasks.

The [frozen plan](BATCHED_LEARNING_STUDY.md) trains all 1,711,376,384 parameters of SmolLM2-1.7B-Instruct for 128 updates. Both arms use the same 8,192 training documents, global batch of 256, document schedule, response-token weights, learning rate and shared AdamW rule. The distributed arm uses rank-32 PowerSGD with full-precision error feedback, eight dense warmup updates and 120 compressed updates. Single-GPU microbatches contain sixteen documents; each distributed worker processes eight. Retained gradient buffers and expandable CUDA allocation segments let both controls use the larger tested batches.

Numerical sources and [prepared inputs](../config/experiments/batched-learning-study-inputs.json) were committed at `073599b5619737c803c4bbf706fa6ef050e45805`, with prepared identity `43a704d4a67b6e38abf5c315f51729306775757b6683a7fb20c6c541335911c5`. Both predetermined step-128 models were [committed together](../config/experiments/batched-learning-study-selection.json) at `d52876209c6d3f558c10a993440708a2a9ccd229` before any final scoring. The [machine-readable results](../config/experiments/batched-learning-study-results.json) retain evidence hashes and the declared decisions.

## Learning

| Fresh task cases | Seed | One GPU | Two GPUs, compressed |
| --- | ---: | ---: | ---: |
| Correct / 1,024 | 109 | 748 | 738 |
| Accuracy | 10.64% | 73.05% | 72.07% |
| Lookup / 256 | 104 | 256 | 255 |
| Filtering / 256 | 3 | 252 | 250 |
| Sorting / 256 | 2 | 235 | 230 |
| Invoice totals / 256 | 0 | 5 | 3 |
| Mean retention-loss change, nats | — | +0.022167 | +0.005077 |
| Approximate 99% upper retention bound | — | +0.034610 | +0.015325 |

The compressed candidate gains 61.43 accuracy points over the seed. Its paired accuracy difference against one GPU is −0.98 points, with an approximate 99% lower bound of −2.10 points, above the declared −3-point margin. Its retention upper bound stays below +0.02 nats. All three quality conditions pass.

The single-GPU candidate scores ten more tasks correctly but **fails the retention threshold**. The distributed candidate's retention mean is also slightly worse than the seed; passing a noninferiority margin does not mean zero forgetting or improved general quality. This one run does not establish a regularization benefit from compression. Invoice arithmetic remains almost entirely unsolved by either candidate.

The 1,024 test cases have no case-ID overlap with earlier experiment roles. They are new instances of four public, deliberately trained task families. The 128 retention documents were previously exposed. Confidence intervals describe paired cases within one development seed; they do not measure variability across training runs, independent operators or task distributions. This is instruction fine-tuning of an existing pretrained model, not general pretraining or a broad assistant benchmark.

## Execution and cost

| Measurement | One GPU | Two GPUs, compressed |
| --- | ---: | ---: |
| Maximum rank active time | 1,304.90 s | 875.56 s |
| Active time plus final checkpoint | 1,320.11 s | 890.87 s |
| Complete process group | 1,343.23 s | 925.27 s |
| Allocated GPU seconds, complete process | 1,343.23 | 1,850.53 |
| All hosts, transmitted GB | 0.000109 | 151.628 |
| Maximum allocated GPU memory | 39.57 GiB | 41.68 GiB |

Active training is 1.490× faster and the complete process is 1.452× faster, exceeding both 1.10× thresholds. Complete-process time includes startup and the final checkpoint, but excludes provisioning, evaluation and backup. Multiplying it by the allocated GPU count yields a **1.378× compute-cost ratio**, before network charges. Faster execution is the established benefit; cheaper execution is not.

Both g6e.2xlarge hosts are in us-east-1, in separate availability zones. The compressed arm ran first and the single-GPU arm second, sequentially without competing GPU jobs or bulk artifact transfers. These are nearby cloud hosts under one operator; the result does not predict performance across arbitrary Internet links, mixed GPUs or frequent departures. Whole-host traffic includes small management overhead. This comparison has no dense distributed arm, so its communication savings against dense synchronization were not measured. The earlier [three-arm study](LEARNING_METHOD_STUDY_RESULTS.md) separately measured a 91.4% reduction with its different recipe.

Both ranks agree on the compressed model's parameter root, `ac2153f0f3e55379f72d46f4ef4a7a9a1f6b7f30a45d471cb3611c9e15a1385b`. The single-GPU root is `743ebceb281d21e900a5ebf0b2c0a5fa6fd90a0289bcff1b1f98b3cd488fb44f`.

## Validation and preservation

A CPU objective test checks the batched implementation against the established per-document AdamW calculation, including unequal lengths, unequal target weights, reversed rank assignment, prompt/padding masking and a partial microbatch. Earlier paired CUDA probes check complete numerical-state equality for retaining gradient buffers and resident compression errors over sixteen updates. Memory failures and slower configurations remain in the experiment evidence.

An independent post-scoring pass re-tokenizes prompts, checks that the assistant answer is excluded from each input, decodes generated token IDs, re-scores every response, and verifies retention target masks, case alignment and finite losses. Its counts agree with all three evaluation files. It validates artifact consistency; it is not an independently operated rerun of inference.

All five final trained models from the two full studies are preserved in content-addressed S3 objects with full readback hash verification. Checkpoints contain model weights. They do **not** contain a resumable distributed optimizer, rank-local compression errors, projection state, data position and RNG state. Consequently, earlier recovery results from a different method cannot establish recovery of this one.

The 12,031,602-byte evidence archive contains the prepared inputs, generated answers, rank receipts, probe failures, exact numerical-source snapshots, objective test and reporting code. Its SHA-256 is `42f0bfc4e5a078cff94624731cf43f5b26be87ae19765833f588181d9f86b31f`. The machine-readable results record its S3 locator and readback verification. The archive index binds each member's bytes; model weights are in the separate verified objects.

Both temporary GPU instances, their root volumes and their dedicated security group were deleted after preservation; the existing native CPU hosts remain. Graceful shutdown was slow, so cleanup used force termination after backup verification. The whole allocation, including both full studies and failed/successful diagnostics, has a conservative **$77.10 planning estimate before credits**: about $12.6 compute through confirmed cleanup, $54.5 from charging all 2,723.27 GB of host-lifetime transmitted traffic at $0.02/GB, and a $10 allowance for other costs. This is not an AWS invoice. Retained S3 storage continues to accrue separately.

## Method decision

Use shared AdamW updates with compressed gradient exchange as the leading learning candidate. [PowerSGD](https://arxiv.org/abs/1905.13727) supplies the underlying low-rank communication method; the experiments here establish its limited NeuroShard result. The earlier local-window recipe remains rejected on its measured quality criteria. Efficient batching on the single-worker control is essential to the comparison.

Before building more protocol machinery around this candidate, resolve these method questions in order:

1. **Continuation and recovery:** preserve the complete distributed numerical state, recover after a worker failure, and train a new cohort from this accepted research checkpoint. Measure new-task gain and retention of the first cohort and broader response tasks. Replay must use previously trained material. A fresh start from the original seed cannot demonstrate continual learning.
2. **Pooled model memory:** measure a sharded implementation whose useful training capacity exceeds one worker's memory, including communication and failure recovery. [ZeRO](https://arxiv.org/abs/1910.02054) provides an established way to partition training state; combining state sharding with this compression method requires new numerical and traffic measurements. These current workers each hold the whole model.
3. **Useful growth:** compare continued dense training with a checkpoint-derived larger candidate under a frozen total budget. [Sparse upcycling](https://arxiv.org/abs/2212.05055) is one research candidate for reusing learned weights while adding experts. It still needs subsequent training, reliable expert execution, retention, serving measurements and a quality advantage over the smaller control. Additional parameters alone do not pass.

These are proposed experiments, not activated network features. The current success proves neither repeated improvement, larger model capacity, economical verification of malicious work, nor a permissionless provider market. Native consensus, the public 0.4.0 model and token issuance are unchanged; these research runs earn zero NEURO and approve no serving promotion.
