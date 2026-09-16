# Persistent model-shard lifecycle, 2026-09-14

A 1,711,376,384-parameter model trained across two physical GPU workers, lost one host during an update, and reproduced the uninterrupted model after restoring that shard on a third host. Neither training worker nor the coordinator constructed the full model. This closes a concrete operated recovery milestone; it does not establish permissionless training or model growth.

The [implementation and reproduction instructions](SHARDED_TRAINING.md), [frozen plan](../config/experiments/persistent-shards.json), and [input/source commitment](../config/experiments/persistent-shards-inputs.json) were fixed before training in commit `1f33493`. Prepared identity: `06203cab858bcbf41a2d62901b9c7a6ad2e9a92d410762473781af9796f28b6e`.

## Actual partition ownership

| Logical shard | Owned model state | Trainable parameters | Peak allocated CUDA memory in the uninterrupted run |
| --- | --- | ---: | ---: |
| 0 | Embedding/tied output head, final norm, layers 0–10 | 838,907,904 | 15.45 GiB |
| 1 | Layers 11–23 | 872,468,480 | 15.32 GiB |

The workers were `g5.2xlarge` instances with one A10G each, in one AWS availability zone. Each retained its own FP32 weights, gradients and AdamW moments, with BF16 computation and activation recomputation. The full FP32 training state would require 27,382,022,144 bytes before activations, exceeding either worker's available GPU memory under this profile. There was no CPU optimizer offload. Workers exchanged activations and their backward gradients, with a shared normalization denominator and clipping norm.

The uninterrupted first 32 optimizer updates took 222.26 seconds of update time and 349.07 seconds for the launched group including loading and checkpoint work. The two workers transmitted 13.91 GB of boundary tensors in total. This is a capacity and persistence demonstration; no training-speed comparison or wide-area scalability claim follows from these measurements.

## Physical-host replacement

1. Both workers committed step 16, including their complete AdamW and RNG state. All checkpoint objects were uploaded to operator S3 storage and verified by full readback.
2. The original group completed an uninterrupted reference at step 32.
3. A fresh group restored step 16. During step 20, after the first microbatch's forward pass and before a completed optimizer update, the controller stopped the original shard-1 EC2 instance. Its SSH process failed and the controller aborted the surviving group. Updates 17–19 and the partial update were uncommitted work.
4. A third physical host received only logical shard 1's seed files and 10,469,704,521 bytes of recovery artifacts from S3. Recovery used no disk access to the failed host. The survivor and replacement both restored the common step-16 checkpoint.
5. They replayed to step 32 in a fresh process group. Both produced the **identical full common manifest**, including weights, Adam moments/cursors, RNG files, numerical binding and parent checkpoint. Replayed losses and global gradient norms also matched exactly.

The uninterrupted and recovered step-32 root is `5c07f02f01b23cf99d425816573ba8c84a72b08e1e4fb8f21e9da5a804b0cf84`. The recovered group completed in 179.02 seconds, including checkpoint loading, 16 replayed updates and final checkpoint writing. The spare host was provisioned and artifact staging began before the injected failure. This timing excludes replacement provisioning, artifact transfer and failure detection.

Membership and rollback were controlled by the operator. The experiment tests one mid-update host-loss event on matching A10G hardware; it does not establish automatic discovery, uninterrupted availability, heterogeneous numerical equivalence, malicious-worker detection, or survival without a complete preserved checkpoint.

## Continued learning and distributed inference

The recovered group continued from update 32 through 64, preserving AdamW and RNG state. Its second cohort contained 512 new generated task examples, 256 task examples actually trained in the first cohort, and the same 256 conversation replay documents. Update time was 230.18 seconds; the launched group took 299.05 seconds including loading and checkpoint work. The final root, `3f8e4b1f15151d452f7711c5fff2b0ecb91299472ec082995603848bbf7bac60`, commits the recovered step-32 root as its parent.

The starting weights were the already trained 1.7B checkpoint from the [batched study](BATCHED_LEARNING_STUDY_RESULTS.md). That earlier experiment retained weights only, so this lifecycle initialized AdamW once at its start. No optimizer reset occurred during host replacement or between its two cohorts.

All three stages were scored on the same precommitted examples, using only their distributed shards. Loss columns are mean response cross-entropy per document over 64 documents each; lower is better. Generation used the first 16 committed cases per task cohort, capped at 64 output tokens.

| Checkpoint | Test A correct | Test B correct | Test A loss | Test B loss | Conversation retention loss |
| --- | ---: | ---: | ---: | ---: | ---: |
| Starting weights | 12/16 | 11/16 | 0.14408 | 0.14574 | 0.54436 |
| Recovered step 32 | 12/16 | 11/16 | 0.19811 | 0.20660 | 0.55522 |
| Continued step 64 | 12/16 | 11/16 | 0.18319 | 0.18456 | 0.59741 |

**Continued state and inference worked; improved model quality was not demonstrated.** The second cohort reduced task loss relative to step 32, but both learned stages remained worse than the starting model on these loss measurements. Final retention loss rose by 0.05305 nats per document; its descriptive normal 99% upper bound was +0.06398. Generated accuracy stayed at 23/32, with no correctness wins or losses. These results do not support promoting the final checkpoint as an improvement.

Both workers returned identical losses, generated tokens, decoded text and answer checks. Sequential generation through the two shards averaged approximately 17 output tokens/second on these short JSON tasks. The implementation recomputes the prefix at each token and has no KV cache. This is distributed inference functionality, not a concurrent serving benchmark or a ChatGPT-quality model claim.

The tasks are fresh cases of four public generated families, and retention uses previously exposed public conversations. The evaluated sets are disjoint from the two current training cohorts; replay membership is verified against examples actually trained in phase one. This small study cannot establish perpetual learning, broad assistant quality or useful automatic model growth.

## Validation and preservation

The CPU reference test passed: partitioned updates matched full-model autograd with tied weights, active clipping, unequal sequence lengths/weights and a partial microbatch. Fresh processes reproduced complete checkpoints and generation; corrupted tensor bytes and inconsistent optimizer cursors were rejected. GPU logs and full-state commitments independently establish the measured host-replacement result.

The first GPU evaluation failed because staging copied tokenizer JSON files but omitted `chat_template.jinja`. The missing file was copied from the frozen source assets, and both workers verified the original committed tokenizer identity before evaluation restarted. Numerical sources, data, training runs and checkpoints were unchanged. The failed attempt and correction are retained in the evidence.

Complete step-16, recovered step-32 and continued step-64 training states are retained as separately hashed objects, approximately 20.54 GB per checkpoint. Preservation includes full object readback, not only storage metadata checks. The [machine-readable report](../config/experiments/persistent-shards-results.json) includes paired comparisons, all checkpoint roots, failed-attempt details, evidence hashes and storage/cleanup receipts.

The 218-member evidence archive contains the exact inputs, numerical sources, all declared evaluations, preparation/controller scripts and checkpoint receipts. Its SHA-256 is `a453d3f2c08b66c8f1266fcfdfd851f517598f139f9635eb365c593a3cfef0df`; it is retained in operator storage under `research/sharded-training-20260914/evidence/` and passed full readback verification. Large model/optimizer files remain separate objects. Storage and availability here are operated services, without a permissionless availability proof.

All three temporary GPU instances, their root disks and the dedicated security group were removed, confirmed at 03:56:28 UTC. Both original native CPU instances remained running. The GPU compute upper estimate is $2.72; a conservative planning estimate including all host transmissions and a $10 allowance for other costs is $14.84. These are estimates rather than an AWS invoice, and retained S3 storage accrues separately. The experiment used no token issuance or native serving promotion.
