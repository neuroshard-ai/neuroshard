# Four-worker local training experiment

This development experiment tests a concrete bottleneck from the [two-GPU result](COOPERATIVE_LEARNING_RESULTS.md): synchronizing full FP32 gradients on every update cost more time than a second GPU saved. It compares less frequent synchronization with a matched four-worker control, then tests recovery and serving. It does not change native consensus, pay GPU workers, or establish permissionless verification.

## Frozen comparisons

The [plan](../config/experiments/local-training-windows.json) fixes the same public SmolLM2-1.7B-Instruct revision and [numerical dependencies](learning-reference-requirements.txt). All model parameters train. Each arm starts from the seed and processes the same 1,024-document schedule in 128 updates with global batch eight:

| Arm | GPUs | Documents per worker per update | Synchronization |
| --- | ---: | ---: | --- |
| single | 1 | 8 | Local AdamW |
| ddp-four | 4 | 2 | Token-weighted global gradient each update |
| diloco-four | 4 | 2 | Local AdamW; average endpoint deltas every 16 updates |

There are 768 fresh executable tasks and 256 previously published conversation-replay documents. Each task target token has weight eight; each replay target token has weight one. The task families remain lookup, filtering, invoice arithmetic and ordering. The new seed excludes all task identities and prompts in the prior experiment. Development uses 64 fresh cases; the final test uses 1,024 cases, including prompt variants absent from training. The 128 retention probes are previously exposed public response-loss examples. These probes cannot certify broad retention or general assistant quality.

The numerical implementation and complete input receipts must be committed before training. All three predetermined final candidates and all ranks' acknowledgments must be collected before selection. Final scoring additionally requires that selection to be committed. The driver preserves each started evaluation attempt and refuses silent replacement. Development scoring also uses the complete selection; it does not select intermediate checkpoints.

## Outer update

The implementation follows the local AdamW and outer Nesterov structure of [DiLoCo](https://arxiv.org/abs/2311.08105). For common parent `x`, worker endpoint `y_r`, four workers, outer learning rate `eta = 0.7` and momentum `mu = 0.9`:

```text
d = mean_r(x - y_r)
v = mu * v + d
x = x - eta * (d + mu * v)
```

Every worker replaces its model with `x` and retains its own local Adam moments. Uniform endpoint averaging is used for equal document assignments. Each local inner loss is normalized by that worker's weighted target count; this differs from DDP's global token normalization when response lengths differ. The algorithm and numerical trajectory therefore change. Accuracy preservation must be measured.

Common FP32 parent and momentum arrays live on CPU. Reductions use bounded 32 MiB chunks on GPU, avoiding two extra full-model GPU arrays. The implementation exchanges deltas rather than summing large parent weights. It records logical payload separately from whole-host network counters. NCCL Ring/Simple settings constrain this experiment's reduction profile; they are not a promise of exact replay on arbitrary hardware or optimal production tuning. See the [NCCL documentation](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/env.html).

## Separate acceptance criteria

The primary candidate is always `diloco-four`, independent of its test score. Its exact-answer gain over the seed must be at least five percentage points with a paired one-sided exact p-value at most 0.01. Its exposed retention probes must have an approximate paired 99% upper loss-change bound no greater than +0.02 nats. Against `ddp-four`, the approximate paired 99% lower accuracy-change bound must be at least -0.03. This is one run on a narrow generated distribution, not broad assistant equivalence.

The communication contract requires transmitted bytes no greater than one eighth of DDP's measured traffic. Active training time, including local updates and synchronization, must be no greater than half DDP's. Checkpoint time, startup, complete process time, allocated GPU time and complete EC2 cost are reported separately. An active-time pass cannot conceal expensive recovery or checkpoint storage. Single-GPU results remain visible even if four-worker local training improves on DDP.

## Durable recovery

Each rank stores its full model, local Adam moments, RNG state, local assignment journal and, for local training, outer momentum. Hash-verified file receipts are durable before the group publishes a common manifest. The manifest names every rank's receipt. A restart selects the latest identical manifest actually present on every worker and verifies local files. A local latest pointer alone cannot advance the group.

Rank is part of checkpoint identity: another worker's Adam state cannot replace it. Hostname is recorded but excluded from numerical identity so an identical replacement host can restore the same rank. Outputs beyond the last common durable point are preserved as orphaned attempts. A missing or corrupt checkpoint stops execution. There is no automatic acceptance of an unverified substitute.

The recovery trial starts from copies of the uninterrupted run's step-64 group checkpoint, with original rank identities and deadlines. Rank one is killed after step 75. The coordinator must observe a bounded group failure, stop surviving processes, and restart the same membership from step 64. Final model parameters, outer momentum, and every rank's complete optimizer/RNG contents must equal the uninterrupted result. Compare tensor contents rather than assuming `torch.save` file hashes are canonical. This tests coordinated recovery under one operator; it does not test elastic membership or Byzantine behavior.

## Execution and evidence

Use four separate L40S hosts, one GPU each, with a private NCCL network and identical environments. The operated AWS trial is bounded to $100 and an eight-hour automatic stop, with 300 GiB encrypted gp3 disks and the same throughput/read-ahead configuration for all arms. The reported quota of 30 GPU vCPUs permits the intended 16-vCPU allocation; actual instance launch remains the capacity check. Provisioning and storage are operated infrastructure.

```bash
PYTHONPATH=src python scripts/run_local_training_windows.py prepare \
  --home /path/experiment --model-dir /path/pinned-seed \
  --previous-home /path/previous-cooperative-evidence
```

Commit the exact `prepared.json` as `config/experiments/local-training-windows-data-selection.json`. Run `train --arm single` with one rank. For each four-worker arm, provide identical inputs on all hosts, `WORLD_SIZE=4`, distinct `RANK=0..3`, a common private `MASTER_ADDR`/`MASTER_PORT`, and `NCCL_ALGO=Ring NCCL_PROTO=Simple NCCL_IB_DISABLE=1`. Each host keeps its own experiment directory. Restrict NCCL ports to these experiment hosts.

Collect every rank's `result.json`, checkpoint receipt and common manifest into rank zero's matching directory structure. Run `select`, commit its `selection.json` as `config/experiments/local-training-windows-selection.json`, and then run `evaluate --role test --arm seed|single|ddp-four|diloco-four`. Model-only evaluation copies may omit optimizer and outer-state files; their receipt and every model file are still verified against the selected artifact. Full recovery artifacts must remain available separately.

The selected local-training model also serves 64 fixed development requests at client concurrency four: one replica, four replicas, then four advertised endpoints with one stopped. At most four attempts per request are permitted. Compare model identities, generated tokens, throughput, failures and retries. This fixture accepts committed task IDs only. It is not general public chat or token billing.

Preserve raw schedules, failures, environment and network measurements, all final answers and full recovery states. Content-addressed, read-back-verified operator backups protect the experiment's artifacts; they do not establish permissionless data availability. Stop the experimental GPUs after evidence preservation. The live native network remains a separate CPU training and settlement system.
