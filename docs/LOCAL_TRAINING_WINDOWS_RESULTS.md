# Four-worker training: September 13, 2026

**Four GPUs trained a shared 1.7B model and recovered exactly after a worker-process crash. Local training windows reduced communication substantially, but failed the quality-preservation contract.** The single-GPU and synchronized four-GPU controls learned these narrow tasks while satisfying the exposed retention bound. Local windows specialized more strongly and worsened retention. This recipe is not approved for native serving or continued model growth.

The [public evidence](https://neuroshard.com/experiments/local-training-windows-20260913/) retains all 1,024 test prompts and four answers per prompt, every predetermined candidate, training/recovery traces, inference measurements, failures, resources and checksums. The [methods](LOCAL_TRAINING_WINDOWS.md) and [frozen plan](../config/experiments/local-training-windows.json) distinguish the acceptance criteria. This operated experiment issues **zero NEURO** and changes no native model, consensus rule or paid-inference profile.

## What was fixed and tested

The initial remote launcher returned before its detached child finished. This overlapped training arms and exhausted GPU memory. Those attempts were stopped and preserved before any candidate selection or final scoring. The corrected launcher uses `setsid --wait`, propagates the child's exit status, and the driver takes an OS GPU lock shared by experiment directories under the same Unix user. A live competing process is rejected before model allocation.

The successful numerical source is `d03031ded9ab3a909377e5a117d47e2cc3bed86c`. The [input commitment](../config/experiments/local-training-windows-data-selection.json) has canonical identity `307be733a2ed647db1bdded7443df6569805443c4ad9e02ccb4dfb491b8f5a6e`. All three step-128 candidates and every rank's acknowledgments were [committed together](../config/experiments/local-training-windows-selection.json) at `00ab4a8f4f0d07cd24c2cf4b88a4550ee20b97f9`, before final-test generation. The numerical implementation, data, hyperparameters and quality thresholds were not changed after that commitment.

All arms start from the same pinned SmolLM2-1.7B-Instruct seed and train all **1,711,376,384 parameters**. They use the same 1,024-document schedule, global batch eight and 128 updates. There are 768 new executable training tasks and 256 previously published conversation-replay documents. Task tokens have weight eight and replay tokens weight one. The two four-worker methods assign two documents to each rank per update. Local windows keep separate Adam states and apply a common outer Nesterov update every 16 steps; this is a different optimization method from DDP.

## Learning and retention

| Final test | Seed | One GPU | Four GPUs, DDP | Four GPUs, local windows |
| --- | ---: | ---: | ---: | ---: |
| Correct / 1,024 | 105 | **641** | 613 | 557 |
| Exact accuracy | 10.25% | **62.60%** | 59.86% | 54.39% |
| Lookup / 256 | 102 | 247 | 224 | **256** |
| Status filtering / 256 | 2 | 206 | **223** | 152 |
| Priority sorting / 256 | 1 | **188** | 166 | 149 |
| Invoice totals / 256 | 0 | **0** | **0** | **0** |
| Mean retention loss change, nats | — | −0.002428 | −0.002885 | **+0.083294** |
| Approximate 99% upper retention bound | — | +0.006411 | +0.005859 | **+0.108140** |

Local windows gain 454 correct answers and lose two against the seed: **+44.14 percentage points**, exact one-sided paired p = **5.60 × 10⁻¹³³**. That passes the minimum task-gain condition. It does not rescue the other failures:

- Retention worsens on 114 of the 128 exposed conversation probes. The +0.108140-nat upper bound exceeds the declared +0.02 limit.
- Against DDP, 75 tasks improve and 131 regress, a net **−5.47 percentage points**. The approximate paired 99% lower change bound is **−9.05 points**, below the declared −3-point margin.

The primary quality contract therefore **fails**. The lower communication cost cannot justify promoting this candidate. Outer momentum, local Adam histories and local token normalization all differ from DDP; this comparison does not isolate which difference causes the quality loss. The perfect lookup score also does not imply general improvement: filtering, sorting and conversation retention expose substantial tradeoffs.

The test uses new generated records and includes wording variants absent from training, but the same four task definitions remain public. It measures one training run. Retention probes were already exposed and related SmolTalk data were used upstream. Neither the small p-value nor those probes certify broad assistant quality. All models still fail invoice arithmetic. The failed 135M learning contract and inconclusive earlier general 1.7B reference retain their original outcomes.

## Communication and complete execution cost

| Measurement | One GPU | Four GPUs, DDP | Four GPUs, local windows |
| --- | ---: | ---: | ---: |
| Maximum rank's active time | 208.09 s | 1,979.26 s | 250.08 s |
| Maximum rank's loop, with checkpoints | 266.12 s | 2,048.37 s | 494.51 s |
| Complete process group | 290.76 s | 2,078.61 s | 532.54 s |
| Allocated GPU seconds in the loop | 266.12 | 8,193.20 | 1,978.02 |
| All hosts, transmitted bytes | 0.000020 GB | 5,330.663 GB | 331.778 GB |

Active time includes updates and synchronization. Local windows use **12.64% of DDP's active time** and **6.22% of its transmitted bytes**, passing both efficiency thresholds. They remain **1.20× slower than one GPU in active time** and **1.86× slower through their checkpoints**, with four GPUs allocated. This experiment does not establish an economical training advantage over one GPU.

All hosts are g6e.xlarge instances with one L40S, four vCPUs and 32 GiB RAM. Capacity required three in us-east-1c and one in us-east-1d; both distributed arms use the same topology. Their 2.5 Gbit/s baseline and up-to-20-Gbit/s burst network materially affect timing. DDP's median update changes from 7.76 seconds over steps 2–90 to 33.15 seconds over steps 91–128, with increasing ENA bandwidth-allowance counters. Fixed arm order and checkpoint pauses affect network credits. The observed timing ratio is not an isolated causal estimate of algorithmic speedup. See [AWS's bandwidth rules](https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/ec2-instance-network-bandwidth.html) and [ENA counters](https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/monitoring-network-performance-ena.html).

Whole-host interface counters include transport and management traffic; receive bytes are retained separately and are not added again as independent transmitted data. A simple ring-payload calculation predicts 5,257.348 GB for 128 dense reductions and 328.584 GB for eight delta reductions, before overhead. Startup, checkpoints, recovery, backups and complete allocation costs remain separate. The instance's 125 MB/s baseline EBS limit also constrains the higher-throughput gp3 volumes.

## Recovery from a killed worker process

The recovery trial copies the uninterrupted step-64 checkpoint and retains the original rank identities and deadlines. Rank one is killed with SIGKILL after observed step 75. The coordinator detects the failed process, stops surviving processes and restores all ranks from step 64.

The failed group process lasts 179.62 seconds including checkpoint loading; failure is observed about 2.47 seconds after the injected kill, and survivors finish shutting down about 32.74 seconds after detection. The restarted process group takes 357.04 seconds, including restoration, repeated updates and its final checkpoint.

Every rank exactly matches the uninterrupted model, outer velocity, all Adam/RNG tensors and scalars, and the numerical trajectory. Each comparison covers 1,711,376,384 model elements, the same number of outer-state elements, and 3,422,758,058 optimizer/RNG tensor elements. The large checkpoint file hashes also match. Full host-side comparisons take 199.80–220.41 seconds and overlap operator backup I/O; they are separate from the restart process timer.

This tests process loss followed by coordinated recovery on the same four operated hosts. It does not test EC2 termination, loss of a disk or failure domain, replacing a rank, elastic membership, Byzantine workers or an independent operator. Comparison receipts are measurements, not cryptographic proofs of remote execution.

## Serving the fixed candidate

The predetermined local-window model remains the benchmark subject even though its quality gate failed. Each phase uses the same 64 development tasks, concurrency four, fresh request identities and identical four-case warmups. Bulk checkpoint transfers finish before warmups and measured requests.

| Providers | Completed | Requests / second | Median latency | 95th-percentile latency |
| --- | ---: | ---: | ---: | ---: |
| One | 64 / 64 | 3.008 | 1.285 s | 1.521 s |
| Four | 64 / 64 | 9.025 | 0.321 s | 0.884 s |
| Four advertised, one stopped | 64 / 64 | 6.713 | 0.450 s | 1.028 s |

Four providers deliver **3.00× aggregate request throughput** relative to one under this workload. The outage phase completes all requests with 16 retries. Every successful response has identical output tokens across phases and no response is served from the retry cache.

After all three phases completed, the controller reported a cleanup error because it tried to stop the already removed transient service a second time. Final host checks showed zero GPU memory in use on every host. The executed controller and error are retained; its final cleanup now targets only the three remaining replicas. No serving measurements were rerun.

This is immediate unavailability of a known provider, with bounded retries to other operated endpoints. It does not test a slow partition, a crash between execution and acknowledgment, elastic admission, general private chat or native payments. Replicas increase service capacity here; they do not make this rejected model better.

## Artifact preservation and resources

All three predetermined model archives are downloadable, including the failed local-window candidate. Each archive and every member was verified after transfer. The complete original and recovered checkpoint inventory has 267 references and **177.992 GB of unique objects**, retained in S3 with full hash-checked readback. These full optimizer/outer states are private operator backups; their manifest is public.

The initial backup relayed 95.843 GB through the coordinator. The remaining 82.149 GB uses direct per-worker S3 uploads and readback after single-part and multipart probes. Workers receive only 15-minute capabilities for specific objects or upload parts; AWS secret keys remain on the coordinator. Previously completed readback receipts are reused only after matching object size, metadata and ETag. This removes the coordinator from those bulk transfers, but does not establish permissionless data availability.

All four temporary GPU instances, their root volumes and the dedicated security group were removed at **2026-09-13T21:10:13.731855+00:00**. Compute through confirmed cleanup is bounded above by **$16.97** at $1.861 per instance-hour. A conservative transfer estimate plus a $10 allowance for storage, setup and other traffic brings this allocation to **$88.69 before credits**, within its $100 plan. This is planning accounting, not an AWS invoice. It includes failed attempts, loading, evaluation, recovery, backup, the compression probe and serving. Retained S3/site storage and subsequent public downloads continue to incur costs. The existing CPU network remains running separately.

## Compression feasibility after the quality failure

A separate [12-update probe](../scripts/probe_gradient_compression.py) investigates [PowerSGD](https://arxiv.org/abs/1905.13727): compress each synchronized gradient while retaining a common model and replicated Adam optimizer. Its rank-32 configuration completes eight dense warmup updates before four compressed updates. It uses only the existing training partition on two same-zone GPUs, and makes no new quality claim.

The first real-model attempt, at `cfe9e63`, completes step nine and fails on step ten: rank one's resident compression errors leave insufficient memory for the next response-loss calculation. The failed logs and process receipts remain public. Keeping dense FP32 error feedback on CPU between completed bucket reductions fixes this memory failure. Projection state remains on GPU. The wrapper completes each bucket's collective sequence before dispatching the next and copies residuals without quantization; this sacrifices overlap and adds PCIe transfers.

The fix is at `856c9cb4f32a3edf40ce62e0462585e51f7f44c1`. Small-model conformance checks pass on Gloo and on both real CUDA hosts. A strengthened check at `ebaef040272fa3648eafa42a6f9914fc1e17a574` also passes on both backends, comparing complete projection, input and PyTorch random-generator states, including their positions. Model, Adam, error-feedback, projection and loss states agree exactly between resident and offloaded wrappers, plus the unmodified upstream hook in a single-bucket control. The full 1.7B probe then completes all 12 updates on both GPUs, with maximum allocated CUDA memory **36.89 GiB** and a shared parameter digest `242d50b2ec9850394fbbad82174d61ae65cc8287e8a487fe910f81fdbbfe8642`.

The upstream counters report **45.14× logical compression during the four compressed updates**. This excludes the eight dense warmup updates, initial model broadcast and wire overhead. Complete process time is 106.49 seconds, with operator artifact I/O running concurrently. It is not a matched sustained-throughput result. No held-out set is scored, no probe checkpoint is selected for serving, and no recovery codec is implemented for compression errors, random projections or bucket layout.

This closes a concrete memory/transport feasibility issue. A correction to the failed learning recipe still needs a full-length compressed-gradient comparison, fresh committed test cases, retention checks and recovery that preserves every compression state. The current local-window quality failure remains unchanged.
