# Two-GPU learning and serving: September 13, 2026

**A 1.7B model learned the measured lookup, filtering and sorting tasks; two machines trained a shared checkpoint and served it with higher aggregate throughput.** The shared run was slower to train than the single-GPU baseline, and all candidates failed the arithmetic task. These results establish useful components under one operator, with separate work still required for broad assistant quality, economical distributed training and permissionless verification.

The [complete public evidence](https://neuroshard.com/experiments/cooperative-learning-20260913/) contains all 256 final-test prompts and all four answers per prompt, raw scores, per-rank training journals, serving requests/retries, downloadable weights, resources and checksums. The [experiment contract](COOPERATIVE_LEARNING.md) and [frozen plan](../config/experiments/cooperative-learning.json) define the comparisons. This experiment issued **zero NEURO**, made no native training or promotion transaction and did not replace the public 0.4.0 model.

## Inputs and candidate selection

All three arms begin at the same pinned SmolLM2-1.7B-Instruct seed and train all **1,711,376,384 parameters** for 128 AdamW updates, with a global batch of eight. There are 768 generated grounded tasks plus 256 complete public conversation-replay records. The clean targets are computed and checked from the supplied records. The control deliberately damages 203 of the 768 grounded targets, leaving prompts, replay records and the global document schedule unchanged. This is a controlled artificial corruption, not an estimate of upstream data quality.

Grounded response tokens receive weight eight; replay tokens receive weight one. These weights were frozen from target counts before model scoring. The clean arm has 84,133 unweighted response targets and 150,514 weighted targets; the damaged arm has 84,660 and 154,730 respectively. Changed target strings therefore change token counts even though document count and schedule match. Evaluation is unweighted. Contexts fit within 1,024 tokens without truncation; generation is greedy with a 96-token bound.

Training runs at commit `a7efc6c67f3c7fd7a0e67db3a57675ac668fb153`. The [prepared input commitment](../config/experiments/cooperative-learning-data-selection.json) has canonical identity `4304122faebe6a7bf8619a82f4297c81d213f2343a1c994b6e44317f0bdd8c3b`. All three predetermined step-128 checkpoints were [committed together](../config/experiments/cooperative-learning-selection.json) at `394fb106f7fd7f9fe6b1b36989ebccf54de45c42` before final scoring. The clean-pair checkpoint was chosen for the serving comparison before those scores were available. No intermediate checkpoint was selected by final-test performance.

## Exact task performance and limits

Correctness requires both the requested JSON schema and exact values, including ordering, missing records, ties and integer types. The report rechecks every retained answer rather than trusting stored totals.

| Final test | Seed | Clean, one GPU | Damaged, one GPU | Clean, two GPUs |
| --- | ---: | ---: | ---: | ---: |
| All tasks, out of 256 | 23 | **169** | 152 | 168 |
| Accuracy | 8.98% | **66.02%** | 59.38% | 65.62% |
| Lookup, out of 64 | 23 | 62 | 60 | 63 |
| Status filter, out of 64 | 0 | 60 | 47 | 58 |
| Priority sorting, out of 64 | 0 | 47 | 45 | 47 |
| Invoice totals, out of 64 | 0 | **0** | 0 | 0 |

The primary clean-single comparison gains 146 correct answers and loses none of the seed's correct answers. Its gain is **57.03 percentage points**, with exact one-sided paired McNemar p = **1.12 × 10⁻⁴⁴**. Mean response-loss change on the 128 public retention probes is **−0.002195 nats**; the approximate paired 99% upper bound is **+0.006656**, below the frozen +0.02 limit. It passes this experiment's narrow learning contract. The test is conditional on these generated examples and this one trained seed; its small p-value does not quantify broad intelligence or variation across training runs.

Fresh generated records appear in every test example. Prompt variants 2 and 3 were absent from training: clean-single answers 83/128 correctly on those wordings, versus 9/128 for the seed. The underlying four task definitions are shared with training. All clean-single and clean-pair outputs are valid JSON objects; the seed produces 198/256 valid objects. Format compliance explains part of the improvement, while exact record values remain necessary for credit.

The arithmetic failure is substantive. For example, five invoice rows have a checked total of 365, but clean-single and clean-pair both answer 431. Their outputs satisfy the JSON schema and contain the wrong number. This recipe does not teach reliable multiplication and accumulation. The complete evidence retains all 64 arithmetic failures per model, as well as sorting and lookup errors.

The clean control advantage is descriptive: 27 tasks improve and 10 regress versus the damaged arm, a net 17/256. The paired p-value is 0.00382, without correction for exploratory comparisons or repeated training seeds. This supports testing target quality further; it does not establish a universal data-cleaning effect. The pair differs from clean-single on seven correctness outcomes, gaining three and losing four, within the predetermined two-percentage-point descriptive margin.

Retention probes were already exposed by the earlier GPU reference, and related SmolTalk data were used upstream. They cannot certify general retention. The [failed 135M contract](LEARNING_MILESTONE_RESULTS.md) and [inconclusive general 1.7B reference](LEARNING_REFERENCE_RESULTS.md) keep their original outcomes. This new, narrower pass does not unlock their later phases or approve native serving.

## Cooperation: agreement with substantial communication cost

Each of two g6e.xlarge hosts has one L40S, four vCPUs and 32 GiB RAM. Both are in the same AWS availability zone and communicate over private TCP/NCCL. Parameters, gradients and Adam moments are FP32, with BF16 forward/backward autocast. Each rank processes four documents per global update. Weighted loss normalization preserves the intended eight-document objective under DDP's gradient averaging.

| Measurement | Clean, one GPU | Damaged, one GPU | Clean, two GPUs |
| --- | ---: | ---: | ---: |
| Sum of update timers | 208.70 s | 209.55 s | 917.03 s |
| Median update | 1.61 s | 1.62 s | 7.12 s |
| Training loop through final digest | 986.84 s | 665.01 s | 1,294.04 s |
| Allocated GPU seconds in that loop | 986.84 | 665.01 | 2,587.95 |
| Peak allocated CUDA memory per rank | 31.60 GB | 31.60 GB | 38.44 GB |

The pair is **4.39 times slower in update time**, **1.31 times slower in loop wall time**, and consumes **2.62 times the allocated GPU seconds** of clean-single. These are one-run measurements, not stable hardware benchmarks. Checkpoint I/O varies considerably even between the two single-GPU controls. Loop timers include checkpoints and final hashing but exclude seed verification, model loading and distributed setup; full process resource logs and complete EC2 lifetimes are retained separately.

The pair's hosts transmit 890.33 GB and 884.94 GB respectively during the measured loop, about **1.775 TB combined transmitted traffic**. These are whole-interface counters, including small management traffic, not isolated NCCL telemetry. Received bytes are reported separately and are not added again as independent transferred data. Synchronizing approximately 6.85 GB of FP32 gradients per update makes communication a major cost for this topology.

Both ranks finish with the identical parameter digest `e7f65d52f3a8f40edcc83e2e2ec6d2132d06d427844f566f90efe3b073a70b35`. Their parameters differ from the one-GPU result, as reduction order can change floating-point rounding: relative L2 difference is 0.000322 and maximum absolute difference 0.000801 across all 218 tensors. The comparable task results are measured separately from byte agreement. A preliminary two-host CUDA test also passes the weighted-update check.

The full model and optimizer fit each GPU. This is data parallelism, with complete replicas; it does not pool memory, grow the model, establish independent ownership or verify an adversarial worker. There was no full-model training failure/recovery trial in this experiment. Its serving failover must not be described as training recovery.

## Serving the shared checkpoint

The clean-pair checkpoint is loaded on both GPUs. Each phase uses the same first 32 committed development tasks, concurrency two, fresh request identities and at most two attempts. Separate four-request warmups precede measurement. The phases run once in the fixed order single, pair, failure; no benchmark run is selected from repeated attempts.

| Phase | Successful requests | Wall time | Requests/s | Median latency | p95 latency |
| --- | ---: | ---: | ---: | ---: | ---: |
| One replica | 32/32 | 11.02 s | 2.90 | 0.683 s | 0.873 s |
| Two replicas | 32/32 | 6.61 s | **4.84** | 0.335 s | 0.663 s |
| One of two stopped | 32/32 | 11.01 s | 2.91 | 0.684 s | 0.883 s |

The second replica delivers **1.67 times aggregate throughput** for this workload. All 96 completed requests produce identical task-matched output token sequences across phases, with **zero cache hits**. Stopping rank one's serving process before the failure workload causes 16 failed first attempts and 16 successful retries against the survivor. Each request retains its original identity during retry. The failure was a stopped service with immediate connection refusal, not a slow network partition or an interruption after execution but before acknowledgement.

Completed requests are not necessarily correct answers. These are short fixed tasks on private endpoints, one client, two operated providers and a volatile retry cache. The experiment does not measure long conversations, continuous batching, public provider discovery, durable exactly-once billing, a many-peer load envelope or token settlement. Both replicas return the same trained model; replication increases service capacity without itself making that model smarter.

## Artifacts and next development

The public archives contain exact FP32 weights, tokenizer files, upstream attribution and modification notes for clean-single and clean-pair. Optimizer/RNG state is omitted from these downloads. The operator retains the complete clean-single final checkpoint and verified weights for all three trained arms; other recovery states require rerunning the pinned recipe. All GPU artifacts were verified against the committed candidate receipts before resource cleanup.

Both hosts use 200 GiB encrypted gp3 root volumes. Root-device read-ahead is 128 KiB throughout training and changes to 4,096 KiB afterward for artifact staging and evaluation. The measured training recipe is unchanged. The [resource receipt](https://neuroshard.com/experiments/cooperative-learning-20260913/resources.json) records launch, termination, volume deletion and a complete-lifetime compute upper estimate at $1.861 per host-hour. The $100 ceiling includes an eight-hour automatic stop deadline. The compute estimate includes setup and idle time but excludes storage, transfer, public IPv4 and taxes; it is not an AWS invoice.

The immediate training direction is a matched-quality experiment with less frequent communication between groups, explicit outer-optimizer semantics and fresh task cohorts. [DiLoCo](https://arxiv.org/abs/2311.08105) is relevant prior work, not an algorithm implemented by the synchronous run here. Broad reasoning, arithmetic and instruction-following probes need stronger targets and retention coverage before an assistant-facing promotion. Subsequent cohorts must demonstrate retained gains rather than reusing this exposed test as new confirmation.

The native integration still needs a defined GPU execution/verification profile, funded independent checking, artifact availability and admission rules. Useful model growth requires memory-partitioned execution plus quality and cost evidence. More replicas are already useful for measured service capacity; those separate requirements determine whether many contributed machines can sustain a larger, trustworthy shared model.
