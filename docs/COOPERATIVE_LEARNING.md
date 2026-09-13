# Two-GPU learning and serving experiment

This operated development experiment asks whether checked training targets help a shared model learn a narrow, useful task, what it costs to cooperate across two GPUs, and how serving capacity changes with a second replica. It does not change native consensus, issue NEURO or establish independent ownership. The previous [GPU result](LEARNING_REFERENCE_RESULTS.md) remains inconclusive on general assistant improvement.

The [completed results](COOPERATIVE_LEARNING_RESULTS.md) pass the narrow learning comparison, record slower synchronous training and measure higher inference throughput with bounded failover. All final-test answers and raw measurements are public.

The [frozen plan](../config/experiments/cooperative-learning.json) uses the same pinned 1.7B seed and numerical dependencies as the [reference](LEARNING_REFERENCE.md). Two g6e.xlarge instances each provide one L40S, four vCPUs and 32 GiB host RAM. Together they fit the reported eight-vCPU GPU quota. The observed on-demand price in us-east-1 is $1.861 per instance-hour; an eight-hour automatic stop deadline bounds combined compute to $29.776, with additional room for disk/traffic under a $100 experiment ceiling. Provisioning remains operator infrastructure, not a public admission service.

## Questions and fixed comparisons

1. **Learning:** a clean single-GPU arm trains on 768 generated, executable tasks plus 256 public conversation-replay records. A control uses the same prompts and schedule but deliberately damages approximately one quarter of the generated targets. This is an artificial target-quality ablation, not a claim that an upstream corpus has that error rate. The original seed is evaluated without updates.
2. **Cooperation:** a two-GPU arm trains the clean corpus with the same 128 updates, effective global batch of eight and learning-rate schedule. Each rank contributes four documents per update. This measures synchronous data parallelism over real network links. The complete model fits each GPU; this experiment does not demonstrate pooling memory for a model that fits neither host alone.
3. **Serving:** load the predetermined clean-pair final checkpoint on both hosts, independently of its final-test score. Measure a fixed development workload with one and then two replicas at the same client concurrency, verify output and model identities, stop a replica, and retry bounded requests against the survivor. A second replica may improve throughput and availability; it does not inherently improve answer quality or make a single request faster. This connects the cooperatively trained artifact to an operated serving test, without native promotion.

No checkpoints are selected by final-test score. Every arm's predetermined final checkpoint must be committed before final-test evaluation. All arms and failures remain in the report.

## Generated tasks and learning contract

The [task generator and checker](../src/neuroshard/evolution/grounded_tasks.py) cover lookup, status filtering, integer invoice totals and priority ordering. The answer is a strict JSON object computed from records supplied in the prompt. Missing lookups, zero quantities, ties, malformed JSON, duplicate keys and wrong numeric types have explicit semantics. The test includes fresh generated records and two prompt variants absent from training. This measures these task families; it is not a broad reasoning benchmark or newly acquired world knowledge.

The training targets are checked before tokenization. Each grounded target token has weight eight and each replay target token weight one, with the denominator equal to the global sum of weighted targets. This choice was made from target counts before any model scoring: short JSON targets otherwise contribute only about 11% of supervised tokens despite being 75% of documents. It gives grounded tasks approximately half the target weight. Evaluation remains unweighted. The damaged-target arm is explicitly marked and never eligible for serving. The same 256 unmodified conversation records are replayed in both arms. They come from the already published reference training data; retention uses its previously exposed 128 response-loss probes. The task dataset, source identities, target counts, corruption identities and all role hashes are committed before training. Test content is publicly reconstructable; sequencing is enforced by the driver, not secrecy.

The eligible clean-single arm must improve exact task accuracy on the 256 final-test examples by at least five percentage points, with an exact one-sided paired sign/McNemar p-value at most 0.01, and retention's approximate paired 99% upper loss-change bound must be at most +0.02 nats. This is the only primary comparison. The damaged-target comparison, family/paraphrase breakdowns and pair-versus-single results are descriptive. Previously exposed retention probes cannot certify general retention. Passing this narrow contract does not authorize native serving promotion or unlock the earlier failed 135M contract.

## Distributed arithmetic and limits

PyTorch [DistributedDataParallel](https://docs.pytorch.org/docs/2.9/generated/torch.nn.parallel.DistributedDataParallel.html) averages gradients across ranks. Each local summed response loss is multiplied by `world_size * record_weight / global_weighted_target_count` before backward, so variable answer lengths preserve the intended global token-weighted objective. Accumulation suppresses intermediate reductions with `no_sync`, including the forward pass. Clipping and AdamW occur after the global reduction. Tests compare several real two-process updates against the single-process reference with unequal target counts.

The ranks agree on prepared-input and numerical-profile digests before training and on final parameter hashes afterward. Fixed-byte digest exchanges avoid pickle deserialization. CPU/GPU runtime settings, rank identities, all global/local document assignments, step time and host network counters are retained. Same-group replicas should agree exactly; one-rank and two-rank reduction orders need not produce bit-identical weights. Loss, exact-answer differences and resource cost must be measured separately. Host network counters include other host traffic and are reported as such.

The training-loop timer includes checkpoint writing, verification and final parameter agreement. It excludes initial seed-file verification, model loading and process-group setup. Keep whole-process resource logs and the complete instance lifetime alongside these timers; allocated machine cost also includes initialization, evaluation, staging and idle time.

Synchronous FP32 gradients for this model are about 6.85 GB per update, before transport overhead. A second GPU can therefore make this small training job slower if networking dominates. [DiLoCo](https://arxiv.org/abs/2311.08105) motivates less frequent communication between learning groups, but changing synchronization and the outer optimizer also changes the learning algorithm. That needs a later matched-quality experiment; it is not assumed to preserve this result.

## Reproduction

Use dedicated hosts and the pinned [GPU requirements](learning-reference-requirements.txt), with NCCL communications restricted to the experiment's private network. Keep existing validator environments intact.

```bash
PYTHONPATH=src python scripts/run_cooperative_learning.py prepare \
  --home /path/cooperative --model-dir /path/pinned-seed \
  --reference-home /path/previous-reference

PYTHONPATH=src python scripts/run_cooperative_learning.py train \
  --home /path/cooperative --model-dir /path/pinned-seed --arm clean-single
```

Run the damaged-single arm on the other GPU with its own experiment directory. For the pair, run one process on each host with `WORLD_SIZE=2`, distinct `RANK` values 0 and 1, a shared private `MASTER_ADDR`/`MASTER_PORT`, and `--arm clean-pair`. Every rank must have the same prepared inputs. The rank-zero checkpoint contains full model/Adam/RNG state. A coordinated resume requires the verified checkpoint and pointer on both hosts, with unchanged numerical profile and original deadlines. This is not elastic membership or automatic Byzantine recovery.

Use `evaluate --role dev --arm seed|clean-single|damaged-single|clean-pair` for development evidence. Collect and verify all three final checkpoints before `select`, commit the resulting selection, then use `evaluate --role test --selection config/experiments/cooperative-learning-selection.json`. All generated answers and strict-check outcomes are retained.

The [serving probe](../scripts/serve_compute_probe.py) accepts only prepared development task IDs on a private endpoint. Its bounded in-memory retry cache is a benchmark convenience; it is not durable exactly-once billing. The [replica benchmark](../scripts/benchmark_compute_replicas.py) reports completed requests, latency, wall time and every retry. A common operator and coordinator remain trusted throughout. General chat, private prompts, public GPU settlement and independent participation require further protocol work.

Collect both ranks' result files, all four final-test evaluations, `selection.json`, and the three serving reports (`serving/single.json`, `pair.json`, `failure.json`) alongside the prepared inputs. Recompute comparisons with:

```bash
PYTHONPATH=src python scripts/report_cooperative_learning.py \
  --home /path/collected-evidence --output /path/new-report.json
```

The report checks candidate identities, recomputes strict answers, checks the fixed global schedule and compares output tokens across all serving phases. Failed or inconsistent evidence stops report generation and remains available for inspection. This validates the consistency of an operated experiment's evidence; it is not a cryptographic proof that remote training occurred.
