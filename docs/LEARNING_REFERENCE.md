# A useful-assistant learning reference

**Development experiment, September 2026.** The next objective is a useful conversational model with measurable improvement from additional training. The [earlier frozen experiment](LEARNING_MILESTONE_RESULTS.md) remains rejected. Its continual-learning and scaling phases are not unlocked by this work.

The [reference driver](../scripts/run_learning_reference.py) provides an operated full-model baseline before adapting a successful recipe to the permissionless protocol. It uses the same open model family, complete conversations, AdamW, and CPU or GPU execution. It has no ledger or token-payment interface. GPU arithmetic and AdamW are not supported native transitions merely because this research process can execute them.

## Why this experiment

The original 135M run combined a small instruction model, short response windows, plain SGD, and the distributed execution implementation. Its rejected result cannot by itself identify which choice limits learning. A conventional training baseline gives us a comparison whose optimizer, context, memory and throughput can be measured independently.

The first larger seed is [SmolLM2-1.7B-Instruct](https://huggingface.co/HuggingFaceTB/SmolLM2-1.7B-Instruct/tree/31b70e2e869a7173562077fd711b654946d38674), with 1,711,376,384 parameters and an Apache-2.0 license. Its upstream card reports instruction-following and conversational evaluations; those are upstream results, not NeuroShard measurements. Sharing the Llama architecture with the existing fixture makes numerical comparison more tractable. This is a candidate baseline, not a permanent size ceiling or a claim of frontier-model capability.

The upstream [training recipe](https://github.com/huggingface/alignment-handbook/blob/main/recipes/smollm2/sft/config.yaml) uses an adaptive optimizer, substantially longer contexts and supervised instruction training. Our learning rate and budget differ because we are continuing from an instruction-tuned checkpoint. The reference is not an attempted reproduction of the upstream training run.

## Inputs and observations

The [development plan](../config/experiments/learning-reference.json) specifies:

| Item | First GPU trial |
| --- | --- |
| Model | Pinned SmolLM2-1.7B-Instruct; every parameter trainable |
| Training data | 2,048 complete conversations from pinned Apache-2.0 Smol-SmolTalk |
| Development / retention / final test | 128 / 128 / 256 documents, disjoint source ranges |
| Sequence length | At most 1,024 tokens; complete conversations only |
| Training | 256 steps, 8 documents per effective batch, AdamW, peak learning rate 0.00003 |
| Schedule | 16 warmup steps, cosine decay to 10% of the peak, global clipping at 1 |
| Numerical execution | FP32 parameters and Adam moments; BF16 autocast on CUDA; FP32 on CPU |
| Generation | 20 conversations per comparison, up to 256 new tokens, original complete prompts |
| Resources | Six-hour limit per training/evaluation phase, 120 GiB artifact limit |

Preparation records source revisions, model-file hashes, tokenizer identity, selected document IDs, complete token/label arrays, rejected-document counts and implementation identity. Held-out roles are prepared before training. Exact normalized prompts/documents and the existing SimHash heuristic exclude duplicates across the selected roles. This does not prove semantic non-contamination. The seed already had related SmolTalk training; an unused NeuroShard source range is not proof of unseen upstream data or newly acquired real-world knowledge.

Each real assistant response, including its actual EOS, contributes targets. User prompts, role markers and padding do not. Oversized documents are excluded with an explicit count; there is no silent left truncation, partial-answer supervision or cross-document packing. These exclusions bias the experiment toward conversations within the declared length, and the report must retain that limitation.

Gradient accumulation weights each document by its number of supervised answer tokens. Checkpoints preserve the full model, both Adam moments, random state, the consumed-document journal and input/runtime identity. Resume verifies file hashes and can recover a completely written checkpoint whose pointer was lost. Work after the last checkpoint may be repeated; it receives no network payment. The original phase deadline survives a restart.

## Run on a dedicated GPU host

One [g6e.2xlarge](https://aws.amazon.com/ec2/instance-types/g6e/) provides a 48 GB NVIDIA L40S, 8 vCPUs and 64 GiB host memory. Approximately 27.4 GB are needed for four FP32 arrays across 1.711B parameters before activations and temporary storage. The actual memory and throughput must be measured. A 24 GB device cannot hold this specific full FP32 AdamW state; alternative optimizer/storage profiles need separate comparison.

Use 200 GiB encrypted gp3 storage and an AWS [GPU base image](https://docs.aws.amazon.com/dlami/latest/devguide/aws-deep-learning-x86-base-gpu-ami-ubuntu-22-04.html) with a compatible NVIDIA driver. Keep the existing validator hosts available. The initial operator budget is $100, with one instance, a verified hourly rate and an automatic shutdown deadline no later than 12 hours after launch. A stopped instance still incurs storage charges; archive required evidence and account for retained volumes. An AWS budget alert alone is not a spending cutoff.

Create a separate Python environment; never install CUDA dependencies over a released validator environment:

```bash
python3 -m venv .neuroshard/reference-venv
.neuroshard/reference-venv/bin/python -m pip install -r docs/learning-reference-requirements.txt
.neuroshard/reference-venv/bin/python -m pip install --no-deps -e .

.neuroshard/reference-venv/bin/python scripts/run_learning_reference.py fetch-model \
  --home .neuroshard/reference-gpu --model-dir .neuroshard/reference-model
.neuroshard/reference-venv/bin/python scripts/run_learning_reference.py prepare \
  --home .neuroshard/reference-gpu --model-dir .neuroshard/reference-model
.neuroshard/reference-venv/bin/python scripts/run_learning_reference.py run \
  --home .neuroshard/reference-gpu --model-dir .neuroshard/reference-model --device cuda
```

`fetch-model` checks the pinned revision and verifies the model files against upstream Git/LFS object identities. `prepare` creates immutable inputs and refuses to overwrite an existing prepared home. `run` evaluates the seed, trains, evaluates the candidate on development/retention data, publishes complete response transcripts and writes `selection.json`. Repeating `run` resumes the same environment/checkpoint or returns the completed outcome. It cannot load the final test partition.

After reviewing development evidence, commit the exact selected candidate before opening the final test:

```bash
cp .neuroshard/reference-gpu/selection.json config/experiments/learning-reference-selection.json
git add config/experiments/learning-reference-selection.json
git commit -m "Commit the GPU reference candidate before final evaluation"
.neuroshard/reference-venv/bin/python scripts/run_learning_reference.py score-test \
  --home .neuroshard/reference-gpu --model-dir .neuroshard/reference-model --device cuda \
  --selection config/experiments/learning-reference-selection.json
```

The test documents are publicly reconstructable. The driver enforces sequencing; it does not make the holdout secret or prevent an operator from examining public files. A final-test result is descriptive development evidence. A confirmatory improvement claim needs a separately committed quality contract, unused evaluation examples, and an answer-quality assessment defined before candidate selection. The earlier failed contract cannot be retrospectively changed.

## Verification and next decisions

Tests cover assistant/EOS masking, complete-context rejection, shared-prompt contamination, unequal-length gradient accumulation against an independent padded-batch calculation, Adam-state recovery, lost checkpoint pointers, corrupt artifacts, durable budgets and the Git commitment required to open the test.

A [two-step 135M CPU smoke run](../config/experiments/learning-reference-cpu-smoke.json) exercised the real driver and full optimizer before the GPU experiment. It used four training documents and two documents per evaluation role, with short generation limits to check execution. It completed in 68.73 seconds, retaining about 3.51 GB of artifacts, and a repeated command returned the saved outcome without retraining. Development loss worsened on that tiny sample while retention improved. This is functional verification, not a useful-learning result. The final test remained unopened. To repeat the smoke workflow, use that plan with `--plan`, `--device cpu`, and separate model/experiment directories in the commands above.

Assess the GPU result using answer quality, paired document loss, retention, generated-response completeness, response speed, peak memory and total cost. A loss reduction with worse answers does not qualify the model as a useful assistant. The report publishes every selected response and counts responses that exhaust their token budget; it does not invent an automatic correctness judge for arbitrary text.

If this reference learns usefully, reproduce that recipe through the distributed implementation and measure the cost of verification, communication and failure recovery. Then test a second data cohort using replay from the recorded, actually trained documents. Model expansion follows sustained resources and a better quality/cost result against continued training at the smaller size. Native settlement, independent ownership, economical GPU verification and private inference remain separate requirements in the [scaling design](SCALING_DESIGN.md).
