# Continual model evolution: experimental implementation

NeuroShard now has executable experiments for full-model training, fresh/replay data, model growth, and optimistic native settlement. The public 0.4.0 network still serves its existing adapter model. The evolving-model components have not yet been integrated into a public network.

The [scaling design](SCALING_DESIGN.md) sets the direction for compute groups, funded complete auditing and growth backed by sustained capacity. The opt-in [compact optimizer extension](COMPACT_UPDATE_DISPUTES.md) can refute an inconsistent SGD assertion with bounded evidence. It leaves full-stage auditing in place for gradients, tensor/file relationships and other operators.

The published [working paper](https://neuroshard.com/papers/FINE2026_neuroshard_short.pdf) distinguishes the target protocol from what has actually run. Manuscripts and generated measurements are maintained outside the current code tree; the evidence links below preserve the recorded historical revision.

## What runs

| Mechanism | Implementation | Evidence |
| --- | --- | --- |
| Whole-model training | `evolution/model.py`, `worker.py`, `pipeline.py` | 134,515,008 trainable parameters; three worker processes on two hosts; exact stage replay |
| Model expansion | `grow`, `place`, `audit_growth` | Four identity-initialized blocks add 14,160,384 parameters; the 148,675,392-parameter model needs four workers under the declared 48M cap |
| Continuing data | `evolution/data.py` | Pinned source revisions, durable cursors, immutable windows, document grouping, replay and evaluation exclusion |
| Text semantics and response objectives | `evolution/text.py`, `batches.py` | Content-addressed tokenizer/model identity, bounded windows across assistant turns, ignored prompt/padding labels and actual response endings |
| Quality decisions | `evolution/evaluation.py`, `controller.py` | Paired fresh/retention losses, preserved candidates and selection on restart, rejection leaves the accepted model unchanged |
| Native work settlement | `evolution/settlement.py`, `app.py` | Four validators, signed reservations, exact token accounting, availability challenges and an objective fraud dispute |
| Native growth settlement | `grow` transaction, `audit_growth`, session `start_step` | A separate bonded claim; invalid growth is challengeable from one parent block; accepted growth mints no reward and preserves the training-round counter |
| Distributed generation | `Pipeline.generate` | Greedy inference through the assigned model components |
| Native data and serving lifecycle | `evolution/cohorts.py`, `lifecycle.py`, `forward.py` | Opt-in cohort admission, finite fresh/replay assignments, replayable scores, serving decisions and paid generation; see the [lifecycle profile](NATIVE_LIFECYCLE.md) |

The distributed training and paired quality experiments ran on **two physical machines under one operator**. Later local reproduction checks and the extended native growth lifecycle ran on one host, as identified in their results. The worker parameter cap is a declared allocation; these experiments do not establish that a whole 135M model cannot fit on either physical machine.

## Exact execution profile

The seed is `HuggingFaceTB/SmolLM2-135M-Instruct`, revision `12fd25f77366fa6b3b4b768ec3050bf629380bac`. The importer verifies all eight published seed-file hashes before conversion. Tensors are float32 safetensors components; arbitrary Python model code and pickle checkpoints are not accepted by this path.

The first worker owns the tied embedding/head and final norm. Forward activations traverse contiguous block ranges. The final hidden state returns to the first worker; the output-head gradient then travels backward through the pipeline. Tied embedding/head gradients accumulate on their single owner. All parameters are trainable.

SGD uses learning rate 0.003 and global clipping at 1.0 in the main experiments. The global norm combines committed stage norms, with squared-gradient sums in float64. The profile fixes CPU execution, one thread, deterministic algorithms, disabled MKLDNN, default ATen CPU dispatch and MKL SSE4.2. Startup order is part of correctness: a native experiment initially initialized the CPU path too early and produced the wrong replay result. A fresh-process regression test now checks the corrected order.

Current bounds are 48M resident parameters per worker, 64 stages, four batch rows, 256 tokens per row and 512 total batch tokens. Parameter count does not include runtime, gradient, activation, and serialization memory. The existing implementation is a sequential pipeline, not a throughput-optimized 1F1B schedule.

## Why the first quality result needed correction

The initial experiment scored all next tokens in 64-token conversation prefixes. Only 17/64 retention examples, 18/64 fresh examples and 10/64 test examples contained any first-response tokens. Much of the apparent loss improvement concerned prompts and formatting.

The corrected experiment retains up to 64 prompt-context tokens and 64 first-response tokens. It scores response positions only, and masks right padding. It uses subsequent source rows and separate protected examples. This is still a truncated-context, teacher-forced response-loss test. It does not establish broad reasoning, factual accuracy, agent autonomy, or answer safety.

The maintained epoch command now uses the [versioned text protocol](TEXT_PROTOCOL.md). It retains short answers, splits targets across assistant turns, records omitted training tokens, and requires complete response coverage for held-out documents. Tokenizer/backend/template versions are bound to the model and corpus; evaluation averages real target losses within each document before the paired comparison. The older scripts retain their original objective to reproduce the historical measurements below. Their failed quality decisions have not been replaced with claims about the new objective.

The research gate requires an approximate 99% upper confidence bound below -0.001 nats on fresh examples and below +0.02 nats on retention examples. It uses paired document losses and at least 32 examples per group; the main runs use 64. An untouched test group is reported separately. Public-test exposure, dependence and multiple comparisons remain limits on the claim.

**Neither response-trained candidate passed that gate.** Their fresh-loss changes versus the common parent were -0.001434 and -0.001405 nats, with upper bounds of +0.000639 and +0.000596. The larger model did not establish an advantage at the same training budget. This is a useful rejection result, not evidence that the system already improves continuously. [Raw measurements and reproduction details](https://github.com/neuroshard-ai/neuroshard/blob/108b4ba3d6c6fb5760ff211b447ee95a67fa9112/docs/eval/results/evolution20260911/README.md) include both successful execution checks and failed quality promotion.

A separate exploratory follow-up started from the original seed with 32 steps at learning rate 0.03 and 128 new paired examples per group. It also failed: fresh-response loss increased from 1.129396 to 1.138708, with an upper bound of +0.017282. All three final-step stages replayed correctly. No candidate in these experiments qualifies for promotion; a correct training transition is distinct from a better serving model.

## Native transaction lifecycle

1. `reserve` locks the coordinator's collateral and binds the current model, round and worker reward identities.
2. Workers produce content-addressed activations, gradients, components and step transcripts. Signed receipts bind the native reservation, chain, record and stage.
3. `claim` checks assigned data, exact component coverage, graph links, optimizer scaling, parent/round and signatures. It performs no neural replay on the ordinary path.
4. `challenge` names a bounded stage for a fraud dispute or a related object for an availability request.
5. `upload` publishes sequential 1 MiB chunks through native transactions. `seal` checks their combined digest. Only the responsible publisher can append chunks.
6. `resolve` invokes the deterministic stage referee after every required input has been published in finalized blocks. No off-chain URL fetch participates in block validity.
7. Invalid or unavailable work is rejected. An eligible unchallenged claim settles after its window, updates the learning head and issues one experimental NEURO to the assigned workers.

False/abandoned computation challenges burn their bond. An absolute claim expiry prevents repeated challenges from occupying the work slot forever. The period budget renews without resetting chain history. The prototype's fee and bond amounts are test parameters, not a demonstrated economic equilibrium.

A corrected native experiment published **170,169,856 replay bytes through 184 upload/seal transactions**, taking **310.05 seconds**. Dispute resolution took **10.14 seconds** including transaction processing. Four validators rejected the forged optimizer update, then accepted valid work and issued exactly **1,000,000 atoms**. This is a coarse objective reference, not an inexpensive succinct proof. Ordinary graph checking was about **25 milliseconds** in the separate training conformance run; independent replay still incurred neural compute.

The subsequent [native growth lifecycle](https://github.com/neuroshard-ai/neuroshard/blob/108b4ba3d6c6fb5760ff211b447ee95a67fa9112/docs/eval/results/evolution20260911/native-growth-check.json) ran four validators on one physical host with the real model. It rejected forged training and growth, accepted valid growth, and paid the next training step without resetting the round. Growth verification used **14,161,224 bytes** from one parent block; this is a different, cheaper operation than replaying a full training stage. Growth issued no tokens; the two paid training steps issued 2,000,000 atoms total. A restart check confirmed matching state after the final payment.

The ledger now records paid work by parameter bytes, architecture, effective inputs/targets and optimizer settings. Changing ancestry, job nonces or worker placement cannot mint another payment for that same task. This also rejects a repeated task after exact convergence, when weight bytes stop changing. The `/work` query exposes prior payment for a computed work identity. The growing payment index still needs a more efficient long-lived storage representation before large-scale operation.

## Trust and incentive boundaries

Optimistic correctness assumes an honest online observer checks each accepted graph and can obtain inputs and complete a dispute. One randomly selected stage is not complete coverage. An unchallenged claim is not a cryptographic proof that somebody checked it.

The new settlement prototype pays workers and has fraud bounties. It does not yet fund a sustainable audit market when work is honest. A public design needs explicit audit and availability budgets, collusion analysis, and measured prices. Consensus honesty, numerical correctness, audit participation, data quality and model quality are separate assumptions.

The local epoch registry is intentionally **not** presented as a decentralized serving registry. A separate [native lifecycle profile](NATIVE_LIFECYCLE.md) now implements bounded data-cohort admission, optimistic evaluation adjudication, serving promotion/rejection and paid full-model generation on an isolated chain. It does not change the public 0.4.0 network. Public worker discovery, model replication/retention, funded complete audit coverage, efficient long-lived ledger storage and a balance-preserving migration from 0.4.0 remain required before cutover. Native growth uses declared capacities; turning those advertisements into reliable worker admission remains part of that integration.

## Running the tools from a checkout

Use the pinned numerical dependencies in `docs/llm-requirements.txt` and install the checkout. Do not point existing 0.4.0 validators at a modified working tree.

```bash
python -m neuroshard.evolution --help
python -m neuroshard.evolution download-seed --model-dir ./seed
python -m neuroshard.evolution import-model --model-dir /path/to/pinned/model --objects ./evolution-objects
python -m neuroshard.evolution inspect --model-root MODEL_ROOT --objects ./evolution-objects --workers 3
python -m neuroshard.evolution grow --model-root MODEL_ROOT --objects ./evolution-objects --layers 4
python -m neuroshard.evolution audit --trace-root TRACE_ROOT --objects ./evolution-objects
pytest -q tests/evolution
```

`grow` creates an unevaluated candidate. It does not replace a public checkpoint or issue tokens. `transport.py` binds to loopback and requires an operator-managed random token; use SSH forwarding for a second machine. These are research tools, not a new public mining endpoint.

The ongoing epoch controller is `Epochs` in `evolution/controller.py`. It journals collection, training, candidate commitment, evaluation reservations and decisions. It supports daily budgets, replay, optional capacity-aware growth and rejection. Its acceptance registry is local and remains separate from native serving-model activation.

### Run a complete research epoch

Install the numerical profile and collector dependencies from a checkout:

```bash
python -m pip install -r docs/evolution-requirements.txt '.[collector,dev]'
```

Copy [the example configuration](../config/evolution-epoch.example.json) into a private experiment directory. Paths resolve relative to that configuration. Import the verified seed into its `objects` directory. Use separate worker homes and one long-running worker command per assigned process:

```bash
python -c 'import os,secrets; f=os.open("worker.token",os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o600); os.write(f,secrets.token_hex(32).encode()); os.close(f)'
python -m neuroshard.evolution.transport --home ./worker0 --port 48701 --token-file ./worker.token
```

Start the other three workers with their own homes and ports from the configuration; place them on another machine through SSH forwarding as needed. Use the same pinned dependencies on each host. The example admits four workers and requests a four-block growth candidate. A worker's 48M parameter cap does not include gradients, activations and runtime overhead. One 32-step epoch writes many gigabytes of immutable checkpoints; allocate disk for both successful and rejected candidates. Garbage collection is not yet implemented, so do not schedule unbounded runs.

```bash
python -m neuroshard.evolution run-epoch --config ./epoch.json
```

This consumes subsequent licensed source rows, adds a replay fraction when historical data exists, trains all parameters, selects protected examples after committing the candidate, and records acceptance or rejection. Run it again to resume an interrupted epoch or begin the next budgeted epoch. The example allows one epoch per UTC day. It does not activate a public checkpoint or pay research workers. Use the separate native lifecycle experiment to test ledger activation and settlement.

Use fresh corpus and epoch homes when upgrading from the historical first-response objective. The tokenizer, window budget and document evaluation policy are persistent settings; changing them in an existing corpus is rejected. The current `import-model` command binds the verified seed tokenizer to its embedding rows. `inspect-tokenizer` checks this identity without loading the model tensors. See [text reproduction](TEXT_PROTOCOL.md#reproduce) for a complete text-to-training-to-generation check.

### Reproduce execution and fraud checks

Both scripts require a fresh home and retain their records. The model check uses the real 135M seed; its default workers share one process. Supply `--workers-config epoch.json` to use separately running HTTP workers. The native check starts and stops its own four validators and uses a small synthetic model by default, making it practical as an integration check.

```bash
python scripts/experiment_evolution_model.py --home ./model-check --model-dir ./seed
python scripts/experiment_evolution_native.py --home ./native-check --engine /path/to/cometbft
```

For a full-model native dispute, add `--record ./model-check/result.json --objects ./model-check/objects` to the second command. It uploads the actual replay inputs through native transactions and can take several minutes. Test keys stay in the experiment home and have no public value. The published two-host result is distinguished from the newer single-host reproduction checks.

The source commitment now covers all repository Python modules, including imported legacy helpers. Validators reject unsupported runtime versions at startup. These checks strengthen the declared profile; they do not prove identical arithmetic on all hardware or replace cross-machine conformance testing.

### Reproduce the bounded response follow-up

Start three workers using the earlier transport instructions and a private endpoint configuration. Use a fresh home for a new run; completed runs can be reopened to check their recorded result. The pinned seed files must already be downloaded. The plan fixes 32 training steps, response-target construction, source offsets and 128 evaluation examples per group.

```bash
PYTHONPATH=src python scripts/experiment_evolution_response.py \
  --home ./response-check --model-dir ./seed --workers-config ./workers.json \
  --plan config/experiments/response-from-seed-plan.json \
  --historical-selection config/experiments/response-from-seed-selection.json
```

The worker configuration uses the same `workers` array (`url` and `token_file`) as the epoch example. Use separate worker homes for a separate run. An optional `--upstream-cache` points to an existing verified Parquet cache. Repeating the historical selection reproduces old evidence; it is not another independent successful or failed experiment. This script replays all three stages of the final training step and recomputes the decision from paired values. It does not settle those training steps on the public ledger.
