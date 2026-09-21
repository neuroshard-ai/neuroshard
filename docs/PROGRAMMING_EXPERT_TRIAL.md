# Useful programming expert growth

The next experiment asks one question: **does adding a trained neural expert
make the complete assistant more useful, within a fixed cost per answer?**
The candidate is a programming tail derived from the same original
SmolLM2-1.7B-Instruct checkpoint as its frozen parent. Three owners retain the
complete parent between them. A fourth owns the copied last four layers and
trains only those layers. No serving owner receives the whole model. A
read-only tied output head is replicated at the learner and counted separately.

This is an operated research candidate, not an update to the accepted alpha
graph. Native BFT and existing work verification remain unchanged.

**Measured September 20, 2026: rejected in development.** The 256-update run
completed across four GPU owners. All 24 general responses were preserved,
but the original model passed 12/32 coding tasks and automatic selection
passed 11/32. The final stayed unopened. See the
[committed result](../config/experiments/programming-expert-results.json) and
[actual generated outputs](../config/experiments/programming-expert-development-outputs.json).

## Why this method

The earlier [fusion trial](CAUSAL_FUSION_TRIAL.md) failed to transfer specialist
answers. The subsequent source-interface measurements found that the
directory specialist could answer 8/8 under its native prompt but only 1/8 with
the shared conversational context. That motivates removing this interface
mismatch before attempting another latent fusion method.

Here training and serving share the original model, tokenizer, chat template,
instruction style and prefix. A learned input-only selector chooses one tail
per request. Existing frozen-prefix production, tail optimization and cached
sharded inference do the neural work. The selector is trained only on training
prompts, including general negatives. Supplied conversation history is part
of the input; no evaluation ID, expected next answer or test result is given.

This does **not** establish composition between multiple learned skills. It is
the prerequisite test that one added skill helps an automatically served
assistant. No claim that this architecture is novel follows from this trial.

## Comparison and evidence

The [plan](../config/experiments/programming-expert.json) fixes one recipe and
one terminal checkpoint. Every development and final prompt is generated through
three actual paths, with counterbalanced order:

| Arm | Answering path | Purpose |
| --- | --- | --- |
| Base | Unchanged parent | Measure the starting capability |
| Automatic | Learned selection of parent or new tail | Measure the complete assistant |
| Replacement | New tail on every request | Same learned weights and training budget, without the retained path |

Removing the added expert gives the original graph by construction. Its
ablation score is derived from the recorded base trace and explicitly labelled
as such; it is not a fourth independent generation or a latency measurement.
The four-process CPU test separately checks numerical equality of the original
tail through both layouts before training. The GPU driver repeats that check
on two development prompts before taking any optimizer step.

Generation uses the ordinary tokenizer vocabulary, greedy decoding, a
256-token cap and a local KV cache at each owner. There is no retrieved answer,
program repair, repeated sampling or execution feedback during generation.
Code either passes the supplied tests or fails. The full output, routing
decision and measured serving time are preserved.

## Data and limits

Coding uses the original [Mostly Basic Python Problems dataset](https://github.com/google-research/google-research/tree/master/mbpp)
by Austin et al., with its published task-ID splits: training 601–974,
development 511–600 and test 11–510. IDs 1–10 are unused. Dataset bytes are
pinned to repository revision `f82046ba5aabbbb427dbfd38a254d26bff08b533`.
MBPP is [CC BY 4.0](https://huggingface.co/datasets/google-research-datasets/mbpp);
NeuroShard's source code remains Apache-2.0. Downloaded benchmark rows are
research inputs, not bundled protocol/package data.

One public example is in the user prompt to specify the function interface;
the remaining tests are not shown to the model. Gold solutions must first
pass their own tests in the same evaluator. Invalid references, near-duplicate
training prompts and overlong complete references are reported as exclusions
before model execution. The final sample is 128 eligible tasks and development
uses 32; deterministic selection and exact IDs are committed before training.

These are old public problems that the seed model may have encountered during
pretraining. The comparison can show improvement over that seed, not absence
of contamination, broad programming competence or correctness beyond the
benchmark's few tests. Filtering bad references does not certify their
algorithms on every input.

General prompts come from the pinned Apache-2.0
[smol-smoltalk](https://huggingface.co/datasets/HuggingFaceTB/smol-smoltalk)
test snapshot, independently sampled for selector training, development and
final across conversation, constraints, summarization and rewriting. General
retention means exact preservation of the base model's generated responses,
including any base errors. It is not an independent general-quality score.
All data is public and committed, not secret or an adversarial quality oracle.

Generated programs run under Bubblewrap namespaces and systemd scopes, with
no network, host home or repository mounts, aggregate process/memory limits,
CPU limits and wall timeouts. This is an operated benchmark evaluator. It
trusts the host kernel and does not claim a secure permissionless verification
protocol or a tamper-proof scoring harness for malicious submissions.

## Stop rule

There is one learning run: 256 updates, up to four documents per batch, last four
layers only, fixed learning rate schedule. The existing parent never changes.
The feature bank and final weights, Adam state, router and outputs are saved.

Development must improve by at least two coding tasks, preserve all 24 general
responses and satisfy both latency bounds before the final is opened. The
final requires at least five percentage points of automatic coding gain, a
positive paired-bootstrap one-sided 95% lower bound, all 64 general responses
preserved, p95 latency at most 1.5 times the base and at most 90 seconds.
Failure stops this candidate. There is no tuning on the opened final.

Four temporary g5.xlarge owners have a four-hour maximum allocation and a $50
planning cap including a transfer/storage allowance. CPU preparation, the
scorer and the actual four-process training driver must pass before launch.
Source, input selection and recipe must be committed before training. No
permanent GPU service is created. Evidence is copied off the machines before
termination; a hard deadline retires them even on controller failure.

A pass would establish one bounded instance of useful expert growth. Repeated
cohorts, cross-skill composition, consolidation, independent operators and
economical verification would still need their own evidence.

## What this run established

Preparation retained 289 training programs after reference checks and
near-duplicate exclusions, plus 256 general prompts for selector training.
The selector routed all 32 development coding prompts to the expert and all
24 general prompts to the original path. Before training, two prompts generated
the same eight tokens through the parent and the unmodified added-tail path.
The owners agreed on the cached feature root, and every parent parameter
remained unchanged through training.

| Development measure | Result |
| --- | --- |
| Original model, executable coding tasks | 12/32 |
| Automatic selection | 11/32 |
| Trained expert for every request | 11/32 |
| Coding cases gained / lost versus original | 4 / 5 |
| Automatic general responses identical to original | 24/24 |
| Forced expert general responses identical to original | 6/24 |
| Original / automatic p95 response time | 9.950 / 9.951 seconds |
| Paired coding gain, one-sided 95% lower bound | −0.1875 |

Different general wording in the forced-expert control is not itself evidence
of an incorrect answer: this retention measure checks exact preservation.
The rejection follows the coding gate. Correct domain routing and working
distributed optimization did not produce a net improvement from this recipe.

There is a narrower signal for the next method. The expert generated four
test-passing programs where the original failed, but displaced five passing
original programs. In a **posthoc analysis of the opened development set**,
retaining the original answer when it passes the user-provided first example,
and trying the expert only after that check fails, would score 15/32. This
uses only the example already present in the prompt for selection; all three
tests still decide the reported score. All 32 setup strings were empty.

That is a candidate mechanism, **not an admission result**. It was chosen
after looking at development, uses an additional inference attempt on some
requests, and has not been compared against giving the original model an
extra decode under the same output-token cap. A fresh frozen comparison must
resolve that before opening finals or crediting useful growth. No further
training or model promotion follows from this diagnostic. The frozen follow-up
is the [equal extra-attempt fallback comparison](PROGRAMMING_FALLBACK_COMPARISON.md).
Its leftover result passed +4/32 versus parent and parent repair without
opening this trial's final or promoting the tail; see
[PROGRAMMING_FALLBACK_RESULTS.md](PROGRAMMING_FALLBACK_RESULTS.md). That
fallback configuration is the research baseline for further growth. The
[second-capability growth](PROGRAMMING_GROWTH.md) campaign that followed is
closed. The next experiment is
[learned integration of new capacity](LEARNED_INTEGRATION.md).

An independent CPU rescore exactly matched the GPU host. All four temporary
instances, their volumes and network interfaces are gone; the temporary
security group is deleted and the protected hosts retain their original
states. The conservative GPU instance-time estimate through observed
retirement is **$1.858**, excluding preparation-host, storage and transfer
charges. Full checkpoint, Adam and feature-bank artifacts remain preserved
off the retired hosts. No S3 copy is claimed.

## Reproduction

Install the isolated research requirements from
`docs/learning-reference-requirements.txt`. Scoring additionally requires Linux
Bubblewrap and a working systemd user manager with cgroup resource controls.
Do not install this environment over a node runtime.

Download the model snapshot at the revision above, the pinned `mbpp/mbpp.jsonl`
from Google Research, and
`HuggingFaceTB/smol-smoltalk@f73fe857d519ff6ac5af2ea67c4d3834da7b8bcc`'s
`data/test-00000-of-00001.parquet`. Put the data in a fresh study directory as
`mbpp.jsonl` and `smoltalk.parquet`. Preparation checks the exact byte hashes.

```sh
PYTHONPATH=src python scripts/prepare_programming_expert.py --home STUDY --seed MODEL
```

This writes the prepared rows, original owned tensor objects and a compact
selection under `config/experiments/`. Commit the source, plan and selection;
then use the same command with `--freeze` and commit its execution freeze.
The driver refuses an uncommitted freeze or changed execution source.

Each of four owners needs the source checkout, prepared inputs, tokenizer
files and **only its selection manifest's tensors** in `STUDY/objects`. Owner
3 also needs the declared read-only head objects. The full downloaded seed
belongs on the preparation host, not each serving owner. Run the same command
on all four owners with the appropriate rank:

```sh
PYTHONPATH=src ATEN_CPU_CAPABILITY=default MKL_ENABLE_INSTRUCTIONS=SSE4_2 \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
torchrun --nnodes=4 --nproc-per-node=1 --node-rank=RANK \
  --master-addr=OWNER0_PRIVATE_IP --master-port=29441 \
  scripts/run_programming_expert.py --home STUDY --seed TOKENIZER_DIRECTORY
```

`training.json`, `expert/`, `features/`, `router.json`, runtime identities and
the actual generated outputs are evidence. `result.json` states whether the
final opened and whether its gates passed. Preserve these artifacts off-host
and retire the temporary owners. The original protected network hosts are not
part of this allocation.
