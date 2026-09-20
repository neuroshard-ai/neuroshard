# Second admitted programming capability

**Status: frozen contract. Not executed. Not promoted.**
The leftover fallback system remains the research baseline. The original
128-task final stays closed. This is not an upgrade of the 0.4.0 genesis.

The leftover [fallback result](PROGRAMMING_FALLBACK_RESULTS.md) is the complete
system to beat: parent plus the incumbent tail `46bd2e76…`, public-example
gate, one extra decode, 256 output tokens. The next question is:

> Can another admitted capability improve new answers, preserve previously
> demonstrated successes, and keep the same maximum work per request?

## What is added

A second last-four-layer programming tail, trained on leftover MBPP that is
disjoint from:

- the reserved parent final (128 tasks)
- the leftover preservation set (32 tasks, including the 12 recorded successes)
- the new-answer held-out set (32 tasks)
- the incumbent tail's training IDs (official MBPP train 601–974 after the
  parent exclusions)
- leftover-remainder prompts that are near-duplicates of those frozen
  evaluation sets under the programming-expert Jaccard rule

The new tail is another programming specialist, not a new domain. It starts
from the **parent** last-four-layer weights, not the incumbent checkpoint.
Training on the incumbent first would make `θ_added = θ_incumbent + Δ_new`,
and unit merge would double-count the incumbent update. Hyperparameters copy
the rejected first expert: 256 updates, last four layers only, frozen 1.7B
parent. Leftover-remainder prompts that are near-duplicates of the frozen
preservation, new, or development sets are dropped from training only. Those
evaluation IDs stay fixed.

## Serving rule

Available capacity grows by one stored tail. Activation does not:

1. Generate the unchanged parent.
2. If the public example passes, keep that program.
3. Otherwise spend **one** extra decode on the **unit task-vector merge** of
   both tails:

   `θ_merged = θ_parent + (θ_incumbent − θ_parent) + (θ_added − θ_parent)`

   That is linear task arithmetic of two independently learned deltas, not
   learned routing and not a second decode. Serving still uses one extra
   decode of a last-four-layer tail. Storing another checkpoint is not the
   same as expanding active parameters per answer. Jaccard overlap over
   leftover-like prompts would send those requests to the new tail and can
   drop the incumbent leftover wins; merge tests whether the two skills add
   in weight space instead. A failed unit merge rejects this coefficient
   choice. It does not settle other composition methods.

Paired measurement may generate the incumbent extra and the merged extra so
both policies can be scored. Latency gates count only the attempts the serving
policy would use. That is not equal computation.

## Stages

**Isolation.** On the frozen development 32, the new tail as a checked extra
must pass the leftover gates versus the parent and versus parent repair. Failure
stops the candidate. The new-answer slice stays closed.

**Growth.** Only after isolation passes. Compare the merged extra against
the complete leftover fallback system on:

- **preservation 32:** every leftover fallback success must remain correct
- **new 32:** at least two additional full-test-correct answers versus the
  fallback system, with a positive paired-bootstrap lower bound

p95 serving time (generation plus public-example check for the attempts that
arm used) at most 1.5× the fallback system and 90 seconds.

## Stop rule

Isolation failure ends the run. Growth failure does not promote the second
tail and does not reopen the original final. Neither stage admits a public
graph or changes 0.4.0.

## GPU allocation (after this contract and its freeze are committed)

```sh
PYTHONPATH=src python scripts/prepare_programming_growth.py \
  --home STUDY --mbpp mbpp.jsonl --seed TOKENIZER_DIRECTORY
PYTHONPATH=src python scripts/prepare_programming_growth.py --freeze
# commit the freeze, then launch
PYTHONPATH=src ATEN_CPU_CAPABILITY=default MKL_ENABLE_INSTRUCTIONS=SSE4_2 \
torchrun --nnodes=4 --nproc-per-node=1 --node-rank=RANK \
  --master-addr=OWNER0_PRIVATE_IP --master-port=29441 \
  scripts/run_programming_growth.py --home STUDY --seed TOKENIZER_DIRECTORY \
  --incumbent /path/to/rejected-trial/expert
```

Copy parent owner objects from the rejected programming-expert trial. Wait for
the image `unattended-upgrades` dpkg lock before `apt-get`. Four temporary
g5.xlarge hosts, six-hour cap, $80 planning ceiling. Retire them after
evidence copy. No native issuance.
