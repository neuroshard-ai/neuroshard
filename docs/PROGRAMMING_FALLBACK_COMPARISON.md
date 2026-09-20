# Equal extra-attempt programming fallback

**Status: leftover comparison passed. This configuration is the research
baseline for further growth. Not promoted.**
The original 128-task final stays closed. No further training was run.

The [leftover result](PROGRAMMING_FALLBACK_RESULTS.md) is +4/32 versus both the
first parent attempt and parent repair, with no unique losses. That is
complementary skill under an executable public-example gate. It is not learned
routing, not promotion, and not an ever-growing public model.

## Question

Does the trained tail help the complete assistant when it is used **only after
the parent's public example fails**, compared with spending that same extra
decode on a frozen parent repair prompt?

The arms share one extra attempt and the same 256 output-token cap. They are
**not** equal computation: the repair conversation is longer. The score reports
input tokens, output tokens and public-example checking time alongside
generation latency.

## Policy

1. Generate the unchanged parent on the original prompt.
2. Execute **only** the example that was already in the prompt.
3. If it passes, that program is the answer for every arm.
4. If it fails, spend one extra generation:
   - **expert_fallback:** the trained tail on the original prompt
   - **base_repair:** the parent on the original prompt, the failed reply, and
     an instruction that cites that same public example (no withheld tests)
5. Score the chosen program on the **full** test list, including extra
   attempts that repeat the first-answer tokens.

A second greedy decode of the identical parent prompt is not a control: it
would copy the first attempt. The repair conversation is the extra attempt.
Repeated output tokens from that distinct prompt are a scored outcome.

## Comparison set

32 leftover MBPP tasks from the original final pool (IDs 11–510), excluding the
128 already reserved as the unopened final and the three parent-preparation
exclusions. IDs are committed in
[programming-fallback.json](../config/experiments/programming-fallback.json)
and recomputed by `leftover_task_ids` before preparation and again before
scoring. Runtime loading compares the rows themselves to that committed list.

Pinned identities: parent plan `b792b308…`, selection `fbff46de…`, rejected
expert checkpoint `46bd2e76…`, tokenizer `e9478f6c…`.

## Stop rule

The expert fallback must gain at least two tasks versus the first parent
attempt **and** versus parent repair, with a positive paired-bootstrap lower
bound versus repair, and p95 serving time (generation plus public-example
check for the attempts that arm used) at most 1.5× repair and 90 seconds.
Failure stops this candidate. The opened-development 15/32 number cannot
satisfy these gates.

Pass would show complementary skill under an executable check, not learned
routing, not cross-skill composition, and not promotion of the accepted graph.

## CPU diagnostic of the opened development set

```sh
PYTHONPATH=src python scripts/score_programming_fallback.py \
  --inputs .neuroshard/programming-expert-20260920/inputs
```

That path needs the parent trial's `dev.jsonl` and Bubblewrap. It must report
`admission_evidence: false`. The committed diagnostic is
[programming-fallback-development-diagnostic.json](../config/experiments/programming-fallback-development-diagnostic.json).

## GPU comparison (after this contract and its freeze are committed)

Prepare leftovers, copy the four owner object manifests from the rejected trial,
and load the saved expert tail. Do not copy original `final.jsonl` into the
comparison home.

```sh
PYTHONPATH=src python scripts/prepare_programming_fallback.py \
  --home STUDY --mbpp mbpp.jsonl
PYTHONPATH=src python scripts/prepare_programming_fallback.py --freeze
# commit the freeze, then launch
PYTHONPATH=src ATEN_CPU_CAPABILITY=default MKL_ENABLE_INSTRUCTIONS=SSE4_2 \
torchrun --nnodes=4 --nproc-per-node=1 --node-rank=RANK \
  --master-addr=OWNER0_PRIVATE_IP --master-port=29441 \
  scripts/run_programming_fallback.py --home STUDY --seed TOKENIZER_DIRECTORY \
  --expert /path/to/rejected-trial/expert
```

Four temporary g5.xlarge hosts, two-hour cap, $40 planning ceiling. Evidence
from the executed run is in [PROGRAMMING_FALLBACK_RESULTS.md](PROGRAMMING_FALLBACK_RESULTS.md).
Retire them after evidence copy. No native issuance.
