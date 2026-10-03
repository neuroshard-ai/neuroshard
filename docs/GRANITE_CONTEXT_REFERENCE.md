# Contextual assistant reference contract

September 26, 2026. **Completed and failed; [results](GRANITE_CONTEXT_REFERENCE_RESULTS.md).**
The frozen contract below is unchanged. The earlier
[requirement-check comparison](GRANITE_REFERENCE_RESULTS.md) remains failed.
The [completed audit](GRANITE_ADAPTER_AUDIT_RECOVERY_RESULTS.md) reproduced its
errors and closed that integration diagnosis. This contract tests a different,
useful function through the complete answering path. It authorizes one bounded
CPU execution after commit and passing CI, with no training or GPU launch.

## Why this experiment

An assistant needs to understand follow-up questions before retrieving records
or calling tools. We test whether an independently trained neural function can
improve that operation while the accepted assistant behavior survives. This is
a reference for the modular architecture, not a new consensus mechanism or
another attempt to rescue the failed checker.

The published query-rewrite adapter resolves conversational references into
standalone questions. Its authors evaluate rewrite quality on a private set;
their training uses the MT-RAG Cloud corpus and proprietary conversations.
Those facts support choosing the function, not predicting our results. We use
fresh fictional records and score final answers rather than borrow that corpus
or reproduce the authors' private benchmark.
[Pinned author model card](https://huggingface.co/ibm-granite/granitelib-rag-r1.0/blob/2f0b2c79c6731068625aca8045c2eb2e8912b353/query_rewrite/README.md).

The implementation keeps the already pinned Granite 4.1 3B parent, Granite
Switch checkpoint, upstream code and isolated numerical runtime. Query rewrite
is the existing rank-32 aLoRA, route 2, control token 100353. The composed
checkpoint retains all twelve published adapters in memory, even though this
study invokes only query rewrite. Charge that full storage and runtime.
[Published composition inventory](https://huggingface.co/ibm-granite/granite-switch-4.1-3b-preview/blob/7a3ac02e07868411424ac89440397b475da66fa7/BUILD.md).

## Frozen comparison

The [plan](../config/experiments/granite-context-reference.json),
[execution inventory](../config/experiments/granite-context-reference-execution.json)
and [resource contract](../config/experiments/granite-context-reference-resources.json)
fix the following three complete systems:

| Arm | Retrieval query | Answer generation |
| --- | --- | --- |
| History | All user and assistant turns, without a neural rewrite | Unchanged parent |
| Parent rewrite | One parent decode with the declared rewrite instruction | Unchanged parent |
| Module rewrite | Same instruction and history, plus published adapter activation | Switch backbone, no adapter, fresh cache |

All arms share the documents, BM25 implementation, three-document limit, original
conversation and final answer instruction. Full-history retrieval is a useful
cheap control; the parent rewrite controls for another neural generation.
The rewrite is used only for retrieval. The answerer still receives the original
conversation, so an incorrect rewrite cannot silently replace the user's request.
Selection never sees task identities, categories, target documents or answers.

There are **64 new cases in 16 entity-pair blocks**, with 128 fictional facility
records: 32 contextual questions, 16 standalone questions and 16 questions whose
answers are absent. Contextual cases resolve an earlier venue or a corrected
destination. Values and source IDs are fixed before any generation. This is a
small authored public evaluation, not a secret holdout or a broad chat benchmark.
Related cases are not treated as 64 independent observations: uncertainty uses
10,000 paired bootstrap resamples of the 16 blocks, with the seed in the plan.

The machine interface requests a JSON answer value and source IDs. Success
requires the exact requested value and the correct document actually present in
retrieval. Unsupported questions require `null` and no sources. Extra prose,
duplicate keys, uncompleted output or malformed rewrites fail. No LLM judge,
gold routing, retry, answer repair or hidden fallback can rescue an output.

Parent and module use the same rewrite instruction, 128 output tokens for the
rewrite, 96 for the final answer, and 2,048 input tokens. These are stage caps,
not equal FLOPs: actual input/output tokens, generation calls, retrieval time,
end-to-end latency, worker CPU/wall time and instance time are recorded. Greedy
BF16 eager CPU generation uses the upstream implementation. Every stage starts
a fresh cache; no cache-sharing speedup is claimed.

The [tokenizer-only preflight](../config/experiments/granite-context-reference-preflight.json)
checked all 64 rewrite prompts: parent and module differ only at the activation
token. Their answer prompts are identical. The maximum rewrite is 160 tokens;
the conservative answer-prompt bound is 426, below 2,048. Every known answer's
source is retrievable with its scoring-only oracle query. No model weights were
loaded and no neural answer was generated for these checks.

## Pass and stop rules

Before new questions, repeat all 24 opened assistant anchors for each model.
Both must meet the original 18/24 and 5/8-per-category gates and preserve all
18 previously correct answers. These repetitions establish retention only.
Failure stops before the affected model's new-question comparison.

For the fresh cases, require all of:

- Module answers at least 48/64 correctly, including 20/32 contextual,
  12/16 standalone and 12/16 unsupported questions.
- Net gain of at least four answers against **each** control, with the 2.5th
  percentile of the paired block bootstrap above zero for each comparison.
- **No previously correct parent-rewrite answer lost.** Report losses against
  the history control too; aggregate gains do not hide them.
- Module p95 complete-request latency at most 120 seconds and at most 1.5 times
  the parent-rewrite p95.
- Only after these gates pass, reload the models and repeat three fixed cases
  in every arm. All nine complete pipelines must match their queries, retrieved
  documents, tokens, scores and route traces exactly.

There is one attempt. Broken runtime, failed retention, time exhaustion or a
failed quality gate closes it. Do not retune prompts, retrieval or thresholds
against opened answers. Failure reporting distinguishes malformed rewriting,
missing evidence, incorrect answers despite retrieved evidence, and unsupported
assertions. A later method requires its own prospective contract.

## Resources and interpretation

One `r7i.4xlarge`, 80 GiB gp3, at most two hours, $6 planning cap with a reserve
for storage and transfers. The worker has 80 minutes and 64 GiB; setup has
20 minutes and evidence copy reserves ten. Sources must be committed and the
exact commit's protocol CI must pass before allocation. The prior allocation's
retirement receipt is pinned. OS expiry and the controller terminate the temporary
host, volume and security group after evidence copy, including on failure.
No active model session is required while the background controller waits.

Memory reporting is cumulative process peak RSS, not isolated per-model memory.
The existing [A4 memory and boundary estimates](ASSISTANT_FOUNDATION_DECISION.md#route-to-a4)
still apply. This study does not partition weights or test independent operators.

A pass would establish one published neural module improving this complete,
document-grounded assistant path under a matched-call control. The facts come
from retrieval; the module contributes a learned query transformation. Neither
is newly trained capability, learned semantic routing, general conversation
quality or repeated model growth. A1 still needs an explicit review against its
full checklist; the runner cannot award it. A2's automatic tool-use training,
controls and retention requirements remain unchanged.
