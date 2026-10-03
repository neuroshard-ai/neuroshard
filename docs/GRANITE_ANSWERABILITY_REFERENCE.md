# Fresh answerability module comparison

**Frozen September 26, 2026; [completed result: failed](GRANITE_ANSWERABILITY_REFERENCE_RESULTS.md).** This continues A1
with the published Granite checkpoints and the implemented
[evidence-selection interface](EVIDENCE_SELECTION.md). The earlier failed
requirement-checking and query-rewriting references remain failed. The
[64/64 copying diagnostic](EVIDENCE_SELECTION_RESULTS.md) remains opened data.

The question is whether an added neural module improves the complete answering
path under preservation and cost constraints. Correct serialization alone has
already been demonstrated; it cannot establish that adding weights helps.

## Method and controls

| Path | Selection | Conditional second call | Final answer |
| --- | --- | --- | --- |
| Selection | Parent selects one span or abstains | None | Exact copied value and citation |
| Parent check | Same parent selector | Parent checks selected document | Keep selected value or abstain |
| Module check | Modular checkpoint in base mode selects | Published answerability adapter checks selected document | Keep selected value or abstain |

Every selector receives the same public conversation, complete source records,
span values and invocation-local labels. It uses the diagnostic's unchanged
instruction and one constrained token. If it abstains, checking is skipped.
Otherwise the checker receives the original conversation and **only the selected
complete document**, through the native documents template. An `answerable`
decision keeps the span; `unanswerable` abstains. No alternative choice, second
answer, threshold, hidden fallback, training or learned router is introduced.

Both checkers receive the same instruction and finite output constraint. Native
greedy generation emits one of the JSON strings `"answerable"` or
`"unanswerable"`, including terminal EOS, in at most six tokens. This follows
the pinned upstream IO schema, whose framework later nests the scalar under an
`answerability` key. That wrapper is not the raw neural output. The explicit
instruction and finite constraint are declared deviations from the unconstrained
upstream usage examples; there is no claim to reproduce IBM's benchmark score.

The module is designed for deciding whether supplied documents support the
latest question in a conversation. Its published training description uses the
MT-RAG Government corpus; this study uses new fictional records instead.
[Published answerability model card](https://huggingface.co/ibm-granite/granitelib-rag-r1.0/blob/2f0b2c79c6731068625aca8045c2eb2e8912b353/answerability/README.md).

The existing [artifact inventory](../config/experiments/granite-reference-artifacts.json)
pins the parent, composed model, tokenizer and upstream implementation. The
selected embedded aLoRA has rank 16, alpha 32 and attention/MLP projections,
according to its raw config and composed `BUILD.md`. The model card describes a
different rank/target summary; that discrepancy is recorded, not silently treated
as equivalent. Route 5 / token 100356 explicitly activates answerability at the
assistant header. Every stage gets a fresh cache. All 12 embedded adapters'
storage and the complete composed checkpoint's memory are charged, even though
only one adapter can be active in this path.

## Frozen evaluation

The [80 new cases](../config/experiments/granite-answerability-cases.json) have
40 supported and 40 unsupported requests. Eight semantic families cover dates,
versions, numerical conditions, missing context, proposed versus approved facts,
subject corrections, operational scope and different measures. Each family has
five fictional entities with a paired supported/unsupported question. These are
authored controlled records, not a secret test or an external benchmark. No
pretrained model has been run on them before the freeze.

Each case supplies three public source records with explicit spans. These are
scoped document inputs, not a new retrieval benchmark. Source IDs, values and
scope qualifiers are public; expected answers, task IDs, family/category labels
and pair IDs never enter a model request. A source contains only one candidate
value. The executor never uses expected answers to construct a menu or choose a
document for checking.

The [plan](../config/experiments/granite-answerability-reference.json) requires:

- At least **64/80** complete correct answers, including **32/40** in each class.
- Net **at least four** over both selection and parent checking, with a positive
  lower 95% paired bootstrap bound. Resample the **eight template families**;
  do not pretend the 80 similar cases are independent observations.
- **Zero lost correct answers** against either fresh control. Keep all 64 opened
  lookup successes and all 18 original assistant successes. Both models must
  also satisfy the original assistant total/category gates.
- Exact agreement between parent and modular-base selector outputs on all 144
  source tasks. Module gains cannot be attributed to an unexplained base change.
- Candidate p95 at most **30 seconds** and **1.5 times** parent-check p95;
  isolated worker peak RSS at most **64 GiB**.

The parent check is a competing control. Its own regressions on the old 64 do
not cancel a candidate that might repair them. The unchanged parent selector
must preserve those 64; the complete candidate must preserve them too. The old
cases count only toward retention, never new capability. The original coding
final and opened programming-growth cases are not read by this runner.

The baseline runs first. If its selector or checker scores above 76/80, a +4 gain
is impossible; stop without loading the candidate. Otherwise one modular run
follows. Any gate failure rejects it. Only a passing comparison earns six path
replays, three cases in each of two fresh model processes. All decisions,
answers, prompts, tokens and route traces must match. No automatic retry,
post-result prompt tuning or threshold adjustment is authorized.

## Execution and interpretation

The [execution inventory](../config/experiments/granite-answerability-reference-execution.json)
pins contracts and all executed sources. The
[preflight](../config/experiments/granite-answerability-reference-preflight.json)
checks every possible selected-document prompt, both tokenizers and activation
headers. A tiny random BF16 model tests native finite-output generation with
documents; its output is not quality evidence. Sources must be committed and
that exact commit's protocol CI green before allocation.

One CPU `r7i.4xlarge`, 80 GiB gp3, two-hour expiry, **$6 planning cap** including
storage/transfer reserve. Primary workers get 30 minutes each; conditional replay
workers get ten minutes each. Each runs in a separate process, with its own peak
RSS and CPU/wall accounting. Setup/copy limits are 20/10 minutes. At most 64 MiB
of study evidence is collected before retirement. The previous allocation's
cleanup receipt is pinned in the
[resource contract](../config/experiments/granite-answerability-reference-resources.json).

Record all executed calls, route traces, checkpoint bytes, input/output tokens,
latency and instance time. Selection-only reuses the first stage of the parent
check in the laboratory; report its logical serving cost without charging an
imaginary additional execution. The controls have the same maximum calls and
decode budget, not equal parameter counts or exact FLOPs. Fresh and retained
work, replays, downloads and interruptions all count toward the actual bill.

A pass would establish one bounded published-module benefit through the new
interface. A1 still needs review of the complete foundation/recipe requirements;
A2 still needs a newly trained useful extension. This is not general assistant
competence, unlimited growth, automated module discovery, sharding or independent
operation. Receipts and valid output syntax do not prove neural execution or
semantic correctness. No GPU, training, native-network activation or checklist
completion is authorized by this contract.
