# Evidence selection as a neural module interface

September 26, 2026. **Implemented experimental primitive; model diagnostic pending.**
The previous [contextual reference](GRANITE_CONTEXT_REFERENCE_RESULTS.md) remains
failed. A1 remains open. This work addresses exact copying and serialization;
it does not claim a new learned capability, module admission or a new consensus.

## Decision

Make the model choose the relevant evidence. Let a bounded executor copy the
selected source bytes and construct the answer and citation. The model no longer
has to reproduce an email address, opaque document ID or JSON punctuation to
use that evidence correctly.

The failed study supports this change: ordinary history retrieval already found
48/48 required sources, while the answerer frequently emitted prose or corrupted
values and citations. More query-rewrite tuning cannot supply evidence that is
already there. The published query-rewrite candidate is closed on these cases.

This draws on established ideas. Pointer networks select positions in inputs;
language-model programming and constrained decoding place machine-checkable
boundaries around generation. We are applying that separation to NeuroShard's
module interface, not claiming to have invented pointers or constrained output.
[Pointer Networks](https://arxiv.org/abs/1506.03134),
[LMQL](https://www.sri.inf.ethz.ch/publications/beurerkellner2023prompting),
[XGrammar](https://arxiv.org/abs/2411.15100).

Syntactic restrictions do not establish semantic correctness and can change
which output a model prefers. The selected evidence must still be evaluated
against the user's request; grammar validity is not an intelligence score.
[Grammar-Aligned Decoding](https://arxiv.org/abs/2405.21047).

## Implemented boundary

The [primitive](../src/neuroshard/evolution/evidence_selection.py) takes only
public conversation and scoped source documents with explicit UTF-8 byte spans.
At most 16 candidate spans and 64 KiB of source text enter one invocation.
The input adapter must derive spans from public data, not evaluation answers.

1. Validate source IDs, complete text, span bounds and UTF-8 boundaries.
2. Assign local choices `A` through `P` in a deterministic, request-dependent
   order; `Z` means abstain. No choice is derived from expected answers.
3. Bind the request, complete source snapshot, span inventory and choice mapping
   into an invocation hash. The model receives the original conversation,
   source text, candidate values and local labels.
4. Produce exactly one choice token through the native generation engine with
   a finite allowed-token list. The operation ends after that token; it is not
   a truncated free-form answer waiting for EOS.
5. Reconstruct the invocation before accepting the decision. Copy the selected
   bytes and source ID into a deterministic answer object. Record the source
   hash and byte offsets. A stale decision or altered source snapshot fails.

For example, selecting the source containing `ÅQ-92` copies exactly those bytes
and its original citation ID. The model cannot introduce `AQ-92` or an edited
citation during rendering. It can still choose the wrong source or abstain
incorrectly. Tests explicitly accept such receipts as valid copying and reject
their answers as semantically wrong.

The receipt verifies a deterministic transformation of supplied data. It does
**not** verify the truth of a source, the relevance of a selection, execution of
the neural model, or honest ownership. A worker could select a valid label
without running the model. Neural verification and operator incentives retain
their existing unresolved requirements. No new ledger or fork-choice rule is
introduced.

## Relationship to growing neural capabilities

The proposed interface separates learned decisions from exact data handling.
A future contributed module could improve reference resolution, tool choice or
composition while sharing the same executor and evidence references. Its
weights, selection policy and full serving behavior still need prospective
quality, retention and cost evaluation. Valid outputs alone cannot admit it.

The implemented operation is deliberately narrower: one read-only source span
or abstention. It does not implement arbitrary programs, multi-source synthesis,
autonomous actions, neural module discovery, distributed weights or continual
learning. Extending those capabilities must preserve the same explicit boundary
between correct execution and useful answers. Existing native graph/executor
commitments are the eventual integration point; this change does not migrate
the running network.

## Frozen CPU diagnostic

The [plan](../config/experiments/granite-evidence-diagnostic.json),
[execution inventory](../config/experiments/granite-evidence-diagnostic-execution.json)
and [resource limit](../config/experiments/granite-evidence-diagnostic-resources.json)
authorize one new implementation diagnostic after commit and exact-commit CI.
It uses the **unchanged parent only**, with no adapter, training or GPU.

The 64 previously opened cases are development data. The fixed record adapter
identifies the value in every retrieved `... is VALUE.` record by public syntax.
This is a controlled structured-record interface, not a general extraction
algorithm. The same full-history BM25 retrieves three documents for each arm.

| Arm | Input and instruction placement | Output |
| --- | --- | --- |
| Direct control | Conversation and evidence options quoted as JSON, followed by the task | Up to 96 generated tokens; exact answer and citation JSON, requiring EOS |
| Evidence selection | Identical conversation and evidence payload; same instruction placement | One constrained choice token, then deterministic copying |

Both see candidate values and original source IDs. This controls for moving
the task instruction after the conversation and explicitly marking its data.
An improvement shared by both arms is not credited to pointer selection.
Requests and menus contain no task ID, expected answer, category or oracle query.
Both receive one model call; charge actual input/output tokens and wall/CPU time.

Repeat the 24 original assistant anchors first; preserve all 18 prior successes
and the original total/category gates. On the 64 opened cases, the diagnostic
target is 56 correct, including at least 24/32 contextual, 14/16 standalone and
14/16 unsupported requests, with no correct direct-control answer lost. Require
selection p95 at most 30 seconds and at most 1.25 times direct-control p95.
Every pointer output must have a valid request-bound receipt.

Only if those conditions hold, reload the parent and repeat three declared
cases in both arms; all six generated decisions/answers, menus and receipts
must match. A failed gate stays failed. There is no automatic retry or threshold
tuning. **Even a pass is development evidence only:** it cannot close A1, prove
learning, admit a module or upgrade serving. A later learning/reference claim
needs new cases and an actual neural-capability comparison.

One CPU `r7i.4xlarge`, 80 GiB gp3, two-hour expiry, $6 planning allowance including
a storage/transfer reserve. Worker limit is 60 minutes/64 GiB. Evidence is copied
before automatic retirement, with the previous allocation's retirement receipt
pinned. Source and runtime checks run before inference. Interrupted work counts.

The [preflight](../config/experiments/granite-evidence-diagnostic-preflight.json)
checks all menus, label tokens and prompt lengths without pretrained generation.
A tiny random BF16 model exercises the actual native one-token constraint path;
its arbitrary choice is only an implementation check.
