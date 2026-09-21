# Programming selector evaluation contract

**Status: contract frozen; all four pickers failed their screens. Campaign closed.**
This contract authorized preparation of CPU screens. It did not launch GPUs,
train a tail or a selector, admit a model, or change the failed merge result.
The original 128-task programming-expert final remains closed. The next
experiment is [learned integration](LEARNED_INTEGRATION.md).

Machine-readable contract:
[`programming-selector-contract.json`](../config/experiments/programming-selector-contract.json).
The contract commit freezes the evaluation rules. A later picker freeze must
bind one implementation before its first screen; an absent picker is not a
passing policy and cannot trigger a GPU allocation.

## Evidence and question

An independent sandboxed CPU rescore reproduced the saved complementarity
diagnosis exactly. Across 64 already-opened questions, the parent solves 22,
the incumbent fallback solves 29, always using the added tail solves 31, and
the oracle union solves 32. Among the 38 parent-example failures, the incumbent
and added tails solve seven and nine respectively. Three successes are unique
to the added tail; one is unique to the incumbent. The unit merge remains failed.

The next question is whether a selector can recover that complementary coverage
using information available **before** either extra answer is generated.
The oracle is a bound, not a policy. The 64 questions, all 38 extra-decode cases,
and their disclosed unique IDs are development data, permanently ineligible as
fresh admission evidence.

The diagnosis used the growth freeze `dcc6693` and its unchanged checkpoints,
but the diagnosis extension was outside that freeze's `EXECUTION_SOURCES`.
The artifact hashes in this contract bind the saved evidence retrospectively;
they do not assert that the diagnosis extension was preregistered. The original
growth freeze and the later selector freeze are separate identities.

## Serving and picker input

The parent and both saved tails remain unchanged. Generate the parent with the
existing prompt, tokenizer and greedy 256-token cap. Run only the public example
already present in that prompt. If it passes, return the recorded parent answer
and perform no selection or extra decode. Otherwise call the picker once and
generate once from its selected tail, with the original prompt and the same cap.
Never generate both tails, inspect their answers, or try the other tail afterward
as part of a served request.

The picker receives only this JSON object, with no additional keys:

```json
{
  "question": "the exact original user message, including its public example",
  "failed_parent_program": "the parent's exact decoded response",
  "public_example": "the single example already visible in question",
  "public_feedback": {
    "passed": false,
    "status": "execution-error"
  }
}
```

Feedback comes from the existing Python extractor and sandbox, using the public
example alone and an empty setup. Failure status is one of `extraction-error`,
`execution-error`, `timeout` or `early-exit`. All 64 diagnostic rows have empty setup.
Never supply other tests, reference code, task IDs, row order, split labels,
checkpoint outputs, success labels, oracle membership, file paths, timing,
stderr or random sandbox identifiers. A sandbox infrastructure error invalidates
the run; it is not a model failure or a routing feature.

The picker returns `incumbent`, `added`, or `abstain`. The default is **incumbent**:
ties, uncertainty, abstention, missing features, invalid output, a picker error,
or its one-second deadline select the incumbent. Any confidence thresholds and
tie rules must be explicit in the picker freeze. No threshold may be adjusted
after viewing the screen score.

The picker is deterministic, CPU-only, uses one thread and at most 512 MiB,
has no network access, and reads only its predeclared immutable assets. No tail
forward pass or external model call is allowed. It may use a subset of the
allowed fields. Explicit evaluation-question/ID/hash lookups and case exceptions
are prohibited. Pin every source file, asset, dependency, feature transform,
threshold and inference setting before running it. This contract does not
authorize fitting a picker on the disclosed diagnostic labels.

## CPU screen: one candidate, one measurement

Commit one picker artifact and execution manifest. Generate the 38 input-only
views from pinned parent traces and freshly replayed public-example checks.
Run the picker without mounting either tail's saved outputs, full tests,
reference programs or diagnosis labels. Write and hash all decisions and picker
timings **before** a separate scorer joins the chosen saved outputs. Transport
IDs may map decisions back to cases, but must never enter picker input.

The scorer checks exact, unique coverage of all 38 decisions and all 64 parent
cases. It independently grades the selected responses with the unchanged full
test lists in the existing sandbox. Fixed controls are incumbent fallback,
always-added fallback and the non-deployable oracle union. Raw hidden-test
feedback must never flow back into the picker.

The CPU screen qualifies only if **all** conditions hold:

- recover all three unique added successes;
- preserve all 29 incumbent-policy successes, including all 12 old successes;
- reach 32/64, hence +3 over incumbent and +1 over always-added;
- use at most one extra decode per hypothetical request;
- picker p95 wall time is at most 50 ms, with zero errors or deadline overruns;
- artifact, coverage and input-isolation checks pass.

Time the entire picker call, including per-request parsing and feature work.
Immutable assets may be loaded once; report that startup separately. Use the
nearest-rank p95 over the 38 calls (`ceil(0.95 * 38) - 1` in sorted zero-based
samples). Do not select the fastest repetition or exclude abstentions.

Failure stops this picker candidate. There is no automatic parameter sweep,
second picker, coefficient search, tail training or GPU retry. Any later method
revision needs its own declared experiment; these cases remain opened. A score
without the committed picker manifest is invalid. Even a complete pass must
report `admission_evidence: false` and `gpu_authorized_by_screen: false`.

Reconstructed latency may be reported as parent generation + public check +
picker + the chosen recorded extra generation. Those traces came from separate
allocations, so this is an estimate, not a new end-to-end latency measurement.
No bootstrap interval on these opened cases is admission evidence.

## Fresh confirmation eligibility

There is no untouched, eligible non-final remainder in MBPP 11–510: that pool
has been assigned to training, opened evaluation or exclusions. Do not relabel
training rows or the 21 near-duplicate exclusions as a fresh holdout.

The audit in
[`programming-selector-pool.json`](../config/experiments/programming-selector-pool.json)
instead uses **MBPP 511–600**, excluding the 32 original opened development
questions. Of the remaining 58, 14 match training or opened evaluation prompts
under the existing Jaccard rule; 44 remain. The audit ranks them deterministically
and reserves the first 32 now. All have empty setup. Neither tail's generation
on these 32 has been evaluated in the pinned programming-trial records. This
does not establish absence from upstream model pretraining: MBPP is public.

The audit reads prompt text for mechanical duplicate checks, not model answers
or reference solutions for policy design. The original final is not used.
No selected task may be swapped after observing a candidate output. A source,
reference-validity or runtime preflight failure aborts readiness and requires a
recorded correction before any model generation; it is not a scored task loss.

A passing CPU screen makes a **separately committed** confirmation execution
freeze eligible. It must bind the exact screened picker and inputs, the 32 IDs,
both checkpoint manifests and tensors, tokenizer, numerical/runtime settings,
scorer, all source/dependency hashes and resource controller. Required CI and
all four hosts' hash inventories must pass before starting the workload. The
maximum allocation is four g5.xlarge hosts for two hours and $50; setup attempts
count toward that single budget. Preserve evidence and retire the allocation.
This contract does not create or schedule that allocation.

The confirmation reports incumbent, always-added and selected policies on the
same fresh 32, with paired control generation used only by the evaluator. The
served policy still chooses before generation and uses one tail. Qualification
requires all of:

- the frozen 64-case preservation replay keeps all 29 incumbent successes;
- fresh net gain of at least two tasks over the incumbent policy;
- positive one-sided 95% paired-bootstrap lower bound versus incumbent, using
  10,000 resamples and seed 20260921;
- at least one fresh net gain over always-added, so selecting has measured value
  beyond simply replacing the incumbent tail;
- measured selected-policy p95 latency, including public check, picker and
  request-time tail loading, at most 1.5 times incumbent and at most 90 seconds;
- the unchanged one-extra-decode cap, complete traces and zero picker errors.

A miss stops this candidate. A pass is supporting evidence for bounded
selection; neither result automatically promotes the accepted graph, issues
NEURO, changes 0.4.0, completes a checklist item, or demonstrates independent
operation. The CPU screen's three gains cannot be pooled into the fresh score.

## Immediate next action

Specify and commit **one picker implementation**, its assets and thresholds,
then run the CPU-only screen. No picker was fitted or scored while freezing
this contract. A frozen evaluation contract is not a claim that a useful picker
already exists.
