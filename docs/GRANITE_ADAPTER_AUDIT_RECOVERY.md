# Granite adapter audit: vocabulary comparison amendment

September 26, 2026. **One amended execution, committed before allocation.**
The [first attempt](GRANITE_ADAPTER_AUDIT_RESULTS.md) remains stopped and its
receipts remain unchanged. This amendment completes the implementation comparison;
it does not train a model, change the failed reference gate or close A1.

## Correction

The published Switch vocabulary adds 12 control tokens. Whole embedding hashes
cannot match a parent with fewer rows. The amended comparison checks:

- The exact declared shapes: parent **100,352 × 2,560**, Switch
  **100,364 × 2,560**, for input embeddings and the tied output head only.
- Every original vocabulary row matches exactly in BF16. Every other backbone
  tensor still requires exact shape, dtype and value equality.
- The 12 added rows are finite; record their hashes, dimensions, nonzero counts,
  minimum and maximum. They are not assumed to be zero or removed from inference.
- The complete input/output vocabulary matrices are equal and share storage.
  Control IDs are precisely 100352–100363 and substitutions remain in the parent
  vocabulary.
- All original adapter tensor, scaling, padding and non-target checks remain.

The output vocabulary is still larger. This is not a full-logit equality claim;
the original exact generated-answer and replay comparisons remain the behavioral
check. No logits are masked or edited to manufacture agreement.

All 243 published backbone mapping shapes were checked locally on the meta
device. Regression tests accept the declared expansion and reject changed
original rows, nonfinite added rows, an untied or changed head, undeclared shapes,
changed dtype and incorrect control IDs. A worker test verifies that structural
failure stops before any answer generation.

## Execution and unchanged method

The [original plan](../config/experiments/granite-adapter-audit.json) remains
byte-for-byte unchanged. The [amendment](../config/experiments/granite-adapter-audit-recovery.json)
and [new execution freeze](../config/experiments/granite-adapter-audit-recovery-execution.json)
bind that plan and both previous result files.

First load the parent to record mapped tensor hashes, shapes and dtypes. Then
load Switch and perform all backbone and embedded-adapter checks **before
generation**. Generate its 16 answers, reload the parent with the standalone
adapter and repeat its 16 answers. The order changes to avoid repeating
generations before a structural stop. Both arms are generated anew; earlier
standalone answers do not replace new measurements.

Keep exactly the same 16 opened cases, prompts, weights, tokenizer and upstream
revisions, activation offsets, BF16 eager CPU execution, eight threads, seed and
32-token cap. Keep pairwise comparison, replay against the original Switch
outputs and all quality interpretations. No new evaluation set is opened.

## Accounting and stop

The stopped attempt used **192.14 worker wall seconds**, **282.84 process CPU
seconds** and **$0.086284 conservative instance compute**. New receipts carry
that accounting and add all new loading, checking and repeated generation.

One additional `r7i.4xlarge`, at most two hours, 80 GiB disk, 64 GiB worker cap,
60-minute worker limit. The cloud guard includes prior compute plus the full
new instance allowance and a $3 storage/transfer reserve within **$6 combined**.
Maximum combined instance time is 7,494 seconds. Prior resources must be retired.

The existing `granite-adapter-audit` controller profile now selects this new
execution/resource freeze; the historical checkout still identifies the original
attempt. It requires CI to pass for the new commit before allocation. Evidence
is copied and all temporary resources are retired on completion or error.
One amended attempt only; no automatic retry or change to gates.

Agreement would complete this implementation diagnosis while leaving the
published checker's quality failure intact. Any remaining discrepancy is
reported as such. Further capability work requires a separate prospective
decision; neither outcome automatically authorizes training or promotion.
