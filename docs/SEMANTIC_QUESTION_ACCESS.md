# Semantic access to learned facts

The complete opened development diagnostic now reaches **15/16 single answers
and 13/16 combined answers**, versus 13/16 and 9/16 on the same expert weights
with the earlier interface. No previously correct retained answer is lost.
Knowledge is 31/33, skills 20/24 and conversation 14/16. Every earlier neural
reply was reproduced, and a complete combined request replayed exactly.

This passes the diagnostic's absolute accuracy and retention floors. It does
not turn the original frozen trial into a pass: no expert was trained, no new
final was opened, no native promotion occurred, and the paired single-answer
gain interval against these already trained weights still includes zero.
Two formerly correct combined diagnostic cases and one single case regressed;
these are reported even though the absolute diagnostic floors passed.
Checklist items 1 and 2 remain open until prospective repeated learning passes.

## What changed

Separate forced controls established that the trained admission expert answers
all sixteen canonical training questions correctly. The obstruction was access
to those facts under ordinary wording. A 17-way linear classifier on the
existing input embeddings recognized only 17/29 distinct new-domain questions.
Prompted matching with the preserved interpreter was worse: 6/79 total matches
and 50 false-positive selections. Neither method was adopted.

A frozen [BGE-small English encoder](https://huggingface.co/BAAI/bge-small-en-v1.5),
revision `5c38ec7c405ec4b44b94cc5a9bb96e735b38267a`, supplies semantic features.
The model is MIT licensed. It has 33,360,000 parameters and runs on the existing
first owner, inside that owner's resident parameter limit. The full answering
backbone remains partitioned across owners. This adds an encoder, not another
copy of the complete language model.

The index contains only training-question vectors, source identifiers and intent
labels. Its canonical dictionary contains questions and routes, never answers.
Nearest training-question selection recognized 28/29 new-domain queries with
zero false positives on fifty other-domain queries. Those measurements use
opened development cases and are not independent generalization evidence.
The expert still generates each selected fact from its trained weights.

The complete policy binds encoder and tokenizer files, the training index,
canonical questions, source, expert weights and request handling. The trace
records the actual encoder input tokens, features and chosen training example.
Rejection preserves the previous selector. Contextual conversations keep their
existing path. Encoder artifacts are hash checked; the immutable encoder and
its extra resident parameters are checked during execution. Encoder features
are not fictitious text output tokens. Any production cost quotation must
include their actual computational cost; this diagnostic is sponsored.

An initialization failure also exposed a loader bug: the declared 8 MiB policy
artifact limit was accidentally followed by the 2 MB demo-message parser.
Policy loading now uses its own existing limit and retains duplicate-key and
nonfinite-number rejection. Network message limits remain unchanged.

The [complete result](../config/experiments/semantic-answering-diagnostic-20260918b/results/result.json)
and [visible replies](../config/experiments/semantic-answering-diagnostic-20260918b/results/answers.json)
record all outcomes. The next experiment must use new prospective learning
cohorts. The opened admission cases cannot serve as a new final.
