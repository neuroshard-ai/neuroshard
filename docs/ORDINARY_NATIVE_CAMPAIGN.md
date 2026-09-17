# Prospective ordinary learning campaign

This campaign tests checklist items 1 and 2 together. It starts from the exposed
15/15 ordinary development service, then requires three new learning cohorts to
pass complete automatic answering and preservation before native promotion.
The development repair is the initial model, not an additional passing cohort.
The checklist remains open until the operated evidence meets its criteria.

The source-grounded admission, conversation and feed cohorts each contain 16
facts, 384 training conversations and 32 prospectively frozen ordinary questions.
The final contains single questions and combined questions, including questions
that need earlier experts. Twelve general assistant questions and eight
multi-turn conversations are fixed preservation anchors. These are bounded
assistant tasks, not a claim of frontier-model capability. Evaluation labels and
answers reach the scorer only; the answering system receives the conversation.

The initial bootstrap exposed a general-answering failure before any full cohort
was trained. Its real prefix and one update were independently replayed and
accepted; the measured quality failure was audited and rejected, keeping serving
unchanged. The repaired answering system now has the following measured results:

| Screen | Knowledge | General skills | Conversations | Previously correct lost |
| --- | ---: | ---: | ---: | ---: |
| Original native bootstrap baseline | 24/25 | 3/12 | 4/8 | — |
| First general-interface diagnostic, failed | 22/25 | 6/12 | 5/8 | 4 |
| Complete worked-answer development diagnostic | 25/25 | 10/12 | 8/8 | 0 |
| Separately frozen prospective assistant screen | — | 10/12 | 6/8 | Fresh cases |

Both final rows used automatic selection and actual owned shards, and replayed
their first complete response exactly. The prospective screen passed its
precommitted 75% floor in each category. It still failed four questions: one
comparison was routed incorrectly, a conversation calculation was wrong, a
sort was wrong, and a word-position answer was wrong. Those failures remain in
the record; these results establish bounded assistant competence, not broad
ChatGPT-level capability or another learned cohort.

The repair preserves mixed-question decomposition and earlier free-form replies,
fits a general/specialist guard using separate training inputs, and uses bounded
worked reasoning for explicitly constrained general requests. Its six worked
examples contain neither the installed expert facts nor evaluation answers.
The final-answer boundary only extracts generated text; it cannot certify its
truth. All reasoning tokens are included in complete replay and native billing.
A temporary full-model oracle outside the stopped network first matched all
218 preserved tensors and all ten control token streams. Two earlier reasoning
methods scored 12/20; an explicit final-answer reminder scored 18/20 before
integration. The full-model oracle was deleted before distributed evaluation.

The [development outputs](../config/experiments/general-interface-worked/diagnostic-result.json)
and [prospective outputs](../config/experiments/general-assistant-holdout-results/result.json)
bind the measured complete replies. The
[renewed prescription](../config/experiments/ordinary-native-repaired/operation.json)
retains every unopened admission/conversation/feed training and final byte,
domain gate, training recipe and quality threshold. All twenty prospective
assistant cases join retention, yielding 25 knowledge, 24 skill and 16
conversation anchors. No full cohort has passed or been promoted under this
renewed prescription yet. Checklist items 1 and 2 remain open.

Serving allocation now reserves FP32 weights and working memory without
reserving gradients or Adam; training retains its original conservative guard.
The parameter and context limits are unchanged. Numerical checks preserve exact
owned outputs while checking that inference cannot bypass the parameter limit.
The renewed native source commitment includes this allocation correction.

Each full job uses 128 updates, batch size 16, microbatch 4, maximum training
length 256, learning rate 0.00005, four warmup updates, weight decay 0.01 and
gradient clipping at 1. Only its terminal checkpoint is eligible. A new tail
starts from the preserved C expert with fresh Adam state. Earlier experts stay
available. Ordinary training paraphrases fit only the appended selection gate.
This is isolated addition by default; updating or consolidating remains allowed
when the complete service demonstrates preservation and benefit.

Admission requires at least 80% new single answers, 75% combined answers, a
strictly positive 95% paired-bootstrap lower bound on single-answer gain, no
previously correct retained answer lost, and at least 75% accuracy in each
retained knowledge, general-skill and conversation category. Every earlier
admitted evaluation joins retention. All numbers and decoding rules are frozen
before numerical training. Failure is audited, recorded and stops later jobs
under this prescription. It cannot authorize tuning on that final.

The first one-update job exercises rejection using separate frozen questions.
An unexpected pass must be published and stops this prescription; the controller
cannot manufacture a failure. The first full cohort consumes the remaining
fresh source rows and replays the 16 rows actually trained in that first job.
Later cohorts replay 16 actually trained rows from the preceding cohort.

Four native CometBFT validators run a new experimental genesis. The existing
publisher and curator/auditor daemons consume monotonically advancing immutable
public feed heads, review source correspondence and contamination, fund jobs,
assign owned computation, audit every window, and submit complete quality for
promotion or rejection. The publisher restarts from its durable signing journal
between ticks. A separately hashed source window with a substituted answer must
be quarantined before funding. Serving probes continue on the accepted graph
during training, audits, rejection and promotion. Only future verified numerical
windows issue rewards; initial model history cannot mint retrospectively.

Seven disposable GPU hosts own the computation. No host receives the full
backbone or preserved interpreter. Three owned prefix stages feed a separate
tail owner; each of three auditors executes its own fresh stages and updates.
The source-backed curator is installed policy for this reviewed corpus. This
does not claim automatic semantic truth verification for arbitrary web data,
independent operator ownership, cheap verification or cross-hardware equality.

The first full cohort also has a fixed-capacity control replacing C with the
same newly trained weights. Each arm receives seven hosts, the same disk budget
and a 5,400-second interval including computation, three full audits, complete
quality and two serving replicas. The control recomputes work without submitting
it for issuance. A fixed queue of training prompts fills the remaining interval.
Report quality, retained losses, throughput, transfer/storage and request overruns;
completion of the comparison alone does not establish a benefit from growth.
This measures one workload, not lifetime cost or the optimal fixed-capacity method.

`prepare_ordinary_cohorts.py` builds the training-only selectors and scoring
inputs. `freeze_ordinary_campaign.py` binds immutable inputs, policies, source
revision and numerical profile. `run_ordinary_campaign.py` separates allocation,
setup, operation and retirement. Actual runtime observations must match the
committed profile before initialization. The initial L40S allocation could not
place the complete pool. Its partial allocation was retired. The replacement
prescription uses seven `g5.xlarge` A10G instances, at most two concurrent fresh
evaluations, an absolute twelve-hour termination deadline and the same $150
planning allowance. Actual compute and storage costs are reported separately.
The data, learning recipe, gates and matched comparison interval are unchanged.
Allocation and retirement explicitly exclude the three protected network hosts.
