# Continuing an accepted expert

The first continued-expert trial improved some new answers but lost previously
correct answers and was rejected. It is part of tasks 1 and 2 in
[the fixed checklist](../TODO.md); neither task is complete.

Continued jobs bind a named accepted expert in `seed_expert`. Initialization copies
its weights and resets Adam. The parent remains immutable. Both distributed
feature production and the native prefix referee compute replay references from
that accepted expert; a read-only tail replica counts against the final parent's
resident parameter limit. Later audited windows need their current complete
boundary, rather than the previous cohort's seed files or optimization history.

The new continual quality profile generates old and new answers. Any previously
correct retained answer becoming incorrect fails the gate. Its initial assistant
anchors and thresholds remain fixed, and admission requires every earlier admitted
evaluation conversation in subsequent retention sets. A proposer cannot silently
omit a difficult old question, rewrite its reference answer, or put protected
retention conversations into fresh training. Mechanical provenance and duplicate
checks still require semantic curation; they do not prove that arbitrary data is
true or harmless.

The graph executor can hold one alternative expert revision on that expert's
owner for paired evaluation. Both sets of weights stay frozen and count toward
the resident limit. Comparing a candidate does not promote it or overwrite the
accepted serving weights. Native quality funding includes the additional retained
question pairs.

## Frozen method trial

[The prescription](../config/experiments/continual-expert-trial.json) continues the
accepted 134,225,920-parameter tail of the sharded 1.7B model. Four machines own
three parent partitions and the learner. No owner loads the complete backbone.
The learner also needs the frozen output head, and reference production needs the
explicit tail replica described above.

- First cohort: 16 source-anchored facts about newer NeuroShard protocol behavior.
  Four training wrappers and separately worded development/final questions are
  reconstructed from [the curation](../config/experiments/continual-expert-facts.json).
- Training: 192 fixed updates, learning rate 0.00005, fresh Adam, and batches of
  four new plus four replay conversations. Ninety-six replay conversations cover
  all 64 earlier fact topics and come only from batches used in the completed
  560-update expert trajectory. Replay uses accepted-expert KL and margin terms.
- Acceptance: at least 75% single and composed answer accuracy, a positive paired
  bootstrap lower bound on new single-fact gains, and zero lost correct answers
  on the 96 earlier evaluation questions. A loss decrease cannot substitute for
  these checks.
- Selection: only the fixed terminal checkpoint. A failed development gate
  leaves the new final set unopened. A passing development result requires
  publication of the terminal decision before the controller releases the final.
- Resources: four temporary GPUs, 200 GiB disks, a 2.5-hour allocation deadline
  and a $50 planning cap. Preserve evidence and retire the allocation afterward.

This trial measures direct specialist learning on actual distributed neural
computation. It does not demonstrate automatic routing, three admitted cohorts,
a matched-resource advantage from growth, a running public chat service, or an
independent operator. The other two curated cohorts are reserved for subsequent
learning; they are not included in the first training job. Numerical work and
model-quality approval remain separate obligations before native settlement and
serving promotion.

The committed driver was exercised on four CPU processes before GPU allocation:
it produced and persisted a new trajectory from accepted weights, generated both
sets of development answers, rejected the failed candidate, and left final
questions unopened. This preflight is execution evidence, not a quality gain.

## Measured result: rejected

The four GPU owners completed all 192 updates under source commit
`76aeaf5864be9d757959cc57c2c3cedeaf26d259`. Full CI passed for that source.

| Direct expert answers | Before | After | Previously correct answers lost |
|---|---:|---:|---:|
| New single facts | 2/16 | 7/16 | 0 |
| New composed facts | 0/8 | 1/8 | 0 |
| Retained single facts | 57/64 | 55/64 | 4 |
| Retained composed facts | 7/32 | 6/32 | 4 |

The new single-fact paired bootstrap interval was `[0.125, 0.5]`, but both
accuracy thresholds and the retention gate failed. Five gains on other retained
questions cannot offset the eight lost answers. One lost response changed the
number of atomic units per NEURO from `1000000` to `1000000000`: this includes
substantive forgetting, not only output formatting differences.

The final test stayed unopened. No native job was activated, no NEURO was issued,
and the serving model was unchanged. A fresh Python process restored the step-188
input boundary and reproduced all four final updates exactly, including Adam.
This establishes execution reproducibility for that window, not useful continual
learning or an independently administered audit.

The retained composed baseline is **7/32 on direct generation**. The earlier
**23/32 serving result** used two composed expert calls; these are different
execution paths and cannot be compared as the same benchmark. Promotion still
requires evaluation through the actual serving graph.

[Published result and artifacts](https://github.com/neuroshard-ai/neuroshard/releases/tag/research-continual-expert-20260917)
include generated development and retention answers, every window's metadata,
the replay report, source/input freeze, and public step-188/192 tensor catalogs.
All 36 preserved tensor objects passed full public download/hash checks, totaling
3,221,433,520 bytes. The result archive is 4,886,131 bytes with SHA-256
`3659b07782602bca462745ec95055f3479d0c741fd1466c791998e4bc2ef273c`.
The failed trajectory was not settled; intermediate windows retain metadata and
can be reconstructed from the frozen inputs rather than claiming complete paid
retention of every boundary.

All four temporary instances, disks and the security group were deleted.
The conservative compute estimate is $3.16, with storage and transfer additional.
Existing services were unchanged.

This result rejects this update prescription. It does not establish that merely
adding another expert will fix learning: a preserved old expert still needs
correct routing, the new expert must learn its facts, and combined questions
must pass through the serving path. Those remain explicit requirements before
another learning or growth claim.

## Preparing native cohorts

`expert_preparation.prepare` now reads bounded, consecutive rows from immutable
source descriptors and creates the native conversation, provenance, token and
batch commitments. It starts from the ledger's cursors, records every selected
row, and rejects missing rows, unsupported publishers and repeated conversations.
Replay keeps its original source position and token commitment and requires an
accepted training-window entry; an imported checkpoint does not fabricate replay
history. Preparation can restart with the same snapshot and recover identical
content-addressed inputs without advancing a ledger cursor.

The numerical owner must produce the actual zero-update checkpoint of the
prepared job. `expert_preparation.seal` then binds that checkpoint to the candidate
graph and frozen quality contract, rereads the upstream data, and runs the native
data reviewer before returning a proposal. It refuses changed admission history
or serving state. Under the continual quality profile, the previous cohort's
admitted evaluation questions automatically join the next retention obligation,
including questions from a rejected candidate. The initial anchors and scoring
rules remain fixed.

Five preparation/integration checks cover restart, source replacement between
preparation and review, stale state, missing data, replay eligibility, accumulated
retention and actual CPU prefix production, two updates and fresh-process replay.
They prove the prepared proposal drives the numerical executor. They do not
measure model-quality improvement. Curator approval, assignment, audit funding,
transaction submission and promotion still need an operated repeating loop;
this preparation API does not complete task 2.


## Isolated learning result

The next trial kept accepted A/B weights and the earlier router intact, trained a
new 134,225,920-parameter tail from accepted B, and fitted only an appended routing
gate. All 512 sharded updates completed on four GPUs. Evaluation used the actual
learned service and explicit two-question calls:

| Answers | Before | After |
|---|---:|---:|
| New single facts | 2/16 | 11/16 |
| New composed facts | 0/8 | 4/8 |
| Retained single facts | 57/64 | 57/64 |
| Retained composed facts | 23/32 | 23/32 |

All 80 previously correct retained answers survived. Both new-answer accuracy
gates failed. The final set remained unopened, with no native activation,
issuance or serving promotion. A fresh process given only the step-508 boundary
reproduced steps 509–512 exactly, including optimizer state.

[Published evidence](https://github.com/neuroshard-ai/neuroshard/releases/tag/research-isolated-expert-20260917)
includes the complete generated answers, source freezes, preserved boundary
tensors and full public hash checks. All four instances, disks and the temporary
security group were deleted; compute was at most $6.42, storage/transfer additional.
The separately frozen replacement control failed during setup and generated no
answers; it supplies no measured growth comparison. Its failures are preserved.

Every incorrect new answer selected the intended expert. Failed composed replies
contained an incorrect constituent response. The original training variations
mostly changed framing around an identical core question. The next intervention
therefore adds semantic wording variations for every training topic and converts
paired training into the atomic questions the serving parser executes. Targets
come exclusively from the original training inventory. This is explicitly a
development-informed intervention; it restarts from accepted B, keeps the same
512-step recipe and gates, and leaves the original final set unopened. It is not
a result until the frozen run has completed.

The [frozen semantic trial](../config/experiments/semantic-expert-trial.json)
contains 192 atomic training conversations: 12 per topic, interleaved in 24
batches, with targets and provenance checked against all 96 original training
records. Actual tokenizer validation confirmed complete response masking and
supervised EOS on every row. The run compares appended capacity with replacing B
using the very same learned weights; it includes fresh boundary replay, public
terminal publication before final evaluation, and automatic resource retirement.
The four-host allocation has a three-hour/$50 cap. This comparison matches training
work and still does not measure equal lifetime storage and serving costs.
