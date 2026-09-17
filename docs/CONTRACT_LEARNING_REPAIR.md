# Repairing C after ordinary access

The request-preservation and subject-resolution changes bring the unchanged
expert graph from 7/15 to **10/15** on the exposed ordinary development screen.
All 21 standalone routes now select the expected expert. No selection,
decomposition or assembly failures remain; all seven retained cases pass.
Six owners agree on complete transcripts, and automatic and forced replay pass.
The five remaining failures come from three incorrect C answers. This closes
the bounded access repair, not the full learning milestone or public readiness.

The first preservation attempt scored 9/15 and failed one reference repair.
Both attempts are retained locally, with CPU verification, source archives,
all-owner evidence and resource retirement records. No expert was trained in
either inference attempt. See [the access record](ORDINARY_ACCESS_TRIAL.md).

The first [committed prescription](../config/experiments/contract-learning-trial.json)
tested one training-data intervention. C's 192 original conversations contain
144 distinct unwrapped questions: 128 occur under only one wrapper, and 16 under
four. The new preparer crosses every question with all five existing wrappers
and bare wording, producing 864 conversations. Targets come exclusively from
the original training records. The question strings from development and the
opened final are used only for exclusion, before tokenization; no scoring
answers enter training. No fact, evaluation wording or serving answer lookup
is added.

The hypothesis is that independent coverage of wording and prompt format
reduces C's input sensitivity. This is a hypothesis about this model, not an
established explanation of all three errors. Prior work supports systematic
coverage when injecting facts through SFT, but used different models and data:
[Mecklenburg et al.](https://arxiv.org/abs/2404.00213).

Only C's existing 134,225,920-parameter tail receives updates, starting from
`618e3eb0…` with fresh Adam. The fixed recipe is 108 steps, batches of 16,
microbatches of four, learning rate 0.00005, four warmup steps and response-only
cross-entropy. Two full dataset passes cover every conversation twice. The
backbone, A/B experts, selector, question handling and generation rules remain
fixed. The current C weights are not permanently protected from improvement;
the accepted behavior is what the quality gate protects.

Only the terminal checkpoint is evaluated. Acceptance requires all 15 ordinary
cases and all 13 distinct forced controls correct, preservation of all 13
previously correct C standalone development answers under their exact earlier
prompts, exact replay and complete-call metering. No new final opens, no next
cohort starts, no NEURO is issued, and this experiment cannot promote serving.
The actual full-system result decides whether the method helped.

The [completed result](../config/experiments/contract-learning-results.json)
improved ordinary answering **10/15 → 14/15**, forced answers **10/13 → 12/13**,
and C's earlier standalone development questions **13/16 → 15/16**. No previously
correct answer in either preservation inventory was lost. All four training
owners agreed; all six serving transcripts agreed. A fresh process reproduced
updates 104–108 from the input boundary, and both inference replays passed.
The 108 updates took 985 seconds. All four instances, volumes and the temporary
security group were retired; estimated compute was at most $2.28, with storage
and transfer separate. The candidate still **failed** the unchanged all-answer
gate. These are exposed development results, not a fresh final or new cohort.

The remaining error confuses a single claim's window with a complete job:
the expert answers `4096` where the window limit is `16`. Source inspection
establishes the relationship: `planner_work.apply` passes `claim_planner.window`
to `planner_window.validate`, which bounds that window to 16 updates; the
full recipe may prescribe 4096. Existing training variations described a
window without connecting it to a submitted claim.

The separately [frozen scope continuation](../config/experiments/scope-learning-trial.json)
starts from terminal C `f1aa7624…`. Its
[training variations](../config/experiments/planner-scope-questions.json) add
six claim-window questions and six contrasting whole-job questions. The same
original training inventory supplies every target; all other topics remain.
Crossing with the six input contracts gives 936 conversations. Development
and opened-final question strings remain excluded. This intervention is
explicitly informed by development failures and cannot establish independent
generalization, even if the diagnostic passes.

The continuation freezes 64 updates: all 59 balanced batches once, followed
by the first five balanced batches. Learning rate, optimizer, microbatch,
question handling, selector, A/B experts and backbone use the previous recipe.
Only the terminal C checkpoint is tested. It must preserve all **14** currently
correct ordinary cases and all **15** currently correct C development answers,
as well as pass the original all-answer, replay and metering gates. This is a
small repair within the same cohort, not completion of TODO items 1 or 2.

The first scope allocation stopped during object download, before training:
publication receipts lacked the fetch inventory's `folder` field. Preparation
now derives the owned `objects` destination from the checkpoint, validates
hash and size against that checkpoint, and rejects conflicting destinations.
The recovered allocation uses identical training bytes, recipe, schedule and
retention inventory. Its freeze changes only to bind the corrected handoff.
The original attempt remains recorded and is not counted as a quality run.

One four-host allocation performs sharded prefix production, C training,
replay of the final four-update window from its input state, and six-owner
serving. Its absolute deadline is two hours, training deadline one hour and
compute planning cap $25. The final two boundaries are retained in the existing
immutable project object store before retirement. Earlier window metadata alone
does not establish payload availability for settling all those windows.
