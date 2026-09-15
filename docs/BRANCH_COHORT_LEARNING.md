# Learning another expert while the established model serves

The [next-cohort plan](../config/experiments/branch-cohort-learning.json) tests
one complete continuation of the [retained transformer branch](BRANCH_GROWTH.md).
It adds a second 134,225,920-parameter tail on a fifth owner, bringing the
retained graph from 1,845,602,304 to 1,979,828,224 parameters. The original
1,711,376,384-parameter model and the first learned expert keep their weights
and actual optimizer histories. Each process holds its assigned partition or
expert. This is a prepared experiment, with no second-cohort GPU result yet.

The new expert starts as a copy of the original model's last two blocks with
fresh Adam state. The three parent owners prepare 28 exact padded training
batches once. The new owner reuses those frozen features for 560 local AdamW
updates and has an explicitly read-only replica of the tied output head. That
replica is extra runtime storage, outside the uniquely trained parameters.
The earlier parent and expert use independent process groups to continue
answering during this learning job.

The learning material contains 64 source-anchored facts about the historical
NeuroShard 0.4.0 public profile. Training uses three distinct core questions
per fact, with varied instructions, and 32 combinations of two known facts.
Development has 64 independently worded single-fact questions and 16 new
combinations; the final has another 64 independently worded questions and
32 other combinations. No combination crosses these roles. These are known
facts with held-out wording and combinations, not unseen facts or a broad
assistant benchmark. The answer table supplies training targets and scoring;
the neural decoder and router do not access it.

Both development and final require at least 75% correct single-fact answers,
50% correct two-fact answers, and a paired lower confidence bound above 0.10
for single-fact improvement over the established model. Each fact counts
once in that confidence calculation. Actual generated text is scored;
saved correctness flags cannot authorize a pass. Commands, identifiers and
paths retain case-sensitive checking. Two-fact answers must give both
answers in order. Ordinary greedy generation uses the original tokenizer
and a 64-token cap, with no forced answer prefix or repair.

The fixed terminal step 560 is the only candidate. All earlier branch
knowledge and skill answers must match exactly, as must the retained
conversation losses. The controller must observe responses from both old
paths after the first new update and before the last. It must also observe
the new expert process exit before the retained expert and parent answer
again. All development evidence is committed before final evaluation.

Training requires a passing and preserved result from the first branch,
then a committed numerical source and exact input preparation. The proposed
allocation is five `g5.xlarge` instances, bounded to six hours and a $100
planning allowance, after the earlier allocation is retired. Failed trials
and complete checkpoint artifacts must be preserved before cleanup. Merely
committing the plan does not satisfy the preparation requirement.

The implementation is [cohort_job.py](../src/neuroshard/evolution/sharded/cohort_job.py),
with [run_branch_cohort.py](../scripts/run_branch_cohort.py) as its experiment
entrypoint. A five-process CPU integration test executes the actual worker,
feature production, local updates, checkpoints, serving receipts, retained
answer checks and observed departure. Separate tests compare every update
against independent full-model autograd and reproduce a checkpoint exactly
after restoring the learner's Adam state.

Routing remains an explicit, ordered policy on user text: the first domain
keeps precedence, the historical protocol domain chooses the second expert,
and other questions use the original model. The trial does not establish
general learned routing, hot peer admission, independent operators, native
settlement or economic security. It issues no tokens and promotes no public
serving checkpoint. Earlier failed question sets remain part of the record.
