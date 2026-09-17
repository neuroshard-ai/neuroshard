# Continuing an accepted expert

The next learning decision is whether the existing sharded expert can acquire
fresh knowledge without losing its earlier correct answers. This precedes another
capacity expansion. It is part of tasks 1 and 2 in [the fixed checklist](../TODO.md).

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
