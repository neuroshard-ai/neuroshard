# Compose the learned second expert

The fifth peer completed 560 updates while earlier paths served requests. Its
single-fact development answers improved from 7/64 to 49/64. Direct generation of
two answers reached 2/16 and failed. Retention also caught a prompt serialization
bug: saving the configuration reordered the interpreter's example keys, changing
the actual model input despite an unchanged semantic configuration identity.
The [failed trial](../config/experiments/interpreted-cohort-results.json) remains
failed; it never opened the second cohort's final.

This experiment restores that trial's terminal checkpoint. It performs no further
training. Its five owners retain the same parent, first expert, original neural
interpreter and second expert; no owner holds a complete 1.7B model.

Two changes address the observed failures:

1. Serialize interpreter example arguments explicitly as `name`, then `field`,
   with the exact spacing used by the successful first graph. Bind both the
   rendered messages and actual prefix token IDs into the new serving graph.
2. For an explicit `First: … Second: …` request asking for two semicolon-separated
   answers, derive two questions from the raw text and execute the learned expert
   twice. Join the unchanged neural outputs. Record both generated token sequences,
   both input questions, and the separately labeled rendered response.

The composer receives no source document, expected answer, topic identifier or
evaluation role. It is a bounded typed request primitive, not a general language
planner. Single questions and unrecognized request formats execute the existing
greedy expert directly. Each constituent call has a 64-token limit; a composed
request may therefore use twice the neural generation budget. No comparison at
equal inference cost is claimed.

Offline reuse of the already observed single-question development calls yields
10/16 correct pairs. This diagnoses a useful decomposition; it is not a new live
result. The operated experiment must actually execute both calls on five separate
owners and validate their complete transcripts.

The [frozen plan](../config/experiments/composed-cohort.json) preserves the earlier
quality gates: at least 75% single-fact accuracy, 50% paired accuracy, paired gain
lower bound above 0.10, and zero changes to retained answers or losses. Development
has 64 singles and 16 pairs. The previously unexecuted final has 64 newly worded
singles and 32 other fact combinations. Facts were training material; wording and
final combinations were held out. Previously exposed first-expert questions serve
only as retention probes.

Preparation must commit source, input hashes, the existing training identity,
checkpoint and exact interpreter prompt tokens. Development evidence and an
eligible selection must be archived and committed before final execution. A
failed gate closes the trial. Both phases observe the fifth process exit before
the earlier expert and parent serve again. Model objects already have complete
storage readbacks; all temporary resources retire after evidence preservation.

The CPU integration test performs real training, restores its saved tail into five
fresh processes, executes both composed calls, compares them with a complete-model
oracle, checks old answers and losses, and observes the fifth process exit. This
does not substitute for the frozen GPU quality experiment.

This is operated model research. It adds no native issuance, consensus change,
public serving promotion or claim of independent ownership. The original training
job identity remains attached to the checkpoint; composing its calls creates no
new payable training work.
