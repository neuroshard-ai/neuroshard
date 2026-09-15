# Compose the learned second expert

The [closed five-owner experiment](../config/experiments/composed-cohort-results.json)
passed development and its previously unexecuted final. New single answers rose
from **7/64 to 57/64**, and paired answers from **0/32 to 23/32**. All **1,024**
earlier knowledge answers, **768** skill answers and **256** conversation losses
remained exact. All five owners agreed on answer identity
`47a9803d48f4cd2418268f5af69750803169437dc477450f3d2d1a027e9fb09d`.
The fifth process actually exited successfully; the other four then served the
earlier paths. No owner held a complete 1.7B model.

This establishes a bounded second-domain learning result through an added
134M expert and explicit request composition. It uses one operator and known
training facts with held-out wording and combinations. Retention preserves
previously incorrect answers too. It does not establish general assistant
quality, arbitrary routing, independent ownership or an equal-budget scaling
advantage. The expert's original 560 updates were retained without retraining;
this read-only evaluation issued no tokens and did not activate native serving.

The full evidence archive has a verified storage readback. All five temporary
GPUs, disks and their security group were retired. The retirement helper initially
required a missing administrative no-replacement record; cleanup was recovered
using the actual unchanged allocation, and that failure and recovery are recorded
in the result. The numerical result and frozen helpers were unchanged.

See the [five-owner reproduction procedure](REPRODUCE_COMPOSED_COHORT.md).
The [public model release](https://github.com/neuroshard-ai/neuroshard/releases/tag/research-cohorts-20260915) contains the new expert, training bank and
numerical bundle. All 24 new assets passed complete anonymous SHA-256 readbacks;
[publication records](../config/experiments/composed-cohort-public-artifacts.json)
bind the result, public files and each owner's download requests.
The experiment's method and original failure remain documented below.


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

Offline reuse of the already observed single-question development calls first
suggested 10/16 correct pairs. The frozen operated development then executed both
calls on five separate owners and confirmed that score, with every retention and
exit check passing. Its selection was committed before opening the final, which
reached 23/32 correct pairs from 64 actual child calls.

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
