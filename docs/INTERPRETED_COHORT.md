# Learn a second expert while the established model serves

This experiment adds a fifth neural worker to the preserved-interpreter graph.
The new worker learns a separately owned two-layer documentation expert. The
three parent owners and the existing expert continue answering earlier
questions, including the original model's question interpretation.

**Status: the first graph passed its final; the second-cohort GPU run is being
retried after an isolated controller I/O failure.** The earlier branch-cohort proposal remains
deferred because its different prerequisite graph failed. This proposal has a
new plan and identities; it does not change that failed result.

The [plan](../config/experiments/interpreted-cohort.json) requires the actual
published final for graph
`2a81df24652ac4a434ad9ffb3eec09e6038801104645869d33ee35c6a4a2826d`.
Preparation binds the complete result bytes, verified archive, numerical
sources, original interpreter assets, and every earlier answer to retain. A
development-only pass, another graph's pass, missing archive verification or
an unfinished allocation cannot authorize training.

The first expert keeps its four-owner process group. The second uses a separate
group containing the same three parent owners and the fifth worker. The new
owner receives only its tail and a frozen output-head replica. After the parent
owners produce its 28 fixed training batches, the new owner performs 560 local
updates while the established paths serve requests. No participant holds a
whole 1.7B model. The graph has 3,691,204,608 stored parameters; this measures
added capability with additional capacity, not an equal-budget advantage.

The interpreter, parent and first expert remain fixed, with their real optimizer
ages preserved. The new tail starts from the parent weights with fresh Adam
state. Its loss trains only the new task. Earlier answers are protected by
independent fixed paths and checked directly. An added rule follows existing
rules, preserving their precedence. These are explicit domain choices, not a
general learned router.

The new training material contains 64 source-anchored facts about the historical
NeuroShard 0.4.0 interface. The [question specification](../config/experiments/branch-cohort-questions.json)
separates training wording, development wording, final wording and two-fact
combinations. Facts and answers are supervision; they never enter inference as
a lookup table. Responses use ordinary greedy generation over the unchanged
tokenizer vocabulary.

Only terminal update 560 may qualify. It must answer at least 75% of new
single-fact questions and 50% of composed questions, with a fact-cluster gain
lower bound above 0.10. Every retained answer and conversation loss must remain
exact. Both established paths must answer after the first new update and before
the last. After the controller observes the new expert process exit, both must
answer again. Complete development evidence is archived and selection committed
before the independent final opens.

The allocation is bounded to five GPUs, six hours and a $100 planning cap, and
starts only after the preceding allocation has been retired. The trial issues
no tokens or serving promotion. Native graph settlement, arbitrary peer
admission and general assistant quality require their own execution evidence.

## First GPU attempt and control-file correction

The [first attempt](../config/experiments/interpreted-cohort-attempt-0.json)
completed 43 new-tail updates before a worker observed an empty control file.
The controller used `cat > destination`, which exposes a file before its full
JSON content arrives. The worker's JSON reader failed and its peers then lost
their connection. Development quality and the independent final were unopened.
The step-zero checkpoint, feature bank, complete source and observations were
archived with full readback. All five instances, disks and the study security
group were removed; compute was bounded at $1.48.

The corrected controller validates each signal and publishes it with an atomic
temporary-file replacement. It reads live service logs only through complete
newline-terminated records. A regression check reproduced the original race,
verified the actual replacement command, interrupted writes, duplicate and
conflicting signals, and partial service records. Its decision checker still
accepts the actual five-process CPU evidence and rejects altered answer tokens.
The retry retains the exact prepared inputs, neural source, training recipe,
unopened final questions, original overall deadline and $100 planning cap.

Every operator must publish complete control JSON atomically. With the pinned
repository on `PYTHONPATH`, use its existing durable helper:

```python
from neuroshard.evolution.reference_data import save
save(home / 'new-learner-started.json', receipt)
```

The same requirement applies to the later service and process-exit receipts.
Only write a receipt after observing the event it describes.

## Completed training attempt and failed development

The retry finished all 560 prescribed updates while the parent and first expert served earlier requests. Single-fact development answers improved from 7/64 to 49/64, but paired answers reached only 2/16 and failed the unchanged 50% gate. Retention then stopped at question 113: configuration serialization had reordered the interpreter examples from `name, field` to `field, name`, changing the prompt. The first 112 answers matched; the remaining retention and departure checks were not completed. No final was opened.

The 0/280/560 checkpoints and feature bank were preserved with full readbacks, and all five temporary hosts, disks and their security group were retired. The terminal checkpoint is `698d9ae2ba3a89de5b1bf19ec42e6680fe2d408a96f92efa84b383ac5d95b530`. See [the complete failed result](../config/experiments/interpreted-cohort-results.json). A separate composition experiment must bind prompt token identity and execute both constituent neural calls; it cannot change this outcome.
