# Learn a second expert while the established model serves

This experiment adds a fifth neural worker to the preserved-interpreter graph.
The new worker learns a separately owned two-layer documentation expert. The
three parent owners and the existing expert continue answering earlier
questions, including the original model's question interpretation.

**Status: implementation and CPU integration tests; GPU training is gated by
the first graph's completed final.** The earlier branch-cohort proposal remains
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
