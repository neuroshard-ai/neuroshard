# Owned expert conversation interfaces

The probability mixture recovered 9/16 individual specialist answers, compared
with 0/16 from the preserved assistant. It still answered 0/16 combined questions
and exceeded the general-retention bound. All five owners recorded the same
rejection; the final remains unopened. Complete outputs and optimizer checkpoints
are in the [public evidence](https://github.com/neuroshard-ai/neuroshard/releases/download/research-causal-fusion-20260916/probability-mixture-evidence.tar.gz).

This next experiment targets the demonstrated interface problem. A source can
remember a fact yet fail to retrieve it from a combined conversation. Small
rank-8 adapters attach to the existing expert projections, while every original
backbone and expert weight stays frozen. The output owner jointly trains its
causal probability gate. The prefix is computed by its existing shard owners;
each expert owner trains its own adapters using returned hidden-state gradients.
No owner constructs the whole backbone.

The [prospective plan](../config/experiments/expert-interface-trial.json) freezes
512 updates for each arm, the existing data splits and acceptance gates, AdamW
settings and a two-hour/$25 allocation limit. Both arms receive identical gate
training, including stronger general retention and training-only provenance
supervision. The candidate additionally trains the expert interfaces. This
comparison measures the interfaces' contribution; it is not the required
matched-total-resource comparison against a model without growth.

Auxiliary source loss applies only to the source's own answer spans. The joint
objective trains the complete response, including ordering and delimiters.
Inference receives the original conversation and generates one shared token
stream. It receives no reference answer, task label or source assignment.

The final 16 updates must reproduce after restoring all three trainable owners'
weights and Adam state. The terminal development score must pass every gate
before the final is evaluated once. There is no intermediate checkpoint
selection. Development has informed this method; its scores are diagnostic,
and the final is the remaining unscored evaluation.

Adapters have separate source-bound commitments. Adapted generation binds all
of them in the request; feature banks bind the interfaces that produced their
activations. Scoped hooks preserve the original parameter inventory and are
removed after each operation. A rejected candidate leaves the frozen expert
paths available. These execution bindings do not yet implement native admission
or an independent audit for the new method.

This is still a controlled experiment under one administrator. It cannot complete
the independent-operator, repeated-cohort or public-chat checklist items.

The terminal run was rejected: 13/16 single-fact answers, 2/16 combined answers,
7/8 structured answers, and a passing general-retention upper bound of +0.01412.
Both optimizer restarts passed. The [result](../config/experiments/expert-interface-results.json)
and [bounded continuation](EXPERT_INTERFACE_CONTINUATION.md) preserve the failed
gate and leave the final unopened.
