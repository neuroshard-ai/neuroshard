# Input-only expert routing

The current serving graph selects its two experts using fixed phrases. The next
router uses the existing model's frozen input embeddings and small fitted
classifiers. The user supplies ordinary text; task identities and reference
answers are absent from the inference interface. Adding a route does not require
another complete language model on the coordinator.

The development source can include system messages and multiple turns. Fitting
and evaluation join only its user turns in order. All assistant text, including
the reference response, stays outside the feature extractor.

The embedding owner quantizes the committed input table once, pools only user
tokens and normalizes the result with integer arithmetic. Deterministic spherical
clustering fits several prototypes per route, including the ordinary parent.
An integer averaged margin perceptron learns separating directions from the
same training observations; prototypes remain the distance guard. Low-margin
and distant requests fall back to that parent. The model binds the
embedding/feature recipe, tokenizer and training observations. This makes the
router reproducible; it does not prove that its routing decisions are useful.

The method is motivated by work on prompt-dependent routing in
[RouteLLM](https://arxiv.org/abs/2406.18665), and by the frozen-embedding approach
reported in [Latent-LoRA](https://arxiv.org/abs/2607.23837). This implementation
uses integer prototypes and a margin perceptron; neither is those papers'
specific router.
Their reported results do not establish performance for NeuroShard's experts.

The [frozen development screen](../config/experiments/embedding-router.json)
uses previously exposed development data, with disjoint fitting and evaluation
records. All variants of a directory entity stay on one side. It measures the
original questions and versions with the old fixed routing phrases changed.
The gate requires at least 95% accuracy for each expert in both forms and no
change to the parent route for the selected ordinary questions. It runs on CPU
and cannot promote a graph, issue tokens or count as an independent final test.

Run `scripts/experiment_embedding_router.py prepare` with the existing input
directory, commit the generated selection, then use `run` with the pinned seed,
embedding file and a fresh output directory. The run rejects changed source or
data. Results must be retained before changing the method.

Task 1 additionally requires response quality, retention across three admitted
cohorts, general composition and a resource-matched control. A passing classifier
alone does not complete it. Provider assignment, native admission and chat must
bind the same accepted graph and router version.

The prototype-only [development result](../config/experiments/embedding-router-prototype-result.json)
failed: all 706 original prompts selected the expected route, but only 60 of
80 reworded protocol questions did. The directory variants passed 480/480.
The discriminative follow-up uses identical fitting/evaluation identities and
gates; no reworded evaluation text is added to fitting. Neither result is an
independent quality claim.
