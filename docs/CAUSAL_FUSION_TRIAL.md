# Causal fusion of owned model shards

**Result: rejected on development.** Both connections completed 512 updates and
exactly replayed their final 16 updates from saved GPU optimizer state. The
learned-source connection answered 0/32 specialist or mixed questions correctly;
the source-information ablation answered 1/32. Structured answers remained 2/8.
General-response loss increased by 0.01309 nats on average, with an upper bound
of 0.02840 against the permitted 0.02. The final set stayed unopened.

The [complete result](../config/experiments/causal-fusion-results.json) records
the initial hostname-agreement failure, corrected execution, limits and resource
retirement. [Public evidence](https://github.com/neuroshard-ai/neuroshard/releases/tag/research-causal-fusion-20260916)
includes frozen inputs, source, all saved optimizer checkpoints and generated
responses. This small connection did not demonstrate useful knowledge transfer.

The next [frozen source-interface diagnostic](../config/experiments/fusion-source-interface-probe.json)
isolates input framing and output-head compatibility on existing development
questions. It deliberately names the source under test; it does not demonstrate
automatic routing or count as a new learning cohort.

The question-rewriting experiments did not reliably compose answers: the latest
scoped routing regression completed 7 of 9 answers, and reviewing the planner's
own output worsened its result. Those failures are retained in
`config/experiments/conversation-routing-results.json`.

This experiment trains a connection that reads causal hidden activations from
the existing frozen models. All model paths see the original conversation and
the same generated tokens. The receiver produces one autoregressive answer;
it does not parse subquestions, look up answer labels or stitch generated text.
The trained backbone prefix executes once and supplies both learned tails.
The preserved assistant and trained backbone each remain split across three
owners, with the two learned tails on their own owners.

Cross-attention model composition is motivated by
[CALM](https://arxiv.org/abs/2401.02412) and
[BTS](https://arxiv.org/abs/2502.00075). This experiment's low-rank connection,
initialization, small corpus and ownership arrangement are separate design
choices; the papers do not establish that this particular method will work.

## Frozen decision

The complete prescription is
[`causal-fusion-trial.json`](../config/experiments/causal-fusion-trial.json).
It fixes source models, tokenizer, input hashes, batches, optimizer schedule,
answer scoring, retention bounds and resource limits before training.

- Train 1,048,576 connection parameters over 1,024 complete conversations for
  512 updates. All existing language-model weights and output-head weights stay
  frozen. General conversation examples also constrain divergence from the
  preserved assistant.
- Directory people and protocol topics are disjoint between connection fitting,
  development and final evaluation. The frozen specialist shards previously
  learned these facts. This tests transfer of existing distributed knowledge,
  rather than expecting unavailable facts to be inferred.
- Train an identical connection with unchanged parent activations substituted
  for learned tail activations. This is an information ablation; it does not
  establish the best achievable use of an equal total compute budget without
  growth. Record its extra transport separately.
- Check exact initialization behavior on the real GPU paths and replay the last
  16 updates from persisted Adam state. Evaluate only the terminal connections.
- Evaluate 48 development conversations once. Failure rejects this prescription
  and leaves the 128 final conversations unopened. A pass opens that final once.
  Paired bootstrap intervals resample connected knowledge groups; greedy answer
  gains and general-response retention are separate required checks.

The controlled five-host allocation has an absolute two-hour lifetime and a
$30 planning cap. Preserve both successes and failures, checkpoints, commitments,
generated answers, timings and traffic before terminating the allocation.

## Reproduction and limits

`scripts/prepare_fusion_corpus.py` reconstructs the committed selection from the
original source conversations and local tokenizer files. Exact prompt duplicates
are excluded across roles. Complete examples longer than 384 tokens and tasks
explicitly requiring a scratch-work block before JSON are excluded during
preparation. No training or final scores informed those preparation rules.

`scripts/run_fusion_trial.py` executes the frozen prescription against installed
owned tensor manifests. Only the receiver stores the causal activation bank.
The small connection is replicated to owners for generation; no owner acquires
the full pretrained backbone. Frozen source weights can reproduce the activation
bank, but commitments alone are not proofs of honest source execution.

All model sources execute every generated token. Sparse activation, cheap
independent verification, broader assistant quality and an optimal resource
comparison remain unproven. This path also requires separate native admission
before it can replace a settled serving graph. A passing trial supplies evidence
for item 1 of [the six-item checklist](../TODO.md); it does not complete that item
or imply independent operation or a public chat release.
