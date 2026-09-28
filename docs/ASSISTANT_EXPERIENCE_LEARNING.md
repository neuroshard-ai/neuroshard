# Verified-experience learning contract

September 27, 2026. New method for A2; it supersedes the training and gate
sections of the [workspace learning contract](ASSISTANT_WORKFLOW_LEARNING.md)
and reuses its data partitions, tools, policy and arms. The
[JSON plan](../config/experiments/assistant-experience-learning.json) is
authoritative for numerical settings.

## Why a different method

Earlier module experiments supervised the model on fixed target text, then relied
on a separate selector to protect old behavior. Several lost most protected
answers: 11 of 15 in [staged recovery](STAGED_ANSWERING_RECOVERY_RESULTS.md) and
8 of 8 in the [block-expert measurement](BLOCK_EXPERT_MEASURE_RESULTS.md).
Forgetting tracks how far training moves the model's output distribution on the
new task ([RL's Razor](https://arxiv.org/abs/2509.04259)). Imitating arbitrary
target text can move it far; training on the model's own successful outputs
moves it little.

The workspace gives something those experiments lacked: a deterministic check of
whether a whole conversation succeeded. This contract uses that check as the
source of training data instead of authored answers.

## Method

1. **Verified self-experience.** The parent samples eight complete conversations
   per training case in the sandbox. The frozen scorer replays and scores each
   one. Fully successful conversations, and the leading successful rounds of
   partial ones, become training data. Rejected tool calls stay in context
   without loss.
2. **Coached practice.** Where the parent never succeeds alone, it retries with a
   fixed procedure card: how to pick the latest approved revision, use tools for
   arithmetic and dates, cite what it read, and apply corrections to its last
   draft. The card is the same for every case and contains no values. Verified
   successes are stored with the card removed, so the trained module learns the
   procedure itself (context distillation). The card never appears at evaluation.
3. **Near-policy check.** Each accepted trajectory's likelihood under the parent,
   without the card, is recorded. Coached trajectories less likely than every
   natural success are dropped. This bounds distribution shift and, in a network,
   rejects injected text the model would not have produced.
4. **Preservation replay.** The parent's own greedy answers on 256 generated
   conversation, instruction and tool prompts, disjoint from the anchors, are
   mixed into every update.
5. **Selection from success rates.** On each integration episode, the parent and
   trained arm each run once greedily and four times sampled. The gate targets the
   arm only where its success rate is higher, weighted by the difference; ties
   carry no weight. A single greedy run per episode is too noisy for 64 labels.

The update and addition arms, optimizer, 128-step schedule and episode-level
routing are unchanged from the previous contract. Each accepted conversation
has equal loss weight.

## Gates

Development requires at least 18/24 complete episodes, net +4 over the parent,
no lost parent or anchor success, and at most one fewer success than the update
control. Confirmation opens once, after a development pass: at least 77/96, eight
per family, net +10 over the parent, no lost parent success, a positive
family-bootstrapped lower bound over the parent, and a lower bound over the
update control of at least −5 percentage points. Latency limits are unchanged; a
routed system's episode latency includes the parent forward pass that selects its
model, timed per episode.

The previous contract required the addition to beat the equal-data update by +4
and +10. Both arms see the same data and both keep the parent for fallback, so
that margin tests noise rather than method. The addition instead has to match the
update while training 60 times fewer parameters (1.05M versus 62.9M), which is
what makes it cheap to distribute and roll back. Whether separate modules beat
repeated shared-weight updates is an A3 question: cumulative retention across
cohorts under a declared budget. This amendment precedes any training.

## Conditional second round: verified decision preferences

Declared before any round-1 development outcome was read; it runs only if the
round-1 development gate fails. On integration episodes the round-1 arms complete
95–98% of cases whose latest approved revision is listed first, but only 35–36% of
cases whose older revision is listed first. The pinned round-1 rollouts already
contain both outcomes of that decision on the same case.

A pair shares every message up to the first `read_document` call. The chosen
continuation opens the latest approved revision in a rollout whose first round
passed; the rejected one opens an older approved revision in a rollout whose first
round failed. Both arms continue from their round-1 checkpoints for 64 steps on the
same experience, replay and pairs, adding a DPO term on that decision message with
the round-1 arm as reference. Gates are refitted on integration episodes, and the
same development and confirmation gates apply. The development split is reused, so
confirmation remains the decisive fresh test.

## Serving runtime amendment: prefix cache

Declared after the [round-2 development result](ASSISTANT_EXPERIENCE_DEVELOPMENT_ROUND2_RESULTS.md)
failed only on p95 latency, and before any cached run. Prefill was 49% of generation
time: compound workflows re-processed about 17,000 prompt tokens across twelve
generations. The routed systems now reuse the key/value cache of the longest
unchanged token prefix between generations of one episode. Every request is still
rendered and tokenized in full, at least one token is recomputed, and each episode
starts empty. Arms, gates and every threshold are unchanged, and the parent control
remains the canonical recompute baseline. The complete development gate must pass
under this runtime before confirmation opens.

## Third round: verified divergence preferences

Declared after the [cached development result](ASSISTANT_EXPERIENCE_DEVELOPMENT_CACHED_RESULTS.md)
and before any round-3 rollout. Round 2 fixed most version choices; the remaining
brittle behavior, follow-up date arithmetic, was never targeted. The round-2
addition samples six fresh rollouts per training case, each rescored. For every
success and failure of the same case, the first message where they differ, after
identical earlier messages, becomes a pair: the success's tool call is chosen, and
the failure's message is rejected only if its round failed. At most four pairs per
case. Both round-2 arms continue for 64 steps on the round-1 experience and replay
plus these pairs, with DPO against the round-2 arm. Gates are refitted on
integration; the development gate under prefix-cache serving must pass before
confirmation opens.

## Second attempt: goal-guided repairs and fresh confirmation

Declared on September 28, 2026 after the [confirmation](ASSISTANT_EXPERIENCE_CONFIRMATION_RESULTS.md)
failed only the update-parity bound, and before any round-4 rollout or any look at
new evaluation data. That confirmation stays failed, and its 96 episodes are now
spent.

On confirmation, the addition's difference failures read approved revision 2 and
the draft instead of revision 1. Its latest follow-up failures re-read revision 2
when asked for revision 1. Round-2 preferences had over-generalized into avoiding
older revisions. Round 4 targets that error without looking at evaluation goals.
The round-3 addition samples four rollouts per training case. In each failed
rollout, the first read that opens a document outside its round's training goal
is replaced by a goal source not yet read in that round. Earlier generations are
replayed exactly, and the same model then continues by sampling. Only repaired
rollouts that pass the frozen scorer in every round are kept. The repaired message
is chosen and the original rejected (at most four pairs per case), and verified
repaired rollouts (at most two per case) join the round-1 experience. Both round-3
arms continue for 64 steps; gates are refitted on integration.

Development then runs under prefix-cache serving with an A1 served-system check on
the version that would be served:

- It must pass at least 6 of the 8 primitive development episodes, with at least
  one from each primitive family.
- The original anchor gate must hold, with every protected anchor routed to the
  parent.
- Its p95 latency must be at most 180 seconds, including selection time.
- Two development episodes are replayed in fresh processes, and each must match
  its original episode exactly.

If development and the A1 check pass, a fresh confirmation opens once. It has 192
episodes (24 per family, from a new generator seed, with no overlap in project or
ID). All three systems, the parent included, run under the prefix cache. It
needs:

- At least 154 successes, and at least 16 in each family.
- At least 20 more successes than the parent, with no parent success lost.
- Family-cluster bootstrap lower bounds (10,000 resamples) above 0 against the
  parent and at least -0.05 against the update.
- p95 latency of at most 180 seconds, and at most twice the update arm's p95.

The [contract](../config/experiments/assistant-experience-learning.json) records
these rules under `goal_guided_repairs`, `a1_served` and `confirmation2_gate`. The
budget ceiling for this attempt is $1,000.

## Role in the network

Rollouts are the parallel contributor workload. Anyone can replay a submitted
transcript against the sandbox and rescore it without trusting its producer or
rerunning generation; one teacher-forced pass under the parent checks that it is
near-policy. More participants mean more verified experience per hour. One compute
group trains the module (about 2 MiB in BF16); replicas serve it. A3 repeats the
cycle per cohort. Settlement, payment and adversarial auditing are outside this
experiment.

## Preconditions and limits

The [canonical re-baseline](ASSISTANT_WORKFLOW_CANONICAL_RESULTS.md) completed
at 9/24 with conforming tokenization and development headroom, but failed primitive
qualification on version choice and the call budget. On September 27, 2026 the
user authorized collection and training anyway. A1 stays open, and this comparison
cannot close it: a pass shows learned capability over a parent that is not yet a
qualified workspace assistant.

Collection and training run once on one L40S host under the
[resource contract](../config/experiments/assistant-experience-resources.json)
(six-hour expiry, $15 allowance) and the pinned
[execution inventory](../config/experiments/assistant-experience-execution.json).
Development evaluation reuses the CPU runtime of the parent baseline; confirmation
opens only after a development pass.
Precedents for filtered self-training ([STaR](https://arxiv.org/abs/2203.14465),
[ReST-EM](https://arxiv.org/abs/2312.06585)) and
[context distillation](https://arxiv.org/abs/2209.15189) do not establish that
this combination passes these gates.
