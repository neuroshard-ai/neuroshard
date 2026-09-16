# Fixed blocks for affordable checked inference

The preceding GPU stream passed all eight complete-response checks, both
first-token forgeries and both alternate draft-chunk comparisons. It corrected
two real disagreements between cached decoding and fixed-context checking.
However, its 128-token responses took 28.09–29.26 seconds and sent 1.46–1.61 GB
of tensor traffic. Each complete-response audit took about 0.65 seconds and
83.9 MB. The [complete result](../config/experiments/checked-streaming-results.json)
includes both allocations; estimated combined compute was $2.25, with storage
and transfer separate. All hosts, volumes and temporary network resources are
retired. This execution pass did not change the rejected model-quality result.

`sharded.blocked_inference` addresses the repeated full-context cost. Every owner
keeps attention state for previously verified blocks. A target call evaluates
exactly eight new positions at a boundary aligned from the start of the input.
The preserved assistant proposes tokens through its own three owners. Target
predictions accept the matching prefix and supply the first correction. No
proposal can determine an accepted token without target execution.

If a proposal diverges, the entire speculative cache block is removed. The next
call starts from the same verified boundary with the same query width and a
longer verified input prefix. Complete verified blocks remain cached. Each call
emits at least one token, so target calls are bounded by the output allowance,
in addition to prompt prefill. A separate verifier reconstructs every cache from
the complete reported input; it never accepts a provider's cache as evidence.

Drafting and target checking follow the general approach of
[speculative sampling](https://arxiv.org/abs/2302.01318). Our greedy block
prescription is distinct: block size and alignment define the floating-point
program. Equivalence to ordinary cached decoding or the earlier full-context
checker is not assumed. Changing draft strategy must preserve output; changing
the target block size requires a different numerical profile.

The five-process CPU check reproduced output using both assistant drafts and
EOS-only proposals, verified the complete response, and rejected a forged first
token. It also checked exact cache tensors after rollback and exact hidden
states after continuation compared with a fresh cache. The check passed in
17.65 seconds. These are CPU results, not GPU performance evidence.

The [frozen GPU trial](../config/experiments/blocked-streaming-trial.json) reuses
the eight exposed execution workloads and the exact same gate and expert
interface weights. It records a separate eight-token warm-up. The three
preselected long responses must each contain at least 64 tokens, then satisfy:

- At most 0.12 seconds and 1 MiB of tensor traffic per generated token.
- At most 3 seconds and 32 MiB of tensor traffic for complete fresh replay.
- Every measured request emits its first checked output within 2.5 seconds.

All eight responses must verify. Both first-token forgeries must be rejected,
and two EOS-only proposal runs must produce identical canonical output. The
allocation is limited to one hour and a $10 planning cap. No training or model
selection occurs. The model remains a rejected quality candidate, every target
block executes all installed sources, and native acceptance, complete service
economics, public concurrency and independent operation remain separate work.
