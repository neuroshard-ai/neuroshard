# Audited sharded serving results

The [second attempt](GRANITE_SHARD_AUDIT.md#amendment-for-the-second-attempt)
**passed all six declared checks** (commit `ee323e4`). On the real 3B assistant,
light auditors that each held one owner's shard verified that owner's serving
work by replaying its signed log. When owner 1 flipped one bit in one message,
its auditor named that exact message, and a fresh verifier accepted the fraud
proof. Owner 2 received the corrupted tensor but computed honestly on it, and it
audited clean. Evidence: [report](../config/experiments/granite-shard-audit2-report.json).
The [first attempt](../config/experiments/granite-shard-audit-report.json) failed
at fetch and stays failed.

## Checks

| Check | Result |
|---|---|
| Honest pass matches the single-host development result | 18/24 correct, no token mismatch, p95 102.4 s |
| Both honest logs replay exactly and are signed by the published keys | 8830 of 8830 forward passes each |
| Auditor 1 names the declared fault | Forward pass 200 (log entry 203), as declared |
| A fresh verifier accepts the fraud proof | Accepted: owner 1's signed log plus 201 inputs |
| Owner 2's log from the faulty pass audits clean | 8830 of 8830, no blame |
| Each auditor fetched exactly its owner's bytes | 2.20 of 6.8 GB each |

## Measurements

- **Audit cost.** A complete replay took 485 s for owner 1 and 499 s for owner 2,
  against 568 s and 558 s of owner busy time (0.85× and 0.89×). Auditing every
  message cost slightly less compute than serving it.
- **Fault detection.** Replay stops at the first wrong output: auditor 1 found
  the fault after 11.5 s. The fresh verifier process loaded its shard in 5.8 s,
  then replayed the proof's 201 inputs; that check was not timed separately.
- **Logs.** 8902 entries per owner per pass (8830 forward passes plus commands),
  about 242 MB compressed with the retained inputs.

## Evidence gap

Only owner 0's evidence was archived. Each of the other four hosts held more
than the 96 MiB per-host evidence cap, mostly the two signed logs, so the
controller refused to copy them and retired the hosts as designed. Every phase
result the checks use was read from the hosts during the run and is in the
[result file](../config/experiments/granite-shard-audit2-result.json). The raw
logs and the fraud-proof files were not kept, so the proof cannot be re-verified
offline. Future audited runs should archive the proof and log digests within
the cap, or declare a larger cap for logs.

## Limits

One operator ran every host. Bonds, challenge windows and slashing are not yet
on chain, and independent operators belong to A5. Owner 0 was not audited,
because in the design it runs on the user's own machine. No checklist credit.

## Resources

Five r7i.4xlarge allocations, all retired with no instance or volume
remaining: 24,459 instance-seconds, $7.19. The first attempt cost $0.32.
