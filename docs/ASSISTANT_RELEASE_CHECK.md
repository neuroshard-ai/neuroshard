# Assistant network release qualification

The next network version is the accepted A3 assistant: Granite 4.1 3B, the
separable U1 update, scheduling unit L2, drafting upgrade L3, the learned router,
canonical tokenizer, tool policies and bounded decoding. It is an experimental
workspace assistant. Independent operation and general assistant quality remain
open; the existing small-model chain and PyPI 0.4.1 are separate releases.

Before promotion, reproduce all **432 already-opened confirmation episodes**:
192 drafting, 192 scheduling and 48 cross-capability conversations. Compare every
input/output token, route, tool result, workspace snapshot, completion and failure
with the recorded accepted system. Only timing and resource telemetry are omitted
from the equality check. This is deployment equivalence, not new learning evidence.

The [execution contract](../config/experiments/assistant-network-release-check.json)
pins sources, records, modules, policies and runtime. All bytes must be committed
before execution. A separate temporary chain pays two owners on different hosts;
the client hosts stage 0 on a third host. No execution worker holds the complete
backbone. The first six episodes exercise every route before the remaining work.
The first mismatch or execution error stops the attempt. No training, GPU, new
evaluation set or tuning is authorized by this contract.

The ceiling is 16 hours and $30 of instance time across three m7i.2xlarge hosts,
including the two existing owners. Only the additional client instance is
temporary. Retire it and the shadow services after evidence copy; never retire
the protected ledger or owner instances. Save partial evidence on timeout. A
repair requires a committed execution amendment before another attempt.

After an exact pass, publish the report and immutable module assets. Configure
validators for both current and candidate replay; enable automatic owner-log
audits and proof delivery. Drain paid jobs, submit an authorized native
`model_promote` transaction, switch owners and the default client descriptor,
then check paid chat, audits, settlement, refunds and restart recovery. Preserve
keys, balances, bonds and chain history. Keep A2 available for rollback.

PR #85 is the release integration. Merge only when checks on its final commit
pass. A merge does not make A5 complete: four independently administered
operators are still required.

## Attempt 1 and the amendment for attempt 2

Attempt 1 ran from commit `80f65c8`, setting up from 18:07 UTC on October 9 and
serving from 19:08 UTC. The supervisor retired it at 10:06 UTC on October 10, on
its 16-hour budget. Each shadow owner holds 355 signed job logs, so 355 of the
432 conversations were served, at about 2.53 minutes each; no mismatch was
reported at the last progress reading (333). The full workload therefore needs
about 18.2 hours of serving. Two supervision faults lost the per-conversation
evidence:

- The client unit's `RuntimeMaxSec` (15 h 45 min) was shorter than the worker
  budget, so systemd killed the worker before it could write `result.json`.
- The supervisor stopped each service before copying its evidence, and the stop
  command timed out on the client.

Attempt 1 therefore neither passes nor fails; it gives no evidence either way.
The committed amendment in the [execution contract](../config/experiments/assistant-network-release-check.json)
(`attempt: 2`) raises the worker budget to 21 hours and the cumulative ceiling
to $50 across both attempts. It also makes the client stop cleanly between
conversations on SIGTERM and always record a result. The supervisor copies
evidence before stopping anything, and the client unit outlives the worker
budget. The workload, pass rule, model root, runtime, modules, policies, ports,
positions and price are unchanged. Attempt 2 uses a fresh shadow chain and a
fresh client; it does not resume attempt 1.
