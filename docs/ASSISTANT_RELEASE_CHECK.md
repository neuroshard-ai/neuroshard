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
