# Operated alpha

The [running alpha](OPERATED_ALPHA_RESULT.md) serves the accepted growing graph
on separately hosted AWS machines. One administrator still controls the deployment. Different machines,
availability zones, wallets and consensus keys test failures and protocol access;
they do not establish independent administration. TODO 4 remains open until its
independent-operator criterion is met. A working public deployment addresses
TODO 6; it does not establish ChatGPT-level general capability.

AWS can simulate many participants with separate wallets, provider processes,
machines and network conditions. These participants exercise admission,
concurrent use, payments, failures and recovery without requiring outside
volunteers. Independent operators may also use AWS, but must control their own
keys and infrastructure. Simulated participation is sufficient for the operated
alpha; the independent-operation requirement remains part of item 4.

The implementation uses a separate alpha genesis and matching pinned source;
[the joining guide](JOIN_ALPHA.md) identifies both. The existing 0.4.0 chain and balances are not upgraded by
these changes. No new learning or final evaluation is needed to test deployment
of the already accepted graph.

## Funded admission

The opt-in `service_admission` profile replaces the unaccepted audit-offer queue
with standing services explicitly offered by native voting owners. Each offer
binds a purpose, graph or cohort prescription, maximum stage count, expiry and
prepaid audit collateral. It is an authorization to perform verification, not a
verification result.

One signed `admit_work` reserves an actual hosted request, feature-production
assignment, training window or quality claim together with its complete audit.
It requires services covering strictly more than two thirds of the native
voting snapshot. Missing capacity or an exceeded price ceiling rejects the
entire transition. It cannot leave an unaccepted or unreserved audit budget.
Hosted customers use one durable payment operation with an absolute inclusion
deadline. A lost acknowledgement never authorizes another payment or nonce.

Full replay fees and occupancy have separate escrows. Occupancy costs a positive
number of native atoms for every occupied block, including admission; unused
occupancy and verification funding are refunded. A requester that holds a slot
without executing pays its complete occupied lifetime. Failed audited inference
releases its job and provider slots immediately. Registration and offer lifetime
are also prepaid and nonrefundable. These are measurable native costs, **not a
proof that an attacker cannot afford denial of service**: trial NEURO has no
demonstrated external price, and sponsorship must itself remain bounded.

Complete native replay quorum is the sole adjudicator for this profile. The
legacy external challenge/upload/referee path is disabled. Consequently an
outside observer cannot pause the global candidate queue through that path.
Correctness and liveness rely on the stated native voting assumption and actual
complete replay by honest auditors. This is not a cryptographic proof of neural
computation, nor independent verification when one administrator owns the quorum.
Other geneses retain their existing dispute semantics.

## Provider maintenance and recovery

Each runtime publishes native heartbeats while downloading or executing a job.
Only recently heard-from providers can acquire new assignments. The same
runtime key can occupy one rank in an assignment and only its configured number
of concurrent jobs across all offers. Renewing an offer extends its existing
identity instead of creating another capacity slot. Heartbeats demonstrate key
activity, not hardware ownership, useful computation or separate administration.

Paid provider registrations and offers renew within their declared lifetime
bounds. Runtime configuration must opt into maintenance and specify a finite
service duration. Initial registration and initial offer publication remain
explicit funded actions. Losing registration or letting an offer expire does
not trigger unbounded automatic purchases.

A local recovery controller watches native deadlines. When an assignment times
out, it checks available replacement capacity and submits an epoch-bound,
deadline-bound `recover_hosted_job`. Native settlement discovers providers again,
preserving the original request, graph, audit funding and fee ceiling. A
replacement requires an already available provider; the protocol does not create
cloud resources. A live malicious provider can still withhold computation, and
heartbeats alone do not resolve that admission problem.

Control and payment transactions retain their exact signed bytes until receipt
or a committed state proves their signed deadline has permanently elapsed. An
unknown outcome stays unknown; retirement is not a fabricated refund or receipt.

## Deployment checks

The original native preflight passed on source
`ca0752bb993afa1da11c38b2583b9062f9455782`. Five assigned CPU shards answered each
request; advertised standby owners replaced a killed coordinator and backbone
owner through the recovery daemon. Each of the three responses received three
complete numerical replays. All four validators agreed. Ledger replay reproduced
651 headers and 147 accepted transactions, with zero rejected transactions or
issued tokens, and final state
`229aa4e5ed9a9daad04a9101de76625b99b8861f54689d4e804f76318edb1314`.
Execution took 617.41 seconds; all trial processes stopped. No AWS instances or
GPUs were allocated. [Public evidence, including the CPU model and ledger](https://dwquwt9gkkeil.cloudfront.net/research/native-expert-live-20260916/objects/1161002838ecfdf912d5a6948e850f866336fbc4a3c0487c2b8b7d8056ca88d3)
and [exact source](https://dwquwt9gkkeil.cloudfront.net/research/native-expert-live-20260916/objects/575ed90fc2864b5779cc1487fd44e3a1caee39794127c24bdf7c80d5e21b182e)
were uploaded with complete checksum readback. These are synthetic-model
operability results, not another LLM quality result or the independent soak.

The current [funded deployment contract](../config/experiments/operated-alpha.json)
specifies seven GPUs for at most 60 hours, four separate ledger hosts for at most
seven days, two serving replicas on different physical hosts, and an $800
aggregate planning ceiling with a spending watch. The ledger outlives GPU
service so native expiries can refund unfinished work. Public starter credits
are finite sponsor transfers; the larger declared bootstrap stake keeps that
credit pool below a blocking minority. It still belongs to one administrator.
The [deployed result](OPERATED_ALPHA_RESULT.md) now passes the declared service
gate and publishes its access descriptor, ledger replay and numerical evidence.
The [joining guide](JOIN_ALPHA.md) covers source installation, a validating local
observer, bounded starter credits and the existing chat client. The GPU gate
stops one provider service in each recovery case so systemd cannot immediately
restart it, then requires native recovery using an already advertised replica
on another machine. It does not claim recovery of a failed physical host or
independent administration.

Before a GPU allocation, the native CPU preflight must execute actual shards,
settle complete numerical audits, recover a coordinator and a backbone owner
using the recovery controller, and reproduce the ledger from its blocks. The
source and experiment contract must be committed first. A subsequent allocation
must publish its spending ceiling, service window, topology and stop rules
before launch. Protected hosts and their existing services stay available.

Public chat retains the [visibility and storage rules](HOSTED_CHAT.md): prompts,
conversation context and settled replies are public ledger data. Provisional
text is unverified until native settlement. Native trial balances are service
accounting units; deployment does not establish a market value for them.

## Preparation race found during deployment

The first deployment on `ba0e19e` reached the two-customer warm-up but three
providers abandoned preparation while their original heartbeat acknowledgement
was unresolved. They never signed acceptance; no response was accepted at the
failure observation, and public availability was not published. The
[declared retry](../config/experiments/operated-alpha-preparation-retry.json) fixes
that control boundary: resolve the same pending transaction before another
operation, and retry only preparation that has not begun neural frames. It
preserves the original service gates and all numerical behavior. The failed
GPU allocation is retired; its separate ledger stays available for native
expiry/refunds before retirement. The aggregate $800 ceiling includes a $50
reserve for the failed deployment, including its remaining ledger lifetime.


The corrected source `944a0d4` repeated all three CPU cases in 618.26 seconds:
651 headers, 147 accepted transactions, no rejected transactions or issuance,
and all four saved states reproduced. Its final state is
`b53769d0b92e0c496c3bd6a7fce3bfbbc37235f8dcc2bae69b2b5fad9c751529`.
[Corrected preflight evidence](https://dwquwt9gkkeil.cloudfront.net/research/native-expert-live-20260916/objects/a3e96efd4785d7162fdd18eb1ccd98bbd2e32eccf1971d8f96e6cc02465bd77a)
and [exact corrected source](https://dwquwt9gkkeil.cloudfront.net/research/native-expert-live-20260916/objects/d17b896babaab5b983e3d4ec223c8493589b0c6479890e4d8fb4c26a7956dae5)
include full checksum readback. The
[first deployment's failure and ledger snapshot](https://dwquwt9gkkeil.cloudfront.net/research/native-expert-live-20260916/objects/7c292c6a5f8f43822cd00896a21d8adc09193adfeaa8f5020bf4df66624bc629)
reproduce 1,493 headers and 226 accepted transactions against all four states,
with zero issuance or accepted responses. Its
[original deployment source](https://dwquwt9gkkeil.cloudfront.net/research/native-expert-live-20260916/objects/7847a7133c595af0a6882497b0ad839a55be36e2123249b7686f74cf44281c02)
remains available. Retired GPU compute was at most $3.38; the complete failed
allocation remains covered by the declared $50 reserve.

The second deployment exposed a separate launch configuration error: persistent
systemd services omitted the image's GPU library path. The numerical guard
refused that runtime before execution. The
[committed loader repair](../config/experiments/operated-alpha-loader-repair.json)
preserves the same library selection as the existing reference launcher.
All seven physical owners passed the unchanged numerical preflight before
provider restart. Native recovery alone created new assignment epochs; original
start markers, requests, deadlines and gate timing were preserved. No model,
consensus source or genesis changed.


## CPU capacity repair

The next ordinary screen settled all six responses with complete replay, but
failed its first-visible target on multi-turn context (55.24 seconds) and the
combined question (69.90 seconds). The original 45-second limit remains in force.
File checks finished before a long queue of native acceptances. CloudWatch showed
the customer-facing T3 validator at zero CPU credits and its 20% CPU baseline;
the other validators still had credits. The one-transaction-per-block native
profile makes delayed acceptance visible to every assigned owner.

The [prospective repair](../config/experiments/operated-alpha-credit-repair.json)
enables paid CPU bursting on the four existing validators, preserving their keys,
addresses, source, genesis and ledger. To retain the $800 aggregate ceiling, the
same seven GPUs now retire after at most **66 hours**, on **September 22 at
16:27 UTC**. The ledger window remains seven days. The cost watch includes a
$67.20 upper allowance for continuous use of both vCPUs on all four validators
for all 168 hours, even though baseline credits are free. The
[AWS Linux T3 surplus rate](https://aws.amazon.com/ec2/instance-types/t3/) is
$0.05 per vCPU-hour. The complete-window forecast at repair was $729.76;
this is a conservative plan, not an invoice.

The failed gate is retained, and its in-progress request settled through native
recovery. A separate gate must repeat all ten requests and both failure cases.
An ordinary latency failure now stops the driver before lengthy fault trials.
This repair allocates no additional machines and changes no neural behavior.


CPU bursting brought the combined answer down to 45.016492 seconds; the other
five ordinary cases passed. The result remains failed rather than being rounded
into a pass. A [declared timing repair](../config/experiments/operated-alpha-commit-wait-repair.json)
reduces CometBFT's inter-block wait from 500 to 250 milliseconds across the
validators and observers. Every validator advanced through its rolling restart.
Quorum rules, application source, genesis and neural execution remain unchanged.

That run's two warm-up requests settled at heights 3,938 and 4,027 with complete
numerical audits. Its checker incorrectly read history from another validator
before that node caught up. The [receipt-view repair](../config/experiments/operated-alpha-receipt-view-repair.json)
checks subsequent payment history through the same pinned customer node that
confirmed completion. The native ledger contains exactly one settlement for
each request. The checker failure and the actual results are preserved separately.


## Registration renewal repair

The corrected checker reached all six ordinary results. Five met the frozen
latency limits; the combined request took 309.69 seconds to first text and failed.
The provider maintainer clipped an offer duration to the registration end at the
query height. Inclusion in any later block put the offer beyond that end, causing
repeated rejection and delaying model acceptance. All 27 requests on this second
candidate settled after complete audits; its failed gate remains preserved.

The [next repair](../config/experiments/operated-alpha-renewal-repair.json) renews
registration before a complete configured offer, with 64 blocks of inclusion
headroom. Native-transition checks cover immediate and latest allowed inclusion.
Changed source requires a new pinned genesis; no running chain silently changes
its source commitment. Both failed candidates are included in a $100 reserve.
The next GPU window is bounded to 60 hours, with paid CPU bursting declared at
ledger launch, under the same $800 aggregate ceiling and unchanged service gates.


The corrected source `f593b3c` passed ordinary CPU serving and both automatic
process replacements, with nine complete numerical replays. Application replay
matched 620 headers, 143 accepted transactions, zero rejected transactions and
all four stored states, with zero issuance. [The complete CPU evidence](https://dwquwt9gkkeil.cloudfront.net/research/native-expert-live-20260916/objects/4c3af5187d8e1bd215e5baa77d2796d0854ef1f08372754db70347eafdcfb4f1)
also records a launcher configuration deviation: its initial CPU ceiling was
300%, corrected to the declared 150% after the ordinary case. This establishes
protocol behavior, not conformity to that CPU resource envelope or a performance
improvement. The corrected source passed all five CI checks.

The [second candidate's complete failed gates and ledger](https://dwquwt9gkkeil.cloudfront.net/research/native-expert-live-20260916/objects/4d8ea87af7ab6470c63b916973ff831fb58c123ec9c24a3262850e168c9cfbf9)
reproduce 6,536 headers and 1,288 transactions against all four states. All 27
requests settled and future audit services closed. Its only starter grant went
to the operator's onboarding wallet. Private signing state was copied after
stopping consensus; all eleven instances and their volumes are retired. GPU
compute was at most $11.21, with complete costs covered by the failure reserve.

A subsequent allocation stopped before installation or model execution because
its bootstrap invoked `git archive` from an ignored exported source directory.
The [checkout repair](../config/experiments/operated-alpha-checkout-repair.json)
uses a detached Git worktree at the identical tested commit and checks the full
archive command before allocation. It reuses the passing CPU evidence, changes
no model or protocol bytes, and keeps the 60-hour window, $800 total ceiling and
all service gates. This failed setup also remains within the $100 prior-attempt
reserve.

The fourth setup reached provider funding but failed during its first offer.
Its launcher had discarded remote stderr, so the exact cause is unknown.
The [bootstrap repair](../config/experiments/operated-alpha-bootstrap-repair.json)
closes an observed readiness gap: each provider must confirm its funding
transaction on its own synchronized, pinned full node before registering.
Remote startup errors are now preserved privately before cleanup. All eleven
R4 hosts and their volumes are retired; no public inference was admitted. The
protocol and neural source, passed CPU preflight and service gates are unchanged.
