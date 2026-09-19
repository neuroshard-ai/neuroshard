# Finite funding for the learning and serving demonstration

The [frozen sponsorship](../config/experiments/finite-learning-service-sponsorship.json)
covers the complete bounded research deployment with a **$1,000 aggregate cap**.
It includes rejected attempts, training, all full numerical audits, serving,
idle time, storage, transfer, public IPv4 and the existing controllers. It does
not assign a cash value to NEURO or assume outside customer revenue.

This measures affordability under finite operator sponsorship. It does not
establish perpetual economic sustainability, independent verification or a
competitive provider market. Earlier experiment budgets and failed results
remain unchanged; a later aggregate budget cannot turn an earlier failure into
a pass. No new learning cohort or indefinite public serving obligation is
authorized by this experiment file.

## Native payment and verification

Every hosted request prepays its maximum execution, provider and complete audit
cost before work starts. The audit reservation binds the customer, graph and
conversation. Final settlement conserves these escrows and refunds unused
capacity. Inference issues no new tokens. Training can issue only the prescribed
reward for a previously unsettled, verified update; quality approval remains a
separate decision. Rewards from different research geneses are not one ledger.

The operated profile uses [native replay quorums](NATIVE_SHARD_REPLAY.md), not
sponsor-selected attestations. A verdict requires strictly more than two thirds
of snapshotted native voting weight. Each honest signer replays the complete
obligation. Honest negative reports are paid without posting a separate fraud
dispute bond; absent reports cannot authorize issuance. This assumes less than
one third Byzantine voting weight and the committed numerical environment.
Threshold collusion can accept forged work. Our different audit keys and fresh
executions are all administered by one operator, so they do not demonstrate the
independence assumption.

Full replay remains a real replicated computation cost. The accounts below
include it already; charging it again as a separate GPU allocation would double
count it. Immutable inputs and outputs let an audit start at its committed
checkpoint instead of replaying the model's entire training history.

## Accounting scope

Three distinct seven-GPU allocations cover the accepted continual-learning
campaign and its rejected/recovery work. Reused study directories do not create
new allocation charges. Raw five-minute EC2 `NetworkIn` and `NetworkOut` samples
cover their complete allocated lifetimes. Provisioning is priced through
confirmed retirement, including time an instance may already have been stopped.

| Learning allocation | Conservative cost before shared retention/controllers |
| --- | ---: |
| Conversation campaign, including rejected trials | $311.09 |
| Admission recovery | $103.04 |
| Audit/storage continuation | $148.68 |
| Total | $562.82 |

These are price-based estimates, not invoices. For example, the first allocation's
complete conservative estimate exceeds its original $250 campaign cap. The
continuation's $148.68 subtotal leaves little of its original $150 cap before
shared retention. Neither earlier cap is retroactively declared a complete-cost
pass. Billing API access is unavailable, so the report does not claim these
deliberately conservative estimates equal charges actually billed.

Provider attempts separately measure every host's non-loopback traffic counters.
Each standalone result retains its original reserve and cost gate. Aggregate
accounting removes prior-attempt reserves and repeated retention charges, then
counts the whole public research object pool once. That pool includes failed
research. Current controller disk provisioning is conservatively charged for
the entire measurement interval, including the older services sharing those hosts.

Both incoming and outgoing measured bytes are priced at $0.15 per decimal GB,
even when AWS does not charge for that direction or route. Disk and 93-day
retention estimates use a conservative 28-day month. A separate $25 allowance
covers unmetered object request charges; it is an allowance, not a measurement.
The source/evidence archive allowance is 1 GiB. Unlimited future downloads, new
objects and perpetual retention are outside this finite scope.

The offline [reconciliation script](../scripts/report_llm_costs.py) checks the
declared allocation set, raw network sums, original prices, unique allocation
identities, complete provider-attempt count and retained-object inventory. It
recalculates the learning costs and reports earlier provider caps separately.
The final provider cost, controller cut-off, public inventory and aggregate
result will be recorded when the ongoing frozen service correction finishes.

## Remaining public-market constraint

Funded complete replay prevents an unfunded claim from earning tokens. It does
not prevent a wealthy or subsidized adversary from occupying scarce work slots.
The current unaccepted audit-offer pool also has a demonstrated saturation
attack. [Atomic admission and priced reservations](AUDIT_ADMISSION_RFC.md)
remains a proposal, not a shipped mitigation. This finite operated funding
evidence must not be advertised as permissionless-market availability or as
proof that NEURO rewards alone will finance future model growth.
