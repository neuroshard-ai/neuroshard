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
The complete report accounts through **20:00 UTC on September 19**, after the
last disposable allocation was retired. Its conservative total is **$792.54**
against the frozen **$1,000** sponsorship cap:

| Cost category | Conservative USD estimate |
| --- | ---: |
| Three learning allocations, including rejected/recovery work | $562.82 |
| All seven provider allocations, removing duplicate retention/reserves | $115.59 |
| Both controllers, including shared older services | $43.83 |
| Entire public object pool for 93 days, plus archive allowance | $45.30 |
| Additional object-request allowance | $25.00 |
| Total | **$792.54** |

The inventory contains 11,303 objects and 591,927,169,917 bytes before the 1 GiB
archive allowance. That allowance covers the subsequently published final
service evidence/source and this cost archive. Controller traffic totals
115,482,829,645 measured incoming/outgoing bytes. All seven provider allocations
meet their own frozen cost gates; the earlier learning-cap caveats above remain.

The [public cost archive](https://dwquwt9gkkeil.cloudfront.net/research/native-expert-live-20260916/objects/235414f949555b375456c4e515ae1dd89cb8ec5e88e548e9123efc1034cde767)
passed full SHA-256 readback and contains 68 allowlisted files: raw measurements,
allocation/retirement records, prices, inventory, funding contract and the offline
reporter. It excludes private keys, credentials and billing-account balances.
After checking its digest and extracting it, reproduce the report without AWS:

```bash
python3 costs/report_llm_costs.py --evidence costs \
  --expected-provider-attempts 7 --output recomputed-costs.json
```

Missing controller series and altered raw measurements are rejected. This
completes checklist item 5 against its explicitly allowed **finite-sponsorship**
criterion. It establishes the cost and funding of these bounded obligations;
it does not establish affordable permissionless verification at arbitrary model
size or a self-funding public economy.

## Remaining public-market constraint

Funded complete replay prevents an unfunded claim from earning tokens. It does
not prevent a wealthy or subsidized adversary from occupying scarce work slots.
The current unaccepted audit-offer pool also has a demonstrated saturation
attack. [Atomic admission and priced reservations](AUDIT_ADMISSION_RFC.md)
remains a proposal, not a shipped mitigation. This finite operated funding
evidence must not be advertised as permissionless-market availability or as
proof that NEURO rewards alone will finance future model growth.
