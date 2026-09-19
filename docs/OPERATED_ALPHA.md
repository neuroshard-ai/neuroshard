# Operated alpha

The next deployment uses the accepted growing graph on separately hosted AWS
machines. One administrator still controls the deployment. Different machines,
availability zones, wallets and consensus keys test failures and protocol access;
they do not establish independent administration. TODO 4 remains open until its
independent-operator criterion is met. A working public deployment addresses
TODO 6; it does not establish ChatGPT-level general capability.

The implementation is on `development/operated-alpha`. It requires a new genesis
and matching source. The existing 0.4.0 chain and balances are not upgraded by
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
