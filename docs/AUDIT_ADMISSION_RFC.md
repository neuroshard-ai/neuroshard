# RFC: atomic audit admission and priced reservations

Status: proposed next-profile design, **not implemented or activated**. This RFC
addresses the offer-pool denial of service demonstrated by the
[funded candidate](FUNDED_AUDITING.md). It does not solve auditor collusion,
prove useful learning or define a production token economy.

## Problem and requirements

Sixteen unaccepted offers occupy the current global pool. At reference prices,
one account can lock 1,600,000 refundable atoms and pay 16,000 atoms in initial
transaction fees to exclude other sponsors for up to 100,000 blocks. More keys
are unnecessary. Increasing the pool or limiting offers per account does not
resolve this attack under permissionless identities.

The replacement should leave unaccepted offers outside native state, lock every
party's funds atomically before training starts, bound state and execution cost,
and charge for occupied capacity. It must preserve the supply invariant and
numerical work identity. Receiving a signed proposal must create no liability
until its agreed obligation is finalized.

## Proposed authorization flow

Sponsors negotiate bounded authorizations with auditors outside consensus. An
auditor signs an exact work intent containing the protocol domain, genesis and
source commitments, sponsor, publisher, workers, work kind, model/data parent,
batch or inference-job identity, stage bound, service price, collateral,
reporting windows, expiry height and a dedicated authorization nonce. A changed
parent, worker group or obligation requires fresh authorization. Signatures do
not prove independent ownership.

One native admission transaction carries the sponsor's payment authorization
and all required auditor signatures. Its state transition validates the full
intent, available balances, distinct signers, current nonces, expiry and capacity
before atomically consuming the authorizations, locking collateral and service
funds, and reserving the work. No intermediate unaccepted offer enters the global
pool. Workers begin after this reservation finalizes.

Use a separate monotone authorization nonce per auditor account, rather than
consuming its ordinary transaction nonce. Successful admission increments each
nonce once. A rejected admission changes neither authorization nor escrow,
apart from the specified submitting-account transaction fee. An explicit native
invalidation can advance an auditor's authorization nonce. Competing permits
with the same nonce deliberately allow at most one successful admission; a
short expiry limits the publisher's option to submit later. A signature does
not guarantee that collateral will still be available at submission time.

This borrows the narrow idea of signed approval with domain separation, a nonce
and expiry from [ERC-2612](https://eips.ethereum.org/EIPS/eip-2612). Its treatment
of withholding and replay is relevant. NeuroShard would specify its own native
encoding and state transition; the standard is not an implementation or security
proof for this work intent.

## Capacity charges and interference

Removing unaccepted offers leaves the existing active reservation as a scarce
resource. A sponsor must prepay a bounded occupancy allowance separate from
auditor compensation and fraud collateral. At a fixed positive rate `r` atoms
per occupied block and maximum lease length `L`, reserve `r × L` and burn the
charge for elapsed occupied blocks. Refund only unused allowance. The exact
activation, release and same-block charging rules need a versioned specification;
no concrete rate is proposed for launch here.

Acceptance must separately prefinance the auditor's worst-case supported
refutation and completion costs. Locking an audit bond is insufficient: the
current regression test shows an honest fraud detector unable to post its
challenge bond, followed by a missed-report penalty. An authorization must bind
the supported evidence-size and fee bounds and their funding source. A reserve
calculated only for honest commit/reveal transactions fails this requirement.

Under that simplified accounting, occupying one slot for `T` blocks consumes
at least `r × T` atoms, excluding transaction costs. Refunding service fees,
cycling identities or paying one's own auditor must not refund this occupancy
charge. Burning a congestion charge instead of returning it to the block
producer is a relevant precedent in
[EIP-1559](https://eips.ethereum.org/EIPS/eip-1559). Copying its fee formula would
not calibrate NeuroShard's very different neural, storage and dispute costs.

This is a conditional token-cost bound, not fair access or an economic security
theorem. A wealthy attacker can still buy capacity. Fraudulent issuance under
colluding audits undermines the assumed scarcity of the attack budget. Empty
leases, repeated cancellation and self-funded service require adversarial tests.

A third party can interrupt honest work with a dispute. The signed maximum
exposure and absolute claim lifetime must bound the sponsor's loss; the protocol
must separately specify who bears occupancy charges during interruption. Giving
every allegation free extension would recreate cheap denial of service. Charging
the innocent sponsor without a remedy creates a different griefing strategy.
This allocation question remains open and blocks adoption of the proposal.

Admission must also leave execution and byte capacity for reports, availability
responses, fraud evidence, refunds and completion. A slot fee cannot stop
validator censorship or protect an unbounded parser. The current 113 MB head
dispute and whole-stage referee require measured transaction, state-byte and
execution limits; admission pricing alone does not make adjudication scalable.

## Required falsification tests

- Flood unsigned or unaccepted proposals from many identities: no native offer
  records accumulate, and measured ingress handling remains bounded.
- Replay an authorization on another genesis, source, job or parent; alter a
  price, worker or deadline; invalidate its nonce; reuse it concurrently. Every
  unauthorized admission must fail without partial locks or nonce consumption.
- Crash after submission, change balances before inclusion and race two valid
  authorizations. At most one compatible reservation may consume each nonce.
- Saturate active capacity with finite attacker funds. Account for every occupied
  block, refund and burn, and publish honest-work waiting times under stated
  inclusion assumptions. Do not label a token-cost bound as censorship resistance.
- Interrupt honest work, withhold reports and upload maximum-sized evidence.
  Demonstrate bounded loss, available completion capacity and supply conservation.

## Alternatives and migration

Per-account quotas are bypassed with more accounts. Non-refundable rent on the
existing offer pool could price its occupancy, but retains a queue of work that
no auditor accepted. Larger deposits raise capital requirements while leaving
refundable occupancy cheap. Atomic admission removes that particular queue;
it still requires resource pricing and a separately defensible audit mechanism.

Implementation requires another source-bound protocol version. The existing
public and experimental balances, account nonces and validator signing state
must not be reset to introduce it. Adoption needs a reviewed activation or
explicit new-genesis policy, reproduction of the authorization and accounting
tests, and a full lifecycle trial on that exact source. This RFC is a proposed
direction following a measured failure, not an authorization to upgrade a node.
