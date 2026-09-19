# Native provider assignments for the accepted answering graph

Status: implementation in progress on `development/provider-market`. This is an
opt-in research genesis extension, not an upgrade of the public 0.4.0 chain.
TODO items 4–6 retain their original completion criteria.

The accepted graph currently executes on an operator-installed group of hosts.
Clients must instead be able to discover signed, collateralized offers for the
graph's logical owners, reserve a complete assignment and recover it without an
SSH controller. Offers describe obligations, not proof of independent ownership
or available hardware. The native ledger remains the assignment authority.

## Assignment and recovery

Any funded key may register an HTTPS endpoint and certificate fingerprint, post
collateral and advertise a rank of the currently accepted graph. Offers have a
price, capacity and expiry. No operator allowlist grants provider membership.
The client selects offers; this is an explicit market choice, not an unbiased
random committee or a claim of Sybil-resistant provider diversity.

A hosted inference reservation binds the existing expert job, complete graph,
conversation, tokenizer, output bound, provider offers, prepaid complete audit
and a bounded provider fee. Every assigned owner acknowledges the assignment
before execution. Provider collateral is unavailable for other obligations
while reserved. Identities, endpoints and certificate fingerprints are copied
into the assignment: later advertisements cannot mutate an in-flight request.

Preparation and execution have native block deadlines. After either deadline,
any participant can propose available replacement offers within the original
provider-fee ceiling. The original request, graph and execution escrow stay
fixed. The assignment epoch changes; receipts bind that epoch so delayed owners
cannot settle the new attempt. Attempts and outstanding assignments are capped.
Recovery restarts the deterministic request from its pinned model; it does not
invent lost activations or accept a provider's cache as evidence.

Timeouts release assignments. Missing acknowledgement or a failed multi-owner
execution is not by itself proof that an individual worker cheated. In
particular, a client must not burn an honest provider's bond by submitting a
request that fails the provider's numerical preflight. The coordinator's native
claim bond is reserved with the assignment, then exposed to the existing
adjudication rules when it submits an execution claim. Service fees pay only
after accepted complete replay. No
hosting transaction issues tokens, and every transition conserves supply.

## Funding and delivery

The existing funded audit quorum must accept a complete budget before a hosted
request reserves work. Its reservation remains attached to the job across a
coordinator change; only the native replacement rule can change its publisher.
A failed adjudicated attempt consumes its own audit obligation. Any retry needs
another explicitly funded obligation, within the finite attempt limit.

The existing token-based execution payment is supplemented by each provider's
fixed fee for the *complete bounded request*: prompt processing, selection,
auxiliary models, transfer and its stated availability period. A token count
alone does not price all those costs. Retention beyond that service period,
auditing and ledger fees must be separately specified and funded. Measured costs
and a finite sponsorship policy are prerequisites for claiming item 5 complete.

The serving transport must authenticate the registered identities, enforce
message and memory bounds, fence assignment epochs and restore only committed
objects. Reusable model partitions may remain resident between requests; request
state and conversation context remain separate. The client must distinguish
provisional streamed delivery from final, replay-authorized ledger payment.
Prompts and responses remain public under the existing native request format.

The implementation now uses signed, certificate-pinned TLS 1.3 frames. A
provider's own synchronized full node authorizes its peers from one committed
job/assignment snapshot; arbitrary remote RPC responses are not accepted as
state proofs. Frames have explicit tensor/control limits, sequence numbers,
assignment epochs, retry identity and a bounded mailbox. A stopped local chain
or changed assignment stops authorization. After a process loses its frame
state, execution requires a new native epoch. The durable transaction outbox
preserves signed bytes when acknowledgement of submission is lost.

Five separate CPU processes now reproduce the fixed-group executor's complete
outputs over this transport, including ordinary and multi-turn answering.
Every owner signs the same graph/request/epoch/transcript statement. Fresh
owners restore only their own model partitions from hash-addressed mirrors.
These are numerical and protocol fixtures, not the remaining operated LLM
failure/latency trial or an independent-operator soak.

The experimental runtime is described in [the provider guide](PROVIDER_RUNTIME.md).
The subsequent [operated native preflight](PROVIDER_NATIVE_PREFLIGHT.md) passes
fresh-key replacement after coordinator and backbone-owner loss, followed by
complete funded replay and exactly-once settlement on the CPU fixture.

## Validation before an LLM allocation

- Native tests: open registration, complete assignments, collateral and supply,
  capacity exhaustion, graph pinning, prepaid audits, coordinator replacement,
  stale receipts, duplicate payment, failed audits, cancellation and expiry.
- Transport tests: authenticated bounded messages, wrong identity/epoch, corrupt
  objects, reconstruction on a replacement owner and coordinator restart.
- One committed serving trial reuses the accepted checkpoint, includes required
  backbone loss and concurrent clients, and freezes latency, availability and
  total-cost limits before allocation. It does not repeat learning or count a
  transport migration as a new learning cohort.

Independent administration cannot be demonstrated by additional keys or AWS
instances under this operator. Item 4 stays open until independently administered
operators complete its soak and ownership/voting-power requirements.

Relevant engineering precedents are [Petals' fault-tolerant distributed
inference](https://arxiv.org/abs/2312.08361) and [libp2p's binding of peer identity
to secure channels](https://github.com/libp2p/specs/blob/master/tls/tls.md).
Neither supplies NeuroShard's settlement, verification or ownership guarantee.
