# NeuroShard protocol candidate v2

Status: executable, bounded reference protocol, 2026-09-10. The [experiment report](https://github.com/neuroshard-ai/neuroshard/blob/108b4ba3d6c6fb5760ff211b447ee95a67fa9112/docs/PROTOCOL_EXPERIMENTS.md) distinguishes observations from assumptions. These inherited ledger rules are retained for implementation context; the current application is specified in the [LLM protocol](LLM_PROTOCOL.md), with separate [model-evolution work](EVOLUTION_PROTOCOL.md).

This candidate specifies the complete lifecycle for its supported tasks: genesis, accounts, resource-based validator entry, native finality, task reservation, sharded execution, verification, payment, expiry, evidence, exit, and recovery. Its execution profile is intentionally small and uses full replay. This is not a claim that economical permissionless LLM pretraining, arbitrary heterogeneous hardware, or production economics have been solved.

## 1. Scope and trust boundaries

NeuroShard is its own CometBFT 0.38.26 chain. It has its own genesis, validator keys, native balances, voting-power updates, and model history. No external blockchain settles its transactions.

Participants may hold several roles:

| Role | Required resource and authority |
|---|---|
| Full node | Published genesis, compatible numerical runtime, dataset and full state, and enough compute to replay the supported tasks |
| Validator | Full-node requirements, an account with spendable tokens, and proof of control of a newly bonded consensus key |
| Task sponsor/coordinator | Transaction fees and reservation collateral; selects counterparties and coordinates its assigned task |
| Training worker | One supported model stage and an account signing key; no account registration or token balance is needed to receive a reward |
| Inference provider | A full accepted model for this implementation and an account signing key |
| Content peer | Authenticated corpus/checkpoint chunks; retrieval does not depend on one named provider |

Any funded account can transfer, reserve work, or bond a consensus key using the same transaction rules. No admission-key signature or operator approval appears in those rules. A worker can earn its initial balance by performing a sponsor's task and subsequently bond it. This establishes an open transaction path; it does not guarantee that every newcomer finds a sponsor or that task assignment is fair. The sponsor bears the risk of its selected workers failing to finish.

The protocol assumes secure SHA-256/SHAKE-256 and signatures, a valid genesis or appropriate recent trusted checkpoint, compatible execution semantics, available task data, and less than one third faulty voting weight in every effective validator set. Safety and liveness assumptions follow the native BFT engine; progress additionally requires eventual synchrony and enough responsive voting weight. The model/shard participant count is not a substitute for a voting-resource bound. [CometBFT consensus specification](https://github.com/cometbft/cometbft/blob/v0.38.26/spec/consensus/consensus.md).

The recorded deployments use one host and, subsequently, two separate machines; all nodes and keys remain under one operator's control. Public entry in the state machine is distinct from decentralized ownership. A stake-based history also needs a long-range/weak-subjectivity policy for clients that have been offline beyond the collateral/evidence horizon. A recent checkpoint must be acquired through an independently trusted process; accepting whichever RPC endpoint answers first is not a solution. The candidate does not claim Bitcoin-style objective bootstrap from an arbitrary untrusted peer.

## 2. Genesis and numerical contract

Genesis fixes the chain ID, native validator public keys/powers, account allocations and initial bonds, protocol parameters, model program, dataset digest, initial parameter root, numerical profile, and three-step conformance vectors. The application rejects a native validator set whose keys or powers do not match its collateral ledger. It also rejects native consensus parameters that differ from its evidence windows or execution bounds: 65,536-byte blocks, 16,384-byte evidence budget, no gas limit, Ed25519 consensus keys, and disabled vote extensions. A mismatch between engine evidence age and application unbond timing cannot silently initialize.

The recorded development allocation is four validators, each with 2,500,000 bonded atoms and 20,000,000 liquid atoms: 90,000,000 total atoms. One development NEURO has 1,000,000 atoms. These are test allocations and constants, not a public distribution or supply proposal.

The task model is the repository's 34,976-parameter NeuroLLM: two decoder layers, byte vocabulary 256, hidden width 32, four attention heads, two KV heads, intermediate width 64, and context limit 64. Training uses next-token cross-entropy, four sequences of 32 bytes, SGD at 0.1, no momentum, and no dropout. The last 5% of the recorded TinyShakespeare corpus is held out. Training batch selection is deterministic from the round, with the RNG procedure bound by the source manifest. There are no uncommitted Adam moments, compression residuals, or mutable architecture changes in this profile.

`scripts/protocol_lab.py` defaults to `ATEN_CPU_CAPABILITY=default` and `MKL_ENABLE_INSTRUCTIONS=SSE4_2` before importing PyTorch. Validators use one CPU thread and disable MKLDNN. Genesis includes the dispatch selection and exact roots/loss encodings from three successive reference updates. Each validator recomputes and compares that manifest when initializing or reopening saved state. A mismatch prevents it from starting the application.

The source manifest also binds the numerical program and candidate code. The old reference's reward/lease constants are excluded from the candidate execution description; v2 economics are defined only by the parameter table below.

Conformance vectors are a finite compatibility gate, not a proof of bitwise equivalence for all inputs and hardware. Twenty successive training updates matched between a Xeon 8259CL/Python 3.10 host and a Xeon 8175M/Python 3.12 host under the pinned CPU profile, including gradients, stage receipts, inference, and the entire genesis manifest. Every actual task still undergoes full replay. Reproducible operators remain necessary for a broad, heterogeneous numerical contract; Verde's RepOps work is a relevant primary reference. [Verde](https://arxiv.org/abs/2502.19405).

## 3. State and encoding

The canonical application state contains:

- Chain ID, manifest, committed height, and consensus block timestamp in integer nanoseconds.
- Accounts indexed by canonical compressed secp256k1 public key, each with a nonnegative integer balance and next nonce.
- Consensus bonds indexed by an Ed25519 public key, with owner, amount, lifecycle status, scheduled update, effective voting-power history, and withdrawal deadlines.
- Current model parameters/root, training round, one optional reserved task, completed task IDs, and the most recent paid inference output.
- Initial supply, issued amount, burned amount, processed evidence IDs, and a deferred verifier-payment escrow.

JSON uses sorted keys, compact separators, ASCII escaping, and no NaN or Infinity. Duplicate JSON keys are rejected. Integer fields reject booleans and out-of-range values. The signed envelope is `{body, public_key, signature}`; ECDSA signs the canonical body. Every body includes exact `kind`, `chain_id`, and account `nonce` fields. Each kind has an exact field set. Public-key encodings are canonical lowercase hex.

Training tensors use bounded shape metadata and base64 little-endian float32. They are never deserialized with pickle. Network messages are capped at 2,000,000 bytes. Candidate transactions are capped at 16,384 bytes and carry commitments/signatures rather than model tensors. This implementation accepts at most one application transaction per block.

Each accepted transaction consumes its nonce and burns its fee. Invalid transactions leave account, nonce, fees, model, and rewards unchanged. Re-signing the same message cannot bypass the nonce. A reservation ID hashes the signed body and signer, excluding ECDSA's randomized signature bytes.

## 4. Parameters of the recorded lab profile

| Parameter | Value |
|---|---:|
| Transaction fee | 1,000 atoms |
| Voting unit / minimum bond | 250,000 atoms |
| Membership epoch | 8 blocks |
| Minimum activation scheduling delay | 4 blocks |
| Evidence age in blocks | 24 blocks |
| Evidence age in time | 6 seconds |
| Reservation bond | 2,000,000 atoms |
| Execution lease | 16 blocks after reservation height |
| Training issuance budget per accepted task | 1,000,000 atoms |
| Total training-worker budget | 800,000 atoms |
| Verifier/consensus budget per accepted training task | 200,000 atoms |
| Maximum rewarded training tasks | 1,000 |

The short time windows exist to exercise failure cases quickly. They have not been sized for WAN conditions, public evidence distribution, or production operators. A deployment requires windows justified by state retrieval, execution, propagation, and finality costs.

The deployment candidate also defines a genesis-committed `testnet` profile: 60-block membership epochs and activation delay, evidence windows of 172,800 blocks and 172,800 seconds, 120-block leases, and a 100,000-task issuance cap. All transitions read the profile stored in chain state. The table above and the concrete H+16 examples below describe the accelerated lab profile. See [public operation](PUBLIC_TESTNET.md) for the deployment parameters and bootstrap policy. Neither profile defines a production token distribution.

## 5. Block transition and native finality

For a block at height H, the application applies these deterministic operations:

1. Set height/time from the agreed block. Pay the previous block's deferred verifier budget using its actual commit signatures and the voting-power snapshot at the task's accepted height.
2. Process new consensus-validated evidence, schedule jailing, emit due membership changes, and establish withdrawal deadlines at the effective removal height.
3. Expire an unfinished task whose deadline is less than H, burn its reservation bond, and refund any unused inference price.
4. Execute the optional transaction and check conservation invariants.
5. Return the resulting state hash and validator updates from `FinalizeBlock`. Persist the entire transition atomically in SQLite at ABCI `Commit`.

The native engine supplies and validates commits and evidence; clients cannot submit fabricated evidence metadata directly to the application. `PrepareProposal` and `ProcessProposal` use the block's same time/evidence and commit information. Mempool checks conservatively use committed state and do not make uncertain future rewards spendable.

ABCI updates emitted by block H become effective voting power at H+2. The application stores that effective-height history and uses it for rewards and offense lookup. A bond's display status `active` means its activation update has been emitted; actual voting eligibility is determined by the history at the queried height. [CometBFT ABCI update semantics](https://github.com/cometbft/cometbft/blob/v0.38.26/spec/abci/abci%2B%2B_methods.md).

Application state H is committed in the next header's `AppHash`. A client checking a state commitment against a header must account for this one-height relationship. The current convenience RPCs provide current-state reads without inclusion proofs, and explicitly reject proof/history requests they cannot satisfy. Independently checking application state currently requires a full node that replays the history from a trusted starting point.

## 6. Bonded membership and evidence

`bond(consensus_key, amount, possession)` debits spendable funds. Amount must be a positive multiple of 250,000 atoms. The Ed25519 key must never have been used for another bond in this chain. Its possession signature covers a domain separator, chain ID, account key, consensus key, amount, and account nonce, preventing key registration by someone who merely copied a public key or another owner's proof.

If the transaction is included at H, let E be the least multiple of eight with E ≥ H+4. The application emits the new power at E, and consensus applies it at E+2. Power is exactly `amount / 250000`; splitting the same stake among eligible keys cannot increase aggregate power. Each additional registration still costs a fee.

`unbond(consensus_key)` is authorized only by the bond owner. It schedules removal at the next membership epoch, effective two heights later. The last active validator cannot initiate exit until another activation has been emitted; a merely pending replacement is insufficient. This prevents a scheduling gap with zero voting power. A key that exits remains in the historical record and cannot silently re-register against a different owner or bond.

If R is the effective removal height, the latest possible voting height is R−1. The remaining collateral is withdrawable only once both conditions hold:

\[
H\ge R+24, \qquad t_H>t_R+6\text{ seconds}.
\]

Using removal-confirmation time is conservative relative to the last voting time. CometBFT expires evidence only when **both** its height age and time age exceed their configured bounds; a block-only unbond timer could release collateral while evidence was still valid. [Upstream evidence-age predicate](https://github.com/cometbft/cometbft/blob/v0.38.26/evidence/verify.go).

Fresh duplicate-vote or light-client-attack evidence burns `ceil(remaining_bond / 4)`. An active validator is scheduled to zero voting power at H+2 and becomes jailed; an already removed validator's still-reserved collateral can also be slashed. Evidence IDs include offense kind, validator address, and offense height, so replaying the same offense cannot repeatedly debit collateral. Withdrawal returns the remaining bond, less the ordinary transaction fee, only after both deadlines. The native test exercises genuine duplicate-vote evidence; the light-client-attack branch is not equivalently tested.

Native BFT does not become safe if one third or more of voting weight is faulty merely because collateral exists. Valid evidence can reduce future power and collect a penalty; it cannot repair a violated consensus assumption or retroactively guarantee availability.

## 7. Work assignment and execution

`reserve(task_kind, parent, round, workers, request, price)` binds the current model root/round and exact eligible worker keys. A training job names two workers, has an empty additional request, and has price zero because its budget comes from issuance. An inference job names one provider and a public prompt/greedy decoding limit; its price must be at least ten fees.

Only one job is outstanding in this profile. The sponsor locks the 2,000,000-atom reservation bond plus the inference price, if any. This freezes the task's parent until completion/expiry. A job accepted at height H may be completed through H+16; at H+17 it expires before transactions execute. A reservation cannot reuse collateral already held for another obligation.

For training, stage zero holds the embedding and first decoder layer; stage one holds the second decoder layer, normalization, and output head. The coordinator sends each only its parameter subset, relays forward activations, obtains stage one's loss/adjoint/gradients, and sends the adjoint back to stage zero. Stage zero recomputes its forward pass before backpropagation. Both sign task-bound commitments to their inputs, outputs, gradients, and loss.

`submit(task_id, result_root, receipts)` is authorized by the sponsor account. Every validator independently executes the complete reference task from its accepted local state. A training submission must match the full parameter-update root and every stage receipt. An inference submission must match the full output commitment and provider signature, with its prompt, decoding rule, and model root bound to the reservation. Signatures establish authorship; exact replay establishes the prescribed result.

Worker identity does not prove which physical processor ran the computation. Subcontracting and precomputation compatible with the task are allowed. Public deterministic results earn at most the assigned task's one reward; copying or re-signing a completed transaction does not create another payment. The profile does not attempt to prove expended joules or unique hardware ownership.

Incorrect submissions receive no reward and cannot advance the model. They may be corrected within the lease. On expiry, the reservation bond is burned and the inference price refunded. The next sponsor can reserve the same still-accepted model round. There is no timeout path that invents a proxy gradient, advances an unverified descendant, or converts absent work into issuance.

The sponsor's collateral addresses reservation griefing, not all denial of service. It neither prevents a well-funded actor from repeatedly reserving work nor proves fair transaction inclusion. Public worker discovery, adversarial worker selection, and sponsor/subcontractor risk remain separate market/network concerns.

## 8. Settlement and monetary invariants

On a valid training submission, the same application transition:

1. Installs the verified model and advances its training round.
2. Marks the task completed and clears its reservation.
3. Returns the sponsor's reservation bond.
4. Issues exactly 1,000,000 atoms: 400,000 to each worker and 200,000 into deferred verifier escrow.

On a valid inference submission, the accepted model is unchanged. Eighty percent of the funded price goes to the provider; the rest enters deferred verifier escrow. Inference mints no tokens. The output is retained for retrieval in this bounded demonstration. Prompt privacy, streaming billing, and KV-cache migration are not provided.

For task block H, the verifier budget is paid at H+1 using H's actual commit certificate. Each committing validator receives `floor(budget * power / total_active_power)` at the task's height. Absent voting weight receives zero. Unpaid shares and integer remainders are burned. This avoids an offline validator automatically earning the same verifier share as a participant. The certificate proves a protocol signature, not that each signer physically repeated the computation; the honest-voting assumption still matters.

Flooring each key's share and burning remainders means splitting one voting stake among keys cannot increase the aggregate proportional payment. The worker budget is fixed per task, so one operator controlling both stages still receives at most 800,000 atoms.

After every transition:

\[
S_{\mathrm{initial}}+S_{\mathrm{issued}}
=S_{\mathrm{liquid}}+S_{\mathrm{bonded}}+S_{\mathrm{task\ escrow}}
+S_{\mathrm{verifier\ escrow}}+S_{\mathrm{burned}}.
\]

All terms are nonnegative integers. Issued atoms equal accepted training rounds times 1,000,000, capped at 1,000 tasks. Supply accounting includes pending activation, cooling/jailed bonds, and deferred rewards. Exhausting training issuance does not disable transfers, funded inference, or exits. These constants test accounting; they do not establish profitable mining or a sustainable public reward schedule.

Correct execution also does not imply that each stochastic step improves validation loss. The protocol pays the prescribed computation and measures model quality separately. Concentrated ownership of correctly earned rewards can eventually violate the consensus resource bound; delayed activation alone does not prevent this. The [economic scenario results](https://github.com/neuroshard-ai/neuroshard/blob/108b4ba3d6c6fb5760ff211b447ee95a67fa9112/docs/eval/results/protocol_economics.json) make that failure mode explicit.

## 9. Availability, restart, and invalid state

The supported corpus is split into 16,384-byte chunks. A manifest binds ordered chunk hashes, total length, and full-object SHA-256. A joining full node can retrieve chunks from interchangeable peers, reject corrupt/oversized/truncated responses, retry another peer, and publish the assembled object only after all checks pass. All-corrupt/unavailable sources cause bootstrap to fail closed.

Each validator retains the entire accepted model and the corpus required for replay. Consequently, this version does not accept a separate worker availability assertion and hope an auditor can later retrieve an unknown result: it reconstructs the result locally before acceptance. The corpus-fetch experiment does not establish a large-scale storage market, erasure-coded availability, or availability proofs for arbitrary checkpoints. New nodes replay the native history; scalable snapshot sync is not implemented.

The application persists model, accounting, history, and effective membership together. A crash after `FinalizeBlock` but before `Commit` leaves the preceding durable state; replay returns the same state hash and validator updates. CometBFT signing state must be preserved across restarts. The test-only equivocation helper deliberately bypasses that protection on a disposable key to create real evidence; it is not part of a normal validator's signing path.

Unavailable input or numerical mismatch cannot authorize an update. It can reduce liveness if too much voting weight cannot execute. The protocol has no emergency admin transaction that bypasses verification, rewrites balances, or accepts a poisoned model. Current queries are conveniences for local full nodes, not authenticated light-client proofs.

## 10. Verification cost and the experimental alternative

Full replay is the authoritative predicate in v2. It requires every validating node to hold and execute the complete small model, so it has poor scaling despite distributing the worker computation. Per-job execution is bounded here by the fixed small model, batch, decoding cap, and one-transaction block rule. The implementation also copies/hashes its full application state; long-lived account/history growth and paginated authenticated queries have not been benchmarked or optimized. These choices are unsuitable for arbitrary large jobs or public-scale state without further work.

The separate integer-matrix experiment checks `A(BR) = CR` with 128 binary projections. Inputs are bounded integers in [−127,127], dimensions are at most 2,048, and the output is range checked. The largest unreduced intermediate magnitude is bounded by `K*N*127^2 < 2^63`; exact int64 arithmetic therefore avoids overflow and floating tolerance ambiguity. This is a concrete application of probabilistic matrix verification. [Freivalds, *Fast probabilistic algorithms*](https://link.springer.com/chapter/10.1007/3-540-09526-8_5).

For any fixed false product and independent uniform binary challenges, at least one nonzero error row survives with probability at least one half per projection, giving false acceptance at most 2^−128. The implementation derives projections from the complete domain-separated statement, including dimensions, task/operation context, A, B, and proposed C. Treating this as a noninteractive check additionally requires a random-oracle assumption and an explicit adversarial query/grinding bound: at most approximately `Q * 2^-128` over Q attempted false statements, plus hash-failure terms.

The prover still computes C, and the verifier still reads/authenticates all matrices. This is not a succinct proof or a privacy protocol. A known projection admits a constructed false product; a tolerance-based floating check also accepted a changed result in the experiment. The candidate therefore does not use those shortcuts to accept floating-point NeuroLLM gradients. Linking every linear/nonlinear/backward/optimizer operation into a valid training graph and preserving useful learning quality under a portable numerical profile remain necessary before this verifier can replace full replay. Slalom demonstrates a related linear-layer verification direction under a different trusted-hardware threat model. [Slalom](https://arxiv.org/abs/1806.03287).

## 11. Coverage and extension boundary

The implementation defines all accepted transactions and their normal/error/expiry outcomes for this profile. Unsupported job kinds, fields, numerical manifests, and model upgrades are rejected. Protocol/model changes require a new version and development genesis; this candidate defines no automatic migration or ownership rights across such chains.

Tested mechanisms include real native admission/removal, actual commit participation, slashing after removal, both withdrawal deadlines, nonces, escrow, supply conservation, authentic pipeline computation, paid inference, corrupt-peer fallback, and restart determinism. The recorded result is not an exhaustive adversarial proof.

The largest unresolved gates are economical verification of an entire useful training graph, broader hardware conformance, independent operator deployments, long-range client bootstrap, Sybil-resistant work discovery/assignment, sustainable genesis/emission choices, measured WAN cost, and broader consensus partition/adaptive-attack testing. The two-machine experiment uses SSH forwarding for actual native P2P and stage traffic; it does not test public peer discovery or independent ownership. These limits are explicitly outside the empirical claim of v2 rather than hidden behind a generic “Proof of Neural Work” label.

## 12. Public node and outbound worker interface

The deployment wrapper accepts independently generated, signed genesis declarations, assembles a checksum-pinned genesis bundle, and initializes a local full node only if its source/data/numerical manifest matches. Account and consensus private keys stay on the participant's machine. It enables native P2P discovery, bounded public account/validator/block reads, and a signed-transaction relay; native administrative RPC and application gRPC remain on loopback. The public gateway supplies observations from a full node, not account inclusion proofs.

An optional coordinator supplies collateral and maintains a bounded registry of outbound workers. Workers poll over HTTPS, authenticate sponsor assignments, and check their local finalized lease, assigned account, stage, source model, batch, and expiry. Per-operation durable records prevent repeated or modified requests from triggering unbounded computation under one lease. Returned receipts still require complete validator replay before payment. A coordinator can censor assignments or withhold submission; workers can withhold results and expose the sponsor to expiry losses. This transport defines an open contribution path, with no claim of Sybil-resistant scheduling or guaranteed payment.

The node refuses public initialization of an old stake history without an explicitly supplied checkpoint and terminates if the replayed block differs. Checkpoint recency and authenticity remain operator responsibilities. Direct public-port P2P, fresh joining, earned-stake admission with testnet delays, quorum halt/recovery, and common block/model agreement were exercised between two machines. These remain one-operator experiments. The [deployment guide](PUBLIC_TESTNET.md) specifies the supported commands, port exposure, and release limits.
