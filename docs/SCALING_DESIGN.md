# A network that grows a shared model

**Design direction, September 2026.** NeuroShard's objective is a collectively trained, openly retrievable language model, maintained by a permissionless network with its own consensus. More usable resources should increase the network's ability to learn, retain knowledge and serve requests. Parameter count is one possible result of that capacity, not the admission rule or reward metric.

The public 0.4.0 adapter network, the experimental full-model native lifecycle, and the proposed scaling protocol below are different maturity levels. This document chooses an architecture for further work. It does not activate unimplemented transactions or claim that the deployed network has an independent audit market.

## What changes in the thesis

| Earlier intuition | Working design rule |
| --- | --- |
| Each new peer makes the model larger | Use sustained capacity first for missing replicas, complete audits, serving and useful training; grow only when those obligations remain covered |
| Training work secures block production | Native bonded BFT orders the ledger; prescribed neural work earns a bounded allocation of NEURO under its own verification assumptions |
| A correct update improves the model | Computation acceptance and serving promotion are separate; correct but unsuccessful experiments can be paid within a finite research budget |
| A challenge bounty pays for verification | Honest observation needs a budget even when there is no fraud; a signature is not evidence of independent computation |
| New downloaded text is new knowledge | Provenance, permission, deduplication, task relevance and contamination checks precede admission; unused rows do not establish recent facts |
| More weights imply more intelligence | Compare quality and serving cost at equal total resources, including rejected work, retention, verification and storage |

The useful-computation mining concept survives. It is a work allocation and settlement mechanism, not a claim that a gradient supplies Bitcoin-like block-selection security. Token issuance alone does not finance hardware in the real world or create demand for inference.

## Architecture: local groups, global commitments

Keep a single native ledger for model versions, budgets, data admission, work obligations, disputes, quality decisions and payment. Keep bulk numerical work and immutable artifacts outside block production. The chain must be able to resolve every supported disputed transition from bounded evidence available to all validators.

Use **compute groups**: a set of peers that can jointly hold and execute a model with tolerable latency. Model parallelism works inside a group; multiple groups provide replication and, later, distinct training contributions. A peer with insufficient capacity for one supported task can supply another bounded service. No peer count is treated as measured FLOPs, memory, independence or reliability.

The existing pipeline proves the mechanics of partitioned full-model execution. It is sequential and has a coordinator. A coordinator can be replaceable without being trusted for correctness, but a single coordinator is still a liveness dependency. The next coordinator implementation must reconstruct assignments and progress from finalized obligations and immutable journals, with bounded lease expiry and no second reward for a retry.

[DiLoCo](https://arxiv.org/abs/2311.08105) motivates reducing communication between cooperating groups. The later [Decoupled DiLoCo report](https://deepmind.google/blog/decoupled-diloco/) also studies isolation of failures between learner units. Those results concern distributed optimization and infrastructure. They do not establish Byzantine security, public admission, or economical verification of NeuroShard's work. Its published 88% goodput comparison is from simulation; its actual multi-region training experiment is separate evidence.

For the first implementation of multiple learning groups, use **bounded synchronous windows** from a common accepted parent. Fix each group's data, optimizer, step count, seed, deadline and aggregation weight before work begins. Commit all outputs before accepting an aggregate. Define the outer optimizer and reduction order exactly, and give aggregation its own reproducible dispute path. A missing group either triggers the precommitted reduced-membership rule or cancels that window; the coordinator cannot silently invent a replacement contribution.

Do not accept arbitrary stale updates or assign reward in proportion to gradient norm. Adding asynchronous windows, optimizer momentum, sparse experts or a changed numerical profile requires a separately tested state transition. Sparse experts may eventually reduce active compute per token, but routing, expert availability, shared layers, load imbalance and training verification remain obligations. Merely creating more experts does not solve them.

## Resource admission before growth

Admission should describe a **service obligation**, not a self-reported machine specification. Bind the provider key, execution profile, task bounds, price, collateral, availability period and artifact-retention deadline. A resource advertisement guides scheduling; objective completion and retrieval checks establish whether its assigned service was supplied.

The planner must account for weights, gradients, optimizer state, peak activations, temporary buffers and runtime overhead. The current 48M-parameter worker bound is not a memory proof. Serving capacity and complete replay capacity must be tested for a proposed larger model as well as training capacity.

The current profile permits at most 64 partitions, caps each worker at 48M parameters, bounds individual artifacts at 256 MiB and training metadata at 512 KiB. Adding peers does not remove these bounds. A much larger model or a new numerical layout needs an explicitly versioned profile and new conformance measurements; enlarging constants without measuring verification and availability is not a scaling result.

Before proposing growth, require all of the following under a published policy:

1. Sustained completion and retrieval measurements over a defined observation window, including failed and timed-out jobs.
2. Enough spare resources to finish the cohort, audit every obligation, retain its artifacts and serve the candidate if promoted.
3. Replicas in distinct declared failure domains, with a tested recovery path if the largest domain disappears. Host or owner labels are observations, not cryptographic identity proofs.
4. A funded finite training/evaluation plan comparing growth with continued training of the smaller model at comparable total cost.
5. A checkpoint-compatible growth transform, followed by retention and fresh-task evaluation. Identity initialization preserves an initial function; it does not prove improved usefulness after training.

Growth remains a non-issuing bonded operation. Training a larger model may earn its assigned work budget; manufacturing parameters does not earn additional tokens. Failure to sustain a larger candidate leaves the last accepted serving checkpoint available. Shrinking or distilling a model is also a valid capacity response, subject to its own quality decision.

## Verification and payment are separate mechanisms

The [compact SGD extension](COMPACT_UPDATE_DISPUTES.md) implements a bounded negative proof for one asserted optimizer relation. It reduces the cost of one class of adjudication. It does not prove the gradients, attach the Merkle tensors cryptographically to flat safetensors hashes, or pay for observing an honest computation.

The complete target is a committed graph covering parameter reads, forward operators, loss, backward operators, reductions, clipping, optimizer state and parameter writes. A dispute must identify a first inconsistent edge or operation starting from an accepted checkpoint. Tiled operators need explicit reduction order, domains, shapes, byte encodings, padding rules and links to their inputs. An authenticated but invented gradient is still wrong.

[Verde](https://arxiv.org/abs/2502.19405) provides relevant prior work on narrowing an ML disagreement and reproducible arithmetic. Its acceptance guarantee assumes at least one honest participant in the delegated computation. NeuroShard cannot inherit that assumption merely by counting keys, nor can a narrow optimizer witness stand in for its complete graph protocol.

Keep the full-stage implementation as a differential oracle until each replacement path has adversarial coverage, cross-machine conformance and measured total cost. An unavailable object, an unsupported operator or an incomplete dispute must not become a successful quality decision or an unbounded reward stream.

## A budget for honest auditing

For each work window, reserve a maximum budget before leasing work:

`B = training + complete_audit + availability + evaluation + settlement_reserve`

These are separate liabilities. Any issuance contribution must come from the published period cap; prepaid inference, sponsorship or fees supply transfers of existing NEURO. Unspent funds return to their source. Fraud bounties come from locked collateral under the adjudication rule. No failed claim, repeated work identity or self-challenge creates a second training emission.

The first credible audit service is explicitly purchased complete replay of a defined obligation, with signed coverage and retrievable evidence. Its payment should be reserved when work is assigned and remain payable for a correct completed audit when training is honest. This buys an accountable service under stated operator assumptions. It **does not cryptographically prove that separate auditors used separate computers**.

Auditors should commit before disclosing their results, have collateral exposed to objective false attestations, and retain the required evidence through the dispute window. A commitment prevents simple copying after public revelation. It cannot stop private sharing with a producer or colluding auditors. Paying each matching signature, requiring a key-specific hash, or dividing one replay among many Sybil identities does not establish independent coverage.

[Proof of Diligence](https://arxiv.org/abs/2402.07241) explicitly analyzes both lazy collusion and sharing the cost of a real replay. [Peer-prediction work](https://arxiv.org/abs/2406.01794) studies richer reports under assumptions about other participants and information. These are research inputs, not a deployed solution to arbitrary collusion in this network. A future reward mechanism must publish its assumptions and adversarial equilibria before controlling public issuance.

Two simple checks constrain any proposal:

- A fraud-only observer with replay cost `C`, probability `p` of being the successful rewarded detector, and bounty `R` has expected net reward `p*R - C`. As successful fraud becomes rare, that cannot cover a fixed positive cost. Paying honest auditing therefore belongs in the normal budget.
- Under genuinely independent draws from a population with adversarial resource fraction `a`, `k` complete auditors all being adversarial has probability `a^k`. With fully correlated ownership or one shared replay supplier, the corresponding risk can remain `a`. For `a = 0.25` and `k = 4`, those are 0.39% and 25%, respectively. Neither number describes the current single-operator deployment, and neither is a consensus-security theorem.

Require a funded audit obligation for **every** accepted training stage and all dependencies, not one cheap sampled SGD chunk per model. The [funded candidate](FUNDED_AUDITING.md) now reserves existing tokens for selected auditors and gates all execution claim kinds on complete-coverage reports. Its daemon replays the entire graph, but signatures do not prove that work occurred independently. This remains an operated service with explicit sponsor selection; the public 0.4.0 reward split is unchanged.

## Membership and assignment

Native stake remains linear in bonded value with explicit activation, withdrawal and evidence windows. Identity splitting does not increase total voting power, while concentration of ownership still determines the practical security assumption. Adding one independent operator does not by itself produce an independent quorum.

Provider admission needs bounded outstanding leases and value-proportional collateral. Objective non-delivery can expire a lease; mere disagreement or poor validation performance is not automatically provable misconduct. Reward identity is fixed before result disclosure so that copying cannot acquire someone else's reserved task reward. Retries of the same numerical task preserve the existing paid-work identity.

Selection rules must analyze stake concentration, lease monopolization, withholding, censorship and producer grinding. A hash of a proposer-influenced block or a data proposal is not automatically an unbiased lottery. Random committees require a separately specified randomness mechanism and its failure behavior. Stake-weighted selection without that work is an experimental assignment policy, not a proof of independent auditing.

Newcomers need a published route to small funded obligations and then larger collateralized ones. A permanent operator faucet or manual provider allowlist cannot be the final admission authority. Bootstrap distribution, cost of acquiring stake and access to the ledger remain part of the protocol's economics.

## Learning and serving that users can assess

The tokenizer is versioned with the model. Arbitrary vocabulary edits would change the meaning of token IDs and invalidate existing weights, data commitments and serving requests. A tokenizer change is a model migration with retokenized data and compatibility checks; adding peers requires no tokenizer change.

Data collection stays separate from activation. S3 is a mirror for content-addressed objects; neither its owner nor a changing object path decides training inputs. Admission binds exact source revisions and bytes, documents provenance and permissions, checks train/evaluation separation and includes tokenization review. Replication must outlive the mirror. Poisoning, benchmark leakage and subjective relevance need explicit review assumptions.

The current native evaluation cohort is public before training. Selecting a subset after the candidate commitment limits one form of choice but **does not make those examples a secret holdout**. For evidence that a stranger can feel, add independent post-commit response tasks whose inputs were not used to choose the candidate, disclose the full scored set after the decision, and audit scoring. Curator secrecy and representativeness are additional trust assumptions. Public tests should remain for reproduction, with repeated testing and contamination accounted for.

Quality reports must compare with the serving seed and a same-compute smaller-model baseline, publish failed runs, track multiple domains and retention, and include actual generated responses and latency. A better mean loss on a small cohort does not establish general reasoning ability. No finite gate guarantees that the LLM becomes better on every task forever.

Separate fast response delivery from final payment while defining which party bears pending-dispute risk. Public prompts and answers are a product restriction of the present transparent ledger. A future private inference mode needs its own correctness and privacy design; encrypting a prompt without explaining execution and disputes is insufficient.

## Measurable release gates

| Gate | Evidence required | Present status |
| --- | --- | --- |
| Bounded neural adjudication | Real-model dispute bytes, validator time, observer cost and cross-CPU agreement; adversarial coverage of all supported operators | Optimizer refutation implemented; full graph still uses stage replay |
| Honest audit service | Complete purchased coverage, conserved budgets, objective false-report disputes, collusion analysis and independent operators | Prepaid complete replay and reporting implemented in the candidate; sponsor selection and ownership remain limitations |
| Useful learning | Post-commit independent tasks, retention, equal-cost baseline and accessible responses with measured latency | [Frozen contract](LEARNING_MILESTONE.md): 135M, no growth, sealed test; training blocked until the selection manifest is committed |
| Additional peers add capacity | Same task/quality target at measured total cost, loss of a worker/domain, recovery without duplicate reward | Two owned hosts demonstrate execution; the milestone measures reliability and overhead after a learning pass, not implied 135M capacity |
| Public independence | Independent ownership of sufficient voting power and services, admission and exit, recovery logs, public genesis | Four public validator keys under one operator |
| Sustainable service | Artifact retention/recovery, storage bounds, audit and inference funding, usable latency, privacy policy | Bounded testnet mechanisms; no established public compute economy |

The next public milestone is a reproducible, independently operated **training and settlement experiment** with useful measured output. That is the path toward the larger vision. A mainnet launch should follow those measurements, a specified upgrade path and independent protocol/security review; an attractive token loop cannot substitute for them.
