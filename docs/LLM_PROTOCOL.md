# NeuroShard LLM protocol — testnet 1

This specification describes the implemented `neuroshard-llm-v1` execution profile in release 0.4.0. The canonical genesis, signed allocations, dataset and network descriptor are in [networks/neuroshard-llm-testnet-1](../networks/neuroshard-llm-testnet-1). The [v2 specification](PROTOCOL_CANDIDATE_V2.md) remains the definition of the inherited bonded ledger; the changes and complete LLM lifecycle are specified here. This is an experimental native chain, not a production economic or scalability claim.

## 1. Participants and trust assumptions

An account has a locally generated 256-bit seed. The native secp256k1 private scalar is SHA-256 of the seed's lowercase hexadecimal text, preserving the existing account format. Compressed public keys identify accounts. ECDSA/SHA-256 signs canonical ASCII JSON; an envelope contains `body`, `public_key`, and DER `signature`. Chain ID and monotonically increasing account nonce prevent cross-chain and account replay. Transaction identity hashes the body and public key, excluding the potentially variable ECDSA signature. Native CometBFT transaction hashes include the complete serialized envelope; these two identifiers are intentionally different.

Full nodes replay all accepted state transitions. Workers perform either frozen model feature extraction or adapter updates. Sponsors reserve training tasks and choose two workers. Providers answer inference requests addressed to their public key. Validators bond NEURO and use separate Ed25519 consensus keys. The same person may perform several roles. There is no website registration, admission token, external blockchain, or external settlement service.

CometBFT 0.38.26 provides NeuroShard's own replicated consensus. Safety requires less than one third Byzantine voting weight. Liveness additionally requires eventual synchrony and enough compatible, available validators. Genesis, data availability, correct pinned software and arithmetic, and a recent trusted checkpoint for an old stake history remain assumptions. Signatures authenticate claims; full execution replay verifies their contents. The protocol does not prove the physical location of computation or independence of participants.

The launch deployment has four genesis validators under **one operator across two hosts** and one advertised public bootstrap peer. Bonding is permissionless; actual operator decentralization has not yet been achieved. HTTPS discovery, the project sponsor, model mirrors and the website are optional conveniences. The initial serving provider is a single advertised service; users can choose another provider public key through the CLI.

## 2. Immutable execution profile

Genesis binds all model/tokenizer assets by SHA-256, the exact execution dataset, critical Python source files, library versions, arithmetic settings, initial adapter, validation loss, three gradient/update test vectors and an inference test vector. Nodes refuse initialization or startup when these differ.

The frozen base is `HuggingFaceTB/SmolLM2-135M-Instruct`, revision `12fd25f77366fa6b3b4b768ec3050bf629380bac`, Apache-2.0: 134,515,008 parameters. The trainable state consists of matrices A in R^(4×576) and B in R^(576×4), totaling 4,608 parameters. For the pretrained post-normalization hidden vector h, the output head receives:

```
h_adapted = h + (h @ A.T) @ B.T
logits = h_adapted @ frozen_output_head.T
```

A has deterministic small Gaussian initialization; B starts at zero. The backbone and output head remain frozen. Training minimizes next-token cross entropy on one 64-token sequence, using float32 SGD at learning rate 0.02 and global adapter gradient clipping at norm 1. No backbone gradient update is claimed.

Execution uses Linux x86_64 CPU, one thread, eager attention, MKLDNN disabled, `ATEN_CPU_CAPABILITY=default`, and `MKL_ENABLE_INSTRUCTIONS=SSE4_2`. [Pinned requirements](llm-requirements.txt) and startup conformance checks constrain supported machines. Matching two tested machines is empirical evidence, not a proof that every CPU will match. Unsupported arithmetic is rejected rather than rounded by a consensus tolerance.

State tensors are bounded shape/base64 little-endian float32 JSON, checked for valid shape and finite values. Model loading uses pinned safetensors and local files with `trust_remote_code=False`. Public training and inference do not deserialize arbitrary pickled models.

## 3. Dataset and validation

The initial source is the Apache-2.0 `HuggingFaceTB/smol-smoltalk` dataset at revision `f73fe857d519ff6ac5af2ea67c4d3834da7b8bcc`. The first 512 source training records are rendered with the pinned model chat template. Text SHA-256 determines document identity and a fixed train/validation split (`int(hash,16) mod 20 == 0` is validation). The resulting immutable snapshot contains 487 training and 25 validation documents.

The execution dataset selects the first 128 eligible training and four eligible validation documents, taking their first 64 tokens. The training sequence for round t is selected by SHA-256 of `neuroshard/adapter-batch/v1:t`, reduced modulo the number of training sequences. This is deterministic, not secret randomness.

Dataset snapshot root: `e5562adf7f4cf8418004e70bb27d68b22e383acf2c0ab4fc68ded430bf6a2bf7`.
Execution dataset SHA-256: `83c85650d10e28515f0129c9b87dee667da6b50abb37e16410e96df1c41cbb08`.

After an accepted update, validators evaluate the new adapter on the four fixed held-out sequences. The training checkpoint always advances for correctly executed work. The serving checkpoint advances only when validation loss is strictly lower than its previous best. This prevents automatic deployment of a measured regression, but four short public sequences can be overfit and do not establish general model quality. No hidden evaluation, broad benchmark improvement, or anti-poisoning theorem is claimed.

## 4. Native accounting and validator lifecycle

One NEURO is 1,000,000 integer atoms. Genesis allocates 90 NEURO: each of four disclosed owners has 2.5 bonded and 20 liquid NEURO. These allocations are explicit; the chain does not claim a zero-premine distribution.

Ordinary accepted transactions burn a fee of 1,000 atoms. Every successful prescribed training task issues exactly 1 NEURO: 0.4 to each worker and 0.2 to native commit verifiers. The profile caps training issuance at 10,000 tasks. There is no automatic schedule reset or perpetual inflation. Maximum genesis plus training issuance is 10,090 NEURO before burns. Inference transfers existing balances and issues nothing.

Native money conservation counts liquid accounts, consensus bonds, the training reservation escrow, deferred verifier rewards and burns. Inference locks remain inside account totals and are not counted as additional money. Every transfer, bond, reserve and ordinary fee payment enforces `available = total − sum(pending inference budgets)`. Native assertions also check checkpoint roots and bounded queues.

Bonding requires an unused Ed25519 consensus key, a possession proof bound to account/chain/nonce/amount, and a multiple of 0.25 NEURO. Activation is scheduled after 60 blocks at a 60-block epoch boundary, with the native validator-update delay included. Unbonding schedules removal; withdrawal waits for both the block and time evidence windows after removal. The LLM profile uses 172,800 blocks and 172,800 seconds. Accepted native equivocation evidence burns one quarter of the remaining bond, rounded up, and schedules removal. Evidence is consumed from CometBFT's verified evidence channel, not accepted merely on an HTTP allegation. The inherited ledger specification details historical voting power and duplicate-evidence handling.

Verifier rewards are deferred until the next native commit identifies actual committers. Rewards are proportional to the stored voting powers among eligible committers; integer remainder or an undistributable budget is burned. Block creation is determined by native stake consensus, not by treating an unverifiable training hash as proof of work.

## 5. Training lifecycle

1. A sponsor signs `reserve` against the current model root and round, naming two public worker keys. The chain permits one training lease at a time and locks a 2 NEURO sponsor bond. The reservation fee is spent.
2. The lease fixes the task ID, parent checkpoint, round, workers and expiry (240 blocks). A worker obtains the finalized task from its own full node and rejects assignments inconsistent with it.
3. Stage 0 computes the frozen hidden features. Its signed receipt binds task, parent checkpoint, input and feature roots.
4. Stage 1 computes adapter gradients and SGD update from the assigned features. Its receipt also binds gradient root, result root and hexadecimal training loss.
5. The sponsor submits both receipts and the result commitment. Validators independently run the complete frozen forward pass, adapter backward pass, optimizer and fixed validation evaluation. Both receipts must agree exactly with replay.
6. Acceptance advances training state, returns the reservation bond and distributes the fixed issuance. Serving state promotes only under the validation rule above. The submission fee is spent.
7. A missed deadline burns the reservation bond and releases the lease. No training reward is issued for a failed or invalid task. Honest workers can go unpaid if a sponsor fails to submit; signatures alone are not payment.

Workers durably journal an operation before executing it and persist its result before returning it. A repeated delivery reuses the receipt. Conflicting assignments for an existing operation are rejected. Interrupted computation without a saved result requires a new lease.

The reference sponsor persists its attempt budget before an attempt. Restarting cannot replenish it; after an uncertain run it waits for any existing native lease to settle or expire. Transaction submission uses the exact signed bytes and queries the corresponding native hash after a lost connection. The sponsor's selection policy prefers available public workers over configured operator fallbacks, with arrival order among public workers. This is a service policy, not Sybil-resistant fairness. Anyone may fund another sponsor. Malicious worker selection and failed leases can consume a sponsor's collateral; operators must monitor it.

## 6. Paid inference lifecycle

Inference uses a separate queue of at most 32 jobs, so an outstanding training lease does not block a customer request.

An `infer` body contains `kind`, `chain_id`, `nonce`, `provider`, `model_root`, `request`, `price`, and `expires`. The request has a nonempty prompt of at most 2,048 UTF-8 bytes and 256 model input tokens, and a maximum output of 1–64 tokens. The model root must match the current serving checkpoint at acceptance. The minimum price is 1,000 atoms per requested maximum output token. The signed expiry must be at least two and at most 240 blocks ahead of acceptance. The default client uses 120 blocks.

The owner pays the submission fee; the inference budget becomes unavailable for spending. The job stores its serving adapter snapshot. Later training or promotion cannot change what this customer purchased. Prompts, payment information and responses are public ledger data.

The chosen provider reads the job from its own finalized state, checks the checkpoint root, runs deterministic greedy generation with the pinned tokenizer and EOS behavior, and signs `respond` with the job ID and result root. Validators regenerate the same answer before accepting it. A provider with zero starting NEURO can respond because the response has no separate fee; it is authorized by an already funded job.

On success, 80% of the budget goes to the provider and 20% to deferred commit verifiers. The price covers the requested token limit even when EOS ends the answer early; it is not metered per emitted token. A 32-token request therefore costs 0.033 NEURO including the 0.001 fee, and pays the provider 0.0256 NEURO.

At the first block strictly after expiry, the budget unlocks in full. The submission fee is not refunded. Wrong providers, wrong outputs, expired claims and nonce replay cannot collect. The owner can choose itself as provider, but still pays the verifier share and fee; no inference issuance is created.

The application keeps the most recent 128 completion/expiry records. Older requests require retained block history. The public API does not provide account inclusion proofs or a durable archival inference search service. Native completion receipts commit an output hash, while the replayed output appears in application state; historical reconstruction may require replay through that height.

## 7. Consensus integration and operational bounds

ABCI CheckTx, PrepareProposal, ProcessProposal and FinalizeBlock run the same transition rules. A block contains at most one application transaction. Full replay is cached by immutable task or request ID within a node to avoid recomputing the same pending claim at each ABCI phase. The cache changes performance, not acceptance. SQLite state uses WAL and FULL synchronization; native signing state and block databases must be backed up consistently and never shared by concurrent validator processes.

Public HTTP relays bound message sizes, allowed RPC methods and request rates. Model/download files are hash verified before use. Public peers use native TCP; the launch operator also maintains a private secondary transport between its two hosts. This auxiliary transport is not required for a public participant to connect to the advertised peer.

There is no on-chain arbitrary-code execution, automatic dataset activation, automatic runtime upgrade, GPU tolerance rule, trustless large tensor availability market, or training-derived leader lottery in this version. Changing a consensus-bound execution profile requires an explicitly coordinated new chain or separately specified upgrade. An S3 upload or website deployment cannot change the model's accepted dataset.

## 8. What the current result establishes

The implementation demonstrates a complete native transaction lifecycle: sponsored neural work, deterministic multi-host acceptance, native issuance, earned balances, model promotion, customer spend locks, generated output verification, payment, timeout refunds and stake-based validator lifecycle. It provides a useful experimental baseline on which to test better computation markets and verification.

It does not establish economical verification at LLM scale: each validator performs substantially the same expensive work. Frozen pretrained features can also be cached and reused; receipts prove correct results, not newly expended physical energy. Work incentives, task allocation, evaluation robustness, independent ownership, available stake history and long-term monetary demand remain distinct research and deployment requirements. See [experiments](LLM_EXPERIMENTS.md) and the [research roadmap](RESEARCH_ROADMAP.md).
