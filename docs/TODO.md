> Historical prototype document. For the supported native release, see [PUBLIC_TESTNET.md](PUBLIC_TESTNET.md) and [PROTOCOL_CANDIDATE_V2.md](PROTOCOL_CANDIDATE_V2.md).

# NeuroShard v2.1 Implementation TODO

> Implementation tracking for the fully dynamic architecture.  
> See [ARCHITECTURE_V2.md](./ARCHITECTURE_V2.md) for full design.

---

## Overview

**Goal**: Fully dynamic, decentralized training at any scale with any hardware.

**Core Innovations**:
- Quorum-based training (speed-matched node groups)
- Multi-mode contribution (pipeline, async, data, verify)
- Adaptive protocol (1 to 1M+ nodes, same code)
- Governance-based vocabulary (no coordinator attack)
- Sqrt-weighted gradient aggregation

**Estimated Timeline**: 10 weeks

---

## Phase 1: Foundation (Week 1-2)

### 1.1 Versioning Infrastructure
- [ ] Add `arch_version` and `vocab_version` to DynamicLayerPool
- [ ] Store/load network state from DHT
- [ ] Implement version compatibility checks

### 1.2 Speed Tier System
- [ ] Define SpeedTier enum (T1-T5)
- [ ] Implement benchmark_self() function
- [ ] Add speed tier thresholds

### 1.3 Contribution Modes
- [ ] Define ContributionMode enum (pipeline, async, data, verify, inference, idle)
- [ ] Implement select_contribution_mode() logic
- [ ] Genesis/solo mode (only for n=1)

### 1.4 Protocol Updates
- [ ] Add versioning to protobuf ActivationPacket
- [ ] Add quorum_id and speed_tier fields
- [ ] Regenerate Python bindings

---

## Phase 2: Quorum System (Week 2-3)

### 2.1 Quorum Formation
- [ ] Define Quorum and QuorumProposal dataclasses
- [ ] Implement form_quorum() algorithm using DHT
- [ ] Implement proposal/acceptance protocol

### 2.2 Quorum Lifecycle
- [ ] Session management (start, renew, end)
- [ ] Health monitoring and member replacement
- [ ] Graceful dissolution
- [ ] DHT registration

### 2.3 Within-Quorum Pipeline
- [ ] Implement QuorumTrainer class
- [ ] Initiator, processor, finisher logic
- [ ] Activation/gradient packet routing

---

## Phase 3: Cross-Quorum Sync (Week 3-4)

### 3.1 Cohort Discovery
- [ ] Implement find_layer_cohort() across all quorums
- [ ] Include async contributors in cohort

### 3.2 Weighted Aggregation
- [ ] Implement weighted_robust_aggregate() with sqrt-batch weighting
- [ ] Add freshness decay (1.0 to 0.3 based on age)
- [ ] Minimum weight ensures all contribute

### 3.3 Robust Methods
- [ ] Trimmed mean (default, trim 20%)
- [ ] Coordinate-wise median
- [ ] Krum selection

### 3.4 Async Contributor Flow
- [ ] Weight download from peers
- [ ] Local training loop
- [ ] Pseudo-gradient submission to cohort

---

## Phase 4: Layer Growth (Week 4-5)

### 4.1 Adaptive Triggers
- [ ] check_layer_growth() with adaptive threshold
- [ ] Coverage and stability checks

### 4.2 Layer Addition Sequence
- [ ] Announcement, grace period, activation, reformation
- [ ] Handle failed upgrades

### 4.3 Initialization
- [ ] Identity initialization (output approx input)
- [ ] Warmup LR schedule (0.1x during warmup)

---

## Phase 5: Vocabulary Governance (Week 5-6)

### 5.1 Proposal System
- [ ] VocabProposal dataclass
- [ ] Validation (stake, frequency, no sensitive patterns)

### 5.2 Voting
- [ ] Discussion period (7 days)
- [ ] Voting period (7 days, stake-weighted)
- [ ] 66% approval, 30% quorum

### 5.3 Implementation
- [ ] Grace period (7 days)
- [ ] Embedding/LM head expansion
- [ ] Emergency rollback procedure

---

## Phase 6: Inference and Rewards (Week 6-7)

### 6.1 Inference Routing
- [ ] Quorum discovery with latency/price/reputation
- [ ] Dynamic pricing: base x demand x speed x reputation

### 6.2 Payment
- [ ] Escrow locking
- [ ] Proportional distribution
- [ ] Refund on failure

### 6.3 Training Rewards
- [ ] Base rate x batches x layers x multipliers
- [ ] Scarcity, position, stake, reputation bonuses

---

## Phase 7: Adversarial Resistance (Week 7-8)

### 7.1 PoNW Proofs
- [ ] PipelineProof and CohortSyncProof generation
- [ ] Merkle commitment for gradients

### 7.2 Optimistic Verification
- [ ] Challenge window (10 min)
- [ ] Auto-acceptance after window
- [ ] Adaptive challenge probability

### 7.3 Challenge Protocol
- [ ] Verification data request
- [ ] Recomputation and comparison
- [ ] Slash on fraud (2x stake)

### 7.4 Cross-Quorum Audit
- [ ] Random proof selection from other quorums
- [ ] Recompute and verify

---

## Phase 8: Testing (Week 8-10)

### 8.1 Unit Tests
- [ ] All new functions tested
- [ ] Edge cases covered

### 8.2 Integration Tests
- [ ] Multi-quorum training
- [ ] Layer growth with active quorums
- [ ] Vocab upgrade end-to-end

### 8.3 Adversarial Simulation
- [ ] Lazy, slow, poisoning nodes
- [ ] Collusion attempts

### 8.4 Scale Testing
- [ ] 10, 100, 1000 simulated nodes

---

## New Files

| File | Purpose |
|------|---------|
| `core/network/tiers.py` | Speed tier system |
| `core/network/quorum.py` | Quorum formation/lifecycle |
| `core/training/async_contributor.py` | Async gradient flow |
| `core/governance/vocab_proposal.py` | Vocabulary governance |
| `core/inference/router.py` | Inference routing/pricing |
| `core/inference/payment.py` | Payment escrow |
| `core/economics/rewards.py` | Training rewards |
| `core/economics/slashing.py` | Stake slashing |
| `core/consensus/ponw.py` | Proof of Neural Work v2 |

---

## Success Criteria

- [ ] Works from 1 to 100+ nodes
- [ ] T5 nodes contribute via async mode
- [ ] Layer growth without training pause
- [ ] Vocab governance end-to-end
- [ ] Adversarial nodes detected and slashed
- [ ] Inference with payment works

---

*Last updated: December 2025*
