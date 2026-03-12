# Governance Overview

NeuroShard uses a **decentralized governance system** to manage protocol upgrades. Any changes to the LLM architecture, training algorithms, or economics must go through a formal proposal and voting process.

## Why Governance Matters

In a decentralized AI network, the **model and economics are tightly coupled**:

| Component | Economic Impact |
|-----------|-----------------|
| Training algorithm | Determines reward efficiency |
| Model architecture | Affects hardware requirements |
| Inference speed | Impacts market pricing |
| Layer distribution | Influences node earnings |

Changing one component without adjusting others can:
- Inflate or deflate NEURO earnings unfairly
- Exclude nodes that don't meet new requirements
- Break verification mechanisms

**Governance ensures all stakeholders have a voice in these decisions.**

## Core Principles

### 1. Transparency
All proposed changes are public. Anyone can review the technical specification and economic impact before voting.

### 2. Economic Parity
Every proposal must include an **Economic Impact Analysis** that quantifies how earnings change.

### 3. Stake-Weighted Voting
Voting power is proportional to staked NEURO. Those with skin in the game make decisions.

### 4. Grace Periods
Approved changes include upgrade windows so nodes have time to adapt.

## The NEP Process

**NEP** = NeuroShard Enhancement Proposal

```
┌──────────────────────────────────────────────────────────────────┐
│                        NEP LIFECYCLE                              │
├──────────────────────────────────────────────────────────────────┤
│                                                                  │
│   ┌─────┐    ┌────────┐    ┌────────┐    ┌──────────┐    ┌──────┐│
│   │DRAFT│───►│ REVIEW │───►│ VOTING │───►│SCHEDULED │───►│ACTIVE││
│   └─────┘    └────────┘    └────────┘    └──────────┘    └──────┘│
│      │           │             │              │              │    │
│      │           │             │              │              │    │
│   Author      7 days        7 days      Activation       Applied │
│   submits    technical    stake-weighted    block          to     │
│              review          vote           set          network  │
│                                                                   │
└───────────────────────────────────────────────────────────────────┘
```

## NEP Types

| Type | Code | Description |
|------|------|-------------|
| **Architecture** | `NEP-ARCH` | Model changes (attention, layers, embeddings) |
| **Economics** | `NEP-ECON` | Reward rates, fees, staking parameters |
| **Training** | `NEP-TRAIN` | DiLoCo params, gradient handling, aggregation |
| **Network** | `NEP-NET` | P2P protocol, gossip, routing |
| **Governance** | `NEP-GOV` | Changes to governance itself |
| **Emergency** | `NEP-EMERG` | Critical security patches (fast-track) |

## Quick Links

- [How to Create a Proposal](/governance/proposals)
- [Voting Guide](/governance/voting)
- [Current Active NEPs](/governance/active)
- [Protocol Versions](/governance/versioning)

## Governance at a Glance

```
┌─────────────────────────────────────────────────────────────┐
│                    GOVERNANCE FLOW                          │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│   ┌─────────────┐                                           │
│   │  PROPOSER   │  Stake: 100+ NEURO                        │
│   │  (Any Node) │  Fee: 10 NEURO (burned)                   │
│   └──────┬──────┘                                           │
│          │                                                  │
│          ▼                                                  │
│   ┌─────────────┐                                           │
│   │     NEP     │  Title, Motivation, Specification         │
│   │  PROPOSAL   │  Parameter Changes, Economic Impact       │
│   └──────┬──────┘                                           │
│          │                                                  │
│          ▼                                                  │
│   ┌─────────────┐                                           │
│   │   VOTING    │  1 NEURO staked = 1 vote                  │
│   │   (7 days)  │  66% approval, 20% quorum                 │
│   └──────┬──────┘                                           │
│          │                                                  │
│     ┌────┴────┐                                             │
│     ▼         ▼                                             │
│ ┌───────┐ ┌────────┐                                        │
│ │APPROVE│ │ REJECT │                                        │
│ └───┬───┘ └────────┘                                        │
│     │                                                       │
│     ▼                                                       │
│ ┌───────────┐                                               │
│ │ SCHEDULED │  Grace period (7-30 days)                     │
│ │           │  Nodes upgrade                                │
│ └─────┬─────┘                                               │
│       │                                                     │
│       ▼                                                     │
│ ┌───────────┐                                               │
│ │  ACTIVE   │  New parameters enforced                      │
│ │           │  Protocol version bumped                      │
│ └───────────┘                                               │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

## Example: Adding Multi-Token Prediction

Here's how a major training change would be proposed:

```python
nep = create_proposal(
    title="Add Multi-Token Prediction Training",
    nep_type=NEPType.TRAINING,
    
    economic_impact=EconomicImpact(
        training_efficiency_multiplier=2.0,  # 2x faster training
        training_reward_multiplier=1.0,      # Same reward per batch
        net_earnings_change_percent=0.0,     # Neutral (quality gains)
    ),
    
    upgrade_path=UpgradePath(
        grace_period_days=14,
        backward_compatible=True,
    ),
)
```

The economic impact shows:
- Training becomes **2x more efficient**
- Per-batch rewards **stay the same**
- Net effect: Model improves faster, making tokens more valuable

This ensures miners aren't suddenly earning half as much while the network benefits from faster progress.
