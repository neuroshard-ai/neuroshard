# Active NEPs

This page lists all currently active and pending NeuroShard Enhancement Proposals.

## Status Overview

| Status | Count | Description |
|--------|-------|-------------|
| 🟢 **ACTIVE** | 0 | Currently enforced |
| 🟡 **VOTING** | 0 | Open for voting |
| 🔵 **SCHEDULED** | 0 | Approved, waiting activation |
| ⚪ **DRAFT** | 0 | In preparation |

*Updated automatically from the network.*

---

## 🟢 Active NEPs

*No NEPs are currently active. The protocol is running on base version 1.0.0.*

When NEPs are activated, they will appear here with:
- Implementation date
- Economic impact summary
- Version bump details

---

## 🟡 Currently Voting

*No proposals are currently in the voting phase.*

When a proposal enters voting, you can:
1. Review the specification
2. Understand the economic impact
3. Cast your vote (if you have staked NEURO)

---

## 🔵 Scheduled for Activation

*No proposals are currently scheduled.*

Approved proposals appear here with:
- Activation block number
- Days until activation
- Upgrade instructions

---

## ⚪ Draft Proposals

*No proposals are currently in draft phase.*

Draft proposals are being prepared and discussed before formal voting.

---

## Proposed NEPs (Not Yet Submitted)

The following improvements are being discussed by the community:

### NEP-DRAFT-001: Multi-Token Prediction (MTP)

**Type:** Training  
**Status:** Community Discussion

**Summary:** Enable models to predict multiple future tokens per forward pass, extracting 2-3x more training signal.

**Economic Impact:**
- Training efficiency: +100%
- Per-batch reward: No change
- Net effect: Faster model improvement

**Discussion:** [Discord #governance](https://discord.gg/neuroshard)

---

### NEP-DRAFT-002: Multi-Head Latent Attention (MLA)

**Type:** Architecture  
**Status:** Technical Review

**Summary:** Compress KV cache by 20x using low-rank projection, enabling longer context on consumer hardware.

**Economic Impact:**
- Memory requirement: -2GB
- Inference speed: +30%
- Net effect: More nodes can participate

**Discussion:** [Discord #governance](https://discord.gg/neuroshard)

---

### NEP-DRAFT-003: Auxiliary-Loss-Free MoE Balancing

**Type:** Training  
**Status:** Research

**Summary:** Dynamic bias terms for MoE expert routing without auxiliary losses.

**Economic Impact:**
- Training stability: Improved
- Convergence: Faster
- Net effect: Better model quality

**Discussion:** [Discord #governance](https://discord.gg/neuroshard)

---

## How to Track Proposals

### Subscribe to Updates

```bash
# CLI notifications
neuroshard governance watch --notify email

# Or via API
curl https://api.neuroshard.com/governance/subscribe \
  -d '{"email": "you@example.com"}'
```

### Query Programmatically

```python
from neuroshard.core.governance import NEPRegistry, NEPStatus

registry = NEPRegistry()

# Get all voting proposals
voting = registry.list_proposals(status=NEPStatus.VOTING)
for nep in voting:
    print(f"{nep.nep_id}: {nep.title}")
    print(f"  Approval: {nep.approval_threshold:.0%}")
    print(f"  Ends: {nep.voting_end}")

# Get active NEPs
active = registry.list_proposals(status=NEPStatus.ACTIVE)
```

### Via Website

Visit [neuroshard.com/governance](https://neuroshard.com/governance) for:
- Real-time proposal status
- Vote casting interface
- Historical NEP archive

---

## Historical Archive

All past proposals (approved and rejected) are archived for transparency.

| NEP | Title | Status | Date |
|-----|-------|--------|------|
| *None yet* | *Protocol launched with base version* | — | — |

*This table will populate as proposals are processed.*

---

## Submit Your Own

Have an idea to improve NeuroShard? 

1. Discuss on [Discord #governance](https://discord.gg/neuroshard)
2. Write a proposal following the [proposal guide](/governance/proposals)
3. Submit with 100+ NEURO staked
4. Engage with community feedback

See [How to Create a Proposal](/governance/proposals) for detailed instructions.
