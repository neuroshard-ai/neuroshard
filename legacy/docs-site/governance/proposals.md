# Creating Proposals

This guide explains how to create and submit a NeuroShard Enhancement Proposal (NEP).

## Requirements to Propose

Before you can submit a proposal:

| Requirement | Amount | Purpose |
|-------------|--------|---------|
| **Minimum Stake** | 100 NEURO | Ensures proposer has skin in the game |
| **Proposal Fee** | 10 NEURO | Held in escrow (refunded if approved, burned if spam) |

## Proposer Rewards (Stake-Proportional)

Rewards scale with how much stake participates in voting—more impactful proposals earn more:

| Outcome | Reward Formula | Example (100K stake voted) |
|---------|----------------|---------------------------|
| ✅ **Approved** | 0.1% of total stake voted + fee refund | ~110 NEURO |
| 🟡 **Rejected w/ quorum** | 0.01% of total stake voted | ~10 NEURO |
| ❌ **Rejected, no quorum** | Fee burned | 0 NEURO |

**Caps:** Min 1 NEURO, Max 1000 NEURO (prevents gaming)

::: tip Why Stake-Proportional?
This aligns incentives—proposals that engage the community more (more stake voting) earn bigger rewards. It encourages proposers to write quality proposals that matter to the network.
:::

## NEP Structure

Every proposal must include these components:

### 1. Metadata

```python
{
    "nep_id": "NEP-001",          # Assigned on submission
    "title": "Short descriptive title",
    "nep_type": "arch",           # arch, econ, train, net, gov, emergency
    "author_node_id": "abc123...",
    "created_at": 1704067200,
}
```

### 2. Content

| Field | Description |
|-------|-------------|
| **Abstract** | 1-2 sentence summary |
| **Motivation** | Why is this change needed? What problem does it solve? |
| **Specification** | Technical details, code changes, parameter modifications |

### 3. Economic Impact Analysis

**This is the most important part.** Every proposal must quantify its economic effects:

```python
economic_impact = EconomicImpact(
    # Training effects
    training_reward_multiplier=1.0,      # 1.0 = no change
    training_efficiency_multiplier=1.0,  # How much more efficient
    
    # Inference effects
    inference_reward_multiplier=1.0,
    inference_cost_multiplier=1.0,
    
    # Hardware requirements
    min_memory_change_mb=0,              # +/- memory needed
    min_compute_change_tflops=0.0,
    
    # Net effect (positive = nodes earn more)
    net_earnings_change_percent=0.0,
)
```

### 4. Parameter Changes

Specify exactly what constants/configs will change:

```python
parameter_changes = [
    ParameterChange(
        module="economics.constants",
        parameter="TRAINING_REWARD_PER_BATCH",
        old_value=0.0005,
        new_value=0.0003,
        rationale="MTP extracts 2x signal, halving reward maintains parity"
    ),
]
```

### 5. Upgrade Path

How will nodes transition?

```python
upgrade_path = UpgradePath(
    min_version="1.0.0",           # Minimum version that can upgrade
    target_version="1.1.0",        # Version after upgrade
    grace_period_days=14,          # Days to upgrade
    backward_compatible=True,      # Can old nodes still participate?
    requires_checkpoint_reload=False,
    migration_steps=[
        "Download new node version",
        "Restart node with --upgrade flag",
        "Wait for gradient sync",
    ],
)
```

## Submitting a Proposal

### Via CLI (Recommended)

```bash
# Install the CLI
pip install neuroshard-ai

# Create a proposal interactively
neuroshard-governance propose \
  --title "Add Multi-Token Prediction Training" \
  --type train \
  --abstract "Enable MTP for 2-3x training efficiency"

# Create from JSON file (for complex proposals)
neuroshard-governance propose \
  --title "Add MTP Training" \
  --type train \
  --file my_proposal.json
```

The CLI will:
1. Show you a preview of your proposal
2. Display the fee and potential rewards
3. Ask for confirmation before submitting

### Via Python SDK

```python
from neuroshard.core.governance import (
    create_proposal,
    NEPType,
    EconomicImpact,
    ParameterChange,
)

# Create the proposal
nep = create_proposal(
    title="Add Multi-Token Prediction Training",
    nep_type=NEPType.TRAINING,
    abstract="Enable MTP for 2-3x training efficiency",
    motivation="""
    Current single-token prediction wastes training signal.
    MTP predicts multiple future tokens, extracting more value
    from each forward pass.
    """,
    specification="""
    ## Changes
    1. Add MTP heads to NeuroLLM
    2. Modify loss function: loss = main_loss + 0.3 * mtp_loss
    3. Update verification to accept mtp_enabled proofs
    """,
    author_node_id=node.node_id,
    economic_impact=EconomicImpact(
        training_efficiency_multiplier=2.0,
        net_earnings_change_percent=0.0,
    ),
)

# Submit to registry
from neuroshard.core.governance import NEPRegistry
registry = NEPRegistry(ledger=node.ledger)

success, message, nep_id = registry.submit_proposal(
    nep=nep,
    node_id=node.node_id,
    signature=node.sign(nep.content_hash),
)

print(f"Submitted: {nep_id}")  # e.g., "NEP-001"
```

### Via Website

1. Go to [neuroshard.com/governance](https://neuroshard.com/governance)
2. Log in (must have 100+ NEURO staked)
3. Click "Create Proposal" tab
4. Follow instructions to use CLI or SDK

## Best Practices

### DO ✅

- **Explain the "why"** clearly in motivation
- **Be specific** about parameter changes
- **Show economic neutrality** when possible
- **Provide migration steps** for complex changes
- **Link to research** if applicable (papers, benchmarks)

### DON'T ❌

- Submit incomplete proposals
- Hide economic impacts
- Propose changes without testing
- Rush emergency proposals for non-emergencies
- Propose changes that benefit only proposer

## Proposal Templates

### Architecture Change (NEP-ARCH)

```markdown
# Title: Replace Standard Attention with MLA

## Abstract
Compress KV cache by 20x using low-rank projection.

## Motivation
Consumer GPUs have limited memory. MLA enables longer context.

## Specification
1. Replace MultiHeadAttention with MultiHeadLatentAttention
2. KV compression: hidden_dim → 512 → num_heads * head_dim
3. Cache only compressed latent

## Economic Impact
- Memory requirement: -2000 MB
- Inference speed: +30%
- Net earnings: +5% (more nodes can participate)

## Upgrade Path
- Breaking change: Requires checkpoint migration
- Grace period: 14 days
- Migration: Download new checkpoint, restart node
```

### Economic Change (NEP-ECON)

```markdown
# Title: Adjust Training Reward Rate

## Abstract
Increase training reward to incentivize participation.

## Motivation
Current reward rate doesn't cover electricity costs for small nodes.

## Specification
- OLD: TRAINING_REWARD_PER_BATCH = 0.0005
- NEW: TRAINING_REWARD_PER_BATCH = 0.0007

## Economic Impact
- Training rewards: +40%
- Net earnings: +25% (training is 60% of typical earnings)
- Inflation impact: +2% annual supply increase

## Upgrade Path
- Non-breaking: Old nodes earn at new rate automatically
- No migration needed
```

## After Submission

Once submitted, your proposal enters the **REVIEW** phase:

1. **Technical Review** (7 days)
   - Community discusses on Discord/GitHub
   - Experts review specification
   - Author can make minor edits

2. **Voting** (7 days)
   - Stake-weighted voting begins
   - 66% approval required
   - 20% quorum required

3. **Outcome**
   - **Approved**: Scheduled for activation
   - **Rejected**: Can resubmit after 30 days with changes

See [Voting Guide](/governance/voting) for details on the voting process.
