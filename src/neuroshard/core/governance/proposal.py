"""
NeuroShard Enhancement Proposal (NEP) System

A NEP is a formal proposal for changing the NeuroShard protocol.
It includes the change specification, economic impact analysis,
and upgrade path for existing nodes.

Inspired by:
- Ethereum EIPs
- Bitcoin BIPs
- On-chain governance in Cosmos/Polkadot
"""

import hashlib
import json
import time
from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import Dict, List, Optional, Any
import logging

logger = logging.getLogger(__name__)


class NEPType(Enum):
    """Categories of protocol changes."""
    
    ARCHITECTURE = "arch"      # Model architecture (MLA, MTP, attention)
    ECONOMICS = "econ"         # Reward rates, fees, staking params
    TRAINING = "train"         # Training algorithms (DiLoCo, aggregation)
    NETWORK = "net"            # P2P, gossip, routing protocols
    GOVERNANCE = "gov"         # Changes to governance itself
    EMERGENCY = "emergency"    # Critical security patches


class NEPStatus(Enum):
    """Lifecycle status of a proposal."""
    
    DRAFT = "draft"            # Being written
    REVIEW = "review"          # Open for technical review
    VOTING = "voting"          # Stake-weighted voting active
    APPROVED = "approved"      # Passed vote threshold
    REJECTED = "rejected"      # Failed vote threshold
    SCHEDULED = "scheduled"    # Waiting for activation height
    ACTIVE = "active"          # Currently enforced
    DEPRECATED = "deprecated"  # Superseded by newer NEP


@dataclass
class EconomicImpact:
    """
    Quantified economic impact of a protocol change.
    
    This is CRITICAL for governance - every change must declare
    how it affects NEURO earnings/costs.
    """
    
    # Training economics
    training_reward_multiplier: float = 1.0  # 1.0 = no change, 2.0 = 2x rewards
    training_efficiency_multiplier: float = 1.0  # How much more efficient training is
    
    # Inference economics  
    inference_reward_multiplier: float = 1.0
    inference_cost_multiplier: float = 1.0
    
    # Hardware requirements
    min_memory_change_mb: int = 0  # +/- memory requirement
    min_compute_change_tflops: float = 0.0
    
    # Staking impacts
    staking_multiplier_change: float = 0.0  # Change to staking formula
    
    # Transition costs
    upgrade_cost_neuro: float = 0.0  # Cost to upgrade (e.g., redownload shards)
    
    # Net effect (positive = nodes earn more, negative = earn less)
    net_earnings_change_percent: float = 0.0
    
    def to_dict(self) -> Dict:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'EconomicImpact':
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})
    
    def is_neutral(self) -> bool:
        """Check if change has no economic impact."""
        return (
            self.training_reward_multiplier == 1.0 and
            self.inference_reward_multiplier == 1.0 and
            self.net_earnings_change_percent == 0.0
        )
    
    def describe(self) -> str:
        """Human-readable impact description."""
        effects = []
        
        if self.training_reward_multiplier != 1.0:
            change = (self.training_reward_multiplier - 1) * 100
            effects.append(f"Training rewards: {change:+.1f}%")
        
        if self.training_efficiency_multiplier != 1.0:
            change = (self.training_efficiency_multiplier - 1) * 100
            effects.append(f"Training efficiency: {change:+.1f}%")
            
        if self.inference_reward_multiplier != 1.0:
            change = (self.inference_reward_multiplier - 1) * 100
            effects.append(f"Inference rewards: {change:+.1f}%")
            
        if self.min_memory_change_mb != 0:
            effects.append(f"Memory requirement: {self.min_memory_change_mb:+d} MB")
            
        if self.net_earnings_change_percent != 0:
            effects.append(f"Net earnings: {self.net_earnings_change_percent:+.1f}%")
        
        return "; ".join(effects) if effects else "No economic impact"


@dataclass
class ParameterChange:
    """
    Specification of a parameter change.
    
    Links to the exact constant/config that will change.
    """
    module: str           # e.g., "economics.constants"
    parameter: str        # e.g., "TRAINING_REWARD_PER_BATCH_PER_LAYER"
    old_value: Any        # Current value
    new_value: Any        # Proposed value
    rationale: str        # Why this change
    
    def to_dict(self) -> Dict:
        return {
            "module": self.module,
            "parameter": self.parameter,
            "old_value": self.old_value,
            "new_value": self.new_value,
            "rationale": self.rationale,
        }


@dataclass
class UpgradePath:
    """
    How nodes should transition to the new protocol.
    """
    
    # Version requirements
    min_version: str = "0.0.0"  # Minimum version that can upgrade
    target_version: str = "0.0.0"  # Version after upgrade
    
    # Timing
    grace_period_days: int = 7  # Days nodes have to upgrade
    activation_delay_blocks: int = 10000  # Blocks after approval before activation
    
    # Compatibility
    backward_compatible: bool = False  # Can old nodes still participate?
    requires_checkpoint_reload: bool = False  # Must reload model checkpoint?
    
    # Migration steps
    migration_steps: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict:
        return asdict(self)


@dataclass
class NEP:
    """
    NeuroShard Enhancement Proposal.
    
    A complete specification for a protocol change including:
    - What changes (technical specification)
    - Why it changes (motivation)
    - Economic impact (how earnings/costs change)
    - Upgrade path (how nodes transition)
    """
    
    # Identity
    nep_id: str = ""              # e.g., "NEP-001" (assigned on creation)
    title: str = ""               # Short descriptive title
    nep_type: NEPType = NEPType.ARCHITECTURE
    status: NEPStatus = NEPStatus.DRAFT
    
    # Authorship
    author_node_id: str = ""      # Node that created this proposal
    created_at: float = 0.0       # Timestamp
    updated_at: float = 0.0
    
    # Content
    abstract: str = ""            # 1-2 sentence summary
    motivation: str = ""          # Why is this needed?
    specification: str = ""       # Technical details (markdown)
    
    # Changes
    parameter_changes: List[ParameterChange] = field(default_factory=list)
    code_changes: Dict[str, str] = field(default_factory=dict)  # file -> diff
    
    # Impact
    economic_impact: EconomicImpact = field(default_factory=EconomicImpact)
    upgrade_path: UpgradePath = field(default_factory=UpgradePath)
    
    # Voting
    voting_start: Optional[float] = None
    voting_end: Optional[float] = None
    approval_threshold: float = 0.66  # 66% stake-weighted approval
    quorum_threshold: float = 0.20    # 20% of staked NEURO must vote
    
    # Activation
    activation_block: Optional[int] = None
    
    # Hash for integrity
    content_hash: str = ""
    signature: str = ""
    
    def __post_init__(self):
        if not self.created_at:
            self.created_at = time.time()
        if not self.updated_at:
            self.updated_at = self.created_at
        if not self.content_hash:
            self.content_hash = self._compute_hash()
    
    def _compute_hash(self) -> str:
        """Compute content hash for integrity verification."""
        content = {
            "title": self.title,
            "nep_type": self.nep_type.value,
            "abstract": self.abstract,
            "motivation": self.motivation,
            "specification": self.specification,
            "parameter_changes": [pc.to_dict() for pc in self.parameter_changes],
            "economic_impact": self.economic_impact.to_dict(),
            "upgrade_path": self.upgrade_path.to_dict(),
        }
        return hashlib.sha256(json.dumps(content, sort_keys=True).encode()).hexdigest()[:32]
    
    def to_dict(self) -> Dict:
        return {
            "nep_id": self.nep_id,
            "title": self.title,
            "nep_type": self.nep_type.value,
            "status": self.status.value,
            "author_node_id": self.author_node_id,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "abstract": self.abstract,
            "motivation": self.motivation,
            "specification": self.specification,
            "parameter_changes": [pc.to_dict() for pc in self.parameter_changes],
            "economic_impact": self.economic_impact.to_dict(),
            "upgrade_path": self.upgrade_path.to_dict(),
            "voting_start": self.voting_start,
            "voting_end": self.voting_end,
            "approval_threshold": self.approval_threshold,
            "quorum_threshold": self.quorum_threshold,
            "activation_block": self.activation_block,
            "content_hash": self.content_hash,
            "signature": self.signature,
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'NEP':
        """Reconstruct NEP from dictionary."""
        nep = cls(
            nep_id=data.get("nep_id", ""),
            title=data.get("title", ""),
            nep_type=NEPType(data.get("nep_type", "arch")),
            status=NEPStatus(data.get("status", "draft")),
            author_node_id=data.get("author_node_id", ""),
            created_at=data.get("created_at", 0.0),
            updated_at=data.get("updated_at", 0.0),
            abstract=data.get("abstract", ""),
            motivation=data.get("motivation", ""),
            specification=data.get("specification", ""),
            parameter_changes=[
                ParameterChange(**pc) for pc in data.get("parameter_changes", [])
            ],
            economic_impact=EconomicImpact.from_dict(
                data.get("economic_impact", {})
            ),
            upgrade_path=UpgradePath(**data.get("upgrade_path", {})),
            voting_start=data.get("voting_start"),
            voting_end=data.get("voting_end"),
            approval_threshold=data.get("approval_threshold", 0.66),
            quorum_threshold=data.get("quorum_threshold", 0.20),
            activation_block=data.get("activation_block"),
            content_hash=data.get("content_hash", ""),
            signature=data.get("signature", ""),
        )
        return nep
    
    def is_voting_active(self) -> bool:
        """Check if voting period is currently active."""
        if self.status != NEPStatus.VOTING:
            return False
        now = time.time()
        return (
            self.voting_start is not None and
            self.voting_end is not None and
            self.voting_start <= now <= self.voting_end
        )
    
    def summary(self) -> str:
        """One-line summary for listings."""
        return f"[{self.nep_id}] {self.title} ({self.status.value})"


def create_proposal(
    title: str,
    nep_type: NEPType,
    abstract: str,
    motivation: str,
    specification: str,
    author_node_id: str,
    parameter_changes: List[ParameterChange] = None,
    economic_impact: EconomicImpact = None,
) -> NEP:
    """
    Create a new NEP proposal.
    
    This is the main entry point for proposing protocol changes.
    
    Example:
        nep = create_proposal(
            title="Add Multi-Token Prediction Training",
            nep_type=NEPType.TRAINING,
            abstract="Enable MTP to extract 2-3x more training signal per batch",
            motivation="Faster convergence, better sample efficiency",
            specification="...",
            author_node_id=node.node_id,
            parameter_changes=[
                ParameterChange(
                    module="economics.constants",
                    parameter="TRAINING_REWARD_PER_BATCH_PER_LAYER",
                    old_value=0.0005,
                    new_value=0.0003,  # Reduced because more efficient
                    rationale="MTP extracts 2x signal, so half reward maintains parity"
                )
            ],
            economic_impact=EconomicImpact(
                training_efficiency_multiplier=2.0,
                training_reward_multiplier=0.6,  # 0.6 * 2.0 = 1.2x net (slight increase)
                net_earnings_change_percent=20.0,
            )
        )
    """
    nep = NEP(
        title=title,
        nep_type=nep_type,
        abstract=abstract,
        motivation=motivation,
        specification=specification,
        author_node_id=author_node_id,
        parameter_changes=parameter_changes or [],
        economic_impact=economic_impact or EconomicImpact(),
    )
    
    # Generate NEP ID (will be formalized when submitted to registry)
    nep.nep_id = f"NEP-DRAFT-{nep.content_hash[:8]}"
    
    logger.info(f"Created proposal: {nep.summary()}")
    logger.info(f"Economic impact: {nep.economic_impact.describe()}")
    
    return nep


# =============================================================================
# EXAMPLE NEPs (for reference)
# =============================================================================

def example_nep_multi_token_prediction() -> NEP:
    """
    Example: Adding Multi-Token Prediction to NeuroShard.
    
    This shows how a major training change would be proposed with
    proper economic impact analysis.
    """
    return create_proposal(
        title="Add Multi-Token Prediction (MTP) Training Objective",
        nep_type=NEPType.TRAINING,
        abstract=(
            "Enable models to predict D additional tokens beyond next-token. "
            "Extracts 2-3x more training signal per sample, accelerating convergence."
        ),
        motivation="""
## Motivation

Current single-token prediction wastes training signal. Each forward pass only
uses the target token for loss, ignoring predictable future tokens.

MTP (from DeepSeek V3) predicts multiple future tokens simultaneously:
- 2-3x denser training signal
- Faster convergence (fewer epochs to same loss)
- Enables speculative decoding at inference (1.8x faster)

## Economic Consideration

If MTP makes training 2x more efficient, nodes do the same work but extract
more value. We have two options:

A) Keep rewards same → Model improves faster → Tokens become more valuable
B) Reduce per-batch reward → Same earnings rate → Faster model improvement

This NEP proposes option A: Let efficiency gains flow to model quality,
which increases inference demand, which increases token value naturally.
        """,
        specification="""
## Specification

### 1. Model Architecture Changes

Add MTP heads to NeuroLLM:
- 1 additional prediction head per MTP depth (D=1 recommended)
- Each head shares base transformer, adds projection layer
- ~2% parameter increase

### 2. Training Changes

Modify loss function:
```python
loss = main_ce_loss + mtp_weight * sum(mtp_losses)
```

MTP weight schedule:
- 0.3 for first 70% of training
- 0.1 for remaining 30%

### 3. Verification Changes

Update ProofVerifier:
- Accept proofs with `mtp_enabled=True`
- Same batch count (MTP is internal to forward pass)
- No change to training rate limits

### 4. Backward Compatibility

- Nodes without MTP can still participate
- MTP nodes produce compatible gradients (main loss only for aggregation)
- Grace period: 30 days before MTP becomes required
        """,
        author_node_id="example_node_id",
        parameter_changes=[
            ParameterChange(
                module="model.llm",
                parameter="MTP_ENABLED",
                old_value=False,
                new_value=True,
                rationale="Enable multi-token prediction training objective"
            ),
            ParameterChange(
                module="model.llm", 
                parameter="MTP_DEPTH",
                old_value=0,
                new_value=1,
                rationale="Predict 1 additional token (D=1)"
            ),
            ParameterChange(
                module="model.llm",
                parameter="MTP_LOSS_WEIGHT",
                old_value=0.0,
                new_value=0.3,
                rationale="30% weight for MTP loss"
            ),
        ],
        economic_impact=EconomicImpact(
            training_efficiency_multiplier=2.0,  # 2x more signal per batch
            training_reward_multiplier=1.0,      # No change to per-batch reward
            inference_reward_multiplier=1.0,     # No change
            min_memory_change_mb=200,            # ~200MB for MTP heads
            net_earnings_change_percent=0.0,     # Neutral (efficiency → quality)
        ),
    )


def example_nep_mla_attention() -> NEP:
    """
    Example: Adding Multi-Head Latent Attention to reduce memory.
    """
    return create_proposal(
        title="Replace Standard Attention with Multi-Head Latent Attention (MLA)",
        nep_type=NEPType.ARCHITECTURE,
        abstract=(
            "Compress KV cache by 20x using low-rank projection. "
            "Enables longer context and lower memory requirements."
        ),
        motivation="""
## Motivation

Consumer GPUs have limited memory. Current attention mechanism requires
storing full KV cache, limiting context length and excluding low-memory nodes.

MLA (from DeepSeek V3) compresses KV to 512-dim latent space:
- 20x smaller KV cache
- 8GB GPU can handle 128K context (vs 6K currently)
- More nodes can participate
- Faster inference (less memory bandwidth)

## Economic Consideration

This EXPANDS the network by lowering hardware requirements.
More nodes = more decentralization = stronger network.

Low-memory nodes gain ability to participate → increases supply
But demand also increases as model supports longer context.
Net effect: slightly positive for all participants.
        """,
        specification="""
## Specification

### 1. Architecture Changes

Replace MultiHeadAttention with MultiHeadLatentAttention:
- KV down-projection: hidden_dim → 512
- KV up-projection: 512 → num_heads * head_dim
- Cache only compressed latent (not full KV)

### 2. Checkpoint Compatibility

- New checkpoints include MLA weights
- Old checkpoints can be migrated (initialize projections to identity-like)
- Migration takes ~5 minutes on consumer hardware

### 3. Verification Changes

None - MLA is internal to forward pass.
Same tokens in, same tokens out.

### 4. Backward Compatibility

BREAKING CHANGE:
- Old attention and MLA are not compatible
- Hard fork at activation block
- 14-day grace period for upgrades
        """,
        author_node_id="example_node_id",
        parameter_changes=[
            ParameterChange(
                module="model.llm",
                parameter="ATTENTION_TYPE",
                old_value="standard",
                new_value="mla",
                rationale="Use Multi-Head Latent Attention"
            ),
            ParameterChange(
                module="model.llm",
                parameter="KV_LORA_RANK",
                old_value=0,
                new_value=512,
                rationale="KV compression dimension"
            ),
        ],
        economic_impact=EconomicImpact(
            training_efficiency_multiplier=1.0,
            training_reward_multiplier=1.0,
            inference_reward_multiplier=1.0,
            min_memory_change_mb=-2000,  # 2GB LESS memory needed
            requires_checkpoint_reload=True,
            net_earnings_change_percent=5.0,  # Slightly positive from efficiency
        ),
    )
