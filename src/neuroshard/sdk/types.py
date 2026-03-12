"""
NeuroShard SDK Type Definitions

Data classes for API responses and structured data.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
from datetime import datetime, date


# ============================================================================
# STATUS TYPES
# ============================================================================

@dataclass
class TrainingStatus:
    """Training progress information."""
    enabled: bool = False
    epoch: int = 0
    step: int = 0
    loss: float = 0.0


@dataclass
class ResourceStatus:
    """System resource usage."""
    gpu_memory_used: int = 0
    gpu_memory_total: int = 0
    cpu_percent: float = 0.0
    ram_used: int = 0
    ram_total: int = 0


@dataclass
class NodeStatus:
    """Complete node status information."""
    node_id: str
    version: str
    uptime_seconds: int
    status: str  # "running", "syncing", "training", "error"
    role: str  # "driver", "worker", "validator", "full"
    layers: List[int]
    peer_count: int
    training: TrainingStatus
    resources: ResourceStatus
    has_embedding: bool = False
    has_lm_head: bool = False


# ============================================================================
# METRICS TYPES
# ============================================================================

@dataclass
class InferenceMetrics:
    """Inference performance metrics."""
    requests_total: int = 0
    requests_per_minute: float = 0.0
    avg_latency_ms: float = 0.0
    p99_latency_ms: float = 0.0
    tokens_generated: int = 0


@dataclass
class TrainingMetrics:
    """Training performance metrics."""
    steps_total: int = 0
    steps_per_hour: float = 0.0
    gradients_submitted: int = 0
    gradients_accepted: int = 0


@dataclass
class NetworkMetrics:
    """Network usage metrics."""
    bytes_sent: int = 0
    bytes_received: int = 0
    active_connections: int = 0
    rpc_calls: int = 0
    peer_count: int = 0


@dataclass
class RewardMetrics:
    """Reward earning metrics."""
    earned_today: float = 0.0
    earned_total: float = 0.0
    pending: float = 0.0


@dataclass
class Metrics:
    """Complete node metrics."""
    timestamp: str
    inference: InferenceMetrics
    training: TrainingMetrics
    network: NetworkMetrics
    rewards: RewardMetrics


# ============================================================================
# INFERENCE TYPES
# ============================================================================

@dataclass
class TokenUsage:
    """Token usage for an inference request."""
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


@dataclass
class Cost:
    """Cost information for an operation."""
    amount: float = 0.0
    currency: str = "NEURO"


@dataclass
class Timing:
    """Timing breakdown for an inference request."""
    queue_ms: float = 0.0
    inference_ms: float = 0.0
    total_ms: float = 0.0


@dataclass
class InferenceResponse:
    """Response from an inference request."""
    id: str
    text: str
    tokens_generated: int
    finish_reason: str  # "stop", "length", "error"
    usage: TokenUsage
    cost: Cost
    timing: Timing


@dataclass
class InferenceChunk:
    """Single chunk from a streaming inference response."""
    token: str
    index: int
    logprob: Optional[float] = None


# ============================================================================
# NETWORK TYPES
# ============================================================================

@dataclass
class PeerInfo:
    """Information about a connected peer."""
    id: str
    address: str
    role: str  # "driver", "worker", "validator"
    layers: List[int]
    latency_ms: float = 0.0
    connected_since: Optional[str] = None


@dataclass
class LayerInfo:
    """Information about a model layer."""
    index: int
    type: str  # "transformer", "embedding", "lm_head"
    memory_mb: int = 0
    status: str = "active"  # "active", "loading", "error"


# ============================================================================
# WALLET TYPES
# ============================================================================

@dataclass
class Balance:
    """Wallet balance information."""
    address: str
    available: float = 0.0
    staked: float = 0.0
    pending: float = 0.0
    total: float = 0.0


@dataclass
class Transaction:
    """A NEURO token transaction."""
    id: str
    from_address: str
    to_address: str
    amount: float
    fee: float
    status: str  # "pending", "confirmed", "failed"
    timestamp: datetime
    type: str = "transfer"  # "transfer", "reward", "stake", "unstake"
    memo: Optional[str] = None


@dataclass
class StakeInfo:
    """Staking information."""
    amount: float = 0.0
    duration_days: int = 0
    start_date: Optional[date] = None
    unlock_date: Optional[date] = None
    multiplier: float = 1.0
    pending_unstake: float = 0.0


@dataclass
class StakeResult:
    """Result of a stake operation."""
    amount: float
    duration_days: int
    start_date: date
    unlock_date: date
    multiplier: float


@dataclass
class UnstakeResult:
    """Result of an unstake request."""
    amount: float
    cooldown_days: int
    available_date: date


@dataclass
class DailyReward:
    """Daily reward breakdown."""
    date: date
    amount: float
    proofs: int = 0


@dataclass
class RewardSummary:
    """Summary of rewards."""
    total: float = 0.0
    by_day: List[DailyReward] = field(default_factory=list)
    by_type: Dict[str, float] = field(default_factory=dict)


# ============================================================================
# CONFIGURATION TYPES
# ============================================================================

@dataclass
class NodeConfig:
    """Node configuration."""
    node_id: str
    port: int
    grpc_port: int
    tracker_url: str
    training: Dict[str, Any] = field(default_factory=dict)
    resources: Dict[str, Any] = field(default_factory=dict)


# ============================================================================
# PROOF TYPES
# ============================================================================

@dataclass
class ProofInfo:
    """Information about a PoNW proof."""
    id: str
    type: str  # "forward", "training", "uptime"
    layer: Optional[int] = None
    timestamp: Optional[str] = None
    verified: bool = False
    reward: float = 0.0


@dataclass
class ProofStats:
    """Statistics about proofs."""
    total_submitted: int = 0
    total_verified: int = 0
    total_reward: float = 0.0

