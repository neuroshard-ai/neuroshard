"""
NeuroShard Python SDK

High-level Python client for interacting with NeuroShard nodes.

Quick Start:
    from neuroshard import NeuroNode, NEUROLedger
    
    # Connect to local node
    node = NeuroNode("http://localhost:8000", api_token="YOUR_TOKEN")
    
    # Check status
    status = node.get_status()
    print(f"Node: {status.node_id}")
    
    # Run inference
    response = node.inference("What is NeuroShard?", max_tokens=100)
    print(response.text)
    
    # Check balance
    ledger = NEUROLedger(node)
    balance = ledger.get_balance()
    print(f"Balance: {balance.available} NEURO")
"""

from neuroshard.sdk.client import (
    NeuroNode,
    NEUROLedger,
    AsyncNeuroNode,
    AsyncNEUROLedger,
)

from neuroshard.sdk.types import (
    NodeStatus,
    Metrics,
    InferenceResponse,
    InferenceChunk,
    PeerInfo,
    LayerInfo,
    Balance,
    Transaction,
    StakeInfo,
    RewardSummary,
    TrainingStatus,
    ResourceStatus,
    NetworkMetrics,
    InferenceMetrics,
    TrainingMetrics,
    RewardMetrics,
)

from neuroshard.sdk.errors import (
    NeuroShardError,
    AuthenticationError,
    InsufficientBalanceError,
    RateLimitError,
    NodeOfflineError,
    InvalidRequestError,
)

__all__ = [
    # Clients
    "NeuroNode",
    "NEUROLedger",
    "AsyncNeuroNode",
    "AsyncNEUROLedger",
    # Types
    "NodeStatus",
    "Metrics",
    "InferenceResponse",
    "InferenceChunk",
    "PeerInfo",
    "LayerInfo",
    "Balance",
    "Transaction",
    "StakeInfo",
    "RewardSummary",
    "TrainingStatus",
    "ResourceStatus",
    "NetworkMetrics",
    "InferenceMetrics",
    "TrainingMetrics",
    "RewardMetrics",
    # Errors
    "NeuroShardError",
    "AuthenticationError",
    "InsufficientBalanceError",
    "RateLimitError",
    "NodeOfflineError",
    "InvalidRequestError",
]

