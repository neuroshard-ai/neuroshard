"""
NeuroShard - Decentralized AI Training Network

Train LLMs together, earn NEURO tokens.

Quick Start:
    # Install
    pip install neuroshard-ai

    # Run a node
    neuroshard --port 8000 --token YOUR_TOKEN

    # Open dashboard
    # http://localhost:8000/

SDK Usage:
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

For more information:
    https://neuroshard.com
    https://docs.neuroshard.com
"""

from neuroshard.version import __version__

# SDK Exports
from neuroshard.sdk import (
    # Clients
    NeuroNode,
    NEUROLedger,
    AsyncNeuroNode,
    AsyncNEUROLedger,
    # Types
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
    # Errors
    NeuroShardError,
    AuthenticationError,
    InsufficientBalanceError,
    RateLimitError,
    NodeOfflineError,
)

__all__ = [
    "__version__",
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
    # Errors
    "NeuroShardError",
    "AuthenticationError",
    "InsufficientBalanceError",
    "RateLimitError",
    "NodeOfflineError",
]

