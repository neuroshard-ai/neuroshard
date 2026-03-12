# neuroshard/core/__init__.py
"""
NeuroShard Core Module

Organized into subpackages:
- swarm: THE architecture (SwarmEnabledDynamicNode, SwarmRouter, etc.)
- model: Model components (DynamicNeuroNode, DynamicNeuroLLM)
- network: P2P, DHT, NAT traversal
- training: Training coordination, gradient compression
- economics: Token economics, ledger, wallet
- crypto: ECDSA cryptography

Primary entry point:
    from neuroshard.core.swarm import create_swarm_node, SwarmNodeConfig
"""

__all__ = [
    # Swarm (Primary) - See neuroshard.core.swarm for full list
    'SwarmEnabledDynamicNode',
    'SwarmNodeConfig',
    'SwarmComponents',
    'create_swarm_node',
    'SwarmRouter',
    'SwarmHeartbeatService',
    'ActivationBuffer',
    'OutboundBuffer',
    'ComputeEngine',
    'DiLoCoTrainer',
    'RobustAggregator',
    'SpeculativeCheckpointer',
    'SwarmServiceMixin',
    'SwarmLogger',
    # Model
    'DynamicNeuroNode',
    'DynamicLayerPool',
    'DynamicNeuroLLM',
    'create_dynamic_node',
    # Network
    'P2PManager',
    # Training
    'GradientContribution',
    'GradientCompressor',
    'GenesisDataLoader',
    'GlobalTrainingTracker',
    # Economics
    'NEUROLedger',
    # Crypto
    'sign_message',
    'verify_signature',
]


def __getattr__(name):
    """Lazy loading to avoid circular dependencies."""
    # Swarm components
    if name in ('SwarmEnabledDynamicNode', 'SwarmNodeConfig', 'SwarmComponents', 'create_swarm_node',
                'SwarmRouter', 'PeerCandidate', 'SwarmHeartbeatService', 'CapacityBitmask',
                'ActivationBuffer', 'OutboundBuffer', 'ActivationPacket', 'ActivationPriority',
                'ComputeEngine', 'StepOutcome',
                'DiLoCoTrainer', 'DiLoCoConfig', 'OuterOptimizer',
                'RobustAggregator', 'GradientValidator', 'AggregationStrategy', 'AggregationConfig', 'ValidationConfig',
                'SpeculativeCheckpointer', 'CheckpointConfig',
                'SwarmServiceMixin', 'SwarmNodeState',
                'SwarmLogger', 'LogCategory', 'NodeRole', 'get_swarm_logger', 'init_swarm_logger'):
        from neuroshard.core import swarm
        return getattr(swarm, name)
    # Model components  
    elif name in ('DynamicNeuroNode', 'DynamicLayerPool', 'DynamicNeuroLLM', 'create_dynamic_node'):
        from neuroshard.core import model
        return getattr(model, name)
    # Network components
    elif name in ('P2PManager', 'P2PDataManager', 'DHT', 'DHTProtocol', 'DHTService',
                  'NATTraverser', 'NATTraversalManager', 'STUNClient', 'NATType',
                  'ConnectionPool', 'get_channel', 'TensorSerializer'):
        from neuroshard.core import network
        return getattr(network, name)
    # Training components
    elif name in ('GradientContribution', 'GradientCompressor', 'GenesisDataLoader',
                  'CompressionConfig', 'CompressionMethod',
                  'GlobalTrainingTracker', 'TrainingSnapshot', 'GlobalTrainingStats'):
        from neuroshard.core import training
        return getattr(training, name)
    # Economics components
    elif name in ('NEURO_DECIMALS', 'NEURO_TOTAL_SUPPLY', 'VALIDATOR_MIN_STAKE',
                  'VALIDATOR_MIN_MEMORY_MB', 'NEUROLedger', 'PoNWProof', 'ProofType',
                  'Wallet', 'InferenceMarket'):
        from neuroshard.core import economics
        return getattr(economics, name)
    # Crypto components
    elif name in ('sign_message', 'verify_signature', 'generate_keypair', 'is_valid_node_id_format'):
        from neuroshard.core import crypto
        return getattr(crypto, name)
    raise AttributeError(f"module 'neuroshard.core' has no attribute '{name}'")
