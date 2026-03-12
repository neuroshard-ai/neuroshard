# neuroshard/core/swarm/__init__.py
"""
Swarm Architecture

This package contains all swarm-related components:
- factory: SwarmEnabledDynamicNode, create_swarm_node
- router: SwarmRouter, PeerCandidate
- heartbeat: SwarmHeartbeatService, CapacityBitmask
- buffers: ActivationBuffer, OutboundBuffer
- compute: ComputeEngine
- diloco: DiLoCoTrainer, OuterOptimizer
- aggregation: RobustAggregator, GradientValidator
- checkpoint: SpeculativeCheckpointer
- service: SwarmServiceMixin
- logger: SwarmLogger
- quorum: Quorum, QuorumRegistry, QuorumMember
"""

__all__ = [
    # Factory
    'SwarmEnabledDynamicNode',
    'SwarmNodeConfig',
    'SwarmComponents',
    'create_swarm_node',
    # Router
    'SwarmRouter',
    'PeerCandidate',
    # Heartbeat
    'SwarmHeartbeatService',
    'CapacityBitmask',
    # Buffers
    'ActivationBuffer',
    'OutboundBuffer',
    'ActivationPacket',
    'ActivationPriority',
    # Compute
    'ComputeEngine',
    'StepOutcome',
    'ComputeStats',
    # DiLoCo
    'DiLoCoTrainer',
    'DiLoCoConfig',
    'OuterOptimizer',
    # Aggregation
    'RobustAggregator',
    'GradientValidator',
    'AggregationStrategy',
    'AggregationConfig',
    'ValidationConfig',
    # Checkpoint
    'SpeculativeCheckpointer',
    'CheckpointConfig',
    # Service
    'SwarmServiceMixin',
    'SwarmNodeState',
    # Logger
    'SwarmLogger',
    'LogCategory',
    'NodeRole',
    'get_swarm_logger',
    'init_swarm_logger',
    # Quorum
    'Quorum',
    'QuorumMember',
    'QuorumRegistry',
    'QuorumLifecycle',
    'QuorumRole',
    'QuorumFormationService',
    'QuorumLifecycleManager',
    'QuorumTrainer',
    'QuorumInferenceRouter',
    'InferenceQuorumInfo',
    'calculate_inference_price',
    'SYNC_INTERVAL',
    'OUTER_LR',
    # Async Training
    'AsyncTrainer',
    'calculate_async_freshness',
]

def __getattr__(name):
    """Lazy loading of submodules."""
    # Factory
    if name in ('SwarmEnabledDynamicNode', 'SwarmNodeConfig', 'SwarmComponents', 'create_swarm_node'):
        from neuroshard.core.swarm import factory
        return getattr(factory, name)
    # Router
    elif name in ('SwarmRouter', 'PeerCandidate', 'RoutingResult'):
        from neuroshard.core.swarm import router
        return getattr(router, name)
    # Heartbeat
    elif name in ('SwarmHeartbeatService', 'CapacityBitmask'):
        from neuroshard.core.swarm import heartbeat
        return getattr(heartbeat, name)
    # Buffers
    elif name in ('ActivationBuffer', 'OutboundBuffer', 'ActivationPacket', 'ActivationPriority'):
        from neuroshard.core.swarm import buffers
        return getattr(buffers, name)
    # Compute
    elif name in ('ComputeEngine', 'StepOutcome', 'ComputeStats', 'InferenceEngine'):
        from neuroshard.core.swarm import compute
        return getattr(compute, name)
    # DiLoCo
    elif name in ('DiLoCoTrainer', 'DiLoCoConfig', 'OuterOptimizer', 'DiLoCoStats', 'DiLoCoPhase'):
        from neuroshard.core.swarm import diloco
        return getattr(diloco, name)
    # Aggregation
    elif name in ('RobustAggregator', 'GradientValidator', 'AggregationStrategy', 
                  'AggregationConfig', 'ValidationConfig', 'GradientContribution'):
        from neuroshard.core.swarm import aggregation
        return getattr(aggregation, name)
    # Checkpoint
    elif name in ('SpeculativeCheckpointer', 'CheckpointConfig', 'CheckpointMetadata', 'CheckpointType'):
        from neuroshard.core.swarm import checkpoint
        return getattr(checkpoint, name)
    # Service
    elif name in ('SwarmServiceMixin', 'SwarmNodeState'):
        from neuroshard.core.swarm import service
        return getattr(service, name)
    # Logger
    elif name in ('SwarmLogger', 'LogCategory', 'NodeRole', 'LogStats', 'get_swarm_logger', 'init_swarm_logger'):
        from neuroshard.core.swarm import logger
        return getattr(logger, name)
    # Quorum
    elif name in ('Quorum', 'QuorumMember', 'QuorumRegistry', 'QuorumLifecycle', 'QuorumRole',
                  'QuorumFormationService', 'QuorumLifecycleManager', 'QuorumTrainer',
                  'QuorumInferenceRouter', 'InferenceQuorumInfo', 'calculate_inference_price',
                  'BASE_SESSION_DURATION', 'MAX_SESSION_DURATION', 'RENEWAL_CHECK_RATIO',
                  'SYNC_INTERVAL', 'OUTER_LR',
                  'AsyncTrainer', 'calculate_async_freshness'):
        from neuroshard.core.swarm import quorum
        return getattr(quorum, name)
    raise AttributeError(f"module 'neuroshard.core.swarm' has no attribute '{name}'")
