"""
Swarm Node Factory - Creates and configures SwarmEnabledNodes

This is THE architecture for NeuroShard. Every node runs the full swarm stack:
- SwarmRouter for fault-tolerant multipath routing
- ActivationBuffer for async compute decoupling
- SwarmHeartbeatService for capacity advertisement
- DiLoCoTrainer for lazy gradient sync
- SpeculativeCheckpointer for fast crash recovery
- RobustAggregator for Byzantine-tolerant gradient aggregation

There are NO toggles, NO fallbacks, NO backward compatibility modes.
Swarm IS the architecture.

Usage:
    from neuroshard.core.swarm import create_swarm_node, SwarmNodeConfig
    
    config = SwarmNodeConfig(
        diloco_inner_steps=500,
        checkpoint_interval=120,
    )
    
    swarm_node = create_swarm_node(
        node_token=token,
        port=port,
        tracker_url=tracker,
        config=config,
    )
"""

import asyncio
import logging
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple

import torch

from neuroshard.core.model.dynamic import (
    DynamicNeuroNode, 
    create_dynamic_node,
    DynamicNeuroLLM,
)

logger = logging.getLogger(__name__)

# Convention: gRPC port = HTTP port + GRPC_PORT_OFFSET
GRPC_PORT_OFFSET = 1000

# Maximum gradient norm for sanity checking peer contributions
MAX_GRAD_NORM = 1000.0


@dataclass
class SwarmNodeConfig:
    """
    Configuration for SwarmEnabledNode.
    
    All values have sensible defaults but can be customized.
    
    NETWORK SIZE CONSIDERATIONS:
    ===========================
    Small network (1-10 nodes):
    - diloco_inner_steps: 100-200 (sync more often for convergence)
    - aggregation_method: "mean" (few Byzantine concerns)
    - cosine_threshold: 0.3 (more lenient, peers vary)
    
    Medium network (10-100 nodes):
    - diloco_inner_steps: 300-500 (balance communication/compute)
    - aggregation_method: "trimmed_mean" (some Byzantine protection)
    - cosine_threshold: 0.5 (stricter validation)
    
    Large network (100+ nodes):
    - diloco_inner_steps: 500-1000 (reduce communication overhead)
    - aggregation_method: "trimmed_mean" or "krum" (Byzantine-tolerant)
    - cosine_threshold: 0.5 (strict validation)
    """
    
    # Buffer Sizes
    inbound_buffer_size: int = 100
    outbound_buffer_size: int = 50
    soft_overflow_threshold: float = 0.9
    hard_overflow_threshold: float = 0.99
    
    # Routing
    ack_timeout_ms: int = 200
    k_candidates: int = 3
    
    # Heartbeat
    heartbeat_interval: float = 5.0
    heartbeat_port: int = 9999
    
    # DiLoCo - Use get_diloco_inner_steps() for dynamic adjustment
    diloco_inner_steps: int = 500          # Base value (adjusted by network size)
    diloco_inner_steps_min: int = 50       # Minimum for very small networks
    diloco_inner_steps_max: int = 1000     # Maximum for very large networks
    diloco_outer_lr: float = 0.7
    diloco_outer_momentum: float = 0.9
    diloco_auto_scale: bool = True         # Auto-adjust inner_steps based on network size
    
    # Checkpointing
    checkpoint_interval: int = 120  # 2 minutes
    max_checkpoints: int = 5
    checkpoint_dir: Optional[str] = None  # Default: ~/.neuroshard/checkpoints
    
    # Aggregation
    aggregation_method: str = "trimmed_mean"  # "mean", "median", "trimmed_mean", "krum"
    krum_f: int = 0  # Byzantine workers for Krum
    trimmed_mean_beta: float = 0.1  # Trim fraction
    
    # Gradient Validation
    cosine_threshold: float = 0.5
    magnitude_ratio_threshold: float = 10.0
    
    # Compute Engine
    num_micro_batches: int = 4
    
    def get_checkpoint_dir(self) -> Path:
        """Get checkpoint directory, creating if needed."""
        if self.checkpoint_dir:
            path = Path(self.checkpoint_dir)
        else:
            path = Path.home() / ".neuroshard" / "checkpoints"
        path.mkdir(parents=True, exist_ok=True)
        return path
    
    def get_diloco_inner_steps(self, num_peers: int = 1) -> int:
        """
        Get optimal DiLoCo inner steps based on network size.
        
        Rationale:
        - Small networks: Sync more often for faster convergence
        - Large networks: Sync less often to reduce communication overhead
        
        Formula: base * (1 + log10(num_peers) / 3), clamped to [min, max]
        
        Examples:
        - 1 peer: 500 * 1.0 = 500 steps
        - 10 peers: 500 * 1.33 = 665 steps
        - 100 peers: 500 * 1.67 = 833 steps
        - 1000 peers: 500 * 2.0 = 1000 steps (capped)
        """
        if not self.diloco_auto_scale or num_peers <= 1:
            return self.diloco_inner_steps
        
        import math
        scale_factor = 1.0 + math.log10(max(1, num_peers)) / 3
        scaled_steps = int(self.diloco_inner_steps * scale_factor)
        
        return max(self.diloco_inner_steps_min, min(self.diloco_inner_steps_max, scaled_steps))


class SwarmComponents:
    """
    Container for all swarm components.
    
    Provides unified lifecycle management for:
    - SwarmRouter (multipath routing)
    - ActivationBuffer/OutboundBuffer (async compute)
    - SwarmHeartbeatService (capacity advertisement)
    - ComputeEngine (GPU worker)
    - DiLoCoTrainer (lazy gradient sync)
    - SpeculativeCheckpointer (crash recovery)
    - RobustAggregator (Byzantine-tolerant aggregation)
    """
    
    def __init__(self):
        # Core components
        self.swarm_router = None
        self.inbound_buffer = None
        self.outbound_buffer = None
        self.heartbeat_service = None
        self.compute_engine = None
        
        # Training components
        self.diloco_trainer = None
        self.outer_optimizer = None
        self.speculative_checkpointer = None
        self.robust_aggregator = None
        self.gradient_validator = None
        
        # State
        self.running = False
        self._tasks: List[asyncio.Task] = []
        
    async def start_async(self):
        """Start all async components."""
        self.running = True
        
        if self.swarm_router:
            await self.swarm_router.start()
            logger.info("[SWARM] SwarmRouter started")
        
        if self.compute_engine:
            task = asyncio.create_task(self.compute_engine.run())
            self._tasks.append(task)
            logger.info("[SWARM] ComputeEngine started")
    
    def start_sync(self):
        """Start all synchronous components (threads)."""
        if self.heartbeat_service:
            self.heartbeat_service.start()
            logger.info("[SWARM] HeartbeatService started")
        
        if self.speculative_checkpointer:
            self.speculative_checkpointer.start()
            logger.info("[SWARM] SpeculativeCheckpointer started")
    
    async def stop_async(self):
        """Stop all async components."""
        self.running = False
        
        for task in self._tasks:
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
        self._tasks.clear()
        
        if self.swarm_router:
            await self.swarm_router.stop()
        
        if self.compute_engine:
            self.compute_engine.running = False
    
    def stop_sync(self):
        """Stop all synchronous components."""
        if self.heartbeat_service:
            self.heartbeat_service.stop()
        
        if self.speculative_checkpointer:
            self.speculative_checkpointer.stop()
    
    def get_stats(self) -> Dict[str, Any]:
        """Get combined stats from all components."""
        stats = {"running": self.running}
        
        if self.inbound_buffer:
            stats["inbound_buffer"] = self.inbound_buffer.get_stats()
        
        if self.outbound_buffer:
            stats["outbound_buffer"] = self.outbound_buffer.get_stats()
        
        if self.swarm_router:
            stats["router"] = self.swarm_router.get_stats()
        
        if self.heartbeat_service:
            stats["heartbeat"] = self.heartbeat_service.get_stats()
        
        if self.compute_engine:
            stats["compute"] = self.compute_engine.get_stats()
        
        if self.diloco_trainer:
            stats["diloco"] = {
                "inner_step_count": self.diloco_trainer.stats.inner_step_count,
                "inner_steps_total": self.diloco_trainer.config.inner_steps,
                "outer_step_count": self.diloco_trainer.stats.outer_step_count,
            }
        
        return stats


class SwarmEnabledDynamicNode:
    """
    A DynamicNeuroNode with full swarm capabilities.
    
    This is THE node type for NeuroShard. Every node runs:
    - Fault-tolerant multipath routing (SwarmRouter)
    - Async activation buffering (ActivationBuffer, OutboundBuffer)
    - Capacity-aware peer selection (SwarmHeartbeatService)
    - Decoupled GPU compute (ComputeEngine)
    - DiLoCo lazy gradient sync (DiLoCoTrainer)
    - Speculative checkpointing (SpeculativeCheckpointer)
    - Byzantine-tolerant aggregation (RobustAggregator)
    
    There are NO toggles. This IS the architecture.
    """
    
    def __init__(
        self,
        base_node: DynamicNeuroNode,
        config: SwarmNodeConfig,
        p2p_manager: Optional[Any] = None,
    ):
        """
        Initialize SwarmEnabledDynamicNode.
        
        Args:
            base_node: The DynamicNeuroNode to enhance (REQUIRED)
            config: SwarmNodeConfig with settings (REQUIRED)
            p2p_manager: Optional P2P manager (uses base_node.p2p_manager if not provided)
        """
        self.base_node = base_node
        self.config = config
        self.p2p_manager = p2p_manager or base_node.p2p_manager
        
        # Expose base node properties directly
        self.node_id = base_node.node_id
        self.node_token = base_node.node_token
        self.model = base_node.model
        self.my_layer_ids = base_node.my_layer_ids
        self.layer_pool = base_node.layer_pool
        self.enable_training = base_node.enable_training
        self.device = base_node.device
        self.available_memory_mb = base_node.available_memory_mb
        
        # Expose training infrastructure for gRPC pipeline handlers
        self._training_lock = base_node._training_lock
        # NOTE: optimizer is accessed via @property (line ~1144) which returns base_node.optimizer
        
        # Training state - sync from base node (may have been loaded from checkpoint)
        self._total_training_rounds = base_node.total_training_rounds
        # Handle None/inf loss values - default to inf for fresh start
        base_loss = base_node.current_loss
        if base_loss is None or (isinstance(base_loss, float) and base_loss == float('inf')):
            self._current_loss = float('inf')
        else:
            self._current_loss = base_loss
        
        # Initialize swarm components (router, heartbeat, compute, etc.)
        # NOTE: Named swarm_components to avoid conflict with base_node.swarm (DataSwarm)
        self.swarm_components = SwarmComponents()
        self._init_swarm_components()
        
        logger.info(f"[SWARM] SwarmEnabledNode initialized for {self.node_id[:16]}...")
        logger.info(f"[SWARM]   - DiLoCo: inner_steps={config.diloco_inner_steps}")
        logger.info(f"[SWARM]   - Checkpointing: interval={config.checkpoint_interval}s")
        logger.info(f"[SWARM]   - Heartbeat: interval={config.heartbeat_interval}s")
        
        # Make swarm_components accessible from base_node BEFORE restoring pending state
        # (DiLoCo restore needs access to swarm_components)
        # NOTE: Don't overwrite base_node.swarm - that's the DataSwarm for P2P downloads!
        base_node.swarm_components = self.swarm_components
        
        # Restore pending state from checkpoint (DiLoCo, optimizer)
        # This must happen AFTER swarm components are initialized AND base_node.swarm is set
        if hasattr(base_node, '_restore_pending_state'):
            base_node._restore_pending_state()
    
    # ==================== PROPERTIES ====================
    # Expose training state for runner/dashboard access
    
    @property
    def total_training_rounds(self) -> int:
        """Total training rounds completed (used by runner for PoNW proofs)."""
        return self._total_training_rounds
    
    @total_training_rounds.setter
    def total_training_rounds(self, value: int):
        self._total_training_rounds = value
    
    @property
    def current_loss(self) -> float:
        """Current training loss (used by dashboard)."""
        return self._current_loss
    
    @current_loss.setter
    def current_loss(self, value: float):
        self._current_loss = value
    
    @property
    def total_tokens_processed(self) -> int:
        """Total tokens processed (delegate to base node)."""
        return self.base_node.total_tokens_processed
    
    @total_tokens_processed.setter
    def total_tokens_processed(self, value: int):
        self.base_node.total_tokens_processed = value
    
    @property
    def current_training_round(self) -> int:
        """Current DiLoCo outer round (for gradient sync coordination)."""
        if self.swarm_components.diloco_trainer:
            return self.swarm_components.diloco_trainer.stats.outer_step_count
        return 0
    
    def _init_swarm_components(self):
        """Initialize all swarm components."""
        from neuroshard.core.swarm.buffers import ActivationBuffer, OutboundBuffer
        from neuroshard.core.swarm.router import SwarmRouter
        from neuroshard.core.swarm.heartbeat import SwarmHeartbeatService
        from neuroshard.core.swarm.compute import ComputeEngine
        
        # Create buffers
        self.swarm_components.inbound_buffer = ActivationBuffer(
            max_size=self.config.inbound_buffer_size
        )
        self.swarm_components.outbound_buffer = OutboundBuffer(
            max_size=self.config.outbound_buffer_size,
            soft_overflow_threshold=self.config.soft_overflow_threshold,
            hard_overflow_threshold=self.config.hard_overflow_threshold,
        )
        
        # Create router
        dht = self.p2p_manager.dht if self.p2p_manager else None
        self.swarm_components.swarm_router = SwarmRouter(dht_protocol=dht)
        self.swarm_components.swarm_router.K_CANDIDATES = self.config.k_candidates
        self.swarm_components.swarm_router.ACK_TIMEOUT_MS = self.config.ack_timeout_ms
        
        # Create heartbeat service
        self.swarm_components.heartbeat_service = SwarmHeartbeatService(
            node_id=self.node_id,
            udp_port=self.config.heartbeat_port,
        )
        self.swarm_components.heartbeat_service.HEARTBEAT_INTERVAL = self.config.heartbeat_interval
        self.swarm_components.heartbeat_service.set_capacity_callback(self._get_capacity_bitmask)
        self.swarm_components.heartbeat_service.set_peer_update_callback(
            self.swarm_components.swarm_router.update_peer_from_heartbeat
        )
        
        # Create compute engine
        self.swarm_components.compute_engine = ComputeEngine(
            model=self.model,
            inbound=self.swarm_components.inbound_buffer,
            outbound=self.swarm_components.outbound_buffer,
            diloco_trainer=None,  # Set after DiLoCo init
            num_micro_batches=self.config.num_micro_batches,
        )
        
        # Initialize training components if training is enabled
        if self.enable_training:
            self._init_diloco()
            self._init_checkpointer()
            self._init_global_tracker()  # Initialize tracker at startup to restore state
    
    def _init_diloco(self):
        """Initialize DiLoCo trainer and related components."""
        from neuroshard.core.swarm.diloco import DiLoCoTrainer, OuterOptimizer, DiLoCoConfig
        from neuroshard.core.swarm.aggregation import (
            RobustAggregator, 
            GradientValidator,
            AggregationConfig, 
            AggregationStrategy,
            ValidationConfig,
        )
        
        # Create outer optimizer
        self.swarm_components.outer_optimizer = OuterOptimizer(
            lr=self.config.diloco_outer_lr,
            momentum=self.config.diloco_outer_momentum,
        )
        
        # Create DiLoCo trainer
        diloco_config = DiLoCoConfig(
            inner_steps=self.config.diloco_inner_steps,
            outer_lr=self.config.diloco_outer_lr,
            outer_momentum=self.config.diloco_outer_momentum,
        )
        
        self.swarm_components.diloco_trainer = DiLoCoTrainer(
            model=self.model,
            config=diloco_config,
            inner_optimizer=self.base_node.optimizer,
        )
        
        # Connect to compute engine
        self.swarm_components.compute_engine.diloco = self.swarm_components.diloco_trainer
        
        # Create gradient validator
        validation_config = ValidationConfig(
            min_cosine_similarity=self.config.cosine_threshold,
            max_magnitude_ratio=self.config.magnitude_ratio_threshold,
        )
        self.swarm_components.gradient_validator = GradientValidator(config=validation_config)
        
        # Create robust aggregator
        strategy_map = {
            "mean": AggregationStrategy.MEAN,
            "median": AggregationStrategy.MEDIAN,
            "trimmed_mean": AggregationStrategy.TRIMMED_MEAN,
            "krum": AggregationStrategy.KRUM,
        }
        strategy = strategy_map.get(self.config.aggregation_method, AggregationStrategy.TRIMMED_MEAN)
        
        agg_config = AggregationConfig(
            strategy=strategy,
            num_byzantine=self.config.krum_f,
            trim_fraction=self.config.trimmed_mean_beta,
        )
        
        self.swarm_components.robust_aggregator = RobustAggregator(
            aggregation_config=agg_config,
            validation_config=validation_config,
        )
    
    def _init_checkpointer(self):
        """Initialize speculative checkpointer."""
        from neuroshard.core.swarm.checkpoint import SpeculativeCheckpointer, CheckpointConfig
        
        checkpoint_config = CheckpointConfig(
            checkpoint_dir=str(self.config.get_checkpoint_dir()),
            snapshot_interval=float(self.config.checkpoint_interval),
            max_hot_snapshots=self.config.max_checkpoints,
        )
        
        self.swarm_components.speculative_checkpointer = SpeculativeCheckpointer(
            model=self.model,
            optimizer=self.base_node.optimizer,
            config=checkpoint_config,
            diloco_trainer=self.swarm_components.diloco_trainer,
            p2p_manager=self.p2p_manager,
        )
    
    def _get_model_hash(self) -> str:
        """
        Get hash of current model WEIGHTS for chained PoNW verification.
        
        This hashes actual parameter VALUES (sampled for speed), not just
        architecture structure. The hash MUST change when training modifies
        weights, otherwise PoNW proofs will be rejected because
        model_hash_start == model_hash_end (no evidence of training).
        
        Sampling strategy: 5 evenly-spaced parameters, first 1000 elements each.
        This is fast (~1ms) while reliably detecting weight changes from training.
        """
        if not self.model:
            return ""
        
        import hashlib
        hasher = hashlib.sha256()
        
        # Include architecture info for context
        if hasattr(self.model, 'architecture'):
            hidden_dim = self.model.architecture.hidden_dim
            num_heads = self.model.architecture.num_heads
        else:
            hidden_dim = getattr(self.model, 'hidden_dim', 0)
            num_heads = getattr(self.model, 'num_heads', 0)
        arch_str = f"{hidden_dim}:{len(self.my_layer_ids)}:{num_heads}"
        hasher.update(arch_str.encode())
        
        # Hash actual parameter VALUES (sampled for speed)
        # This ensures the hash changes when training modifies weights
        params = list(self.model.named_parameters())
        if params:
            sample_indices = [0, len(params)//4, len(params)//2, 3*len(params)//4, len(params)-1]
            for idx in sample_indices:
                if 0 <= idx < len(params):
                    name, param = params[idx]
                    hasher.update(name.encode())
                    flat = param.data.flatten()[:1000]
                    hasher.update(flat.cpu().detach().numpy().tobytes())
        
        return hasher.hexdigest()[:16]
    
    def _get_capacity_bitmask(self):
        """Get current capacity for heartbeat broadcast."""
        from neuroshard.core.swarm.heartbeat import CapacityBitmask
        
        # Get memory info
        available_mb = 0
        gpu_util = 0.0
        
        if torch.cuda.is_available():
            try:
                free, total = torch.cuda.mem_get_info()
                available_mb = free // (1024 * 1024)
            except:
                pass
        else:
            available_mb = self.available_memory_mb
        
        # Determine layer range
        layer_range = (0, 0)
        if self.my_layer_ids:
            layer_range = (min(self.my_layer_ids), max(self.my_layer_ids) + 1)
        
        # Get buffer status
        queue_depth = len(self.swarm_components.inbound_buffer) if self.swarm_components.inbound_buffer else 0
        is_backpressured = self.swarm_components.inbound_buffer.is_backpressured if self.swarm_components.inbound_buffer else False
        
        return CapacityBitmask(
            node_id=self.node_id,
            timestamp=time.time(),
            available_memory_mb=available_mb,
            queue_depth=queue_depth,
            layer_range=layer_range,
            gpu_utilization=gpu_util,
            network_saturation=0.0,
            is_training=self.enable_training,
            is_accepting_inference=True,
            is_accepting_activations=not is_backpressured,
            grpc_addr=self.base_node.grpc_addr,
        )
    
    # ==================== LIFECYCLE ====================
    
    def start(self):
        """Start all swarm components."""
        self.swarm_components.start_sync()
        logger.info("[SWARM] Node started")
    
    def stop(self):
        """Stop all swarm components."""
        self.swarm_components.stop_sync()
        logger.info("[SWARM] Node stopped")
    
    async def start_async(self):
        """Start async swarm components."""
        await self.swarm_components.start_async()
    
    async def stop_async(self):
        """Stop async swarm components."""
        await self.swarm_components.stop_async()
    
    # ==================== BUFFER ACCESS ====================
    
    def receive_activation(self, packet: 'ActivationPacket') -> bool:
        """
        Receive an activation packet from a peer.
        
        Called by gRPC handler when SwarmForward is received.
        """
        return self.swarm_components.inbound_buffer.put_nowait(packet)
    
    def get_buffer_status(self) -> Dict[str, Any]:
        """Get status of inbound and outbound buffers."""
        return {
            'inbound': self.swarm_components.inbound_buffer.get_stats(),
            'outbound': self.swarm_components.outbound_buffer.get_stats(),
        }
    
    # ==================== ROUTING ====================
    
    def get_swarm_route(self) -> Dict[int, List['PeerCandidate']]:
        """
        Get swarm route with K candidates per layer.
        
        Returns dict of layer_id -> list of K candidates.
        """
        from neuroshard.core.swarm.router import PeerCandidate
        
        route: Dict[int, List[PeerCandidate]] = {}
        num_layers = self.layer_pool.current_num_layers if self.layer_pool else 12
        
        for layer_id in range(num_layers):
            candidates = self.swarm_components.swarm_router.get_candidates(layer_id)
            if candidates:
                route[layer_id] = candidates
        
        return route
    
    # ==================== TRAINING ====================
    
    def _init_global_tracker(self):
        """Initialize global training tracker for verification."""
        try:
            from neuroshard.core.training.global_tracker import GlobalTrainingTracker
            self._global_tracker = GlobalTrainingTracker(
                node_id=self.node_id,
                model=self.model,
            )
            logger.info("[SWARM] Global training tracker initialized")
        except Exception as e:
            logger.warning(f"[SWARM] Could not init global tracker: {e}")
            self._global_tracker = None
    
    def train_step(self) -> Optional[float]:
        """
        Execute a training step with DiLoCo lazy gradient sync.
        
        Supports both local and distributed training:
        - Full nodes (embedding + LM head + NO next_hop): Train locally with DiLoCo
        - DRIVER with next_hop: Use pipeline training (forward to next node)
        - Workers (no embedding): Wait for activations via gRPC
        
        KEY DESIGN: Check for next_hop FIRST before deciding to train locally.
        If another node has layers after ours, use distributed training.
        
        Returns loss value or None if no data available.
        """
        if not self.enable_training:
            return None
        
        # Ensure global tracker is initialized
        if not hasattr(self, '_global_tracker') or self._global_tracker is None:
            self._init_global_tracker()
        
        # STEP 1: Check if there's a next hop (another node with higher layers)
        # This is THE KEY check - if someone else has layers after us, use pipeline!
        # IMPORTANT: Use for_training=True to only find training-capable peers!
        my_last_layer = max(self.base_node.my_layer_ids) if self.base_node.my_layer_ids else 0
        next_layer = my_last_layer + 1
        
        has_next_hop = False
        if hasattr(self.base_node, 'p2p_manager') and self.base_node.p2p_manager:
            # Only find peers that can participate in training (not observers)
            next_hop_url = self.base_node.p2p_manager.get_next_hop(next_layer, for_training=True)
            if next_hop_url:
                has_next_hop = True
                logger.debug(f"[SWARM] Found training-capable next hop for layer {next_layer}: {next_hop_url}")
        
        # STEP 2: Decide training mode based on role and network state
        if not self.model.has_embedding:
            # WORKER: No embedding = wait for activations via gRPC
            # Training happens reactively in forward_pipeline/backward_pipeline
            return None
        
        # I am DRIVER (has embedding)
        if has_next_hop:
            # DISTRIBUTED TRAINING: Another node has layers after mine
            # Forward to them via base_node.train_step() which handles the pipeline
            # Even if I have a temporary LM head, use distributed mode!
            logger.debug(f"[SWARM] DRIVER using distributed training (next_hop exists)")
            loss = self.base_node.train_step()
            if loss is not None:
                # Increment swarm's counter even for distributed training
                self._total_training_rounds += 1
                self._current_loss = loss
            return loss
        
        # No next_hop - check if we can do local training
        if not self.model.has_lm_head:
            # DRIVER with no next_hop and no LM head - can't train yet
            logger.debug("[SWARM] DRIVER has no LM head and no next hop - waiting for Validator")
            return None
        
        # LOCAL TRAINING: Full node with embedding + LM head + NO next_hop
        # This means we're truly solo (no other nodes have layers after us)
        diloco = self.swarm_components.diloco_trainer
        
        # Get training data
        batch = self.base_node._get_training_batch()
        if batch is None:
            return None
        
        input_ids, labels = batch
        
        # Check if data is actually ready (tuple might contain None values)
        if input_ids is None or labels is None:
            return None
        
        # Forward pass
        self.model.train()
        outputs, _ = self.model.forward_my_layers(
            self.model.embed(input_ids.to(self.device))
        )
        
        # Compute loss locally
        logits = self.model.compute_logits(outputs)
        loss = torch.nn.functional.cross_entropy(
            logits.view(-1, logits.size(-1)),
            labels.view(-1).to(self.device)
        )
        
        # DiLoCo inner step (backward + optimizer step + zero_grad)
        diloco.inner_step(loss)
        self._current_loss = loss.item()
        self._total_training_rounds += 1
        
        # Record loss to data loader for plateau detection
        # This enables adaptive shard rotation when the model stops learning
        if hasattr(self.base_node, 'genesis_loader') and self.base_node.genesis_loader:
            self.base_node.genesis_loader.record_loss(self._current_loss)
        
        # Note: grad_norm is 0 after inner_step because zero_grad() was called
        # To properly track gradient norm, we'd need to compute it inside inner_step
        # before zero_grad(). For now, we use pseudo-gradient norm from DiLoCo stats.
        grad_norm = diloco.stats.avg_pseudo_grad_norm if diloco.stats.avg_pseudo_grad_norm > 0 else 0.0
        
        # Record step in global tracker (for verification)
        if self._global_tracker:
            current_shard = 0
            if hasattr(self.base_node, 'genesis_loader') and self.base_node.genesis_loader:
                stats = self.base_node.genesis_loader.get_stats()
                current_shard = stats.get('current_shard_id', 0)
            
            self._global_tracker.record_step(
                loss=self._current_loss,
                step=self._total_training_rounds,
                shard_id=current_shard,
                tokens_in_batch=input_ids.numel(),
                gradient_norm=grad_norm,
                inner_step=diloco.stats.inner_step_count,
                outer_step=diloco.stats.outer_step_count,
            )
        
        # Check if outer sync needed
        if diloco.should_sync():
            self._do_diloco_sync()
        
        # NOTE: Checkpoint saving is handled by DiLoCo outer sync (every 500 steps)
        # We removed the per-100-step checkpoint because:
        # 1. It caused 70-80s delays on memory-constrained systems
        # 2. DiLoCo outer sync already saves after each 500-step cycle
        # 3. Worst case loss on crash: 500 steps (acceptable)
        
        return self._current_loss
    
    def get_global_training_status(self) -> Dict[str, Any]:
        """
        Get global training verification status.
        
        Returns comprehensive stats showing:
        - Whether training is actually improving the model
        - Whether nodes are converging (same model hash)
        - Network-wide loss metrics
        """
        if not hasattr(self, '_global_tracker') or not self._global_tracker:
            return {
                "error": "Global tracker not initialized",
                "training_verified": False,
            }
        
        return self._global_tracker.get_global_status()
    
    def _do_diloco_sync(self):
        """
        Execute DiLoCo outer synchronization with REAL gradient exchange.
        
        This is the key step that makes distributed training ACTUALLY work:
        1. Compute local pseudo-gradient (delta from initial weights)
        2. Gossip to peers via GossipGradient RPC
        3. Collect peer contributions
        4. Aggregate using Byzantine-tolerant aggregator
        5. Apply aggregated update to model
        
        If no peers respond, falls back to local gradient only.
        """
        import zlib
        import random
        from neuroshard.protos import neuroshard_pb2, neuroshard_pb2_grpc
        from neuroshard.core.network.connection_pool import get_channel
        from urllib.parse import urlparse
        
        diloco = self.swarm_components.diloco_trainer
        
        # Step 1: Compute local pseudo-gradient
        pseudo_grad = diloco.compute_pseudo_gradient()
        
        # Compress gradients for transmission
        from neuroshard.core.training.production import GradientCompressor
        compressor = GradientCompressor()
        
        compressed_grads = {}
        for name, grad in pseudo_grad.items():
            compressed_grads[name] = compressor.compress(grad, name)
        
        # Step 2: Gossip to peers
        peers_synced = 0
        peer_contributions = []  # List of (peer_id, decompressed_grads, batch_size, loss)
        
        if self.p2p_manager:
            # Get list of peers
            peers = list(self.p2p_manager.known_peers.keys())
            routing_peers = []
            if self.p2p_manager.routing_table:
                for n in self.p2p_manager.routing_table.get_all_nodes():
                    routing_peers.append(f"http://{n.ip}:{n.port}")
            
            peers.extend(routing_peers)
            # Deduplicate
            peers = list(set(peers))
            
            logger.debug(f"[SWARM] Peer discovery: known_peers={len(self.p2p_manager.known_peers)}, "
                        f"routing_table={len(routing_peers)}, total_unique={len(peers)}")
            
            if peers:
                # DYNAMIC GOSSIP FANOUT for network scaling
                # Formula: 2 * sqrt(N) + 3, capped at 50
                # This ensures gossip reaches the whole network in O(log N) rounds
                # - 1-10 nodes: 5-9 peers (ensures full coverage)
                # - 100 nodes: ~23 peers (good redundancy)
                # - 1000 nodes: ~66 peers (capped at 50)
                # - 10000+ nodes: 50 peers (bandwidth limit)
                import math
                num_peers = len(peers)
                # Use 2*sqrt(N) for better convergence in large networks
                fanout = min(int(2 * math.sqrt(num_peers) + 3), 50)
                targets = random.sample(peers, min(num_peers, fanout))
                
                # Create request with model hash for architecture validation
                model_hash = self._get_model_hash()
                
                # Include architecture dimensions for debugging compatibility issues
                arch_info = ""
                if self.model:
                    arch_info = f"{getattr(self.model, 'hidden_dim', '?')}x{len(self.my_layer_ids)}L"
                
                req = neuroshard_pb2.GossipGradientRequest(
                    node_id=str(self.node_id),
                    round_id=diloco.stats.outer_step_count,
                    model_hash=model_hash,  # Critical for architecture validation
                    timestamp=time.time(),
                    batch_size=diloco.config.inner_steps,
                    loss=diloco.stats.avg_inner_loss,
                    layer_gradients=compressed_grads,
                    signature=f"diloco_{diloco.stats.outer_step_count}_{self.node_id[:8]}_{arch_info}",
                    ttl=3,
                )
                
                logger.info(f"[SWARM] DiLoCo sync: gossiping to {len(targets)} peers...")
                
                # Gossip to each peer and collect responses
                for target_url in targets:
                    try:
                        clean_url = target_url
                        if "|" in clean_url:
                            # Strip LAN suffix from pipe-format endpoints
                            if "://" in clean_url:
                                clean_url = clean_url.split("://")[0] + "://" + clean_url.split("://")[1].split("|")[0]
                            else:
                                clean_url = clean_url.split("|")[0]
                        parsed = urlparse(clean_url)
                        ip = parsed.hostname
                        port = (parsed.port or 80) + 1000  # gRPC port
                        
                        channel = get_channel(f"{ip}:{port}")
                        stub = neuroshard_pb2_grpc.NeuroShardServiceStub(channel)
                        
                        resp = stub.GossipGradient(req, timeout=10.0)
                        
                        if resp.accepted:
                            peers_synced += 1
                            logger.debug(f"[SWARM] Peer {ip}:{port} accepted gradient (round={resp.current_round})")
                    except Exception as e:
                        logger.debug(f"[SWARM] Gossip to {target_url} failed: {e}")
        
        # Step 3: Collect contributions (our own + any received from peers)
        # Note: Peer contributions are received asynchronously via GossipGradient RPC handler
        # For now, we use our local gradient + any cached peer gradients
        all_contributions = [
            {
                "node_id": self.node_id,
                "gradients": pseudo_grad,
                "batch_size": diloco.config.inner_steps,
                "loss": diloco.stats.avg_inner_loss,
            }
        ]
        
        # Add any peer contributions we've received (stored in _received_peer_grads)
        if hasattr(self, '_received_peer_grads'):
            for peer_id, peer_data in self._received_peer_grads.items():
                # Only use fresh contributions (< 60s old)
                if time.time() - peer_data.get('timestamp', 0) < 60:
                    all_contributions.append({
                        "node_id": peer_id,
                        "gradients": peer_data['gradients'],
                        "batch_size": peer_data.get('batch_size', 1),
                        "loss": peer_data.get('loss', float('inf')),
                    })
            # Clear after using
            self._received_peer_grads = {}
        
        # Step 4: Aggregate using robust aggregator
        aggregated_grads = pseudo_grad  # Default to local if no peers
        
        if len(all_contributions) > 1 and self.swarm_components.robust_aggregator:
            logger.info(f"[SWARM] Aggregating {len(all_contributions)} contributions")
            try:
                # Clear previous contributions and add new ones
                self.swarm_components.robust_aggregator.clear()
                
                # Add peer contributions (validate against our local gradient)
                rejected_peers = []
                for contrib in all_contributions:
                    if contrib["node_id"] == self.node_id:
                        continue  # Skip self - add as local_grads
                    
                    accepted, reason = self.swarm_components.robust_aggregator.add_contribution(
                        peer_id=contrib["node_id"],
                        gradients=contrib["gradients"],
                        reference_grads=pseudo_grad,  # Validate against our gradient
                        trust_score=1.0,
                        validate=True
                    )
                    if not accepted:
                        rejected_peers.append((contrib["node_id"][:16], reason))
                
                # Aggregate with our local gradients included
                aggregated_grads = self.swarm_components.robust_aggregator.aggregate(
                    local_grads=pseudo_grad
                )
                
                if rejected_peers:
                    logger.warning(f"[SWARM] Rejected {len(rejected_peers)} contributions: {rejected_peers}")
                    
            except Exception as e:
                logger.error(f"[SWARM] Aggregation failed, using local: {e}")
                import traceback
                logger.error(traceback.format_exc())
                aggregated_grads = pseudo_grad
        
        # Capture stats BEFORE apply_outer_update (which resets them)
        avg_loss_before_reset = diloco.stats.avg_inner_loss
        outer_step_before = diloco.stats.outer_step_count
        
        # Step 5: Apply aggregated update
        diloco.apply_outer_update(aggregated_grads)
        
        # Record sync result for tracking
        if hasattr(self, '_global_tracker') and self._global_tracker:
            self._global_tracker.record_sync_result(
                success=peers_synced > 0,
                peers_synced=peers_synced
            )
        
        sync_status = "distributed" if peers_synced > 0 else "local-only"
        logger.info(
            f"[SWARM] DiLoCo outer sync #{outer_step_before + 1} ({sync_status}): "
            f"peers={peers_synced}, contributions={len(all_contributions)}, "
            f"avg_loss={avg_loss_before_reset:.4f}"
        )
        
        # IMPORTANT: Save checkpoint after each outer sync (this is a major milestone!)
        self._save_checkpoint()
        logger.info(f"[SWARM] Checkpoint saved after outer sync #{diloco.stats.outer_step_count}")
    
    def receive_peer_gradients(self, contribution) -> bool:
        """
        Receive gradient contribution from a peer (called by GossipGradient RPC).
        
        Stores the contribution for use in next sync round.
        
        Security Checks (in order):
        1. Model hash validation (architecture compatibility)
        2. Round ID check (freshness)
        3. Gradient shape validation (per-parameter)
        4. Gradient magnitude sanity check
        """
        from neuroshard.core.training.production import GradientCompressor
        
        if not hasattr(self, '_received_peer_grads'):
            self._received_peer_grads = {}
        
        try:
            # === SECURITY CHECK 1: Model Hash Validation (REQUIRED for gradients) ===
            # Unlike proofs (which are historical records), gradient sync REQUIRES
            # matching architectures - you cannot average tensors of different shapes.
            # Peers with different architectures form separate training cohorts.
            if hasattr(contribution, 'model_hash') and contribution.model_hash:
                our_hash = self._get_model_hash()
                if our_hash and contribution.model_hash != our_hash:
                    logger.info(
                        f"[SWARM] Skipping gradient from {contribution.node_id[:8]}... - "
                        f"different architecture cohort (peer={contribution.model_hash[:8]}..., "
                        f"ours={our_hash[:8]}...). Will sync with matching peers."
                    )
                    return False
            
            # === SECURITY CHECK 2: Round ID Freshness ===
            # Only accept gradients from recent rounds (within 2 rounds)
            if self.swarm_components.diloco_trainer:
                our_round = self.swarm_components.diloco_trainer.stats.outer_step_count
                peer_round = getattr(contribution, 'round_id', 0)
                if abs(our_round - peer_round) > 2:
                    logger.warning(
                        f"[SWARM] REJECTED stale gradient from {contribution.node_id[:8]}... - "
                        f"round mismatch: peer={peer_round}, ours={our_round}"
                    )
                    return False
            
            # Decompress gradients
            compressor = GradientCompressor()
            gradients = {}
            
            for name, compressed in contribution.layer_gradients.items():
                gradients[name] = compressor.decompress(compressed)
            
            # === SECURITY CHECK 3: Architecture Shape Validation ===
            # Ensures gradient shapes match our model parameters
            if self.model and hasattr(self.model, 'named_parameters'):
                our_params = dict(self.model.named_parameters())
                mismatched = []
                
                for name, grad in gradients.items():
                    if name in our_params:
                        if grad.shape != our_params[name].shape:
                            mismatched.append((name, grad.shape, our_params[name].shape))
                
                if mismatched:
                    logger.warning(
                        f"[SWARM] REJECTED gradient from {contribution.node_id[:8]}... - "
                        f"{len(mismatched)} shape mismatches (incompatible architecture!)"
                    )
                    for name, peer_shape, our_shape in mismatched[:3]:  # Log first 3
                        logger.warning(f"  {name}: peer={peer_shape}, ours={our_shape}")
                    return False
            
            # === SECURITY CHECK 4: Gradient Magnitude Sanity ===
            # Reject obviously malicious gradients (too large)
            total_norm = sum(g.norm().item() ** 2 for g in gradients.values()) ** 0.5
            if total_norm > MAX_GRAD_NORM:
                logger.warning(
                    f"[SWARM] REJECTED gradient from {contribution.node_id[:8]}... - "
                    f"gradient norm {total_norm:.2f} exceeds max {MAX_GRAD_NORM}"
                )
                return False
            
            # Store for aggregation
            self._received_peer_grads[contribution.node_id] = {
                'gradients': gradients,
                'batch_size': getattr(contribution, 'batch_size', 1),
                'loss': getattr(contribution, 'loss', float('inf')),
                'timestamp': getattr(contribution, 'timestamp', time.time()),
            }
            
            logger.info(f"[SWARM] Accepted gradient from {contribution.node_id[:8]}... "
                       f"(round={getattr(contribution, 'round_id', '?')}, "
                       f"loss={getattr(contribution, 'loss', 0):.4f}, "
                       f"norm={total_norm:.2f})")
            
            return True
            
        except Exception as e:
            logger.error(f"[SWARM] Failed to process peer gradient: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return False
    
    # ==================== STATS ====================
    
    def get_stats(self) -> Dict[str, Any]:
        """Get combined stats from base node and swarm components."""
        # Safety check for shutdown race condition
        base_node = getattr(self, 'base_node', None)
        if not base_node:
            return {"status": "shutting_down"}
        
        try:
            stats = base_node.get_stats()
        except (AttributeError, RuntimeError):
            # Node is shutting down
            return {"status": "shutting_down"}
        
        # Override with swarm node's actual training values (these are updated
        # in train_step() but base_node's values are only synced on checkpoint)
        stats["total_training_rounds"] = getattr(self, '_total_training_rounds', 0)
        stats["current_loss"] = getattr(self, '_current_loss', 0.0)
        
        swarm = getattr(self, 'swarm', None)
        if swarm:
            stats["swarm"] = swarm.get_stats()
        return stats
    
    def get_swarm_status(self) -> Dict[str, Any]:
        """Get detailed swarm status."""
        return self.swarm_components.get_stats()
    
    def get_diloco_progress(self) -> Dict[str, Any]:
        """Get DiLoCo training progress."""
        if not self.swarm_components.diloco_trainer:
            return {"enabled": False}
        
        diloco = self.swarm_components.diloco_trainer
        return {
            "enabled": True,
            "inner_step_count": diloco.stats.inner_step_count,
            "inner_steps_total": diloco.config.inner_steps,
            "progress": diloco.stats.inner_step_count / diloco.config.inner_steps,
            "outer_step_count": diloco.stats.outer_step_count,
        }
    
    # ==================== DELEGATION ====================
    
    @property
    def grpc_addr(self):
        """Get gRPC address."""
        return self.base_node.grpc_addr
    
    @property
    def data_manager(self):
        """Get data manager."""
        return self.base_node.data_manager
    
    @property
    def genesis_loader(self):
        """Get genesis loader."""
        return self.base_node.genesis_loader
    
    @property
    def optimizer(self):
        """Get optimizer."""
        return self.base_node.optimizer
    
    def forward(self, input_ids: torch.Tensor, **kwargs):
        """Forward pass - delegates to base node."""
        return self.base_node.forward(input_ids, **kwargs)
    
    def _save_checkpoint(self, async_save: bool = True):
        """
        Save checkpoint with swarm state.
        
        Syncs the swarm node's training counters to base node before saving.
        
        Args:
            async_save: If True, save in background thread (default).
                        If False, block until save completes (for shutdown).
        """
        # Sync training state to base node before saving
        self.base_node.total_training_rounds = self._total_training_rounds
        self.base_node.current_loss = self._current_loss
        
        # Delegate to base node's checkpoint saving
        return self.base_node._save_checkpoint(async_save=async_save)
    
    def __getattr__(self, name):
        """Delegate unknown attributes to base node."""
        return getattr(self.base_node, name)

    def __setattr__(self, name, value):
        """Set attributes, delegating base_node-owned attributes properly."""
        # Attributes that belong to base_node (not the wrapper)
        BASE_NODE_ATTRS = {
            'genesis_loader', 'swarm', 'is_running', 'enable_training',
            '_contribution_mode', '_quorum_id', '_speed_tier', '_latency_ms',
            'total_training_rounds', 'total_tokens_processed',
        }
        # If this is an attribute that belongs to base_node and we have a base_node
        if name in BASE_NODE_ATTRS and hasattr(self, 'base_node') and self.base_node is not None:
            setattr(self.base_node, name, value)
        else:
            # Normal attribute setting for wrapper's own attributes
            object.__setattr__(self, name, value)

    def verify_training_work(self, proof: Any) -> Tuple[bool, str]:
        """
        Verify training work against THIS NODE's internal state.
        
        Called by ProofVerifier after universal checks pass.
        
        This method enforces "personal integrity" - checks that only
        the node itself can verify:
        - Did I actually do the work I'm claiming?
        - Does the model hash match MY architecture?
        - Is training actually enabled on MY node?
        
        Universal checks (rate limits, required fields) are handled
        by ProofVerifier before this method is called.
        
        Args:
            proof: PoNWProof object
            
        Returns:
            (is_valid, reason) tuple
        """
        # =====================================================================
        # NODE-SPECIFIC CHECK 1: Model Hash Recording (NOT rejection)
        # =====================================================================
        # The model hash in a proof records WHAT architecture the work was done on.
        # In a growing network, architectures evolve - a proof from a month ago
        # was valid for THAT architecture, even if the network has since grown.
        # 
        # We LOG mismatches for monitoring but DO NOT REJECT - the work was valid
        # when it was done. The hash is stored with the proof as historical metadata.
        current_hash = self._get_model_hash()
        proof_hash = getattr(proof, 'model_hash', None)
        
        if proof_hash and current_hash and proof_hash != current_hash:
            # Different architecture - this is NORMAL in a growing network.
            # The proof was valid for its architecture at the time.
            logger.info(
                f"[PoNW] Proof from different architecture epoch: "
                f"proof={proof_hash[:8]}..., current={current_hash[:8]}... "
                f"(network has evolved - this is expected)"
            )
            # Continue validation - don't reject based on hash alone
        
        # =====================================================================
        # NODE-SPECIFIC CHECK 2: Internal Counter Verification
        # =====================================================================
        # This is the core "Lazy Mining" prevention.
        # Only verify our OWN proofs against internal counter - we can't
        # know other nodes' internal state.
        proof_node_id = getattr(proof, 'node_id', None)
        claimed_batches = getattr(proof, 'training_batches', 0)
        
        if proof_node_id == self.node_id:
            # This is OUR proof - verify against our actual work count.
            # Allow a small buffer (10) for timing/sync issues.
            buffer = 10
            if claimed_batches > (self._total_training_rounds + buffer):
                logger.warning(
                    f"[PoNW] Lazy mining attempt blocked: "
                    f"claimed={claimed_batches}, actual={self._total_training_rounds}"
                )
                return False, (
                    f"Claimed batches ({claimed_batches}) exceeds "
                    f"internal counter ({self._total_training_rounds})"
                )
            
            # =====================================================================
            # NODE-SPECIFIC CHECK 3: Training Enabled Status
            # =====================================================================
            # Can't claim training rewards if training is disabled.
            if not self.enable_training and claimed_batches > 0:
                return False, "Claimed training work but training is disabled"
        
        # All node-specific checks passed
        return True, "Training work verified against local state"

    # ==================== FACTORY ====================

def create_swarm_node(
    node_token: str,
    port: int,
    tracker_url: str,
    config: SwarmNodeConfig,
    available_memory_mb: Optional[int] = None,
    enable_training: bool = False,
    max_storage_mb: int = 10000,
    max_cpu_threads: int = 4,
    p2p_manager: Optional[Any] = None,
    device: str = "auto",
) -> SwarmEnabledDynamicNode:
    """
    Factory function to create a SwarmEnabledDynamicNode.
    
    This is the main entry point for creating nodes.
    
    Args:
        node_token: Authentication token for the node
        port: HTTP port for the node
        tracker_url: URL of the tracker server
        config: SwarmNodeConfig with settings (REQUIRED)
        available_memory_mb: Override memory detection
        enable_training: Whether to enable training
        max_storage_mb: Max disk space for data shards
        max_cpu_threads: Max CPU threads to use
        p2p_manager: Optional P2P manager (created if not provided)
    
    Returns:
        SwarmEnabledDynamicNode ready for use
    """
    # Create base DynamicNeuroNode
    # Pass P2P manager so DHT is available during layer assignment!
    base_node = create_dynamic_node(
        node_token=node_token,
        port=port,
        tracker_url=tracker_url,
        available_memory_mb=available_memory_mb,
        enable_training=enable_training,
        max_storage_mb=max_storage_mb,
        max_cpu_threads=max_cpu_threads,
        device=device,
        p2p_manager=p2p_manager,  # NEW: Pass P2P for DHT discovery
    )
    
    # Wrap with swarm capabilities
    swarm_node = SwarmEnabledDynamicNode(
        base_node=base_node,
        config=config,
        p2p_manager=p2p_manager,
    )
    
    # INJECT INTO LEDGER: Connect the node to the ledger for PoNW verification
    # This fixes the "Lazy Mining" vulnerability by allowing the ledger to 
    # verify work against the actual model state.
    if base_node.p2p_manager and base_node.p2p_manager.ledger:
        # Set the model interface for the verifier
        if hasattr(base_node.p2p_manager.ledger, 'verifier'):
            base_node.p2p_manager.ledger.verifier.model_interface = swarm_node
            logger.info("[FACTORY] Connected SwarmNode to Ledger Verifier (PoNW security active)")
            
    return swarm_node


# Alias for backwards compatibility and clarity
create_swarm_node_with_p2p = create_swarm_node
