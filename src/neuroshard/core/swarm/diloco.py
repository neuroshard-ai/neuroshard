"""
DiLoCo Trainer - Distributed Low-Communication Training

Implements the DiLoCo algorithm for bandwidth-efficient distributed training:
- Inner Loop: Each node trains independently for N steps (local SGD)
- Outer Loop: Periodically sync pseudo-gradients across peers
- Outer Optimizer: Nesterov momentum on the aggregated delta

Key Benefits:
- N× reduction in communication (sync every 500 steps vs every step)
- More robust to stragglers (nodes train at their own pace)
- Better for high-latency residential networks
- Naturally supports the "Don't Stop" soft overflow mechanism

Based on: "DiLoCo: Distributed Low-Communication Training of Language Models"
(Douillard et al., 2023)

Usage:
    trainer = DiLoCoTrainer(model, optimizer, inner_steps=500)
    
    # Training loop
    while training:
        loss = trainer.inner_step(batch)
        
        if trainer.should_sync():
            pseudo_grads = trainer.compute_pseudo_gradient()
            aggregated = await gossip_gradients(pseudo_grads)
            trainer.apply_outer_update(aggregated)
"""

import asyncio
import copy
import io
import logging
import threading
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Callable, Tuple
from enum import Enum

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


class DiLoCoPhase(Enum):
    """Current phase of DiLoCo training."""
    INNER_LOOP = "inner_loop"      # Local training
    COMPUTING_DELTA = "computing_delta"  # Computing pseudo-gradient
    SYNCING = "syncing"            # Waiting for peer aggregation
    OUTER_STEP = "outer_step"      # Applying outer update
    IDLE = "idle"


@dataclass
class DiLoCoConfig:
    """Configuration for DiLoCo training."""
    # Inner loop settings
    inner_steps: int = 500              # Steps before sync
    inner_lr: float = 1e-4              # Inner optimizer learning rate
    inner_weight_decay: float = 0.1     # Weight decay for inner optimizer
    
    # Outer loop settings
    outer_lr: float = 0.7               # Outer optimizer learning rate
    outer_momentum: float = 0.9         # Nesterov momentum
    outer_weight_decay: float = 0.0     # Outer weight decay (usually 0)
    
    # Gradient settings
    max_grad_norm: float = 1.0          # Gradient clipping
    gradient_accumulation: int = 1      # Accumulation steps
    
    # Sync settings
    sync_timeout: float = 60.0          # Timeout waiting for peers
    min_peers_for_sync: int = 1         # Minimum peers to average with
    
    # Validation
    validate_gradients: bool = True     # Enable gradient validation
    gradient_cosine_threshold: float = 0.5  # Min cosine similarity
    
    # Learning rate scheduling
    use_lr_scheduler: bool = True       # Enable cosine annealing LR
    warmup_steps: int = 1000            # LR warmup steps (linear ramp)
    min_lr_ratio: float = 0.1           # Min LR = min_lr_ratio * inner_lr
    lr_decay_steps: int = 50000         # Steps for full cosine cycle
    
    # Quorum integration
    use_quorum_cohort: bool = True      # Find cohort from quorum system
    use_freshness_weighting: bool = True  # Weight by sqrt(n) * freshness


@dataclass
class DiLoCoStats:
    """Statistics for DiLoCo training."""
    inner_step_count: int = 0
    outer_step_count: int = 0
    total_inner_steps: int = 0
    
    # Loss tracking
    inner_loss_sum: float = 0.0
    inner_loss_count: int = 0
    
    # Sync tracking
    successful_syncs: int = 0
    failed_syncs: int = 0
    local_only_outer_steps: int = 0
    
    # Timing
    inner_loop_time: float = 0.0
    outer_loop_time: float = 0.0
    sync_time: float = 0.0
    
    # Gradient stats
    avg_pseudo_grad_norm: float = 0.0
    avg_cosine_with_peers: float = 0.0
    
    @property
    def avg_inner_loss(self) -> float:
        if self.inner_loss_count == 0:
            return 0.0
        return self.inner_loss_sum / self.inner_loss_count
    
    def reset_inner_stats(self):
        """Reset stats for new inner loop."""
        self.inner_loss_sum = 0.0
        self.inner_loss_count = 0
        self.inner_step_count = 0


class OuterOptimizer:
    """
    Nesterov momentum optimizer for DiLoCo outer loop.
    
    Applies Nesterov-style momentum to pseudo-gradients:
    v_t = momentum * v_{t-1} + delta
    w_t = w_{t-1} + lr * (momentum * v_t + delta)
    
    This provides better convergence than simple averaging.
    """
    
    def __init__(
        self,
        lr: float = 0.7,
        momentum: float = 0.9,
        weight_decay: float = 0.0,
    ):
        self.lr = lr
        self.momentum = momentum
        self.weight_decay = weight_decay
        
        # Momentum buffers
        self.velocity: Dict[str, torch.Tensor] = {}
    
    def step(
        self,
        model: nn.Module,
        pseudo_gradients: Dict[str, torch.Tensor],
    ):
        """
        Apply outer optimizer step.
        
        Args:
            model: Model to update
            pseudo_gradients: Dict of name -> pseudo-gradient tensor
        """
        with torch.no_grad():
            for name, param in model.named_parameters():
                if name not in pseudo_gradients:
                    continue
                
                delta = pseudo_gradients[name]
                
                # Weight decay (applied to delta, not param)
                if self.weight_decay > 0:
                    delta = delta + self.weight_decay * param.data
                
                # Initialize velocity if needed
                if name not in self.velocity:
                    self.velocity[name] = torch.zeros_like(delta)
                
                v = self.velocity[name]
                
                # Nesterov momentum update
                # v_new = momentum * v + delta
                v.mul_(self.momentum).add_(delta)
                
                # Update: w = w + lr * (momentum * v_new + delta)
                # This is the "look ahead" part of Nesterov
                update = self.lr * (self.momentum * v + delta)
                param.data.add_(update)
                
                # Save velocity
                self.velocity[name] = v
    
    def state_dict(self) -> Dict[str, Any]:
        """Get optimizer state."""
        return {
            'lr': self.lr,
            'momentum': self.momentum,
            'velocity': {k: v.clone() for k, v in self.velocity.items()},
        }
    
    def load_state_dict(self, state: Dict[str, Any], device: str = None):
        """Load optimizer state.
        
        Args:
            state: State dict to load
            device: Target device for tensors (if None, keeps original device)
        """
        self.lr = state.get('lr', self.lr)
        self.momentum = state.get('momentum', self.momentum)
        if device:
            self.velocity = {k: v.clone().to(device) for k, v in state.get('velocity', {}).items()}
        else:
            self.velocity = {k: v.clone() for k, v in state.get('velocity', {}).items()}


class DiLoCoTrainer:
    """
    Distributed Low-Communication Training Coordinator.
    
    Manages the DiLoCo algorithm:
    1. Save initial weights at start of inner loop
    2. Train locally for N steps (inner loop)
    3. Compute pseudo-gradient (delta from initial)
    4. Sync with peers via gossip
    5. Apply outer optimizer update
    6. Repeat
    
    Integrates with:
    - SwarmRouter for peer discovery
    - ActivationBuffer for async compute
    - RobustAggregator for Byzantine-tolerant sync
    """
    
    def __init__(
        self,
        model: nn.Module,
        config: Optional[DiLoCoConfig] = None,
        inner_optimizer: Optional[torch.optim.Optimizer] = None,
        device: str = "cpu",
        quorum: Optional[Any] = None,
        dht_protocol: Optional[Any] = None,
        layer_range: Optional[Tuple[int, int]] = None,
    ):
        """
        Initialize DiLoCo trainer.
        
        Args:
            model: Model to train
            config: DiLoCo configuration
            inner_optimizer: Optimizer for inner loop (creates AdamW if None)
            device: Device for training
            quorum: Current quorum
            dht_protocol: DHT for cohort discovery
            layer_range: Layer range this node holds
        """
        self.model = model
        self.config = config or DiLoCoConfig()
        self.device = device
        
        # Quorum integration
        self.quorum = quorum
        self.dht = dht_protocol
        self.layer_range = layer_range
        
        # Inner optimizer
        if inner_optimizer is None:
            self.inner_optimizer = torch.optim.AdamW(
                model.parameters(),
                lr=self.config.inner_lr,
                weight_decay=self.config.inner_weight_decay,
                betas=(0.9, 0.95),
            )
        else:
            self.inner_optimizer = inner_optimizer
        
        # Outer optimizer
        self.outer_optimizer = OuterOptimizer(
            lr=self.config.outer_lr,
            momentum=self.config.outer_momentum,
            weight_decay=self.config.outer_weight_decay,
        )
        
        # Initial weights (saved at start of each inner loop)
        self.initial_weights: Dict[str, torch.Tensor] = {}
        
        # State
        self.phase = DiLoCoPhase.IDLE
        self.stats = DiLoCoStats()
        
        # Gradient accumulation
        self._accumulated_loss = 0.0
        self._accumulation_count = 0
        
        # Callbacks
        self._sync_callback: Optional[Callable] = None
        self._on_outer_step: Optional[Callable] = None
        
        # Learning rate scheduling
        self._base_lr = self.config.inner_lr
        self._current_lr = self._base_lr
        self._min_lr = self._base_lr * self.config.min_lr_ratio
        
        # Thread safety
        self._lock = threading.RLock()
        
        logger.info(f"DiLoCoTrainer initialized: inner_steps={self.config.inner_steps}, "
                   f"outer_lr={self.config.outer_lr}, outer_momentum={self.config.outer_momentum}")
        if self.config.use_lr_scheduler:
            logger.info(f"  LR scheduler: warmup={self.config.warmup_steps}, "
                       f"decay_steps={self.config.lr_decay_steps}, min_ratio={self.config.min_lr_ratio}")
        if quorum:
            logger.info(f"  Quorum: {quorum.quorum_id[:8]}...")
    
    def set_sync_callback(self, callback: Callable):
        """
        Set callback for pseudo-gradient synchronization.
        
        Callback signature:
            async def sync(pseudo_grads: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]
        
        Should return aggregated pseudo-gradients from peers.
        """
        self._sync_callback = callback
    
    def set_outer_step_callback(self, callback: Callable):
        """Set callback called after each outer step."""
        self._on_outer_step = callback
    
    # ==================== LIFECYCLE ====================
    
    def start_inner_loop(self):
        """Start a new inner loop by saving initial weights."""
        with self._lock:
            self._save_initial_weights()
            self.stats.reset_inner_stats()
            self.phase = DiLoCoPhase.INNER_LOOP
            
            logger.debug(f"Started inner loop {self.stats.outer_step_count + 1}")
    
    def _save_initial_weights(self):
        """Save current weights as initial for pseudo-gradient computation."""
        self.initial_weights = {}
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.initial_weights[name] = param.data.clone()
    
    # ==================== LEARNING RATE SCHEDULING ====================
    
    def _compute_lr(self, step: int) -> float:
        """
        Compute learning rate with warmup and cosine annealing.
        
        Schedule:
        1. Warmup phase (0 -> warmup_steps): Linear ramp from 0 to base_lr
        2. Decay phase (warmup_steps -> decay_steps): Cosine decay to min_lr
        3. After decay_steps: Hold at min_lr
        
        Args:
            step: Current total training step
            
        Returns:
            Learning rate for this step
        """
        import math
        
        warmup_steps = self.config.warmup_steps
        decay_steps = self.config.lr_decay_steps
        base_lr = self._base_lr
        min_lr = self._min_lr
        
        if step < warmup_steps:
            # Linear warmup
            return base_lr * (step / warmup_steps)
        elif step < decay_steps:
            # Cosine annealing decay
            progress = (step - warmup_steps) / (decay_steps - warmup_steps)
            cosine_decay = 0.5 * (1 + math.cos(math.pi * progress))
            return min_lr + (base_lr - min_lr) * cosine_decay
        else:
            # After full decay, hold at min_lr
            return min_lr
    
    def _apply_lr(self, lr: float):
        """Apply learning rate to the inner optimizer."""
        for param_group in self.inner_optimizer.param_groups:
            param_group['lr'] = lr
        self._current_lr = lr
    
    def get_current_lr(self) -> float:
        """Get the current learning rate."""
        return self._current_lr
    
    # ==================== INNER LOOP ====================
    
    def inner_step(self, loss: torch.Tensor) -> float:
        """
        Execute one inner optimization step.
        
        This is normal SGD/AdamW training - no communication needed.
        Applies learning rate scheduling if enabled.
        
        Args:
            loss: Loss tensor from forward pass
            
        Returns:
            Loss value as float
        """
        with self._lock:
            if self.phase == DiLoCoPhase.IDLE:
                self.start_inner_loop()
            
            loss_value = loss.item()
            
            # Gradient accumulation
            scaled_loss = loss / self.config.gradient_accumulation
            scaled_loss.backward()
            
            self._accumulated_loss += loss_value
            self._accumulation_count += 1
            
            # Only step optimizer after accumulation
            if self._accumulation_count >= self.config.gradient_accumulation:
                # Apply learning rate scheduling before step
                if self.config.use_lr_scheduler:
                    new_lr = self._compute_lr(self.stats.total_inner_steps)
                    self._apply_lr(new_lr)
                
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.config.max_grad_norm
                )
                
                # Inner optimizer step
                self.inner_optimizer.step()
                self.inner_optimizer.zero_grad()
                
                # Update stats
                self.stats.inner_step_count += 1
                self.stats.total_inner_steps += 1
                self.stats.inner_loss_sum += self._accumulated_loss / self._accumulation_count
                self.stats.inner_loss_count += 1
                
                # Reset accumulation
                self._accumulated_loss = 0.0
                self._accumulation_count = 0
            
            return loss_value
    
    def should_sync(self) -> bool:
        """Check if we should trigger outer sync."""
        return self.stats.inner_step_count >= self.config.inner_steps
    
    # ==================== OUTER LOOP ====================
    
    def compute_pseudo_gradient(self) -> Dict[str, torch.Tensor]:
        """
        Compute pseudo-gradient (delta from initial weights).
        
        Pseudo-gradient = current_weights - initial_weights
        This represents the DIRECTION we improved during the inner loop.
        
        The outer optimizer will then AMPLIFY this direction with momentum,
        effectively saying "we moved this way and it reduced loss, so let's
        continue moving this way".
        
        NOTE: The sign here is CRITICAL for training to work!
        - current - initial = direction we moved (positive = training progress)
        - The outer optimizer ADDs this to weights, amplifying the improvement
        
        Returns:
            Dict of name -> pseudo-gradient tensor
        """
        with self._lock:
            self.phase = DiLoCoPhase.COMPUTING_DELTA
            
            pseudo_grads = {}
            total_norm = 0.0
            
            for name, param in self.model.named_parameters():
                if name in self.initial_weights:
                    # Delta = current - initial (direction we moved during training)
                    # This is POSITIVE when training made progress
                    delta = param.data - self.initial_weights[name]
                    pseudo_grads[name] = delta
                    total_norm += delta.norm().item() ** 2
            
            # Update stats
            self.stats.avg_pseudo_grad_norm = (total_norm ** 0.5)
            
            logger.info(f"Computed pseudo-gradient: "
                       f"norm={self.stats.avg_pseudo_grad_norm:.4f}, "
                       f"params={len(pseudo_grads)}")
            
            return pseudo_grads
    
    async def sync_with_peers(
        self,
        pseudo_grads: Dict[str, torch.Tensor],
    ) -> Optional[Dict[str, torch.Tensor]]:
        """
        Synchronize pseudo-gradients with peers.
        
        Uses the sync callback to gossip and aggregate.
        
        Args:
            pseudo_grads: Local pseudo-gradients
            
        Returns:
            Aggregated pseudo-gradients from all peers, or None on failure
        """
        with self._lock:
            self.phase = DiLoCoPhase.SYNCING
        
        if self._sync_callback is None:
            logger.warning("No sync callback set - using local gradients only")
            return pseudo_grads
        
        start_time = time.time()
        
        try:
            # Call sync callback (should gossip to peers)
            aggregated = await asyncio.wait_for(
                self._sync_callback(pseudo_grads),
                timeout=self.config.sync_timeout
            )
            
            sync_time = time.time() - start_time
            self.stats.sync_time += sync_time
            self.stats.successful_syncs += 1
            
            logger.info(f"Sync completed in {sync_time:.2f}s")
            
            return aggregated
            
        except asyncio.TimeoutError:
            logger.warning(f"Sync timeout after {self.config.sync_timeout}s")
            self.stats.failed_syncs += 1
            return None
            
        except Exception as e:
            logger.error(f"Sync failed: {e}")
            self.stats.failed_syncs += 1
            return None
    
    def apply_outer_update(
        self,
        aggregated_pseudo_grads: Optional[Dict[str, torch.Tensor]] = None,
    ):
        """
        Apply outer optimizer step with aggregated pseudo-gradients.
        
        If no aggregated gradients provided, uses local pseudo-gradients.
        
        Args:
            aggregated_pseudo_grads: Aggregated pseudo-gradients from peers
        """
        with self._lock:
            self.phase = DiLoCoPhase.OUTER_STEP
            start_time = time.time()
            
            # Use local gradients if sync failed
            if aggregated_pseudo_grads is None:
                logger.warning("Using local pseudo-gradients (sync failed)")
                aggregated_pseudo_grads = self.compute_pseudo_gradient()
                self.stats.local_only_outer_steps += 1
            
            # Apply outer optimizer
            self.outer_optimizer.step(self.model, aggregated_pseudo_grads)
            
            # Update stats
            self.stats.outer_step_count += 1
            self.stats.outer_loop_time += time.time() - start_time
            
            logger.info(f"Outer step {self.stats.outer_step_count} complete "
                       f"(after {self.config.inner_steps} inner steps, "
                       f"avg_loss={self.stats.avg_inner_loss:.4f})")
            
            # Callback
            if self._on_outer_step:
                self._on_outer_step(self.stats)
            
            # Start new inner loop
            self.start_inner_loop()
    
    async def outer_step_async(self) -> bool:
        """
        Execute full outer step: compute, sync, apply.
        
        Async version that handles the full sync flow.
        
        Returns:
            True if sync succeeded, False if used local gradients
        """
        # Compute pseudo-gradient
        pseudo_grads = self.compute_pseudo_gradient()
        
        # Sync with peers
        aggregated = await self.sync_with_peers(pseudo_grads)
        
        # Apply update
        self.apply_outer_update(aggregated)
        
        return aggregated is not None
    
    def outer_step_sync(self) -> bool:
        """
        Synchronous version of outer step.
        
        Runs async outer step in new event loop.
        """
        loop = asyncio.new_event_loop()
        try:
            return loop.run_until_complete(self.outer_step_async())
        finally:
            loop.close()
    
    # ==================== UTILITIES ====================
    
    def get_stats(self) -> Dict[str, Any]:
        """Get training statistics."""
        with self._lock:
            return {
                'phase': self.phase.value,
                'inner_step_count': self.stats.inner_step_count,
                'outer_step_count': self.stats.outer_step_count,
                'total_inner_steps': self.stats.total_inner_steps,
                'avg_inner_loss': self.stats.avg_inner_loss,
                'successful_syncs': self.stats.successful_syncs,
                'failed_syncs': self.stats.failed_syncs,
                'local_only_outer_steps': self.stats.local_only_outer_steps,
                'avg_pseudo_grad_norm': self.stats.avg_pseudo_grad_norm,
                'inner_loop_time': self.stats.inner_loop_time,
                'outer_loop_time': self.stats.outer_loop_time,
                'sync_time': self.stats.sync_time,
                # Learning rate info
                'current_lr': self._current_lr,
                'base_lr': self._base_lr,
                'min_lr': self._min_lr,
                'lr_scheduler_enabled': self.config.use_lr_scheduler,
                'quorum_id': self.quorum.quorum_id if self.quorum else None,
                'layer_range': self.layer_range,
            }
    
    # ==================== QUORUM COHORT ====================
    
    def find_layer_cohort(self) -> List[str]:
        """
        Find nodes holding the same layers across different quorums.
        
        Used for DiLoCo cross-quorum sync where nodes with same layers
        exchange pseudo-gradients.
        
        Returns:
            List of peer endpoints in the cohort
        """
        if not self.layer_range or not self.dht:
            return []
        
        import hashlib
        import json
        
        # Build set of our own IPs for self-filtering
        my_ips = set()
        local_node = getattr(self.dht, 'local_node', None)
        if local_node:
            my_ips.add(local_node.ip)
        lan_ip = getattr(self.dht, '_local_lan_ip', None)
        if lan_ip:
            my_ips.add(lan_ip)
        
        # Resolve LAN addresses
        try:
            from neuroshard.core.network.nat import resolve_peer_address
            my_public_ip = local_node.ip if local_node else None
        except ImportError:
            resolve_peer_address = None
            my_public_ip = None
        
        cohort = []
        seen = set()
        layer_start, layer_end = self.layer_range
        
        # Query DHT for first layer only (all full replicas hold the same layers)
        for layer_id in [layer_start]:
            try:
                key_str = f"layer_{layer_id}"
                key_id = int(hashlib.sha1(key_str.encode()).hexdigest(), 16)
                
                # Check local storage first, then network lookup
                # The trainer's DHT may be a separate instance from P2P's DHT,
                # so local storage may miss remote nodes' announcements.
                value = None
                if hasattr(self.dht, 'storage'):
                    value = self.dht.storage.get(key_id)
                if not value and hasattr(self.dht, 'lookup_value'):
                    try:
                        value = self.dht.lookup_value(key_id)
                    except Exception:
                        pass
                
                if value:
                    try:
                        endpoints = json.loads(value) if isinstance(value, str) else [str(value)]
                    except json.JSONDecodeError:
                        endpoints = [str(value)] if value else []
                    
                    for endpoint in endpoints:
                        raw = str(endpoint)
                        if raw in seen:
                            continue
                        seen.add(raw)
                        
                        # Resolve LAN address (handle pipe format)
                        resolved = raw
                        if resolve_peer_address and my_public_ip:
                            resolved = resolve_peer_address(raw, my_public_ip)
                        elif "|" in resolved:
                            resolved = resolved.split("|", 1)[0]
                        
                        # Filter out self
                        peer_ip = resolved.rsplit(":", 1)[0] if ":" in resolved else resolved
                        if peer_ip in my_ips:
                            continue
                        
                        cohort.append(f"http://{resolved}")
            except Exception as e:
                logger.debug(f"Error finding cohort for layer {layer_id}: {e}")
        
        return cohort
    
    def set_quorum(self, quorum: Any, layer_range: Tuple[int, int]):
        """
        Update quorum assignment (e.g., after quorum reformation).
        
        Args:
            quorum: New quorum
            layer_range: New layer range
        """
        self.quorum = quorum
        self.layer_range = layer_range
        logger.info(f"DiLoCo quorum updated: {quorum.quorum_id[:8]}..., layers={layer_range}")
    
    def state_dict(self) -> Dict[str, Any]:
        """Get full trainer state for checkpointing."""
        with self._lock:
            return {
                'config': self.config.__dict__,
                'inner_optimizer': self.inner_optimizer.state_dict(),
                'outer_optimizer': self.outer_optimizer.state_dict(),
                'initial_weights': {k: v.clone() for k, v in self.initial_weights.items()},
                'stats': {
                    'inner_step_count': self.stats.inner_step_count,
                    'outer_step_count': self.stats.outer_step_count,
                    'total_inner_steps': self.stats.total_inner_steps,
                },
                'phase': self.phase.value,
            }
    
    def load_state_dict(self, state: Dict[str, Any]):
        """Load trainer state from checkpoint."""
        with self._lock:
            # Load config
            for k, v in state.get('config', {}).items():
                if hasattr(self.config, k):
                    setattr(self.config, k, v)
            
            # Load optimizers (move tensors to model's device)
            device = next(self.model.parameters()).device if list(self.model.parameters()) else 'cpu'
            if 'inner_optimizer' in state:
                self.inner_optimizer.load_state_dict(state['inner_optimizer'])
            if 'outer_optimizer' in state:
                self.outer_optimizer.load_state_dict(state['outer_optimizer'], device=str(device))
            
            # Load initial weights (move to model's device)
            device = next(self.model.parameters()).device if list(self.model.parameters()) else 'cpu'
            self.initial_weights = {
                k: v.clone().to(device) for k, v in state.get('initial_weights', {}).items()
            }
            
            # Load stats
            stats = state.get('stats', {})
            self.stats.inner_step_count = stats.get('inner_step_count', 0)
            self.stats.outer_step_count = stats.get('outer_step_count', 0)
            self.stats.total_inner_steps = stats.get('total_inner_steps', 0)
            
            # Load phase
            phase_str = state.get('phase', 'idle')
            try:
                self.phase = DiLoCoPhase(phase_str)
            except ValueError:
                self.phase = DiLoCoPhase.IDLE


# ==================== GOSSIP INTEGRATION ====================

class DiLoCoGossipProtocol:
    """
    Gossip protocol for DiLoCo pseudo-gradient synchronization.
    
    Handles the communication aspect of DiLoCo:
    - Broadcast pseudo-gradients to peers
    - Collect and aggregate responses
    - Handle stragglers with timeout
    """
    
    def __init__(
        self,
        node_id: str,
        router: Any = None,  # SwarmRouter
        min_peers: int = 1,
        timeout: float = 30.0,
    ):
        self.node_id = node_id
        self.router = router
        self.min_peers = min_peers
        self.timeout = timeout
        
        # Pending contributions
        self.pending_contributions: Dict[str, Dict[str, torch.Tensor]] = {}
        self._lock = threading.Lock()
    
    async def sync_pseudo_gradients(
        self,
        round_id: int,
        local_grads: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        """
        Synchronize pseudo-gradients with peers.
        
        Args:
            round_id: Outer step round ID
            local_grads: Local pseudo-gradients
            
        Returns:
            Averaged pseudo-gradients
        """
        # Get peers
        if self.router is None:
            logger.warning("No router - returning local gradients")
            return local_grads
        
        # Broadcast to peers
        await self._broadcast_grads(round_id, local_grads)
        
        # Wait for responses
        contributions = await self._collect_contributions(round_id)
        
        # Add our own contribution
        contributions[self.node_id] = local_grads
        
        # Aggregate
        if len(contributions) < self.min_peers:
            logger.warning(f"Only {len(contributions)} peers - below minimum {self.min_peers}")
        
        aggregated = self._aggregate_contributions(contributions)
        
        return aggregated
    
    async def _broadcast_grads(
        self,
        round_id: int,
        grads: Dict[str, torch.Tensor],
    ):
        """
        Broadcast pseudo-gradients to all peers via GossipGradient RPC.

        Serialises each parameter tensor to raw bytes and sends them
        concurrently to every known peer returned by the router.  TTL is
        set to 1 so peers do NOT re-forward (we are the origin).
        """
        if self.router is None:
            return

        # Collect destination addresses from router's peer cache
        try:
            peers = list(self.router.peer_stats.values())
        except Exception:
            return

        if not peers:
            return

        # Serialise gradients: param_name -> raw float32 bytes
        serialised: Dict[str, bytes] = {}
        for name, tensor in grads.items():
            buf = io.BytesIO()
            torch.save(tensor.cpu().to(torch.float32), buf)
            serialised[name] = buf.getvalue()

        # Import proto lazily to avoid circular imports at module load time
        try:
            from neuroshard.protos import neuroshard_pb2, neuroshard_pb2_grpc
            from neuroshard.core.network.connection_pool import get_channel
        except ImportError:
            logger.warning("Proto/gRPC not available — skipping gradient broadcast")
            return

        req = neuroshard_pb2.GossipGradientRequest(
            node_id=self.node_id,
            round_id=round_id,
            timestamp=time.time(),
            batch_size=1,
            loss=0.0,
            layer_gradients=serialised,
            ttl=1,  # Do not re-forward
        )

        async def _send_to(addr: str):
            try:
                channel = get_channel(addr)
                stub = neuroshard_pb2_grpc.NeuroShardServiceStub(channel)
                await asyncio.get_event_loop().run_in_executor(
                    None,
                    lambda: stub.GossipGradient(req, timeout=self.timeout),
                )
            except Exception as e:
                logger.debug(f"GossipGradient to {addr} failed: {e}")

        await asyncio.gather(*[_send_to(p.grpc_addr) for p in peers], return_exceptions=True)
    
    async def _collect_contributions(
        self,
        round_id: int,
    ) -> Dict[str, Dict[str, torch.Tensor]]:
        """Collect contributions from peers."""
        # Wait up to timeout for contributions
        start = time.time()
        
        while time.time() - start < self.timeout:
            with self._lock:
                if len(self.pending_contributions) >= self.min_peers:
                    contributions = dict(self.pending_contributions)
                    self.pending_contributions.clear()
                    return contributions
            
            await asyncio.sleep(0.1)
        
        # Timeout - return what we have
        with self._lock:
            contributions = dict(self.pending_contributions)
            self.pending_contributions.clear()
            return contributions
    
    def _aggregate_contributions(
        self,
        contributions: Dict[str, Dict[str, torch.Tensor]],
    ) -> Dict[str, torch.Tensor]:
        """Average contributions from all peers."""
        if not contributions:
            return {}
        
        # Get all param names
        param_names = set()
        for grads in contributions.values():
            param_names.update(grads.keys())
        
        # Average each parameter
        aggregated = {}
        for name in param_names:
            tensors = [
                grads[name] for grads in contributions.values()
                if name in grads
            ]
            if tensors:
                aggregated[name] = torch.stack(tensors).mean(dim=0)
        
        return aggregated
    
    def receive_contribution(
        self,
        peer_id: str,
        grads: Dict[str, torch.Tensor],
    ):
        """Receive pseudo-gradient contribution from peer."""
        with self._lock:
            self.pending_contributions[peer_id] = grads


# ==================== FACTORY FUNCTIONS ====================

def create_diloco_trainer(
    model: nn.Module,
    inner_steps: int = 500,
    outer_lr: float = 0.7,
    inner_lr: float = 1e-4,
    device: str = "cpu",
    **config_kwargs,
) -> DiLoCoTrainer:
    """
    Factory function to create a DiLoCo trainer.
    
    Args:
        model: Model to train
        inner_steps: Steps before each sync
        outer_lr: Outer optimizer learning rate
        inner_lr: Inner optimizer learning rate
        device: Training device
        **config_kwargs: Additional config options
        
    Returns:
        Configured DiLoCoTrainer
    """
    config = DiLoCoConfig(
        inner_steps=inner_steps,
        outer_lr=outer_lr,
        inner_lr=inner_lr,
        **config_kwargs,
    )
    
    return DiLoCoTrainer(model, config=config, device=device)

