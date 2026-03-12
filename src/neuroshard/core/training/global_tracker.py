"""
Global Training Tracker - Verify Distributed Training is Working

This module provides network-wide training verification:
1. Tracks loss across ALL nodes in the network
2. Monitors model hash convergence (ensures nodes sync)
3. Computes global training metrics (not just local batch loss)
4. Provides dashboards/APIs for monitoring

Key Concepts:
- Moving Average Loss: Smoothed loss over time (local)
- Global Loss: Average loss across all network nodes
- Model Hash: SHA256 of model weights (should converge across nodes)
- Sync Rate: How often nodes successfully sync gradients
"""

import time
import hashlib
import threading
import logging
import json
from collections import deque
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple
from pathlib import Path
import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


@dataclass
class TrainingSnapshot:
    """A snapshot of training state at a point in time."""
    timestamp: float
    node_id: str
    
    # Loss metrics
    batch_loss: float              # Raw batch loss
    moving_avg_loss: float         # Smoothed loss (EMA)
    min_loss_seen: float           # Best loss achieved
    
    # Training progress
    training_step: int             # Global step count
    inner_step: int                # DiLoCo inner step (0-500)
    outer_step: int                # DiLoCo outer step (sync count)
    
    # Convergence metrics
    model_hash: str                # Hash of model weights
    gradient_norm: float           # L2 norm of gradients
    
    # Data coverage
    shard_id: int                  # Current data shard
    tokens_trained: int            # Total tokens seen


@dataclass 
class GlobalTrainingStats:
    """Network-wide training statistics."""
    # Aggregated metrics
    global_avg_loss: float = 0.0
    global_min_loss: float = float('inf')
    
    # Convergence tracking
    model_hashes: Dict[str, str] = field(default_factory=dict)  # node_id -> hash
    hash_agreement_rate: float = 0.0  # % of nodes with same hash
    
    # Network health
    total_nodes_training: int = 0
    successful_syncs: int = 0
    failed_syncs: int = 0
    
    # Progress
    global_steps: int = 0
    global_tokens: int = 0
    data_shards_covered: set = field(default_factory=set)
    
    # Time tracking
    last_update: float = 0.0


class GlobalTrainingTracker:
    """
    Tracks and verifies distributed training across the network.
    
    Each node runs this to:
    1. Track its own training progress
    2. Receive training stats from peers via gossip
    3. Compute global metrics
    4. Verify convergence (all nodes should have similar model hash)
    
    Usage:
        tracker = GlobalTrainingTracker(node_id, model)
        
        # During training
        tracker.record_step(loss, step, shard_id)
        
        # Get status
        status = tracker.get_global_status()
        print(f"Global loss: {status['global_avg_loss']:.4f}")
        print(f"Hash agreement: {status['hash_agreement_rate']*100:.1f}%")
    """
    
    # EMA smoothing factor (lower = smoother)
    EMA_ALPHA = 0.1
    
    # History window for computing metrics
    HISTORY_WINDOW = 100
    
    # Minimum nodes for global metrics
    MIN_NODES_FOR_GLOBAL = 1
    
    def __init__(
        self,
        node_id: str,
        model: nn.Module,
        checkpoint_dir: Optional[Path] = None,
    ):
        self.node_id = node_id
        self.model = model
        self.checkpoint_dir = checkpoint_dir or Path.home() / ".neuroshard" / "training_logs"
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # Local tracking
        self._local_history: deque = deque(maxlen=self.HISTORY_WINDOW)
        self._moving_avg_loss = 0.0
        self._min_loss = float('inf')
        self._total_steps = 0
        self._total_tokens = 0
        self._current_shard = 0
        
        # Peer tracking (received via gossip)
        self._peer_stats: Dict[str, TrainingSnapshot] = {}
        self._peer_stats_lock = threading.Lock()
        
        # Global aggregated stats
        self._global_stats = GlobalTrainingStats()
        
        # Sync tracking
        self._sync_history: deque = deque(maxlen=50)  # (timestamp, success)
        self._last_model_hash = ""
        
        # Verification
        self._loss_checkpoints: List[Tuple[int, float]] = []  # (step, loss)
        
        # Try to restore previous state
        self._load_state()
        
        logger.info(f"GlobalTrainingTracker initialized for node {node_id[:8]}...")
    
    def _get_state_path(self) -> Path:
        """Get path to state file."""
        return self.checkpoint_dir / f"tracker_state_{self.node_id[:16]}.json"
    
    def _save_state(self):
        """Persist tracker state to disk."""
        try:
            state = {
                "node_id": self.node_id,
                "saved_at": time.time(),
                "moving_avg_loss": self._moving_avg_loss,
                "min_loss": self._min_loss if self._min_loss != float('inf') else None,
                "total_steps": self._total_steps,
                "total_tokens": self._total_tokens,
                "current_shard": self._current_shard,
                "last_model_hash": self._last_model_hash,
                "loss_checkpoints": self._loss_checkpoints[-100:],  # Keep last 100
                "global_stats": {
                    "global_avg_loss": self._global_stats.global_avg_loss,
                    "global_min_loss": self._global_stats.global_min_loss if self._global_stats.global_min_loss != float('inf') else None,
                    "total_nodes_training": self._global_stats.total_nodes_training,
                    "global_steps": self._global_stats.global_steps,
                    "global_tokens": self._global_stats.global_tokens,
                    "successful_syncs": self._global_stats.successful_syncs,
                    "failed_syncs": self._global_stats.failed_syncs,
                },
            }
            
            with open(self._get_state_path(), 'w') as f:
                json.dump(state, f, indent=2, default=str)
            
            logger.debug(f"Tracker state saved: {self._total_steps} steps, loss={self._moving_avg_loss:.4f}")
        except Exception as e:
            logger.warning(f"Failed to save tracker state: {e}")
    
    def _load_state(self):
        """Load tracker state from disk."""
        path = self._get_state_path()
        if not path.exists():
            logger.debug("No previous tracker state found")
            return
        
        try:
            with open(path, 'r') as f:
                state = json.load(f)
            
            # Restore local state
            self._moving_avg_loss = state.get("moving_avg_loss", 0.0)
            self._min_loss = state.get("min_loss") or float('inf')
            self._total_steps = state.get("total_steps", 0)
            self._total_tokens = state.get("total_tokens", 0)
            self._current_shard = state.get("current_shard", 0)
            self._last_model_hash = state.get("last_model_hash", "")
            self._loss_checkpoints = state.get("loss_checkpoints", [])
            
            # Restore global stats
            global_stats = state.get("global_stats", {})
            self._global_stats.global_avg_loss = global_stats.get("global_avg_loss", 0.0)
            self._global_stats.global_min_loss = global_stats.get("global_min_loss") or float('inf')
            self._global_stats.total_nodes_training = global_stats.get("total_nodes_training", 0)
            self._global_stats.global_steps = global_stats.get("global_steps", 0)
            self._global_stats.global_tokens = global_stats.get("global_tokens", 0)
            self._global_stats.successful_syncs = global_stats.get("successful_syncs", 0)
            self._global_stats.failed_syncs = global_stats.get("failed_syncs", 0)
            
            logger.info(f"Restored tracker state: {self._total_steps} steps, avg_loss={self._moving_avg_loss:.4f}")
        except Exception as e:
            logger.warning(f"Failed to load tracker state: {e}")
    
    def record_step(
        self,
        loss: float,
        step: int,
        shard_id: int = 0,
        tokens_in_batch: int = 0,
        gradient_norm: Optional[float] = None,
        inner_step: int = 0,
        outer_step: int = 0,
    ) -> TrainingSnapshot:
        """
        Record a training step and update metrics.
        
        Args:
            loss: Raw loss value for this batch
            step: Global training step number
            shard_id: Data shard being trained on
            tokens_in_batch: Tokens processed in this batch
            gradient_norm: Optional gradient L2 norm
            inner_step: DiLoCo inner step (0-500)
            outer_step: DiLoCo outer sync count
            
        Returns:
            TrainingSnapshot with current state
        """
        # Update EMA loss
        if self._moving_avg_loss == 0.0:
            self._moving_avg_loss = loss
        else:
            self._moving_avg_loss = (
                self.EMA_ALPHA * loss + 
                (1 - self.EMA_ALPHA) * self._moving_avg_loss
            )
        
        # Track minimum loss
        if loss < self._min_loss:
            self._min_loss = loss
        
        # Update counters
        self._total_steps = step
        self._total_tokens += tokens_in_batch
        self._current_shard = shard_id
        
        # Compute model hash (every 50 steps to save compute)
        if step % 50 == 0:
            self._last_model_hash = self._compute_model_hash()
        
        # Create snapshot
        snapshot = TrainingSnapshot(
            timestamp=time.time(),
            node_id=self.node_id,
            batch_loss=loss,
            moving_avg_loss=self._moving_avg_loss,
            min_loss_seen=self._min_loss,
            training_step=step,
            inner_step=inner_step,
            outer_step=outer_step,
            model_hash=self._last_model_hash,
            gradient_norm=gradient_norm or 0.0,
            shard_id=shard_id,
            tokens_trained=self._total_tokens,
        )
        
        self._local_history.append(snapshot)
        
        # Record loss checkpoint every 100 steps
        if step % 100 == 0:
            self._loss_checkpoints.append((step, self._moving_avg_loss))
            # Keep only last 100 checkpoints
            if len(self._loss_checkpoints) > 100:
                self._loss_checkpoints = self._loss_checkpoints[-100:]
        
        # Update global stats
        self._update_global_stats()
        
        # Periodically save state (every 10 steps to match checkpoint frequency)
        if step % 10 == 0:
            self._save_state()
        
        return snapshot
    
    def _compute_model_hash(self) -> str:
        """Compute SHA256 hash of model weights (sampled for speed)."""
        hasher = hashlib.sha256()
        
        # Sample some parameters for speed
        params_to_hash = list(self.model.named_parameters())[:10]
        
        for name, param in params_to_hash:
            hasher.update(name.encode())
            # Sample first 1000 values
            data = param.data.flatten()[:1000].cpu().numpy().tobytes()
            hasher.update(data)
        
        return hasher.hexdigest()[:16]
    
    def receive_peer_stats(self, peer_id: str, snapshot_data: Dict[str, Any]):
        """
        Receive training stats from a peer via gossip.
        
        Args:
            peer_id: ID of peer node
            snapshot_data: Serialized TrainingSnapshot
        """
        with self._peer_stats_lock:
            snapshot = TrainingSnapshot(
                timestamp=snapshot_data.get("timestamp", time.time()),
                node_id=peer_id,
                batch_loss=snapshot_data.get("batch_loss", 0.0),
                moving_avg_loss=snapshot_data.get("moving_avg_loss", 0.0),
                min_loss_seen=snapshot_data.get("min_loss_seen", float('inf')),
                training_step=snapshot_data.get("training_step", 0),
                inner_step=snapshot_data.get("inner_step", 0),
                outer_step=snapshot_data.get("outer_step", 0),
                model_hash=snapshot_data.get("model_hash", ""),
                gradient_norm=snapshot_data.get("gradient_norm", 0.0),
                shard_id=snapshot_data.get("shard_id", 0),
                tokens_trained=snapshot_data.get("tokens_trained", 0),
            )
            
            self._peer_stats[peer_id] = snapshot
            
            # Clean up stale peers (>5 min old)
            now = time.time()
            stale_peers = [
                pid for pid, s in self._peer_stats.items()
                if now - s.timestamp > 300
            ]
            for pid in stale_peers:
                del self._peer_stats[pid]
        
        self._update_global_stats()
    
    def record_sync_result(self, success: bool, peers_synced: int = 0):
        """Record a gradient sync attempt result."""
        self._sync_history.append((time.time(), success, peers_synced))
        
        if success:
            self._global_stats.successful_syncs += 1
        else:
            self._global_stats.failed_syncs += 1
        
        # Persist state after each sync (important milestone)
        self._save_state()
    
    def _update_global_stats(self):
        """Recompute global statistics from local + peer data."""
        with self._peer_stats_lock:
            # Collect all snapshots (local + peers)
            all_snapshots = list(self._peer_stats.values())
            
            # Add our own latest
            if self._local_history:
                all_snapshots.append(self._local_history[-1])
        
        if not all_snapshots:
            return
        
        # Compute global averages
        self._global_stats.global_avg_loss = sum(
            s.moving_avg_loss for s in all_snapshots
        ) / len(all_snapshots)
        
        self._global_stats.global_min_loss = min(
            s.min_loss_seen for s in all_snapshots
        )
        
        self._global_stats.total_nodes_training = len(all_snapshots)
        
        self._global_stats.global_steps = max(
            s.training_step for s in all_snapshots
        )
        
        self._global_stats.global_tokens = sum(
            s.tokens_trained for s in all_snapshots
        )
        
        # Track data shard coverage
        self._global_stats.data_shards_covered = set(
            s.shard_id for s in all_snapshots
        )
        
        # Check model hash convergence
        hashes = [s.model_hash for s in all_snapshots if s.model_hash]
        if hashes:
            # Count most common hash
            from collections import Counter
            hash_counts = Counter(hashes)
            most_common_hash, count = hash_counts.most_common(1)[0]
            self._global_stats.hash_agreement_rate = count / len(hashes)
            self._global_stats.model_hashes = {
                s.node_id: s.model_hash for s in all_snapshots
            }
        
        self._global_stats.last_update = time.time()
    
    @staticmethod
    def _sanitize_float(value: float) -> Optional[float]:
        """Convert inf/nan to None for JSON serialization."""
        import math
        if value is None or math.isinf(value) or math.isnan(value):
            return None
        return value
    
    def get_local_status(self) -> Dict[str, Any]:
        """Get this node's training status."""
        return {
            "node_id": self.node_id,
            "training_step": self._total_steps,
            "moving_avg_loss": self._sanitize_float(self._moving_avg_loss),
            "min_loss_seen": self._sanitize_float(self._min_loss),
            "tokens_trained": self._total_tokens,
            "current_shard": self._current_shard,
            "model_hash": self._last_model_hash,
            "loss_trend": self._compute_loss_trend(),
        }
    
    def get_global_status(self) -> Dict[str, Any]:
        """
        Get network-wide training status.
        
        This is the key method for verifying distributed training is working.
        
        Returns:
            Dict with:
            - global_avg_loss: Average loss across all nodes
            - global_min_loss: Best loss achieved by any node
            - hash_agreement_rate: % of nodes with same model hash (should be 100%)
            - total_nodes_training: Number of active nodes
            - is_converging: Whether the network appears to be converging
            - training_verified: Whether training is definitely improving the model
        """
        # Check if training is actually improving
        is_converging = self._check_convergence()
        training_verified = self._verify_training()
        
        return {
            # Global metrics (sanitized for JSON)
            "global_avg_loss": self._sanitize_float(self._global_stats.global_avg_loss),
            "global_min_loss": self._sanitize_float(self._global_stats.global_min_loss),
            
            # Convergence
            "hash_agreement_rate": self._global_stats.hash_agreement_rate,
            "model_hashes": dict(self._global_stats.model_hashes),
            
            # Network health
            "total_nodes_training": self._global_stats.total_nodes_training,
            "successful_syncs": self._global_stats.successful_syncs,
            "failed_syncs": self._global_stats.failed_syncs,
            "sync_success_rate": (
                self._global_stats.successful_syncs / 
                max(1, self._global_stats.successful_syncs + self._global_stats.failed_syncs)
            ),
            
            # Progress
            "global_steps": self._global_stats.global_steps,
            "global_tokens": self._global_stats.global_tokens,
            "data_shards_covered": list(self._global_stats.data_shards_covered),
            
            # Verification
            "is_converging": is_converging,
            "training_verified": training_verified,
            "loss_trend": self._compute_loss_trend(),
            
            # Timestamp
            "last_update": self._global_stats.last_update,
        }
    
    def _compute_loss_trend(self) -> str:
        """Compute loss trend over recent history."""
        if len(self._loss_checkpoints) < 2:
            return "insufficient_data"
        
        # Compare first half to second half
        mid = len(self._loss_checkpoints) // 2
        first_half_avg = sum(l for _, l in self._loss_checkpoints[:mid]) / mid
        second_half_avg = sum(l for _, l in self._loss_checkpoints[mid:]) / (len(self._loss_checkpoints) - mid)
        
        improvement = (first_half_avg - second_half_avg) / first_half_avg if first_half_avg > 0 else 0
        
        if improvement > 0.1:
            return "improving_strongly"
        elif improvement > 0.02:
            return "improving"
        elif improvement > -0.02:
            return "stable"
        elif improvement > -0.1:
            return "degrading_slightly"
        else:
            return "degrading"
    
    def _check_convergence(self) -> bool:
        """Check if the network appears to be converging."""
        # Need at least 2 nodes with matching hashes
        if self._global_stats.hash_agreement_rate < 0.5:
            return False
        
        # Loss should be trending down
        trend = self._compute_loss_trend()
        return trend in ["improving", "improving_strongly", "stable"]
    
    def _verify_training(self) -> bool:
        """
        Verify that training is actually improving the model.
        
        Returns True if we can confirm the model is learning.
        """
        # Need sufficient data
        if len(self._loss_checkpoints) < 5:
            return False
        
        # Check that loss has decreased overall
        first_losses = [l for _, l in self._loss_checkpoints[:3]]
        recent_losses = [l for _, l in self._loss_checkpoints[-3:]]
        
        first_avg = sum(first_losses) / len(first_losses)
        recent_avg = sum(recent_losses) / len(recent_losses)
        
        # Loss should have decreased by at least 10%
        return recent_avg < first_avg * 0.9
    
    def get_snapshot_for_gossip(self) -> Dict[str, Any]:
        """Get current snapshot data to send to peers."""
        if not self._local_history:
            return {}
        
        latest = self._local_history[-1]
        return {
            "timestamp": latest.timestamp,
            "batch_loss": latest.batch_loss,
            "moving_avg_loss": latest.moving_avg_loss,
            "min_loss_seen": latest.min_loss_seen,
            "training_step": latest.training_step,
            "inner_step": latest.inner_step,
            "outer_step": latest.outer_step,
            "model_hash": latest.model_hash,
            "gradient_norm": latest.gradient_norm,
            "shard_id": latest.shard_id,
            "tokens_trained": latest.tokens_trained,
        }
    
    def save_training_log(self, filename: str = None):
        """Save training history to disk for analysis."""
        if filename is None:
            filename = f"training_log_{self.node_id[:8]}_{int(time.time())}.json"
        
        filepath = self.checkpoint_dir / filename
        
        log_data = {
            "node_id": self.node_id,
            "saved_at": time.time(),
            "local_status": self.get_local_status(),
            "global_status": self.get_global_status(),
            "loss_checkpoints": self._loss_checkpoints,
            "sync_history": list(self._sync_history),
        }
        
        with open(filepath, 'w') as f:
            json.dump(log_data, f, indent=2, default=str)
        
        logger.info(f"Training log saved to {filepath}")
        return filepath


def format_training_status(tracker: GlobalTrainingTracker) -> str:
    """Format training status for display."""
    local = tracker.get_local_status()
    global_stats = tracker.get_global_status()
    
    lines = [
        "=" * 60,
        "NEUROSHARD GLOBAL TRAINING STATUS",
        "=" * 60,
        "",
        "LOCAL NODE:",
        f"  Step: {local['training_step']:,}",
        f"  Loss: {local['moving_avg_loss']:.4f} (min: {local['min_loss_seen']:.4f})",
        f"  Tokens: {local['tokens_trained']:,}",
        f"  Trend: {local['loss_trend']}",
        f"  Model Hash: {local['model_hash']}",
        "",
        "GLOBAL NETWORK:",
        f"  Nodes Training: {global_stats['total_nodes_training']}",
        f"  Global Avg Loss: {global_stats['global_avg_loss']:.4f}",
        f"  Global Min Loss: {global_stats['global_min_loss']:.4f}",
        f"  Hash Agreement: {global_stats['hash_agreement_rate']*100:.1f}%",
        f"  Shards Covered: {len(global_stats['data_shards_covered'])}",
        "",
        "VERIFICATION:",
        f"  Is Converging: {'✓' if global_stats['is_converging'] else '✗'}",
        f"  Training Verified: {'✓' if global_stats['training_verified'] else '✗'}",
        f"  Sync Success Rate: {global_stats['sync_success_rate']*100:.1f}%",
        "",
        "=" * 60,
    ]
    
    return "\n".join(lines)

