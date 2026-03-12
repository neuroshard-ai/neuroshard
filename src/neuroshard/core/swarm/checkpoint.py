"""
Speculative Checkpointing - High-Frequency Snapshots for Fast Recovery

Implements background checkpointing for resilient distributed training:
- Saves snapshots every 2 minutes (configurable)
- Keeps rolling window of last N snapshots
- Announces availability to DHT for peer recovery
- Enables fast crash recovery vs full restart

Key Insight: "Cheaper to over-checkpoint than to re-train."

On crash, neighbors can fetch the "hot" snapshot and resume
with minimal loss of training progress.

Usage:
    checkpointer = SpeculativeCheckpointer(
        model=model,
        optimizer=optimizer,
        diloco_trainer=trainer,
        checkpoint_dir="/path/to/checkpoints"
    )
    checkpointer.start()
    
    # On crash recovery:
    checkpoint = await checkpointer.fetch_neighbor_snapshot(peer_id)
    checkpointer.restore_from_checkpoint(checkpoint)
"""

import asyncio
import gzip
import hashlib
import io
import logging
import os
import shutil
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Any, Callable, Tuple
from enum import Enum

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class CheckpointType(Enum):
    """Type of checkpoint."""
    HOT = "hot"          # Frequent speculative snapshot
    COLD = "cold"        # Less frequent, more complete
    RECOVERY = "recovery"  # Fetched from peer for recovery


@dataclass
class CheckpointMetadata:
    """Metadata for a checkpoint."""
    checkpoint_id: str
    timestamp: float
    checkpoint_type: CheckpointType
    
    # Training state
    training_step: int
    outer_step: int
    inner_step: int
    
    # Model info
    model_hash: str
    num_params: int
    layer_ids: List[int]
    
    # Storage
    file_path: str
    compressed_size: int
    original_size: int
    
    # Node info
    node_id: str
    
    @property
    def age_seconds(self) -> float:
        return time.time() - self.timestamp
    
    @property
    def compression_ratio(self) -> float:
        if self.original_size == 0:
            return 1.0
        return self.compressed_size / self.original_size


@dataclass
class CheckpointConfig:
    """Configuration for speculative checkpointing."""
    # Timing
    snapshot_interval: float = 120.0     # 2 minutes
    cold_checkpoint_interval: float = 3600.0  # 1 hour
    
    # Storage
    max_hot_snapshots: int = 5           # Keep last 5 hot snapshots
    max_cold_checkpoints: int = 3        # Keep last 3 cold checkpoints
    checkpoint_dir: str = "./checkpoints"
    
    # Compression
    compression_level: int = 6           # gzip compression (1-9)
    
    # Networking
    announce_to_dht: bool = True         # Announce availability
    serve_to_peers: bool = True          # Allow peers to fetch
    
    # Recovery
    auto_fetch_on_start: bool = True     # Try to fetch from peers on start
    recovery_timeout: float = 60.0       # Timeout for fetching


class SpeculativeCheckpointer:
    """
    Background checkpointer for resilient distributed training.
    
    Runs in a background thread, periodically saving:
    - Hot snapshots (every 2 minutes) - for fast recovery
    - Cold checkpoints (hourly) - more complete, for long-term storage
    
    Integrates with:
    - DiLoCoTrainer for training state
    - P2P/DHT for checkpoint announcement and fetching
    - gRPC for serving checkpoints to peers
    """
    
    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        diloco_trainer: Optional[Any] = None,  # DiLoCoTrainer
        config: Optional[CheckpointConfig] = None,
        node_id: str = "",
        p2p_manager: Optional[Any] = None,
    ):
        """
        Initialize speculative checkpointer.
        
        Args:
            model: Model to checkpoint
            optimizer: Optimizer to checkpoint
            diloco_trainer: Optional DiLoCo trainer for additional state
            config: Checkpoint configuration
            node_id: This node's ID
            p2p_manager: P2P manager for DHT announcements
        """
        self.model = model
        self.optimizer = optimizer
        self.diloco = diloco_trainer
        self.config = config or CheckpointConfig()
        self.node_id = node_id
        self.p2p = p2p_manager
        
        # Ensure checkpoint directory exists
        self.checkpoint_dir = Path(self.config.checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # Snapshot tracking
        self.hot_snapshots: List[CheckpointMetadata] = []
        self.cold_checkpoints: List[CheckpointMetadata] = []
        
        # Current state
        self.training_step = 0
        self.outer_step = 0
        self.inner_step = 0
        
        # Background thread
        self.running = False
        self._thread: Optional[threading.Thread] = None
        self._last_hot_snapshot = 0.0
        self._last_cold_checkpoint = 0.0
        
        # Stats
        self.snapshots_saved = 0
        self.snapshots_served = 0
        self.recoveries_performed = 0
        
        # Lock for thread safety
        self._lock = threading.RLock()
        
        logger.info(f"SpeculativeCheckpointer initialized: "
                   f"hot_interval={self.config.snapshot_interval}s, "
                   f"cold_interval={self.config.cold_checkpoint_interval}s")
    
    # ==================== LIFECYCLE ====================
    
    def start(self):
        """Start background checkpointing."""
        if self.running:
            return
        
        self.running = True
        self._thread = threading.Thread(
            target=self._checkpoint_loop,
            daemon=True,
            name="SpeculativeCheckpointer"
        )
        self._thread.start()
        
        logger.info("Speculative checkpointing started")
    
    def stop(self):
        """Stop background checkpointing."""
        self.running = False
        if self._thread:
            self._thread.join(timeout=5.0)
        
        logger.info("Speculative checkpointing stopped")
    
    def _checkpoint_loop(self):
        """Background loop for periodic checkpointing."""
        while self.running:
            try:
                now = time.time()
                
                # Hot snapshot check
                if (now - self._last_hot_snapshot) >= self.config.snapshot_interval:
                    self._save_hot_snapshot()
                    self._last_hot_snapshot = now
                
                # Cold checkpoint check
                if (now - self._last_cold_checkpoint) >= self.config.cold_checkpoint_interval:
                    self._save_cold_checkpoint()
                    self._last_cold_checkpoint = now
                
                # Sleep for a bit
                time.sleep(10)  # Check every 10 seconds
                
            except Exception as e:
                logger.error(f"Checkpoint loop error: {e}")
                time.sleep(30)  # Back off on error
    
    # ==================== SAVING ====================
    
    def _save_hot_snapshot(self) -> Optional[CheckpointMetadata]:
        """Save a hot snapshot for fast recovery."""
        with self._lock:
            try:
                checkpoint_id = f"hot_{int(time.time())}_{self.node_id[:8]}"
                filename = f"{checkpoint_id}.pt.gz"
                filepath = self.checkpoint_dir / filename
                
                # Build checkpoint
                checkpoint = self._build_checkpoint(CheckpointType.HOT)
                
                # Save compressed
                original_size, compressed_size = self._save_compressed(
                    checkpoint, filepath
                )
                
                # Create metadata
                metadata = CheckpointMetadata(
                    checkpoint_id=checkpoint_id,
                    timestamp=time.time(),
                    checkpoint_type=CheckpointType.HOT,
                    training_step=self.training_step,
                    outer_step=self.outer_step,
                    inner_step=self.inner_step,
                    model_hash=self._compute_model_hash(),
                    num_params=sum(p.numel() for p in self.model.parameters()),
                    layer_ids=self._get_layer_ids(),
                    file_path=str(filepath),
                    compressed_size=compressed_size,
                    original_size=original_size,
                    node_id=self.node_id,
                )
                
                # Track snapshot
                self.hot_snapshots.append(metadata)
                self.snapshots_saved += 1
                
                # Cleanup old snapshots
                self._cleanup_old_snapshots()
                
                # Announce to DHT
                if self.config.announce_to_dht:
                    self._announce_checkpoint(metadata)
                
                logger.info(f"Hot snapshot saved: {checkpoint_id} "
                           f"({compressed_size/1024:.1f}KB, "
                           f"ratio={metadata.compression_ratio:.2f})")
                
                return metadata
                
            except Exception as e:
                logger.error(f"Failed to save hot snapshot: {e}")
                return None
    
    def _save_cold_checkpoint(self) -> Optional[CheckpointMetadata]:
        """Save a cold checkpoint with full state."""
        with self._lock:
            try:
                checkpoint_id = f"cold_{int(time.time())}_{self.node_id[:8]}"
                filename = f"{checkpoint_id}.pt.gz"
                filepath = self.checkpoint_dir / filename
                
                # Build checkpoint (more complete than hot)
                checkpoint = self._build_checkpoint(CheckpointType.COLD)
                
                # Save compressed
                original_size, compressed_size = self._save_compressed(
                    checkpoint, filepath
                )
                
                # Create metadata
                metadata = CheckpointMetadata(
                    checkpoint_id=checkpoint_id,
                    timestamp=time.time(),
                    checkpoint_type=CheckpointType.COLD,
                    training_step=self.training_step,
                    outer_step=self.outer_step,
                    inner_step=self.inner_step,
                    model_hash=self._compute_model_hash(),
                    num_params=sum(p.numel() for p in self.model.parameters()),
                    layer_ids=self._get_layer_ids(),
                    file_path=str(filepath),
                    compressed_size=compressed_size,
                    original_size=original_size,
                    node_id=self.node_id,
                )
                
                # Track checkpoint
                self.cold_checkpoints.append(metadata)
                
                # Cleanup old checkpoints
                self._cleanup_old_checkpoints()
                
                logger.info(f"Cold checkpoint saved: {checkpoint_id} "
                           f"({compressed_size/1024/1024:.1f}MB)")
                
                return metadata
                
            except Exception as e:
                logger.error(f"Failed to save cold checkpoint: {e}")
                return None
    
    def _build_checkpoint(self, checkpoint_type: CheckpointType) -> Dict[str, Any]:
        """Build checkpoint dictionary."""
        checkpoint = {
            'checkpoint_type': checkpoint_type.value,
            'timestamp': time.time(),
            'node_id': self.node_id,
            
            # Model state
            'model_state_dict': self.model.state_dict(),
            
            # Optimizer state
            'optimizer_state_dict': self.optimizer.state_dict(),
            
            # Training progress
            'training_step': self.training_step,
            'outer_step': self.outer_step,
            'inner_step': self.inner_step,
        }
        
        # Add DiLoCo state if available
        if self.diloco is not None:
            checkpoint['diloco_state'] = {
                'initial_weights': {
                    k: v.clone() for k, v in self.diloco.initial_weights.items()
                },
                'outer_optimizer': self.diloco.outer_optimizer.state_dict(),
                'stats': {
                    'inner_step_count': self.diloco.stats.inner_step_count,
                    'outer_step_count': self.diloco.stats.outer_step_count,
                    'total_inner_steps': self.diloco.stats.total_inner_steps,
                },
                'phase': self.diloco.phase.value,
            }
        
        # For cold checkpoints, add extra info
        if checkpoint_type == CheckpointType.COLD:
            checkpoint['model_hash'] = self._compute_model_hash()
            checkpoint['layer_ids'] = self._get_layer_ids()
        
        return checkpoint
    
    def _save_compressed(
        self,
        checkpoint: Dict[str, Any],
        filepath: Path,
    ) -> Tuple[int, int]:
        """
        Save checkpoint with gzip compression.
        
        Returns:
            (original_size, compressed_size)
        """
        # Serialize to buffer
        buffer = io.BytesIO()
        torch.save(checkpoint, buffer)
        original_data = buffer.getvalue()
        original_size = len(original_data)
        
        # Compress
        compressed_data = gzip.compress(
            original_data,
            compresslevel=self.config.compression_level
        )
        compressed_size = len(compressed_data)
        
        # Write to file
        with open(filepath, 'wb') as f:
            f.write(compressed_data)
        
        return original_size, compressed_size
    
    # ==================== CLEANUP ====================
    
    def _cleanup_old_snapshots(self):
        """Remove old hot snapshots beyond max limit."""
        while len(self.hot_snapshots) > self.config.max_hot_snapshots:
            oldest = self.hot_snapshots.pop(0)
            try:
                Path(oldest.file_path).unlink(missing_ok=True)
                logger.debug(f"Removed old snapshot: {oldest.checkpoint_id}")
            except Exception as e:
                logger.warning(f"Failed to remove snapshot: {e}")
    
    def _cleanup_old_checkpoints(self):
        """Remove old cold checkpoints beyond max limit."""
        while len(self.cold_checkpoints) > self.config.max_cold_checkpoints:
            oldest = self.cold_checkpoints.pop(0)
            try:
                Path(oldest.file_path).unlink(missing_ok=True)
                logger.debug(f"Removed old checkpoint: {oldest.checkpoint_id}")
            except Exception as e:
                logger.warning(f"Failed to remove checkpoint: {e}")
    
    # ==================== LOADING ====================
    
    def load_latest_checkpoint(self) -> Optional[Dict[str, Any]]:
        """Load the most recent checkpoint (hot or cold)."""
        with self._lock:
            # Find latest
            all_checkpoints = self.hot_snapshots + self.cold_checkpoints
            if not all_checkpoints:
                return None
            
            latest = max(all_checkpoints, key=lambda c: c.timestamp)
            return self.load_checkpoint(latest.file_path)
    
    def load_checkpoint(self, filepath: str) -> Optional[Dict[str, Any]]:
        """Load checkpoint from file."""
        try:
            filepath = Path(filepath)
            
            if filepath.suffix == '.gz' or str(filepath).endswith('.pt.gz'):
                # Compressed
                with gzip.open(filepath, 'rb') as f:
                    buffer = io.BytesIO(f.read())
                    return torch.load(buffer, map_location='cpu')
            else:
                # Uncompressed
                return torch.load(filepath, map_location='cpu')
                
        except Exception as e:
            logger.error(f"Failed to load checkpoint {filepath}: {e}")
            return None
    
    def restore_from_checkpoint(self, checkpoint: Dict[str, Any]) -> bool:
        """
        Restore model/optimizer/trainer from checkpoint.
        
        Args:
            checkpoint: Checkpoint dictionary
            
        Returns:
            True if successful
        """
        with self._lock:
            try:
                # Restore model
                if 'model_state_dict' in checkpoint:
                    self.model.load_state_dict(checkpoint['model_state_dict'])
                
                # Restore optimizer
                if 'optimizer_state_dict' in checkpoint:
                    self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                
                # Restore training state
                self.training_step = checkpoint.get('training_step', 0)
                self.outer_step = checkpoint.get('outer_step', 0)
                self.inner_step = checkpoint.get('inner_step', 0)
                
                # Restore DiLoCo state
                if self.diloco is not None and 'diloco_state' in checkpoint:
                    self.diloco.load_state_dict(checkpoint['diloco_state'])
                
                self.recoveries_performed += 1
                
                logger.info(f"Restored from checkpoint: "
                           f"step={self.training_step}, "
                           f"outer={self.outer_step}")
                
                return True
                
            except Exception as e:
                logger.error(f"Failed to restore from checkpoint: {e}")
                return False
    
    # ==================== PEER RECOVERY ====================
    
    async def fetch_neighbor_snapshot(
        self,
        peer_id: str,
    ) -> Optional[Dict[str, Any]]:
        """
        Fetch hot snapshot from a neighbor peer.
        
        Used for fast recovery after crash.
        
        Args:
            peer_id: ID of peer to fetch from
            
        Returns:
            Checkpoint dict if successful, None otherwise
        """
        if self.p2p is None:
            logger.warning("No P2P manager - cannot fetch from peer")
            return None
        
        try:
            # Look up peer's checkpoint in DHT
            key = f"checkpoint_{peer_id}"
            
            if hasattr(self.p2p, 'dht') and self.p2p.dht:
                checkpoint_info = self.p2p.dht.lookup_value(key)
                
                if checkpoint_info:
                    # Fetch via gRPC
                    # This would use a GetHotSnapshot RPC
                    logger.info(f"Found checkpoint from peer {peer_id}")
                    # Implementation would go here
            
            return None
            
        except Exception as e:
            logger.error(f"Failed to fetch from peer {peer_id}: {e}")
            return None
    
    async def try_auto_recovery(self) -> bool:
        """
        Attempt automatic recovery from peers.
        
        Tries to find and load a recent checkpoint from any available peer.
        
        Returns:
            True if recovery succeeded
        """
        if not self.config.auto_fetch_on_start:
            return False
        
        if self.p2p is None:
            return False
        
        # Get list of known peers
        peers = []
        if hasattr(self.p2p, 'get_peers'):
            peers = self.p2p.get_peers()
        
        # Try each peer
        for peer_id in peers:
            checkpoint = await self.fetch_neighbor_snapshot(peer_id)
            if checkpoint:
                return self.restore_from_checkpoint(checkpoint)
        
        return False
    
    def _announce_checkpoint(self, metadata: CheckpointMetadata):
        """Announce checkpoint availability to DHT."""
        if self.p2p is None:
            return
        
        try:
            if hasattr(self.p2p, 'dht') and self.p2p.dht:
                key = f"checkpoint_{self.node_id}"
                value = {
                    'checkpoint_id': metadata.checkpoint_id,
                    'timestamp': metadata.timestamp,
                    'training_step': metadata.training_step,
                    'model_hash': metadata.model_hash,
                }
                self.p2p.dht.store(key, str(value))
                logger.debug(f"Announced checkpoint to DHT: {metadata.checkpoint_id}")
                
        except Exception as e:
            logger.warning(f"Failed to announce checkpoint: {e}")
    
    # ==================== SERVING ====================
    
    def get_latest_snapshot_for_serving(self) -> Optional[bytes]:
        """
        Get latest snapshot data for serving to peers.
        
        Returns compressed checkpoint bytes.
        """
        if not self.config.serve_to_peers:
            return None
        
        with self._lock:
            if not self.hot_snapshots:
                return None
            
            latest = self.hot_snapshots[-1]
            
            try:
                with open(latest.file_path, 'rb') as f:
                    data = f.read()
                
                self.snapshots_served += 1
                return data
                
            except Exception as e:
                logger.error(f"Failed to read snapshot for serving: {e}")
                return None
    
    # ==================== UTILITIES ====================
    
    def _compute_model_hash(self) -> str:
        """Compute hash of model parameters."""
        hasher = hashlib.sha256()
        
        for name, param in sorted(self.model.named_parameters()):
            hasher.update(name.encode())
            hasher.update(param.data.cpu().numpy().tobytes()[:1000])  # First 1000 bytes
        
        return hasher.hexdigest()[:16]
    
    def _get_layer_ids(self) -> List[int]:
        """Get layer IDs from model if available."""
        if hasattr(self.model, 'my_layer_ids'):
            return list(self.model.my_layer_ids)
        return []
    
    def update_training_state(
        self,
        training_step: int,
        outer_step: int = 0,
        inner_step: int = 0,
    ):
        """Update training state for checkpoints."""
        with self._lock:
            self.training_step = training_step
            self.outer_step = outer_step
            self.inner_step = inner_step
    
    def get_stats(self) -> Dict[str, Any]:
        """Get checkpointer statistics."""
        with self._lock:
            return {
                'running': self.running,
                'snapshots_saved': self.snapshots_saved,
                'snapshots_served': self.snapshots_served,
                'recoveries_performed': self.recoveries_performed,
                'hot_snapshot_count': len(self.hot_snapshots),
                'cold_checkpoint_count': len(self.cold_checkpoints),
                'latest_snapshot_age': (
                    self.hot_snapshots[-1].age_seconds 
                    if self.hot_snapshots else None
                ),
                'training_step': self.training_step,
            }
    
    def force_snapshot(self) -> Optional[CheckpointMetadata]:
        """Force an immediate hot snapshot."""
        return self._save_hot_snapshot()


# ==================== FACTORY FUNCTIONS ====================

def create_checkpointer(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    checkpoint_dir: str = "./checkpoints",
    snapshot_interval: float = 120.0,
    node_id: str = "",
    **config_kwargs,
) -> SpeculativeCheckpointer:
    """
    Factory function to create a speculative checkpointer.
    
    Args:
        model: Model to checkpoint
        optimizer: Optimizer to checkpoint
        checkpoint_dir: Directory for checkpoints
        snapshot_interval: Seconds between hot snapshots
        node_id: This node's ID
        **config_kwargs: Additional config options
        
    Returns:
        Configured SpeculativeCheckpointer
    """
    config = CheckpointConfig(
        checkpoint_dir=checkpoint_dir,
        snapshot_interval=snapshot_interval,
        **config_kwargs,
    )
    
    return SpeculativeCheckpointer(
        model=model,
        optimizer=optimizer,
        config=config,
        node_id=node_id,
    )

