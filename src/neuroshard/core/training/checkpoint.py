"""
Checkpoint Sharding System for NeuroShard

This module enables distributing model checkpoints across multiple nodes,
allowing the network to scale beyond what any single node can hold.

Architecture:
- Full model is split into N shards (typically by layer groups)
- Each shard is replicated across M nodes (default M=3 for redundancy)
- Shards are announced via DHT for discovery
- Training updates are coordinated across shard holders

Sharding Strategy:
- Phase 1-2 (Bootstrap/Early): Full model per node (no sharding needed)
- Phase 3 (Growth): Optional sharding for 7B model
- Phase 4 (Mature): Mandatory sharding for 70B+ model

Example for 70B model (80 layers):
- Shard 0: Embedding + Layers 0-19   (~70GB / 4 = ~17.5GB)
- Shard 1: Layers 20-39              (~17.5GB)
- Shard 2: Layers 40-59              (~17.5GB)
- Shard 3: Layers 60-79 + LM Head    (~17.5GB)
"""

import hashlib
import json
import time
import threading
import logging
from typing import Dict, List, Optional, Tuple, Set, Any
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path

import torch

logger = logging.getLogger(__name__)


class ShardingStrategy(Enum):
    """How to split the model."""
    NONE = "none"              # Full model per node (Phase 1-2)
    LAYER_GROUPS = "layer_groups"  # Split by layer ranges (Phase 3-4)
    TENSOR_PARALLEL = "tensor_parallel"  # Split within layers (future)


@dataclass
class ShardConfig:
    """Configuration for a single shard."""
    shard_id: int
    total_shards: int
    
    # Layer range this shard covers
    start_layer: int
    end_layer: int  # Exclusive
    
    # Special flags
    has_embedding: bool = False
    has_lm_head: bool = False
    
    # Size estimates
    estimated_size_mb: float = 0.0
    
    def __post_init__(self):
        """Validate shard config."""
        assert 0 <= self.shard_id < self.total_shards
        assert self.start_layer < self.end_layer


@dataclass
class ShardState:
    """State of a shard on this node."""
    config: ShardConfig
    
    # Version tracking
    version: int = 0
    model_hash: str = ""
    
    # Weights (loaded or None)
    weights: Optional[Dict[str, torch.Tensor]] = None
    
    # Status
    is_loaded: bool = False
    last_updated: float = field(default_factory=time.time)
    
    # Replication
    replicas: Set[str] = field(default_factory=set)  # Node URLs holding copies


@dataclass
class ShardAnnouncement:
    """Announcement of shard availability."""
    node_id: str
    node_url: str
    grpc_addr: str
    
    shard_id: int
    total_shards: int
    version: int
    model_hash: str
    
    # Capacity info
    available_memory_mb: float
    current_load: float  # 0-1
    
    timestamp: float = field(default_factory=time.time)
    
    def to_dict(self) -> Dict:
        return {
            "node_id": self.node_id,
            "node_url": self.node_url,
            "grpc_addr": self.grpc_addr,
            "shard_id": self.shard_id,
            "total_shards": self.total_shards,
            "version": self.version,
            "model_hash": self.model_hash,
            "available_memory_mb": self.available_memory_mb,
            "current_load": self.current_load,
            "timestamp": self.timestamp,
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'ShardAnnouncement':
        return cls(**data)


class ShardRegistry:
    """
    Registry for tracking which nodes hold which shards.
    
    Uses DHT for decentralized discovery:
    - Key: "shard_{model_id}_{shard_id}"
    - Value: JSON list of ShardAnnouncements
    """
    
    def __init__(self, dht_protocol=None, model_id: str = "neuro_llm"):
        self.dht = dht_protocol
        self.model_id = model_id
        
        # Local cache
        self.shard_holders: Dict[int, List[ShardAnnouncement]] = {}
        self.lock = threading.Lock()
        
        # Refresh interval
        self.cache_ttl = 60  # seconds
        self.last_refresh: Dict[int, float] = {}
    
    def announce_shard(self, announcement: ShardAnnouncement) -> bool:
        """Announce that we hold a shard."""
        if not self.dht:
            logger.warning("No DHT available for shard announcement")
            return False
        
        try:
            key = f"shard_{self.model_id}_{announcement.shard_id}"
            
            # Get existing announcements
            existing = self._get_shard_holders_from_dht(announcement.shard_id)
            
            # Add/update our announcement
            updated = False
            for i, ann in enumerate(existing):
                if ann.node_id == announcement.node_id:
                    existing[i] = announcement
                    updated = True
                    break
            
            if not updated:
                existing.append(announcement)
            
            # Store back to DHT
            value = json.dumps([a.to_dict() for a in existing])
            self.dht.store(key, value)
            
            # Update local cache
            with self.lock:
                self.shard_holders[announcement.shard_id] = existing
            
            logger.info(f"Announced shard {announcement.shard_id} "
                       f"(version={announcement.version}, holders={len(existing)})")
            return True
            
        except Exception as e:
            logger.error(f"Failed to announce shard: {e}")
            return False
    
    def get_shard_holders(self, shard_id: int, refresh: bool = False) -> List[ShardAnnouncement]:
        """Get nodes holding a specific shard."""
        with self.lock:
            # Check cache
            if not refresh and shard_id in self.shard_holders:
                if time.time() - self.last_refresh.get(shard_id, 0) < self.cache_ttl:
                    return self.shard_holders[shard_id]
        
        # Refresh from DHT
        holders = self._get_shard_holders_from_dht(shard_id)
        
        with self.lock:
            self.shard_holders[shard_id] = holders
            self.last_refresh[shard_id] = time.time()
        
        return holders
    
    def _get_shard_holders_from_dht(self, shard_id: int) -> List[ShardAnnouncement]:
        """Query DHT for shard holders."""
        if not self.dht:
            return []
        
        try:
            key = f"shard_{self.model_id}_{shard_id}"
            value = self.dht.lookup_value(key)
            
            if not value:
                return []
            
            data = json.loads(value)
            return [ShardAnnouncement.from_dict(d) for d in data]
            
        except Exception as e:
            logger.debug(f"DHT lookup failed for shard {shard_id}: {e}")
            return []
    
    def find_best_holder(self, shard_id: int) -> Optional[ShardAnnouncement]:
        """Find the best node to request a shard from."""
        holders = self.get_shard_holders(shard_id)
        
        if not holders:
            return None
        
        # Sort by: highest version, lowest load, most recent
        holders.sort(key=lambda h: (-h.version, h.current_load, -h.timestamp))
        
        return holders[0]
    
    def get_all_shards_status(self, total_shards: int) -> Dict[int, Dict]:
        """Get status of all shards."""
        status = {}
        
        for shard_id in range(total_shards):
            holders = self.get_shard_holders(shard_id)
            
            status[shard_id] = {
                "shard_id": shard_id,
                "holder_count": len(holders),
                "is_available": len(holders) > 0,
                "is_redundant": len(holders) >= 3,
                "max_version": max((h.version for h in holders), default=0),
                "holders": [h.node_url for h in holders],
            }
        
        return status


class CheckpointShardManager:
    """
    Manages sharded checkpoints for a NeuroLLM model.
    
    Responsibilities:
    1. Determine sharding strategy based on model size and node capacity
    2. Load/save individual shards
    3. Coordinate with ShardRegistry for discovery
    4. Handle shard replication
    """
    
    # Minimum replicas per shard
    MIN_REPLICAS = 3
    
    def __init__(
        self,
        model,  # NeuroLLMForCausalLM
        node_id: str,
        node_url: str,
        registry: Optional[ShardRegistry] = None,
        checkpoint_dir: Optional[Path] = None
    ):
        self.model = model
        self.node_id = node_id
        self.node_url = node_url
        self.registry = registry
        self.checkpoint_dir = checkpoint_dir or Path.home() / ".neuroshard" / "shards"
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # Current shard state
        self.my_shards: Dict[int, ShardState] = {}
        self.shard_configs: List[ShardConfig] = []
        
        # Threading
        self.lock = threading.Lock()
    
    def compute_sharding_strategy(
        self,
        total_params: int,
        available_memory_mb: float,
        num_layers: int
    ) -> Tuple[ShardingStrategy, List[ShardConfig]]:
        """
        Determine optimal sharding strategy.
        
        Args:
            total_params: Total model parameters
            available_memory_mb: Available memory on this node
            num_layers: Number of transformer layers
            
        Returns:
            (strategy, list of shard configs)
        """
        # Estimate model size (4 bytes per param for float32)
        model_size_mb = (total_params * 4) / (1024 * 1024)
        
        # Add overhead for optimizer state, activations (2x)
        required_memory_mb = model_size_mb * 2
        
        logger.info(f"Model size: {model_size_mb:.1f}MB, "
                   f"Required: {required_memory_mb:.1f}MB, "
                   f"Available: {available_memory_mb:.1f}MB")
        
        # Strategy decision
        if required_memory_mb <= available_memory_mb:
            # Can hold full model
            return ShardingStrategy.NONE, [
                ShardConfig(
                    shard_id=0,
                    total_shards=1,
                    start_layer=0,
                    end_layer=num_layers,
                    has_embedding=True,
                    has_lm_head=True,
                    estimated_size_mb=model_size_mb
                )
            ]
        
        # Need to shard - calculate number of shards needed
        shard_memory_target = available_memory_mb * 0.8  # Leave 20% headroom
        num_shards = max(2, int(required_memory_mb / shard_memory_target) + 1)
        
        # Round up to power of 2 for cleaner division
        num_shards = 2 ** (num_shards - 1).bit_length()
        num_shards = min(num_shards, num_layers)  # Can't have more shards than layers
        
        # Create shard configs
        layers_per_shard = num_layers // num_shards
        shard_size_mb = model_size_mb / num_shards
        
        configs = []
        for i in range(num_shards):
            start = i * layers_per_shard
            end = (i + 1) * layers_per_shard if i < num_shards - 1 else num_layers
            
            configs.append(ShardConfig(
                shard_id=i,
                total_shards=num_shards,
                start_layer=start,
                end_layer=end,
                has_embedding=(i == 0),
                has_lm_head=(i == num_shards - 1),
                estimated_size_mb=shard_size_mb
            ))
        
        logger.info(f"Sharding strategy: {num_shards} shards, "
                   f"{layers_per_shard} layers each, "
                   f"~{shard_size_mb:.1f}MB per shard")
        
        return ShardingStrategy.LAYER_GROUPS, configs
    
    def extract_shard_weights(self, shard_config: ShardConfig) -> Dict[str, torch.Tensor]:
        """Extract weights for a specific shard from the full model."""
        state_dict = self.model.state_dict()
        shard_weights = {}
        
        for name, param in state_dict.items():
            # Check if this parameter belongs to this shard
            if self._param_belongs_to_shard(name, shard_config):
                shard_weights[name] = param.clone()
        
        return shard_weights
    
    def _param_belongs_to_shard(self, param_name: str, config: ShardConfig) -> bool:
        """Check if a parameter belongs to a shard."""
        # Embedding layer
        if "embed" in param_name.lower() or "wte" in param_name.lower():
            return config.has_embedding
        
        # LM head
        if "lm_head" in param_name.lower() or "output" in param_name.lower():
            return config.has_lm_head
        
        # Transformer layers - extract layer number
        import re
        match = re.search(r'layers?[._](\d+)', param_name.lower())
        if match:
            layer_num = int(match.group(1))
            return config.start_layer <= layer_num < config.end_layer
        
        # Final norm (belongs to last shard)
        if "final" in param_name.lower() or "ln_f" in param_name.lower():
            return config.has_lm_head
        
        # Default: include in first shard
        return config.shard_id == 0
    
    def save_shard(self, shard_id: int, version: int) -> Optional[Path]:
        """Save a shard to disk."""
        if shard_id not in self.my_shards:
            logger.error(f"Shard {shard_id} not loaded")
            return None
        
        shard_state = self.my_shards[shard_id]
        
        if not shard_state.weights:
            # Extract from model
            shard_state.weights = self.extract_shard_weights(shard_state.config)
        
        # Compute hash
        model_hash = self._compute_weights_hash(shard_state.weights)
        
        # Save
        filename = f"shard_{shard_id}_v{version}.pt"
        path = self.checkpoint_dir / filename
        
        torch.save({
            "shard_id": shard_id,
            "config": {
                "shard_id": shard_state.config.shard_id,
                "total_shards": shard_state.config.total_shards,
                "start_layer": shard_state.config.start_layer,
                "end_layer": shard_state.config.end_layer,
                "has_embedding": shard_state.config.has_embedding,
                "has_lm_head": shard_state.config.has_lm_head,
            },
            "weights": shard_state.weights,
            "version": version,
            "model_hash": model_hash,
            "timestamp": time.time(),
        }, path)
        
        # Update state
        shard_state.version = version
        shard_state.model_hash = model_hash
        shard_state.last_updated = time.time()
        
        logger.info(f"Saved shard {shard_id} v{version} to {path}")
        
        return path
    
    def load_shard(self, shard_id: int, path: Optional[Path] = None) -> bool:
        """Load a shard from disk."""
        if path is None:
            # Find latest version
            pattern = f"shard_{shard_id}_v*.pt"
            files = list(self.checkpoint_dir.glob(pattern))
            if not files:
                logger.warning(f"No shard {shard_id} found on disk")
                return False
            path = max(files, key=lambda p: p.stat().st_mtime)
        
        try:
            data = torch.load(path, map_location="cpu", weights_only=False)
            
            config = ShardConfig(**data["config"])
            
            self.my_shards[shard_id] = ShardState(
                config=config,
                version=data["version"],
                model_hash=data["model_hash"],
                weights=data["weights"],
                is_loaded=True,
                last_updated=data.get("timestamp", time.time()),
            )
            
            logger.info(f"Loaded shard {shard_id} v{data['version']} from {path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load shard {shard_id}: {e}")
            return False
    
    def load_shard_into_model(self, shard_id: int) -> bool:
        """Load shard weights into the model."""
        if shard_id not in self.my_shards:
            logger.error(f"Shard {shard_id} not loaded")
            return False
        
        shard_state = self.my_shards[shard_id]
        
        if not shard_state.weights:
            logger.error(f"Shard {shard_id} has no weights")
            return False
        
        # Load into model
        current_state = self.model.state_dict()
        
        for name, param in shard_state.weights.items():
            if name in current_state:
                current_state[name].copy_(param)
            else:
                logger.warning(f"Parameter {name} not found in model")
        
        logger.info(f"Loaded shard {shard_id} weights into model")
        return True
    
    def _compute_weights_hash(self, weights: Dict[str, torch.Tensor]) -> str:
        """Compute hash of weights."""
        hasher = hashlib.sha256()
        
        for name in sorted(weights.keys()):
            hasher.update(name.encode())
            hasher.update(weights[name].cpu().numpy().tobytes()[:1000])
        
        return hasher.hexdigest()[:16]
    
    def get_shard_status(self) -> Dict[str, Any]:
        """Get status of all shards on this node."""
        return {
            "node_id": self.node_id,
            "shards": {
                shard_id: {
                    "shard_id": shard_id,
                    "version": state.version,
                    "model_hash": state.model_hash,
                    "is_loaded": state.is_loaded,
                    "layers": f"{state.config.start_layer}-{state.config.end_layer}",
                    "has_embedding": state.config.has_embedding,
                    "has_lm_head": state.config.has_lm_head,
                    "replicas": len(state.replicas),
                }
                for shard_id, state in self.my_shards.items()
            },
            "total_shards": len(self.shard_configs),
        }


def compute_shard_assignment(
    node_capacities: Dict[str, float],  # node_id -> available_memory_mb
    model_size_mb: float,
    num_layers: int,
    min_replicas: int = 3
) -> Dict[str, List[int]]:
    """
    Compute optimal shard assignment across nodes.
    
    This is a simplified greedy algorithm. A production system would use
    more sophisticated optimization (e.g., constraint satisfaction, ILP).
    
    Args:
        node_capacities: Available memory per node
        model_size_mb: Total model size
        num_layers: Number of transformer layers
        min_replicas: Minimum replicas per shard
        
    Returns:
        Dict mapping node_id -> list of shard_ids to hold
    """
    # Sort nodes by capacity (largest first)
    sorted_nodes = sorted(node_capacities.items(), key=lambda x: -x[1])
    
    if not sorted_nodes:
        return {}
    
    # Determine number of shards based on largest node
    max_capacity = sorted_nodes[0][1]
    shard_size = max_capacity * 0.8  # Leave headroom
    num_shards = max(1, int(model_size_mb / shard_size) + 1)
    num_shards = min(num_shards, num_layers)
    
    actual_shard_size = model_size_mb / num_shards
    
    # Assign shards to nodes (round-robin with capacity check)
    assignments: Dict[str, List[int]] = {node_id: [] for node_id, _ in sorted_nodes}
    shard_counts = {shard_id: 0 for shard_id in range(num_shards)}
    
    # First pass: assign one shard per node (ensure coverage)
    for shard_id in range(num_shards):
        for node_id, capacity in sorted_nodes:
            current_load = len(assignments[node_id]) * actual_shard_size
            if current_load + actual_shard_size <= capacity:
                assignments[node_id].append(shard_id)
                shard_counts[shard_id] += 1
                break
    
    # Second pass: add replicas until min_replicas reached
    for shard_id in range(num_shards):
        while shard_counts[shard_id] < min_replicas:
            # Find node with capacity that doesn't already have this shard
            assigned = False
            for node_id, capacity in sorted_nodes:
                if shard_id in assignments[node_id]:
                    continue
                current_load = len(assignments[node_id]) * actual_shard_size
                if current_load + actual_shard_size <= capacity:
                    assignments[node_id].append(shard_id)
                    shard_counts[shard_id] += 1
                    assigned = True
                    break
            
            if not assigned:
                logger.warning(f"Could not assign {min_replicas} replicas for shard {shard_id}")
                break
    
    return assignments

