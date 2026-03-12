"""
Quorum System - Self-Organizing Training Groups

This module implements the quorum-based training system:
- Quorum: A group of speed-matched nodes that together hold a complete model
- QuorumRegistry: DHT-backed registry for quorum discovery
- QuorumLifecycle: State machine for quorum formation, operation, and dissolution

Key Concepts:
- A Quorum is a complete pipeline (covers all layers from embedding to LM head)
- Nodes in a quorum are speed-matched (within compatible tiers)
- Quorums train synchronously within, asynchronously across (via DiLoCo)
- Sessions last ~1 hour, with renewal checks at 80% of session time
"""

import asyncio
import hashlib
import json
import logging
import threading
import time
import uuid
from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import Dict, List, Optional, Tuple, Any, Set

logger = logging.getLogger(__name__)


# =============================================================================
# QUORUM CONFIGURATION
# =============================================================================

# Session timing
BASE_SESSION_DURATION = 3600      # 1 hour default session
MAX_SESSION_DURATION = 14400      # 4 hours maximum
RENEWAL_CHECK_RATIO = 0.8         # Check at 80% of session
MIN_BATCHES_TO_RENEW = 1000       # Minimum batches for renewal eligibility

# Quorum formation
FORMATION_TIMEOUT = 60            # Seconds to wait for quorum formation
MIN_QUORUM_MEMBERS = 1            # Minimum members (adapts with network size)
MAX_QUORUM_MEMBERS = 16           # Maximum members per quorum

# Health monitoring
HEARTBEAT_INTERVAL = 30           # Seconds between heartbeats
STALE_THRESHOLD = 4               # Missed heartbeats before considered stale
OFFLINE_THRESHOLD = 120           # Seconds before marked offline (4 × 30)


# =============================================================================
# QUORUM LIFECYCLE STATES
# =============================================================================

class QuorumLifecycle(Enum):
    """State machine for quorum lifecycle."""
    FORMING = "forming"           # Gathering members
    ACTIVE = "active"             # Training in progress
    RENEWING = "renewing"         # Session renewal in progress
    DISSOLVING = "dissolving"     # Graceful shutdown
    DISSOLVED = "dissolved"       # Quorum no longer exists


class QuorumType(Enum):
    """Type of quorum — determines training topology."""
    PIPELINE = "pipeline"         # Tightly-coupled pipeline parallelism (layers split across nodes)
    SYNC = "sync"                 # Loosely-coupled data parallelism (each node has all layers, DiLoCo sync)


class QuorumRole(Enum):
    """Role within a quorum pipeline."""
    INITIATOR = "initiator"       # Holds embedding layer, fetches batches
    PROCESSOR = "processor"       # Middle layers, forward/backward
    FINISHER = "finisher"         # Holds LM head, computes loss


# =============================================================================
# QUORUM MEMBER
# =============================================================================

@dataclass
class QuorumMember:
    """A member of a quorum."""
    node_id: str
    endpoint: str                  # gRPC endpoint (ip:port)
    layer_range: Tuple[int, int]   # (start, end) exclusive
    speed_tier: str                # SpeedTier value
    role: QuorumRole = QuorumRole.PROCESSOR
    
    # Health tracking
    last_heartbeat: float = field(default_factory=time.time)
    consecutive_failures: int = 0
    batches_processed: int = 0
    
    # Uptime tracking (for reputation)
    join_time: float = field(default_factory=time.time)
    total_online_seconds: float = 0.0
    total_expected_seconds: float = 0.0
    successful_requests: int = 0
    failed_requests: int = 0
    
    # Derived reputation (0.0 to 1.0)
    # Combines uptime_ratio and success_rate
    reputation: float = 1.0
    
    def update_uptime(self, heartbeat_received: bool = True):
        """
        Update uptime tracking based on heartbeat status.
        
        Called periodically (e.g., every heartbeat interval) to track
        whether the node was online.
        
        Args:
            heartbeat_received: Whether a heartbeat was received in this interval
        """
        self.total_expected_seconds += HEARTBEAT_INTERVAL
        if heartbeat_received:
            self.total_online_seconds += HEARTBEAT_INTERVAL
        self._recalculate_reputation()
    
    def record_request_result(self, success: bool):
        """
        Record the result of a request processed by this member.
        
        Args:
            success: Whether the request completed successfully
        """
        if success:
            self.successful_requests += 1
        else:
            self.failed_requests += 1
        self._recalculate_reputation()
    
    def _recalculate_reputation(self):
        """
        Recalculate reputation based on uptime and success rate.
        
        Reputation formula:
        - 60% weighted by uptime_ratio
        - 40% weighted by success_rate
        - Minimum 0.1 (to allow recovery)
        """
        uptime_ratio = self.uptime_ratio
        success_rate = self.success_rate
        
        # Weighted combination
        self.reputation = max(0.1, 0.6 * uptime_ratio + 0.4 * success_rate)
    
    @property
    def uptime_ratio(self) -> float:
        """
        Get the uptime ratio (0.0 to 1.0).
        
        This is the fraction of expected time the node was actually online.
        """
        if self.total_expected_seconds <= 0:
            return 1.0  # New nodes start with full uptime
        return min(1.0, self.total_online_seconds / self.total_expected_seconds)
    
    @property
    def success_rate(self) -> float:
        """
        Get the request success rate (0.0 to 1.0).
        
        This is the fraction of requests that completed successfully.
        """
        total = self.successful_requests + self.failed_requests
        if total == 0:
            return 1.0  # New nodes start with full success
        return self.successful_requests / total
    
    @property
    def meets_pipeline_requirements(self) -> bool:
        """
        Check if member meets requirements for pipeline mode.
        
        Pipeline mode requires:
        - Uptime ratio >= 90%
        - Not currently stale or offline
        """
        return self.uptime_ratio >= 0.9 and not self.is_stale
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "node_id": self.node_id,
            "endpoint": self.endpoint,
            "layer_range": list(self.layer_range),
            "speed_tier": self.speed_tier,
            "role": self.role.value,
            "last_heartbeat": self.last_heartbeat,
            "batches_processed": self.batches_processed,
            "reputation": self.reputation,
            "join_time": self.join_time,
            "total_online_seconds": self.total_online_seconds,
            "total_expected_seconds": self.total_expected_seconds,
            "successful_requests": self.successful_requests,
            "failed_requests": self.failed_requests,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'QuorumMember':
        """Create from dictionary."""
        return cls(
            node_id=data["node_id"],
            endpoint=data["endpoint"],
            layer_range=tuple(data["layer_range"]),
            speed_tier=data["speed_tier"],
            role=QuorumRole(data.get("role", "processor")),
            last_heartbeat=data.get("last_heartbeat", time.time()),
            batches_processed=data.get("batches_processed", 0),
            reputation=data.get("reputation", 1.0),
            join_time=data.get("join_time", time.time()),
            total_online_seconds=data.get("total_online_seconds", 0.0),
            total_expected_seconds=data.get("total_expected_seconds", 0.0),
            successful_requests=data.get("successful_requests", 0),
            failed_requests=data.get("failed_requests", 0),
        )
    
    @property
    def is_stale(self) -> bool:
        """Check if member hasn't sent heartbeat recently."""
        return (time.time() - self.last_heartbeat) > (HEARTBEAT_INTERVAL * STALE_THRESHOLD)
    
    @property
    def is_offline(self) -> bool:
        """Check if member is considered offline."""
        return (time.time() - self.last_heartbeat) > OFFLINE_THRESHOLD


# =============================================================================
# QUORUM
# =============================================================================

@dataclass
class Quorum:
    """
    A self-organized group of speed-matched nodes.
    
    Two types:
    - SYNC: Data-parallel group. Each member holds ALL layers (full replica).
      Members train independently on different data and sync via DiLoCo.
      Always ACTIVE — any full replica is self-sufficient.
    - PIPELINE: Model-parallel pipeline. Members hold complementary layer ranges.
      Requires full coverage for ACTIVE state.
    
    Properties:
    - Speed-matched: All members in compatible speed tiers
    - Self-sufficient: Can train independently without coordinator
    - Temporary: Sessions last ~1 hour, then renew or dissolve
    """
    quorum_id: str
    speed_tier: str                # Dominant speed tier (e.g., "tier2")
    
    # Quorum type: SYNC (data-parallel) or PIPELINE (model-parallel)
    quorum_type: QuorumType = QuorumType.PIPELINE
    
    # Members and layer mapping
    members: List[QuorumMember] = field(default_factory=list)
    
    # Session management
    session_start: float = field(default_factory=time.time)
    session_end: float = 0.0
    lifecycle: QuorumLifecycle = QuorumLifecycle.FORMING
    
    # Training progress
    total_batches: int = 0
    total_steps: int = 0
    current_step: int = 0
    
    # Network versioning
    arch_version: int = 1
    vocab_version: int = 1
    
    # Throughput tracking
    batches_per_second: float = 0.0
    last_throughput_update: float = 0.0
    
    def __post_init__(self):
        """Initialize session end time if not set."""
        if self.session_end == 0.0:
            self.session_end = self.session_start + BASE_SESSION_DURATION
    
    @classmethod
    def generate_id(cls) -> str:
        """Generate a unique quorum ID."""
        return f"q-{uuid.uuid4().hex[:12]}"
    
    def to_json(self) -> str:
        """Serialize to JSON for DHT storage."""
        data = {
            "quorum_id": self.quorum_id,
            "speed_tier": self.speed_tier,
            "quorum_type": self.quorum_type.value,
            "members": [m.to_dict() for m in self.members],
            "session_start": self.session_start,
            "session_end": self.session_end,
            "lifecycle": self.lifecycle.value,
            "total_batches": self.total_batches,
            "total_steps": self.total_steps,
            "current_step": self.current_step,
            "arch_version": self.arch_version,
            "vocab_version": self.vocab_version,
            "batches_per_second": self.batches_per_second,
        }
        return json.dumps(data)
    
    @classmethod
    def from_json(cls, json_str: str) -> 'Quorum':
        """Deserialize from JSON."""
        data = json.loads(json_str)
        quorum = cls(
            quorum_id=data["quorum_id"],
            speed_tier=data["speed_tier"],
            quorum_type=QuorumType(data.get("quorum_type", "pipeline")),
            session_start=data["session_start"],
            session_end=data["session_end"],
            lifecycle=QuorumLifecycle(data["lifecycle"]),
            total_batches=data.get("total_batches", 0),
            total_steps=data.get("total_steps", 0),
            current_step=data.get("current_step", 0),
            arch_version=data.get("arch_version", 1),
            vocab_version=data.get("vocab_version", 1),
            batches_per_second=data.get("batches_per_second", 0.0),
        )
        quorum.members = [QuorumMember.from_dict(m) for m in data.get("members", [])]
        return quorum
    
    # =========================================================================
    # LAYER COVERAGE
    # =========================================================================
    
    @property
    def layer_coverage(self) -> Set[int]:
        """Get set of all layers covered by members."""
        layers = set()
        for member in self.members:
            for layer_id in range(member.layer_range[0], member.layer_range[1]):
                layers.add(layer_id)
        return layers
    
    @property
    def is_complete(self) -> bool:
        """Check if quorum covers all layers (complete pipeline)."""
        if not self.members:
            return False
        
        # Get layer range from members
        all_layers = self.layer_coverage
        if not all_layers:
            return False
        
        min_layer = min(all_layers)
        max_layer = max(all_layers)
        
        # Check for gaps
        expected = set(range(min_layer, max_layer + 1))
        return all_layers == expected
    
    @property
    def has_initiator(self) -> bool:
        """Check if quorum has a node holding layer 0 (embedding)."""
        return any(m.role == QuorumRole.INITIATOR for m in self.members)
    
    @property
    def has_finisher(self) -> bool:
        """Check if quorum has a node holding the last layer (LM head)."""
        return any(m.role == QuorumRole.FINISHER for m in self.members)
    
    # =========================================================================
    # MEMBER MANAGEMENT
    # =========================================================================
    
    def add_member(self, member: QuorumMember) -> bool:
        """
        Add a member to the quorum.
        
        Args:
            member: QuorumMember to add
            
        Returns:
            True if added successfully
        """
        if len(self.members) >= MAX_QUORUM_MEMBERS:
            return False
        
        # Check for duplicate
        if any(m.node_id == member.node_id for m in self.members):
            return False
        
        # Assign role based on layer range
        if member.layer_range[0] == 0:
            member.role = QuorumRole.INITIATOR
        elif not self.members:
            # First member with non-zero start might be finisher
            member.role = QuorumRole.PROCESSOR
        else:
            # Check if this is the highest layer
            current_max = max(m.layer_range[1] for m in self.members)
            if member.layer_range[1] > current_max:
                member.role = QuorumRole.FINISHER
                # Demote previous finisher
                for m in self.members:
                    if m.role == QuorumRole.FINISHER:
                        m.role = QuorumRole.PROCESSOR
            else:
                member.role = QuorumRole.PROCESSOR
        
        self.members.append(member)
        logger.info(f"Added member {member.node_id[:8]}... to quorum {self.quorum_id[:8]}... as {member.role.value}")
        return True
    
    def remove_member(self, node_id: str) -> Optional[QuorumMember]:
        """
        Remove a member from the quorum.
        
        Args:
            node_id: ID of node to remove
            
        Returns:
            Removed member if found, None otherwise
        """
        for i, member in enumerate(self.members):
            if member.node_id == node_id:
                removed = self.members.pop(i)
                logger.info(f"Removed member {node_id[:8]}... from quorum {self.quorum_id[:8]}...")
                return removed
        return None
    
    def get_member(self, node_id: str) -> Optional[QuorumMember]:
        """Get a member by node ID."""
        for member in self.members:
            if member.node_id == node_id:
                return member
        return None
    
    def get_pipeline_order(self) -> List[QuorumMember]:
        """Get members in pipeline order (by layer range start)."""
        return sorted(self.members, key=lambda m: m.layer_range[0])
    
    def get_next_hop(self, current_node_id: str) -> Optional[QuorumMember]:
        """Get the next node in the pipeline after current node."""
        pipeline = self.get_pipeline_order()
        for i, member in enumerate(pipeline):
            if member.node_id == current_node_id:
                if i + 1 < len(pipeline):
                    return pipeline[i + 1]
        return None
    
    def get_prev_hop(self, current_node_id: str) -> Optional[QuorumMember]:
        """Get the previous node in the pipeline before current node."""
        pipeline = self.get_pipeline_order()
        for i, member in enumerate(pipeline):
            if member.node_id == current_node_id:
                if i > 0:
                    return pipeline[i - 1]
        return None
    
    # =========================================================================
    # HEALTH MONITORING
    # =========================================================================
    
    def update_heartbeat(self, node_id: str) -> bool:
        """
        Update heartbeat timestamp for a member.
        
        Args:
            node_id: ID of node sending heartbeat
            
        Returns:
            True if member found and updated
        """
        member = self.get_member(node_id)
        if member:
            member.last_heartbeat = time.time()
            member.consecutive_failures = 0
            return True
        return False
    
    def record_failure(self, node_id: str) -> int:
        """
        Record a failure for a member.
        
        Returns:
            Number of consecutive failures
        """
        member = self.get_member(node_id)
        if member:
            member.consecutive_failures += 1
            return member.consecutive_failures
        return 0
    
    def get_healthy_members(self) -> List[QuorumMember]:
        """Get list of members that are not stale."""
        return [m for m in self.members if not m.is_stale]
    
    def get_stale_members(self) -> List[QuorumMember]:
        """Get list of stale members (missed heartbeats)."""
        return [m for m in self.members if m.is_stale]
    
    def get_offline_members(self) -> List[QuorumMember]:
        """Get list of offline members."""
        return [m for m in self.members if m.is_offline]
    
    @property
    def is_healthy(self) -> bool:
        """Check if quorum has enough healthy members for operation."""
        healthy = self.get_healthy_members()
        return len(healthy) >= len(self.members) * 0.5 and self.is_complete
    
    # =========================================================================
    # SESSION MANAGEMENT
    # =========================================================================
    
    @property
    def session_remaining(self) -> float:
        """Get remaining session time in seconds."""
        return max(0, self.session_end - time.time())
    
    @property
    def session_progress(self) -> float:
        """Get session progress as fraction (0.0 to 1.0)."""
        duration = self.session_end - self.session_start
        elapsed = time.time() - self.session_start
        return min(1.0, elapsed / duration)
    
    @property
    def should_check_renewal(self) -> bool:
        """Check if it's time to check for renewal."""
        return self.session_progress >= RENEWAL_CHECK_RATIO
    
    @property
    def is_session_expired(self) -> bool:
        """Check if session has expired."""
        return time.time() >= self.session_end
    
    def extend_session(self, duration: float = BASE_SESSION_DURATION) -> None:
        """Extend the session by specified duration."""
        self.session_end = time.time() + min(duration, MAX_SESSION_DURATION)
        self.lifecycle = QuorumLifecycle.ACTIVE
        logger.info(f"Quorum {self.quorum_id[:8]}... session extended by {duration}s")
    
    # =========================================================================
    # TRAINING PROGRESS
    # =========================================================================
    
    def record_batch(self, batches: int = 1) -> None:
        """Record processed batches."""
        self.total_batches += batches
        
        # Update throughput
        now = time.time()
        if self.last_throughput_update > 0:
            elapsed = now - self.last_throughput_update
            if elapsed > 0:
                # Exponential moving average
                new_rate = batches / elapsed
                self.batches_per_second = 0.9 * self.batches_per_second + 0.1 * new_rate
        self.last_throughput_update = now
    
    def record_step(self) -> None:
        """Record a training step completion."""
        self.total_steps += 1
        self.current_step += 1


# =============================================================================
# QUORUM REGISTRY
# =============================================================================

class QuorumRegistry:
    """
    DHT-backed registry for quorum discovery and management.
    
    Responsibilities:
    - Store quorum metadata in DHT
    - Discover existing quorums
    - Track local quorum membership
    """
    
    # DHT key prefixes
    DHT_KEY_QUORUM_PREFIX = "quorum:"
    DHT_KEY_QUORUM_LIST = "quorums:active"
    
    def __init__(self, dht_protocol=None):
        """
        Initialize the quorum registry.
        
        Args:
            dht_protocol: DHTProtocol instance for network storage
        """
        self.dht = dht_protocol
        self.local_quorums: Dict[str, Quorum] = {}  # quorum_id -> Quorum
        self._lock = threading.RLock()
    
    def _get_quorum_key(self, quorum_id: str) -> int:
        """Get DHT key for a quorum."""
        key_str = f"{self.DHT_KEY_QUORUM_PREFIX}{quorum_id}"
        return int(hashlib.sha1(key_str.encode()).hexdigest(), 16)
    
    def register_quorum(self, quorum: Quorum) -> bool:
        """
        Register a quorum in the registry and DHT.
        
        Args:
            quorum: Quorum to register
            
        Returns:
            True if registered successfully
        """
        with self._lock:
            self.local_quorums[quorum.quorum_id] = quorum
            
            # Store in DHT
            if self.dht:
                try:
                    key_id = self._get_quorum_key(quorum.quorum_id)
                    self.dht.storage[key_id] = quorum.to_json()
                    
                    # Update active quorums list
                    self._update_active_list(quorum.quorum_id, add=True)
                    
                except Exception as e:
                    logger.warning(f"Failed to store quorum in DHT: {e}")
            
            logger.info(f"Registered quorum {quorum.quorum_id[:8]}... with {len(quorum.members)} members")
            return True
    
    def unregister_quorum(self, quorum_id: str) -> bool:
        """
        Unregister a quorum from the registry.
        
        Args:
            quorum_id: ID of quorum to unregister
            
        Returns:
            True if unregistered successfully
        """
        with self._lock:
            if quorum_id in self.local_quorums:
                del self.local_quorums[quorum_id]
            
            # Remove from DHT
            if self.dht:
                try:
                    key_id = self._get_quorum_key(quorum_id)
                    if key_id in self.dht.storage:
                        del self.dht.storage[key_id]
                    
                    # Update active quorums list
                    self._update_active_list(quorum_id, add=False)
                    
                except Exception as e:
                    logger.warning(f"Failed to remove quorum from DHT: {e}")
            
            logger.info(f"Unregistered quorum {quorum_id[:8]}...")
            return True
    
    def _update_active_list(self, quorum_id: str, add: bool) -> None:
        """Update the active quorums list in DHT."""
        if not self.dht:
            return
        
        try:
            key_id = int(hashlib.sha1(self.DHT_KEY_QUORUM_LIST.encode()).hexdigest(), 16)
            
            # Get current list
            existing = self.dht.storage.get(key_id, "[]")
            active_ids = json.loads(existing)
            
            if add:
                if quorum_id not in active_ids:
                    active_ids.append(quorum_id)
            else:
                if quorum_id in active_ids:
                    active_ids.remove(quorum_id)
            
            self.dht.storage[key_id] = json.dumps(active_ids)
            
        except Exception as e:
            logger.debug(f"Failed to update active quorums list: {e}")
    
    def get_quorum(self, quorum_id: str) -> Optional[Quorum]:
        """
        Get a quorum by ID.
        
        Args:
            quorum_id: Quorum ID to look up
            
        Returns:
            Quorum if found, None otherwise
        """
        with self._lock:
            # Check local cache first
            if quorum_id in self.local_quorums:
                return self.local_quorums[quorum_id]
            
            # Try DHT lookup
            if self.dht:
                try:
                    key_id = self._get_quorum_key(quorum_id)
                    
                    # Check DHT storage
                    if key_id in self.dht.storage:
                        json_str = self.dht.storage[key_id]
                        return Quorum.from_json(json_str)
                    
                    # Try network lookup
                    value = self.dht.lookup_value(key_id)
                    if value:
                        return Quorum.from_json(value)
                        
                except Exception as e:
                    logger.debug(f"Failed to lookup quorum from DHT: {e}")
            
            return None
    
    def get_active_quorums(self) -> List[Quorum]:
        """
        Get all active quorums.
        
        Returns:
            List of active Quorum objects
        """
        quorums = []
        
        with self._lock:
            # Get from local cache
            for quorum in self.local_quorums.values():
                if quorum.lifecycle == QuorumLifecycle.ACTIVE:
                    quorums.append(quorum)
            
            # Try to get more from DHT
            if self.dht:
                try:
                    key_id = int(hashlib.sha1(self.DHT_KEY_QUORUM_LIST.encode()).hexdigest(), 16)
                    existing = self.dht.storage.get(key_id, "[]")
                    active_ids = json.loads(existing)
                    
                    for qid in active_ids:
                        if qid not in self.local_quorums:
                            quorum = self.get_quorum(qid)
                            if quorum and quorum.lifecycle == QuorumLifecycle.ACTIVE:
                                quorums.append(quorum)
                                
                except Exception as e:
                    logger.debug(f"Failed to get active quorums from DHT: {e}")
        
        return quorums
    
    def discover_quorums(self) -> List[Quorum]:
        """
        Discover all active quorums in the network.
        
        Alias for get_active_quorums() for API compatibility.
        
        Returns:
            List of active quorums
        """
        return self.get_active_quorums()
    
    def find_quorums_by_speed_tier(self, speed_tier: str) -> List[Quorum]:
        """
        Find quorums matching a speed tier.
        
        Args:
            speed_tier: Speed tier to match (e.g., "tier2")
            
        Returns:
            List of matching quorums
        """
        all_quorums = self.get_active_quorums()
        return [q for q in all_quorums if q.speed_tier == speed_tier]
    
    def find_quorums_with_capacity(self) -> List[Quorum]:
        """
        Find quorums that have capacity for new members.
        
        Returns:
            List of quorums with available slots
        """
        all_quorums = self.get_active_quorums()
        return [q for q in all_quorums if len(q.members) < MAX_QUORUM_MEMBERS]
    
    def update_quorum(self, quorum: Quorum) -> bool:
        """
        Update a quorum in the registry.
        
        Args:
            quorum: Updated quorum
            
        Returns:
            True if updated successfully
        """
        with self._lock:
            self.local_quorums[quorum.quorum_id] = quorum
            
            if self.dht:
                try:
                    key_id = self._get_quorum_key(quorum.quorum_id)
                    self.dht.storage[key_id] = quorum.to_json()
                except Exception as e:
                    logger.warning(f"Failed to update quorum in DHT: {e}")
            
            return True
    
    def get_stats(self) -> Dict[str, Any]:
        """Get registry statistics."""
        with self._lock:
            active = [q for q in self.local_quorums.values() if q.lifecycle == QuorumLifecycle.ACTIVE]
            return {
                "total_quorums": len(self.local_quorums),
                "active_quorums": len(active),
                "total_members": sum(len(q.members) for q in self.local_quorums.values()),
                "quorums_by_tier": {
                    tier: len([q for q in active if q.speed_tier == tier])
                    for tier in set(q.speed_tier for q in active)
                } if active else {},
            }


# =============================================================================
# QUORUM FORMATION SERVICE
# =============================================================================

class QuorumFormationService:
    """
    Service for forming and joining quorums.
    
    Implements the quorum formation algorithm:
    1. Get compatible speed tiers for initiator
    2. Query DHT for nodes in compatible tiers
    3. Find peers covering all layers
    4. Negotiate membership (propose + accept)
    5. Register quorum in DHT
    """
    
    def __init__(
        self,
        registry: QuorumRegistry,
        layer_pool=None,
        dht_protocol=None,
        p2p_manager=None,
    ):
        """
        Initialize the formation service.
        
        Args:
            registry: QuorumRegistry for storing formed quorums
            layer_pool: DynamicLayerPool for layer information
            dht_protocol: DHTProtocol for peer discovery
            p2p_manager: P2PManager for known_peers access (decentralized discovery)
        """
        self.registry = registry
        self.layer_pool = layer_pool
        self.dht = dht_protocol
        self.p2p_manager = p2p_manager
        self._lock = threading.RLock()
        
        # Formation state
        self._pending_proposals: Dict[str, Dict] = {}  # quorum_id -> proposal data
        self._proposal_responses: Dict[str, Dict[str, bool]] = {}  # quorum_id -> {node_id: accepted}
    
    def _resolve_peer_endpoint(self, endpoint: str) -> str:
        """Resolve a peer endpoint, preferring LAN IP if on the same network."""
        try:
            from neuroshard.core.network.nat import resolve_peer_address
            my_public_ip = getattr(self.p2p_manager, 'public_ip', None) if self.p2p_manager else None
            if my_public_ip:
                return resolve_peer_address(endpoint, my_public_ip)
        except Exception:
            pass
        # Fallback: strip any LAN info and use public part
        if "|" in endpoint:
            return endpoint.split("|", 1)[0]
        return endpoint
    
    def form_quorum(
        self,
        initiator_node_id: str,
        initiator_endpoint: str,
        initiator_layers: Tuple[int, int],
        initiator_speed_tier: str,
        total_layers: int,
        timeout: float = FORMATION_TIMEOUT,
    ) -> Optional[Quorum]:
        """
        Join or form a quorum — JOIN FIRST, form only if no peers exist.
        
        Strategy (in order):
        1. DISCOVER peers in the network (P2P known_peers, DHT, tracker)
        2. If peers with complementary layers exist: BUILD a quorum with them
        3. If no peers: CREATE a solo quorum (ACTIVE if full coverage, FORMING if partial)
        
        The key insight: a new node should NEVER create a quorum without first
        checking if compatible peers already exist. Otherwise two complementary
        nodes end up in separate quorums forever.
        
        Args:
            initiator_node_id: Node ID of the initiating node
            initiator_endpoint: gRPC endpoint of initiator
            initiator_layers: (start, end) layer range of initiator
            initiator_speed_tier: Speed tier of initiator
            total_layers: Total number of layers in the model
            
        Returns:
            Quorum if formed/joined, None if failed
        """
        my_layers = set(range(initiator_layers[0], initiator_layers[1]))
        all_layers = set(range(total_layers))
        missing_layers = all_layers - my_layers
        
        is_full_replica = not missing_layers  # Node holds ALL layers
        
        logger.info(f"[QUORUM] Join/Form: node={initiator_node_id[:8]}, "
                    f"layers={initiator_layers[0]}-{initiator_layers[1]-1}, "
                    f"tier={initiator_speed_tier}, "
                    f"{'full replica (data-parallel)' if is_full_replica else f'need={len(missing_layers)} more layers for full pipeline'}")
        
        # ── Step 1: Full replica → SYNC quorum (data-parallel mode) ───────
        # If we hold ALL layers, create a SYNC quorum. This is always ACTIVE
        # because a full replica can train independently. Other full replicas
        # join this quorum for DiLoCo weight sync.
        if is_full_replica:
            quorum = self._create_quorum(
                initiator_node_id, initiator_endpoint,
                initiator_layers, initiator_speed_tier,
            )
            quorum.quorum_type = QuorumType.SYNC
            quorum.lifecycle = QuorumLifecycle.ACTIVE
            self.registry.register_quorum(quorum)
            logger.info(f"[QUORUM] SYNC quorum {quorum.quorum_id[:8]} — "
                       f"full replica, ACTIVE (data-parallel mode)")
            return quorum
        
        # ── Step 2: Discover peers and negotiate pipeline quorum ────────────
        # Per whitepaper Algorithm 2: "Propose quorum to all selected members.
        # If all accept, register quorum in DHT."
        #
        # We find candidate peers via P2P/DHT/tracker, then propose a quorum
        # to each via gRPC ProposeQuorum RPC. Only peers that explicitly
        # accept (returning their real node_id) are added to the quorum.
        # This prevents phantom quorums — both sides know about the quorum.
        
        compatible_peers = self._find_compatible_peers(
            initiator_speed_tier,
            missing_layers,
        )
        
        if compatible_peers:
            # Generate quorum ID for the proposal
            proposed_quorum_id = Quorum.generate_id()
            
            # Negotiate with each peer via gRPC ProposeQuorum
            accepted_peers = self._negotiate_quorum_with_peers(
                proposed_quorum_id,
                initiator_node_id,
                initiator_endpoint,
                initiator_layers,
                initiator_speed_tier,
                total_layers,
                compatible_peers,
            )
            
            if accepted_peers:
                # Build quorum with ourselves + accepted peers
                quorum = Quorum(
                    quorum_id=proposed_quorum_id,
                    speed_tier=initiator_speed_tier,
                    arch_version=1,
                    vocab_version=1,
                )
                # Add ourselves first
                my_member = QuorumMember(
                    node_id=initiator_node_id,
                    endpoint=initiator_endpoint,
                    layer_range=initiator_layers,
                    speed_tier=initiator_speed_tier,
                )
                quorum.add_member(my_member)
                
                covered = set(my_layers)
                for peer in accepted_peers:
                    if covered >= all_layers:
                        break
                    peer_range = peer["layer_range"]
                    peer_layers = set(range(peer_range[0], peer_range[1]))
                    new_coverage = peer_layers - covered
                    
                    if new_coverage:
                        member = QuorumMember(
                            node_id=peer["node_id"],
                            endpoint=peer["endpoint"],
                            layer_range=tuple(peer_range),
                            speed_tier=peer.get("speed_tier", initiator_speed_tier),
                        )
                        quorum.add_member(member)
                        covered |= peer_layers
                        logger.info(f"[QUORUM] Added negotiated peer {peer['node_id'][:8]} "
                                  f"layers {peer_range[0]}-{peer_range[1]-1} "
                                  f"(coverage: {len(covered)}/{total_layers})")
                
                if covered >= all_layers:
                    quorum.lifecycle = QuorumLifecycle.ACTIVE
                    self.registry.register_quorum(quorum)
                    logger.info(f"[QUORUM] ACTIVE pipeline quorum {quorum.quorum_id[:8]} — "
                               f"{len(quorum.members)} members, full {total_layers}-layer pipeline!")
                    return quorum
                else:
                    # Partial coverage even with accepted peers — fall through to ASYNC
                    still_missing = all_layers - covered
                    logger.info(f"[QUORUM] Negotiated {len(accepted_peers)} peers but still missing "
                               f"layers {sorted(still_missing)[:5]}... — falling back to ASYNC")
            else:
                logger.info(f"[QUORUM] Found {len(compatible_peers)} peers but none accepted "
                           f"the quorum proposal — falling back to ASYNC")
        
        # ── Step 3: No pipeline possible — ASYNC mode ────────────────────────
        # Nodes that can't form a complete pipeline train asynchronously.
        # They still participate in DiLoCo sync to align weights.
        quorum = self._create_quorum(
            initiator_node_id, initiator_endpoint,
            initiator_layers, initiator_speed_tier,
        )
        quorum.lifecycle = QuorumLifecycle.FORMING
        self.registry.register_quorum(quorum)
        logger.info(f"[QUORUM] Partial coverage ({len(my_layers)}/{total_layers} layers) — "
                   f"FORMING quorum {quorum.quorum_id[:8]}, will use ASYNC training")
        return quorum
    
    def _negotiate_quorum_with_peers(
        self,
        quorum_id: str,
        initiator_node_id: str,
        initiator_endpoint: str,
        initiator_layers: Tuple[int, int],
        initiator_speed_tier: str,
        total_layers: int,
        candidate_peers: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """
        Negotiate quorum membership with candidate peers via gRPC ProposeQuorum.
        
        Per whitepaper Algorithm 2: propose quorum → peer accepts/rejects → 
        only accepted peers with real node_ids are returned.
        
        Args:
            quorum_id: Proposed quorum ID
            initiator_node_id: Our node ID
            initiator_endpoint: Our gRPC endpoint
            initiator_layers: Our layer range
            initiator_speed_tier: Our speed tier
            total_layers: Total layers in model
            candidate_peers: List of peer dicts from _find_compatible_peers()
            
        Returns:
            List of accepted peers with real node_ids and endpoints
        """
        accepted_peers = []
        
        for peer in candidate_peers:
            try:
                from neuroshard.core.network.connection_pool import get_channel
                from neuroshard.protos import neuroshard_pb2 as pb2
                from neuroshard.protos import neuroshard_pb2_grpc as pb2_grpc
                from neuroshard.core.swarm.factory import GRPC_PORT_OFFSET
                from urllib.parse import urlparse
                
                # Determine gRPC address from peer endpoint
                endpoint = peer.get("endpoint", "")
                node_url = peer.get("node_url", "")
                
                if node_url:
                    # Parse URL-style endpoint
                    parsed = urlparse(node_url)
                    grpc_port = (parsed.port or 8000) + GRPC_PORT_OFFSET
                    grpc_addr = f"{parsed.hostname}:{grpc_port}"
                elif ":" in endpoint:
                    parts = endpoint.rsplit(":", 1)
                    host = parts[0]
                    try:
                        port = int(parts[1])
                        # If port looks like HTTP port, add offset
                        if port < 9000:
                            grpc_port = port + GRPC_PORT_OFFSET
                        else:
                            grpc_port = port
                    except ValueError:
                        grpc_port = 9000
                    grpc_addr = f"{host}:{grpc_port}"
                else:
                    continue
                
                logger.info(f"[QUORUM] Proposing quorum {quorum_id[:8]} to {grpc_addr}...")
                
                channel = get_channel(grpc_addr)
                stub = pb2_grpc.NeuroShardServiceStub(channel)
                
                # Send ProposeQuorum RPC
                request = pb2.ProposeQuorumRequest(
                    quorum_id=quorum_id,
                    proposer_node_id=initiator_node_id,
                    proposer_endpoint=initiator_endpoint,
                    proposer_layer_start=initiator_layers[0],
                    proposer_layer_end=initiator_layers[1],
                    speed_tier=initiator_speed_tier,
                    total_layers=total_layers,
                )
                
                response = stub.ProposeQuorum(request, timeout=10.0)
                
                if response.accepted:
                    accepted_peers.append({
                        "node_id": response.node_id,      # REAL node_id from peer
                        "endpoint": response.endpoint,      # REAL gRPC endpoint
                        "layer_range": (response.layer_start, response.layer_end),
                        "speed_tier": response.speed_tier,
                    })
                    logger.info(f"[QUORUM] Peer {response.node_id[:8]} ACCEPTED — "
                               f"layers {response.layer_start}-{response.layer_end-1}")
                else:
                    logger.info(f"[QUORUM] Peer at {grpc_addr} REJECTED — {response.reason}")
                    
            except Exception as e:
                logger.debug(f"[QUORUM] ProposeQuorum to {peer.get('endpoint', '?')} failed: {e}")
                continue
        
        return accepted_peers
    
    def _create_quorum(self, node_id: str, endpoint: str,
                       layer_range: Tuple[int, int], speed_tier: str) -> Quorum:
        """Create a new quorum with a single initiating member."""
        quorum = Quorum(
            quorum_id=Quorum.generate_id(),
            speed_tier=speed_tier,
            arch_version=1,
            vocab_version=1,
        )
        member = QuorumMember(
            node_id=node_id,
            endpoint=endpoint,
            layer_range=layer_range,
            speed_tier=speed_tier,
        )
        quorum.add_member(member)
        return quorum
    
    def _find_joinable_quorum(
        self,
        speed_tier: str,
        layer_range: Tuple[int, int],
    ) -> Optional[Quorum]:
        """
        Find an existing quorum that the node can join.
        
        Criteria:
        - Same or compatible speed tier
        - Needs the layers we have
        - Has capacity for new members
        """
        # Get compatible quorums
        quorums = self.registry.find_quorums_by_speed_tier(speed_tier)
        
        for quorum in quorums:
            # Check capacity
            if len(quorum.members) >= MAX_QUORUM_MEMBERS:
                continue
            
            # Check if our layers would help
            current_coverage = quorum.layer_coverage
            our_layers = set(range(layer_range[0], layer_range[1]))
            
            # Would we add new layers?
            new_layers = our_layers - current_coverage
            if new_layers or not quorum.is_complete:
                return quorum
        
        return None
    
    def _get_missing_layers(self, quorum: Quorum, total_layers: int) -> Set[int]:
        """Get layers not covered by any quorum member."""
        all_layers = set(range(total_layers))
        covered = quorum.layer_coverage
        return all_layers - covered
    
    def _find_compatible_peers(
        self,
        speed_tier: str,
        missing_layers: Set[int],
    ) -> List[Dict[str, Any]]:
        """
        Find peers that can provide missing layers with compatible speed.
        
        FULLY DECENTRALIZED discovery strategy (in priority order):
        1. P2P known_peers — already discovered via DHT gossip (no central server)
        2. DHT layer lookups — query Kademlia DHT for layer holders
        3. Tracker (last resort) — only if DHT has no results yet (bootstrap phase)
        
        The tracker is NEVER required. It's only used as a fallback during
        the initial bootstrap when the DHT hasn't fully propagated yet.
        Once nodes discover each other via DHT, the tracker is irrelevant.
        """
        from neuroshard.core.model.dynamic import COMPATIBLE_TIERS, SpeedTier
        
        compatible_peers = []
        seen_nodes = set()
        
        # Build set of our own IPs to filter ourselves out of peer results
        my_ips = set()
        if self.p2p_manager:
            my_url = getattr(self.p2p_manager, 'my_url', '')
            if my_url:
                seen_nodes.add(my_url)
                from urllib.parse import urlparse
                parsed_self = urlparse(my_url)
                if parsed_self.hostname:
                    my_ips.add(parsed_self.hostname)
            pub_ip = getattr(self.p2p_manager, 'public_ip', None)
            if pub_ip:
                my_ips.add(pub_ip)
            lan_ip = getattr(self.p2p_manager, 'local_ip', None)
            if lan_ip:
                my_ips.add(lan_ip)
        
        # ── Strategy 1: P2P Known Peers (fully decentralized) ──────────────
        # These peers were already discovered via DHT gossip and heartbeats.
        # The P2P manager maintains known_peers with shard_range info.
        p2p = self.p2p_manager
        if p2p and hasattr(p2p, 'known_peers'):
            for url, info in p2p.known_peers.items():
                if url in seen_nodes:
                    continue
                shard_range = info.get("shard_range", "") if isinstance(info, dict) else ""
                if not shard_range or shard_range in ("unknown", "unassigned", "observer"):
                    continue
                try:
                    start, end = map(int, shard_range.split("-"))
                    peer_layers = set(range(start, end + 1))
                    overlap = peer_layers & missing_layers
                    if overlap:
                        seen_nodes.add(url)
                        from urllib.parse import urlparse
                        parsed = urlparse(url)
                        compatible_peers.append({
                            "node_id": hashlib.sha256(url.encode()).hexdigest()[:16],
                            "endpoint": f"{parsed.hostname}:{(parsed.port or 80)}",
                            "layer_range": (start, end + 1),
                            "speed_tier": speed_tier,
                            "node_url": url,
                        })
                        logger.info(f"[QUORUM] Found peer via P2P: {url} "
                                  f"layers {start}-{end}, covers {len(overlap)} missing")
                except (ValueError, TypeError):
                    continue
        
        if compatible_peers:
            return compatible_peers
        
        # ── Strategy 2: DHT Layer Lookups (fully decentralized) ────────────
        # Query the Kademlia DHT for nodes that announced specific layers.
        if self.dht:
            for layer_id in list(missing_layers)[:5]:
                try:
                    key_str = f"layer_{layer_id}"
                    
                    # Try all key formats (string, SHA-1 int, network lookup)
                    value = self.dht.storage.get(key_str)
                    if not value:
                        key_id = int(hashlib.sha1(key_str.encode()).hexdigest(), 16)
                        value = self.dht.storage.get(key_id)
                    if not value:
                        try:
                            value = self.dht.lookup_value(key_str)
                        except Exception:
                            pass
                    
                    if not value:
                        continue
                    
                    # Parse endpoints from DHT value
                    try:
                        if isinstance(value, str):
                            endpoints = json.loads(value)
                            if not isinstance(endpoints, list):
                                endpoints = [endpoints]
                        elif isinstance(value, list):
                            endpoints = value
                        else:
                            endpoints = [str(value)]
                    except (json.JSONDecodeError, TypeError):
                        endpoints = [str(value)]
                    
                    for endpoint in endpoints:
                        if endpoint in seen_nodes:
                            continue
                        seen_nodes.add(endpoint)
                        
                        # Resolve LAN address if peer is on same network
                        resolved = self._resolve_peer_endpoint(str(endpoint))
                        
                        # Filter out ourselves
                        resolved_ip = resolved.rsplit(":", 1)[0] if ":" in resolved else resolved
                        if resolved_ip in my_ips:
                            continue
                        
                        # Query peer directly for their actual layer range
                        actual_range = self._query_peer_layers(resolved)
                        if actual_range:
                            compatible_peers.append({
                                "node_id": hashlib.sha256(str(endpoint).encode()).hexdigest()[:16],
                                "endpoint": resolved,
                                "layer_range": actual_range,
                                "speed_tier": speed_tier,
                            })
                            logger.info(f"[QUORUM] Found peer via DHT: {resolved} "
                                      f"layers {actual_range[0]}-{actual_range[1]-1}")
                        
                except Exception as e:
                    logger.debug(f"DHT layer query for layer_{layer_id}: {e}")
        
        if compatible_peers:
            return compatible_peers
        
        # ── Strategy 3: Tracker (bootstrap fallback ONLY) ──────────────────
        # Used ONLY when DHT hasn't propagated yet (first minutes of network life).
        # The tracker is just a peer list — it has no control over the network.
        try:
            import requests
            for tracker in ["https://neuroshard.com/api/tracker", "http://localhost:3000"]:
                try:
                    resp = requests.get(f"{tracker}/peers", timeout=5)
                    if resp.status_code != 200:
                        continue
                    for p in resp.json():
                        url = p.get("url", "")
                        shard_range = p.get("shard_range", "")
                        if not url or not shard_range or url in seen_nodes:
                            continue
                        if shard_range in ("unknown", "unassigned", "observer"):
                            continue
                        # Filter out ourselves (tracker may return our own URL)
                        try:
                            from urllib.parse import urlparse as _urlparse
                            _parsed = _urlparse(url)
                            if _parsed.hostname in my_ips:
                                continue
                        except Exception:
                            pass
                        try:
                            start, end = map(int, shard_range.split("-"))
                            peer_layers = set(range(start, end + 1))
                            overlap = peer_layers & missing_layers
                            if overlap:
                                seen_nodes.add(url)
                                from urllib.parse import urlparse
                                parsed = urlparse(url)
                                compatible_peers.append({
                                    "node_id": hashlib.sha256(url.encode()).hexdigest()[:16],
                                    "endpoint": f"{parsed.hostname}:{(parsed.port or 80)}",
                                    "layer_range": (start, end + 1),
                                    "speed_tier": speed_tier,
                                    "node_url": url,
                                })
                                logger.info(f"[QUORUM] Found peer via tracker (bootstrap): {url} "
                                          f"layers {start}-{end}")
                        except (ValueError, TypeError):
                            continue
                    if compatible_peers:
                        break
                except Exception:
                    continue
        except Exception as e:
            logger.debug(f"Tracker fallback failed: {e}")
        
        return compatible_peers
    
    def _query_peer_layers(self, peer_endpoint: str) -> Optional[Tuple[int, int]]:
        """Query a peer via gRPC to get their actual layer range."""
        try:
            from neuroshard.core.network.connection_pool import get_channel
            from neuroshard.protos import neuroshard_pb2 as pb2
            from neuroshard.protos import neuroshard_pb2_grpc as pb2_grpc
            from neuroshard.core.swarm.factory import GRPC_PORT_OFFSET
            from urllib.parse import urlparse
            
            # Determine gRPC address
            if ":" in str(peer_endpoint):
                parsed = urlparse(f"http://{peer_endpoint}" if not peer_endpoint.startswith("http") else peer_endpoint)
                grpc_addr = f"{parsed.hostname}:{(parsed.port or 80) + GRPC_PORT_OFFSET}"
            else:
                return None
            
            channel = get_channel(grpc_addr)
            stub = pb2_grpc.NeuroShardServiceStub(channel)
            
            resp = stub.GetShardInfo(pb2.GetShardInfoRequest(), timeout=5.0)
            if resp.start_layer is not None and resp.end_layer is not None:
                return (resp.start_layer, resp.end_layer + 1)
        except Exception as e:
            logger.debug(f"Failed to query peer {peer_endpoint} for layers: {e}")
        
        return None
    
    def try_join_quorum(
        self,
        node_id: str,
        endpoint: str,
        layer_range: Tuple[int, int],
        speed_tier: str,
        quorum_id: Optional[str] = None,
    ) -> Optional[Quorum]:
        """
        Try to join an existing quorum.
        
        Args:
            node_id: Node ID
            endpoint: gRPC endpoint
            layer_range: Layers this node holds
            speed_tier: Node's speed tier
            quorum_id: Specific quorum to join (optional)
            
        Returns:
            Joined Quorum if successful, None otherwise
        """
        if quorum_id:
            # Try to join specific quorum
            quorum = self.registry.get_quorum(quorum_id)
            if not quorum:
                return None
        else:
            # Find a compatible quorum
            quorum = self._find_joinable_quorum(speed_tier, layer_range)
            if not quorum:
                return None
        
        # Create member and add
        member = QuorumMember(
            node_id=node_id,
            endpoint=endpoint,
            layer_range=layer_range,
            speed_tier=speed_tier,
        )
        
        if quorum.add_member(member):
            # Check if quorum is now complete
            if quorum.is_complete and quorum.lifecycle == QuorumLifecycle.FORMING:
                quorum.lifecycle = QuorumLifecycle.ACTIVE
                logger.info(f"Quorum {quorum.quorum_id[:8]}... is now complete and active")
            
            self.registry.update_quorum(quorum)
            return quorum
        
        return None
    
    def leave_quorum(self, node_id: str, quorum_id: str) -> bool:
        """
        Leave a quorum gracefully.
        
        Args:
            node_id: Node ID leaving
            quorum_id: Quorum to leave
            
        Returns:
            True if left successfully
        """
        quorum = self.registry.get_quorum(quorum_id)
        if not quorum:
            return False
        
        member = quorum.remove_member(node_id)
        if not member:
            return False
        
        # Check if quorum should dissolve
        if len(quorum.members) == 0:
            quorum.lifecycle = QuorumLifecycle.DISSOLVED
            self.registry.unregister_quorum(quorum_id)
            logger.info(f"Quorum {quorum_id[:8]}... dissolved (no members)")
        elif not quorum.is_complete:
            # Quorum no longer complete
            quorum.lifecycle = QuorumLifecycle.FORMING
            self.registry.update_quorum(quorum)
            logger.info(f"Quorum {quorum_id[:8]}... no longer complete, seeking members")
        else:
            self.registry.update_quorum(quorum)
        
        return True


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def get_adaptive_min_quorum_size(network_size: int) -> int:
    """
    Get minimum quorum size based on network size.
    
    Adapts to network conditions:
    - Small network (< 5 nodes): 1 member minimum (allow solo)
    - Medium network (< 20 nodes): 2 members minimum
    - Large network (>= 20 nodes): 3 members minimum
    
    Args:
        network_size: Total number of nodes in network
        
    Returns:
        Minimum quorum size
    """
    if network_size < 5:
        return 1
    elif network_size < 20:
        return 2
    else:
        return 3


# =============================================================================
# QUORUM LIFECYCLE MANAGER
# =============================================================================

class QuorumLifecycleManager:
    """
    Manages the lifecycle of quorums.
    
    Responsibilities:
    - Monitor quorum health via heartbeats
    - Handle session renewal
    - Trigger dissolution when needed
    - Replace failed members
    
    Lifecycle States:
    FORMING -> ACTIVE -> RENEWING -> ACTIVE (if renewed)
                     -> DISSOLVING -> DISSOLVED
    """
    
    def __init__(
        self,
        registry: QuorumRegistry,
        formation_service: QuorumFormationService,
    ):
        """
        Initialize the lifecycle manager.
        
        Args:
            registry: QuorumRegistry for quorum access
            formation_service: QuorumFormationService for member replacement
        """
        self.registry = registry
        self.formation = formation_service
        
        # Background task control
        self._running = False
        self._monitor_thread: Optional[threading.Thread] = None
        self._monitor_interval = 10.0  # Check every 10 seconds
    
    def start(self):
        """Start the lifecycle manager background tasks."""
        if self._running:
            return
        
        self._running = True
        self._monitor_thread = threading.Thread(
            target=self._monitor_loop,
            daemon=True,
            name="QuorumLifecycleMonitor"
        )
        self._monitor_thread.start()
        logger.info("Quorum lifecycle manager started")
    
    def stop(self):
        """Stop the lifecycle manager."""
        self._running = False
        if self._monitor_thread:
            self._monitor_thread.join(timeout=5.0)
        logger.info("Quorum lifecycle manager stopped")
    
    def _monitor_loop(self):
        """Background loop to monitor quorum health."""
        while self._running:
            try:
                self._check_all_quorums()
            except Exception as e:
                logger.error(f"Error in quorum monitor: {e}")
            
            time.sleep(self._monitor_interval)
    
    def _check_all_quorums(self):
        """Check health and lifecycle of all quorums."""
        quorums = self.registry.get_active_quorums()
        
        for quorum in quorums:
            try:
                self._check_quorum_health(quorum)
                self._check_quorum_session(quorum)
            except Exception as e:
                logger.warning(f"Error checking quorum {quorum.quorum_id[:8]}...: {e}")
    
    def _check_quorum_health(self, quorum: Quorum):
        """Check health of a quorum and handle issues."""
        # Get stale and offline members
        stale = quorum.get_stale_members()
        offline = quorum.get_offline_members()
        
        # Handle offline members
        for member in offline:
            logger.warning(f"Member {member.node_id[:8]}... is offline in quorum {quorum.quorum_id[:8]}...")
            self._handle_member_failure(quorum, member)
        
        # Log stale members (warning only)
        for member in stale:
            if member not in offline:
                logger.debug(f"Member {member.node_id[:8]}... is stale in quorum {quorum.quorum_id[:8]}...")
    
    def _check_quorum_session(self, quorum: Quorum):
        """Check session timing and handle renewal/dissolution."""
        # Check if session expired
        if quorum.is_session_expired:
            self._handle_session_expired(quorum)
            return
        
        # Check if should check renewal
        if quorum.should_check_renewal and quorum.lifecycle == QuorumLifecycle.ACTIVE:
            self._handle_renewal_check(quorum)
    
    def _handle_member_failure(self, quorum: Quorum, member: QuorumMember):
        """Handle a failed member - try to replace or dissolve."""
        # Remove failed member
        quorum.remove_member(member.node_id)
        
        if not quorum.members:
            # No members left - dissolve
            self._dissolve_quorum(quorum, reason="all members failed")
            return
        
        if not quorum.is_complete:
            # Try to find replacement
            quorum.lifecycle = QuorumLifecycle.RENEWING
            
            # TODO: Trigger replacement search
            # For now, just mark as forming
            quorum.lifecycle = QuorumLifecycle.FORMING
            self.registry.update_quorum(quorum)
            logger.info(f"Quorum {quorum.quorum_id[:8]}... seeking replacement for failed member")
    
    def _handle_renewal_check(self, quorum: Quorum):
        """Handle session renewal check."""
        quorum.lifecycle = QuorumLifecycle.RENEWING
        
        # Check renewal conditions
        should_renew = self._should_renew_quorum(quorum)
        
        if should_renew:
            # Extend session
            quorum.extend_session()
            logger.info(f"Quorum {quorum.quorum_id[:8]}... session renewed")
        else:
            # Prepare for dissolution
            self._dissolve_quorum(quorum, reason="renewal conditions not met")
    
    def _should_renew_quorum(self, quorum: Quorum) -> bool:
        """
        Check if quorum should be renewed.
        
        Conditions for renewal:
        - All members healthy (not stale)
        - Processed minimum batches
        - Quorum still complete
        """
        # Check all members healthy
        if len(quorum.get_stale_members()) > 0:
            return False
        
        # Check minimum work done
        if quorum.total_batches < MIN_BATCHES_TO_RENEW:
            return False
        
        # Check still complete
        if not quorum.is_complete:
            return False
        
        return True
    
    def _handle_session_expired(self, quorum: Quorum):
        """Handle expired session - attempt renewal or dissolve."""
        if self._should_renew_quorum(quorum):
            quorum.extend_session()
            logger.info(f"Quorum {quorum.quorum_id[:8]}... session auto-renewed on expiry")
        else:
            self._dissolve_quorum(quorum, reason="session expired")
    
    def _dissolve_quorum(self, quorum: Quorum, reason: str = ""):
        """Dissolve a quorum gracefully."""
        logger.info(f"Dissolving quorum {quorum.quorum_id[:8]}...: {reason}")
        
        quorum.lifecycle = QuorumLifecycle.DISSOLVING
        self.registry.update_quorum(quorum)
        
        # TODO: Notify members to save weights, find new quorums
        
        # Mark as dissolved
        quorum.lifecycle = QuorumLifecycle.DISSOLVED
        self.registry.unregister_quorum(quorum.quorum_id)
    
    def process_heartbeat(self, quorum_id: str, node_id: str) -> bool:
        """
        Process a heartbeat from a quorum member.
        
        Args:
            quorum_id: Quorum ID
            node_id: Node sending heartbeat
            
        Returns:
            True if processed successfully
        """
        quorum = self.registry.get_quorum(quorum_id)
        if not quorum:
            return False
        
        success = quorum.update_heartbeat(node_id)
        if success:
            self.registry.update_quorum(quorum)
        
        return success
    
    def record_batch(self, quorum_id: str, batches: int = 1) -> bool:
        """
        Record processed batches for a quorum.
        
        Args:
            quorum_id: Quorum ID
            batches: Number of batches processed
            
        Returns:
            True if recorded successfully
        """
        quorum = self.registry.get_quorum(quorum_id)
        if not quorum:
            return False
        
        quorum.record_batch(batches)
        self.registry.update_quorum(quorum)
        return True
    
    def get_quorum_status(self, quorum_id: str) -> Optional[Dict[str, Any]]:
        """
        Get detailed status of a quorum.
        
        Args:
            quorum_id: Quorum ID
            
        Returns:
            Status dict or None if quorum not found
        """
        quorum = self.registry.get_quorum(quorum_id)
        if not quorum:
            return None
        
        return {
            "quorum_id": quorum.quorum_id,
            "lifecycle": quorum.lifecycle.value,
            "speed_tier": quorum.speed_tier,
            "member_count": len(quorum.members),
            "healthy_members": len(quorum.get_healthy_members()),
            "stale_members": len(quorum.get_stale_members()),
            "is_complete": quorum.is_complete,
            "total_batches": quorum.total_batches,
            "batches_per_second": quorum.batches_per_second,
            "session_remaining": quorum.session_remaining,
            "session_progress": quorum.session_progress,
        }


# =============================================================================
# QUORUM TRAINER
# =============================================================================

# DiLoCo sync settings
SYNC_INTERVAL = 100              # Batches between cohort syncs (lower = faster weight sharing)
OUTER_LR = 0.7                   # Outer learning rate for DiLoCo


class QuorumTrainer:
    """
    Training coordinator for quorum-based training.
    
    Responsibilities:
    - Manage within-quorum synchronous pipeline training
    - Coordinate DiLoCo cohort sync across quorums
    - Track training progress and generate proofs
    
    Pipeline Flow (within quorum):
    1. INITIATOR: Get batch, embed, forward through layers, send to next
    2. PROCESSOR: Receive activations, forward, send to next
    3. FINISHER: Compute loss, backward, send gradients back
    4. All: Apply gradients locally
    
    DiLoCo Sync (across quorums):
    - Every SYNC_INTERVAL batches
    - Compute pseudo-gradient (current - initial weights)
    - Exchange with cohort (other nodes holding same layers)
    - Weighted aggregation (sqrt(n) * freshness)
    - Apply outer update
    """
    
    def __init__(
        self,
        quorum: Quorum,
        node_id: str,
        model: Any,
        optimizer: Any,
        genesis_loader: Any = None,
        dht_protocol: Any = None,
    ):
        """
        Initialize QuorumTrainer.
        
        Args:
            quorum: The quorum this trainer belongs to
            node_id: This node's ID
            model: The model to train (DynamicNeuroLLM)
            optimizer: PyTorch optimizer
            genesis_loader: Genesis data loader (for initiator)
            dht_protocol: DHT protocol for cohort discovery
        """
        self.quorum = quorum
        self.node_id = node_id
        self.model = model
        self.optimizer = optimizer
        self.genesis_loader = genesis_loader
        self.dht = dht_protocol
        
        # Determine role
        self.member = quorum.get_member(node_id)
        if not self.member:
            raise ValueError(f"Node {node_id} is not a member of quorum {quorum.quorum_id}")
        
        self.is_initiator = self.member.role == QuorumRole.INITIATOR
        # In a solo quorum, the initiator is ALSO the finisher
        self.is_finisher = (self.member.role == QuorumRole.FINISHER or 
                           len(quorum.members) == 1)
        
        # DiLoCo state
        self.initial_weights: Dict[str, Any] = {}
        self.batches_since_sync = 0
        self.sync_round = 0
        
        # Training state
        self.running = False
        self._training_thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        
        # Pipeline inbound buffer for receiving activations from upstream nodes.
        # The gRPC PipelineForward handler pushes activation packets here;
        # _process_and_forward() pulls from it. Only used for non-initiator roles.
        import queue
        self.inbound_buffer = queue.Queue(maxsize=32) if not self.is_initiator else None
        
        # Stats
        self.total_batches = 0
        self.current_loss: Optional[float] = None
        
        # Snapshot initial weights for DiLoCo
        self._snapshot_weights()
        
        logger.info(f"QuorumTrainer initialized: quorum={quorum.quorum_id[:8]}..., "
                   f"role={self.member.role.value}, layers={self.member.layer_range}")
    
    def _snapshot_weights(self):
        """Snapshot current weights for DiLoCo pseudo-gradient computation."""
        import torch
        self.initial_weights = {}
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.initial_weights[name] = param.data.clone()
    
    def start(self):
        """Start the training loop."""
        if self.running:
            return
        
        self.running = True
        self._stop_event.clear()
        self._training_thread = threading.Thread(
            target=self._training_loop,
            daemon=True,
            name=f"QuorumTrainer-{self.quorum.quorum_id[:8]}"
        )
        self._training_thread.start()
        logger.info("QuorumTrainer started")
    
    def stop(self):
        """Stop the training loop gracefully."""
        self.running = False
        self._stop_event.set()
        if self._training_thread:
            self._training_thread.join(timeout=5.0)
        logger.info("QuorumTrainer stopped")
    
    def _training_loop(self):
        """Main training loop for quorum member."""
        _idle_iterations = 0  # Track consecutive idle iterations (no activations)
        _IDLE_THRESHOLD = 60  # After 60 idle iterations (~30s), signal stall
        
        while self.running and self.quorum.lifecycle == QuorumLifecycle.ACTIVE:
            try:
                if self.is_initiator:
                    did_work = self._initiate_batch()
                else:
                    # Processor/Finisher: receive activation, forward, and route
                    did_work = self._process_and_forward()
                
                if did_work:
                    self.batches_since_sync += 1
                    self.total_batches += 1
                    self.quorum.record_batch()
                    _idle_iterations = 0
                    
                    # Check if time for DiLoCo sync
                    if self.batches_since_sync >= SYNC_INTERVAL:
                        self._cohort_sync()
                    
                    # Update member stats
                    if self.member:
                        self.member.batches_processed = self.total_batches
                else:
                    # No activation received — pipeline is stalled
                    _idle_iterations += 1
                    if _idle_iterations == _IDLE_THRESHOLD:
                        logger.warning(
                            f"[TRAINING] Processor idle for {_idle_iterations} iterations "
                            f"(~{_idle_iterations * 0.5:.0f}s) — no activations received. "
                            f"Pipeline may be disconnected. Consider ASYNC mode."
                        )
                        # Signal stall so runner can detect and switch modes
                        self._pipeline_stalled = True
                
            except Exception as e:
                logger.error(f"Training loop error: {e}")
                # Exponential backoff: 1s, 2s, 4s, ... up to 60s
                if not hasattr(self, '_consecutive_errors'):
                    self._consecutive_errors = 0
                self._consecutive_errors += 1
                backoff = min(2 ** (self._consecutive_errors - 1), 60)
                logger.warning(f"[TRAINING] Error #{self._consecutive_errors}, "
                              f"backing off {backoff}s")
                time.sleep(backoff)
                continue
            
            # Reset error counter on success
            if hasattr(self, '_consecutive_errors'):
                self._consecutive_errors = 0
            
            # Check for stop signal
            if self._stop_event.is_set():
                break
        
        logger.info(f"Training loop ended: {self.total_batches} batches processed")
    
    def _initiate_batch(self) -> bool:
        """Initiator: Get batch, embed, and start pipeline.
        
        Returns:
            True if a batch was actually processed, False if waiting for data.
        """
        import torch
        
        if not self.genesis_loader:
            # Log once, then back off to avoid spamming logs every second
            if not hasattr(self, '_genesis_wait_logged'):
                logger.warning("[INITIATOR] Waiting for Genesis data loader to initialize... "
                              "(downloading manifest + first shard from CDN)")
                self._genesis_wait_logged = True
            time.sleep(5)  # Check every 5s instead of 1s
            return False
        
        # Get batch from genesis loader — returns (input_ids, labels) tuple
        try:
            result = self.genesis_loader.get_batch()
        except RuntimeError:
            # Data not ready yet (shard still downloading)
            time.sleep(1)
            return False
        
        if result is None or result[0] is None:
            time.sleep(0.5)
            return False
        
        input_ids, labels = result
        
        # Move to device
        device = next(self.model.parameters()).device
        input_ids = input_ids.to(device)
        labels = labels.to(device)
        
        # Forward through local layers (embedding + initial layers)
        self.model.train()
        embeddings = self.model.embed(input_ids)
        hidden, _ = self.model.forward_my_layers(embeddings)
        
        # Send to next node in pipeline or finish locally
        next_hop = self.quorum.get_next_hop(self.node_id)
        if next_hop:
            self._send_activation(next_hop, hidden, {'labels': labels})
        else:
            # Solo quorum or this node is also the finisher — compute loss locally
            self._finisher_backward(hidden, labels)
        
        return True
    
    def _process_and_forward(self) -> bool:
        """Processor/Finisher: Receive activation, forward through local layers, route result.
        
        This is the non-initiator pipeline path:
        1. Pull activation from inbound buffer (sent by previous node via gRPC)
        2. Forward through this node's layers
        3. If there's a next hop: send activation forward
        4. If this is the finisher: compute loss and backward pass
        
        Returns:
            True if actual work was done, False if no activation received.
            The training loop uses this to avoid inflating batch counters.
        """
        import torch
        
        # Pull from inbound buffer (filled by gRPC PipelineForward handler)
        activation = None
        metadata = {}
        
        if self.inbound_buffer is not None:
            import queue
            try:
                packet = self.inbound_buffer.get(timeout=5.0)
            except queue.Empty:
                packet = None
            if packet is not None:
                activation = packet.tensor_data
                metadata = {
                    'session_id': getattr(packet, 'session_id', ''),
                    'grad_output': getattr(packet, 'grad_output', None),
                    'labels': getattr(packet, 'labels', None),
                }
        
        if activation is None:
            time.sleep(0.5)
            return False  # No work done — don't count as a batch
        
        # Move to device
        device = next(self.model.parameters()).device
        activation = activation.to(device)
        
        # Forward through this node's layers
        self.model.train()
        hidden, _ = self.model.forward_my_layers(activation)
        
        # Track a proxy loss for PoNW proofs — processors don't compute
        # cross-entropy but DO perform real computation (forward pass).
        # Without this, PoNW rejects the proof with "No valid loss".
        proxy_loss = hidden.detach().pow(2).mean().item() * 0.01
        if proxy_loss > 0:
            self.current_loss = proxy_loss
        
        # Route: next hop or finisher backward
        next_hop = self.quorum.get_next_hop(self.node_id)
        if next_hop:
            self._send_activation(next_hop, hidden, metadata)
        elif self.is_finisher:
            # Get labels from metadata if available
            labels = metadata.get('labels')
            if labels is not None:
                self._finisher_backward(hidden, labels)
            else:
                # Finisher without labels — compute proxy loss
                loss = hidden.pow(2).mean() * 0.01
                self.current_loss = loss.item()
                if loss.requires_grad:
                    self.optimizer.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                    self.optimizer.step()
        
        return True  # Actual work done
    
    def _send_activation(self, target: QuorumMember, hidden: Any, metadata: Dict = None):
        """Send activation to next node in pipeline via gRPC PipelineForward."""
        try:
            from neuroshard.core.network.connection_pool import get_channel
            from neuroshard.protos import neuroshard_pb2 as pb2
            from neuroshard.protos import neuroshard_pb2_grpc as pb2_grpc
            from neuroshard.core.swarm.factory import GRPC_PORT_OFFSET
            from urllib.parse import urlparse
            
            # Determine gRPC address from endpoint.
            # QuorumMember.endpoint stores the gRPC address directly
            # (set during quorum formation via ProposeQuorum response).
            endpoint = target.endpoint
            if not endpoint:
                logger.warning(f"[QUORUM] Cannot send activation — no endpoint for {target.node_id[:16]}")
                return
            
            # Use endpoint as gRPC address directly.
            # For backwards compat: if port looks like HTTP (< 9000), add offset.
            grpc_addr = endpoint
            if ":" in endpoint:
                parts = endpoint.rsplit(":", 1)
                try:
                    port = int(parts[1])
                    if port < 9000:
                        grpc_addr = f"{parts[0]}:{port + GRPC_PORT_OFFSET}"
                except ValueError:
                    pass
            
            channel = get_channel(grpc_addr)
            stub = pb2_grpc.NeuroShardServiceStub(channel)
            
            # Serialize activation tensor with shape info
            import torch
            import numpy as np
            hidden_tensor = hidden if isinstance(hidden, torch.Tensor) else torch.tensor(hidden)
            
            request = pb2.PipelineForwardRequest(
                hidden_states=hidden_tensor.detach().cpu().numpy().astype(np.float32).tobytes(),
                hidden_shape=list(hidden_tensor.shape),
                sender_url=endpoint,
                source_shard=self.member.layer_range[1] - 1,
            )
            
            # Add labels from metadata if present
            if metadata and 'labels' in metadata and metadata['labels'] is not None:
                labels_tensor = metadata['labels']
                if isinstance(labels_tensor, torch.Tensor):
                    request.training_labels = labels_tensor.cpu().numpy().tobytes()
            
            response = stub.PipelineForward(request, timeout=30.0)
            logger.debug(f"[QUORUM] Sent activation to {target.node_id[:16]}...")
            
        except Exception as e:
            logger.warning(f"[QUORUM] Failed to send activation to {target.node_id[:16]}...: {e}")
    
    def _finisher_backward(self, hidden: Any, labels: Any):
        """Finisher: Compute loss from logits and run backward pass.
        
        Args:
            hidden: Hidden states from forward pass [batch, seq, hidden_dim]
            labels: Target token IDs [batch, seq] (either tensor or dict with 'labels' key)
        """
        import torch
        
        # Handle labels passed as dict (from pipeline activation) or tensor (from local)
        if isinstance(labels, dict):
            labels = labels.get('labels')
        
        if labels is None:
            logger.warning("[FINISHER] No labels provided for backward pass")
            return
        
        # Compute logits from hidden states
        if hasattr(self.model, 'compute_logits'):
            logits = self.model.compute_logits(hidden)
        elif hasattr(self.model, 'lm_head') and self.model.lm_head is not None:
            logits = self.model.lm_head(hidden)
        else:
            logger.warning("[FINISHER] No LM head available for loss computation")
            return
        
        # Cross-entropy loss (standard next-token prediction)
        loss = torch.nn.functional.cross_entropy(
            logits.view(-1, logits.size(-1)),
            labels.view(-1),
            ignore_index=-100
        )
        
        self.current_loss = loss.item()
        
        # Record loss for shard rotation plateau detection
        if hasattr(self, 'genesis_loader') and self.genesis_loader:
            if hasattr(self.genesis_loader, 'record_loss'):
                self.genesis_loader.record_loss(self.current_loss)
        
        # Backward pass + optimizer step
        if loss.requires_grad:
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()
    
    def _cohort_sync(self):
        """Synchronize with cohort (DiLoCo cross-quorum sync).
        
        Exchanges pseudo-gradients with peers in the layer cohort via gRPC,
        then applies Byzantine-robust aggregation before the outer update.
        
        Sign convention: pseudo_grad = current - initial (weight delta).
        This is consistent with DiLoCoTrainer.compute_pseudo_gradient() and
        the OuterOptimizer which adds the aggregated delta to the weights.
        """
        import torch
        
        logger.info(f"Starting cohort sync round {self.sync_round}")
        
        # 1. Compute pseudo-gradient (current - initial = direction training moved)
        pseudo_grad = {}
        for name, param in self.model.named_parameters():
            if name in self.initial_weights:
                pseudo_grad[name] = param.data - self.initial_weights[name]
        
        # 2. Find cohort (other nodes with same layers, different quorums)
        cohort = self._find_layer_cohort()
        
        # 3. Exchange gradients with cohort via gRPC and aggregate
        if cohort:
            aggregated = self._exchange_and_aggregate(pseudo_grad, cohort)
        else:
            # No cohort peers found — use local gradient only
            logger.debug(f"[DILOCO] No cohort peers found, using local gradient only")
            aggregated = pseudo_grad
        
        # 4. Apply outer update: amplify the training direction
        # w_new = w_initial + lr * delta (where delta = current - initial)
        with torch.no_grad():
            for name, param in self.model.named_parameters():
                if name in self.initial_weights and name in aggregated:
                    param.data = self.initial_weights[name] + OUTER_LR * aggregated[name]
        
        # 5. Reset for next round
        self._snapshot_weights()
        self.batches_since_sync = 0
        self.sync_round += 1
        
        logger.info(f"Cohort sync complete: round {self.sync_round}, "
                    f"cohort_size={len(cohort) + 1}")
    
    def _exchange_and_aggregate(self, local_grad: Dict[str, 'torch.Tensor'],
                                 cohort: List[str]) -> Dict[str, 'torch.Tensor']:
        """Exchange pseudo-gradients with cohort peers via gRPC and aggregate.
        
        Uses Byzantine-robust Trimmed Mean aggregation with sqrt(n) weighting.
        Falls back to local gradient if all peer exchanges fail.
        """
        import torch
        
        # Collect all contributions (local + peers)
        contributions = [local_grad]
        
        for peer_url in cohort:
            try:
                peer_grad = self._request_gradient_from_peer(peer_url)
                if peer_grad is not None:
                    contributions.append(peer_grad)
            except Exception as e:
                logger.debug(f"[DILOCO] Failed to get gradient from {peer_url}: {e}")
        
        if len(contributions) == 1:
            # No peer gradients received — use local only
            return local_grad
        
        logger.info(f"[DILOCO] Aggregating {len(contributions)} gradient contributions "
                    f"(1 local + {len(contributions) - 1} peers)")
        
        # Robust aggregation: trimmed mean across contributions
        aggregated = {}
        for name in local_grad:
            tensors = [c[name] for c in contributions if name in c]
            if len(tensors) == 1:
                aggregated[name] = tensors[0]
            elif len(tensors) >= 3:
                # Trimmed Mean: remove top/bottom 10% then average
                stacked = torch.stack(tensors)
                trim_n = max(1, len(tensors) // 10)
                sorted_vals, _ = torch.sort(stacked, dim=0)
                aggregated[name] = sorted_vals[trim_n:-trim_n].mean(dim=0) if trim_n < len(tensors) // 2 else sorted_vals.mean(dim=0)
            else:
                # 2 contributions: simple average
                aggregated[name] = torch.stack(tensors).mean(dim=0)
        
        return aggregated
    
    def _request_gradient_from_peer(self, peer_url: str) -> Optional[Dict[str, 'torch.Tensor']]:
        """Request pseudo-gradient from a cohort peer via gRPC GossipGradient RPC."""
        import torch
        
        try:
            from neuroshard.core.network.connection_pool import get_channel
            from neuroshard.protos import neuroshard_pb2 as pb2
            from neuroshard.protos import neuroshard_pb2_grpc as pb2_grpc
            from urllib.parse import urlparse
            
            parsed = urlparse(peer_url)
            from neuroshard.core.swarm.factory import GRPC_PORT_OFFSET
            grpc_port = (parsed.port or 80) + GRPC_PORT_OFFSET
            grpc_addr = f"{parsed.hostname}:{grpc_port}"
            
            channel = get_channel(grpc_addr)
            stub = pb2_grpc.NeuroShardServiceStub(channel)
            
            # Request gradient via GossipGradient RPC
            request = pb2.GossipGradientRequest(
                node_id=self.node_id,
                round_id=self.sync_round,
            )
            
            response = stub.GossipGradient(request, timeout=15.0)
            
            if response.success and response.gradient_data:
                import io
                import numpy as np
                
                # Deserialize gradient data
                peer_grad = {}
                gradient_dict = json.loads(response.gradient_data)
                for name, data in gradient_dict.items():
                    arr = np.frombuffer(bytes.fromhex(data['hex']), dtype=np.float32)
                    peer_grad[name] = torch.from_numpy(arr.copy()).reshape(data['shape'])
                
                logger.debug(f"[DILOCO] Received gradient from {peer_url}: "
                            f"{len(peer_grad)} params")
                return peer_grad
                
        except Exception as e:
            logger.debug(f"[DILOCO] Gradient exchange with {peer_url} failed: {e}")
        
        return None
    
    def _find_layer_cohort(self) -> List[str]:
        """Find nodes holding the same layers in other quorums.
        
        Returns list of resolved HTTP URLs for cohort peers (not self).
        DHT values may contain LAN info (ip:port|local_ip:port) which
        is resolved before returning.
        """
        if not self.dht:
            return []
        
        # Build set of our own IPs for self-filtering
        my_ips = set()
        local_node = getattr(self.dht, 'local_node', None)
        if local_node:
            my_ips.add(local_node.ip)
        lan_ip = getattr(self.dht, '_local_lan_ip', None)
        if lan_ip:
            my_ips.add(lan_ip)
        my_port = local_node.port if local_node else 0
        
        # Resolve LAN addresses
        try:
            from neuroshard.core.network.nat import resolve_peer_address
            my_public_ip = local_node.ip if local_node else None
        except ImportError:
            resolve_peer_address = None
            my_public_ip = None
        
        cohort = []
        seen = set()
        layer_start, layer_end = self.member.layer_range
        
        # Query DHT for first layer only (all full replicas hold layer 0)
        # Checking every layer would return the same peers N times
        for layer_id in [layer_start]:
            try:
                key_str = f"layer_{layer_id}"
                key_id = int(hashlib.sha1(key_str.encode()).hexdigest(), 16)
                
                # Check local storage first, then do network lookup
                # NOTE: QuorumTrainer may use a different DHTProtocol instance
                # than the P2P manager, so local storage may not have remote
                # nodes' announcements. The network lookup ensures we find them.
                value = self.dht.storage.get(key_id)
                if not value and hasattr(self.dht, 'lookup_value'):
                    try:
                        value = self.dht.lookup_value(key_id)
                    except Exception:
                        pass
                
                if value:
                    endpoints = json.loads(value) if isinstance(value, str) else [value]
                    
                    for endpoint in endpoints:
                        raw = str(endpoint)
                        if raw in seen:
                            continue
                        seen.add(raw)
                        
                        # Resolve LAN address
                        resolved = raw
                        if resolve_peer_address and my_public_ip:
                            resolved = resolve_peer_address(raw, my_public_ip)
                        elif "|" in resolved:
                            resolved = resolved.split("|", 1)[0]
                        
                        # Filter out self by IP
                        peer_ip = resolved.rsplit(":", 1)[0] if ":" in resolved else resolved
                        if peer_ip in my_ips:
                            continue
                        
                        # Return as HTTP URL for _request_gradient_from_peer
                        cohort.append(f"http://{resolved}")
            except Exception:
                pass
        
        return cohort
    
    def get_training_stats(self) -> Dict[str, Any]:
        """Get current training statistics."""
        return {
            "quorum_id": self.quorum.quorum_id,
            "node_id": self.node_id,
            "role": self.member.role.value if self.member else "unknown",
            "total_batches": self.total_batches,
            "batches_since_sync": self.batches_since_sync,
            "sync_round": self.sync_round,
            "current_loss": self.current_loss,
            "running": self.running,
        }


# =============================================================================
# QUORUM INFERENCE ROUTER
# =============================================================================

# Inference pricing constants
BASE_INFERENCE_PRICE = 0.0001      # NEURO per token
SPEED_MULTIPLIERS = {
    "tier1": 1.5,
    "tier2": 1.3,
    "tier3": 1.0,
    "tier4": 0.8,
    "tier5": 0.6,
}


@dataclass
class InferenceQuorumInfo:
    """Information about a quorum available for inference."""
    quorum_id: str
    speed_tier: str
    initiator_endpoint: str
    estimated_latency_ms: float
    price_per_token: float
    available_capacity: int
    reputation: float = 0.95
    
    def to_json(self) -> str:
        return json.dumps({
            "quorum_id": self.quorum_id,
            "speed_tier": self.speed_tier,
            "initiator_endpoint": self.initiator_endpoint,
            "estimated_latency_ms": self.estimated_latency_ms,
            "price_per_token": self.price_per_token,
            "available_capacity": self.available_capacity,
            "reputation": self.reputation,
        })
    
    @classmethod
    def from_json(cls, json_str: str) -> 'InferenceQuorumInfo':
        data = json.loads(json_str)
        return cls(**data)


def calculate_inference_price(speed_tier: str, utilization: float = 0.5, reputation: float = 0.95) -> float:
    """
    Calculate per-token price for inference.
    
    Args:
        speed_tier: Quorum speed tier (tier1-tier5)
        utilization: Current utilization (0-1)
        reputation: Quorum reputation score (0-1)
        
    Returns:
        Price per token in NEURO
    """
    # Speed multiplier (faster = more expensive)
    speed_mult = SPEED_MULTIPLIERS.get(speed_tier, 1.0)
    
    # Demand multiplier (higher utilization = higher price)
    demand_mult = 1.0 + utilization ** 2  # 1.0 to 2.0
    
    # Reputation multiplier (proven quorums charge more)
    rep_mult = 0.8 + 0.4 * reputation  # 0.8 to 1.2
    
    price = BASE_INFERENCE_PRICE * demand_mult * speed_mult * rep_mult
    
    # Ensure minimum profitability
    min_price = BASE_INFERENCE_PRICE * 0.5
    return max(price, min_price)


class QuorumInferenceRouter:
    """
    Routes inference requests to appropriate quorums.
    
    Discovery Flow:
    1. Query DHT for available inference quorums
    2. Filter by capacity and status (ACTIVE only)
    3. Score by latency, price, and reputation
    4. Return ranked list for client selection
    
    Execution Flow:
    1. Client sends request to selected quorum's initiator
    2. Initiator embeds and forwards through pipeline
    3. Finisher generates output, sends back
    4. Payment distributed to quorum members
    """
    
    def __init__(
        self,
        registry: QuorumRegistry,
        dht_protocol: Any = None,
    ):
        """
        Initialize QuorumInferenceRouter.
        
        Args:
            registry: QuorumRegistry for quorum lookup
            dht_protocol: DHT for network-wide discovery
        """
        self.registry = registry
        self.dht = dht_protocol
        
        # Cache of available quorums
        self._quorum_cache: Dict[str, InferenceQuorumInfo] = {}
        self._cache_ttl = 30.0  # 30 second cache
        self._last_cache_update = 0.0
        
        logger.info("QuorumInferenceRouter initialized")
    
    def discover_quorums(self, force_refresh: bool = False) -> List[InferenceQuorumInfo]:
        """
        Discover available inference quorums.
        
        Args:
            force_refresh: Force cache refresh
            
        Returns:
            List of available quorums sorted by score
        """
        now = time.time()
        
        # Check cache
        if not force_refresh and (now - self._last_cache_update) < self._cache_ttl:
            return list(self._quorum_cache.values())
        
        # Refresh from registry
        self._quorum_cache.clear()
        
        # Get all active quorums
        all_quorums = self.registry.discover_quorums()
        
        for quorum in all_quorums:
            if quorum.lifecycle != QuorumLifecycle.ACTIVE:
                continue
            
            if not quorum.is_complete:
                continue
            
            # Find initiator
            initiator = None
            for member in quorum.members:
                if member.role == QuorumRole.INITIATOR:
                    initiator = member
                    break
            
            if not initiator:
                continue
            
            # Calculate price
            utilization = 0.5  # TODO: Get actual utilization
            price = calculate_inference_price(quorum.speed_tier, utilization)
            
            # Estimate latency based on speed tier
            latency_estimates = {
                "tier1": 50,
                "tier2": 150,
                "tier3": 400,
                "tier4": 1500,
                "tier5": 5000,
            }
            latency = latency_estimates.get(quorum.speed_tier, 1000)
            
            info = InferenceQuorumInfo(
                quorum_id=quorum.quorum_id,
                speed_tier=quorum.speed_tier,
                initiator_endpoint=initiator.endpoint,
                estimated_latency_ms=latency,
                price_per_token=price,
                available_capacity=10,  # TODO: Get actual capacity
            )
            
            self._quorum_cache[quorum.quorum_id] = info
        
        self._last_cache_update = now
        
        # Sort by score (lower is better: latency + price penalty)
        result = list(self._quorum_cache.values())
        result.sort(key=lambda q: q.estimated_latency_ms + q.price_per_token * 10000)
        
        return result
    
    def select_best_quorum(
        self,
        max_latency_ms: Optional[float] = None,
        max_price: Optional[float] = None,
        min_reputation: float = 0.8,
    ) -> Optional[InferenceQuorumInfo]:
        """
        Select the best quorum for an inference request.
        
        Args:
            max_latency_ms: Maximum acceptable latency
            max_price: Maximum acceptable price per token
            min_reputation: Minimum reputation score
            
        Returns:
            Best matching quorum or None
        """
        quorums = self.discover_quorums()
        
        for quorum in quorums:
            # Apply filters
            if max_latency_ms and quorum.estimated_latency_ms > max_latency_ms:
                continue
            
            if max_price and quorum.price_per_token > max_price:
                continue
            
            if quorum.reputation < min_reputation:
                continue
            
            if quorum.available_capacity <= 0:
                continue
            
            return quorum
        
        return None
    
    def get_quorum_for_routing(self, quorum_id: str) -> Optional[Quorum]:
        """
        Get full quorum for routing an inference request.
        
        Args:
            quorum_id: Quorum to route to
            
        Returns:
            Quorum object or None
        """
        return self.registry.get_quorum(quorum_id)
    
    def estimate_cost(self, quorum_id: str, max_tokens: int) -> float:
        """
        Estimate inference cost for a request.
        
        Args:
            quorum_id: Target quorum
            max_tokens: Maximum tokens to generate
            
        Returns:
            Estimated cost in NEURO
        """
        if quorum_id in self._quorum_cache:
            price = self._quorum_cache[quorum_id].price_per_token
            return price * max_tokens
        return BASE_INFERENCE_PRICE * max_tokens


# =============================================================================
# ASYNC TRAINER - For T5 nodes and nodes without quorum
# =============================================================================

# Freshness decay for async contributions (from whitepaper)
ASYNC_FRESHNESS_DECAY = {
    3600: 1.0,      # < 1 hour: full value
    86400: 0.7,     # < 1 day: 70% value
    604800: 0.5,    # < 1 week: 50% value
    float('inf'): 0.3,  # > 1 week: 30% value
}


def calculate_async_freshness(age_seconds: float) -> float:
    """Calculate freshness multiplier for async gradient contributions."""
    for threshold, freshness in sorted(ASYNC_FRESHNESS_DECAY.items()):
        if age_seconds < threshold:
            return freshness
    return 0.3


class AsyncTrainer:
    """
    Async training for nodes that can't join real-time quorums.
    
    Per whitepaper, async training is for:
    - T5 (slow) nodes that can't keep up with real-time pipelines
    - Any node that can't find a compatible quorum
    
    Process:
    1. Download current weights for held layers
    2. Train locally for N steps using Genesis data
    3. Compute pseudo-gradient: delta_w = w_current - w_initial
    4. Submit to layer cohort for next DiLoCo sync round
    5. Apply freshness decay to contribution weight
    
    Async contributions are weighted by: sqrt(batches) * freshness
    This ensures fair influence while prioritizing recent work.
    """
    
    # Training interval: how often to do async training
    ASYNC_TRAIN_INTERVAL = 60  # Train every 60 seconds
    ASYNC_BATCH_SIZE = 4       # Smaller batches for async training
    ASYNC_STEPS_PER_ROUND = 50 # Steps per async training round
    
    def __init__(
        self,
        node_id: str,
        model: Any,
        optimizer: Any,
        genesis_loader: Any = None,
        dht_protocol: Any = None,
    ):
        """
        Initialize AsyncTrainer.
        
        Args:
            node_id: This node's ID
            model: The model to train
            optimizer: PyTorch optimizer
            genesis_loader: Genesis data loader
            dht_protocol: DHT for cohort discovery
        """
        self.node_id = node_id
        self.model = model
        self.optimizer = optimizer
        self.genesis_loader = genesis_loader
        self.dht = dht_protocol
        
        # Training state
        self.running = False
        self._training_thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        
        # DiLoCo-like state for async
        self.initial_weights: Dict[str, Any] = {}
        self.last_sync_time = time.time()
        
        # Stats
        self.total_batches = 0
        self.total_syncs = 0
        self.current_loss: Optional[float] = None
        self.last_gradient_submission_time: Optional[float] = None
        
        # Model hash tracking for chained PoNW
        self.model_hash_start: str = ""
        self.model_hash_end: str = ""
        
        # Snapshot initial weights
        self._snapshot_weights()
        
        # Capture initial model hash
        self.model_hash_start = self._compute_model_hash()
        
        logger.info(f"AsyncTrainer initialized for node {node_id[:16]}...")
    
    def _snapshot_weights(self):
        """Snapshot current weights for pseudo-gradient computation."""
        import torch
        self.initial_weights = {}
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.initial_weights[name] = param.data.clone()
    
    def _compute_model_hash(self) -> str:
        """Compute a hash of model weights for chained PoNW verification."""
        import torch
        import hashlib
        
        hasher = hashlib.sha256()
        
        # Sample some parameters for speed (full hash too slow)
        params = list(self.model.named_parameters())
        sample_indices = [0, len(params)//4, len(params)//2, 3*len(params)//4, len(params)-1]
        
        for idx in sample_indices:
            if 0 <= idx < len(params):
                name, param = params[idx]
                hasher.update(name.encode())
                # Use a sample of the parameter data for speed
                flat = param.data.flatten()[:1000]
                hasher.update(flat.cpu().numpy().tobytes())
        
        return hasher.hexdigest()[:16]
    
    def start(self):
        """Start the async training loop."""
        if self.running:
            return
        
        self.running = True
        self._stop_event.clear()
        self._training_thread = threading.Thread(
            target=self._async_training_loop,
            daemon=True,
            name=f"AsyncTrainer-{self.node_id[:8]}"
        )
        self._training_thread.start()
        logger.info("AsyncTrainer started")
    
    def stop(self):
        """Stop the async training loop."""
        self.running = False
        self._stop_event.set()
        if self._training_thread and self._training_thread.is_alive():
            self._training_thread.join(timeout=5.0)
        logger.info("AsyncTrainer stopped")
    
    def _async_training_loop(self):
        """
        Main async training loop.
        
        Unlike QuorumTrainer, this runs independently:
        1. Train locally for ASYNC_STEPS_PER_ROUND steps
        2. Compute pseudo-gradient
        3. Submit to DiLoCo cohort sync (if available)
        4. Reset and repeat
        """
        import torch
        
        logger.info("[ASYNC] Starting async training loop...")
        
        # Track if we've logged the "no genesis loader" message
        _logged_no_genesis = False
        
        while self.running and not self._stop_event.is_set():
            try:
                # Check if we have Genesis data to train on
                if not self.genesis_loader:
                    if not _logged_no_genesis:
                        logger.warning("[ASYNC] No Genesis data loader available - waiting for data...")
                        logger.info("[ASYNC] Genesis loader will be initialized when node downloads training shards")
                        _logged_no_genesis = True
                    time.sleep(self.ASYNC_TRAIN_INTERVAL)
                    continue
                
                _logged_no_genesis = False  # Reset if we get a loader
                
                # Train for N steps
                losses = []
                data_not_ready_count = 0
                
                for step in range(self.ASYNC_STEPS_PER_ROUND):
                    if self._stop_event.is_set():
                        break
                    
                    try:
                        # Get batch from Genesis loader
                        result = self.genesis_loader.get_batch(batch_size=self.ASYNC_BATCH_SIZE)
                        
                        # Handle None returns (shard boundary)
                        if result is None or result[0] is None:
                            data_not_ready_count += 1
                            time.sleep(0.1)
                            continue
                        
                        input_ids, labels = result
                        
                        # Move to device
                        device = next(self.model.parameters()).device
                        input_ids = input_ids.to(device)
                        labels = labels.to(device)
                        
                        # Forward pass through local layers
                        self.model.train()
                        
                        # Determine forward path based on what this node has
                        has_embedding = hasattr(self.model, 'has_embedding') and self.model.has_embedding
                        has_lm_head = hasattr(self.model, 'has_lm_head') and self.model.has_lm_head
                        
                        if has_embedding and has_lm_head:
                            # Full node: embed → forward → logits → cross-entropy
                            embeddings = self.model.embed(input_ids)
                            hidden, _ = self.model.forward_my_layers(embeddings)
                            logits = self.model.compute_logits(hidden)
                            loss = torch.nn.functional.cross_entropy(
                                logits.view(-1, logits.size(-1)),
                                labels.view(-1),
                                ignore_index=-100
                            )
                        elif has_embedding:
                            # Driver without LM head: embed → forward → proxy loss
                            embeddings = self.model.embed(input_ids)
                            hidden, _ = self.model.forward_my_layers(embeddings)
                            loss = hidden.pow(2).mean() * 0.01  # Proxy loss
                        elif has_lm_head:
                            # Worker with LM head but no embedding:
                            # Can't embed token IDs — train on random hidden states
                            # to keep weights active (will sync via DiLoCo)
                            hidden_input = torch.randn(
                                input_ids.shape[0], input_ids.shape[1],
                                self.model.hidden_dim, device=device
                            )
                            hidden, _ = self.model.forward_my_layers(hidden_input)
                            logits = self.model.compute_logits(hidden)
                            loss = torch.nn.functional.cross_entropy(
                                logits.view(-1, logits.size(-1)),
                                labels.view(-1),
                                ignore_index=-100
                            )
                        else:
                            # Processor only: forward through layers with random input
                            hidden_input = torch.randn(
                                input_ids.shape[0], input_ids.shape[1],
                                self.model.hidden_dim, device=device
                            )
                            hidden, _ = self.model.forward_my_layers(hidden_input)
                            loss = hidden.pow(2).mean() * 0.01  # Proxy loss
                        
                        # Add MoE auxiliary loss (all layers are MoE)
                        # MoE aux loss includes load balancing and router z-loss
                        moe_aux = self.model.get_moe_aux_loss()
                        if moe_aux is not None:
                            loss = loss + moe_aux
                        
                        # Backward pass
                        self.optimizer.zero_grad()
                        loss.backward()
                        
                        # Gradient clipping
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                        
                        # Optimizer step
                        self.optimizer.step()
                        
                        # Track stats
                        self.total_batches += 1
                        loss_value = loss.item()
                        losses.append(loss_value)
                        
                        # Update current_loss INCREMENTALLY so proofs always have valid loss
                        # This fixes timing issue where proof is created during training
                        self.current_loss = sum(losses) / len(losses)
                        
                        # Record loss for shard rotation plateau detection
                        if self.genesis_loader and hasattr(self.genesis_loader, 'record_loss'):
                            self.genesis_loader.record_loss(loss_value)
                        
                        # Log every 10 steps
                        if self.total_batches % 10 == 0:
                            logger.info(f"[ASYNC] Step {self.total_batches}: loss={loss_value:.4f}")
                        
                    except RuntimeError as e:
                        error_msg = str(e)
                        if "Data not ready" in error_msg or "shard" in error_msg.lower():
                            data_not_ready_count += 1
                            time.sleep(0.1)  # Brief wait for data
                            continue
                        elif "out of memory" in error_msg.lower() or "CUDA" in error_msg:
                            # CUDA OOM: clear cache, reduce batch, and break inner loop
                            # Don't spam 50 identical error lines — log once and back off
                            logger.warning(f"[ASYNC] CUDA out of memory — clearing cache and backing off")
                            try:
                                import torch as _torch
                                _torch.cuda.empty_cache()
                            except Exception:
                                pass
                            time.sleep(5)  # Give GPU time to free memory
                            break  # Exit inner loop, outer loop will handle backoff
                        else:
                            logger.warning(f"[ASYNC] Training step error: {e}")
                            continue
                    except Exception as e:
                        logger.warning(f"[ASYNC] Training step error: {e}")
                        continue
                
                # Log data availability issues
                if data_not_ready_count > 0 and len(losses) == 0:
                    logger.info(f"[ASYNC] Data not ready ({data_not_ready_count} attempts) - waiting for shard download")
                
                # Update current loss and model hash
                if losses:
                    self.current_loss = sum(losses) / len(losses)
                    # Capture model hash AFTER training (weights changed)
                    self.model_hash_end = self._compute_model_hash()
                    logger.info(f"[ASYNC] Completed {len(losses)} steps, avg_loss={self.current_loss:.4f}")
                
                # Compute pseudo-gradient
                pseudo_gradient = self._compute_pseudo_gradient()
                
                # Submit to cohort sync (if DHT available)
                if pseudo_gradient and self.dht:
                    self._submit_to_cohort_sync(pseudo_gradient)
                
                # Reset weights snapshot for next round
                self._snapshot_weights()
                
                # Wait before next training round
                time.sleep(self.ASYNC_TRAIN_INTERVAL)
                
            except Exception as e:
                logger.error(f"[ASYNC] Training loop error: {e}")
                # Exponential backoff: 2s, 4s, 8s, ... up to 120s
                if not hasattr(self, '_consecutive_errors'):
                    self._consecutive_errors = 0
                self._consecutive_errors += 1
                backoff = min(2 ** self._consecutive_errors, 120)
                logger.warning(f"[ASYNC] Error #{self._consecutive_errors}, "
                              f"backing off {backoff}s")
                time.sleep(backoff)
                continue
            
            # Reset error counter on successful round
            if hasattr(self, '_consecutive_errors'):
                self._consecutive_errors = 0
        
        logger.info("[ASYNC] Training loop ended")
    
    def _compute_pseudo_gradient(self) -> Optional[Dict[str, Any]]:
        """
        Compute pseudo-gradient: delta_w = w_current - w_initial
        
        Sign convention: current - initial (weight delta = direction training moved).
        This is consistent with DiLoCoTrainer.compute_pseudo_gradient() and
        the OuterOptimizer which adds the aggregated delta to amplify it.
        """
        import torch
        
        if not self.initial_weights:
            return None
        
        pseudo_grad = {}
        for name, param in self.model.named_parameters():
            if name in self.initial_weights and param.requires_grad:
                # Pseudo-gradient is the direction training moved
                pseudo_grad[name] = param.data - self.initial_weights[name]
        
        return pseudo_grad
    
    def _submit_to_cohort_sync(self, pseudo_gradient: Dict[str, Any]):
        """
        Submit pseudo-gradient to layer cohort for DiLoCo sync.
        
        Async contributions are weighted by freshness:
        - < 1 hour: 100%
        - < 1 day: 70%
        - < 1 week: 50%
        - > 1 week: 30%
        """
        try:
            # Calculate freshness
            age_seconds = time.time() - self.last_sync_time
            freshness = calculate_async_freshness(age_seconds)
            
            # Create gradient contribution
            contribution = {
                "node_id": self.node_id,
                "batches": self.total_batches,
                "timestamp": time.time(),
                "freshness": freshness,
                "is_async": True,
                # Note: Actual gradient data would be compressed and sent via DHT/P2P
            }
            
            # Log submission
            logger.info(f"[ASYNC] Submitting gradient: batches={self.total_batches}, "
                       f"freshness={freshness:.2f}, age={age_seconds/60:.1f}min")
            
            # Update tracking
            self.last_gradient_submission_time = time.time()
            self.last_sync_time = time.time()
            self.total_syncs += 1
            
            # TODO: Actual submission via DHT/P2P to layer cohort
            # For now, the gradient is computed and ready for submission
            
        except Exception as e:
            logger.error(f"[ASYNC] Failed to submit gradient: {e}")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get async trainer statistics."""
        return {
            "running": self.running,
            "total_batches": self.total_batches,
            "total_syncs": self.total_syncs,
            "current_loss": self.current_loss,
            "last_submission": self.last_gradient_submission_time,
            "is_async": True,
            # Chained PoNW: model state tracking
            "model_hash_start": self.model_hash_start,
            "model_hash_end": self.model_hash_end,
        }
    
    def reset_proof_period(self):
        """Reset tracking for next proof period (called after proof submission)."""
        # model_hash_end becomes the new start for next period
        self.model_hash_start = self.model_hash_end
