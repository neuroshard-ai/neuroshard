"""
Swarm Router - Fault-Tolerant Multipath Routing for NeuroShard

Implements the "Swarm" network layer with:
- K-candidate routing (multiple peers per layer range)
- Automatic failover within 200ms
- Weighted scoring (latency + queue_depth)
- Probabilistic send with retry

Key Directive: "If Node A hangs, the packet must automatically flow to Node B 
without crashing the run."

Reference: SWARM Parallelism papers for randomized pipeline construction.
"""

import asyncio
import grpc
import logging
import random
import time
import torch
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
from collections import defaultdict

# Try importing proto definitions
try:
    from neuroshard.protos import neuroshard_pb2
    from neuroshard.protos import neuroshard_pb2_grpc
    GRPC_AVAILABLE = True
except ImportError:
    GRPC_AVAILABLE = False

logger = logging.getLogger(__name__)


@dataclass
class PeerCandidate:
    """
    A candidate peer for routing.
    
    Candidates are scored by a weighted combination of latency and queue depth.
    Lower score = better candidate.
    """
    node_id: str
    grpc_addr: str
    layer_range: Tuple[int, int]  # (start_layer, end_layer) exclusive
    
    # Performance metrics (rolling averages)
    latency_ms: float = 100.0             # Default 100ms
    queue_depth: int = 0                   # Current work queue size
    
    # Health tracking
    last_heartbeat: float = field(default_factory=time.time)
    consecutive_failures: int = 0
    total_requests: int = 0
    successful_requests: int = 0
    
    # Capacity info from heartbeat
    available_memory_mb: int = 0
    gpu_utilization: float = 0.0
    is_accepting_activations: bool = True
    
    def score(self, weights: Optional[Dict[str, float]] = None) -> float:
        """
        Compute weighted score for peer selection.
        
        Lower score = better candidate.
        
        Default weights prioritize queue depth (60%) over latency (40%)
        because queue depth is a better indicator of immediate availability.
        """
        w = weights or {"latency": 0.4, "queue": 0.6}
        
        # Normalize: latency 0-500ms maps to 0-1, queue 0-100 maps to 0-1
        latency_norm = min(1.0, self.latency_ms / 500.0)
        queue_norm = min(1.0, self.queue_depth / 100.0)
        
        base_score = w["latency"] * latency_norm + w["queue"] * queue_norm
        
        # Penalty for recent failures (exponential backoff)
        if self.consecutive_failures > 0:
            failure_penalty = min(1.0, 0.1 * (2 ** self.consecutive_failures))
            base_score += failure_penalty
            
        # Penalty for stale heartbeat (>15s)
        staleness = time.time() - self.last_heartbeat
        if staleness > 15.0:
            stale_penalty = min(1.0, staleness / 60.0)
            base_score += stale_penalty
            
        return base_score
    
    @property
    def success_rate(self) -> float:
        """Success rate of requests to this peer."""
        if self.total_requests == 0:
            return 1.0  # Assume good until proven otherwise
        return self.successful_requests / self.total_requests
    
    def covers_layer(self, layer: int) -> bool:
        """Check if this peer handles the given layer."""
        return self.layer_range[0] <= layer < self.layer_range[1]
    
    def update_latency(self, latency_ms: float, alpha: float = 0.3):
        """Update latency with exponential moving average."""
        self.latency_ms = alpha * latency_ms + (1 - alpha) * self.latency_ms
        
    def record_success(self, latency_ms: float):
        """Record a successful request."""
        self.total_requests += 1
        self.successful_requests += 1
        self.consecutive_failures = 0
        self.update_latency(latency_ms)
        
    def record_failure(self):
        """Record a failed request."""
        self.total_requests += 1
        self.consecutive_failures += 1


@dataclass
class RoutingResult:
    """Result of a routing operation."""
    success: bool
    peer_used: Optional[str] = None
    latency_ms: float = 0.0
    attempts: int = 0
    error: Optional[str] = None
    response_data: Any = None


class SwarmRouter:
    """
    Fault-tolerant multipath router for distributed inference/training.
    
    Key Features:
    - Returns K candidates per layer range (not just one)
    - Automatic failover: if primary doesn't ACK in 200ms, try secondary
    - Weighted scoring: 0.4 * latency + 0.6 * queue_depth
    - Probabilistic load balancing for equal-scored candidates
    
    Integration:
    - Replaces single-hop routing in p2p.py
    - Works with SwarmHeartbeatService for capacity awareness
    - Uses DHT for decentralized peer discovery 
    """
    
    ACK_TIMEOUT_MS = 200       # Failover if no ACK in 200ms
    K_CANDIDATES = 3           # Return top-K candidates per layer range
    CACHE_TTL_SECONDS = 30     # How long to cache peer info
    
    def __init__(
        self,
        dht_protocol: Optional[Any] = None,
        layer_pool: Optional[Any] = None,
    ):
        """
        Initialize SwarmRouter.
        
        Args:
            dht_protocol: DHT protocol for decentralized peer discovery
            layer_pool: DynamicLayerPool for local layer info
        """
        self.dht = dht_protocol
        self.layer_pool = layer_pool
        
        # Peer state cache
        self.peer_stats: Dict[str, PeerCandidate] = {}
        self._peer_lock = asyncio.Lock()
        
        # gRPC connection pool
        self._channels: Dict[str, grpc.aio.Channel] = {}
        self._stubs: Dict[str, Any] = {}
        
        # Routing metrics
        self.total_routes = 0
        self.successful_routes = 0
        self.failover_count = 0
        
        # Background tasks
        self._cleanup_task: Optional[asyncio.Task] = None
    
    # Property aliases for compatibility with SwarmServiceMixin
    @property
    def total_sends(self) -> int:
        """Alias for total_routes."""
        return self.total_routes
    
    @property
    def successful_sends(self) -> int:
        """Alias for successful_routes."""
        return self.successful_routes
        
    async def start(self):
        """Start background maintenance tasks."""
        self._cleanup_task = asyncio.create_task(self._cleanup_loop())
        logger.info("SwarmRouter started")
        
    async def stop(self):
        """Stop router and cleanup."""
        if self._cleanup_task:
            self._cleanup_task.cancel()
            try:
                await self._cleanup_task
            except asyncio.CancelledError:
                pass
                
        # Close all gRPC channels
        for channel in self._channels.values():
            await channel.close()
        self._channels.clear()
        self._stubs.clear()
        
        logger.info("SwarmRouter stopped")
        
    async def _cleanup_loop(self):
        """Periodically cleanup stale peers and connections."""
        while True:
            try:
                await asyncio.sleep(60)  # Every minute
                await self._cleanup_stale_peers()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Cleanup error: {e}")
                
    async def _cleanup_stale_peers(self):
        """Remove peers that haven't sent heartbeat in >60s."""
        now = time.time()
        stale_threshold = 60.0
        
        async with self._peer_lock:
            stale_peers = [
                node_id for node_id, peer in self.peer_stats.items()
                if (now - peer.last_heartbeat) > stale_threshold
            ]
            
            for node_id in stale_peers:
                del self.peer_stats[node_id]
                if node_id in self._channels:
                    await self._channels[node_id].close()
                    del self._channels[node_id]
                if node_id in self._stubs:
                    del self._stubs[node_id]
                    
            if stale_peers:
                logger.info(f"Cleaned up {len(stale_peers)} stale peers")
    
    def register_peer(self, peer: PeerCandidate):
        """
        Register or update a peer from heartbeat.
        
        Called by SwarmHeartbeatService when receiving capacity broadcasts.
        """
        self.peer_stats[peer.node_id] = peer
        
    def update_peer_from_heartbeat(
        self,
        node_id: str,
        grpc_addr: str,
        layer_range: Tuple[int, int],
        queue_depth: int,
        available_memory_mb: int,
        gpu_utilization: float,
        is_accepting: bool,
    ):
        """Update peer info from heartbeat message."""
        if node_id in self.peer_stats:
            peer = self.peer_stats[node_id]
            peer.queue_depth = queue_depth
            peer.available_memory_mb = available_memory_mb
            peer.gpu_utilization = gpu_utilization
            peer.is_accepting_activations = is_accepting
            peer.last_heartbeat = time.time()
        else:
            # New peer
            self.peer_stats[node_id] = PeerCandidate(
                node_id=node_id,
                grpc_addr=grpc_addr,
                layer_range=layer_range,
                queue_depth=queue_depth,
                available_memory_mb=available_memory_mb,
                gpu_utilization=gpu_utilization,
                is_accepting_activations=is_accepting,
                last_heartbeat=time.time(),
            )
    
    def get_candidates(self, target_layer: int) -> List[PeerCandidate]:
        """
        Get K candidates for a target layer, sorted by score.
        
        Returns candidates from:
        1. Local cache (fastest) - peers we know from heartbeats
        2. DHT lookup (if cache miss) - decentralized peer discovery
        
        Args:
            target_layer: The layer index we need to route to
            
        Returns:
            List of up to K candidates, sorted by score (best first)
        """
        candidates: List[PeerCandidate] = []
        
        # Strategy 1: Local cache (from heartbeats)
        for node_id, info in self.peer_stats.items():
            if info.covers_layer(target_layer) and info.is_accepting_activations:
                candidates.append(info)
        
        # Strategy 2: DHT lookup (if not enough candidates)
        if len(candidates) < self.K_CANDIDATES and self.dht:
            dht_peers = self._dht_lookup_layer(target_layer)
            for peer in dht_peers:
                if peer.node_id not in {c.node_id for c in candidates}:
                    candidates.append(peer)
        
        # Sort by score, return top K
        candidates.sort(key=lambda c: c.score())
        
        # Add small random shuffle among similarly-scored candidates
        # to distribute load
        if len(candidates) > 1:
            candidates = self._probabilistic_shuffle(candidates)
            
        return candidates[:self.K_CANDIDATES]
    
    def _probabilistic_shuffle(
        self, 
        candidates: List[PeerCandidate], 
        epsilon: float = 0.1
    ) -> List[PeerCandidate]:
        """
        Add small random shuffle among similarly-scored candidates.
        
        Candidates with scores within epsilon of each other may be swapped.
        This provides load balancing without completely ignoring scores.
        """
        if len(candidates) < 2:
            return candidates
            
        result = candidates.copy()
        
        for i in range(len(result) - 1):
            # Check if next candidate has similar score
            score_diff = abs(result[i].score() - result[i + 1].score())
            if score_diff < epsilon:
                # 30% chance to swap similar candidates
                if random.random() < 0.3:
                    result[i], result[i + 1] = result[i + 1], result[i]
                    
        return result
    
    def _dht_lookup_layer(self, target_layer: int) -> List[PeerCandidate]:
        """Look up peers for a layer via DHT."""
        if not self.dht:
            return []
            
        try:
            # DHT key format: "layer:{layer_num}"
            key = f"layer:{target_layer}"
            values = self.dht.lookup_values(key, k=self.K_CANDIDATES * 2)
            
            candidates = []
            for value in values:
                # Parse peer info from DHT value
                try:
                    import json
                    info = json.loads(value)
                    peer = PeerCandidate(
                        node_id=info.get("node_id", ""),
                        grpc_addr=info.get("grpc_addr", ""),
                        layer_range=(info.get("start_layer", 0), info.get("end_layer", 0)),
                    )
                    candidates.append(peer)
                except:
                    continue
                    
            return candidates
        except Exception as e:
            logger.debug(f"DHT lookup failed: {e}")
            return []
            
    async def _get_channel(self, grpc_addr: str) -> grpc.aio.Channel:
        """Get or create gRPC channel for address."""
        if grpc_addr not in self._channels:
            # Create new channel for P2P network
            # Fast keepalive to detect dead nodes quickly in decentralized network
            options = [
                ('grpc.keepalive_time_ms', 30000),  # Ping every 30 seconds
                ('grpc.keepalive_timeout_ms', 10000),  # 10 second timeout
                ('grpc.keepalive_permit_without_calls', True),  # Ping even when idle
                ('grpc.http2.max_pings_without_data', 0),  # Unlimited pings
            ]
            
            # Handle different address formats
            if grpc_addr.startswith("http://"):
                addr = grpc_addr.replace("http://", "")
            elif grpc_addr.startswith("https://"):
                addr = grpc_addr.replace("https://", "")
            else:
                addr = grpc_addr
                
            self._channels[grpc_addr] = grpc.aio.insecure_channel(addr, options=options)
            
        return self._channels[grpc_addr]
    
    async def _get_stub(self, grpc_addr: str) -> Any:
        """Get or create gRPC stub for address."""
        if not GRPC_AVAILABLE:
            raise RuntimeError("gRPC not available")
            
        if grpc_addr not in self._stubs:
            channel = await self._get_channel(grpc_addr)
            self._stubs[grpc_addr] = neuroshard_pb2_grpc.NeuroShardServiceStub(channel)
            
        return self._stubs[grpc_addr]
    
    async def send_with_failover(
        self,
        tensor: torch.Tensor,
        target_layer: int,
        session_id: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> RoutingResult:
        """
        Send tensor to target layer with automatic failover.
        
        Algorithm:
        1. Get K candidates sorted by score
        2. Try primary candidate
        3. If no ACK in 200ms, try secondary
        4. Continue until success or all candidates exhausted
        
        Args:
            tensor: Activation tensor to send
            target_layer: Target layer index
            session_id: Session identifier for routing
            metadata: Optional metadata to include
            
        Returns:
            RoutingResult with success status and response data
        """
        self.total_routes += 1
        candidates = self.get_candidates(target_layer)
        
        if not candidates:
            logger.error(f"No candidates available for layer {target_layer}")
            return RoutingResult(
                success=False,
                error=f"No candidates for layer {target_layer}",
                attempts=0,
            )
        
        last_error: Optional[Exception] = None
        
        for i, candidate in enumerate(candidates):
            start_time = time.time()
            
            try:
                # Attempt send with timeout
                result = await asyncio.wait_for(
                    self._send_to_peer(candidate, tensor, session_id, metadata),
                    timeout=self.ACK_TIMEOUT_MS / 1000.0
                )
                
                latency_ms = (time.time() - start_time) * 1000
                
                # Record success
                candidate.record_success(latency_ms)
                self.successful_routes += 1
                
                if i > 0:
                    self.failover_count += 1
                    logger.info(f"Failover to candidate {i}: {candidate.node_id[:8]}...")
                
                return RoutingResult(
                    success=True,
                    peer_used=candidate.node_id,
                    latency_ms=latency_ms,
                    attempts=i + 1,
                    response_data=result,
                )
                
            except asyncio.TimeoutError:
                candidate.record_failure()
                logger.warning(
                    f"Peer {candidate.node_id[:8]}... timed out after {self.ACK_TIMEOUT_MS}ms, "
                    f"trying next ({i+1}/{len(candidates)})"
                )
                last_error = TimeoutError(f"Peer {candidate.node_id[:8]} timed out")
                continue
                
            except grpc.aio.AioRpcError as e:
                candidate.record_failure()
                logger.warning(
                    f"Peer {candidate.node_id[:8]}... gRPC error: {e.code()}, "
                    f"trying next ({i+1}/{len(candidates)})"
                )
                last_error = e
                continue
                
            except Exception as e:
                candidate.record_failure()
                logger.warning(
                    f"Peer {candidate.node_id[:8]}... failed: {e}, "
                    f"trying next ({i+1}/{len(candidates)})"
                )
                last_error = e
                continue
        
        # All candidates failed
        error_msg = f"All {len(candidates)} candidates failed for layer {target_layer}"
        if last_error:
            error_msg += f": {last_error}"
            
        logger.error(error_msg)
        
        return RoutingResult(
            success=False,
            error=error_msg,
            attempts=len(candidates),
        )
    
    async def _send_to_peer(
        self,
        candidate: PeerCandidate,
        tensor: torch.Tensor,
        session_id: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Any:
        """
        Send activation tensor to a specific peer.
        
        Uses gRPC SwarmForward RPC (or falls back to existing Forward).
        """
        if not GRPC_AVAILABLE:
            raise RuntimeError("gRPC not available")
            
        stub = await self._get_stub(candidate.grpc_addr)
        
        # Serialize tensor
        tensor_bytes = tensor.cpu().numpy().tobytes()
        shape = list(tensor.shape)
        dtype_str = str(tensor.dtype).replace("torch.", "")
        
        # Build request - use SwarmForward if available
        try:
            # Try SwarmForward (new swarm-aware RPC)
            request = neuroshard_pb2.SwarmForwardRequest(
                session_id=session_id,
                request_id=f"swarm_{time.time()}_{session_id[:8]}",
                hidden_states=tensor_bytes,
                hidden_shape=shape,
                # tensor_dtype inferred from shape/context
                target_layer=candidate.layer_range[0],
                sender_url=metadata.get("source_node", "") if metadata else "",
                micro_batch_id=metadata.get("micro_batch_id", 0) if metadata else 0,
                priority=metadata.get("priority", 10) if metadata else 10,
            )
            
            response = await stub.SwarmForward(request)
            return response
            
        except grpc.aio.AioRpcError as e:
            if e.code() == grpc.StatusCode.UNIMPLEMENTED:
                logger.warning(f"SwarmForward not implemented on {candidate.node_id}, falling back")
                # Fall back to PipelineForward
                request = neuroshard_pb2.PipelineForwardRequest(
                    session_id=session_id,
                    request_id=f"pipe_{time.time()}",
                    hidden_states=tensor_bytes,
                    hidden_shape=shape,
                    target_shard=candidate.layer_range[0],
                    sender_url=metadata.get("source_node", "") if metadata else "",
                )
                response = await stub.PipelineForward(request)
                return response
            raise
    
    def get_stats(self) -> Dict[str, Any]:
        """Get router statistics."""
        success_rate = (
            self.successful_routes / self.total_routes 
            if self.total_routes > 0 else 0.0
        )
        failover_rate = (
            self.failover_count / self.total_routes 
            if self.total_routes > 0 else 0.0
        )
        
        return {
            "total_routes": self.total_routes,
            "successful_routes": self.successful_routes,
            "success_rate": success_rate,
            "failover_count": self.failover_count,
            "failover_rate": failover_rate,
            "known_peers": len(self.peer_stats),
            "active_channels": len(self._channels),
        }


# Convenience function for backwards compatibility
async def get_swarm_candidates(
    router: SwarmRouter,
    target_layer: int,
) -> List[PeerCandidate]:
    """Get swarm routing candidates for a layer."""
    return router.get_candidates(target_layer)

