"""
Swarm Service Mixin for gRPC Integration

Integrates Phase 1 swarm components with the existing gRPC server:
- SwarmRouter for multipath routing with failover
- ActivationBuffer for async compute decoupling
- ComputeEngine for "Don't Stop" soft overflow handling
- SwarmHeartbeat for capacity advertisement

This mixin is designed to be added to the existing NeuroShardServiceServicer.
"""

import asyncio
import time
import logging
import threading
import numpy as np
from typing import Dict, Optional, Tuple, Any
from dataclasses import dataclass, field
from concurrent.futures import ThreadPoolExecutor

import torch

from neuroshard.core.swarm.router import SwarmRouter, PeerCandidate
from neuroshard.core.swarm.buffers import (
    ActivationBuffer,
    OutboundBuffer,
    ActivationPacket,
)
from neuroshard.core.swarm.heartbeat import SwarmHeartbeatService, CapacityBitmask
from neuroshard.core.swarm.compute import ComputeEngine, StepOutcome, ComputeStats

logger = logging.getLogger(__name__)


@dataclass
class SwarmNodeState:
    """State container for swarm node operation."""
    node_id: str
    layer_range: Tuple[int, int]
    grpc_addr: str
    
    # Capacity
    available_memory_mb: int = 0
    gpu_utilization: float = 0.0
    
    # Status
    is_training: bool = False
    is_accepting_inference: bool = True
    is_accepting_activations: bool = True
    
    # Timestamps
    last_updated: float = field(default_factory=time.time)


class SwarmServiceMixin:
    """
    Mixin class that adds Swarm functionality to NeuroShardServiceServicer.
    
    Provides:
    - Async activation handling via ActivationBuffer
    - Fault-tolerant routing via SwarmRouter
    - Capacity broadcasting via SwarmHeartbeatService
    - Decoupled compute via ComputeEngine
    
    Usage:
        class NeuroShardServiceServicer(SwarmServiceMixin, DHTServiceMixin, ...):
            def __init__(self, ...):
                SwarmServiceMixin.__init__(self, model, layer_pool)
                ...
    """
    
    # Configuration
    INBOUND_BUFFER_SIZE = 100
    OUTBOUND_BUFFER_SIZE = 50
    HEARTBEAT_PORT = 9999
    
    def __init_swarm__(
        self,
        model,
        layer_pool,
        p2p_manager=None,
        enable_heartbeat: bool = True,
        enable_compute_engine: bool = True,
    ):
        """
        Initialize swarm components.
        
        Args:
            model: DynamicNeuroNode or DynamicNeuroLLM instance
            layer_pool: DynamicLayerPool for peer discovery
            p2p_manager: P2PManager for network communication
            enable_heartbeat: Whether to start heartbeat service
            enable_compute_engine: Whether to start compute engine
        """
        self.swarm_model = model
        self.layer_pool = layer_pool
        self.p2p = p2p_manager
        
        # Extract node info
        self.swarm_node_id = getattr(model, 'node_id', 'unknown')
        self.swarm_layer_ids = getattr(model, 'my_layer_ids', [])
        
        # Initialize buffers
        self.inbound_buffer = ActivationBuffer(max_size=self.INBOUND_BUFFER_SIZE)
        self.outbound_buffer = OutboundBuffer(
            max_size=self.OUTBOUND_BUFFER_SIZE,
            soft_overflow_threshold=0.9,
            hard_overflow_threshold=0.99
        )
        
        # Initialize router (uses DHT/layer_pool for peer discovery)
        dht = getattr(p2p_manager, 'routing_table', None) if p2p_manager else None
        self.swarm_router = SwarmRouter(dht_protocol=dht, layer_pool=layer_pool)
        
        # Initialize heartbeat service
        self.heartbeat_service = None
        if enable_heartbeat:
            self.heartbeat_service = SwarmHeartbeatService(
                node_id=self.swarm_node_id,
                udp_port=self.HEARTBEAT_PORT
            )
        
        # Initialize compute engine (runs async)
        self.compute_engine = None
        self._compute_thread = None
        self._async_loop = None
        
        # State tracking
        self.swarm_state = SwarmNodeState(
            node_id=self.swarm_node_id,
            layer_range=self._get_layer_range(),
            grpc_addr=getattr(model, 'grpc_addr', ''),
        )
        
        # Statistics
        self.swarm_stats = {
            'activations_received': 0,
            'activations_sent': 0,
            'failovers': 0,
            'soft_overflows': 0,
            'hard_overflows': 0,
        }
        
        logger.info(f"SwarmServiceMixin initialized for node {self.swarm_node_id}")
        logger.info(f"  Layers: {self.swarm_layer_ids}")
        logger.info(f"  Inbound buffer: {self.INBOUND_BUFFER_SIZE}")
        logger.info(f"  Outbound buffer: {self.OUTBOUND_BUFFER_SIZE}")
    
    def _get_layer_range(self) -> Tuple[int, int]:
        """Get the layer range for this node."""
        if not self.swarm_layer_ids:
            return (0, 0)
        return (min(self.swarm_layer_ids), max(self.swarm_layer_ids) + 1)
    
    # ==================== LIFECYCLE ====================
    
    def start_swarm_services(self):
        """Start all swarm background services."""
        # Start heartbeat
        if self.heartbeat_service:
            self._update_heartbeat_capacity()
            self.heartbeat_service.start()
            logger.info("Swarm heartbeat service started")
        
        # Start compute engine in background thread with async loop
        self._start_compute_engine()
        
        # Start outbound send loop
        self._start_outbound_sender()
        
        logger.info("All swarm services started")
    
    def stop_swarm_services(self):
        """Stop all swarm background services."""
        # Stop compute engine
        if self.compute_engine:
            self.compute_engine.running = False
        
        # Stop heartbeat
        if self.heartbeat_service:
            self.heartbeat_service.stop()
        
        # Stop async loop
        if self._async_loop:
            self._async_loop.call_soon_threadsafe(self._async_loop.stop)
        
        logger.info("All swarm services stopped")
    
    def _start_compute_engine(self):
        """Start the compute engine in a background thread."""
        if not hasattr(self, 'swarm_model') or self.swarm_model is None:
            logger.warning("No model available for compute engine")
            return
        
        # Create compute engine
        self.compute_engine = ComputeEngine(
            model=self.swarm_model,
            inbound=self.inbound_buffer,
            outbound=self.outbound_buffer,
            num_micro_batches=4
        )
        
        def run_async_loop():
            """Run asyncio event loop in thread."""
            self._async_loop = asyncio.new_event_loop()
            asyncio.set_event_loop(self._async_loop)
            
            try:
                self._async_loop.run_until_complete(self.compute_engine.run())
            except Exception as e:
                logger.error(f"Compute engine error: {e}")
            finally:
                self._async_loop.close()
        
        self._compute_thread = threading.Thread(
            target=run_async_loop,
            daemon=True,
            name="SwarmComputeEngine"
        )
        self._compute_thread.start()
        logger.info("Compute engine started in background thread")
    
    def _start_outbound_sender(self):
        """Start the outbound buffer send loop."""
        def send_loop():
            """Continuously send packets from outbound buffer."""
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            
            async def _send():
                while True:
                    try:
                        packet = await self.outbound_buffer.get()
                        if packet is None:
                            await asyncio.sleep(0.01)
                            continue
                        
                        # Send via swarm router with failover
                        try:
                            await self.swarm_router.send_with_failover(
                                tensor=packet.tensor_data,
                                target_layer=packet.target_layer,
                                session_id=packet.session_id
                            )
                            self.swarm_stats['activations_sent'] += 1
                        except Exception as e:
                            logger.warning(f"Failed to send activation: {e}")
                            self.swarm_stats['failovers'] += 1
                    except Exception as e:
                        logger.error(f"Outbound send error: {e}")
                        await asyncio.sleep(0.1)
            
            try:
                loop.run_until_complete(_send())
            except Exception as e:
                logger.error(f"Outbound sender error: {e}")
            finally:
                loop.close()
        
        threading.Thread(
            target=send_loop,
            daemon=True,
            name="SwarmOutboundSender"
        ).start()
        logger.info("Outbound sender started")
    
    def _update_heartbeat_capacity(self):
        """Update heartbeat capacity from current state."""
        if not self.heartbeat_service:
            return
        
        # Get memory info
        available_mb = 0
        gpu_util = 0.0
        
        try:
            if torch.cuda.is_available():
                free, total = torch.cuda.mem_get_info()
                available_mb = free // (1024 * 1024)
                # GPU utilization would need nvidia-smi or pynvml
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                # MPS doesn't have direct memory query
                available_mb = 1024  # Estimate
            else:
                import psutil
                mem = psutil.virtual_memory()
                available_mb = mem.available // (1024 * 1024)
        except Exception as e:
            logger.debug(f"Could not get memory info: {e}")
        
        self.swarm_state.available_memory_mb = available_mb
        self.swarm_state.gpu_utilization = gpu_util
        
        # Update heartbeat
        self.heartbeat_service.update_local_capacity(
            available_memory_mb=available_mb,
            queue_depth=len(self.inbound_buffer),
            layer_range=self.swarm_state.layer_range,
            gpu_utilization=gpu_util,
            is_training=self.swarm_state.is_training,
            is_accepting_activations=self.swarm_state.is_accepting_activations,
        )
    
    # ==================== gRPC HANDLERS ====================
    
    def SwarmForward(self, request, context):
        """
        Handle incoming activation packets for swarm routing.
        
        This is the primary RPC for the async pipeline:
        1. Deserialize activation tensor
        2. Push to inbound ActivationBuffer
        3. ComputeEngine processes asynchronously
        4. Return immediately (non-blocking)
        
        This replaces the synchronous PipelineForward for swarm mode.
        """
        try:
            import numpy as np
            from neuroshard.protos import neuroshard_pb2
            
            # Deserialize activation
            hidden_states = torch.from_numpy(
                np.frombuffer(request.hidden_states, dtype=np.float32)
            ).reshape(list(request.hidden_shape))
            
            # Create activation packet
            packet = ActivationPacket(
                priority=request.priority if hasattr(request, 'priority') else 10,
                session_id=request.session_id,
                micro_batch_id=request.micro_batch_id if hasattr(request, 'micro_batch_id') else 0,
                tensor_data=hidden_states,
                source_node=request.sender_url if hasattr(request, 'sender_url') else '',
                target_layer=request.target_layer if hasattr(request, 'target_layer') else 0,
                is_backward=request.is_backward if hasattr(request, 'is_backward') else False,
            )
            
            # Non-blocking put to inbound buffer
            success = self.inbound_buffer.put_nowait(packet)
            
            if success:
                self.swarm_stats['activations_received'] += 1
                return neuroshard_pb2.SwarmForwardResponse(
                    success=True,
                    request_id=request.request_id,
                    buffer_depth=len(self.inbound_buffer),
                )
            else:
                # Buffer full - backpressure signal
                return neuroshard_pb2.SwarmForwardResponse(
                    success=False,
                    request_id=request.request_id,
                    error_message="Inbound buffer full - backpressure",
                    buffer_depth=len(self.inbound_buffer),
                )
                
        except Exception as e:
            logger.error(f"SwarmForward error: {e}")
            from neuroshard.protos import neuroshard_pb2
            return neuroshard_pb2.SwarmForwardResponse(
                success=False,
                request_id=getattr(request, 'request_id', ''),
                error_message=str(e),
            )
    
    def GetSwarmStatus(self, request, context):
        """
        Get swarm node status including buffer fill rates and capacity.
        
        Used by peers for routing decisions.
        """
        try:
            from neuroshard.protos import neuroshard_pb2
            
            # Get buffer stats
            inbound_stats = self.inbound_buffer.get_stats()
            outbound_stats = self.outbound_buffer.get_stats()
            
            # Get compute stats if available
            compute_stats = {}
            if self.compute_engine:
                compute_stats = self.compute_engine.get_stats()
            
            return neuroshard_pb2.SwarmStatusResponse(
                node_id=self.swarm_node_id,
                layer_start=self.swarm_state.layer_range[0],
                layer_end=self.swarm_state.layer_range[1],
                
                # Buffer status
                inbound_fill_rate=inbound_stats.get('fill_rate', 0.0),
                outbound_fill_rate=outbound_stats.get('fill_rate', 0.0),
                inbound_queue_depth=inbound_stats.get('queue_size', 0),
                outbound_queue_depth=outbound_stats.get('queue_size', 0),
                
                # Capacity
                available_memory_mb=self.swarm_state.available_memory_mb,
                gpu_utilization=self.swarm_state.gpu_utilization,
                
                # Status
                is_training=self.swarm_state.is_training,
                is_accepting_activations=self.swarm_state.is_accepting_activations,
                
                # Compute stats
                total_steps=compute_stats.get('total_steps', 0),
                local_only_rate=compute_stats.get('local_only_rate', 0.0),
            )
            
        except Exception as e:
            logger.error(f"GetSwarmStatus error: {e}")
            from neuroshard.protos import neuroshard_pb2
            return neuroshard_pb2.SwarmStatusResponse(
                node_id=self.swarm_node_id,
                error_message=str(e),
            )
    
    def UpdatePeerCapacity(self, request, context):
        """
        Receive peer capacity update (alternative to UDP heartbeat).
        
        Used when UDP heartbeat is blocked by firewalls.
        """
        try:
            from neuroshard.protos import neuroshard_pb2
            
            # Update router's peer cache
            peer = PeerCandidate(
                node_id=request.node_id,
                grpc_addr=request.grpc_addr,
                layer_range=(request.layer_start, request.layer_end),
                latency_ms=0.0,  # Will be measured on first send
                queue_depth=request.queue_depth,
                last_heartbeat=time.time(),
            )
            
            self.swarm_router.update_peer_from_heartbeat(peer)
            
            return neuroshard_pb2.UpdatePeerCapacityResponse(
                success=True,
            )
            
        except Exception as e:
            logger.error(f"UpdatePeerCapacity error: {e}")
            from neuroshard.protos import neuroshard_pb2
            return neuroshard_pb2.UpdatePeerCapacityResponse(
                success=False,
                error_message=str(e),
            )
    
    # ==================== INTEGRATION WITH EXISTING PIPELINE ====================
    
    def handle_pipeline_forward_swarm(self, request, context):
        """
        Adapter for existing PipelineForward to use swarm buffers.
        
        Converts synchronous PipelineForward to async buffer-based processing.
        Can be called from PipelineForward() to enable swarm mode.
        """
        import numpy as np
        from neuroshard.protos import neuroshard_pb2
        
        try:
            # Deserialize hidden states
            hidden_states = torch.from_numpy(
                np.frombuffer(request.hidden_states, dtype=np.float32)
            ).reshape(list(request.hidden_shape))
            
            # Determine priority
            priority = 10  # Default: normal inference
            if hasattr(request, 'is_training') and request.is_training:
                priority = 20  # Training forward
            
            # Create packet
            packet = ActivationPacket(
                priority=priority,
                session_id=request.session_id,
                micro_batch_id=0,
                tensor_data=hidden_states,
                source_node=request.sender_url if hasattr(request, 'sender_url') else '',
                target_layer=request.target_shard if hasattr(request, 'target_shard') else 0,
                is_backward=False,
            )
            
            # Check buffer pressure before accepting
            if self.inbound_buffer.is_backpressured:
                return neuroshard_pb2.PipelineForwardResponse(
                    request_id=request.request_id,
                    success=False,
                    error_message="Node backpressured - try another peer",
                )
            
            # Non-blocking put
            if not self.inbound_buffer.put_nowait(packet):
                return neuroshard_pb2.PipelineForwardResponse(
                    request_id=request.request_id,
                    success=False,
                    error_message="Buffer full",
                )
            
            self.swarm_stats['activations_received'] += 1
            
            # In async mode, we return immediately
            # The compute engine will process and send to next peer
            return neuroshard_pb2.PipelineForwardResponse(
                request_id=request.request_id,
                success=True,
                # hidden_states not returned - async processing
            )
            
        except Exception as e:
            logger.error(f"handle_pipeline_forward_swarm error: {e}")
            return neuroshard_pb2.PipelineForwardResponse(
                request_id=request.request_id,
                success=False,
                error_message=str(e),
            )
    
    # ==================== UTILITIES ====================
    
    def get_swarm_stats(self) -> Dict[str, Any]:
        """Get comprehensive swarm statistics."""
        stats = dict(self.swarm_stats)
        
        # Add buffer stats
        stats['inbound'] = self.inbound_buffer.get_stats()
        stats['outbound'] = self.outbound_buffer.get_stats()
        
        # Add compute stats
        if self.compute_engine:
            stats['compute'] = self.compute_engine.get_stats()
        
        # Add router stats
        stats['router'] = {
            'cached_peers': len(self.swarm_router.peer_stats),
            'total_sends': self.swarm_router.total_sends,
            'successful_sends': self.swarm_router.successful_sends,
            'failovers': self.swarm_router.failover_count,
        }
        
        # Add heartbeat stats
        if self.heartbeat_service:
            stats['heartbeat'] = {
                'known_peers': len(self.heartbeat_service.peer_capacities),
                'broadcast_count': self.heartbeat_service.broadcast_count,
            }
        
        return stats
    
    def set_training_mode(self, enabled: bool):
        """Set training mode (affects prioritization)."""
        self.swarm_state.is_training = enabled
        self._update_heartbeat_capacity()
    
    def set_accepting_activations(self, enabled: bool):
        """Set whether this node accepts new activations."""
        self.swarm_state.is_accepting_activations = enabled
        self._update_heartbeat_capacity()


# ==================== PROTO EXTENSION ====================
# These message types need to be added to neuroshard.proto

SWARM_PROTO_EXTENSION = """
// --- Swarm Routing Messages ---

message SwarmForwardRequest {
    string session_id = 1;
    string request_id = 2;
    
    // Activation data
    bytes hidden_states = 3;
    repeated int64 hidden_shape = 4;
    
    // Routing
    int32 target_layer = 5;
    string sender_url = 6;
    
    // Priority (0=highest)
    int32 priority = 7;
    int32 micro_batch_id = 8;
    bool is_backward = 9;
    
    // Training metadata
    bool requires_grad = 10;
    bytes grad_output = 11;
}

message SwarmForwardResponse {
    string request_id = 1;
    bool success = 2;
    string error_message = 3;
    int32 buffer_depth = 4;  // For backpressure signaling
}

message SwarmStatusRequest {
    // Empty - just get status
}

message SwarmStatusResponse {
    string node_id = 1;
    int32 layer_start = 2;
    int32 layer_end = 3;
    
    // Buffer status
    float inbound_fill_rate = 4;
    float outbound_fill_rate = 5;
    int32 inbound_queue_depth = 6;
    int32 outbound_queue_depth = 7;
    
    // Capacity
    int32 available_memory_mb = 8;
    float gpu_utilization = 9;
    
    // Status
    bool is_training = 10;
    bool is_accepting_activations = 11;
    
    // Compute stats
    int64 total_steps = 12;
    float local_only_rate = 13;
    
    string error_message = 14;
}

message UpdatePeerCapacityRequest {
    string node_id = 1;
    string grpc_addr = 2;
    int32 layer_start = 3;
    int32 layer_end = 4;
    int32 queue_depth = 5;
    int32 available_memory_mb = 6;
    float gpu_utilization = 7;
}

message UpdatePeerCapacityResponse {
    bool success = 1;
    string error_message = 2;
}
"""

