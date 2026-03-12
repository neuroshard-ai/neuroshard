"""
Swarm Heartbeat Service - Capacity Advertisement Protocol

Implements UDP-based heartbeat broadcasting for swarm awareness:
- Every node broadcasts capacity bitmask every 5 seconds
- Peers learn about available compute before routing
- Enables load-aware routing decisions

Key Directive: "Nodes must broadcast a lightweight Capacity Bitmask every 5 seconds 
so peers know who is ready to receive work."

Capacity Bitmask contains:
- Available memory (MB)
- Queue depth (current work queue size)
- Layer range (which layers this node handles)
- GPU utilization (0-100%)
- Status flags (training, accepting inference, accepting activations)
"""

import asyncio
import json
import logging
import socket
import struct
import threading
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Callable, Tuple, Any

logger = logging.getLogger(__name__)


@dataclass
class CapacityBitmask:
    """
    Lightweight capacity advertisement broadcast every 5 seconds.
    
    Designed to fit in a single UDP packet (~64 bytes serialized).
    Contains all info needed for routing decisions.
    """
    node_id: str
    timestamp: float = field(default_factory=time.time)
    
    # Network address (for gRPC connections)
    grpc_addr: str = ""
    
    # Capacity info
    available_memory_mb: int = 0      # Free GPU/system memory
    queue_depth: int = 0              # Current work queue size (0-65535)
    layer_range: Tuple[int, int] = (0, 0)  # (start_layer, end_layer)
    
    # Utilization metrics  
    gpu_utilization: float = 0.0      # 0-100%
    network_saturation: float = 0.0   # 0-100%
    
    # Status flags
    is_training: bool = False
    is_accepting_inference: bool = True
    is_accepting_activations: bool = True
    
    # Optional: current training step for sync coordination
    training_step: int = 0
    
    def to_bytes(self) -> bytes:
        """
        Serialize to compact binary format (~64 bytes).
        
        Format:
        - 4 bytes: magic number (0x4E455552 = "NEUR")
        - 1 byte:  version
        - 32 bytes: node_id (truncated/padded)
        - 8 bytes: timestamp (double)
        - 4 bytes: available_memory_mb (uint32)
        - 2 bytes: queue_depth (uint16)
        - 2 bytes: layer_start (uint16)
        - 2 bytes: layer_end (uint16)
        - 1 byte:  gpu_utilization (uint8, 0-100)
        - 1 byte:  network_saturation (uint8, 0-100)
        - 1 byte:  flags (bit field)
        - 4 bytes: training_step (uint32)
        - remaining: grpc_addr (variable, null-terminated)
        """
        flags = 0
        if self.is_training:
            flags |= 0x01
        if self.is_accepting_inference:
            flags |= 0x02
        if self.is_accepting_activations:
            flags |= 0x04
            
        # Truncate/pad node_id to 32 bytes
        node_id_bytes = self.node_id.encode('utf-8')[:32].ljust(32, b'\x00')
        
        # Pack fixed fields
        header = struct.pack(
            '>I B 32s d I H H H B B B I',
            0x4E455552,                    # Magic "NEUR"
            1,                             # Version
            node_id_bytes,                 # Node ID (32 bytes)
            self.timestamp,                # Timestamp
            self.available_memory_mb,      # Memory
            min(65535, self.queue_depth),  # Queue depth
            self.layer_range[0],           # Layer start
            self.layer_range[1],           # Layer end
            int(min(100, max(0, self.gpu_utilization))),      # GPU util
            int(min(100, max(0, self.network_saturation))),   # Network sat
            flags,                         # Status flags
            self.training_step,            # Training step
        )
        
        # Append grpc_addr (null-terminated)
        addr_bytes = (self.grpc_addr + '\x00').encode('utf-8')[:128]
        
        return header + addr_bytes
    
    @classmethod
    def from_bytes(cls, data: bytes) -> Optional['CapacityBitmask']:
        """Deserialize from binary format."""
        if len(data) < 62:  # Minimum size without grpc_addr
            return None
            
        try:
            # Unpack fixed header
            magic, version, node_id_bytes, timestamp, memory, queue, \
                layer_start, layer_end, gpu_util, net_sat, flags, step = \
                struct.unpack('>I B 32s d I H H H B B B I', data[:62])
            
            # Validate magic number
            if magic != 0x4E455552:
                logger.debug(f"Invalid magic number: {magic:#x}")
                return None
                
            # Version check
            if version > 1:
                logger.debug(f"Unknown version: {version}")
                # Continue anyway for forward compatibility
                
            # Decode node_id (strip null padding)
            node_id = node_id_bytes.rstrip(b'\x00').decode('utf-8', errors='replace')
            
            # Extract grpc_addr from remaining bytes
            grpc_addr = ""
            if len(data) > 62:
                addr_data = data[62:]
                null_idx = addr_data.find(b'\x00')
                if null_idx >= 0:
                    grpc_addr = addr_data[:null_idx].decode('utf-8', errors='replace')
                else:
                    grpc_addr = addr_data.decode('utf-8', errors='replace')
            
            return cls(
                node_id=node_id,
                timestamp=timestamp,
                grpc_addr=grpc_addr,
                available_memory_mb=memory,
                queue_depth=queue,
                layer_range=(layer_start, layer_end),
                gpu_utilization=float(gpu_util),
                network_saturation=float(net_sat),
                is_training=bool(flags & 0x01),
                is_accepting_inference=bool(flags & 0x02),
                is_accepting_activations=bool(flags & 0x04),
                training_step=step,
            )
            
        except Exception as e:
            logger.error(f"Failed to deserialize heartbeat: {e}")
            return None
    
    def to_json(self) -> str:
        """Serialize to JSON (for debugging/logging)."""
        return json.dumps({
            "node_id": self.node_id,
            "timestamp": self.timestamp,
            "grpc_addr": self.grpc_addr,
            "available_memory_mb": self.available_memory_mb,
            "queue_depth": self.queue_depth,
            "layer_range": list(self.layer_range),
            "gpu_utilization": self.gpu_utilization,
            "network_saturation": self.network_saturation,
            "is_training": self.is_training,
            "is_accepting_inference": self.is_accepting_inference,
            "is_accepting_activations": self.is_accepting_activations,
            "training_step": self.training_step,
        })
    
    @classmethod
    def from_json(cls, data: str) -> Optional['CapacityBitmask']:
        """Deserialize from JSON."""
        try:
            d = json.loads(data)
            return cls(
                node_id=d.get("node_id", ""),
                timestamp=d.get("timestamp", time.time()),
                grpc_addr=d.get("grpc_addr", ""),
                available_memory_mb=d.get("available_memory_mb", 0),
                queue_depth=d.get("queue_depth", 0),
                layer_range=tuple(d.get("layer_range", [0, 0])),
                gpu_utilization=d.get("gpu_utilization", 0.0),
                network_saturation=d.get("network_saturation", 0.0),
                is_training=d.get("is_training", False),
                is_accepting_inference=d.get("is_accepting_inference", True),
                is_accepting_activations=d.get("is_accepting_activations", True),
                training_step=d.get("training_step", 0),
            )
        except Exception as e:
            logger.error(f"Failed to parse heartbeat JSON: {e}")
            return None


class SwarmHeartbeatService:
    """
    Broadcasts and receives capacity heartbeats over UDP.
    
    Features:
    - Periodic broadcast (default: every 5 seconds)
    - Multicast or direct peer-to-peer UDP
    - Stale peer detection (dead after 15s)
    - Integration with SwarmRouter for routing updates
    
    Usage:
        service = SwarmHeartbeatService(node_id="abc123", udp_port=9999)
        service.set_capacity_callback(get_my_capacity)
        service.set_peer_update_callback(router.update_peer_from_heartbeat)
        service.start()
    """
    
    HEARTBEAT_INTERVAL = 5.0   # Broadcast every 5 seconds
    STALE_THRESHOLD = 15.0     # Consider peer dead after 15s
    DEAD_THRESHOLD = 60.0      # Remove peer after 60s
    
    # Multicast group for local network discovery
    MULTICAST_GROUP = "239.255.42.42"
    MULTICAST_PORT = 9999
    
    def __init__(
        self,
        node_id: str,
        udp_port: int = MULTICAST_PORT,
        grpc_addr: str = "",
        use_multicast: bool = True,
        known_peers: Optional[List[str]] = None,
    ):
        """
        Initialize heartbeat service.
        
        Args:
            node_id: This node's unique identifier
            udp_port: UDP port for heartbeat traffic
            grpc_addr: This node's gRPC address (for inclusion in heartbeat)
            use_multicast: Whether to use multicast (True) or unicast (False)
            known_peers: List of peer UDP addresses for unicast mode
        """
        self.node_id = node_id
        self.udp_port = udp_port
        self.grpc_addr = grpc_addr
        self.use_multicast = use_multicast
        self.known_peer_addrs = known_peers or []
        
        # Peer state
        self.peer_capacities: Dict[str, CapacityBitmask] = {}
        self._peer_lock = threading.Lock()
        
        # Local capacity (updated via update_local_capacity())
        self._local_capacity: Optional[CapacityBitmask] = None
        
        # Callbacks
        self._capacity_callback: Optional[Callable[[], CapacityBitmask]] = None
        self._peer_update_callback: Optional[Callable[[CapacityBitmask], None]] = None
        
        # State
        self.running = False
        self._broadcast_thread: Optional[threading.Thread] = None
        self._listen_thread: Optional[threading.Thread] = None
        self._socket: Optional[socket.socket] = None
        
        # Metrics
        self.heartbeats_sent = 0
        self.heartbeats_received = 0
        self.peers_discovered = 0
        self.peers_lost = 0
    
    @property
    def broadcast_count(self) -> int:
        """Alias for heartbeats_sent."""
        return self.heartbeats_sent
        
    def set_capacity_callback(self, callback: Callable[[], CapacityBitmask]):
        """
        Set callback to get current node capacity.
        
        The callback should return a CapacityBitmask with current state.
        Called every heartbeat interval.
        """
        self._capacity_callback = callback
        
    def set_peer_update_callback(self, callback: Callable[[CapacityBitmask], None]):
        """
        Set callback for peer capacity updates.
        
        Called when a heartbeat is received from a peer.
        Typically used to update SwarmRouter's peer stats.
        """
        self._peer_update_callback = callback
        
    def add_known_peer(self, peer_addr: str):
        """Add a peer address for unicast heartbeats."""
        if peer_addr not in self.known_peer_addrs:
            self.known_peer_addrs.append(peer_addr)
    
    def update_local_capacity(
        self,
        available_memory_mb: int,
        queue_depth: int,
        layer_range: Tuple[int, int],
        gpu_utilization: float = 0.0,
        is_training: bool = False,
        is_accepting_activations: bool = True,
    ):
        """
        Update local capacity info for next heartbeat broadcast.
        
        Called by SwarmServiceMixin to update capacity state.
        The actual broadcast happens in the background thread.
        """
        self._local_capacity = CapacityBitmask(
            node_id=self.node_id,
            grpc_addr=self.grpc_addr,
            available_memory_mb=available_memory_mb,
            queue_depth=queue_depth,
            layer_range=layer_range,
            gpu_utilization=gpu_utilization,
            is_training=is_training,
            is_accepting_activations=is_accepting_activations,
        )
            
    def start(self):
        """Start heartbeat broadcast and listener."""
        if self.running:
            return
            
        self.running = True
        
        # Setup UDP socket
        self._setup_socket()
        
        # Start broadcast thread
        self._broadcast_thread = threading.Thread(
            target=self._broadcast_loop,
            daemon=True,
            name="SwarmHeartbeat-Broadcast"
        )
        self._broadcast_thread.start()
        
        # Start listener thread
        self._listen_thread = threading.Thread(
            target=self._listen_loop,
            daemon=True,
            name="SwarmHeartbeat-Listen"
        )
        self._listen_thread.start()
        
        logger.info(f"SwarmHeartbeatService started on port {self.udp_port}")
        
    def stop(self):
        """Stop heartbeat service."""
        self.running = False
        
        if self._socket:
            try:
                self._socket.close()
            except:
                pass
                
        if self._broadcast_thread:
            self._broadcast_thread.join(timeout=2.0)
        if self._listen_thread:
            self._listen_thread.join(timeout=2.0)
            
        logger.info("SwarmHeartbeatService stopped")
        
    def _setup_socket(self):
        """Setup UDP socket for heartbeat traffic."""
        self._socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        
        # Allow multiple processes on same host (for testing)
        try:
            self._socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEPORT, 1)
        except AttributeError:
            pass  # Not available on all platforms
            
        if self.use_multicast:
            # Join multicast group
            self._socket.bind(('', self.udp_port))
            group = socket.inet_aton(self.MULTICAST_GROUP)
            mreq = struct.pack('4sL', group, socket.INADDR_ANY)
            self._socket.setsockopt(
                socket.IPPROTO_IP,
                socket.IP_ADD_MEMBERSHIP,
                mreq
            )
            # Set TTL for multicast
            self._socket.setsockopt(
                socket.IPPROTO_IP,
                socket.IP_MULTICAST_TTL,
                2
            )
        else:
            self._socket.bind(('0.0.0.0', self.udp_port))
            
        # Non-blocking with timeout
        self._socket.settimeout(1.0)
        
    def _broadcast_loop(self):
        """Periodically broadcast capacity heartbeat."""
        while self.running:
            try:
                self._broadcast_heartbeat()
                time.sleep(self.HEARTBEAT_INTERVAL)
            except Exception as e:
                if self.running:
                    logger.error(f"Broadcast error: {e}")
                time.sleep(1.0)
                
    def _broadcast_heartbeat(self):
        """Send a single heartbeat broadcast."""
        if not self._capacity_callback:
            # Create default capacity
            capacity = CapacityBitmask(
                node_id=self.node_id,
                grpc_addr=self.grpc_addr,
            )
        else:
            capacity = self._capacity_callback()
            # Ensure our info is set
            capacity.node_id = self.node_id
            if not capacity.grpc_addr:
                capacity.grpc_addr = self.grpc_addr
                
        data = capacity.to_bytes()
        
        if self.use_multicast:
            # Send to multicast group
            self._socket.sendto(data, (self.MULTICAST_GROUP, self.udp_port))
        else:
            # Send to known peers
            for peer_addr in self.known_peer_addrs:
                try:
                    host, port = self._parse_addr(peer_addr)
                    self._socket.sendto(data, (host, port))
                except Exception as e:
                    logger.debug(f"Failed to send to {peer_addr}: {e}")
                    
        self.heartbeats_sent += 1
        
    def _parse_addr(self, addr: str) -> Tuple[str, int]:
        """Parse address string to (host, port)."""
        if ':' in addr:
            host, port = addr.rsplit(':', 1)
            return host, int(port)
        return addr, self.udp_port
        
    def _listen_loop(self):
        """Listen for heartbeats from peers."""
        while self.running:
            try:
                data, addr = self._socket.recvfrom(256)
                self._handle_heartbeat(data, addr)
            except socket.timeout:
                # Check for stale peers periodically
                self._cleanup_stale_peers()
            except Exception as e:
                if self.running:
                    logger.debug(f"Listen error: {e}")
                    
    def _handle_heartbeat(self, data: bytes, addr: Tuple[str, int]):
        """Process received heartbeat packet."""
        capacity = CapacityBitmask.from_bytes(data)
        if not capacity:
            return
            
        # Ignore our own heartbeats
        if capacity.node_id == self.node_id:
            return
            
        self.heartbeats_received += 1
        
        with self._peer_lock:
            is_new = capacity.node_id not in self.peer_capacities
            self.peer_capacities[capacity.node_id] = capacity
            
            if is_new:
                self.peers_discovered += 1
                logger.info(
                    f"Discovered peer {capacity.node_id[:8]}... "
                    f"layers={capacity.layer_range}, "
                    f"queue={capacity.queue_depth}"
                )
                
                # Add to known peers for unicast
                if capacity.grpc_addr and not self.use_multicast:
                    self.add_known_peer(f"{addr[0]}:{self.udp_port}")
                    
        # Notify callback
        if self._peer_update_callback:
            try:
                self._peer_update_callback(capacity)
            except Exception as e:
                logger.error(f"Peer update callback error: {e}")
                
    def _cleanup_stale_peers(self):
        """Remove peers that haven't sent heartbeat recently."""
        now = time.time()
        
        with self._peer_lock:
            dead_peers = []
            
            for node_id, capacity in self.peer_capacities.items():
                staleness = now - capacity.timestamp
                
                if staleness > self.DEAD_THRESHOLD:
                    dead_peers.append(node_id)
                elif staleness > self.STALE_THRESHOLD:
                    # Mark as not accepting (but keep for history)
                    capacity.is_accepting_activations = False
                    capacity.is_accepting_inference = False
                    
            for node_id in dead_peers:
                del self.peer_capacities[node_id]
                self.peers_lost += 1
                logger.info(f"Lost peer {node_id[:8]}... (no heartbeat for {self.DEAD_THRESHOLD}s)")
    
    def get_available_peers(
        self,
        min_memory: int = 0,
        target_layer: Optional[int] = None,
    ) -> List[CapacityBitmask]:
        """
        Get peers with available capacity.
        
        Args:
            min_memory: Minimum required memory (MB)
            target_layer: Optional layer to filter by
            
        Returns:
            List of CapacityBitmask for available peers
        """
        now = time.time()
        
        with self._peer_lock:
            result = []
            
            for capacity in self.peer_capacities.values():
                # Check freshness
                if (now - capacity.timestamp) > self.STALE_THRESHOLD:
                    continue
                    
                # Check availability
                if not capacity.is_accepting_activations:
                    continue
                    
                # Check memory
                if capacity.available_memory_mb < min_memory:
                    continue
                    
                # Check layer coverage
                if target_layer is not None:
                    if not (capacity.layer_range[0] <= target_layer < capacity.layer_range[1]):
                        continue
                        
                result.append(capacity)
                
            return result
    
    def get_stats(self) -> Dict[str, Any]:
        """Get heartbeat service statistics."""
        with self._peer_lock:
            active_peers = sum(
                1 for c in self.peer_capacities.values()
                if c.is_accepting_activations
            )
            
        return {
            "running": self.running,
            "heartbeats_sent": self.heartbeats_sent,
            "heartbeats_received": self.heartbeats_received,
            "peers_discovered": self.peers_discovered,
            "peers_lost": self.peers_lost,
            "active_peers": active_peers,
            "total_known_peers": len(self.peer_capacities),
            "use_multicast": self.use_multicast,
        }


def create_capacity_callback(
    layer_range: Tuple[int, int],
    inbound_buffer: Any,  # ActivationBuffer
    outbound_buffer: Any,  # OutboundBuffer
    get_memory_fn: Optional[Callable[[], int]] = None,
    get_gpu_util_fn: Optional[Callable[[], float]] = None,
    is_training: bool = False,
) -> Callable[[], CapacityBitmask]:
    """
    Factory to create a capacity callback function.
    
    Args:
        layer_range: (start, end) layer range
        inbound_buffer: ActivationBuffer for queue depth
        outbound_buffer: OutboundBuffer for network saturation
        get_memory_fn: Optional function returning available memory (MB)
        get_gpu_util_fn: Optional function returning GPU utilization (0-100)
        is_training: Whether node is currently training
        
    Returns:
        Callback function returning CapacityBitmask
    """
    def _get_default_memory() -> int:
        """Default: report 4GB available."""
        try:
            import torch
            if torch.cuda.is_available():
                free, total = torch.cuda.mem_get_info()
                return int(free / 1024 / 1024)
        except:
            pass
        return 4096
        
    def _get_default_gpu_util() -> float:
        """Default: report 0% if no GPU."""
        try:
            import torch
            if torch.cuda.is_available():
                # Rough estimate from memory usage
                free, total = torch.cuda.mem_get_info()
                return 100.0 * (1 - free / total)
        except:
            pass
        return 0.0
        
    memory_fn = get_memory_fn or _get_default_memory
    gpu_fn = get_gpu_util_fn or _get_default_gpu_util
    
    def callback() -> CapacityBitmask:
        # Get queue depth from inbound buffer
        queue_depth = 0
        if hasattr(inbound_buffer, '_queue'):
            queue_depth = len(inbound_buffer._queue)
            
        # Get network saturation from outbound buffer
        net_sat = 0.0
        if hasattr(outbound_buffer, 'fill_rate'):
            net_sat = outbound_buffer.fill_rate * 100
            
        return CapacityBitmask(
            node_id="",  # Set by service
            timestamp=time.time(),
            available_memory_mb=memory_fn(),
            queue_depth=queue_depth,
            layer_range=layer_range,
            gpu_utilization=gpu_fn(),
            network_saturation=net_sat,
            is_training=is_training,
            is_accepting_inference=not is_training or queue_depth < 50,
            is_accepting_activations=queue_depth < 80,
        )
        
    return callback

