"""
NAT Traversal & Connectivity - Residential Network Support

Implements NAT traversal for nodes behind residential NATs:
- STUN client for public IP/port discovery
- Hole punching coordination
- TURN relay fallback
- Connection success rate tracking

Target Metric: Connection success rate > 90% across different ISPs

Options:
- Option A: libp2p integration (recommended for production)
- Option B: STUN/TURN servers (simpler, more centralized)

This module implements Option B with hooks for Option A.
"""

import asyncio
import logging
import random
import socket
import struct
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple, Callable, Any

logger = logging.getLogger(__name__)


class NATType(Enum):
    """
    Detected NAT type.
    
    Different NAT types have different hole-punching success rates:
    - OPEN: No NAT, direct connection works
    - FULL_CONE: Easy hole punch (any external host can connect once mapped)
    - RESTRICTED_CONE: Medium (only replied-to hosts can connect)
    - PORT_RESTRICTED: Harder (specific port must match)
    - SYMMETRIC: Hardest (different mapping per destination, requires relay)
    """
    UNKNOWN = "unknown"
    OPEN = "open"                    # Direct connection works
    FULL_CONE = "full_cone"          # Easy hole punch
    RESTRICTED_CONE = "restricted"   # Medium difficulty
    PORT_RESTRICTED = "port_restricted"  # Harder
    SYMMETRIC = "symmetric"          # Requires relay


@dataclass
class PeerConnectivity:
    """Peer connectivity information."""
    peer_id: str
    public_addr: Optional[Tuple[str, int]] = None
    private_addrs: List[Tuple[str, int]] = field(default_factory=list)
    nat_type: NATType = NATType.UNKNOWN
    relay_addrs: List[str] = field(default_factory=list)
    last_seen: float = field(default_factory=time.time)
    
    @property
    def best_addr(self) -> Optional[Tuple[str, int]]:
        """Get best address to try connecting to."""
        if self.public_addr:
            return self.public_addr
        if self.private_addrs:
            return self.private_addrs[0]
        return None


class STUNClient:
    """
    STUN client for public IP/port discovery.
    
    STUN (Session Traversal Utilities for NAT) allows discovering
    the public IP and port assigned by the NAT.
    
    Default STUN servers:
    - stun.l.google.com:19302
    - stun.cloudflare.com:3478
    """
    
    # Public STUN servers (free tier)
    DEFAULT_STUN_SERVERS = [
        ("stun.l.google.com", 19302),
        ("stun.cloudflare.com", 3478),
        ("stun.stunprotocol.org", 3478),
    ]
    
    # STUN message types
    BINDING_REQUEST = 0x0001
    BINDING_RESPONSE = 0x0101
    
    # STUN attribute types
    MAPPED_ADDRESS = 0x0001
    XOR_MAPPED_ADDRESS = 0x0020
    
    # Magic cookie (RFC 5389)
    MAGIC_COOKIE = 0x2112A442
    
    def __init__(self, stun_servers: Optional[List[Tuple[str, int]]] = None):
        """
        Initialize STUN client.
        
        Args:
            stun_servers: List of (host, port) tuples for STUN servers
        """
        self.stun_servers = stun_servers or self.DEFAULT_STUN_SERVERS
        self.public_ip: Optional[str] = None
        self.public_port: Optional[int] = None
        self.nat_type: NATType = NATType.UNKNOWN
        
        # Local socket (reused for hole punching)
        self._socket: Optional[socket.socket] = None
        self._local_port: int = 0
        
    def _create_binding_request(self) -> Tuple[bytes, bytes]:
        """
        Create STUN Binding Request message.
        
        Returns:
            (message_bytes, transaction_id)
        """
        # Transaction ID (96 bits / 12 bytes)
        transaction_id = random.randbytes(12)
        
        # Message header (20 bytes):
        # - Type: 2 bytes (0x0001 = Binding Request)
        # - Length: 2 bytes (0 for empty body)
        # - Magic Cookie: 4 bytes (0x2112A442)
        # - Transaction ID: 12 bytes
        header = struct.pack(
            '>HHI12s',
            self.BINDING_REQUEST,
            0,  # No attributes, length = 0
            self.MAGIC_COOKIE,
            transaction_id
        )
        
        return header, transaction_id
    
    def _parse_binding_response(
        self, 
        data: bytes, 
        expected_tid: bytes
    ) -> Optional[Tuple[str, int]]:
        """
        Parse STUN Binding Response.
        
        Returns:
            (public_ip, public_port) or None if invalid
        """
        if len(data) < 20:
            return None
            
        # Parse header
        msg_type, msg_len, cookie, tid = struct.unpack('>HHI12s', data[:20])
        
        # Validate
        if msg_type != self.BINDING_RESPONSE:
            logger.debug(f"Unexpected message type: {msg_type:#x}")
            return None
            
        if cookie != self.MAGIC_COOKIE:
            logger.debug(f"Invalid magic cookie: {cookie:#x}")
            return None
            
        if tid != expected_tid:
            logger.debug("Transaction ID mismatch")
            return None
            
        # Parse attributes
        pos = 20
        while pos < len(data):
            if pos + 4 > len(data):
                break
                
            attr_type, attr_len = struct.unpack('>HH', data[pos:pos+4])
            pos += 4
            
            if pos + attr_len > len(data):
                break
                
            attr_data = data[pos:pos+attr_len]
            
            # Handle XOR-MAPPED-ADDRESS (preferred)
            if attr_type == self.XOR_MAPPED_ADDRESS:
                return self._parse_xor_mapped_address(attr_data)
                
            # Handle MAPPED-ADDRESS (fallback)
            elif attr_type == self.MAPPED_ADDRESS:
                return self._parse_mapped_address(attr_data)
                
            # Pad to 4-byte boundary
            pos += attr_len
            pos += (4 - (attr_len % 4)) % 4
            
        return None
    
    def _parse_xor_mapped_address(self, data: bytes) -> Optional[Tuple[str, int]]:
        """Parse XOR-MAPPED-ADDRESS attribute."""
        if len(data) < 8:
            return None
            
        # Format: 1 byte reserved, 1 byte family, 2 bytes port, 4 bytes IP
        _, family, xport = struct.unpack('>BBH', data[:4])
        
        # XOR port with magic cookie (high 16 bits)
        port = xport ^ (self.MAGIC_COOKIE >> 16)
        
        if family == 0x01:  # IPv4
            xaddr = struct.unpack('>I', data[4:8])[0]
            addr = xaddr ^ self.MAGIC_COOKIE
            ip = socket.inet_ntoa(struct.pack('>I', addr))
            return (ip, port)
            
        return None
    
    def _parse_mapped_address(self, data: bytes) -> Optional[Tuple[str, int]]:
        """Parse MAPPED-ADDRESS attribute (not XORed)."""
        if len(data) < 8:
            return None
            
        _, family, port = struct.unpack('>BBH', data[:4])
        
        if family == 0x01:  # IPv4
            ip = socket.inet_ntoa(data[4:8])
            return (ip, port)
            
        return None
    
    async def discover_public_address(
        self, 
        timeout: float = 5.0
    ) -> Optional[Tuple[str, int]]:
        """
        Query STUN servers to discover public IP and port.
        
        Args:
            timeout: Total timeout for discovery
            
        Returns:
            (public_ip, public_port) or None if failed
        """
        # Create UDP socket if needed
        if self._socket is None:
            self._socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            self._socket.setblocking(False)
            self._socket.bind(('0.0.0.0', 0))
            self._local_port = self._socket.getsockname()[1]
            
        loop = asyncio.get_event_loop()
        
        for server_host, server_port in self.stun_servers:
            try:
                # Resolve server address
                try:
                    server_ip = socket.gethostbyname(server_host)
                except socket.gaierror:
                    continue
                    
                # Create request
                request, tid = self._create_binding_request()
                
                # Send request
                await loop.sock_sendto(self._socket, request, (server_ip, server_port))
                
                # Wait for response
                try:
                    data = await asyncio.wait_for(
                        loop.sock_recv(self._socket, 1024),
                        timeout=timeout / len(self.stun_servers)
                    )
                except asyncio.TimeoutError:
                    continue
                    
                # Parse response
                result = self._parse_binding_response(data, tid)
                
                if result:
                    self.public_ip, self.public_port = result
                    logger.info(f"Discovered public address: {self.public_ip}:{self.public_port}")
                    return result
                    
            except Exception as e:
                logger.debug(f"STUN query to {server_host} failed: {e}")
                continue
                
        logger.warning("Could not discover public address via STUN")
        return None
    
    async def detect_nat_type(self) -> NATType:
        """
        Detect NAT type using multiple STUN queries.
        
        Algorithm:
        1. Query STUN server A from port P
        2. Query STUN server B from same port P
        3. Compare mapped addresses:
           - Same = Full Cone / Restricted
           - Different = Symmetric
        4. Try to receive from different port to distinguish Cone types
        
        Returns:
            Detected NATType
        """
        if len(self.stun_servers) < 2:
            return NATType.UNKNOWN
            
        # First query
        result1 = await self.discover_public_address()
        if not result1:
            return NATType.UNKNOWN
            
        # Save first result
        first_ip, first_port = result1
        
        # Query second server from same local port
        if self._socket:
            loop = asyncio.get_event_loop()
            
            server2 = self.stun_servers[1]
            try:
                server2_ip = socket.gethostbyname(server2[0])
            except:
                return NATType.UNKNOWN
                
            request, tid = self._create_binding_request()
            await loop.sock_sendto(self._socket, request, (server2_ip, server2[1]))
            
            try:
                data = await asyncio.wait_for(
                    loop.sock_recv(self._socket, 1024),
                    timeout=3.0
                )
                result2 = self._parse_binding_response(data, tid)
                
                if result2:
                    second_ip, second_port = result2
                    
                    if first_ip != second_ip or first_port != second_port:
                        # Different mapping = Symmetric NAT
                        self.nat_type = NATType.SYMMETRIC
                        logger.info("Detected NAT type: SYMMETRIC")
                        return NATType.SYMMETRIC
                    else:
                        # Same mapping = some form of Cone NAT
                        # For now, assume Restricted Cone (conservative)
                        self.nat_type = NATType.RESTRICTED_CONE
                        logger.info("Detected NAT type: RESTRICTED_CONE")
                        return NATType.RESTRICTED_CONE
                        
            except asyncio.TimeoutError:
                pass
                
        # If we got here with a public address, at least it's not Symmetric
        self.nat_type = NATType.RESTRICTED_CONE
        return self.nat_type
    
    def get_local_socket(self) -> Optional[socket.socket]:
        """Get the local socket (for hole punching)."""
        return self._socket
    
    def close(self):
        """Close the STUN client socket."""
        if self._socket:
            self._socket.close()
            self._socket = None


class HolePuncher:
    """
    Coordinates UDP hole punching between two NATed peers.
    
    Protocol:
    1. Both peers register their public address with coordinator
    2. When A wants to connect to B:
       a. A requests B's public address from coordinator
       b. Coordinator tells B that A wants to connect
       c. Both start sending UDP packets to each other's public addr
       d. NAT creates mapping, packets cross, connection established
       
    Works best with Full Cone / Restricted Cone NATs.
    Symmetric NAT requires relay fallback.
    """
    
    PUNCH_ATTEMPTS = 10
    PUNCH_INTERVAL = 0.1  # 100ms between attempts
    PUNCH_TIMEOUT = 5.0   # Total timeout
    
    def __init__(
        self,
        stun_client: STUNClient,
        coordinator_callback: Optional[Callable] = None,
    ):
        """
        Initialize hole puncher.
        
        Args:
            stun_client: STUN client for address discovery
            coordinator_callback: Function to coordinate with peer
        """
        self.stun = stun_client
        self.coordinator = coordinator_callback
        
        # Punch statistics
        self.attempts = 0
        self.successes = 0
        
    async def punch_hole(
        self,
        peer_addr: Tuple[str, int],
        timeout: float = PUNCH_TIMEOUT,
    ) -> bool:
        """
        Attempt to punch hole to peer.
        
        Args:
            peer_addr: (ip, port) of peer's public address
            timeout: Total timeout for hole punching
            
        Returns:
            True if hole punched successfully
        """
        self.attempts += 1
        
        sock = self.stun.get_local_socket()
        if not sock:
            logger.error("No local socket available for hole punching")
            return False
            
        loop = asyncio.get_event_loop()
        
        # Punch packet - simple magic bytes
        punch_data = b'NEUROSHARD_PUNCH\x00'
        
        start_time = time.time()
        attempts = 0
        
        while (time.time() - start_time) < timeout:
            attempts += 1
            
            try:
                # Send punch packet
                await loop.sock_sendto(sock, punch_data, peer_addr)
                
                # Check for incoming punch from peer
                sock.setblocking(False)
                try:
                    data, addr = await asyncio.wait_for(
                        loop.sock_recvfrom(sock, 64),
                        timeout=self.PUNCH_INTERVAL
                    )
                    
                    if data.startswith(b'NEUROSHARD_PUNCH'):
                        # Success!
                        self.successes += 1
                        logger.info(
                            f"Hole punch successful to {peer_addr} "
                            f"after {attempts} attempts"
                        )
                        return True
                        
                except asyncio.TimeoutError:
                    pass
                    
            except Exception as e:
                logger.debug(f"Punch attempt {attempts} failed: {e}")
                
            await asyncio.sleep(self.PUNCH_INTERVAL)
            
        logger.warning(f"Hole punch failed to {peer_addr} after {attempts} attempts")
        return False
    
    @property
    def success_rate(self) -> float:
        """Calculate hole punch success rate."""
        if self.attempts == 0:
            return 0.0
        return self.successes / self.attempts


class NATTraversalManager:
    """
    Unified NAT traversal manager.
    
    Combines STUN discovery, hole punching, and relay fallback
    into a single interface for connection establishment.
    
    Connection Priority:
    1. Direct (if peer has public IP)
    2. Hole punch (if NAT types are compatible)
    3. Relay (fallback, higher latency)
    
    Target Metric: Connection success rate > 90%
    """
    
    def __init__(
        self,
        node_id: str,
        stun_servers: Optional[List[str]] = None,
        relay_servers: Optional[List[str]] = None,
    ):
        """
        Initialize NAT traversal manager.
        
        Args:
            node_id: This node's identifier
            stun_servers: STUN server addresses ("host:port")
            relay_servers: TURN relay server addresses (future)
        """
        self.node_id = node_id
        
        # Parse STUN servers
        stun_list = []
        if stun_servers:
            for s in stun_servers:
                parts = s.split(':')
                host = parts[0]
                port = int(parts[1]) if len(parts) > 1 else 3478
                stun_list.append((host, port))
                
        self.stun = STUNClient(stun_list if stun_list else None)
        self.hole_puncher = HolePuncher(self.stun)
        
        self.relay_servers = relay_servers or []
        
        # Peer connectivity cache
        self.peer_connectivity: Dict[str, PeerConnectivity] = {}
        
        # Connection statistics
        self.direct_success = 0
        self.holepunch_success = 0
        self.relay_success = 0
        self.total_attempts = 0
        self.total_failures = 0
        
        # Our public address
        self.public_addr: Optional[Tuple[str, int]] = None
        self.nat_type: NATType = NATType.UNKNOWN
        
    async def initialize(self):
        """
        Initialize NAT traversal.
        
        Discovers public address and NAT type.
        """
        logger.info("Initializing NAT traversal...")
        
        # Discover public address
        self.public_addr = await self.stun.discover_public_address()
        
        if self.public_addr:
            # Detect NAT type
            self.nat_type = await self.stun.detect_nat_type()
            logger.info(
                f"NAT traversal initialized: "
                f"public={self.public_addr}, type={self.nat_type.value}"
            )
        else:
            logger.warning("Could not discover public address")
            
    def register_peer(self, peer: PeerConnectivity):
        """Register peer connectivity info."""
        self.peer_connectivity[peer.peer_id] = peer
        
    async def connect(
        self,
        peer_id: str,
        peer_addrs: Optional[List[Tuple[str, int]]] = None,
    ) -> Optional[Tuple[str, int]]:
        """
        Connect to peer using best available method.
        
        Args:
            peer_id: Peer identifier
            peer_addrs: Optional list of peer addresses to try
            
        Returns:
            (ip, port) for established connection, or None if failed
        """
        self.total_attempts += 1
        
        # Get peer info
        peer = self.peer_connectivity.get(peer_id)
        if peer is None and peer_addrs:
            peer = PeerConnectivity(
                peer_id=peer_id,
                public_addr=peer_addrs[0] if peer_addrs else None,
                private_addrs=peer_addrs[1:] if len(peer_addrs) > 1 else [],
            )
            
        if peer is None:
            logger.error(f"No connectivity info for peer {peer_id}")
            self.total_failures += 1
            return None
            
        # Try connection methods in order
        
        # 1. Try direct connection
        if peer.public_addr:
            if await self._try_direct(peer.public_addr):
                self.direct_success += 1
                return peer.public_addr
                
        # 2. Try hole punching
        if peer.public_addr and self._can_hole_punch(peer):
            if await self.hole_puncher.punch_hole(peer.public_addr):
                self.holepunch_success += 1
                return peer.public_addr
                
        # 3. Try relay (not implemented yet)
        if self.relay_servers:
            relay = await self._try_relay(peer_id)
            if relay:
                self.relay_success += 1
                return relay
                
        self.total_failures += 1
        logger.warning(f"All connection methods failed for peer {peer_id}")
        return None
        
    def _can_hole_punch(self, peer: PeerConnectivity) -> bool:
        """Check if hole punching is likely to succeed."""
        # Symmetric NAT on both sides won't work
        if self.nat_type == NATType.SYMMETRIC and peer.nat_type == NATType.SYMMETRIC:
            return False
        return True
        
    async def _try_direct(self, addr: Tuple[str, int], timeout: float = 2.0) -> bool:
        """Try direct UDP connectivity test."""
        sock = self.stun.get_local_socket()
        if not sock:
            return False
            
        try:
            loop = asyncio.get_event_loop()
            
            # Send test packet
            test_data = b'NEUROSHARD_TEST\x00'
            await loop.sock_sendto(sock, test_data, addr)
            
            # Wait for response
            sock.setblocking(False)
            data, _ = await asyncio.wait_for(
                loop.sock_recvfrom(sock, 64),
                timeout=timeout
            )
            
            return data.startswith(b'NEUROSHARD_')
            
        except:
            return False
            
    async def _try_relay(self, peer_id: str) -> Optional[Tuple[str, int]]:
        """Try connecting via relay server (placeholder for TURN)."""
        # TODO: Implement TURN relay
        return None
    
    @property
    def connection_success_rate(self) -> float:
        """Calculate overall connection success rate."""
        if self.total_attempts == 0:
            return 0.0
        total_success = self.direct_success + self.holepunch_success + self.relay_success
        return total_success / self.total_attempts
    
    def get_stats(self) -> Dict[str, Any]:
        """Get NAT traversal statistics."""
        return {
            "public_addr": f"{self.public_addr[0]}:{self.public_addr[1]}" if self.public_addr else None,
            "nat_type": self.nat_type.value,
            "total_attempts": self.total_attempts,
            "direct_success": self.direct_success,
            "holepunch_success": self.holepunch_success,
            "relay_success": self.relay_success,
            "total_failures": self.total_failures,
            "success_rate": self.connection_success_rate,
            "holepunch_rate": self.hole_puncher.success_rate,
        }
    
    def close(self):
        """Close all resources."""
        self.stun.close()


# Convenience function
async def discover_public_address() -> Optional[Tuple[str, int]]:
    """Quick helper to discover public address."""
    client = STUNClient()
    try:
        return await client.discover_public_address()
    finally:
        client.close()

