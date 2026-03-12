import grpc
import threading
import time
import logging
import json
import hashlib
from concurrent import futures
from typing import List, Optional, Dict, Any
from dataclasses import dataclass, asdict, field

from neuroshard.protos import neuroshard_pb2
from neuroshard.protos import neuroshard_pb2_grpc
from neuroshard.core.network.dht import Node, RoutingTable, ID_BITS
from neuroshard.core.network.dht_service import node_to_proto, proto_to_node
from neuroshard.core.network.connection_pool import get_channel

logger = logging.getLogger("DHT")


# =============================================================================
# NETWORK STATE - Versioned Network Configuration
# =============================================================================

@dataclass
class NetworkState:
    """
    Network-wide state stored in DHT at key "network:state".
    
    This contains versioned information about the current model architecture
    and network configuration. All nodes should reference this to ensure
    they're compatible with the current network state.
    
    Version fields:
    - arch_version: Increments when model architecture changes (layers added)
    - vocab_version: Increments when vocabulary expands (via governance)
    """
    # Version tracking
    arch_version: int = 1           # Architecture version (increments on layer growth)
    vocab_version: int = 1          # Vocabulary version (increments on vocab expansion)
    
    # Model architecture
    num_layers: int = 11            # Current number of layers
    hidden_dim: int = 512           # Hidden dimension
    vocab_size: int = 32000         # Current vocabulary size
    
    # Network stats
    total_nodes: int = 0            # Total active nodes
    total_memory_mb: int = 0        # Total network memory
    active_quorums: int = 0         # Number of active quorums
    current_step: int = 0           # Global training step
    
    # Timestamps
    last_updated: float = field(default_factory=time.time)
    last_arch_upgrade: float = 0.0  # When architecture was last upgraded
    
    def to_json(self) -> str:
        """Serialize to JSON string for DHT storage."""
        return json.dumps(asdict(self))
    
    @classmethod
    def from_json(cls, json_str: str) -> 'NetworkState':
        """Deserialize from JSON string."""
        data = json.loads(json_str)
        return cls(**data)
    
    def is_compatible_with(self, other: 'NetworkState') -> bool:
        """
        Check if this state is compatible with another state.
        
        States are compatible if they have the same arch_version and vocab_version.
        Different versions require synchronization before nodes can communicate.
        """
        return (self.arch_version == other.arch_version and 
                self.vocab_version == other.vocab_version)


# DHT keys for network state
DHT_KEY_NETWORK_STATE = "network:state"
DHT_KEY_ARCH_VERSION = "network:arch_version"
DHT_KEY_VOCAB_VERSION = "network:vocab_version"


class DHTProtocol:
    def __init__(self, local_node: Node, routing_table: RoutingTable, port: int,
                 data_dir: str = None):
        self.local_node = local_node
        self.routing_table = routing_table
        self.port = port
        
        # Persistent DHT storage backed by shelve (survives restarts)
        # Falls back to in-memory dict if data_dir is not provided or shelve fails
        self._shelve_db = None
        if data_dir:
            try:
                import os
                import shelve
                os.makedirs(data_dir, exist_ok=True)
                db_path = os.path.join(data_dir, 'dht_storage')
                self._shelve_db = shelve.open(db_path, writeback=True)
                self.storage = self._shelve_db
                logger.info(f"DHT storage persisted to {db_path}")
            except Exception as e:
                logger.warning(f"Failed to open persistent DHT storage: {e}, using in-memory")
                self.storage = {}
        else:
            self.storage = {}
        
        # We don't manage the server here anymore, it's mixed into the main gRPC server
        # Rate limiting for "no peers" messages (log max once per 60 seconds)
        self._last_no_peers_log = {}
    
    def close(self):
        """Close persistent storage if open."""
        if self._shelve_db is not None:
            try:
                self._shelve_db.close()
                logger.info("DHT persistent storage closed")
            except Exception as e:
                logger.debug(f"Error closing DHT storage: {e}")
    
    def sync_storage(self):
        """Flush pending writes to disk."""
        if self._shelve_db is not None:
            try:
                self._shelve_db.sync()
            except Exception as e:
                logger.debug(f"DHT storage sync error: {e}")

    def _get_stub(self, target: Node):
        # gRPC port assumption: http port + 1000
        # If target.port is already the gRPC port, we use it directly.
        # Our convention in p2p.py was: p_port = parsed.port or 80.
        # And in grpc_server.py: grpc_port = port + 1000.
        # So the node.port stored in DHT is likely the HTTP port.
        
        # Let's assume Node stores HTTP port, so we add 1000.
        # We should standardize this. For now, consistent with existing logic.
        target_addr = f"{target.ip}:{target.port + 1000}"
        channel = get_channel(target_addr)
        return neuroshard_pb2_grpc.NeuroShardServiceStub(channel)

    def ping(self, target: Node) -> bool:
        """Send PING to target node."""
        try:
            target_addr = f"{target.ip}:{target.port + 1000}"
            logger.debug(f"DHT Ping: {self.local_node.ip}:{self.local_node.port} -> {target_addr}")
            stub = self._get_stub(target)
            req = neuroshard_pb2.DHTPingRequest(
                sender=node_to_proto(self.local_node)
            )
            resp = stub.DHTPing(req, timeout=5.0)  # Increased timeout for cross-network
            # Update routing table with responder
            self.routing_table.add_contact(proto_to_node(resp.responder))
            logger.debug(f"DHT Ping successful: {target_addr} responded as {resp.responder.ip}:{resp.responder.port}")
            return True
        except grpc.RpcError as e:
            # Log at info level for important connectivity issues
            logger.info(f"DHT Ping failed to {target.ip}:{target.port + 1000}: {e.code() if hasattr(e, 'code') else e}")
            return False

    def find_node(self, target: Node, search_id: int) -> List[Node]:
        """Ask target for nodes closest to search_id."""
        try:
            stub = self._get_stub(target)
            req = neuroshard_pb2.DHTFindNodeRequest(
                sender=node_to_proto(self.local_node),
                target_id=search_id.to_bytes(20, byteorder='big')
            )
            resp = stub.DHTFindNode(req, timeout=5.0)
            
            self.routing_table.add_contact(proto_to_node(resp.responder))
            
            nodes = [proto_to_node(n) for n in resp.nodes]
            return nodes
        except grpc.RpcError:
            return []

    def store(self, target: Node, key: int, value: str) -> bool:
        """Ask target to store key=value."""
        try:
            stub = self._get_stub(target)
            req = neuroshard_pb2.DHTStoreRequest(
                sender=node_to_proto(self.local_node),
                key=key.to_bytes(20, byteorder='big'),
                value=value
            )
            resp = stub.DHTStore(req, timeout=5.0)
            
            self.routing_table.add_contact(proto_to_node(resp.responder))
            return resp.success
        except grpc.RpcError as e:
            logger.debug(f"DHT store gRPC error to {target.ip}:{target.port}: {e.code()}")
            return False
        
    def find_value(self, target: Node, key: int):
        """Ask target for value at key."""
        # Returns (value: str | None, nodes: List[Node])
        try:
            stub = self._get_stub(target)
            req = neuroshard_pb2.DHTFindValueRequest(
                sender=node_to_proto(self.local_node),
                key=key.to_bytes(20, byteorder='big')
            )
            resp = stub.DHTFindValue(req, timeout=5.0)
            
            self.routing_table.add_contact(proto_to_node(resp.responder))
            
            if resp.found:
                return (resp.value, [])
            else:
                nodes = [proto_to_node(n) for n in resp.nodes]
                return (None, nodes)
        except grpc.RpcError:
            return (None, [])

    # --- High Level Lookup Algorithm (Iterative) ---
    
    def lookup_node(self, target_id: int) -> List[Node]:
        """
        Standard Kademlia Lookup
        """
        shortlist = self.routing_table.find_closest(target_id)
        if not shortlist:
            return []
            
        visited = set()
        visited.add(self.local_node.id) # Don't query self
        
        active_queries = 0
        alpha = 3
        
        # Iterative lookup
        converged = False
        while not converged:
            candidates = [n for n in shortlist if n.id not in visited][:alpha]
            if not candidates:
                break
                
            results_found = False
            for node in candidates:
                visited.add(node.id)
                
                new_nodes = self.find_node(node, target_id)
                if new_nodes:
                    results_found = True
                    for n in new_nodes:
                        if n.id != self.local_node.id:
                            self.routing_table.add_contact(n)
                            # Add to shortlist if not already there
                            if n not in shortlist:
                                shortlist.append(n)
            
            # Re-sort
            shortlist.sort(key=lambda n: n.id ^ target_id)
            # K-bucket size limit
            shortlist = shortlist[:20]
            
            if not results_found:
                converged = True
            
        return shortlist

    def lookup_value(self, key: int) -> Optional[str]:
        """
        Find a value in the DHT.
        Returns the value if found, None otherwise.
        
        IMPORTANT: Check local storage FIRST, then query peers.
        This is essential for small networks where nodes might store data
        that should be discoverable by peers who ping us directly.
        """
        # CRITICAL: Check local storage first!
        # In small networks, the value might only be stored locally.
        # When a peer pings us and adds us to their routing table,
        # their next lookup should find our local data.
        if key in self.storage:
            return self.storage[key]
        
        # Query peers for the value
        shortlist = self.routing_table.find_closest(key)
        if not shortlist:
            return None
            
        visited = set()
        visited.add(self.local_node.id)
        
        converged = False
        while not converged:
            candidates = [n for n in shortlist if n.id not in visited][:3]
            if not candidates:
                break
                
            results_found = False
            for node in candidates:
                visited.add(node.id)
                
                val, nodes = self.find_value(node, key)
                if val:
                    return val # Found it!
                    
                if nodes:
                    results_found = True
                    for n in nodes:
                        if n.id != self.local_node.id:
                            self.routing_table.add_contact(n)
                            if n not in shortlist:
                                shortlist.append(n)
                                
            shortlist.sort(key=lambda n: n.id ^ key)
            shortlist = shortlist[:20]
            
            if not results_found:
                converged = True
                
        return None

    def announce(self, key_string: str):
        """
        Announce a string key (e.g. "layer_0") to the DHT.
        The key is hashed to find the location.
        Value is our connection info in format: "public_ip:port|local_ip:port"
        (backward compatible — peers without LAN detection ignore the |local part).
        """
        import hashlib
        import json
        # Hash the key string to get the 160-bit Key ID
        key_id = int(hashlib.sha1(key_string.encode()).hexdigest(), 16)
        # Include both public and local IPs so same-LAN peers can use local address
        local_ip = getattr(self, '_local_lan_ip', None)
        if local_ip and local_ip != self.local_node.ip:
            value = f"{self.local_node.ip}:{self.local_node.port}|{local_ip}:{self.local_node.port}"
        else:
            value = f"{self.local_node.ip}:{self.local_node.port}"
        
        # CRITICAL: ALWAYS store locally so other nodes can find us!
        # When we're a solo node, we ARE the closest node to any key.
        # Without local storage, DHT lookups will fail.
        try:
            # Store as list to support multiple holders
            existing = self.storage.get(key_id)
            if existing:
                holders = json.loads(existing)
                if value not in holders:
                    holders.append(value)
            else:
                holders = [value]
            self.storage[key_id] = json.dumps(holders)
        except Exception as e:
            logger.debug(f"Local DHT storage failed: {e}")
        
        # Find K closest nodes to the KEY ID (for replication to other nodes)
        nodes = self.lookup_node(key_id)
        
        store_count = 0
        for node in nodes:
            try:
                if self.store(node, key_id, value):
                    store_count += 1
            except: pass
        
        if store_count > 0:
            # Rate limit success logs to avoid spam (heartbeats happen every 10s)
            now = time.time()
            last_success = getattr(self, '_last_announce_success', {}).get(key_string, 0)
            if not hasattr(self, '_last_announce_success'):
                self._last_announce_success = {}
            
            # Only log first announce and then every 5 minutes
            if last_success == 0 or (now - last_success) > 300:
                logger.info(f"DHT Announce success: Stored '{key_string}' on {store_count} nodes.")
                self._last_announce_success[key_string] = now
        else:
            # Rate limit all DHT announce logs to avoid spam
            now = time.time()
            last_log = self._last_no_peers_log.get(key_string, 0)
            if now - last_log > 60:  # Rate limit: once per minute per key
                if len(nodes) == 0:
                    # No peers found - expected when you're the first/only node
                    logger.debug(f"DHT Announce: No peers found to store '{key_string}' (this is normal when you're the first node).")
                elif len(nodes) <= 3:
                    # Few nodes found and store failed - common in small networks or Docker
                    logger.debug(f"DHT Announce: Store failed for '{key_string}' (found {len(nodes)} nodes). This is normal in small networks.")
                else:
                    # Found many nodes but all failed - this might be a real problem
                    logger.warning(f"DHT Announce failed: Could not store '{key_string}' (found {len(nodes)} nodes but store failed).")
                self._last_no_peers_log[key_string] = now

    def lookup_key(self, key_string: str) -> Optional[str]:
        """
        Look up a value by string key in the DHT.
        
        This is a convenience method that hashes the key string and calls lookup_value.
        Used for finding nodes that have announced specific roles (e.g., "proof_receiver").
        
        Args:
            key_string: The string key to look up (e.g., "proof_receiver", "layer_0")
            
        Returns:
            The value if found, None otherwise
        """
        key_id = int(hashlib.sha1(key_string.encode()).hexdigest(), 16)
        return self.lookup_value(key_id)

    # =========================================================================
    # NETWORK STATE MANAGEMENT
    # =========================================================================
    
    def get_network_state(self) -> Optional[NetworkState]:
        """
        Get the current network state from DHT.
        
        Returns:
            NetworkState if found, None otherwise
        """
        try:
            key_id = int(hashlib.sha1(DHT_KEY_NETWORK_STATE.encode()).hexdigest(), 16)
            
            # Check local storage first
            if key_id in self.storage:
                return NetworkState.from_json(self.storage[key_id])
            
            # Try DHT lookup
            value = self.lookup_value(key_id)
            if value:
                return NetworkState.from_json(value)
            
            return None
            
        except Exception as e:
            logger.warning(f"Failed to get network state: {e}")
            return None
    
    def set_network_state(self, state: NetworkState) -> bool:
        """
        Store network state in DHT.
        
        This should be called when:
        - Network architecture changes
        - Vocabulary expands
        - Periodically to update node counts
        
        Args:
            state: NetworkState to store
            
        Returns:
            True if stored successfully
        """
        try:
            state.last_updated = time.time()
            json_value = state.to_json()
            
            key_id = int(hashlib.sha1(DHT_KEY_NETWORK_STATE.encode()).hexdigest(), 16)
            
            # Store locally
            self.storage[key_id] = json_value
            
            # Replicate to peers
            nodes = self.lookup_node(key_id)
            store_count = 0
            for node in nodes:
                try:
                    if self.store(node, key_id, json_value):
                        store_count += 1
                except:
                    pass
            
            logger.debug(f"Network state stored: arch_v{state.arch_version}, vocab_v{state.vocab_version}, replicated to {store_count} nodes")
            return True
            
        except Exception as e:
            logger.error(f"Failed to set network state: {e}")
            return False
    
    def get_or_create_network_state(self) -> NetworkState:
        """
        Get existing network state or create default if none exists.
        
        This is the main entry point for nodes to get the current state.
        If no state exists (first node), creates and stores a default.
        
        Returns:
            NetworkState (existing or newly created)
        """
        state = self.get_network_state()
        if state:
            return state
        
        # Create default state (genesis)
        state = NetworkState(
            arch_version=1,
            vocab_version=1,
            num_layers=11,
            hidden_dim=512,
            vocab_size=32000,
            total_nodes=1,
            total_memory_mb=0,
            active_quorums=0,
            current_step=0,
        )
        
        self.set_network_state(state)
        logger.info("Created genesis network state")
        return state
    
    def increment_arch_version(self, new_layers: int, new_hidden_dim: int) -> NetworkState:
        """
        Increment architecture version when model grows.
        
        Called when layer growth system triggers an upgrade.
        
        Args:
            new_layers: New number of layers
            new_hidden_dim: New hidden dimension
            
        Returns:
            Updated NetworkState
        """
        state = self.get_or_create_network_state()
        state.arch_version += 1
        state.num_layers = new_layers
        state.hidden_dim = new_hidden_dim
        state.last_arch_upgrade = time.time()
        
        self.set_network_state(state)
        logger.info(f"Architecture upgraded: v{state.arch_version} ({state.num_layers} layers, {state.hidden_dim} hidden)")
        return state
    
    def increment_vocab_version(self, new_vocab_size: int) -> NetworkState:
        """
        Increment vocabulary version when vocab expands via governance.
        
        Args:
            new_vocab_size: New vocabulary size
            
        Returns:
            Updated NetworkState
        """
        state = self.get_or_create_network_state()
        state.vocab_version += 1
        state.vocab_size = new_vocab_size
        
        self.set_network_state(state)
        logger.info(f"Vocabulary upgraded: v{state.vocab_version} ({state.vocab_size} tokens)")
        return state
    
    def update_network_stats(self, total_nodes: int, total_memory_mb: int, active_quorums: int, current_step: int) -> NetworkState:
        """
        Update network statistics without changing versions.
        
        Called periodically to keep network stats current.
        
        Args:
            total_nodes: Current node count
            total_memory_mb: Total network memory
            active_quorums: Active quorum count
            current_step: Global training step
            
        Returns:
            Updated NetworkState
        """
        state = self.get_or_create_network_state()
        state.total_nodes = total_nodes
        state.total_memory_mb = total_memory_mb
        state.active_quorums = active_quorums
        state.current_step = current_step
        
        self.set_network_state(state)
        return state
