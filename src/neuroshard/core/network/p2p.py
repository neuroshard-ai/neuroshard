import requests
import random
import threading
import time
import hashlib
import logging
import os
import sqlite3
from typing import Dict, List, Optional, Any, Tuple
from urllib.parse import urlparse
from dataclasses import dataclass
from enum import Enum

# Convention: gRPC port = HTTP port + GRPC_PORT_OFFSET
GRPC_PORT_OFFSET = 1000

# DHT Imports
try:
    from neuroshard.core.network.dht import Node, RoutingTable
    from neuroshard.core.network.dht_protocol import DHTProtocol
    DHT_AVAILABLE = True
except ImportError:
    DHT_AVAILABLE = False


# =============================================================================
# NETWORK PHASES
# =============================================================================
# The network adapts its parameters based on size.
# Same core protocol scales from 1 node to millions.

class NetworkPhase(Enum):
    """Network phase based on size."""
    GENESIS = "genesis"     # 1 node - bootstrap mode
    MICRO = "micro"         # 2-4 nodes - minimal network
    SMALL = "small"         # 5-19 nodes - emerging network
    MEDIUM = "medium"       # 20-99 nodes - functional network
    GROWING = "growing"     # 100-499 nodes - scaling up
    LARGE = "large"         # 500-4999 nodes - mature network
    MASSIVE = "massive"     # 5000+ nodes - full scale


# Phase thresholds
PHASE_THRESHOLDS = {
    NetworkPhase.GENESIS: 1,
    NetworkPhase.MICRO: 4,
    NetworkPhase.SMALL: 19,
    NetworkPhase.MEDIUM: 99,
    NetworkPhase.GROWING: 499,
    NetworkPhase.LARGE: 4999,
    NetworkPhase.MASSIVE: float('inf'),
}


def get_network_phase(num_nodes: int) -> NetworkPhase:
    """
    Determine network phase based on node count.
    
    Args:
        num_nodes: Number of active nodes
        
    Returns:
        NetworkPhase enum value
    """
    if num_nodes <= 1:
        return NetworkPhase.GENESIS
    elif num_nodes <= 4:
        return NetworkPhase.MICRO
    elif num_nodes <= 19:
        return NetworkPhase.SMALL
    elif num_nodes <= 99:
        return NetworkPhase.MEDIUM
    elif num_nodes <= 499:
        return NetworkPhase.GROWING
    elif num_nodes <= 4999:
        return NetworkPhase.LARGE
    else:
        return NetworkPhase.MASSIVE


@dataclass
class AdaptiveConfig:
    """
    Configuration that adapts to network size.
    
    Parameters scale automatically as network grows to maintain
    security and efficiency at any scale.
    """
    # Quorum parameters
    min_quorum_size: int = 1
    max_quorum_size: int = 16
    
    # Cross-quorum sync (DiLoCo)
    sync_interval: int = 100       # Batches between syncs
    outer_lr: float = 0.7          # Outer learning rate
    
    # Verification
    challenge_rate: float = 0.0    # Probability of proof verification
    challenge_window: int = 600    # Challenge window in seconds
    
    # Heartbeat
    heartbeat_interval: int = 30   # Seconds between heartbeats
    stale_threshold: int = 4       # Missed heartbeats before stale
    
    # Replication
    min_replicas: int = 1          # Minimum layer replicas
    target_replicas: int = 3       # Target layer replicas


# Phase-specific configurations
ADAPTIVE_CONFIGS = {
    NetworkPhase.GENESIS: AdaptiveConfig(
        min_quorum_size=1,
        max_quorum_size=1,
        sync_interval=50,        # Fast sync for solo node
        outer_lr=0.9,            # Higher LR for fast convergence
        challenge_rate=0.0,      # No challenges - trust yourself
        min_replicas=1,
        target_replicas=1,
    ),
    NetworkPhase.MICRO: AdaptiveConfig(
        min_quorum_size=1,
        max_quorum_size=4,
        sync_interval=100,
        outer_lr=0.8,
        challenge_rate=0.0,      # Trust in small group
        min_replicas=1,
        target_replicas=2,
    ),
    NetworkPhase.SMALL: AdaptiveConfig(
        min_quorum_size=2,
        max_quorum_size=8,
        sync_interval=200,
        outer_lr=0.7,
        challenge_rate=0.01,     # 1% random verification
        min_replicas=2,
        target_replicas=3,
    ),
    NetworkPhase.MEDIUM: AdaptiveConfig(
        min_quorum_size=3,
        max_quorum_size=12,
        sync_interval=500,
        outer_lr=0.7,
        challenge_rate=0.05,     # 5% verification
        min_replicas=3,
        target_replicas=5,
    ),
    NetworkPhase.GROWING: AdaptiveConfig(
        min_quorum_size=3,
        max_quorum_size=16,
        sync_interval=1000,
        outer_lr=0.6,
        challenge_rate=0.05,
        min_replicas=3,
        target_replicas=10,
    ),
    NetworkPhase.LARGE: AdaptiveConfig(
        min_quorum_size=4,
        max_quorum_size=16,
        sync_interval=2000,
        outer_lr=0.5,
        challenge_rate=0.02,     # Lower rate, more nodes verify
        min_replicas=5,
        target_replicas=20,
    ),
    NetworkPhase.MASSIVE: AdaptiveConfig(
        min_quorum_size=5,
        max_quorum_size=16,
        sync_interval=5000,
        outer_lr=0.4,
        challenge_rate=0.01,
        min_replicas=10,
        target_replicas=50,
    ),
}


def get_adaptive_config(num_nodes: int) -> AdaptiveConfig:
    """
    Get adaptive configuration for current network size.
    
    Args:
        num_nodes: Number of active nodes
        
    Returns:
        AdaptiveConfig with appropriate parameters
    """
    phase = get_network_phase(num_nodes)
    return ADAPTIVE_CONFIGS.get(phase, ADAPTIVE_CONFIGS[NetworkPhase.MEDIUM])


class NetworkPhaseTracker:
    """
    Tracks network size and provides adaptive parameters.
    
    Updates periodically from DHT and adjusts parameters automatically.
    """
    
    def __init__(self, update_interval: float = 60.0):
        self.update_interval = update_interval
        self._num_nodes: int = 1
        self._phase: NetworkPhase = NetworkPhase.GENESIS
        self._config: AdaptiveConfig = get_adaptive_config(1)
        self._last_update: float = 0
        self._lock = threading.Lock()
    
    @property
    def num_nodes(self) -> int:
        return self._num_nodes
    
    @property
    def phase(self) -> NetworkPhase:
        return self._phase
    
    @property
    def config(self) -> AdaptiveConfig:
        return self._config
    
    def update(self, num_nodes: int) -> bool:
        """
        Update network size and recalculate parameters.
        
        Args:
            num_nodes: Current number of nodes
            
        Returns:
            True if phase changed
        """
        with self._lock:
            old_phase = self._phase
            self._num_nodes = num_nodes
            self._phase = get_network_phase(num_nodes)
            self._config = get_adaptive_config(num_nodes)
            self._last_update = time.time()
            
            if self._phase != old_phase:
                logger.info(f"Network phase changed: {old_phase.value} -> {self._phase.value} "
                           f"({num_nodes} nodes)")
                return True
            return False
    
    def should_update(self) -> bool:
        """Check if it's time for a periodic update."""
        return time.time() - self._last_update > self.update_interval
    
    def get_sync_interval(self) -> int:
        """Get current sync interval for DiLoCo."""
        return self._config.sync_interval
    
    def get_outer_lr(self) -> float:
        """Get current outer learning rate."""
        return self._config.outer_lr
    
    def get_challenge_rate(self) -> float:
        """Get current proof challenge rate."""
        return self._config.challenge_rate
    
    def get_min_quorum_size(self) -> int:
        """Get minimum quorum size."""
        return self._config.min_quorum_size
    
    def get_target_replicas(self) -> int:
        """Get target replicas per layer."""
        return self._config.target_replicas

# Ledger Imports
try:
    from neuroshard.core.economics.ledger import NEUROLedger, ProofType, PoNWProof
    LEDGER_AVAILABLE = True
except ImportError as e:
    LEDGER_AVAILABLE = False

logger = logging.getLogger(__name__)

if not LEDGER_AVAILABLE:
    logger.error(f"Ledger module import failed — PoNW proof generation disabled")

class P2PManager:
    def __init__(self, my_url: str, shard_range: str, tracker_url: str = "http://localhost:3000", node_token: Optional[str] = None, training_enabled: bool = True, observer_mode: bool = False):
        self.my_url = my_url
        self.shard_range = shard_range
        self.tracker_url = tracker_url
        self.node_token = node_token
        self.training_enabled = training_enabled  # Whether this node participates in training pipeline
        self.observer_mode = observer_mode  # Observer mode: sync ledger but don't generate proofs
        self.known_peers: Dict[str, dict] = {} # url -> info
        self._peer_failures: Dict[str, int] = {}  # url -> consecutive failure count
        self._PEER_FAILURE_THRESHOLD = 8  # Remove peer after this many consecutive failures
        self.running = True
        self._stop_event = threading.Event()  # For interruptible sleeps
        
        # Parse local shard range
        try:
            self.start_layer, self.end_layer = map(int, shard_range.split("-"))
        except (ValueError, AttributeError):
            # Default to -1 (unassigned) instead of 0-0 to prevent premature announcement
            self.start_layer, self.end_layer = -1, -1

        # Metrics
        self.current_tps = 0.0
        self.current_latency = 0.0
        
        # Reference to global state (injected by runner)
        self.state_ref = {}
        # Lock for thread-safe access to state_ref (read/written by gossip loop,
        # training loop, and API handlers concurrently)
        self._state_lock = threading.Lock()
        
        # Callback to get fresh training stats (injected by runner)
        # This bypasses the timing issue where state_ref isn't updated yet
        self.get_fresh_training_stats = None
        
        # Callback to notify AsyncTrainer when a proof is accepted
        # This triggers reset_proof_period() so model_hash_start is fresh
        self.on_proof_accepted = None
        
        # --- DHT & Decentralization Init ---
        self.dht = None
        self.routing_table = None
        self.ledger = None
        
        # Node ID will be set from ledger crypto (ECDSA-derived)
        # For DHT we need an integer ID, so we'll derive it from the token
        if self.node_token:
            # Use first 20 bytes of SHA256(token) as DHT node ID (160-bit)
            self.node_id = int(hashlib.sha256(self.node_token.encode()).hexdigest()[:40], 16)
        else:
            # Fallback to random ID
            self.node_id = int(hashlib.sha1(f"{my_url}{time.time()}".encode()).hexdigest(), 16)
        
        # The ledger node_id (32 hex chars from ECDSA public key) will be different
        # but deterministically linked to the same token
        self.ledger_node_id = None  # Set after ledger init

        if DHT_AVAILABLE:
            try:
                parsed = urlparse(my_url)
                ip = parsed.hostname or 'localhost'
                port = parsed.port or (443 if parsed.scheme == 'https' else 80)
                
                self.local_node = Node(self.node_id, ip, port)
                self.routing_table = RoutingTable(self.local_node)
                self.dht = DHTProtocol(self.local_node, self.routing_table, port)
                # Expose internal storage for gRPC inspection
                self.dht_storage = self.dht.storage 
                logger.info(f"DHT Initialized: {self.local_node}")
            except Exception as e:
                logger.error(f"Failed to init DHT: {e}")
                self.dht_storage = {} # Fallback

        if not hasattr(self, 'dht_storage'):
            self.dht_storage = {}

        if LEDGER_AVAILABLE:
            try:
                # Check for explicit path from environment (Docker/production)
                ledger_db_path = os.getenv("LEDGER_DB_PATH")
                
                if not ledger_db_path:
                    # Fallback to ~/.neuroshard/ directory for local development
                    neuroshard_dir = os.path.join(os.path.expanduser("~"), ".neuroshard")
                    os.makedirs(neuroshard_dir, exist_ok=True)
                    ledger_db_path = os.path.join(neuroshard_dir, f"ledger_{self.node_id}.db")
                else:
                    # Ensure directory exists for explicit path
                    os.makedirs(os.path.dirname(ledger_db_path), exist_ok=True)
                
                logger.info(f"Ledger DB path: {ledger_db_path}")
                
                self.ledger = NEUROLedger(
                    db_path=ledger_db_path,
                    node_token=self.node_token
                )
                # Get the ECDSA-derived node_id from ledger
                self.ledger_node_id = self.ledger.node_id
                logger.info(f"NEUROLedger Initialized with ECDSA node_id: {self.ledger_node_id[:16]}...")
                
                # Bootstrap balance from DHT for existing wallets
                # Fully trustless via ECDSA signature verification + Byzantine consensus
                self._bootstrap_balance_from_dht()
            except Exception as e:
                logger.error(f"Failed to init Ledger: {e}")
                self.ledger = None # Ensure explicit None on failure
        else:
            logger.info("Ledger Manager NOT available (dependencies missing or import failed)")

        # Reference to NeuroNode (set later via set_neuro_node)
        self.neuro_node = None
        
        # Start background tasks
        threading.Thread(target=self._announce_loop, daemon=True).start()
        
        # Only start gossip loop if NOT in observer mode
        # Observer nodes receive proofs via gRPC but don't generate their own
        if not self.observer_mode:
            threading.Thread(target=self._gossip_loop, daemon=True).start()
        else:
            logger.info("[OBSERVER] Observer mode enabled - not generating proofs")
        
        if self.ledger:
            threading.Thread(target=self._sync_stakes_loop, daemon=True).start()
    
    def set_neuro_node(self, neuro_node):
        """Set reference to NeuroNode for checkpoint announcements."""
        self.neuro_node = neuro_node
    
    def get_swarm_status(self) -> Dict[str, Any]:
        """
        Get swarm-related status from the connected node.
        
        Returns:
            Dict with swarm status including buffer fill rates, DiLoCo progress, etc.
        """
        if not self.neuro_node:
            return {"swarm_enabled": False, "error": "Node not connected"}
        
        # Check if node has swarm capabilities
        if hasattr(self.neuro_node, 'get_swarm_status'):
            return self.neuro_node.get_swarm_status()
        else:
            return {"swarm_enabled": False}
    
    def get_diloco_progress(self) -> Dict[str, Any]:
        """
        Get DiLoCo training progress from the connected node.
        
        Returns:
            Dict with inner step count, sync progress, etc.
        """
        if not self.neuro_node:
            return {"enabled": False, "error": "Node not connected"}
        
        if hasattr(self.neuro_node, 'get_diloco_progress'):
            return self.neuro_node.get_diloco_progress()
        else:
            return {"enabled": False}
    
    def _resolve_dht_endpoint(self, endpoint: str) -> str:
        """
        Resolve a DHT endpoint that may contain LAN info.
        
        DHT values can be in format "public_ip:port|local_ip:port".
        If the peer shares our public IP, use their local IP (same LAN).
        Otherwise, use just the public part.
        """
        try:
            from neuroshard.core.network.nat import resolve_peer_address
            my_public_ip = getattr(self, 'public_ip', None)
            if my_public_ip:
                return resolve_peer_address(endpoint, my_public_ip)
        except Exception:
            pass
        # Fallback: strip LAN info and use public part
        if "|" in endpoint:
            return endpoint.split("|", 1)[0]
        return endpoint
    
    def get_network_health(self) -> Dict[str, Any]:
        """
        Get overall network health metrics.
        
        Returns:
            Dict with peer count, average latency, routing stats, etc.
        """
        health = {
            "peer_count": len(self.known_peers),
            "avg_latency_ms": self.current_latency * 1000 if self.current_latency else 0,
            "current_tps": self.current_tps,
            "dht_available": self.dht is not None,
            "ledger_available": self.ledger is not None,
        }
        
        # Add swarm stats if available
        swarm_status = self.get_swarm_status()
        if swarm_status.get("swarm_enabled", False):
            health["swarm_enabled"] = True
            if "router" in swarm_status:
                health["swarm_peers"] = swarm_status["router"].get("peer_count", 0)
            if "heartbeat" in swarm_status:
                health["heartbeat_peers"] = swarm_status["heartbeat"].get("peer_count", 0)
        else:
            health["swarm_enabled"] = False
        
        return health
    
    def stop(self):
        """Stop the P2P manager and all background threads."""
        logger.info("Stopping P2P manager...")
        self.running = False
        
        # Signal stop event (for threads that check it)
        if hasattr(self, '_stop_event'):
            self._stop_event.set()
        
        # Close DHT if available
        if self.dht:
            try:
                # DHT doesn't have a stop method, but we can clear its state
                self.dht.storage.clear()
            except Exception:
                pass
        
        # Clear known peers
        self.known_peers.clear()
        
        logger.info("P2P manager stopped")
        
    def update_metrics(self, tps: float, latency: float):
        self.current_tps = tps
        self.current_latency = latency

    def _store_proof_in_dht(self, proof: 'PoNWProof', reward: float):
        """
        Store proof in DHT for decentralized balance sync.
        
        This enables new nodes to bootstrap their balance from DHT
        without relying on a central API.
        
        Args:
            proof: The PoNW proof to store
            reward: The reward amount credited for this proof
        """
        if not self.dht:
            return  # DHT not available
        
        try:
            from neuroshard.core.network.dht_proof_store import DHTProofStore, DHTProofRecord
            
            # Create DHT proof store (lazy init)
            if not hasattr(self, '_dht_proof_store'):
                self._dht_proof_store = DHTProofStore(self.dht)
            
            # Create proof record with ALL fields for verification
            # CRITICAL: Must include nonce, model_hash, request_id, and public_key for ECDSA verification
            proof_record = DHTProofRecord(
                node_id=proof.node_id,
                timestamp=proof.timestamp,
                proof_type=proof.proof_type.value if hasattr(proof.proof_type, 'value') else str(proof.proof_type),
                nonce=proof.nonce,  # 🔒 Required for canonical_payload
                reward=reward,
                signature=proof.signature,
                public_key=self.ledger.crypto.public_key_hex if self.ledger and self.ledger.crypto else "",  # 🔒 Required for verification
                uptime_seconds=proof.uptime_seconds,
                tokens_processed=proof.tokens_processed,
                training_batches=proof.training_batches,
                data_samples=proof.data_samples,
                model_hash=proof.model_hash,  # 🔒 Required for canonical_payload
                request_id=proof.request_id if proof.request_id else "",  # 🔒 Required for canonical_payload
                layers_held=proof.layers_held,
                has_embedding=proof.has_embedding,
                has_lm_head=proof.has_lm_head
            )
            
            # Get wallet_id (first 16 chars of node_id)
            wallet_id = proof.node_id[:16]
            
            # Store in DHT (async in background to not block)
            threading.Thread(
                target=self._dht_proof_store.store_proof_in_dht,
                args=(wallet_id, proof_record),
                daemon=True
            ).start()
            
        except Exception as e:
            logger.debug(f"DHT proof storage error (non-fatal): {e}")
    
    def _bootstrap_balance_from_dht(self):
        """
        Bootstrap balance from DHT (PRODUCTION-READY TRUSTLESS SYSTEM).
        
        This is called on startup to sync historical earnings when running
        the same wallet on a new machine.
        
        SECURITY ARCHITECTURE (Production-Grade):
        ┌─────────────────────────────────────────────────────────────┐
        │ DHT RETRIEVAL (FULLY TRUSTLESS)                            │
        │   1. Query DHT for historical proofs                       │
        │   2. Verify ECDSA signature on EACH proof                  │
        │   3. Cross-validate with 3+ independent DHT nodes          │
        │   4. Require Byzantine consensus                           │
        │   5. Credit only cryptographically verified proofs         │
        │   ✅ Fully decentralized                                   │
        │   ✅ Fully trustless                                       │
        │   ✅ Byzantine-resistant                                   │
        │   ✅ Production-ready                                      │
        └─────────────────────────────────────────────────────────────┘
        
        No Fallbacks:
        - If DHT has no proofs → Start from 0 (new wallet)
        - If network too small → Proofs stored when >=3 nodes
        - No trusted servers required
        
        Similar to: Bitcoin SPV, Ethereum Light Client
        
        Why no API fallback:
        - Would require trusting central server
        - Defeats purpose of decentralization
        - Opens security vulnerabilities
        - Not needed - local DB persists your own proofs
        """
        if not self.ledger or not self.node_token:
            return
        
        try:
            # Get wallet_id (first 16 chars of ECDSA node_id)
            wallet_id = self.ledger.node_id[:16]
            
            # Check if we already have a balance (skip bootstrap if we do)
            current_balance = self.ledger.get_balance()
            if current_balance > 0:
                logger.info(f"Local balance found: {current_balance:.4f} NEURO (skipping bootstrap)")
                return
            
            # ===================================================================
            # PHASE 1: DHT RETRIEVAL (TRUSTLESS)
            # ===================================================================
            dht_success = False
            if self.dht:
                try:
                    from neuroshard.core.network.dht_proof_store import DHTProofStore
                    
                    # Get network size for adaptive behavior
                    all_nodes = self.dht.routing_table.get_all_nodes() if self.dht else []
                    network_size = len(all_nodes) + 1  # +1 for self
                    
                    logger.info(f"[DHT BOOTSTRAP] Querying DHT for wallet {wallet_id}... (network: {network_size} nodes)")
                    
                    dht_store = DHTProofStore(self.dht)
                    
                    # Retrieve proofs from DHT with signature verification
                    verified_proofs, metadata = dht_store.retrieve_proofs_from_dht(
                        wallet_id=wallet_id,
                        max_proofs=100,
                        verify_signatures=True  # 🔒 TRUSTLESS
                    )
                    
                    if verified_proofs:
                        logger.info(f"[DHT BOOTSTRAP] Found {len(verified_proofs)} verified proofs in DHT "
                                   f"(total_reward={metadata.get('total_reward', 0):.6f} NEURO)")
                        
                        # Cross-validate with multiple DHT nodes for Byzantine resistance
                        # Uses adaptive validation (works with 2-node networks)
                        consensus, validation_data = dht_store.cross_validate_proofs(
                            wallet_id=wallet_id,
                            desired_validators=3  # Adapts to actual network size
                        )
                        
                        if consensus:
                            validators_count = validation_data.get('validators_queried', 0)
                            network_size = validation_data.get('network_size', 1)
                            
                            logger.info(f"[DHT BOOTSTRAP] ✅ Cross-validation PASSED "
                                       f"({validators_count} validators, network={network_size} nodes)")
                            
                            # Credit verified proofs to local ledger
                            total_credited = 0.0
                            for proof_record in verified_proofs:
                                # Each proof is ECDSA-verified, safe to credit
                                total_credited += proof_record.reward
                            
                            # Update local ledger with DHT data
                            with self.ledger.lock:
                                with sqlite3.connect(self.ledger.db_path) as conn:
                                    conn.execute("""
                                        INSERT OR REPLACE INTO balances 
                                        (node_id, balance, total_earned, total_spent, proof_count, last_proof_time)
                                        VALUES (?, ?, ?, 0.0, ?, ?)
                                    """, (
                                        self.ledger.node_id,
                                        total_credited,
                                        total_credited,
                                        len(verified_proofs),
                                        time.time()
                                    ))
                                    conn.commit()
                            
                            logger.info(f"[DHT BOOTSTRAP] ✅ Synced from DHT: {total_credited:.6f} NEURO")
                            logger.info(f"[DHT BOOTSTRAP]    {len(verified_proofs)} proofs verified via ECDSA signatures")
                            logger.info(f"[DHT BOOTSTRAP]    Network: {network_size} nodes, {validators_count} validators confirmed")
                            dht_success = True
                            return  # Success!
                        
                        else:
                            logger.warning(f"[DHT BOOTSTRAP] ⚠️ Cross-validation FAILED - nodes disagree")
                            logger.warning(f"[DHT BOOTSTRAP] Validation data: {validation_data}")
                            # Fall through to API
                    else:
                        logger.info(f"[DHT BOOTSTRAP] No proofs found in DHT (new wallet or network still syncing)")
                        # Fall through to API
                        
                except Exception as e:
                    logger.warning(f"[DHT BOOTSTRAP] DHT retrieval failed: {e}")
                    # Fall through to API
            
            # ===================================================================
            # NO API FALLBACK - PRODUCTION-READY TRUSTLESS SYSTEM
            # ===================================================================
            # If DHT doesn't have proofs, we start from zero and earn naturally.
            # This is the CORRECT behavior for a decentralized system.
            # 
            # Why no API fallback:
            # 1. API would be a trusted party (defeats trustless design)
            # 2. Creates centralization point
            # 3. Opens attack vector (malicious API can inflate balances)
            # 
            # Edge case handling:
            # - New wallet: Balance = 0 (correct)
            # - Existing wallet but DHT empty: Proofs are in local DB, will
            #   propagate to DHT as we earn. Other machines bootstrap when DHT
            #   has enough replicas (3+ nodes needed)
            # 
            # This is how Bitcoin/Ethereum work - fully decentralized.
            # ===================================================================
            
            if not dht_success:
                logger.info(f"[BOOTSTRAP] No proofs found in DHT for wallet {wallet_id[:8]}...")
                logger.info(f"[BOOTSTRAP] Starting with zero balance - will earn via PoNW")
                logger.info(f"[BOOTSTRAP] Future earnings will be stored in DHT for other machines")
                logger.info(f"[BALANCE] New wallet - starting from 0 NEURO. Start earning!")
            
        except Exception as e:
            logger.warning(f"[BOOTSTRAP] Error during DHT bootstrap: {e}")
            logger.info("[BOOTSTRAP] Starting with zero balance - future earnings via P2P")

    def _sync_stakes_loop(self):
        """
        P2P stake gossip loop.
        
        Periodically broadcasts our stake to peers so they can:
        1. Verify our PoNW claims have correct multipliers
        2. Maintain a network-wide view of stakes
        """
        while self.running:
            # Interruptible sleep - wakes up immediately on stop()
            if self._stop_event.wait(timeout=300):
                break  # Stop event was set
            
            if not self.ledger:
                continue
            
            try:
                # Get our current stake
                account_info = self.ledger.get_account_info()
                stake = account_info.get("stake", 0.0)
                stake_locked_until = account_info.get("stake_locked_until", 0.0)
                
                if stake <= 0:
                    continue  # Nothing to gossip
                
                # Gossip our stake to peers
                peers = list(self.known_peers.keys())
                if self.routing_table:
                    for n in self.routing_table.get_all_nodes():
                        peers.append(f"http://{n.ip}:{n.port}")
                
                if not peers:
                    continue
                
                # DYNAMIC FANOUT: Scale with network size
                # Formula: 2*sqrt(N) + 3, capped at 50 for stake gossip
                # Stakes are important for security - need higher coverage
                import math
                fanout = min(int(2 * math.sqrt(len(peers)) + 3), 50)
                targets = random.sample(peers, min(len(peers), fanout))
                logger.info(f"Stake gossip: Broadcasting {stake:.2f} NEURO to {len(targets)} peers")
                
                for target in targets:
                    threading.Thread(
                        target=self._send_stake_to_peer, 
                        args=(target, stake, stake_locked_until), 
                        daemon=True
                    ).start()
                    
            except Exception as e:
                logger.error(f"Stake gossip error: {e}")

    def _send_stake_to_peer(self, target_url: str, amount: float, locked_until: float):
        """Send stake update to a peer via gRPC."""
        from neuroshard.protos import neuroshard_pb2
        from neuroshard.protos import neuroshard_pb2_grpc
        from neuroshard.core.network.connection_pool import get_channel
        from urllib.parse import urlparse
        
        try:
            # Resolve LAN-format URLs before parsing
            clean_url = target_url
            if "|" in clean_url:
                if "://" in clean_url:
                    scheme, rest = clean_url.split("://", 1)
                    clean_url = f"{scheme}://{self._resolve_dht_endpoint(rest)}"
                else:
                    clean_url = self._resolve_dht_endpoint(clean_url)
            parsed = urlparse(clean_url)
            ip = parsed.hostname
            port = (parsed.port or 80) + 1000  # gRPC port
            
            channel = get_channel(f"{ip}:{port}")
            stub = neuroshard_pb2_grpc.NeuroShardServiceStub(channel)
            
            # Create stake gossip request using ECDSA node_id
            ledger_node_id = self.ledger.node_id
            payload = f"{ledger_node_id}:{amount}:{locked_until}"
            
            # SECURITY: Include public key for verification
            # This allows peers to verify our signature without prior knowledge
            public_key_hex = self.ledger.crypto.get_public_key_hex()
            
            req = neuroshard_pb2.GossipStakeRequest(
                node_id=ledger_node_id,
                amount=amount,
                locked_until=locked_until,
                timestamp=time.time(),
                signature=self.ledger._sign(payload),
                public_key=public_key_hex  # Required for verification
            )
            
            stub.GossipStake(req, timeout=3.0)
            logger.debug(f"Stake gossip sent to {ip}:{port}")
        except Exception as e:
            logger.debug(f"Stake gossip to {target_url} failed: {e}")

    def _announce_loop(self):
        # Immediate announce on startup (verbose for first time)
        self._announce_once(verbose=True)
        while self.running:
            # Re-announce every 60 seconds (DHT entries have ~5min TTL)
            # This is frequent enough for peer discovery but not spammy
            if self._stop_event.wait(timeout=60):
                break
            self._announce_once(verbose=False)  # Silent re-announce

    def broadcast_transaction(self, recipient_id: str, amount: float, signature: str, tx_hash: str):
        """Broadcast a transaction to the P2P network."""
        threading.Thread(target=self._gossip_transaction, args=(recipient_id, amount, signature, tx_hash), daemon=True).start()

    def _gossip_transaction(self, recipient_id: str, amount: float, signature: str, tx_hash: str):
        """Gossip transaction to peers."""
        from neuroshard.protos import neuroshard_pb2
        from neuroshard.protos import neuroshard_pb2_grpc
        from neuroshard.core.network.connection_pool import get_channel
        from urllib.parse import urlparse
        
        # 1. Gather Peers
        peers = list(self.known_peers.keys())
        if self.routing_table:
            for n in self.routing_table.get_all_nodes():
                peers.append(f"http://{n.ip}:{n.port}")
                
        if not peers: return

        # 2. Gossip to random subset (Epidemic Propagation)
        # DYNAMIC FANOUT: 2*sqrt(N) + 3, capped at 50 for transactions
        # Transactions need high coverage for consistency
        import math
        fanout = min(int(2 * math.sqrt(len(peers)) + 3), 50)
        targets = random.sample(peers, min(len(peers), fanout))
        
        req = neuroshard_pb2.GossipTransactionRequest(
            sender_id=str(self.node_id),
            recipient_id=recipient_id,
            amount=amount,
            timestamp=time.time(),
            signature=signature,
            tx_hash=tx_hash
        )
        
        logger.info(f"Broadcasting TX {tx_hash[:8]} to {len(targets)} peers...")
        
        for target_url in targets:
            try:
                # Resolve LAN-format URLs before parsing
                clean_url = target_url
                if "|" in clean_url:
                    if "://" in clean_url:
                        scheme, rest = clean_url.split("://", 1)
                        clean_url = f"{scheme}://{self._resolve_dht_endpoint(rest)}"
                    else:
                        clean_url = self._resolve_dht_endpoint(clean_url)
                parsed = urlparse(clean_url)
                ip = parsed.hostname
                port = (parsed.port or 80) + 1000
                
                channel = get_channel(f"{ip}:{port}")
                stub = neuroshard_pb2_grpc.NeuroShardServiceStub(channel)
                
                stub.GossipTransaction(req, timeout=3.0)
            except Exception as e:
                pass # Gossip is best effort

    def _gossip_loop(self):
        """Periodically create Proof of Neural Work and gossip to peers."""
        try:
            from neuroshard.protos import neuroshard_pb2
            from neuroshard.protos import neuroshard_pb2_grpc
            from neuroshard.core.network.connection_pool import get_channel
        except Exception as e:
            logger.error(f"[PoNW] Failed to import neuroshard.protos: {e}")
            return
        
        logger.info("[PoNW] Gossip loop started (will generate first proof in 60s)")
        
        loop_iteration = 0
        consecutive_errors = 0
        consecutive_skips = 0  # Track how many times we skip due to equal hashes

        while self.running:
            loop_iteration += 1
            
            try:
                # Interruptible sleep - wakes up immediately on stop()
                if self._stop_event.wait(timeout=60):
                    logger.info("[PoNW] Stop event received, exiting gossip loop")
                    break
                    
                if not self.ledger:
                    logger.info("[NODE] PoNW: No ledger available, skipping proof generation")
                    continue
                
                # Reset error counter on successful loop start
                consecutive_errors = 0
                # Get metrics from state — thread-safe snapshot + reset
                with self._state_lock:
                    tokens_processed = self.state_ref.get("token_count", 0)
                    training_batches = self.state_ref.get("training_batches", 0)
                    pending_request_ids = self.state_ref.get("pending_inference_requests", [])
                    # Reset counters after snapshot
                    self.state_ref["token_count"] = 0
                    self.state_ref["training_batches"] = 0
                    self.state_ref["pending_inference_requests"] = []
                    # Get node info for role multipliers
                    layers_held = len(self.state_ref.get("assigned_layers", []))
                    has_embedding = self.state_ref.get("has_embedding", False)
                    has_lm_head = self.state_ref.get("has_lm_head", False)
                    model_hash = self.state_ref.get("model_hash", "")
                    current_loss = self.state_ref.get("current_loss", None)
                
                # Determine proof type based on activity
                # IMPORTANT: Inference proofs REQUIRE a request_id (paid request)
                # Tokens processed without a request_id don't earn inference rewards
                if training_batches > 0:
                    proof_type = ProofType.TRAINING
                    # Note: tokens_processed during training doesn't count as inference
                    tokens_processed = 0  # Don't double-count training tokens
                elif tokens_processed > 0 and pending_request_ids:
                    # Only create inference proof if we have actual paid requests
                    proof_type = ProofType.INFERENCE
                else:
                    # Default to uptime - unpaid inference doesn't earn rewards
                    proof_type = ProofType.UPTIME
                    tokens_processed = 0  # Unpaid tokens don't count
                
                # =========================================================
                # FRESH STATS: Get latest values directly from trainer
                # =========================================================
                # This bypasses the timing issue where state_ref isn't updated yet
                if self.get_fresh_training_stats:
                    try:
                        fresh_stats = self.get_fresh_training_stats()
                        if fresh_stats:
                            # Use fresh loss if available
                            if fresh_stats.get("current_loss") is not None:
                                current_loss = fresh_stats["current_loss"]
                            # ONLY set model_hash_start if we don't have one yet
                            # After a proof, P2P sets model_hash_start = model_hash_end
                            # We don't want to overwrite that with stale value from trainer
                            if fresh_stats.get("model_hash_start") and not self.state_ref.get("model_hash_start"):
                                self.state_ref["model_hash_start"] = fresh_stats["model_hash_start"]
                            # Always get fresh model_hash_end (current weights after training)
                            if fresh_stats.get("model_hash_end"):
                                self.state_ref["model_hash_end"] = fresh_stats["model_hash_end"]
                                model_hash = fresh_stats["model_hash_end"]
                    except Exception as e:
                        logger.debug(f"[PoNW] Failed to get fresh stats: {e}")
                
                # Sanitize loss (must be a valid float for storage)
                if current_loss is not None:
                    import math
                    if math.isinf(current_loss) or math.isnan(current_loss):
                        current_loss = None
                
                # =========================================================
                # CHAINED PoNW: Get model state and gradient info
                # =========================================================
                # These prove actual training work was done:
                # - model_hash_start: Weights at start of proof period
                # - model_hash_end: Weights now (after training)
                # - gradient_commitment: Hash of gradient stats
                # - gradient_norm: L2 norm of gradients
                model_hash_start = self.state_ref.get("model_hash_start", "")
                model_hash_end = self.state_ref.get("model_hash_end", model_hash)
                
                # Skip training proof if no training happened (hashes are equal)
                # This happens if gossip loop fires before AsyncTrainer completes a round
                if proof_type == ProofType.TRAINING and model_hash_start and model_hash_end:
                    if model_hash_start == model_hash_end:
                        consecutive_skips += 1
                        if consecutive_skips <= 3:
                            logger.info(f"[PoNW] Skipping proof - training round not complete yet (skip {consecutive_skips}/3)")
                            continue
                        else:
                            # After 3 skips (~3 minutes), force reset the model hash
                            # This handles cases where the hash tracking gets out of sync
                            logger.warning(f"[PoNW] Resetting model hash after {consecutive_skips} skips - forcing re-sync")
                            self.state_ref["model_hash_start"] = ""
                            consecutive_skips = 0
                            # Still skip this iteration, but next one will start fresh
                            continue
                    else:
                        # Hashes differ - training happened, reset skip counter
                        consecutive_skips = 0
                gradient_commitment = self.state_ref.get("gradient_commitment", "")
                data_hash = self.state_ref.get("data_hash", "")
                gradient_norm = self.state_ref.get("gradient_norm", 0.0)
                
                # If model_hash_start not set, use current as start (first proof)
                if not model_hash_start and model_hash:
                    model_hash_start = model_hash
                
                # Compute gradient commitment if we have training data
                if proof_type == ProofType.TRAINING and current_loss and not gradient_commitment:
                    # Create commitment from available data
                    import hashlib as hl
                    commitment_data = f"{model_hash_start}:{model_hash_end}:{current_loss:.6f}:{training_batches}"
                    gradient_commitment = hl.sha256(commitment_data.encode()).hexdigest()[:32]
                
                # Create PoNW proof using new NEUROLedger API
                proof = self.ledger.create_proof(
                    proof_type=proof_type,
                    uptime_seconds=60,
                    tokens_processed=tokens_processed,
                    training_batches=training_batches,
                    layers_held=layers_held,
                    has_embedding=has_embedding,
                    has_lm_head=has_lm_head,
                    model_hash=model_hash,
                    current_loss=current_loss,
                    # Chained PoNW fields
                    model_hash_start=model_hash_start,
                    model_hash_end=model_hash_end,
                    gradient_commitment=gradient_commitment,
                    data_hash=data_hash,
                    gradient_norm=gradient_norm,
                )
                
                # Process proof locally (credit ourselves)
                success, reward, msg = self.ledger.process_proof(proof)
                
                if success:
                    if proof_type == ProofType.TRAINING:
                        logger.info(f"[NODE] Earned {reward:.6f} NEURO (training, {training_batches} batches in last 60s)")
                    elif proof_type == ProofType.INFERENCE:
                        logger.info(f"[NODE] Earned {reward:.6f} NEURO (inference, {tokens_processed} tokens in last 60s)")
                    else:
                        logger.info(f"[NODE] Earned {reward:.6f} NEURO (uptime, 60s)")
                    
                    # 🔥 NEW: Store proof in DHT for decentralized balance sync
                    self._store_proof_in_dht(proof, reward)
                    
                    # Reset model_hash_start for next proof period
                    # So next period tracks fresh weight changes
                    if proof_type == ProofType.TRAINING:
                        self.state_ref["model_hash_start"] = model_hash_end
                        # Notify AsyncTrainer to reset its internal tracking
                        if self.on_proof_accepted:
                            try:
                                self.on_proof_accepted()
                            except Exception as e:
                                logger.debug(f"on_proof_accepted callback error: {e}")
                else:
                    logger.info(f"[NODE] ❌ PoNW rejected: {msg}")

                # Gossip to random peers (deduplicated, excluding stale peers)
                peers_set = set(self.known_peers.keys())
                if self.routing_table:
                    for n in self.routing_table.get_all_nodes():
                        peers_set.add(f"http://{n.ip}:{n.port}")
                
                # Also look for proof receivers (observers) in DHT
                if self.dht:
                    try:
                        import json
                        proof_receivers = self.dht.lookup_value(
                            int(hashlib.sha1(b"proof_receiver").hexdigest(), 16)
                        )
                        if proof_receivers:
                            receivers = json.loads(proof_receivers)
                            for receiver in receivers:
                                # Resolve LAN-format endpoints (e.g. "1.2.3.4:8000|192.168.1.5:8000")
                                resolved = self._resolve_dht_endpoint(str(receiver))
                                peers_set.add(f"http://{resolved}")
                    except Exception:
                        pass  # Best effort
                
                # Remove self and stale peers (too many consecutive failures)
                # Discard both public and LAN forms of our own URL
                peers_set.discard(self.my_url)
                local_ip = getattr(self, 'local_ip', None)
                if local_ip:
                    from urllib.parse import urlparse
                    parsed_self = urlparse(self.my_url)
                    self_port = parsed_self.port or 80
                    peers_set.discard(f"http://{local_ip}:{self_port}")
                stale_peers = {url for url, count in self._peer_failures.items() 
                              if count >= self._PEER_FAILURE_THRESHOLD}
                peers_set -= stale_peers
                peers = list(peers_set)
                
                if not peers:
                    logger.info("PoNW: Solo mining (no peers to gossip)")
                else:
                    # DYNAMIC FANOUT: sqrt(N) + 3, capped at 30 for PoNW proofs
                    # Proofs need reasonable coverage for DHT consistency
                    import math
                    fanout = min(int(math.sqrt(len(peers)) + 3), 30)
                    targets = random.sample(peers, min(len(peers), fanout))
                    logger.info(f"PoNW: Gossiping to {len(targets)} peers: {targets[:5]}")
                    
                    for target in targets:
                        threading.Thread(target=self._send_proof_to_peer, args=(target, proof)).start()
                         
            except Exception as e:
                consecutive_errors += 1
                logger.error(f"[PoNW] Gossip loop error (attempt {consecutive_errors}): {e}")
                import traceback
                logger.debug(f"[PoNW] Traceback: {traceback.format_exc()}")
                
                # If too many consecutive errors, log warning but keep trying
                if consecutive_errors >= 5:
                    logger.warning(f"[PoNW] {consecutive_errors} consecutive errors - loop is struggling")
                
                # Small backoff on errors to avoid tight error loops
                time.sleep(min(consecutive_errors * 2, 30))
        
        # Log when loop exits (important for debugging)
        logger.info(f"[PoNW] Gossip loop exited after {loop_iteration} iterations (running={self.running})")

    def _send_proof_to_peer(self, target_url: str, proof: PoNWProof):
        """Send PoNW proof to a peer via gRPC."""
        from neuroshard.protos import neuroshard_pb2
        from neuroshard.protos import neuroshard_pb2_grpc
        from neuroshard.core.network.connection_pool import get_channel
        from urllib.parse import urlparse
        
        try:
            # Resolve LAN-format URLs (e.g. "http://1.2.3.4:8000|192.168.1.5:8000")
            # The pipe character breaks urlparse, so resolve before parsing
            clean_url = target_url
            if "|" in clean_url:
                # Extract scheme, resolve endpoint, reconstruct
                if "://" in clean_url:
                    scheme, rest = clean_url.split("://", 1)
                    resolved = self._resolve_dht_endpoint(rest)
                    clean_url = f"{scheme}://{resolved}"
                else:
                    clean_url = self._resolve_dht_endpoint(clean_url)
            
            parsed = urlparse(clean_url)
            ip = parsed.hostname
            # gRPC port = HTTP port + 1000
            port = (parsed.port or 80) + 1000
            
            logger.info(f"[GOSSIP] Sending proof to {ip}:{port}")
            
            channel = get_channel(f"{ip}:{port}")
            stub = neuroshard_pb2_grpc.NeuroShardServiceStub(channel)
            
            # Send FULL proof data for proper verification
            # CRITICAL: Include public key for trustless verification
            # CRITICAL: Include data_samples, model_hash, request_id for canonical_payload match
            req = neuroshard_pb2.GossipProofRequest(
                node_id=proof.node_id,
                timestamp=proof.timestamp,
                uptime=proof.uptime_seconds,
                signature=proof.signature,
                token_count=proof.tokens_processed,
                training_batches=proof.training_batches,
                layers_held=proof.layers_held,
                has_embedding=proof.has_embedding,
                has_lm_head=proof.has_lm_head,
                proof_type=proof.proof_type,
                nonce=proof.nonce,
                public_key=self.ledger.crypto.get_public_key_hex(),
                data_samples=proof.data_samples,
                model_hash=proof.model_hash,
                request_id=proof.request_id or "",
                current_loss=proof.current_loss if proof.current_loss is not None else 0.0,
                # Chained PoNW fields - CRITICAL for canonical_payload signature verification
                epoch_id=proof.epoch_id,
                model_hash_start=proof.model_hash_start or "",
                model_hash_end=proof.model_hash_end or "",
                gradient_commitment=proof.gradient_commitment or "",
                data_hash=proof.data_hash or "",
                gradient_norm=proof.gradient_norm if proof.gradient_norm is not None else 0.0
            )
            
            response = stub.GossipProof(req, timeout=10.0)
            logger.info(f"[GOSSIP] Proof sent to {ip}:{port}, accepted={response.accepted}")
            # Reset failure counter on success
            self._peer_failures.pop(target_url, None)
        except Exception as e:
            # Track consecutive failures per peer
            failures = self._peer_failures.get(target_url, 0) + 1
            self._peer_failures[target_url] = failures
            
            # Brief log (not the full gRPC debug string)
            error_detail = str(e).split("details = ")[1].split('"')[1] if "details = " in str(e) else str(e)[:80]
            
            if failures < self._PEER_FAILURE_THRESHOLD:
                logger.warning(f"[GOSSIP] Failed to reach {ip}:{port} ({failures}/{self._PEER_FAILURE_THRESHOLD}): {error_detail}")
            elif failures == self._PEER_FAILURE_THRESHOLD:
                # Log removal once, then suppress further logs for this peer
                self.known_peers.pop(target_url, None)
                logger.info(f"[GOSSIP] Removing unreachable peer {ip}:{port} — "
                           f"{failures} consecutive failures, will stop gossiping to it")
            # else: failures > threshold, already removed, silently skip (filtered in gossip loop)

    def _sync_with_new_peer(self, peer_url: str):
        """
        Sync state with a newly discovered peer.
        
        IMPORTANT: Historical proofs CANNOT be replayed because of the 
        PROOF_FRESHNESS_WINDOW (5 minutes). This is BY DESIGN - it prevents
        nodes from fabricating work while running solo.
        
        How balance sync works (like Bitcoin):
        ┌─────────────────────────────────────────────────────────────────┐
        │ LOCAL BALANCE = All proofs I generated (witnessed by me)        │
        │ NETWORK BALANCE = Proofs gossiped within 5 min (witnessed by N) │
        │                                                                 │
        │ If you run SOLO, only LOCAL balance increases.                  │
        │ NETWORK balance only increases when peers witness your work.    │
        │                                                                 │
        │ This is the SECURITY MODEL: No free NEURO from fabricated work! │
        └─────────────────────────────────────────────────────────────────┘
        
        What we DO sync:
        1. Current peer list (for gossip)
        2. DHT routing table (for lookups)
        3. Training state (for DiLoCo)
        
        What we DON'T sync (by design):
        - Historical proofs older than 5 minutes (prevents fraud)
        """
        # Log the connection for transparency
        logger.info(f"[SYNC] Connected to new peer: {peer_url}")
        
        # NOTE: Historical proof replay removed because:
        # 1. PROOF_FRESHNESS_WINDOW = 300s (5 min) - old proofs rejected
        # 2. This is correct security behavior (like Bitcoin confirmations)
        # 3. Solo-earned NEURO is LOCAL only - needs witnesses to be NETWORK-confirmed

    def _announce_once(self, verbose: bool = True):
        # 1. DHT Announce (Primary)
        # Announces all layers so peers can find us for pipeline routing
        if self.dht:
            # Observer mode: announce as proof receiver even without layers
            if self.observer_mode:
                try:
                    # Announce as a proof receiver so training nodes can find us
                    self.dht.announce("proof_receiver")
                    self.dht.announce("observer")
                    if verbose:
                        logger.info("[OBSERVER] DHT Announce: registered as proof receiver")
                except Exception as e:
                    logger.debug(f"[OBSERVER] DHT Announce error: {e}")
                # Don't return here - also announce to tracker for discoverability!
            
            # Skip DHT layer announcement if layers are not yet assigned
            # But still continue to tracker announce!
            elif self.start_layer < 0:
                if verbose:
                    logger.info("[P2P] Skipping DHT layer announcement (layers not assigned yet)")
                # Don't return here - also announce to tracker!
            
            # Normal training node with assigned layers
            else:
                try:
                    num_layers = self.end_layer - self.start_layer + 1
                    success_count = 0
                    
                    # Announce ALL layers we hold so peers can find us for any layer
                    # This is critical for distributed training pipeline routing!
                    for layer_id in range(self.start_layer, self.end_layer + 1):
                        try:
                            self.dht.announce(f"layer_{layer_id}")
                            success_count += 1
                        except (Exception,) as e:
                            logger.debug(f"DHT layer announce failed for layer_{layer_id}: {e}")
                    
                    # Also announce as proof receiver (all nodes can receive proofs)
                    try:
                        self.dht.announce("proof_receiver")
                    except (Exception,) as e:
                        logger.debug(f"DHT proof_receiver announce failed: {e}")
                    
                    # Log summary (only on first announce or if verbose)
                    if verbose and num_layers > 0:
                        logger.info(f"DHT Announce: {success_count}/{num_layers} layers announced (layers {self.start_layer}-{self.end_layer})")
                    
                    # Also announce checkpoint info for distributed training sync
                    if hasattr(self, 'neuro_node') and self.neuro_node:
                        checkpoint_info = self.neuro_node.get_checkpoint_info()
                        self.dht.announce(f"checkpoint_v{checkpoint_info['version']}")
                except Exception as e:
                    logger.debug(f"DHT Announce error: {e}")
        
        # Tracker Announce (for bootstrap discovery)
        # NOTE: This is NOT centralized control - the tracker is just a peer list.
        # New nodes query it to find existing peers, then DHT takes over.
        # This is how ALL P2P networks work (Bitcoin DNS seeds, IPFS bootstrap nodes, etc.)
        self._announce_to_tracker(verbose)

    def _announce_to_tracker(self, verbose: bool = False):
        """
        Announce this node to the tracker for peer discovery.
        
        This is NOT centralized control - the tracker is just a peer list:
        - New nodes query tracker to find existing peers
        - Once connected, DHT handles all discovery
        - The network works without tracker after bootstrap
        
        This is identical to how Bitcoin uses DNS seeds, IPFS uses bootstrap nodes, etc.
        """
        if not self.tracker_url:
            return
        
        try:
            import requests
            from urllib.parse import urlparse
            
            parsed = urlparse(self.my_url)
            ip = parsed.hostname
            port = parsed.port or 80
            
            # Ensure IP is valid (never None or empty string)
            if not ip or ip == "None":
                logger.debug("[TRACKER] Skipping announce - no valid IP address")
                return
            
            # Build shard range string
            if self.observer_mode:
                shard_range = "observer"
            elif self.start_layer >= 0:
                shard_range = f"{self.start_layer}-{self.end_layer}"
            else:
                shard_range = "unassigned"
            
            # Safely determine is_exit
            is_exit = False
            if not self.observer_mode:
                try:
                    if hasattr(self, 'neuro_node') and self.neuro_node:
                        if hasattr(self.neuro_node, 'model') and self.neuro_node.model:
                            is_exit = getattr(self.neuro_node.model, 'has_lm_head', False)
                except Exception:
                    pass
            
            # Announce to tracker
            announce_data = {
                "ip": ip,
                "port": port,
                "shard_range": shard_range,
                "is_entry": self.start_layer == 0 if self.start_layer >= 0 else False,
                "is_exit": is_exit,
                "tps": self.current_tps,
                "latency": self.current_latency,
                "node_token": self.node_token,
                "training_enabled": self.training_enabled and not self.observer_mode
            }
            
            resp = requests.post(
                f"{self.tracker_url}/announce",
                json=announce_data,
                timeout=5
            )
            
            if resp.status_code == 200 and verbose:
                data = resp.json()
                peer_count = data.get("peer_count", 0)
                logger.info(f"[TRACKER] Announced to tracker ({peer_count} peers in network)")
            elif resp.status_code != 200:
                logger.debug(f"[TRACKER] Announce failed: {resp.status_code}")
                
        except Exception as e:
            logger.debug(f"[TRACKER] Announce error: {e}")

    def get_next_hop(self, current_end_layer: int, session_id: Optional[str] = None, for_training: bool = False) -> Optional[str]:
        """
        Find a peer that can handle the next layer.
        
        Args:
            current_end_layer: The layer we need a peer for
            session_id: Optional session ID for sticky routing
            for_training: If True, only return peers with training enabled (for pipeline training)
        """
        candidates = []
        
        # Strategy 1: DHT Lookup (Scalable)
        # NOTE: DHT doesn't track training_enabled yet, so for training we'll filter below
        if self.dht:
            import json
            key_str = f"layer_{current_end_layer}"
            key = int(hashlib.sha1(key_str.encode()).hexdigest(), 16)
            
            # Use iterative lookup
            val = self.dht.lookup_value(key)
            if val:
                logger.debug(f"[ROUTING] DHT lookup for layer {current_end_layer}: found {val[:100] if len(val) > 100 else val}")
                try:
                    # Try parsing as list of peers
                    dht_candidates = json.loads(val)
                    if isinstance(dht_candidates, list):
                        for c in dht_candidates:
                            # Resolve LAN-format endpoints (e.g. "1.2.3.4:8000|192.168.1.5:8000")
                            resolved = self._resolve_dht_endpoint(str(c))
                            if not resolved.startswith("http"):
                                candidates.append(f"http://{resolved}")
                            else:
                                candidates.append(resolved)
                    else:
                         # Legacy single value
                         if not isinstance(dht_candidates, str):
                             dht_candidates = str(dht_candidates)
                         resolved = self._resolve_dht_endpoint(dht_candidates)
                         if not resolved.startswith("http"):
                             candidates.append(f"http://{resolved}")
                         else:
                             candidates.append(resolved)
                except (TypeError, ValueError, AttributeError) as e:
                    logger.debug(f"DHT candidate parsing fallback: {e}")
                    # Simple string fallback
                    resolved = self._resolve_dht_endpoint(str(val))
                    if not resolved.startswith("http"):
                        candidates.append(f"http://{resolved}")
                    else:
                        candidates.append(resolved)
            else:
                logger.debug(f"[ROUTING] DHT lookup for layer {current_end_layer}: no value found")

        # Strategy 2: Local Cache (from heartbeat-discovered peers)
        # Check if target layer is WITHIN the peer's range (not just at start)
        for url, info in self.known_peers.items():
            try:
                r = info.get("shard_range", "0-0")
                start, end = map(int, r.split("-"))
                # Peer can handle layer if it's within their range
                if start <= current_end_layer <= end:
                    # For training pipeline, only include peers with training enabled
                    if for_training:
                        training_enabled = info.get("training_enabled", True)
                        if not training_enabled:
                            logger.debug(f"[ROUTING] Skipping {url} for training (training_enabled=False)")
                            continue
                    if url not in candidates:  # Avoid duplicates
                        candidates.append(url)
                        logger.debug(f"[ROUTING] Local cache: {url} has layer {current_end_layer} (range {start}-{end})")
            except (ValueError, TypeError, KeyError):
                continue
        
        # For training, filter DHT candidates against known_peers training status
        if for_training and candidates:
            filtered = []
            for url in candidates:
                # Check if we know this peer's training status
                if url in self.known_peers:
                    if self.known_peers[url].get("training_enabled", True):
                        filtered.append(url)
                    else:
                        logger.debug(f"[ROUTING] Filtering out {url} (training_enabled=False)")
                else:
                    # Unknown peer - assume training enabled (will fail gracefully if not)
                    filtered.append(url)
            candidates = filtered
            
        if not candidates:
            # Log why we couldn't find a next hop
            peer_info = {url: f"{info.get('shard_range', '?')},train={info.get('training_enabled', '?')}" 
                        for url, info in self.known_peers.items()}
            logger.debug(f"[ROUTING] No next hop for layer {current_end_layer} (for_training={for_training}). Known peers: {peer_info}")
            return None
        
        if session_id:
            # Sticky routing
            candidates.sort()
            hash_val = int(hashlib.sha256(session_id.encode()).hexdigest(), 16)
            return candidates[hash_val % len(candidates)]
            
        return random.choice(candidates)
    
    def get_redundant_hop(self, current_end_layer: int, primary_hop: str) -> Optional[str]:
        candidates = []
        for url, info in self.known_peers.items():
            try:
                r = info.get("shard_range", "0-0")
                start, end = map(int, r.split("-"))
                # Peer can handle layer if it's within their range
                if start <= current_end_layer <= end and url != primary_hop:
                    candidates.append(url)
            except (ValueError, TypeError, KeyError):
                continue
            
        if not candidates: return None
        return random.choice(candidates)
        
    def get_sync_peers(self) -> List[str]:
        candidates = []
        for url, info in self.known_peers.items():
            if info.get("shard_range") == self.shard_range:
                candidates.append(url)
        return candidates
    
    # =========================================================================
    # EPOCH CHAIN GOSSIPING
    # =========================================================================
    
    def gossip_epoch(self, epoch: 'Epoch'):
        """
        Gossip a finalized epoch to peers.
        
        Epochs are gossiped with higher priority than regular proofs because
        they represent finalized state that all nodes should agree on.
        
        Args:
            epoch: The finalized epoch to gossip
        """
        from neuroshard.protos import neuroshard_pb2
        from neuroshard.protos import neuroshard_pb2_grpc
        from neuroshard.core.network.connection_pool import get_channel
        from urllib.parse import urlparse
        
        peers = list(self.known_peers.keys())
        
        if not peers:
            logger.info(f"[EPOCH] No peers to gossip epoch {epoch.epoch_id}")
            return
        
        # Higher fanout for epochs (important consensus data)
        import math
        fanout = min(int(math.sqrt(len(peers)) + 5), 50)
        targets = random.sample(peers, min(len(peers), fanout))
        
        logger.info(f"[EPOCH] Gossiping epoch {epoch.epoch_id} to {len(targets)} peers")
        
        for target in targets:
            threading.Thread(
                target=self._send_epoch_to_peer,
                args=(target, epoch),
                daemon=True
            ).start()
    
    def _send_epoch_to_peer(self, target_url: str, epoch: 'Epoch'):
        """Send epoch announcement to a peer."""
        from neuroshard.protos import neuroshard_pb2
        from neuroshard.protos import neuroshard_pb2_grpc
        from neuroshard.core.network.connection_pool import get_channel
        from urllib.parse import urlparse
        
        try:
            clean_url = target_url
            if "|" in clean_url:
                if "://" in clean_url:
                    scheme, rest = clean_url.split("://", 1)
                    clean_url = f"{scheme}://{self._resolve_dht_endpoint(rest)}"
                else:
                    clean_url = self._resolve_dht_endpoint(clean_url)
            parsed = urlparse(clean_url)
            ip = parsed.hostname
            port = (parsed.port or 80) + 1000  # gRPC port
            
            channel = get_channel(f"{ip}:{port}")
            stub = neuroshard_pb2_grpc.NeuroShardServiceStub(channel)
            
            req = neuroshard_pb2.EpochAnnouncementRequest(
                epoch_id=epoch.epoch_id,
                epoch_hash=epoch.epoch_hash,
                prev_epoch_hash=epoch.prev_epoch_hash,
                proposer_node_id=epoch.proposer_node_id,
                proposer_signature=epoch.proposer_signature,
                proof_count=epoch.proof_count,
                total_reward=epoch.total_reward,
                timestamp_start=epoch.timestamp_start,
                timestamp_end=epoch.timestamp_end,
                model_state_hash_start=epoch.model_state_hash_start,
                model_state_hash_end=epoch.model_state_hash_end,
                proofs_merkle_root=epoch.proofs_merkle_root,
                gradient_commitments_root=epoch.gradient_commitments_root,
            )
            
            stub.AnnounceEpoch(req, timeout=5.0)
            logger.debug(f"[EPOCH] Sent epoch {epoch.epoch_id} to {target_url}")
            
        except Exception as e:
            logger.debug(f"[EPOCH] Failed to send to {target_url}: {e}")
    
    def request_epoch_from_peer(self, peer_url: str, epoch_id: int) -> Optional['Epoch']:
        """
        Request a specific epoch from a peer.
        
        Used for syncing missing epochs when joining the network.
        
        Args:
            peer_url: Peer to request from
            epoch_id: Epoch ID to request
            
        Returns:
            The Epoch if successful, None otherwise
        """
        from neuroshard.protos import neuroshard_pb2
        from neuroshard.protos import neuroshard_pb2_grpc
        from neuroshard.core.network.connection_pool import get_channel
        from urllib.parse import urlparse
        
        try:
            clean_url = peer_url
            if "|" in clean_url:
                if "://" in clean_url:
                    scheme, rest = clean_url.split("://", 1)
                    clean_url = f"{scheme}://{self._resolve_dht_endpoint(rest)}"
                else:
                    clean_url = self._resolve_dht_endpoint(clean_url)
            parsed = urlparse(clean_url)
            ip = parsed.hostname
            port = (parsed.port or 80) + 1000
            
            channel = get_channel(f"{ip}:{port}")
            stub = neuroshard_pb2_grpc.NeuroShardServiceStub(channel)
            
            req = neuroshard_pb2.EpochRequest(epoch_id=epoch_id)
            resp = stub.GetEpoch(req, timeout=10.0)
            
            if resp.found:
                from neuroshard.core.consensus.epoch import Epoch
                epoch = Epoch(
                    epoch_id=resp.epoch_id,
                    epoch_hash=resp.epoch_hash,
                    prev_epoch_hash=resp.prev_epoch_hash,
                    timestamp_start=resp.timestamp_start,
                    timestamp_end=resp.timestamp_end,
                    model_state_hash_start=resp.model_state_hash_start,
                    model_state_hash_end=resp.model_state_hash_end,
                    proofs_merkle_root=resp.proofs_merkle_root,
                    proof_count=resp.proof_count,
                    total_reward=resp.total_reward,
                    total_batches=resp.total_batches,
                    average_loss=resp.average_loss,
                    gradient_commitments_root=resp.gradient_commitments_root,
                    proposer_node_id=resp.proposer_node_id,
                    proposer_signature=resp.proposer_signature,
                )
                
                # Verify epoch hash
                if epoch.compute_epoch_hash() == epoch.epoch_hash:
                    return epoch
                else:
                    logger.warning(f"[EPOCH] Invalid epoch hash from {peer_url}")
                    
            return None
            
        except Exception as e:
            logger.debug(f"[EPOCH] Failed to get epoch {epoch_id} from {peer_url}: {e}")
            return None
    
    def sync_epoch_chain(self, target_epoch_id: Optional[int] = None) -> Tuple[int, str]:
        """
        Sync epoch chain from peers.
        
        Args:
            target_epoch_id: Target epoch to sync to (None = get latest from peers)
            
        Returns:
            (synced_count, status_message)
        """
        if not self.ledger:
            return 0, "No ledger available"
        
        # Get our latest epoch
        local_latest = self._get_local_latest_epoch()
        
        # Get target from peers if not specified
        if target_epoch_id is None:
            target_epoch_id = self._get_network_latest_epoch()
            if target_epoch_id is None:
                return 0, "Could not determine network latest epoch"
        
        if target_epoch_id <= local_latest:
            return 0, "Already up to date"
        
        # Fetch missing epochs from peers
        synced = 0
        peers = list(self.known_peers.keys())
        
        for epoch_id in range(local_latest + 1, target_epoch_id + 1):
            # Try each peer until we get the epoch
            for peer in random.sample(peers, min(len(peers), 3)):
                epoch = self.request_epoch_from_peer(peer, epoch_id)
                if epoch:
                    # Store in local ledger
                    self._store_epoch_locally(epoch)
                    synced += 1
                    break
            else:
                return synced, f"Failed to sync epoch {epoch_id}"
        
        return synced, f"Synced {synced} epochs"
    
    def _get_local_latest_epoch(self) -> int:
        """Get latest epoch ID from local ledger."""
        if not self.ledger:
            return -1
        
        try:
            import sqlite3
            with sqlite3.connect(self.ledger.db_path) as conn:
                result = conn.execute(
                    "SELECT MAX(epoch_id) FROM epochs WHERE finalized = 1"
                ).fetchone()
                return result[0] if result and result[0] else -1
        except Exception as e:
            logger.debug(f"Failed to get local latest epoch: {e}")
            return -1
    
    def _get_network_latest_epoch(self) -> Optional[int]:
        """Query peers for latest epoch ID."""
        peers = list(self.known_peers.keys())[:5]  # Sample 5 peers
        
        latest_ids = []
        for peer in peers:
            try:
                from neuroshard.protos import neuroshard_pb2
                from neuroshard.protos import neuroshard_pb2_grpc
                from neuroshard.core.network.connection_pool import get_channel
                from urllib.parse import urlparse
                
                clean_peer = peer
                if "|" in clean_peer:
                    if "://" in clean_peer:
                        scheme, rest = clean_peer.split("://", 1)
                        clean_peer = f"{scheme}://{self._resolve_dht_endpoint(rest)}"
                    else:
                        clean_peer = self._resolve_dht_endpoint(clean_peer)
                parsed = urlparse(clean_peer)
                channel = get_channel(f"{parsed.hostname}:{(parsed.port or 80) + 1000}")
                stub = neuroshard_pb2_grpc.NeuroShardServiceStub(channel)
                
                resp = stub.GetLatestEpoch(neuroshard_pb2.Empty(), timeout=3.0)
                if resp.epoch_id > 0:
                    latest_ids.append(resp.epoch_id)
            except (ConnectionError, TimeoutError, Exception) as e:
                logger.debug(f"Failed to get epoch from peer {peer}: {e}")
        
        if latest_ids:
            # Return the most commonly reported latest epoch
            from collections import Counter
            return Counter(latest_ids).most_common(1)[0][0]
        
        return None
    
    def _store_epoch_locally(self, epoch: 'Epoch'):
        """Store epoch in local ledger database."""
        if not self.ledger:
            return
        
        try:
            import sqlite3
            with sqlite3.connect(self.ledger.db_path) as conn:
                conn.execute("""
                    INSERT OR REPLACE INTO epochs 
                    (epoch_id, prev_epoch_hash, timestamp_start, timestamp_end,
                     model_state_hash_start, model_state_hash_end, proofs_merkle_root,
                     proof_count, total_reward, total_batches, average_loss,
                     gradient_commitments_root, proposer_node_id, proposer_signature,
                     epoch_hash, finalized, created_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 1, ?)
                """, (
                    epoch.epoch_id, epoch.prev_epoch_hash,
                    epoch.timestamp_start, epoch.timestamp_end,
                    epoch.model_state_hash_start, epoch.model_state_hash_end,
                    epoch.proofs_merkle_root, epoch.proof_count, epoch.total_reward,
                    epoch.total_batches, epoch.average_loss,
                    epoch.gradient_commitments_root, epoch.proposer_node_id,
                    epoch.proposer_signature, epoch.epoch_hash, time.time()
                ))
                logger.info(f"[EPOCH] Stored epoch {epoch.epoch_id} locally")
        except Exception as e:
            logger.error(f"[EPOCH] Failed to store epoch {epoch.epoch_id}: {e}")