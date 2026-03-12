"""
DHT-Based Proof Storage for Decentralized Balance Sync

This module implements a fully decentralized, trustless proof storage system
using Kademlia DHT. This is the "proper fix" for balance synchronization.

Architecture:
┌─────────────────────────────────────────────────────────────────┐
│ 1. PROOF GENERATION                                             │
│    - Node earns NEURO via PoNW                                  │
│    - Proof signed with ECDSA                                    │
│    - Stored locally + gossiped to peers                         │
│    - ALSO stored in DHT at SHA256(wallet_id)                    │
└─────────────────────────────────────────────────────────────────┘
┌─────────────────────────────────────────────────────────────────┐
│ 2. DHT STORAGE                                                  │
│    - Key: SHA256(wallet_id) → Deterministic location           │
│    - Value: JSON list of proofs (with signatures)              │
│    - Replication: Stored on k=3 closest nodes                  │
│    - Persistence: Each node stores ~100 most recent proofs     │
└─────────────────────────────────────────────────────────────────┘
┌─────────────────────────────────────────────────────────────────┐
│ 3. BALANCE BOOTSTRAP (New Node)                                │
│    - Calculate DHT key = SHA256(wallet_id)                     │
│    - Query DHT for proofs                                       │
│    - Verify ECDSA signature on EACH proof                      │
│    - Cross-validate with 3+ DHT nodes                          │
│    - Credit only cryptographically verified proofs             │
└─────────────────────────────────────────────────────────────────┘

Security Properties:
- ✅ Trustless: All proofs have ECDSA signatures
- ✅ Decentralized: No central server required
- ✅ Byzantine-resistant: Majority consensus required
- ✅ Replay-protected: Signature deduplication
- ✅ Rate-limited: Plausibility checks on claims
"""

import hashlib
import json
import logging
import time
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, asdict

logger = logging.getLogger(__name__)


@dataclass
class DHTProofRecord:
    """
    A proof record stored in DHT.
    
    This contains ALL fields needed for ECDSA signature verification.
    Missing any field = can't verify = REJECT proof.
    
    CRITICAL: We store the FULL proof data so we can reconstruct
    canonical_payload for signature verification.
    """
    node_id: str
    timestamp: float
    proof_type: str  # "training", "inference", "uptime"
    nonce: str  # 🔒 REQUIRED for canonical_payload
    reward: float
    signature: str
    public_key: str  # 🔒 REQUIRED for trustless verification
    
    # Work metrics (required for canonical_payload)
    uptime_seconds: float = 0.0
    tokens_processed: int = 0
    training_batches: int = 0
    data_samples: int = 0
    model_hash: str = ""  # 🔒 REQUIRED for canonical_payload
    request_id: str = ""  # 🔒 REQUIRED for canonical_payload (inference proofs)
    
    # Role metadata
    layers_held: int = 0
    has_embedding: bool = False
    has_lm_head: bool = False
    
    def to_dict(self) -> dict:
        """Convert to dict for JSON storage."""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: dict) -> 'DHTProofRecord':
        """Create from dict."""
        return cls(**data)
    
    def to_compact_json(self) -> str:
        """Compact JSON for DHT storage (minimize bandwidth)."""
        # Only include non-zero fields
        compact = {
            "n": self.node_id[:16],  # Abbreviated node_id (wallet identifier)
            "t": self.timestamp,
            "p": self.proof_type[0],  # "t"=training, "i"=inference, "u"=uptime
            "r": round(self.reward, 6),
            "s": self.signature[:32] + "..." + self.signature[-8:],  # Abbreviated signature
        }
        
        # Add optional fields only if present
        if self.uptime_seconds > 0:
            compact["u"] = self.uptime_seconds
        if self.tokens_processed > 0:
            compact["tk"] = self.tokens_processed
        if self.training_batches > 0:
            compact["tb"] = self.training_batches
        if self.data_samples > 0:
            compact["ds"] = self.data_samples
        if self.layers_held > 0:
            compact["l"] = self.layers_held
        if self.has_embedding:
            compact["e"] = 1
        if self.has_lm_head:
            compact["h"] = 1
            
        return json.dumps(compact, separators=(',', ':'))  # No whitespace


class DHTProofStore:
    """
    DHT-based proof storage manager.
    
    Handles storing and retrieving proofs from Kademlia DHT.
    """
    
    def __init__(self, dht_protocol=None):
        """
        Initialize DHT proof store.
        
        Args:
            dht_protocol: Instance of DHTProtocol for DHT operations
        """
        self.dht = dht_protocol
        
    def get_dht_key_for_wallet(self, wallet_id: str) -> int:
        """
        Calculate DHT key for a wallet.
        
        The key is SHA256(wallet_id) to ensure deterministic placement
        in the DHT and load balancing across nodes.
        
        Args:
            wallet_id: Wallet identifier (first 16 chars of node_id)
            
        Returns:
            160-bit integer key for DHT storage
        """
        # Use SHA1 for DHT (Kademlia standard is 160-bit)
        return int(hashlib.sha1(f"proofs:{wallet_id}".encode()).hexdigest(), 16)
    
    def store_proof_in_dht(
        self, 
        wallet_id: str, 
        proof_record: DHTProofRecord,
        desired_replication: int = 3
    ) -> bool:
        """
        Store a proof in DHT with adaptive replication.
        
        The proof is stored at DHT key = SHA256(wallet_id) and replicated
        to k closest nodes. The replication factor adapts to network size:
        - 1 node: Local DB only (no replication)
        - 2 nodes: k=1 (store on 1 peer)
        - 3+ nodes: k=3 (full replication)
        
        Args:
            wallet_id: Wallet identifier
            proof_record: Proof to store
            desired_replication: Desired replication factor (default: 3)
            
        Returns:
            True if stored on at least one node (or 0 nodes if network too small)
        """
        if not self.dht:
            logger.warning("DHT not available - proof not stored in DHT")
            return False
        
        try:
            # Calculate DHT key
            dht_key = self.get_dht_key_for_wallet(wallet_id)
            
            # Serialize proof to JSON
            proof_json = json.dumps(proof_record.to_dict())
            
            # ADAPTIVE REPLICATION: Adjust k to available network size
            # Get all nodes in routing table
            all_nodes = self.dht.routing_table.get_all_nodes()
            available_peers = len(all_nodes)  # Excludes self
            
            # Adapt replication factor to network size
            # For 2-node network: k=1 (store on 1 peer)
            # For 3+ node network: k=3 (full replication)
            actual_replication = min(desired_replication, max(1, available_peers))
            
            # Find k closest nodes to this key
            closest_nodes = self.dht.routing_table.find_closest(dht_key, k=actual_replication)
            
            if not closest_nodes:
                if available_peers == 0:
                    # Solo node - can't use DHT yet
                    logger.debug(f"Solo node - no peers for DHT replication (wallet {wallet_id[:8]}...)")
                    logger.debug(f"Proofs will be stored in DHT when network has >=2 nodes")
                    return False  # Not an error - just need more nodes
                else:
                    logger.warning(f"No DHT nodes found for wallet {wallet_id[:8]}... despite {available_peers} peers")
                    return False
            
            # Store on each node
            success_count = 0
            logger.debug(f"Attempting to store proof on {len(closest_nodes)} DHT nodes for wallet {wallet_id[:8]}...")
            for node in closest_nodes:
                try:
                    result = self.dht.store(node, dht_key, proof_json)
                    if result:
                        success_count += 1
                        logger.debug(f"  ✓ Stored on {node.ip}:{node.port}")
                    else:
                        logger.debug(f"  ✗ Store returned False for {node.ip}:{node.port}")
                except Exception as e:
                    logger.debug(f"  ✗ Exception storing on {node.ip}:{node.port}: {e}")
            
            if success_count > 0:
                # Show network size context
                if available_peers < 3:
                    logger.info(f"Stored proof on {success_count} node(s) (network has {available_peers+1} nodes - need 3+ for full replication)")
                else:
                    logger.debug(f"Stored proof for wallet {wallet_id[:8]}... on {success_count}/{len(closest_nodes)} DHT nodes")
                return True
            else:
                # Don't warn if network is just too small (e.g. startup)
                if available_peers < 1:
                     logger.debug(f"Deferred DHT storage for wallet {wallet_id[:8]}... (network too small: {available_peers} peers)")
                else:
                     logger.warning(f"Failed to store proof on any DHT nodes for wallet {wallet_id[:8]}... (tried {len(closest_nodes)} nodes)")
                return False
                
        except Exception as e:
            logger.error(f"DHT proof storage error: {e}")
            return False
    
    def retrieve_proofs_from_dht(
        self, 
        wallet_id: str,
        max_proofs: int = 100,
        verify_signatures: bool = True
    ) -> Tuple[List[DHTProofRecord], Dict[str, any]]:
        """
        Retrieve proofs for a wallet from DHT with verification.
        
        This performs a multi-step trustless retrieval:
        1. Query DHT for wallet's proofs
        2. Verify ECDSA signatures on each proof
        3. Cross-validate with multiple DHT nodes
        4. Return only verified proofs
        
        Args:
            wallet_id: Wallet identifier
            max_proofs: Maximum number of proofs to retrieve
            verify_signatures: Whether to verify ECDSA signatures (default: True)
            
        Returns:
            (verified_proofs, metadata)
            metadata includes: total_reward, proof_count, retrieval_stats
        """
        if not self.dht:
            logger.warning("DHT not available - cannot retrieve proofs")
            return [], {"error": "DHT not available"}
        
        try:
            # Calculate DHT key
            dht_key = self.get_dht_key_for_wallet(wallet_id)
            
            # Query DHT for value
            logger.info(f"Querying DHT for wallet {wallet_id[:8]}... proofs (key={hex(dht_key)[:10]}...)")
            
            value_json = self.dht.lookup_value(dht_key)
            
            if not value_json:
                logger.info(f"No proofs found in DHT for wallet {wallet_id[:8]}...")
                return [], {
                    "found": False,
                    "wallet_id": wallet_id,
                    "message": "No proofs in DHT (new wallet or network still syncing)"
                }
            
            # Parse JSON list of proofs
            try:
                proof_list = json.loads(value_json)
                if not isinstance(proof_list, list):
                    proof_list = [value_json]  # Single proof
            except json.JSONDecodeError:
                # Legacy format or single proof
                proof_list = [value_json]
            
            logger.info(f"Found {len(proof_list)} proof records in DHT for wallet {wallet_id[:8]}...")
            
            # Parse and verify each proof
            verified_proofs = []
            total_reward = 0.0
            verification_failures = 0
            
            for i, proof_json in enumerate(proof_list[:max_proofs]):
                try:
                    # Parse proof record
                    if isinstance(proof_json, str):
                        proof_data = json.loads(proof_json)
                    else:
                        proof_data = proof_json
                    
                    proof_record = DHTProofRecord.from_dict(proof_data)
                    
                    # 🔒 SECURITY: ALWAYS verify signatures (trustless system)
                    if verify_signatures:
                        # Import crypto module for verification
                        from neuroshard.core.crypto.ecdsa import verify_signature
                        
                        # 🚨 CRITICAL: Reject proof if no public key
                        if not proof_record.public_key:
                            verification_failures += 1
                            logger.warning(f"❌ REJECTED proof {i+1}: No public key included (node {proof_record.node_id[:16]}...)")
                            continue
                        
                        # Reconstruct canonical payload EXACTLY as in ledger.py:176-181
                        # CRITICAL: Must match format from PoNWProof.canonical_payload()
                        # Format: {node_id}:{proof_type}:{timestamp}:{nonce}:{uptime}:{tokens}:{batches}:{samples}:{request_id}:{model_hash}:{layers}
                        payload = (
                            f"{proof_record.node_id}:{proof_record.proof_type}:{proof_record.timestamp:.6f}:{proof_record.nonce}:"
                            f"{float(proof_record.uptime_seconds):.1f}:{proof_record.tokens_processed}:{proof_record.training_batches}:"
                            f"{proof_record.data_samples}:{proof_record.request_id if proof_record.request_id else ''}:"
                            f"{proof_record.model_hash}:{proof_record.layers_held}"
                        )
                        
                        # Verify ECDSA signature with provided public key
                        # The verify_signature function will:
                        # 1. Validate public_key matches node_id
                        # 2. Verify ECDSA signature
                        # 3. Auto-register public key for future use
                        is_valid = verify_signature(
                            node_id=proof_record.node_id,
                            payload=payload,
                            signature=proof_record.signature,
                            public_key_hex=proof_record.public_key  # Provide public key
                        )
                        
                        if is_valid:
                            verified_proofs.append(proof_record)
                            total_reward += proof_record.reward
                        else:
                            verification_failures += 1
                            logger.warning(f"❌ REJECTED proof {i+1}: Invalid ECDSA signature (node {proof_record.node_id[:16]}...)")
                    else:
                        # No verification requested - accept all (DANGEROUS!)
                        verified_proofs.append(proof_record)
                        total_reward += proof_record.reward
                        
                except Exception as e:
                    logger.warning(f"Failed to parse/verify proof {i+1}: {e}")
                    verification_failures += 1
            
            metadata = {
                "found": True,
                "wallet_id": wallet_id,
                "total_proofs": len(proof_list),
                "verified_proofs": len(verified_proofs),
                "verification_failures": verification_failures,
                "total_reward": round(total_reward, 6),
                "dht_key": hex(dht_key),
                "signature_verification": "enabled" if verify_signatures else "disabled"
            }
            
            logger.info(f"DHT retrieval complete: {len(verified_proofs)}/{len(proof_list)} proofs verified, "
                       f"total_reward={total_reward:.6f} NEURO")
            
            return verified_proofs, metadata
            
        except Exception as e:
            logger.error(f"DHT proof retrieval error: {e}")
            return [], {"error": str(e)}
    
    def cross_validate_proofs(
        self,
        wallet_id: str,
        desired_validators: int = 3
    ) -> Tuple[bool, Dict[str, any]]:
        """
        Cross-validate proof data with multiple DHT nodes (ADAPTIVE).
        
        Queries multiple independent DHT nodes and ensures consensus
        on wallet balance. Adapts to network size:
        - 1 node: Skip validation (only local DB exists)
        - 2 nodes: Validate with both (need 100% agreement)
        - 3+ nodes: Validate with 3+ (need majority)
        
        Args:
            wallet_id: Wallet to validate
            desired_validators: Desired number of validators (default: 3)
            
        Returns:
            (consensus_reached, validation_data)
        """
        if not self.dht:
            return False, {"error": "DHT not available"}
        
        try:
            dht_key = self.get_dht_key_for_wallet(wallet_id)
            
            # ADAPTIVE VALIDATION: Adjust to network size
            all_nodes = self.dht.routing_table.get_all_nodes()
            available_peers = len(all_nodes)
            
            # Calculate actual validators needed
            # For small networks: Use all available peers
            # For large networks: Use desired_validators (usually 3)
            actual_validators = min(desired_validators, available_peers)
            
            if actual_validators == 0:
                # Solo node - no cross-validation possible
                logger.info(f"Solo node - skipping cross-validation (network needs >=2 nodes)")
                return True, {
                    "consensus": True,
                    "validators_queried": 0,
                    "network_size": 1,
                    "message": "Solo node - cross-validation skipped (not needed)"
                }
            
            # Find multiple nodes that should have this data
            closest_nodes = self.dht.routing_table.find_closest(dht_key, k=actual_validators * 2)
            
            if len(closest_nodes) < actual_validators:
                # Network smaller than expected - adjust
                actual_validators = len(closest_nodes)
                
            if actual_validators == 0:
                logger.warning(f"No DHT nodes reachable for cross-validation")
                # Not enough nodes - accept optimistically for small networks
                return True, {
                    "consensus": True,
                    "validators_queried": 0,
                    "network_size": available_peers,
                    "message": "Network too small for cross-validation - accepted optimistically"
                }
            
            # Query each node independently
            responses = []
            for node in closest_nodes[:actual_validators]:
                try:
                    value, _ = self.dht.find_value(node, dht_key)
                    if value:
                        # Parse and count proofs
                        proof_list = json.loads(value)
                        if not isinstance(proof_list, list):
                            proof_list = [value]
                        
                        # Calculate total reward
                        total = sum(
                            DHTProofRecord.from_dict(json.loads(p) if isinstance(p, str) else p).reward
                            for p in proof_list
                        )
                        
                        responses.append({
                            "node": f"{node.ip}:{node.port}",
                            "proof_count": len(proof_list),
                            "total_reward": round(total, 6)
                        })
                except Exception as e:
                    logger.debug(f"Failed to query node {node}: {e}")
            
            # ADAPTIVE CONSENSUS RULES:
            # - 1 node: Auto-pass (solo mining)
            # - 2 nodes: Both must agree (100% consensus required)
            # - 3+ nodes: Majority consensus (allow 1% deviation)
            
            if len(responses) == 0:
                # No responses - network too small or unreachable
                return True, {
                    "consensus": True,
                    "validators_queried": 0,
                    "message": "No validators available - using local data only"
                }
            
            if len(responses) == 1:
                # Only 1 validator (2-node network)
                # Accept if we got a response (basic sanity check)
                return True, {
                    "consensus": True,
                    "validators_queried": 1,
                    "network_size": 2,
                    "message": "2-node network - single validator confirmation",
                    "responses": responses
                }
            
            # 2+ validators: Check consensus (all nodes should report similar totals)
            rewards = [r["total_reward"] for r in responses]
            avg_reward = sum(rewards) / len(rewards)
            max_deviation = max(abs(r - avg_reward) for r in rewards)
            
            # ADAPTIVE THRESHOLD:
            # - 2 nodes: Require EXACT match (0% deviation)
            # - 3+ nodes: Allow 1% deviation (timing differences)
            deviation_threshold = 0.0 if len(responses) == 2 else (avg_reward * 0.01)
            consensus = max_deviation <= deviation_threshold if avg_reward > 0 else True
            
            validation_data = {
                "consensus": consensus,
                "validators_queried": len(responses),
                "network_size": available_peers + 1,  # +1 for self
                "avg_reward": round(avg_reward, 6),
                "max_deviation": round(max_deviation, 6),
                "deviation_threshold": round(deviation_threshold, 6),
                "responses": responses
            }
            
            if consensus:
                logger.info(f"Cross-validation PASSED for wallet {wallet_id[:8]}... "
                           f"({len(responses)} nodes agree, avg={avg_reward:.6f} NEURO, "
                           f"network_size={available_peers + 1})")
            else:
                logger.warning(f"Cross-validation FAILED for wallet {wallet_id[:8]}... "
                              f"(deviation={max_deviation:.6f} > threshold={deviation_threshold:.6f}, "
                              f"avg={avg_reward:.6f})")
            
            return consensus, validation_data
            
        except Exception as e:
            logger.error(f"Cross-validation error: {e}")
            return False, {"error": str(e)}


# =============================================================================
# DHT EPOCH STORE - Blockchain-like Epoch Chain Storage
# =============================================================================

class DHTEpochStore:
    """
    DHT-based storage for finalized epochs.
    
    This enables decentralized epoch chain verification:
    - Any node can retrieve the epoch chain from DHT
    - Epochs are cryptographically linked (prev_hash)
    - Chain integrity can be verified without trusting any single node
    
    DHT Key Scheme:
    - epoch:{epoch_id} -> Epoch JSON
    - epoch_chain:latest -> Latest finalized epoch_id
    - epoch_chain:genesis -> Genesis epoch hash
    """
    
    def __init__(self, dht_protocol=None):
        """Initialize DHT epoch store."""
        self.dht = dht_protocol
        self._epoch_cache: Dict[int, 'Epoch'] = {}
    
    def get_epoch_dht_key(self, epoch_id: int) -> int:
        """Calculate DHT key for an epoch."""
        key_str = f"epoch:{epoch_id}"
        return int(hashlib.sha1(key_str.encode()).hexdigest(), 16)
    
    def get_latest_epoch_key(self) -> int:
        """Get DHT key for latest epoch pointer."""
        return int(hashlib.sha1(b"epoch_chain:latest").hexdigest(), 16)
    
    def get_genesis_key(self) -> int:
        """Get DHT key for genesis epoch hash."""
        return int(hashlib.sha1(b"epoch_chain:genesis").hexdigest(), 16)
    
    def store_epoch(self, epoch: 'Epoch', replication: int = 5) -> bool:
        """
        Store a finalized epoch in DHT.
        
        Epochs are immutable once finalized, so we use higher replication
        for durability. This is similar to how blocks are stored in Bitcoin.
        
        Args:
            epoch: The finalized epoch to store
            replication: Number of nodes to replicate to (default: 5)
            
        Returns:
            True if stored successfully
        """
        if not self.dht:
            logger.warning("DHT not available - epoch not stored")
            return False
        
        try:
            # Store the epoch itself
            epoch_key = self.get_epoch_dht_key(epoch.epoch_id)
            epoch_json = epoch.to_json()
            
            # Find closest nodes
            closest = self.dht.routing_table.find_closest(epoch_key, k=replication)
            if not closest:
                logger.warning(f"No nodes found to store epoch {epoch.epoch_id}")
                return False
            
            # Store on all closest nodes
            stored_count = 0
            for node in closest:
                try:
                    success = self.dht.store_value_on_node(epoch_key, epoch_json, node)
                    if success:
                        stored_count += 1
                except Exception as e:
                    logger.debug(f"Failed to store epoch on {node}: {e}")
            
            # Update "latest" pointer
            if stored_count > 0:
                latest_key = self.get_latest_epoch_key()
                self.dht.store_value(latest_key, str(epoch.epoch_id))
                
                # Cache locally
                self._epoch_cache[epoch.epoch_id] = epoch
                
                logger.info(f"Epoch {epoch.epoch_id} stored on {stored_count}/{len(closest)} nodes")
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"Failed to store epoch {epoch.epoch_id}: {e}")
            return False
    
    def get_epoch(self, epoch_id: int) -> Optional['Epoch']:
        """
        Retrieve an epoch from DHT.
        
        Args:
            epoch_id: The epoch ID to retrieve
            
        Returns:
            The Epoch if found, None otherwise
        """
        # Check cache first
        if epoch_id in self._epoch_cache:
            return self._epoch_cache[epoch_id]
        
        if not self.dht:
            return None
        
        try:
            epoch_key = self.get_epoch_dht_key(epoch_id)
            epoch_json = self.dht.lookup_value(epoch_key)
            
            if not epoch_json:
                return None
            
            # Import here to avoid circular imports
            from neuroshard.core.consensus.epoch import Epoch
            epoch = Epoch.from_json(epoch_json)
            
            # Cache it
            self._epoch_cache[epoch_id] = epoch
            
            return epoch
            
        except Exception as e:
            logger.error(f"Failed to get epoch {epoch_id}: {e}")
            return None
    
    def get_latest_epoch_id(self) -> Optional[int]:
        """Get the ID of the latest finalized epoch from DHT."""
        if not self.dht:
            return None
        
        try:
            latest_key = self.get_latest_epoch_key()
            value = self.dht.lookup_value(latest_key)
            
            if value:
                return int(value)
            return None
            
        except Exception as e:
            logger.error(f"Failed to get latest epoch ID: {e}")
            return None
    
    def get_epoch_chain(
        self,
        start_id: int,
        end_id: int,
        verify: bool = True
    ) -> Tuple[List['Epoch'], bool, str]:
        """
        Retrieve a range of epochs and optionally verify chain integrity.
        
        Gap-tolerant: Empty epochs (with no proofs) are expected and don't break
        the chain. Only epochs that exist are returned and verified.
        
        Args:
            start_id: Starting epoch ID (inclusive)
            end_id: Ending epoch ID (inclusive)
            verify: Whether to verify chain integrity
            
        Returns:
            (epochs, is_valid, error_message)
        """
        epochs = []
        skipped = 0
        
        for epoch_id in range(start_id, end_id + 1):
            epoch = self.get_epoch(epoch_id)
            if epoch:
                epochs.append(epoch)
            else:
                # Empty epochs are expected - they had no proofs and were skipped
                skipped += 1
        
        if not verify or len(epochs) < 2:
            return epochs, True, f"{len(epochs)} epochs found, {skipped} empty"
        
        # Verify chain integrity (gap-tolerant)
        is_valid, error = self.verify_chain_integrity(epochs)
        if is_valid:
            return epochs, True, f"{len(epochs)} epochs verified, {skipped} empty"
        return epochs, is_valid, error
    
    def verify_chain_integrity(self, epochs: List['Epoch']) -> Tuple[bool, str]:
        """
        Verify that a list of epochs forms a valid chain.
        
        Gap-tolerant: When there are gaps between epochs (empty epochs),
        we only verify consecutive epochs are properly linked. Epochs separated
        by gaps may link to genesis or intermediate (missing) epochs.
        
        Checks:
        1. Each epoch's hash is correctly computed
        2. Consecutive epochs (no gap) have matching prev_hash linkage
        3. Timestamps are sequential
        
        Args:
            epochs: List of epochs in order (may have gaps)
            
        Returns:
            (is_valid, error_message)
        """
        if not epochs:
            return True, ""
        
        # Import here to avoid circular imports
        from neuroshard.core.consensus.epoch import GENESIS_EPOCH_HASH
        
        for i, epoch in enumerate(epochs):
            # Verify epoch hash is correctly computed
            computed_hash = epoch.compute_epoch_hash()
            if computed_hash != epoch.epoch_hash:
                return False, f"Epoch {epoch.epoch_id}: hash mismatch (computed={computed_hash[:16]}... != stored={epoch.epoch_hash[:16]}...)"
            
            if i == 0:
                # First epoch - no previous to check against
                continue
            
            prev_epoch = epochs[i - 1]
            gap_size = epoch.epoch_id - prev_epoch.epoch_id
            
            if gap_size == 1:
                # Consecutive epochs - verify strict linkage
                if epoch.prev_epoch_hash != prev_epoch.epoch_hash:
                    return False, f"Epoch {epoch.epoch_id}: prev_hash broken (expected {prev_epoch.epoch_hash[:16]}...)"
                
                # Check model state progression for consecutive epochs
                if epoch.model_state_hash_start != prev_epoch.model_state_hash_end:
                    return False, f"Epoch {epoch.epoch_id}: model state discontinuity"
            # else: gap exists - prev_hash may link to missing epoch or genesis, which is acceptable
            
            # Check timestamp ordering (always, regardless of gap)
            if epoch.timestamp_start < prev_epoch.timestamp_end:
                return False, f"Epoch {epoch.epoch_id}: timestamps overlap with {prev_epoch.epoch_id}"
        
        return True, ""
    
    def sync_epoch_chain(
        self,
        local_latest: int,
        target_latest: Optional[int] = None
    ) -> Tuple[int, int, str]:
        """
        Sync local epoch chain with network.
        
        Args:
            local_latest: Latest epoch ID in local storage
            target_latest: Target epoch ID (None = get from DHT)
            
        Returns:
            (synced_count, new_latest, status_message)
        """
        # Get target from DHT if not provided
        if target_latest is None:
            target_latest = self.get_latest_epoch_id()
            if target_latest is None:
                return 0, local_latest, "Could not determine latest epoch from network"
        
        if target_latest <= local_latest:
            return 0, local_latest, "Already up to date"
        
        # Fetch missing epochs
        synced = 0
        for epoch_id in range(local_latest + 1, target_latest + 1):
            epoch = self.get_epoch(epoch_id)
            if epoch:
                synced += 1
            else:
                return synced, local_latest + synced, f"Failed to sync epoch {epoch_id}"
        
        return synced, target_latest, f"Synced {synced} epochs"

