"""
gRPC Server for NeuroShard Node

Handles:
1. Inference requests (forward pass through NeuroLLM)
2. Training gradient exchange
3. DHT operations for peer discovery
4. PoNW proof verification
5. Layer-specific forward (for distributed inference)
"""

import grpc
from concurrent import futures
import torch
import time
import threading
import random
from typing import Optional, Union
import logging

# Import generated protobuf code
from neuroshard.protos import neuroshard_pb2
from neuroshard.protos import neuroshard_pb2_grpc

from neuroshard.utils.serialization import deserialize_tensor, serialize_tensor
from neuroshard.core.network.p2p import P2PManager
from neuroshard.core.network.dht_service import DHTServiceMixin

# Swarm Service Mixin (Phase 4)
try:
    from neuroshard.core.swarm.service import SwarmServiceMixin
    SWARM_SERVICE_AVAILABLE = True
except ImportError:
    SWARM_SERVICE_AVAILABLE = False
    SwarmServiceMixin = object  # Fallback

logger = logging.getLogger(__name__)

# Global state
GRPC_SERVER = None
GRPC_SERVICER = None  # Reference to NeuroShardServiceServicer for quorum buffer wiring


class NeuroShardServiceServicer(DHTServiceMixin, neuroshard_pb2_grpc.NeuroShardServiceServicer):
    """
    gRPC service for NeuroShard nodes.
    
    Supports DynamicNeuroNode with layer-based routing.
    Each node holds specific layers and can process inference requests for those layers.
    
    Swarm Architecture:
    - SwarmForward: Async activation forwarding with failover
    - GetSwarmStatus: Buffer fill rates, capacity info
    - UpdatePeerCapacity: TCP fallback for heartbeat updates
    """
    
    def __init__(self, model, p2p: P2PManager, swap_controller=None):
        """
        Args:
            model: DynamicNeuroNode or SwarmEnabledDynamicNode instance
            p2p: P2P manager for peer discovery
            swap_controller: Deprecated, kept for compatibility
        """
        self.model = model
        self.p2p = p2p
        self.swap_controller = swap_controller
        
        # Detect if this is a DynamicNeuroNode
        self.is_dynamic_node = hasattr(model, 'my_layer_ids') and hasattr(model, 'layer_pool')
        self.is_neuro_node = self.is_dynamic_node  # Alias for legacy compatibility
        
        # Rate limiter for gradient gossip (per-node)
        # Allows max 1 gossip per node per 60 seconds (DiLoCo sync is every ~8-10 min)
        self._gossip_rate_limit: dict = {}  # node_id -> last_gossip_time
        self._gossip_rate_limit_seconds = 60  # Min seconds between gossips from same node
        self._gossip_rate_limit_lock = threading.Lock()
        
        # Quorum pipeline support
        # - quorum_inbound_buffer: Set by runner when QuorumTrainer is active.
        #   PipelineForward routes training activations here instead of processing synchronously.
        # - on_quorum_proposed: Callback invoked when a ProposeQuorum RPC is accepted.
        #   runner.py uses this to set up the quorum on the receiving side.
        self.quorum_inbound_buffer = None  # queue.Queue set by runner
        self.on_quorum_proposed = None     # Callable(quorum_id, proposer_info, my_info) -> None
        self._active_quorum_id = None      # Current quorum ID for routing
        self._active_quorum_type = None    # "sync" or "pipeline" — sync quorums accept new peers
        
        # Initialize DHT Mixin if available
        if p2p.routing_table:
            DHTServiceMixin.__init__(self, p2p.routing_table, p2p.dht_storage, ledger=p2p.ledger)
        else:
            from neuroshard.core.network.dht import RoutingTable, Node
            dummy_rt = RoutingTable(Node(0, "0.0.0.0", 0))
            DHTServiceMixin.__init__(self, dummy_rt, {}, ledger=p2p.ledger)
        
        logger.info(f"gRPC Servicer initialized (DynamicNode={self.is_dynamic_node})")

    def UnaryInference(self, request, context):
        """Handle inference request."""
        try:
            # Deserialize input
            input_str = request.tensor_data.decode('utf-8')
            input_tensor = deserialize_tensor(input_str)
            
            # Forward pass through DynamicNeuroNode
            output = self.model.forward(input_tensor, session_id=request.session_id)
            # Return result directly
            return neuroshard_pb2.InferenceResponse(
                success=True,
                tensor_data=serialize_tensor(output).encode('utf-8')
            )

        except Exception as e:
            logger.error(f"Inference error: {e}")
            return neuroshard_pb2.InferenceResponse(success=False, error_message=str(e))

    def GetWeights(self, request, context):
        """Return model weights (for gradient sync)."""
        try:
            # Get weights for my layers only
            layer_weights = {}
            for layer_id, layer in self.model.model.my_layers.items():
                layer_weights[f"layer_{layer_id}"] = layer.state_dict()
            
            if self.model.model.embedding:
                layer_weights["embedding"] = self.model.model.embedding.state_dict()
            if self.model.model.lm_head:
                layer_weights["lm_head"] = self.model.model.lm_head.state_dict()
            
            data = serialize_tensor(layer_weights, use_quantization=False).encode('utf-8')
            return neuroshard_pb2.WeightResponse(weights_data=data)
        except Exception as e:
            logger.error(f"GetWeights error: {e}")
            return neuroshard_pb2.WeightResponse(weights_data=b"")

    # ==================== PoNW GOSSIP RPCs ====================
    
    def GossipProof(self, request, context):
        """
        Receive a Proof of Neural Work from a peer.
        
        This is the core PoNW gossip handler. When a node earns rewards,
        it gossips the proof to peers who:
        1. Verify the signature (ECDSA - trustless)
        2. Store the proof in their ledger (for DHT propagation)
        3. Optionally forward to more peers
        
        Security: Proofs are cryptographically signed and verified.
        """
        try:
            from neuroshard.core.economics.ledger import PoNWProof
            from neuroshard.core.crypto.ecdsa import register_public_key
            
            # Log incoming proof for debugging
            logger.info(f"[GOSSIP] Received proof from {request.node_id[:16]}... type={request.proof_type}")
            
            # Register sender's public key for verification
            if request.public_key:
                register_public_key(request.node_id, request.public_key)
            
            # Reconstruct the proof object
            # CRITICAL: Include ALL fields that are in canonical_payload for signature verification
            proof = PoNWProof(
                node_id=request.node_id,
                proof_type=request.proof_type,
                timestamp=request.timestamp,
                nonce=request.nonce,
                uptime_seconds=request.uptime,
                tokens_processed=request.token_count,
                training_batches=request.training_batches,
                data_samples=request.data_samples,
                model_hash=request.model_hash,
                layers_held=request.layers_held,
                has_embedding=request.has_embedding,
                has_lm_head=request.has_lm_head,
                current_loss=request.current_loss if request.current_loss != 0.0 else None,
                request_id=request.request_id if request.request_id else None,
                signature=request.signature,
                # Chained PoNW fields - CRITICAL for canonical_payload signature verification
                epoch_id=request.epoch_id,
                model_hash_start=request.model_hash_start if request.model_hash_start else "",
                model_hash_end=request.model_hash_end if request.model_hash_end else "",
                gradient_commitment=request.gradient_commitment if request.gradient_commitment else "",
                data_hash=request.data_hash if request.data_hash else "",
                gradient_norm=request.gradient_norm if request.gradient_norm != 0.0 else None,
            )
            
            # Verify and process the proof using our ledger
            if self.p2p and self.p2p.ledger:
                ledger = self.p2p.ledger
                
                # 1. Deduplication (Idempotency check)
                # Check if we already have this proof BEFORE doing expensive verification
                if ledger.has_proof(proof.signature):
                    logger.debug(f"[GOSSIP] Ignoring duplicate proof from {request.node_id[:8]}... (already processed)")
                    return neuroshard_pb2.GossipProofResponse(accepted=True)
                
                # 2. Verify
                is_valid, reason = ledger.verify_proof(proof)
                
                if is_valid:
                    # 3. Process (credits rewards)
                    # skip_verification=True because we already called verify_proof above
                    success, reward, msg = ledger.process_proof(proof, skip_verification=True)
                    if success:
                        logger.info(f"[GOSSIP] ✓ Accepted proof from {request.node_id[:16]}...: {reward:.6f} NEURO")
                        
                        # 🔥 DECENTRALIZED: Also store proof in DHT for network-wide visibility
                        # This ensures any node can query balances from DHT, not just local ledger
                        self._store_received_proof_in_dht(proof, reward, request.public_key)
                        
                        # 📦 EPOCH TRACKING: Register proof with epoch manager
                        # This enables chained epoch verification for blockchain-like security
                        self._register_proof_with_epoch_manager(proof, reward)
                        
                        return neuroshard_pb2.GossipProofResponse(accepted=True)
                    else:
                        # Should rarely happen given has_proof check above, but handle race conditions
                        if "Duplicate" in msg:
                             logger.debug(f"[GOSSIP] Ignoring duplicate proof (race condition)")
                             return neuroshard_pb2.GossipProofResponse(accepted=True)
                             
                        logger.warning(f"[GOSSIP] ✗ Proof rejected: {msg}")
                        return neuroshard_pb2.GossipProofResponse(accepted=False)
                else:
                    logger.info(f"[GOSSIP] ✗ Proof verification failed: {reason}")
                    return neuroshard_pb2.GossipProofResponse(accepted=False)
            
            # No ledger - just accept silently
            return neuroshard_pb2.GossipProofResponse(accepted=True)
            
        except Exception as e:
            logger.error(f"GossipProof error: {e}")
            return neuroshard_pb2.GossipProofResponse(accepted=False)
    
    def _store_received_proof_in_dht(self, proof, reward: float, public_key: str):
        """
        Store a received proof in DHT for decentralized replication.
        
        This is called when we receive a valid proof via gossip. By storing it
        in DHT, we ensure any node can query balances without relying on the
        original sender's DHT availability.
        
        This makes the system TRULY DECENTRALIZED:
        - Proofs are replicated across multiple DHT nodes
        - Any node can verify balances by querying DHT
        - No single point of failure
        """
        if not self.p2p or not self.p2p.dht:
            return
        
        try:
            import threading
            from neuroshard.core.network.dht_proof_store import DHTProofStore, DHTProofRecord
            
            # Create DHT proof store
            dht_store = DHTProofStore(self.p2p.dht)
            
            # Create proof record with all fields for verification
            # CRITICAL: Must include request_id for canonical_payload to match signature
            proof_record = DHTProofRecord(
                node_id=proof.node_id,
                timestamp=proof.timestamp,
                proof_type=proof.proof_type.value if hasattr(proof.proof_type, 'value') else str(proof.proof_type),
                nonce=proof.nonce,
                reward=reward,
                signature=proof.signature,
                public_key=public_key or "",  # Include public key for trustless verification
                uptime_seconds=proof.uptime_seconds,
                tokens_processed=proof.tokens_processed,
                training_batches=proof.training_batches,
                data_samples=proof.data_samples,
                model_hash=proof.model_hash,
                request_id=proof.request_id if proof.request_id else "",  # 🔒 Required for signature
                layers_held=proof.layers_held,
                has_embedding=proof.has_embedding,
                has_lm_head=proof.has_lm_head
            )
            
            # Get wallet_id (first 16 chars of node_id)
            wallet_id = proof.node_id[:16]
            
            # Store in DHT (async to not block gRPC response)
            def store_async():
                try:
                    success = dht_store.store_proof_in_dht(wallet_id, proof_record)
                    if success:
                        logger.debug(f"[DHT] Replicated proof for {wallet_id}... to DHT")
                except Exception as e:
                    logger.debug(f"[DHT] Replication failed: {e}")
            
            threading.Thread(target=store_async, daemon=True).start()
            
        except Exception as e:
            logger.debug(f"DHT proof replication error: {e}")
    
    def _register_proof_with_epoch_manager(self, proof, reward: float):
        """
        Register a verified proof with the epoch manager for epoch chaining.
        
        This is a key part of the Chained PoNW system:
        - Each proof is bound to a specific epoch
        - Epochs aggregate proofs with merkle roots
        - Epochs are cryptographically linked (like blockchain)
        
        Args:
            proof: The verified PoNW proof
            reward: The reward amount credited
        """
        try:
            # Import here to avoid circular imports
            from neuroshard.runner import STATE
            
            # Get epoch manager from global state (set by observer mode)
            epoch_manager = STATE.get("epoch_manager")
            if not epoch_manager:
                return  # Not in observer mode or epoch manager not initialized
            
            # Extract gradient commitment if present
            gradient_commitment = None
            if hasattr(proof, 'gradient_commitment') and proof.gradient_commitment:
                from neuroshard.core.consensus.epoch import GradientCommitment
                # If we have full gradient data, create the commitment object
                # Otherwise, we just have the hash
                gradient_commitment = None  # Will use hash from proof
            
            # Accept proof into current epoch
            accepted, message = epoch_manager.accept_proof(
                proof_signature=proof.signature,
                node_id=proof.node_id,
                proof_type=proof.proof_type,
                timestamp=proof.timestamp,
                reward=reward,
                gradient_commitment=gradient_commitment,
            )
            
            if accepted:
                logger.debug(f"[EPOCH] Proof registered: {message}")
            else:
                logger.warning(f"[EPOCH] Proof not registered: {message}")
                
        except Exception as e:
            logger.debug(f"[EPOCH] Registration error: {e}")
    
    def GossipTransaction(self, request, context):
        """
        Receive a NEURO transaction from a peer.
        
        Transactions are used for:
        - Paying for inference
        - Staking NEURO
        - Transfers between wallets
        """
        try:
            if self.p2p and self.p2p.ledger:
                ledger = self.p2p.ledger
                
                # Verify signature and process transaction
                success = ledger.create_transaction(
                    from_id=request.sender_id,
                    to_id=request.recipient_id,
                    amount=request.amount,
                    signature=request.signature
                )
                
                if success:
                    logger.debug(f"[GOSSIP] Accepted transaction: {request.amount:.6f} NEURO")
                    return neuroshard_pb2.GossipTransactionResponse(accepted=True)
                else:
                    return neuroshard_pb2.GossipTransactionResponse(
                        accepted=False, 
                        reason="Transaction verification failed"
                    )
            
            return neuroshard_pb2.GossipTransactionResponse(accepted=True)
            
        except Exception as e:
            logger.error(f"GossipTransaction error: {e}")
            return neuroshard_pb2.GossipTransactionResponse(accepted=False, reason=str(e))
    
    def GossipStake(self, request, context):
        """
        Receive a stake update from a peer.
        
        Stake information is gossiped so peers can:
        1. Verify PoNW multipliers
        2. Select validators (stake-weighted)
        3. Display network stake distribution
        """
        try:
            from neuroshard.core.crypto.ecdsa import register_public_key, verify_signature
            
            # Register public key
            if request.public_key:
                register_public_key(request.node_id, request.public_key)
            
            # Verify signature on stake claim
            payload = f"{request.node_id}:{request.amount}:{request.locked_until}:{request.timestamp}"
            if not verify_signature(request.node_id, payload, request.signature):
                return neuroshard_pb2.GossipStakeResponse(
                    accepted=False,
                    reason="Invalid signature"
                )
            
            # Update stake in ledger
            if self.p2p and self.p2p.ledger:
                ledger = self.p2p.ledger
                success = ledger.update_stake(
                    node_id=request.node_id,
                    amount=request.amount,
                    locked_until=request.locked_until
                )
                
                if success:
                    logger.debug(f"[GOSSIP] Updated stake for {request.node_id[:16]}...: {request.amount:.2f} NEURO")
                    return neuroshard_pb2.GossipStakeResponse(accepted=True)
                else:
                    return neuroshard_pb2.GossipStakeResponse(
                        accepted=False,
                        reason="Stake update rejected"
                    )
            
            return neuroshard_pb2.GossipStakeResponse(accepted=True)
            
        except Exception as e:
            logger.error(f"GossipStake error: {e}")
            return neuroshard_pb2.GossipStakeResponse(accepted=False, reason=str(e))
    
    # =========================================================================
    # CHAINED EPOCH SYSTEM RPCs
    # =========================================================================
    
    def AnnounceEpoch(self, request, context):
        """
        Receive an epoch announcement from a peer.
        
        When a finalized epoch is gossiped:
        1. Verify the epoch hash is correct
        2. Verify it links to our previous epoch
        3. Store it locally
        4. Optionally gossip to other peers
        """
        try:
            from neuroshard.core.consensus.epoch import Epoch, EpochStatus
            from neuroshard.runner import STATE
            
            # Reconstruct epoch from request
            epoch = Epoch(
                epoch_id=request.epoch_id,
                epoch_hash=request.epoch_hash,
                prev_epoch_hash=request.prev_epoch_hash,
                timestamp_start=request.timestamp_start,
                timestamp_end=request.timestamp_end,
                model_state_hash_start=request.model_state_hash_start,
                model_state_hash_end=request.model_state_hash_end,
                proofs_merkle_root=request.proofs_merkle_root,
                proof_count=request.proof_count,
                total_reward=request.total_reward,
                total_batches=0,  # Not in gossip message
                average_loss=0.0,  # Not in gossip message
                gradient_commitments_root=request.gradient_commitments_root,
                proposer_node_id=request.proposer_node_id,
                proposer_signature=request.proposer_signature,
                status=EpochStatus.FINALIZED,
            )
            
            # Verify epoch hash
            computed_hash = epoch.compute_epoch_hash()
            if computed_hash != epoch.epoch_hash:
                logger.warning(f"[EPOCH] Invalid hash from peer: {computed_hash[:16]}... != {epoch.epoch_hash[:16]}...")
                return neuroshard_pb2.EpochAnnouncementResponse(
                    accepted=False,
                    reason="Invalid epoch hash"
                )
            
            # Store in epoch manager if available
            epoch_manager = STATE.get("epoch_manager")
            if epoch_manager:
                epoch_manager.epoch_chain[epoch.epoch_id] = epoch
                if epoch.epoch_id > epoch_manager.latest_finalized_epoch_id:
                    epoch_manager.latest_finalized_epoch_id = epoch.epoch_id
            
            # Store in local ledger
            if self.p2p and self.p2p.ledger:
                self.p2p._store_epoch_locally(epoch)
            
            logger.info(f"[EPOCH] Accepted epoch {epoch.epoch_id} from peer")
            return neuroshard_pb2.EpochAnnouncementResponse(accepted=True)
            
        except Exception as e:
            logger.error(f"AnnounceEpoch error: {e}")
            return neuroshard_pb2.EpochAnnouncementResponse(
                accepted=False,
                reason=str(e)
            )
    
    def GetEpoch(self, request, context):
        """
        Return a specific epoch to a requesting peer.
        
        Used for epoch chain synchronization.
        """
        try:
            import sqlite3
            from neuroshard.runner import STATE
            
            epoch_id = request.epoch_id
            
            # Try epoch manager first
            epoch_manager = STATE.get("epoch_manager")
            if epoch_manager and epoch_id in epoch_manager.epoch_chain:
                epoch = epoch_manager.epoch_chain[epoch_id]
                return neuroshard_pb2.EpochResponse(
                    found=True,
                    epoch_id=epoch.epoch_id,
                    epoch_hash=epoch.epoch_hash,
                    prev_epoch_hash=epoch.prev_epoch_hash,
                    timestamp_start=epoch.timestamp_start,
                    timestamp_end=epoch.timestamp_end,
                    model_state_hash_start=epoch.model_state_hash_start,
                    model_state_hash_end=epoch.model_state_hash_end,
                    proofs_merkle_root=epoch.proofs_merkle_root,
                    proof_count=epoch.proof_count,
                    total_reward=epoch.total_reward,
                    total_batches=epoch.total_batches,
                    average_loss=epoch.average_loss,
                    gradient_commitments_root=epoch.gradient_commitments_root,
                    proposer_node_id=epoch.proposer_node_id,
                    proposer_signature=epoch.proposer_signature,
                )
            
            # Try local database
            if self.p2p and self.p2p.ledger:
                with sqlite3.connect(self.p2p.ledger.db_path, timeout=10.0) as conn:
                    conn.row_factory = sqlite3.Row
                    row = conn.execute(
                        "SELECT * FROM epochs WHERE epoch_id = ?",
                        (epoch_id,)
                    ).fetchone()
                    
                    if row:
                        return neuroshard_pb2.EpochResponse(
                            found=True,
                            epoch_id=row["epoch_id"],
                            epoch_hash=row["epoch_hash"],
                            prev_epoch_hash=row["prev_epoch_hash"],
                            timestamp_start=row["timestamp_start"],
                            timestamp_end=row["timestamp_end"],
                            model_state_hash_start=row["model_state_hash_start"],
                            model_state_hash_end=row["model_state_hash_end"],
                            proofs_merkle_root=row["proofs_merkle_root"],
                            proof_count=row["proof_count"],
                            total_reward=row["total_reward"],
                            total_batches=row["total_batches"],
                            average_loss=row["average_loss"],
                            gradient_commitments_root=row["gradient_commitments_root"],
                            proposer_node_id=row["proposer_node_id"],
                            proposer_signature=row["proposer_signature"],
                        )
            
            return neuroshard_pb2.EpochResponse(found=False)
            
        except Exception as e:
            logger.error(f"GetEpoch error: {e}")
            return neuroshard_pb2.EpochResponse(found=False)
    
    def GetLatestEpoch(self, request, context):
        """
        Return the latest finalized epoch ID.
        
        Used for epoch chain synchronization.
        """
        try:
            from neuroshard.runner import STATE
            
            # Try epoch manager first
            epoch_manager = STATE.get("epoch_manager")
            if epoch_manager and epoch_manager.latest_finalized_epoch_id >= 0:
                latest = epoch_manager.latest_finalized_epoch_id
                epoch = epoch_manager.epoch_chain.get(latest)
                return neuroshard_pb2.LatestEpochResponse(
                    epoch_id=latest,
                    epoch_hash=epoch.epoch_hash if epoch else ""
                )
            
            # Try local database
            if self.p2p and self.p2p.ledger:
                import sqlite3
                with sqlite3.connect(self.p2p.ledger.db_path, timeout=10.0) as conn:
                    row = conn.execute(
                        "SELECT epoch_id, epoch_hash FROM epochs WHERE finalized = 1 ORDER BY epoch_id DESC LIMIT 1"
                    ).fetchone()
                    
                    if row:
                        return neuroshard_pb2.LatestEpochResponse(
                            epoch_id=row[0],
                            epoch_hash=row[1]
                        )
            
            return neuroshard_pb2.LatestEpochResponse(epoch_id=0, epoch_hash="")
            
        except Exception as e:
            logger.error(f"GetLatestEpoch error: {e}")
            return neuroshard_pb2.LatestEpochResponse(epoch_id=0, epoch_hash="")
    
    def SpotCheckChallenge(self, request, context):
        """
        Receive a spot-check challenge for a gradient commitment.
        
        This enables economic security via probabilistic verification:
        - Challenger stakes NEURO to challenge
        - If fraud detected: challenger gets reward, prover slashed
        - If legitimate: challenger loses stake
        """
        try:
            from neuroshard.core.crypto.ecdsa import verify_signature, register_public_key
            from neuroshard.runner import STATE
            
            # Verify challenger's signature
            payload = f"{request.commitment_hash}:{request.challenger_id}:{request.challenger_stake}:{request.timestamp}"
            if not verify_signature(request.challenger_id, payload, request.signature):
                return neuroshard_pb2.SpotCheckChallengeResponse(
                    accepted=False,
                    reason="Invalid signature"
                )
            
            # Store challenge for processing
            if self.p2p and self.p2p.ledger:
                import sqlite3
                import uuid
                
                challenge_id = str(uuid.uuid4())
                
                with sqlite3.connect(self.p2p.ledger.db_path, timeout=10.0) as conn:
                    conn.execute("""
                        INSERT INTO spot_check_challenges
                        (challenge_id, commitment_hash, challenger_id, challenger_stake, challenged_at, status)
                        VALUES (?, ?, ?, ?, ?, 'pending')
                    """, (challenge_id, request.commitment_hash, request.challenger_id, 
                          request.challenger_stake, request.timestamp))
                
                logger.info(f"[SPOT-CHECK] Challenge {challenge_id[:8]}... accepted for commitment {request.commitment_hash[:16]}...")
                return neuroshard_pb2.SpotCheckChallengeResponse(
                    accepted=True,
                    challenge_id=challenge_id
                )
            
            return neuroshard_pb2.SpotCheckChallengeResponse(
                accepted=False,
                reason="No ledger available"
            )
            
        except Exception as e:
            logger.error(f"SpotCheckChallenge error: {e}")
            return neuroshard_pb2.SpotCheckChallengeResponse(
                accepted=False,
                reason=str(e)
            )

    # ==================== DISTRIBUTED TRAINING RPCs ====================
    
    def GossipGradient(self, request, context):
        """
        Receive gradient contribution from a peer.
        
        This is the core of distributed training:
        1. Peer computes gradients locally
        2. Peer broadcasts via this RPC
        3. We aggregate and apply updates
        
        Rate limited to prevent spam attacks (1 gossip per node per 60s).
        """
        
        try:
            # Check if training is enabled
            if not self.model.enable_training:
                return neuroshard_pb2.GossipGradientResponse(
                    accepted=False,
                    reason="Training not enabled on this node"
                )
            
            # RATE LIMITING: Prevent spam from malicious nodes
            node_id = request.node_id
            now = time.time()
            
            with self._gossip_rate_limit_lock:
                last_gossip = self._gossip_rate_limit.get(node_id, 0)
                time_since_last = now - last_gossip
                
                if time_since_last < self._gossip_rate_limit_seconds:
                    logger.warning(f"Rate limited gradient from {node_id[:8]}... "
                                   f"(only {time_since_last:.1f}s since last, need {self._gossip_rate_limit_seconds}s)")
                    return neuroshard_pb2.GossipGradientResponse(
                        accepted=False,
                        reason=f"Rate limited: wait {self._gossip_rate_limit_seconds - time_since_last:.0f}s"
                    )
                
                # Update last gossip time
                self._gossip_rate_limit[node_id] = now
                
                # Clean up old entries (older than 10 minutes)
                stale_nodes = [nid for nid, ts in self._gossip_rate_limit.items() if now - ts > 600]
                for nid in stale_nodes:
                    del self._gossip_rate_limit[nid]
            
            # Convert protobuf to GradientContribution
            from neuroshard.core.training.distributed import GradientContribution
            
            contribution = GradientContribution(
                node_id=request.node_id,
                round_id=request.round_id,
                layer_gradients=dict(request.layer_gradients),  # Convert MapContainer to dict
                batch_size=request.batch_size,
                loss=request.loss,
                timestamp=request.timestamp,
                signature=request.signature
            )
            
            # Submit to NeuroNode for processing
            success = self.model.receive_peer_gradients(contribution)
            
            if success:
                logger.info(f"Received gradient from peer {request.node_id[:8]}... "
                           f"(round={request.round_id}, batch={request.batch_size}, loss={request.loss:.4f})")
            
            return neuroshard_pb2.GossipGradientResponse(
                accepted=success,
                reason="" if success else "Failed to process gradient",
                current_round=self.model.current_training_round
            )
            
        except Exception as e:
            logger.error(f"GossipGradient error: {e}")
            return neuroshard_pb2.GossipGradientResponse(
                accepted=False,
                reason=str(e)
            )
    
    def GetCheckpointInfo(self, request, context):
        """Get checkpoint info without downloading the full checkpoint."""
        if not self.is_neuro_node:
            return neuroshard_pb2.GetCheckpointInfoResponse()
        
        try:
            # DynamicNeuroNode may not have get_checkpoint_info() —
            # build the info directly from available attributes
            if hasattr(self.model, 'get_checkpoint_info'):
                info = self.model.get_checkpoint_info()
            else:
                # Build info from DynamicNeuroNode attributes
                model = getattr(self.model, 'model', None)
                info = {
                    "version": getattr(self.model, 'total_training_rounds', 0),
                    "model_hash": self.model._get_model_hash() if hasattr(self.model, '_get_model_hash') else "",
                    "phase": getattr(self.model, 'phase', 'training'),
                    "params": sum(p.numel() for p in model.parameters()) if model else 0,
                    "loss": getattr(self.model, 'current_loss', float('inf')),
                }
            
            return neuroshard_pb2.GetCheckpointInfoResponse(
                version=info.get("version", 0),
                model_hash=info.get("model_hash", ""),
                phase=info.get("phase", "training"),
                params=info.get("params", 0),
                loss=info.get("loss", float('inf'))
            )
            
        except Exception as e:
            logger.error(f"GetCheckpointInfo error: {e}")
            return neuroshard_pb2.GetCheckpointInfoResponse()
    
    def GetCheckpoint(self, request, context):
        """Download full checkpoint from this node."""
        if not self.is_neuro_node:
            return neuroshard_pb2.GetCheckpointResponse(
                success=False,
                error_message="Node does not support checkpoint sync"
            )
        
        try:
            import io
            import zlib
            
            # Get model checkpoint
            if not self.model.model:
                return neuroshard_pb2.GetCheckpointResponse(
                    success=False,
                    error_message="Model not loaded"
                )
            
            # Serialize checkpoint using manual collection (same as save_checkpoint).
            # We cannot use model.state_dict() because DynamicNeuroLLM stores
            # layers in a plain Dict (not nn.ModuleDict), so state_dict() misses them.
            buffer = io.BytesIO()
            model = self.model.model
            arch = getattr(model, 'architecture', None) or getattr(model, 'config', None)
            
            state_dict = {}
            # Collect embedding
            if getattr(model, 'embedding', None) is not None:
                state_dict["embedding.weight"] = model.embedding.weight.cpu()
            # Collect LM head
            if getattr(model, 'lm_head', None) is not None:
                state_dict["lm_head.weight"] = model.lm_head.weight.cpu()
            if getattr(model, 'final_norm', None) is not None:
                for name, param in model.final_norm.named_parameters():
                    state_dict[f"final_norm.{name}"] = param.cpu()
            # Collect ALL layers (my_layers is a plain Dict, not in state_dict())
            for layer_id, layer in getattr(model, 'my_layers', {}).items():
                for name, param in layer.named_parameters():
                    state_dict[f"layer_{layer_id}.{name}"] = param.cpu()
            
            layer_ids = sorted(getattr(model, 'my_layer_ids', []))
            
            checkpoint = {
                "state_dict": state_dict,  # Same format as save_checkpoint
                "layer_ids": layer_ids,
                "config": {
                    "phase": getattr(self.model, 'phase', 'training'),
                    "hidden_dim": arch.hidden_dim if arch else 0,
                    "num_layers": arch.num_layers if arch else 0,
                    "vocab_size": getattr(arch, 'vocab_size', 0) if arch else 0,
                },
                "training_rounds": self.model.total_training_rounds,
                "version": self.model.total_training_rounds,
                "model_hash": self.model._get_model_hash() if hasattr(self.model, '_get_model_hash') else "",
            }
            torch.save(checkpoint, buffer)
            
            # Compress
            raw_data = buffer.getvalue()
            compressed = zlib.compress(raw_data, level=6)
            
            logger.info(f"Serving checkpoint: version={checkpoint['version']}, "
                       f"size={len(compressed)/1024:.1f}KB (compressed from {len(raw_data)/1024:.1f}KB)")
            
            return neuroshard_pb2.GetCheckpointResponse(
                success=True,
                version=checkpoint["version"],
                model_hash=checkpoint["model_hash"],
                phase=self.model.phase,
                checkpoint_data=compressed,
                total_size=len(compressed)
            )
            
        except Exception as e:
            logger.error(f"GetCheckpoint error: {e}")
            return neuroshard_pb2.GetCheckpointResponse(
                success=False,
                error_message=str(e)
            )

    # ==================== PIPELINE PARALLELISM RPCs ====================
    
    def PipelineForward(self, request, context):
        """
        Pipeline Forward for distributed training and inference.
        
        PRODUCTION-READY MULTI-HOP PIPELINE:
        1. Receive hidden states from previous node (DRIVER or WORKER)
        2. Process through MY layers
        3. Check for next_hop:
           - If next_hop exists → forward to next node, return their response
           - If no next_hop (I'm VALIDATOR) → compute loss, send gradients back
        
        This enables: DRIVER → WORKER₁ → WORKER₂ → ... → VALIDATOR
        Any number of nodes in the pipeline!
        
        QUORUM MODE: When a QuorumTrainer is active and has an inbound_buffer,
        training activations are routed through the buffer instead of being
        processed synchronously. The QuorumTrainer's _process_and_forward()
        pulls from the buffer and handles the pipeline locally.
        """
        # ── QUORUM BUFFER PATH ─────────────────────────────────────────────
        # If we have an active QuorumTrainer with an inbound buffer,
        # route training activations through it for the trainer to process.
        if self.quorum_inbound_buffer is not None and request.training_labels:
            try:
                import numpy as np
                import queue
                
                # Deserialize hidden states
                hidden_states = torch.from_numpy(
                    np.frombuffer(request.hidden_states, dtype=np.float32).copy()
                ).reshape(list(request.hidden_shape))
                
                # Deserialize training labels
                batch_size = request.hidden_shape[0]
                seq_len = request.hidden_shape[1]
                training_labels = torch.from_numpy(
                    np.frombuffer(request.training_labels, dtype=np.int64).copy()
                ).reshape(batch_size, seq_len)
                
                # Create a simple packet object for the QuorumTrainer
                class _ActivationPacket:
                    pass
                
                packet = _ActivationPacket()
                packet.tensor_data = hidden_states
                packet.session_id = request.session_id
                packet.grad_output = None
                packet.labels = training_labels
                
                # Push to buffer (non-blocking with short timeout)
                try:
                    self.quorum_inbound_buffer.put(packet, timeout=1.0)
                    logger.debug(f"[PIPELINE] Routed activation to QuorumTrainer buffer "
                               f"(shape={list(hidden_states.shape)})")
                    return neuroshard_pb2.PipelineForwardResponse(
                        request_id=request.request_id,
                        success=True,
                    )
                except queue.Full:
                    logger.warning("[PIPELINE] QuorumTrainer buffer full, processing synchronously")
                    # Fall through to synchronous processing
                    
            except Exception as e:
                logger.warning(f"[PIPELINE] Buffer routing failed ({e}), processing synchronously")
                # Fall through to synchronous processing
        
        # ── SYNCHRONOUS PATH (inference or no quorum buffer) ───────────────
        if not hasattr(self.model, 'forward_pipeline'):
            return neuroshard_pb2.PipelineForwardResponse(
                success=False,
                error_message="Node does not support forward_pipeline"
            )
        
        try:
            import numpy as np
            from neuroshard.core.network.connection_pool import get_channel
            from urllib.parse import urlparse
            
            logger.info(f"[PIPELINE] Received forward request {request.request_id[:8]}... from {request.sender_url}")
            
            # STEP 1: Deserialize hidden states
            hidden_states = torch.from_numpy(
                np.frombuffer(request.hidden_states, dtype=np.float32).copy()
            ).reshape(list(request.hidden_shape))
            
            model_device = self.model.device if hasattr(self.model, 'device') else 'cpu'
            hidden_states = hidden_states.to(model_device)
            
            logger.info(f"[PIPELINE] Hidden states: {hidden_states.shape} on {hidden_states.device}")
            
            # STEP 2: Deserialize attention mask (optional)
            attention_mask = None
            if request.attention_mask:
                attention_mask = torch.from_numpy(
                    np.frombuffer(request.attention_mask, dtype=np.float32).copy()
                ).to(model_device)
            
            # STEP 3: Deserialize training labels (for training mode)
            training_labels = None
            is_training = False
            if request.training_labels:
                batch_size = request.hidden_shape[0]
                seq_len = request.hidden_shape[1]
                training_labels = torch.from_numpy(
                    np.frombuffer(request.training_labels, dtype=np.int64).copy()
                ).reshape(batch_size, seq_len).to(model_device)
                is_training = True
                logger.info(f"[PIPELINE] Training mode: labels shape {training_labels.shape}")
            
            # STEP 4: Forward through MY layers (no loss computation here)
            output, _ = self.model.model.forward_my_layers(hidden_states, attention_mask)
            
            # STEP 5: Check for next_hop - KEY for multi-hop pipeline!
            my_last_layer = max(self.model.my_layer_ids) if self.model.my_layer_ids else 0
            next_layer = my_last_layer + 1
            
            next_hop_url = None
            if hasattr(self.model, 'p2p_manager') and self.model.p2p_manager:
                next_hop_url = self.model.p2p_manager.get_next_hop(next_layer)
            
            # STEP 6: Submit PoNW proof for MY work (before forwarding)
            self._submit_pipeline_proof(request, output, is_training, is_final=(next_hop_url is None))
            
            # STEP 7: Route based on network topology
            if next_hop_url:
                # ═══════════════════════════════════════════════════════════════
                # MULTI-HOP: Forward to next node and return THEIR response
                # ═══════════════════════════════════════════════════════════════
                logger.info(f"[PIPELINE] Forwarding to next hop: {next_hop_url} (layer {next_layer})")
                
                # Derive gRPC address from HTTP URL
                parsed = urlparse(next_hop_url)
                next_http_port = parsed.port or 8000
                next_grpc_port = next_http_port + 1000
                next_grpc_addr = f"{parsed.hostname}:{next_grpc_port}"
                
                # Serialize output for next hop
                output_bytes = output.detach().cpu().numpy().astype(np.float32).tobytes()
                output_shape = list(output.shape)
                
                # Build forward request for next hop
                next_request = neuroshard_pb2.PipelineForwardRequest(
                    request_id=request.request_id,
                    session_id=request.session_id,
                    hidden_states=output_bytes,
                    hidden_shape=output_shape,
                    source_shard=my_last_layer,  # My last layer
                    sender_url=request.sender_url,  # Original DRIVER URL (for gradients)
                    attention_mask=request.attention_mask,
                    training_labels=request.training_labels,  # Pass through labels
                )
                
                # Forward to next node
                try:
                    channel = get_channel(next_grpc_addr)
                    stub = neuroshard_pb2_grpc.NeuroShardServiceStub(channel)
                    
                    # Call next hop and return their response directly
                    next_response = stub.PipelineForward(next_request, timeout=60.0)
                    
                    logger.info(f"[PIPELINE] Next hop response: is_final={next_response.is_final}, loss={next_response.loss if next_response.is_final else 'N/A'}")
                    return next_response
                    
                except Exception as e:
                    logger.error(f"[PIPELINE] Failed to forward to {next_grpc_addr}: {e}")
                    return neuroshard_pb2.PipelineForwardResponse(
                        request_id=request.request_id,
                        success=False,
                        error_message=f"Multi-hop forward failed: {e}"
                    )
            
            else:
                # ═══════════════════════════════════════════════════════════════
                # FINAL NODE (VALIDATOR): Compute loss and send gradients back
                # ═══════════════════════════════════════════════════════════════
                logger.info(f"[VALIDATOR] I am final node - computing loss")
                
                loss = None
                grad_output = None
                
                if is_training and training_labels is not None and self.model.model.has_lm_head:
                    # Save hidden states for gradient computation
                    hidden_states_for_grad = hidden_states.clone().requires_grad_(True)
                    output_for_loss, _ = self.model.model.forward_my_layers(hidden_states_for_grad, attention_mask)
                    
                    with self.model._training_lock:
                        self.model.optimizer.zero_grad()
                        
                        logits = self.model.model.compute_logits(output_for_loss)
                        logits_flat = logits.view(-1, logits.size(-1))
                        labels_flat = training_labels.view(-1)
                        loss_tensor = torch.nn.functional.cross_entropy(logits_flat, labels_flat, ignore_index=-100)
                        
                        loss_tensor.backward()
                        torch.nn.utils.clip_grad_norm_(self.model.model.parameters(), 1.0)
                        self.model.optimizer.step()
                        
                        loss = loss_tensor.item()
                        grad_output = hidden_states_for_grad.grad
                        
                        logger.info(f"[VALIDATOR] ✅ Loss: {loss:.4f}")
                
                # Build response
                output_bytes = output.detach().cpu().numpy().astype(np.float32).tobytes()
                output_shape = list(output.shape)
                
                response = neuroshard_pb2.PipelineForwardResponse(
                    request_id=request.request_id,
                    success=True,
                    hidden_states=output_bytes,
                    hidden_shape=output_shape,
                    is_final=True,
                )
                
                if loss is not None:
                    response.loss = loss
                    
                    # Send gradients back to DRIVER for backpropagation
                    if grad_output is not None and request.sender_url:
                        self._send_gradients_to_driver(
                            grad_output=grad_output,
                            sender_url=request.sender_url,
                            session_id=request.session_id,
                            request_id=request.request_id,
                            source_shard=request.source_shard,
                        )
                
                return response
            
        except Exception as e:
            logger.error(f"[PIPELINE] Error: {e}")
            import traceback
            traceback.print_exc()
            return neuroshard_pb2.PipelineForwardResponse(
                request_id=request.request_id,
                success=False,
                error_message=str(e)
            )
    
    def _submit_pipeline_proof(self, request, output, is_training: bool, is_final: bool):
        """Submit PoNW proof for pipeline work."""
        if not hasattr(self.model, 'ledger') or not self.model.ledger:
            return
        
        try:
            from neuroshard.core.economics.ledger import PoNWProof, sign_proof
            import uuid
            
            tokens_processed = output.shape[1] if len(output.shape) > 1 else output.shape[0]
            
            proof = PoNWProof(
                node_id=self.model.node_id,
                proof_type="training" if is_training else "inference",
                timestamp=time.time(),
                nonce=str(uuid.uuid4()),
                tokens_processed=tokens_processed,
                request_id=request.request_id,
                has_embedding=self.model.model.has_embedding,
                has_lm_head=self.model.model.has_lm_head,
                layers_held=len(self.model.my_layer_ids)
            )
            
            signed_proof = sign_proof(proof, self.model.node_token)
            success, reward, msg = self.model.ledger.process_proof(signed_proof)
            
            if success:
                role = "VALIDATOR" if is_final else "WORKER"
                logger.info(f"[{role}] ✅ Earned {reward:.6f} NEURO")
        except Exception as e:
            logger.warning(f"[PIPELINE] PoNW proof failed: {e}")
    
    def _send_gradients_to_driver(self, grad_output: torch.Tensor, sender_url: str,
                                   session_id: str, request_id: str, source_shard: int):
        """Send gradients back to DRIVER for distributed backpropagation."""
        try:
            import numpy as np
            from neuroshard.core.network.connection_pool import get_channel
            from urllib.parse import urlparse
            
            # Parse sender URL and derive gRPC address
            parsed = urlparse(sender_url)
            sender_http_port = parsed.port or 8000
            sender_grpc_port = sender_http_port + 1000
            sender_grpc_addr = f"{parsed.hostname}:{sender_grpc_port}"
            
            logger.info(f"[VALIDATOR] Sending gradients to DRIVER at {sender_grpc_addr}")
            
            # Serialize gradient
            grad_bytes = grad_output.detach().cpu().numpy().astype(np.float32).tobytes()
            grad_shape = list(grad_output.shape)
            
            # Send PipelineBackward request
            channel = get_channel(sender_grpc_addr)
            stub = neuroshard_pb2_grpc.NeuroShardServiceStub(channel)
            
            backward_request = neuroshard_pb2.PipelineBackwardRequest(
                session_id=session_id,
                request_id=request_id,
                grad_output=grad_bytes,
                grad_shape=grad_shape,
                target_shard=source_shard,
            )
            
            backward_response = stub.PipelineBackward(backward_request, timeout=30)
            
            if backward_response.success:
                logger.info(f"[VALIDATOR] ✅ Gradients sent to DRIVER successfully")
            else:
                logger.warning(f"[VALIDATOR] ⚠️ DRIVER rejected gradients: {backward_response.error_message}")
                
        except Exception as e:
            logger.warning(f"[VALIDATOR] Failed to send gradients to DRIVER: {e}")

    def PipelineBackward(self, request, context):
        """
        Backward pass: propagate gradients back to previous node.
        """
        if not hasattr(self.model, 'backward_pipeline'):
            return neuroshard_pb2.PipelineBackwardResponse(
                success=False,
                error_message="Node does not support backward pipeline"
            )
            
        try:
            import numpy as np
            
            # Deserialize gradients
            grad_output = torch.from_numpy(
                np.frombuffer(request.grad_output, dtype=np.float32).copy()
            ).reshape(list(request.grad_shape))
            
            # CRITICAL: Move gradients to the same device as the model!
            model_device = getattr(self.model, 'device', 'cpu')
            if hasattr(self.model, 'model') and hasattr(self.model.model, 'device'):
                model_device = self.model.model.device
            grad_output = grad_output.to(model_device)
            
            # Run backward pipeline
            self.model.backward_pipeline(
                grad_output=grad_output,
                session_id=request.session_id
            )
            
            return neuroshard_pb2.PipelineBackwardResponse(success=True)
            
        except Exception as e:
            logger.error(f"PipelineBackward error: {e}")
            return neuroshard_pb2.PipelineBackwardResponse(
                success=False, 
                error_message=str(e)
            )

    def GetShardChunk(self, request, context):
        """
        Serve a chunk of a data shard to a peer (Data Swarm).
        """
        if not hasattr(self.model, 'swarm') or not self.model.swarm:
            return neuroshard_pb2.GetShardChunkResponse(
                success=False,
                error_message="Swarm not initialized on this node"
            )
            
        chunk_data = self.model.swarm.serve_chunk(request.shard_id, request.chunk_index)
        
        if chunk_data:
            return neuroshard_pb2.GetShardChunkResponse(
                success=True,
                data=chunk_data
            )
        else:
            return neuroshard_pb2.GetShardChunkResponse(
                success=False,
                error_message="Chunk not found"
            )
    
    def GetShardInfo(self, request, context):
        """Get shard/layer information from this node."""
        try:
            # Get layer range
            layer_ids = self.model.my_layer_ids if hasattr(self.model, 'my_layer_ids') else []
            start_layer = min(layer_ids) if layer_ids else 0
            end_layer = max(layer_ids) + 1 if layer_ids else 0
            
            return neuroshard_pb2.GetShardInfoResponse(
                shard_id=start_layer,  # Use start layer as shard ID
                total_shards=self.model.model.architecture.num_layers if hasattr(self.model.model, 'architecture') else 11,
                start_layer=start_layer,
                end_layer=end_layer,
                has_embedding=self.model.model.has_embedding,
                has_lm_head=self.model.model.has_lm_head,
                version=getattr(self.model, 'total_training_rounds', 0),
                model_hash=self.model._get_model_hash() if hasattr(self.model, '_get_model_hash') else "",
                available_memory_mb=int(self.model.available_memory_mb) if hasattr(self.model, 'available_memory_mb') else 0,
                current_load=0,
            )
            
        except Exception as e:
            logger.error(f"GetShardInfo error: {e}")
            return neuroshard_pb2.GetShardInfoResponse()
    
    # ==================== SWARM RPCs (Phase 4) ====================
    
    def SwarmForward(self, request, context):
        """
        Swarm-style activation forwarding with async buffering.
        
        This is the async-first activation forwarding for the swarm architecture.
        Activations are queued in the inbound buffer and processed by ComputeEngine.
        
        Unlike PipelineForward (sync), this returns immediately after queueing.
        """
        try:
            import numpy as np
            from neuroshard.core.swarm.buffers import ActivationPacket, ActivationPriority
            
            # Deserialize activation
            hidden_states = torch.from_numpy(
                np.frombuffer(request.hidden_states, dtype=np.float32)
            ).reshape(list(request.hidden_shape))
            
            # Create activation packet
            packet = ActivationPacket(
                priority=request.priority or ActivationPriority.INFERENCE_NORMAL,
                timestamp=time.time(),
                session_id=request.session_id,
                micro_batch_id=request.micro_batch_id,
                tensor_data=hidden_states,
                source_node=request.source_node,
                target_layer=request.target_layer,
                is_backward=request.is_backward,
                requires_grad=request.requires_grad,
            )
            
            # Queue in inbound buffer (non-blocking)
            # Note: swarm_components contains SwarmComponents (router, buffers, etc.)
            # Not to be confused with swarm (DataSwarm for P2P downloads)
            inbound = getattr(self.model, 'swarm_components', None)
            inbound = inbound.inbound_buffer if inbound else None
            if inbound:
                queue_success = inbound.put_nowait(packet)
                if queue_success:
                    return neuroshard_pb2.SwarmForwardResponse(
                        request_id=request.request_id,
                        success=True,
                        buffer_depth=len(inbound),
                    )
                else:
                    # Buffer full - backpressure
                    return neuroshard_pb2.SwarmForwardResponse(
                        request_id=request.request_id,
                        success=False,
                        error_message="Inbound buffer full (backpressure)",
                        buffer_depth=inbound.capacity if hasattr(inbound, 'capacity') else 0,
                    )
            else:
                return neuroshard_pb2.SwarmForwardResponse(
                    request_id=request.request_id,
                    success=False,
                    error_message="Inbound buffer not initialized"
                )
                
        except Exception as e:
            logger.error(f"SwarmForward error: {e}")
            return neuroshard_pb2.SwarmForwardResponse(
                request_id=request.request_id if hasattr(request, 'request_id') else "",
                success=False,
                error_message=str(e)
            )
    
    def GetSwarmStatus(self, request, context):
        """
        Get swarm node status: buffer fill rates, capacity, etc.
        
        Used by peers to check node health before routing.
        """
        try:
            status = self.model.get_swarm_status()
            
            # Extract key metrics
            inbound_fill = 0.0
            outbound_fill = 0.0
            inbound_depth = 0
            outbound_depth = 0
            
            if "inbound_buffer" in status:
                inbound_fill = status["inbound_buffer"].get("fill_rate", 0.0)
                inbound_depth = status["inbound_buffer"].get("queue_size", 0)
            if "outbound_buffer" in status:
                outbound_fill = status["outbound_buffer"].get("fill_rate", 0.0)
                outbound_depth = status["outbound_buffer"].get("queue_size", 0)
            
            # Get layer range
            layer_start = min(self.model.my_layer_ids) if self.model.my_layer_ids else 0
            layer_end = max(self.model.my_layer_ids) + 1 if self.model.my_layer_ids else 0
            
            return neuroshard_pb2.SwarmStatusResponse(
                node_id=self.model.node_id,
                layer_start=layer_start,
                layer_end=layer_end,
                inbound_fill_rate=inbound_fill,
                outbound_fill_rate=outbound_fill,
                inbound_queue_depth=inbound_depth,
                outbound_queue_depth=outbound_depth,
                is_accepting_activations=inbound_fill < 0.95,
            )
            
        except Exception as e:
            logger.error(f"GetSwarmStatus error: {e}")
            return neuroshard_pb2.SwarmStatusResponse(error_message=str(e))
    
    def UpdatePeerCapacity(self, request, context):
        """
        Receive capacity update from peer (TCP fallback for UDP heartbeat).
        
        Used when UDP heartbeats fail (firewalls, etc).
        """
        try:
            # Update router with peer info
            # Note: swarm_components contains SwarmComponents (router, buffers, etc.)
            router = getattr(self.model, 'swarm_components', None)
            router = router.swarm_router if router else None
            if router:
                from neuroshard.core.swarm.heartbeat import CapacityBitmask
                
                capacity = CapacityBitmask(
                    node_id=request.node_id,
                    timestamp=time.time(),
                    available_memory_mb=request.available_memory_mb,
                    queue_depth=request.queue_depth,
                    layer_range=(request.layer_start, request.layer_end),
                    gpu_utilization=request.gpu_utilization,
                    network_saturation=request.network_saturation,
                    is_training=request.is_training,
                    is_accepting_inference=request.is_accepting_inference,
                    is_accepting_activations=request.is_accepting_activations,
                    grpc_addr=request.grpc_addr,
                )
                
                router.update_peer_from_heartbeat(capacity)
                
                return neuroshard_pb2.UpdatePeerCapacityResponse(success=True)
            
            return neuroshard_pb2.UpdatePeerCapacityResponse(success=False)
            
        except Exception as e:
            logger.error(f"UpdatePeerCapacity error: {e}")
            return neuroshard_pb2.UpdatePeerCapacityResponse(success=False, error_message=str(e))
    
    # ==================== VALIDATOR CONSENSUS RPCs ====================
    
    def RequestProofValidation(self, request, context):
        """
        Handle proof validation request from another node.
        
        When a node submits a PoNW proof, it requests validation from
        selected validators (stake-weighted random selection).
        
        This node:
        1. Verifies it is an eligible validator
        2. Validates the proof locally
        3. Casts a vote (valid/invalid)
        4. Returns the vote to be gossiped
        
        From whitepaper Section 7: Hybrid Validator System
        """
        try:
            # Check if we're an eligible validator
            if not hasattr(self.model, 'layer_pool') or not self.model.layer_pool:
                return neuroshard_pb2.ProofValidationResponse(
                    accepted=False,
                    reason="Node does not have layer pool"
                )
            
            # Check if we have LM head (required for validator)
            has_lm_head = self.model.model.has_lm_head if hasattr(self.model, 'model') else False
            if not has_lm_head:
                return neuroshard_pb2.ProofValidationResponse(
                    accepted=False,
                    reason="Not a validator (no LM head)"
                )
            
            # Get our stake
            stake = 0.0
            if hasattr(self.model, 'ledger') and self.model.ledger:
                stake = self.model.ledger.get_local_stake(self.model.node_id)
            
            # Check minimum stake requirement
            from neuroshard.core.economics.constants import get_dynamic_validator_stake
            num_validators = len([n for n in self.model.layer_pool.node_capacities 
                                 if n != self.model.node_id]) + 1
            required_stake = get_dynamic_validator_stake(num_validators)
            
            if stake < required_stake:
                return neuroshard_pb2.ProofValidationResponse(
                    accepted=False,
                    reason=f"Insufficient stake: {stake:.2f} < {required_stake:.2f} NEURO required"
                )
            
            logger.info(f"[VALIDATOR] Accepted validation request for proof {request.proof_signature[:16]}...")
            
            return neuroshard_pb2.ProofValidationResponse(
                accepted=True,
                reason="Validation request accepted",
                validator_id=self.model.node_id,
                validator_stake=stake
            )
            
        except Exception as e:
            logger.error(f"RequestProofValidation error: {e}")
            return neuroshard_pb2.ProofValidationResponse(
                accepted=False,
                reason=str(e)
            )
    
    def GossipValidationVote(self, request, context):
        """
        Receive a validation vote from a validator.
        
        Validators cast stake-weighted votes on proofs:
        - vote = True: Proof is valid
        - vote = False: Proof is invalid
        
        Consensus is reached when 66% of stake agrees.
        Validators who vote against consensus are slashed (2x penalty).
        
        From whitepaper Section 7.3: Stake-Weighted Proof Validation
        """
        try:
            logger.info(f"[CONSENSUS] Received vote from {request.validator_id[:16]}... "
                       f"on proof {request.proof_signature[:16]}...: "
                       f"{'VALID' if request.vote else 'INVALID'} (stake={request.validator_stake:.2f})")
            
            # Get or create consensus tracker
            if not hasattr(self, '_pending_consensus'):
                self._pending_consensus = {}
            
            proof_sig = request.proof_signature
            if proof_sig not in self._pending_consensus:
                self._pending_consensus[proof_sig] = {
                    'valid_stake': 0.0,
                    'invalid_stake': 0.0,
                    'votes': [],
                    'created_at': time.time(),
                }
            
            consensus = self._pending_consensus[proof_sig]
            
            # Record vote
            consensus['votes'].append({
                'validator_id': request.validator_id,
                'stake': request.validator_stake,
                'vote': request.vote,
                'timestamp': request.timestamp,
            })
            
            if request.vote:
                consensus['valid_stake'] += request.validator_stake
            else:
                consensus['invalid_stake'] += request.validator_stake
            
            total_stake = consensus['valid_stake'] + consensus['invalid_stake']
            valid_ratio = consensus['valid_stake'] / total_stake if total_stake > 0 else 0
            
            # Check consensus (66% threshold)
            from neuroshard.core.economics.constants import VALIDATION_CONSENSUS_THRESHOLD
            consensus_reached = (
                valid_ratio >= VALIDATION_CONSENSUS_THRESHOLD or
                (1 - valid_ratio) >= VALIDATION_CONSENSUS_THRESHOLD
            )
            
            consensus_result = False
            if consensus_reached:
                consensus_result = valid_ratio >= VALIDATION_CONSENSUS_THRESHOLD
                logger.info(f"[CONSENSUS] Reached for proof {proof_sig[:16]}...: "
                           f"{'ACCEPTED' if consensus_result else 'REJECTED'} "
                           f"({valid_ratio:.1%} valid stake)")
            
            return neuroshard_pb2.ValidationVoteResponse(
                accepted=True,
                reason="Vote recorded",
                total_valid_stake=consensus['valid_stake'],
                total_invalid_stake=consensus['invalid_stake'],
                consensus_reached=consensus_reached,
                consensus_result=consensus_result,
            )
            
        except Exception as e:
            logger.error(f"GossipValidationVote error: {e}")
            return neuroshard_pb2.ValidationVoteResponse(
                accepted=False,
                reason=str(e)
            )

    # ==================== MIXTURE OF EXPERTS (MoE) RPCs ====================
    
    def ExpertForward(self, request, context):
        """
        Forward activations through a specific expert.
        
        This enables distributed MoE where experts are spread across nodes.
        The router runs locally, but expert computation may be remote.
        """
        try:
            import numpy as np
            
            layer_id = request.layer_id
            expert_id = request.expert_id
            
            # Check if we have this expert
            if not self.is_dynamic_node:
                return neuroshard_pb2.ExpertForwardResponse(
                    request_id=request.request_id,
                    success=False,
                    error_message="Not a dynamic node"
                )
            
            # Check if expert is available on this node
            if not hasattr(self.model, 'my_experts'):
                return neuroshard_pb2.ExpertForwardResponse(
                    request_id=request.request_id,
                    success=False,
                    error_message="MoE not enabled on this node"
                )
            
            my_experts = self.model.my_experts.get(layer_id, [])
            if expert_id not in my_experts:
                return neuroshard_pb2.ExpertForwardResponse(
                    request_id=request.request_id,
                    success=False,
                    error_message=f"Expert {expert_id} not on this node (have {my_experts})"
                )
            
            # Deserialize input tensor
            activations = torch.from_numpy(
                np.frombuffer(request.activations, dtype=np.float32)
            ).reshape(list(request.shape))
            
            # Get the layer and run expert
            layer = self.model.my_layers.get(layer_id)
            if layer is None or not hasattr(layer, 'moe'):
                return neuroshard_pb2.ExpertForwardResponse(
                    request_id=request.request_id,
                    success=False,
                    error_message=f"Layer {layer_id} not found or not MoE"
                )
            
            # Forward through the specific expert
            moe = layer.moe
            if str(expert_id) not in moe.local_experts:
                return neuroshard_pb2.ExpertForwardResponse(
                    request_id=request.request_id,
                    success=False,
                    error_message=f"Expert {expert_id} not initialized"
                )
            
            # Move to device and forward
            activations = activations.to(self.model.device)
            with torch.no_grad():
                expert = moe.local_experts[str(expert_id)]
                output = expert(activations)
            
            # Serialize output
            output_np = output.cpu().numpy().astype(np.float32)
            output_bytes = output_np.tobytes()
            
            # Get utilization stats
            utilization = moe.expert_counts[expert_id].item() / max(1, moe.total_tokens.item())
            
            logger.debug(f"[MoE] ExpertForward: layer={layer_id}, expert={expert_id}, "
                        f"shape={list(activations.shape)}")
            
            return neuroshard_pb2.ExpertForwardResponse(
                request_id=request.request_id,
                success=True,
                output=output_bytes,
                shape=list(output.shape),
                utilization=utilization,
                queue_depth=0
            )
            
        except Exception as e:
            logger.error(f"ExpertForward error: {e}")
            return neuroshard_pb2.ExpertForwardResponse(
                request_id=request.request_id,
                success=False,
                error_message=str(e)
            )
    
    def GetExpertStatus(self, request, context):
        """Get status of a specific expert for routing decisions."""
        try:
            layer_id = request.layer_id
            expert_id = request.expert_id
            
            if not self.is_dynamic_node or not hasattr(self.model, 'my_experts'):
                return neuroshard_pb2.ExpertStatusResponse(available=False)
            
            my_experts = self.model.my_experts.get(layer_id, [])
            available = expert_id in my_experts
            
            utilization = 0.0
            queue_depth = 0
            
            if available and layer_id in self.model.my_layers:
                layer = self.model.my_layers[layer_id]
                if hasattr(layer, 'moe'):
                    moe = layer.moe
                    if moe.total_tokens.item() > 0:
                        utilization = moe.expert_counts[expert_id].item() / moe.total_tokens.item()
            
            # Get replica count from layer pool
            replicas = 1
            if hasattr(self.model, 'layer_pool'):
                holders = self.model.layer_pool.get_expert_holders(layer_id, expert_id)
                replicas = len(holders)
            
            return neuroshard_pb2.ExpertStatusResponse(
                available=available,
                utilization=utilization,
                queue_depth=queue_depth,
                capacity=1000.0,  # tokens/second estimate
                replicas=replicas
            )
            
        except Exception as e:
            logger.error(f"GetExpertStatus error: {e}")
            return neuroshard_pb2.ExpertStatusResponse(available=False)
    
    def AnnounceExpert(self, request, context):
        """Receive expert availability announcement."""
        try:
            from neuroshard.core.model.dynamic import ExpertAssignment
            
            layer_id = request.layer_id
            expert_id = request.expert_id
            node_id = request.node_id
            
            # Update layer pool with expert info
            if hasattr(self.model, 'layer_pool'):
                # The layer pool will track this for routing
                self.model.layer_pool.expert_assignments.setdefault(layer_id, {})
                self.model.layer_pool.expert_assignments[layer_id].setdefault(expert_id, [])
                
                # Add this node if not already tracked
                holders = self.model.layer_pool.expert_assignments[layer_id][expert_id]
                if node_id not in holders:
                    holders.append(node_id)
                
                # Calculate scarcity bonus
                scarcity_bonus = self.model.layer_pool.get_expert_scarcity_bonus(layer_id, expert_id)
                total_replicas = len(holders)
                
                logger.debug(f"[MoE] Expert announced: layer={layer_id}, expert={expert_id}, "
                            f"node={node_id[:8]}, replicas={total_replicas}, bonus={scarcity_bonus:.2f}")
                
                return neuroshard_pb2.AnnounceExpertResponse(
                    accepted=True,
                    total_replicas=total_replicas,
                    scarcity_bonus=scarcity_bonus
                )
            
            return neuroshard_pb2.AnnounceExpertResponse(
                accepted=False,
                reason="No layer pool available"
            )
            
        except Exception as e:
            logger.error(f"AnnounceExpert error: {e}")
            return neuroshard_pb2.AnnounceExpertResponse(
                accepted=False,
                reason=str(e)
            )


    # ==================== QUORUM FORMATION RPCs ====================
    
    def ProposeQuorum(self, request, context):
        """
        Handle a quorum proposal from another node.
        
        Per whitepaper Algorithm 2: "Propose quorum to all selected members.
        If all accept, register quorum in DHT."
        
        Accept if:
        - Not already in an ACTIVE pipeline quorum
        - Speed tier is compatible
        - We have layers that complement the proposer
        
        On acceptance, the runner callback (on_quorum_proposed) is invoked to
        set up the quorum, start QuorumTrainer, and switch to PIPELINE mode.
        """
        try:
            from neuroshard.core.model.dynamic import COMPATIBLE_TIERS, SpeedTier
            
            proposer_node_id = request.proposer_node_id
            proposer_endpoint = request.proposer_endpoint
            proposer_layers = (request.proposer_layer_start, request.proposer_layer_end)
            proposed_tier = request.speed_tier
            quorum_id = request.quorum_id
            total_layers = request.total_layers
            
            logger.info(f"[QUORUM] Received ProposeQuorum from {proposer_node_id[:8]}... "
                       f"quorum={quorum_id[:8]}, layers={proposer_layers}, tier={proposed_tier}")
            
            # Check 1: Do we have layer info?
            if not self.is_dynamic_node:
                return neuroshard_pb2.ProposeQuorumResponse(
                    accepted=False,
                    reason="Not a dynamic node"
                )
            
            my_node_id = self.model.node_id
            my_layer_ids = self.model.my_layer_ids
            if not my_layer_ids:
                return neuroshard_pb2.ProposeQuorumResponse(
                    accepted=False,
                    node_id=my_node_id,
                    reason="No layers assigned"
                )
            
            my_layer_start = min(my_layer_ids)
            my_layer_end = max(my_layer_ids) + 1
            
            # Check 2: Are we already in an ACTIVE quorum?
            # SYNC quorums (data-parallel) accept new peers — more sync partners = better.
            # PIPELINE quorums reject — they're already tightly coupled.
            if self._active_quorum_id:
                if self._active_quorum_type == "sync":
                    # SYNC quorum: accept proposal to add sync peer
                    logger.info(f"[QUORUM] In SYNC quorum {self._active_quorum_id[:8]}... — "
                               f"accepting proposal for sync peer")
                else:
                    return neuroshard_pb2.ProposeQuorumResponse(
                        accepted=False,
                        node_id=my_node_id,
                        layer_start=my_layer_start,
                        layer_end=my_layer_end,
                        reason="Already in active pipeline quorum",
                        current_quorum_id=self._active_quorum_id
                    )
            
            # Check 3: Speed tier compatibility
            my_speed_tier = getattr(self.model, '_speed_tier', None) or "tier3"
            # Accept same tier or adjacent tiers
            compatible = False
            try:
                my_tier_enum = SpeedTier(my_speed_tier)
                proposed_tier_enum = SpeedTier(proposed_tier)
                if my_tier_enum in COMPATIBLE_TIERS.get(proposed_tier_enum, set()):
                    compatible = True
                elif proposed_tier_enum in COMPATIBLE_TIERS.get(my_tier_enum, set()):
                    compatible = True
                elif my_speed_tier == proposed_tier:
                    compatible = True
            except (ValueError, KeyError):
                # If tiers can't be parsed, accept same-tier only
                compatible = (my_speed_tier == proposed_tier)
            
            if not compatible:
                return neuroshard_pb2.ProposeQuorumResponse(
                    accepted=False,
                    node_id=my_node_id,
                    layer_start=my_layer_start,
                    layer_end=my_layer_end,
                    speed_tier=my_speed_tier,
                    reason=f"Speed tier incompatible: mine={my_speed_tier}, proposed={proposed_tier}"
                )
            
            # Check 4: Do our layers complement the proposer's?
            my_layers_set = set(range(my_layer_start, my_layer_end))
            proposer_layers_set = set(range(proposer_layers[0], proposer_layers[1]))
            # We need to provide layers the proposer doesn't have
            new_coverage = my_layers_set - proposer_layers_set
            if not new_coverage:
                return neuroshard_pb2.ProposeQuorumResponse(
                    accepted=False,
                    node_id=my_node_id,
                    layer_start=my_layer_start,
                    layer_end=my_layer_end,
                    speed_tier=my_speed_tier,
                    reason="No complementary layers to contribute"
                )
            
            # ACCEPTED — determine gRPC endpoint
            grpc_port = getattr(self.model, '_grpc_port', None)
            if not grpc_port:
                # Derive from HTTP port
                http_port = getattr(self.model, 'port', 8000)
                grpc_port = http_port + 1000
            my_endpoint = f"0.0.0.0:{grpc_port}"  # Will be resolved by proposer
            
            # If we have a public IP, use that
            if hasattr(self.model, 'p2p_manager') and self.model.p2p_manager:
                ip = getattr(self.model.p2p_manager, 'external_ip', None)
                if ip:
                    my_endpoint = f"{ip}:{grpc_port}"
            
            logger.info(f"[QUORUM] ACCEPTING proposal from {proposer_node_id[:8]}... "
                       f"quorum={quorum_id[:8]}, contributing layers {my_layer_start}-{my_layer_end-1}")
            
            # Notify runner to set up quorum on this side
            if self.on_quorum_proposed:
                try:
                    self.on_quorum_proposed(
                        quorum_id=quorum_id,
                        proposer_info={
                            'node_id': proposer_node_id,
                            'endpoint': proposer_endpoint,
                            'layer_range': proposer_layers,
                            'speed_tier': proposed_tier,
                        },
                        my_info={
                            'node_id': my_node_id,
                            'endpoint': my_endpoint,
                            'layer_range': (my_layer_start, my_layer_end),
                            'speed_tier': my_speed_tier,
                        },
                        total_layers=total_layers,
                    )
                except Exception as e:
                    logger.error(f"[QUORUM] on_quorum_proposed callback error: {e}")
            
            return neuroshard_pb2.ProposeQuorumResponse(
                accepted=True,
                node_id=my_node_id,
                endpoint=my_endpoint,
                layer_start=my_layer_start,
                layer_end=my_layer_end,
                speed_tier=my_speed_tier,
            )
            
        except Exception as e:
            logger.error(f"[QUORUM] ProposeQuorum error: {e}")
            import traceback
            traceback.print_exc()
            return neuroshard_pb2.ProposeQuorumResponse(
                accepted=False,
                reason=f"Internal error: {e}"
            )


def serve_grpc(port: int, model, p2p: P2PManager, swap_controller=None):
    """Start the gRPC server."""
    global GRPC_SERVER
    
    # Server options for P2P network - be lenient with keepalive pings
    # This is critical for decentralized networks where nodes ping frequently
    # IMPORTANT: Increase message size for activation tensors in pipeline training!
    # Activation size = batch_size * seq_len * hidden_dim * 4 bytes
    # For batch=4, seq=512, hidden=512: ~4MB, but we need headroom
    MAX_MESSAGE_SIZE = 64 * 1024 * 1024  # 64MB for large batches/sequences
    options = [
        ('grpc.keepalive_time_ms', 10000),  # Send keepalive every 10s
        ('grpc.keepalive_timeout_ms', 5000),  # 5s timeout for response
        ('grpc.keepalive_permit_without_calls', True),  # Allow pings without active RPCs
        ('grpc.http2.min_recv_ping_interval_without_data_ms', 5000),  # Accept pings every 5s
        ('grpc.http2.max_ping_strikes', 0),  # Don't penalize frequent pings
        ('grpc.max_receive_message_length', MAX_MESSAGE_SIZE),  # For receiving activations
        ('grpc.max_send_message_length', MAX_MESSAGE_SIZE),  # For sending responses
    ]
    
    global GRPC_SERVICER
    
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10), options=options)
    servicer = NeuroShardServiceServicer(model, p2p, swap_controller)
    neuroshard_pb2_grpc.add_NeuroShardServiceServicer_to_server(servicer, server)
    GRPC_SERVICER = servicer  # Store reference for quorum buffer wiring
    
    grpc_port = port + 1000
    # Bind to 0.0.0.0 for IPv4 compatibility ([::]  may not work on all systems)
    server.add_insecure_port(f'0.0.0.0:{grpc_port}')
    server.start()
    GRPC_SERVER = server
    
    logger.info(f"gRPC Server started on port {grpc_port}")
    
    try:
        # Wait until server is stopped externally via stop_grpc()
        server.wait_for_termination()
    except KeyboardInterrupt:
        server.stop(0)
    finally:
        GRPC_SERVER = None
        logger.info(f"gRPC Server on port {grpc_port} terminated")


def start_grpc_background(port: int, model, p2p: P2PManager, swap_controller=None):
    """Start gRPC server in background thread."""
    t = threading.Thread(target=serve_grpc, args=(port, model, p2p, swap_controller), daemon=True)
    t.start()


def stop_grpc(timeout: float = 5.0):
    """Stop the gRPC server gracefully."""
    global GRPC_SERVER
    if GRPC_SERVER is not None:
        logger.info("Stopping gRPC server...")
        try:
            # stop() returns an event that is set when shutdown is complete
            event = GRPC_SERVER.stop(grace=timeout)
            event.wait(timeout=timeout)
            logger.info("gRPC server stopped")
        except Exception as e:
            logger.warning(f"Error stopping gRPC server: {e}")
        finally:
            GRPC_SERVER = None
