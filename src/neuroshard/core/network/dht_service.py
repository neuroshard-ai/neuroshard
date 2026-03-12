import logging
import time
import threading
from typing import List, Optional

from neuroshard.protos import neuroshard_pb2
from neuroshard.protos import neuroshard_pb2_grpc
from neuroshard.core.network.dht import Node, RoutingTable, KBucket

logger = logging.getLogger(__name__)

def proto_to_node(proto_node: neuroshard_pb2.DHTNodeInfo) -> Node:
    # Convert bytes to int
    node_id = int.from_bytes(proto_node.node_id, byteorder='big')
    return Node(node_id, proto_node.ip, proto_node.port)

def node_to_proto(node: Node) -> neuroshard_pb2.DHTNodeInfo:
    # Convert int to bytes (20 bytes for 160 bits)
    # Ensure it's 20 bytes length
    try:
        id_bytes = node.id.to_bytes(20, byteorder='big')
    except OverflowError:
        # Handle potential ID size issues if we mess up somewhere
        id_bytes = node.id.to_bytes((node.id.bit_length() + 7) // 8, byteorder='big')
        # Pad to 20 if needed? Or just assume big enough. Kademlia typically fixed size.
        # For now let's stick to standard 20 bytes.
        
    return neuroshard_pb2.DHTNodeInfo(
        node_id=id_bytes,
        ip=node.ip,
        port=node.port
    )

class DHTServiceMixin:
    """
    Mixin to add DHT capabilities to the main NeuroShardServiceServicer.
    We use a mixin to keep the code clean and separated.
    """
    def __init__(self, routing_table: RoutingTable, storage: dict, ledger: Optional[object] = None):
        self.routing_table = routing_table
        self.storage = storage # Simple in-memory dict for now
        self.ledger = ledger # LedgerManager instance

    def _update_routing(self, sender_proto: neuroshard_pb2.DHTNodeInfo):
        """Update routing table with the sender's info"""
        node = proto_to_node(sender_proto)
        self.routing_table.add_contact(node)

    def DHTPing(self, request, context):
        self._update_routing(request.sender)
        
        return neuroshard_pb2.DHTPingResponse(
            responder=node_to_proto(self.routing_table.local_node)
        )

    def DHTStore(self, request, context):
        self._update_routing(request.sender)
        
        key_int = int.from_bytes(request.key, byteorder='big')
        new_val = request.value
        
        import json
        current_val = self.storage.get(key_int)
        
        stored_list = []
        if current_val:
            try:
                # Try to parse as list
                stored_list = json.loads(current_val)
                if not isinstance(stored_list, list):
                    stored_list = [current_val]
            except:
                # Was a simple string
                stored_list = [current_val]
        
        # Add new value if not present
        if new_val not in stored_list:
            stored_list.append(new_val)
            
        # Limit size (keep most recent 50)
        if len(stored_list) > 50:
            stored_list = stored_list[-50:]
            
        self.storage[key_int] = json.dumps(stored_list)
        
        return neuroshard_pb2.DHTStoreResponse(
            responder=node_to_proto(self.routing_table.local_node),
            success=True
        )

    def DHTFindNode(self, request, context):
        self._update_routing(request.sender)
        
        target_id = int.from_bytes(request.target_id, byteorder='big')
        closest_nodes = self.routing_table.find_closest(target_id)
        
        return neuroshard_pb2.DHTFindNodeResponse(
            responder=node_to_proto(self.routing_table.local_node),
            nodes=[node_to_proto(n) for n in closest_nodes]
        )

    def DHTFindValue(self, request, context):
        self._update_routing(request.sender)
        
        key_int = int.from_bytes(request.key, byteorder='big')
        
        if key_int in self.storage:
            return neuroshard_pb2.DHTFindValueResponse(
                responder=node_to_proto(self.routing_table.local_node),
                value=self.storage[key_int],
                found=True
            )
        else:
            # Return closest nodes
            closest_nodes = self.routing_table.find_closest(key_int)
            return neuroshard_pb2.DHTFindValueResponse(
                responder=node_to_proto(self.routing_table.local_node),
                nodes=[node_to_proto(n) for n in closest_nodes],
                found=False
            )
    
    def GossipProof(self, request, context):
        """
        Handle incoming gossip Proof of Neural Work (PoNW).
        
        Security Model:
        ===============
        1. Proof must have valid signature format (64 hex chars)
        2. Proof must have valid node_id format (32 hex or legacy decimal)
        3. Proof must pass timestamp freshness check (< 5 min old)
        4. Proof must not be a replay (signature not seen before)
        5. Proof must pass rate limiting (per-node limits)
        6. Proof must pass plausibility checks (realistic work claims)
        
        All these checks happen in ledger.process_proof().
        """
        if not self.ledger:
            return neuroshard_pb2.GossipProofResponse(accepted=False)

        from neuroshard.core.economics.ledger import PoNWProof, ProofType
        
        try:
            # Determine proof type from request
            proof_type_str = getattr(request, 'proof_type', '')
            if proof_type_str:
                proof_type = proof_type_str
            elif getattr(request, 'training_batches', 0) > 0:
                proof_type = ProofType.TRAINING.value
            elif getattr(request, 'token_count', 0) > 0:
                proof_type = ProofType.INFERENCE.value
            else:
                proof_type = ProofType.UPTIME.value
            
            # CRITICAL: Store public key for trustless verification
            # This allows nodes to verify proofs from previously unknown peers
            public_key = getattr(request, 'public_key', None)
            if public_key and self.ledger and self.ledger.crypto:
                stored = self.ledger.crypto.store_public_key(request.node_id, public_key)
                logger.debug(f"Gossip: Stored public key for {request.node_id[:16]}...: {stored}")
            else:
                logger.debug(f"Gossip: No public key provided by {request.node_id[:16]}...")
            
            # Reconstruct full PoNWProof from gRPC request
            # CRITICAL: Include data_samples, model_hash, request_id for canonical_payload match
            # Extract current_loss (0.0 means "not set" - convert to None)
            loss_val = getattr(request, 'current_loss', 0.0)
            current_loss = loss_val if loss_val > 0.0 else None
            
            proof = PoNWProof(
                node_id=request.node_id,
                proof_type=proof_type,
                timestamp=request.timestamp,
                nonce=getattr(request, 'nonce', '') or f"gossip_{request.signature[:8] if request.signature else 'none'}",
                uptime_seconds=request.uptime,
                tokens_processed=getattr(request, 'token_count', 0),
                training_batches=getattr(request, 'training_batches', 0),
                data_samples=getattr(request, 'data_samples', 0),
                model_hash=getattr(request, 'model_hash', ''),
                request_id=getattr(request, 'request_id', '') or None,
                layers_held=getattr(request, 'layers_held', 0),
                has_embedding=getattr(request, 'has_embedding', False),
                has_lm_head=getattr(request, 'has_lm_head', False),
                signature=request.signature,
                current_loss=current_loss
            )
            
            # Validate and credit - ALL security checks happen here
            success, reward, msg = self.ledger.process_proof(proof)
            
            if success:
                logger.info(f"Gossip: Accepted PoNW from {request.node_id[:16]}... "
                           f"(type={proof_type}, tokens={proof.tokens_processed}, "
                           f"batches={proof.training_batches}, reward={reward:.6f} NEURO)")
            else:
                # Common benign rejections in gossip networks - log as debug to reduce noise
                # - Duplicates: Normal propagation
                # - Proof too old: Syncing nodes or slow propagation
                # - Rate limits: Peer sending too fast (dropped)
                if any(x in msg for x in ["Duplicate", "Proof too old", "Rate limit"]):
                    logger.debug(f"Gossip: Ignored PoNW from {request.node_id[:16]}... ({msg})")
                else:
                    logger.warning(f"❌ Gossip: Rejected PoNW from {request.node_id[:16]}...: {msg}")
                
            return neuroshard_pb2.GossipProofResponse(accepted=success)
        except Exception as e:
            logger.error(f"Gossip processing error: {e}")
            return neuroshard_pb2.GossipProofResponse(accepted=False)

    def GossipTransaction(self, request, context):
        """Handle incoming gossip transaction (P2P Transfer)."""
        if not self.ledger:
            return neuroshard_pb2.GossipTransactionResponse(accepted=False, reason="No ledger")

        try:
            # Reconstruct the signed payload (Canonical String)
            # Format: sender_id:recipient_id:amount:timestamp
            payload = f"{request.sender_id}:{request.recipient_id}:{request.amount}:{request.timestamp}"
            
            # ECDSA Signature Verification — trustless, anyone can verify
            # with the sender's public key (from registry or included in message)
            from neuroshard.core.crypto.ecdsa import verify_signature
            
            public_key_hex = getattr(request, 'public_key', None) or None
            if not verify_signature(request.sender_id, payload, request.signature, public_key_hex):
                logger.warning(f"Gossip: Rejected Transaction from {request.sender_id[:16]}... "
                              f"— ECDSA signature verification failed")
                return neuroshard_pb2.GossipTransactionResponse(
                    accepted=False, reason="Invalid ECDSA signature"
                )
            
            success = self.ledger.create_transaction(
                from_id=request.sender_id,
                to_id=request.recipient_id,
                amount=request.amount,
                signature=request.signature
            )
            
            if success:
                logger.info(f"Gossip: Accepted Transaction {request.amount} NEURO from {request.sender_id} -> {request.recipient_id}")
                return neuroshard_pb2.GossipTransactionResponse(accepted=True)
            else:
                logger.warning(f"❌ Gossip: Rejected Transaction {request.sender_id} -> {request.recipient_id}")
                return neuroshard_pb2.GossipTransactionResponse(accepted=False, reason="Validation failed")
                
        except Exception as e:
            logger.error(f"Gossip Transaction error: {e}")
            return neuroshard_pb2.GossipTransactionResponse(accepted=False, reason=str(e))

    def GossipStake(self, request, context):
        """
        Handle incoming stake gossip from peers.
        
        This allows the network to have a consistent view of stakes across nodes.
        Stakes are used to calculate reward multipliers and validate PoNW claims.
        
        ECDSA signatures enable trustless verification of stake claims.
        """
        if not self.ledger:
            return neuroshard_pb2.GossipStakeResponse(accepted=False, reason="No ledger")

        try:
            import time
            from neuroshard.core.crypto.ecdsa import verify_signature, is_valid_node_id_format
            
            # Basic validation
            if not request.node_id or request.amount < 0:
                return neuroshard_pb2.GossipStakeResponse(accepted=False, reason="Invalid stake data")
            
            # Validate node_id format (32 hex chars)
            if not is_valid_node_id_format(request.node_id):
                return neuroshard_pb2.GossipStakeResponse(accepted=False, reason="Invalid node_id format")
            
            # Timestamp freshness check (5 minute window)
            if abs(time.time() - request.timestamp) > 300:
                return neuroshard_pb2.GossipStakeResponse(accepted=False, reason="Stale stake update")
            
            # Verify ECDSA signature
            # SECURITY: Public key is REQUIRED for verification
            if not request.public_key:
                logger.warning(f"Gossip: Missing public key from {request.node_id[:16]}...")
                return neuroshard_pb2.GossipStakeResponse(accepted=False, reason="Missing public key")
            
            payload = f"{request.node_id}:{request.amount}:{request.locked_until}"
            if not verify_signature(request.node_id, payload, request.signature, request.public_key):
                logger.warning(f"Gossip: Invalid stake signature from {request.node_id[:16]}...")
                return neuroshard_pb2.GossipStakeResponse(accepted=False, reason="Invalid signature")
            
            # Update our local view of this node's stake
            success = self.ledger.update_stake(
                node_id=request.node_id,
                amount=request.amount,
                locked_until=request.locked_until
            )
            
            if success:
                logger.info(f"Gossip: Updated stake for {request.node_id[:16]}... = {request.amount:.2f} NEURO")
                return neuroshard_pb2.GossipStakeResponse(accepted=True)
            else:
                return neuroshard_pb2.GossipStakeResponse(accepted=False, reason="Stake update failed")
                
        except Exception as e:
            logger.error(f"Gossip Stake error: {e}")
            return neuroshard_pb2.GossipStakeResponse(accepted=False, reason=str(e))

    def RequestProofValidation(self, request, context):
        """
        Handle proof validation requests.
        
        When a node submits a proof, it can request validation from validators.
        Validators check the proof and cast their vote.
        """
        if not self.ledger:
            return neuroshard_pb2.ProofValidationResponse(
                accepted=False, 
                reason="No ledger available"
            )

        try:
            from neuroshard.core.economics.ledger import PoNWProof, ProofType
            
            # Check if we're eligible to validate
            eligible, reason = self.ledger.is_eligible_validator()
            if not eligible:
                return neuroshard_pb2.ProofValidationResponse(
                    accepted=False,
                    reason=f"Not eligible validator: {reason}"
                )
            
            # Reconstruct the proof
            proof = PoNWProof(
                node_id=request.submitter_id,
                proof_type=request.proof_type,
                timestamp=request.timestamp,
                nonce=request.nonce,
                uptime_seconds=request.uptime_seconds,
                tokens_processed=request.tokens_processed,
                training_batches=request.training_batches,
                layers_held=request.layers_held,
                has_embedding=request.has_embedding,
                has_lm_head=request.has_lm_head,
                signature=request.proof_signature
            )
            
            # Validate the proof
            # Check plausibility, rate limits, etc.
            is_valid = self._validate_proof_locally(proof)
            
            # Cast our vote
            success, fee, msg = self.ledger.validate_proof_as_validator(
                proof=proof,
                vote=is_valid,
                validation_details=f"Local validation: {'PASS' if is_valid else 'FAIL'}"
            )
            
            my_stake = self.ledger._get_stake(self.ledger.node_id)
            
            logger.info(f"Validation: Voted {'VALID' if is_valid else 'INVALID'} on proof from "
                       f"{request.submitter_id[:16]}... (earned {fee:.4f} NEURO)")
            
            return neuroshard_pb2.ProofValidationResponse(
                accepted=True,
                reason=msg,
                validator_id=self.ledger.node_id,
                validator_stake=my_stake
            )
            
        except Exception as e:
            logger.error(f"Proof validation error: {e}")
            return neuroshard_pb2.ProofValidationResponse(
                accepted=False,
                reason=str(e)
            )
    
    def _validate_proof_locally(self, proof) -> bool:
        """
        Perform local validation checks on a proof.
        
        Returns True if the proof appears legitimate.
        """
        import time
        from neuroshard.core.economics.ledger import (
            MAX_UPTIME_PER_PROOF, 
            MAX_TOKENS_PER_MINUTE,
            PROOF_FRESHNESS_WINDOW
        )
        
        # 1. Timestamp freshness
        age = abs(time.time() - proof.timestamp)
        if age > PROOF_FRESHNESS_WINDOW:
            logger.debug(f"Proof too old: {age:.0f}s > {PROOF_FRESHNESS_WINDOW}s")
            return False
        
        # 2. Uptime plausibility
        if proof.uptime_seconds > MAX_UPTIME_PER_PROOF:
            logger.debug(f"Uptime too high: {proof.uptime_seconds}s > {MAX_UPTIME_PER_PROOF}s")
            return False
        
        # 3. Token rate plausibility
        if proof.uptime_seconds > 0:
            tokens_per_minute = (proof.tokens_processed / proof.uptime_seconds) * 60
            if tokens_per_minute > MAX_TOKENS_PER_MINUTE:
                logger.debug(f"Token rate too high: {tokens_per_minute:.0f}/min > {MAX_TOKENS_PER_MINUTE}/min")
                return False
        
        # 4. Signature format (basic check)
        if not proof.signature or len(proof.signature) < 64:
            logger.debug("Invalid signature format")
            return False

        # 5. Work Content Verification (via Ledger's Verifier)
        if hasattr(self, 'ledger') and self.ledger and hasattr(self.ledger, 'verifier'):
            is_work_valid, reason = self.ledger.verifier.verify_work_content(proof)
            if not is_work_valid:
                logger.debug(f"Work content invalid: {reason}")
                return False
        
        # All checks passed
        return True

    def GossipValidationVote(self, request, context):
        """
        Handle incoming validation votes from other validators.
        
        This allows validators to share their votes across the network.
        """
        if not self.ledger:
            return neuroshard_pb2.ValidationVoteResponse(
                accepted=False,
                reason="No ledger available"
            )

        try:
            from neuroshard.core.crypto.ecdsa import verify_signature, is_valid_node_id_format
            
            # Validate vote format
            if not is_valid_node_id_format(request.validator_id):
                return neuroshard_pb2.ValidationVoteResponse(
                    accepted=False,
                    reason="Invalid validator_id format"
                )
            
            # Verify signature
            payload = f"{request.proof_signature}:{request.validator_id}:{request.vote}:{request.timestamp}"
            if not verify_signature(request.validator_id, payload, request.signature):
                return neuroshard_pb2.ValidationVoteResponse(
                    accepted=False,
                    reason="Invalid vote signature"
                )
            
            # Record the vote in our ledger
            with self.ledger.lock:
                import sqlite3
                with sqlite3.connect(self.ledger.db_path, timeout=60.0) as conn:
                    # Ensure table exists
                    conn.execute("""
                        CREATE TABLE IF NOT EXISTS validation_votes (
                            validation_id TEXT PRIMARY KEY,
                            proof_signature TEXT NOT NULL,
                            validator_id TEXT NOT NULL,
                            validator_stake REAL NOT NULL,
                            vote INTEGER NOT NULL,
                            details TEXT,
                            timestamp REAL NOT NULL,
                            fee_earned REAL DEFAULT 0.0,
                            UNIQUE(proof_signature, validator_id)
                        )
                    """)
                    
                    # Insert vote (ignore if already exists)
                    import hashlib
                    validation_id = hashlib.sha256(
                        f"{request.validator_id}:{request.proof_signature}:{request.timestamp}".encode()
                    ).hexdigest()[:32]
                    
                    conn.execute("""
                        INSERT OR IGNORE INTO validation_votes 
                        (validation_id, proof_signature, validator_id, validator_stake, vote, details, timestamp)
                        VALUES (?, ?, ?, ?, ?, ?, ?)
                    """, (
                        validation_id,
                        request.proof_signature,
                        request.validator_id,
                        request.validator_stake,
                        1 if request.vote else 0,
                        request.details,
                        request.timestamp
                    ))
            
            # Get current validation status
            status = self.ledger.get_proof_validation_status(request.proof_signature)
            
            logger.debug(f"Received validation vote from {request.validator_id[:16]}... "
                        f"({'VALID' if request.vote else 'INVALID'})")
            
            return neuroshard_pb2.ValidationVoteResponse(
                accepted=True,
                total_valid_stake=status["valid_stake"],
                total_invalid_stake=status["invalid_stake"],
                consensus_reached=status["consensus_reached"],
                consensus_result=status.get("consensus_result", False)
            )
            
        except Exception as e:
            logger.error(f"Gossip validation vote error: {e}")
            return neuroshard_pb2.ValidationVoteResponse(
                accepted=False,
                reason=str(e)
            )
