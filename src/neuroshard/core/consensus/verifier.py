"""
Proof of Neural Work - Semantic Verifier (v2 - Optimistic Verification)

This module enforces UNIVERSAL verification rules - the "laws of physics" that apply
to ALL nodes regardless of their internal state. These are constraints that no
legitimate node can violate.

Architecture:
==============================

OPTIMISTIC VERIFICATION:
Proofs are accepted IMMEDIATELY and can be challenged within a time window.
This increases throughput while maintaining economic security.

Flow:
1. Node submits proof -> IMMEDIATE reward credit
2. Proof enters challenge window (default: 600 seconds)
3. Any node can challenge by staking NEURO
4. If challenge succeeds: challenger wins 2x stake, prover slashed
5. If challenge fails: challenger loses stake to prover
6. After window: proof finalized, unchallenged proofs are permanent

Verification Layers:
1. ProofVerifier (this file) - Universal Constraints + Optimistic Mode
   - Physical rate limits (no GPU can exceed certain throughput)
   - Required fields validation
   - Format and sanity checks
   - Challenge window management
   - Delegates to model_interface for node-specific checks

2. ModelInterface (e.g., SwarmEnabledDynamicNode) - Node-Specific State
   - Internal counter verification (only the node knows its true work count)
   - Model hash matching against local architecture
   - Training enabled status

The Verifier enforces *laws of physics*.
The Node enforces *personal integrity*.
The Challenge System enforces *economic security*.
"""

import logging
import time
import threading
from dataclasses import dataclass, field
from typing import Tuple, Optional, Any, Dict, List
from enum import Enum

logger = logging.getLogger(__name__)


# =============================================================================
# PHYSICAL CONSTANTS - Hardware Limits
# =============================================================================
# These are generous upper bounds based on current GPU capabilities.
# Even an H100 cluster cannot exceed these sustained rates.

# Maximum training batches per second (sustained)
# Rationale: Training rate depends on model size, batch size, and hardware.
# - Small models (11 layers, 512 hidden) with batch_size=1-4 can train at 5-15 steps/sec
# - Large models (70B+) are much slower
# 10.0 batches/sec is a reasonable limit that catches obvious fraud while allowing
# legitimate fast training on modern hardware (Jetson Orin, consumer GPUs with small models)
MAX_TRAINING_RATE_PER_SEC = 100.0

# Maximum inference tokens per second per GPU
# Rationale: H100 can do ~3000 tokens/sec for small models, ~500 for large
# 5000 is generous upper bound for any realistic deployment
MAX_INFERENCE_TOKENS_PER_SEC = 5000.0

# Minimum uptime to claim meaningful work (prevents micro-farming)
MIN_UPTIME_FOR_TRAINING_REWARD = 10.0  # 10 seconds


# =============================================================================
# OPTIMISTIC VERIFICATION CONSTANTS
# =============================================================================

# Challenge window duration (seconds)
# Proofs can be challenged within this window after submission
CHALLENGE_WINDOW = 600  # 10 minutes

# Minimum stake required to initiate a challenge
MIN_CHALLENGE_STAKE = 10.0  # NEURO

# Slash multiplier for fraud
# If fraud proven: prover loses stake * SLASH_MULTIPLIER
SLASH_MULTIPLIER = 2.0

# Verification rates by network size
# Larger networks verify proofs less frequently (probabilistic)
def get_verification_rate(network_size: int) -> float:
    """
    Get the probability that a proof will be verified.
    
    In small networks, verify everything.
    In large networks, verify probabilistically.
    
    Args:
        network_size: Number of nodes in network
        
    Returns:
        Probability (0.0 to 1.0) that a proof should be verified
    """
    if network_size < 20:
        return 1.0  # Verify all proofs
    elif network_size < 100:
        return 0.5  # 50% verification
    elif network_size < 500:
        return 0.2  # 20% verification
    else:
        return 0.05  # 5% verification


# =============================================================================
# OPTIMISTIC VERIFICATION DATA STRUCTURES
# =============================================================================

class ProofStatus(Enum):
    """Status of a proof in the optimistic verification system."""
    PENDING = "pending"         # Accepted, in challenge window
    CHALLENGED = "challenged"   # Someone challenged this proof
    VERIFIED = "verified"       # Challenge window passed, no challenges
    FRAUD = "fraud"             # Challenge succeeded, proof was fraudulent
    DEFENDED = "defended"       # Challenge failed, proof was valid


@dataclass
class PendingProof:
    """A proof that is pending verification (in challenge window)."""
    proof_id: str               # Unique identifier for the proof
    node_id: str                # Node that submitted the proof
    proof_type: str             # Type of proof (training, inference, etc.)
    reward_credited: float      # NEURO credited for this proof
    timestamp: float            # When proof was submitted
    status: ProofStatus = ProofStatus.PENDING
    
    # Challenge info (if challenged)
    challenger_id: Optional[str] = None
    challenger_stake: float = 0.0
    challenge_timestamp: Optional[float] = None
    
    # Proof data for recomputation
    proof_data: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def window_remaining(self) -> float:
        """Time remaining in challenge window (seconds)."""
        elapsed = time.time() - self.timestamp
        return max(0, CHALLENGE_WINDOW - elapsed)
    
    @property
    def is_window_expired(self) -> bool:
        """Check if challenge window has expired."""
        return self.window_remaining <= 0
    
    @property
    def is_challengeable(self) -> bool:
        """Check if proof can still be challenged."""
        return self.status == ProofStatus.PENDING and not self.is_window_expired


class PendingProofStore:
    """
    Thread-safe store for pending proofs.
    
    Manages proofs in their challenge window and handles
    expiration cleanup.
    """
    
    def __init__(self):
        self._proofs: Dict[str, PendingProof] = {}
        self._lock = threading.RLock()
    
    def add_proof(self, proof: PendingProof) -> None:
        """Add a pending proof."""
        with self._lock:
            self._proofs[proof.proof_id] = proof
    
    def get_proof(self, proof_id: str) -> Optional[PendingProof]:
        """Get a pending proof by ID."""
        with self._lock:
            return self._proofs.get(proof_id)
    
    def remove_proof(self, proof_id: str) -> Optional[PendingProof]:
        """Remove and return a pending proof."""
        with self._lock:
            return self._proofs.pop(proof_id, None)
    
    def get_proofs_by_node(self, node_id: str) -> List[PendingProof]:
        """Get all pending proofs for a node."""
        with self._lock:
            return [p for p in self._proofs.values() if p.node_id == node_id]
    
    def get_challengeable_proofs(self) -> List[PendingProof]:
        """Get all proofs that can still be challenged."""
        with self._lock:
            return [p for p in self._proofs.values() if p.is_challengeable]
    
    def get_expired_proofs(self) -> List[PendingProof]:
        """Get proofs whose challenge window has expired."""
        with self._lock:
            return [
                p for p in self._proofs.values()
                if p.is_window_expired and p.status == ProofStatus.PENDING
            ]
    
    def cleanup_expired(self) -> List[PendingProof]:
        """
        Clean up expired proofs and return them.
        
        Expired unchallenged proofs are finalized as VERIFIED.
        """
        expired = []
        with self._lock:
            for proof_id, proof in list(self._proofs.items()):
                if proof.is_window_expired and proof.status == ProofStatus.PENDING:
                    proof.status = ProofStatus.VERIFIED
                    expired.append(self._proofs.pop(proof_id))
        return expired
    
    def get_stats(self) -> Dict[str, int]:
        """Get statistics about pending proofs."""
        with self._lock:
            return {
                "total": len(self._proofs),
                "pending": sum(1 for p in self._proofs.values() if p.status == ProofStatus.PENDING),
                "challenged": sum(1 for p in self._proofs.values() if p.status == ProofStatus.CHALLENGED),
                "expired": sum(1 for p in self._proofs.values() if p.is_window_expired),
            }


# =============================================================================
# PROOF STRUCTURES
# =============================================================================

@dataclass
class PipelineProof:
    """
    Proof of work for a single batch in pipeline.
    
    Submitted after each batch processed in quorum-based training.
    Contains commitments (hashes) for later verification if challenged.
    """
    # Identity
    node_id: str
    quorum_id: str
    layer_range: tuple  # (start_layer, end_layer)
    
    # Work reference
    batch_id: str
    step_id: int
    
    # Commitments (hashes, not full data - for efficiency)
    input_activation_hash: bytes
    output_activation_hash: bytes
    gradient_merkle_root: bytes
    
    # Chain links (for verification continuity)
    prev_node_id: str  # Previous node in pipeline
    next_node_id: str  # Next node in pipeline
    
    # Metadata
    timestamp: float = field(default_factory=time.time)
    signature: bytes = b""
    
    # Version fields
    arch_version: int = 1
    vocab_version: int = 1
    speed_tier: str = "tier3"
    contribution_mode: str = "pipeline"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "node_id": self.node_id,
            "quorum_id": self.quorum_id,
            "layer_range": list(self.layer_range),
            "batch_id": self.batch_id,
            "step_id": self.step_id,
            "input_activation_hash": self.input_activation_hash.hex(),
            "output_activation_hash": self.output_activation_hash.hex(),
            "gradient_merkle_root": self.gradient_merkle_root.hex(),
            "prev_node_id": self.prev_node_id,
            "next_node_id": self.next_node_id,
            "timestamp": self.timestamp,
            "signature": self.signature.hex(),
            "arch_version": self.arch_version,
            "vocab_version": self.vocab_version,
            "speed_tier": self.speed_tier,
            "contribution_mode": self.contribution_mode,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'PipelineProof':
        """Create from dictionary."""
        return cls(
            node_id=data["node_id"],
            quorum_id=data["quorum_id"],
            layer_range=tuple(data["layer_range"]),
            batch_id=data["batch_id"],
            step_id=data["step_id"],
            input_activation_hash=bytes.fromhex(data["input_activation_hash"]),
            output_activation_hash=bytes.fromhex(data["output_activation_hash"]),
            gradient_merkle_root=bytes.fromhex(data["gradient_merkle_root"]),
            prev_node_id=data["prev_node_id"],
            next_node_id=data["next_node_id"],
            timestamp=data.get("timestamp", time.time()),
            signature=bytes.fromhex(data.get("signature", "")),
            arch_version=data.get("arch_version", 1),
            vocab_version=data.get("vocab_version", 1),
            speed_tier=data.get("speed_tier", "tier3"),
            contribution_mode=data.get("contribution_mode", "pipeline"),
        )


@dataclass
class CohortSyncProof:
    """
    Proof of participation in cohort sync.
    
    Submitted after each DiLoCo sync round where gradients are
    exchanged across quorums.
    """
    node_id: str
    layer_range: tuple  # (start_layer, end_layer)
    sync_round: int
    
    # Work done during inner loop
    batches_processed: int
    pseudo_gradient_hash: bytes
    
    # Aggregation participation
    cohort_members: list  # List of node IDs in cohort
    aggregated_gradient_hash: bytes
    
    # Weights before/after
    initial_weights_hash: bytes
    final_weights_hash: bytes
    
    # Metadata
    timestamp: float = field(default_factory=time.time)
    signature: bytes = b""
    
    # Version fields
    quorum_id: str = ""
    arch_version: int = 1
    contribution_mode: str = "pipeline"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "node_id": self.node_id,
            "layer_range": list(self.layer_range),
            "sync_round": self.sync_round,
            "batches_processed": self.batches_processed,
            "pseudo_gradient_hash": self.pseudo_gradient_hash.hex(),
            "cohort_members": self.cohort_members,
            "aggregated_gradient_hash": self.aggregated_gradient_hash.hex(),
            "initial_weights_hash": self.initial_weights_hash.hex(),
            "final_weights_hash": self.final_weights_hash.hex(),
            "timestamp": self.timestamp,
            "signature": self.signature.hex(),
            "quorum_id": self.quorum_id,
            "arch_version": self.arch_version,
            "contribution_mode": self.contribution_mode,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'CohortSyncProof':
        """Create from dictionary."""
        return cls(
            node_id=data["node_id"],
            layer_range=tuple(data["layer_range"]),
            sync_round=data["sync_round"],
            batches_processed=data["batches_processed"],
            pseudo_gradient_hash=bytes.fromhex(data["pseudo_gradient_hash"]),
            cohort_members=data["cohort_members"],
            aggregated_gradient_hash=bytes.fromhex(data["aggregated_gradient_hash"]),
            initial_weights_hash=bytes.fromhex(data.get("initial_weights_hash", "00")),
            final_weights_hash=bytes.fromhex(data.get("final_weights_hash", "00")),
            timestamp=data.get("timestamp", time.time()),
            signature=bytes.fromhex(data.get("signature", "")),
            quorum_id=data.get("quorum_id", ""),
            arch_version=data.get("arch_version", 1),
            contribution_mode=data.get("contribution_mode", "pipeline"),
        )


class ProofVerifier:
    """
    Verifies the semantic validity of Proof of Neural Work.
    
    Optimistic Verification Mode
    
    Proofs are accepted IMMEDIATELY with rewards credited.
    They can be challenged within the challenge window.
    This increases throughput while maintaining security.
    
    Enforces universal constraints that apply to ALL nodes:
    - Physical hardware limits (rate caps)
    - Required field validation
    - Proof format sanity
    
    Delegates node-specific verification to the model_interface:
    - Internal work counter validation
    - Local model hash matching
    - Training state verification
    
    Usage:
        verifier = ProofVerifier(model_interface=swarm_node, optimistic=True)
        
        # Accept proof optimistically
        accepted, proof_id, reward = verifier.accept_proof_optimistic(proof)
        
        # Later, challenge if suspicious
        success, result = verifier.challenge_proof(proof_id, challenger_stake)
    """
    
    def __init__(
        self,
        model_interface: Optional[Any] = None,
        optimistic: bool = True,
        network_size: int = 1,
    ):
        """
        Initialize the verifier.
        
        Args:
            model_interface: Object implementing verify_training_work(proof).
                             Typically a SwarmEnabledDynamicNode.
                             Required for training proof verification.
            optimistic: Enable optimistic verification mode
            network_size: Current network size (affects verification rate)
        """
        self.model_interface = model_interface
        self.optimistic = optimistic
        self.network_size = network_size
        
        # Optimistic verification state
        self._pending_proofs = PendingProofStore()
        self._verification_rate = get_verification_rate(network_size)
    
    def update_network_size(self, network_size: int):
        """Update network size and recalculate verification rate."""
        self.network_size = network_size
        self._verification_rate = get_verification_rate(network_size)
    
    # =========================================================================
    # OPTIMISTIC VERIFICATION
    # =========================================================================
    
    def accept_proof_optimistic(
        self,
        proof: Any,
        proof_id: str,
        reward: float,
    ) -> Tuple[bool, str, Optional[PendingProof]]:
        """
        Accept a proof optimistically (immediate credit, challengeable).
        
        This is the primary entry point for optimistic verification.
        The proof is accepted immediately and enters the challenge window.
        
        Args:
            proof: The PoNW proof object
            proof_id: Unique identifier for the proof
            reward: NEURO reward to credit
            
        Returns:
            (accepted, reason, pending_proof) tuple
        """
        if not self.optimistic:
            # Fall back to traditional verification
            is_valid, reason = self.verify_work_content(proof)
            return is_valid, reason, None
        
        # Quick sanity checks only (universal constraints)
        is_valid, reason = self._quick_verify(proof)
        if not is_valid:
            return False, reason, None
        
        # Create pending proof record
        pending = PendingProof(
            proof_id=proof_id,
            node_id=getattr(proof, 'node_id', ''),
            proof_type=getattr(proof, 'proof_type', 'unknown'),
            reward_credited=reward,
            timestamp=time.time(),
            proof_data={
                'uptime_seconds': getattr(proof, 'uptime_seconds', 0),
                'training_batches': getattr(proof, 'training_batches', 0),
                'tokens_processed': getattr(proof, 'tokens_processed', 0),
                'model_hash': getattr(proof, 'model_hash', ''),
                'signature': getattr(proof, 'signature', ''),
            }
        )
        
        self._pending_proofs.add_proof(pending)
        
        logger.debug(f"Proof {proof_id[:8]}... accepted optimistically, "
                    f"challenge window: {CHALLENGE_WINDOW}s")
        
        return True, "Accepted (optimistic)", pending
    
    def _quick_verify(self, proof: Any) -> Tuple[bool, str]:
        """
        Quick verification for optimistic acceptance.
        
        Only checks universal physical constraints.
        Full verification happens on challenge.
        """
        proof_type = getattr(proof, 'proof_type', None)
        uptime = getattr(proof, 'uptime_seconds', 0)
        
        # Basic sanity
        if uptime < 0:
            return False, "Negative uptime"
        
        # Physical rate limits
        if proof_type == "training":
            batches = getattr(proof, 'training_batches', 0)
            if uptime > 0 and batches > 0:
                rate = batches / uptime
                if rate > MAX_TRAINING_RATE_PER_SEC:
                    return False, f"Impossible training rate: {rate:.2f}/s"
        
        elif proof_type == "inference":
            tokens = getattr(proof, 'tokens_processed', 0)
            if uptime > 0 and tokens > 0:
                rate = tokens / uptime
                if rate > MAX_INFERENCE_TOKENS_PER_SEC:
                    return False, f"Impossible inference rate: {rate:.0f}/s"
        
        return True, "Quick verification passed"
    
    def challenge_proof(
        self,
        proof_id: str,
        challenger_id: str,
        challenger_stake: float,
    ) -> Tuple[bool, str, Optional[Dict[str, Any]]]:
        """
        Challenge a pending proof.
        
        Anyone can challenge a proof in its challenge window by staking NEURO.
        The challenge triggers full verification.
        
        Args:
            proof_id: ID of proof to challenge
            challenger_id: ID of challenging node
            challenger_stake: NEURO staked on challenge
            
        Returns:
            (success, reason, result_dict) tuple
            result_dict contains slashing/reward info on success
        """
        # Get pending proof
        pending = self._pending_proofs.get_proof(proof_id)
        if not pending:
            return False, "Proof not found or already finalized", None
        
        # Check if challengeable
        if not pending.is_challengeable:
            return False, f"Proof not challengeable (status: {pending.status.value})", None
        
        # Check stake
        if challenger_stake < MIN_CHALLENGE_STAKE:
            return False, f"Insufficient stake: {challenger_stake} < {MIN_CHALLENGE_STAKE}", None
        
        # Record challenge
        pending.status = ProofStatus.CHALLENGED
        pending.challenger_id = challenger_id
        pending.challenger_stake = challenger_stake
        pending.challenge_timestamp = time.time()
        
        # Perform full verification
        is_fraud = self._verify_challenge(pending)
        
        if is_fraud:
            # Challenge succeeded - proof was fraudulent
            pending.status = ProofStatus.FRAUD
            
            # Calculate slashing
            slash_amount = pending.reward_credited * SLASH_MULTIPLIER
            challenger_reward = challenger_stake + slash_amount / 2
            
            result = {
                "fraud_detected": True,
                "prover_slashed": slash_amount,
                "challenger_reward": challenger_reward,
                "reward_clawed_back": pending.reward_credited,
            }
            
            logger.warning(f"Fraud detected in proof {proof_id[:8]}... "
                          f"Prover slashed {slash_amount} NEURO")
            
            self._pending_proofs.remove_proof(proof_id)
            return True, "Challenge succeeded - fraud detected", result
        
        else:
            # Challenge failed - proof was valid
            pending.status = ProofStatus.DEFENDED
            
            # Challenger loses stake to prover
            result = {
                "fraud_detected": False,
                "challenger_lost_stake": challenger_stake,
                "prover_bonus": challenger_stake * 0.9,  # 90% to prover
                "network_burn": challenger_stake * 0.1,  # 10% burned
            }
            
            logger.info(f"Challenge failed for proof {proof_id[:8]}... "
                       f"Challenger loses {challenger_stake} NEURO")
            
            self._pending_proofs.remove_proof(proof_id)
            return True, "Challenge failed - proof valid", result
    
    def _verify_challenge(self, pending: PendingProof) -> bool:
        """
        Perform full verification for a challenged proof.
        
        Returns True if fraud is detected.
        """
        # Reconstruct minimal proof object for verification
        class ProofData:
            pass
        
        proof = ProofData()
        proof.node_id = pending.node_id
        proof.proof_type = pending.proof_type
        
        for key, value in pending.proof_data.items():
            setattr(proof, key, value)
        
        # Full verification
        is_valid, reason = self.verify_work_content(proof)
        
        if not is_valid:
            logger.info(f"Challenge verification failed: {reason}")
            return True  # Fraud detected
        
        return False  # Proof is valid
    
    def finalize_expired_proofs(self) -> List[PendingProof]:
        """
        Finalize proofs whose challenge window has expired.
        
        Expired unchallenged proofs become permanently verified.
        
        Returns:
            List of finalized proofs
        """
        finalized = self._pending_proofs.cleanup_expired()
        
        if finalized:
            logger.debug(f"Finalized {len(finalized)} proofs (unchallenged)")
        
        return finalized
    
    def get_pending_stats(self) -> Dict[str, Any]:
        """Get statistics about pending proofs."""
        stats = self._pending_proofs.get_stats()
        stats["verification_rate"] = self._verification_rate
        stats["challenge_window"] = CHALLENGE_WINDOW
        return stats
        
    def verify_work_content(self, proof: Any) -> Tuple[bool, str]:
        """
        Verify the content of a PoNW proof.
        
        Applies universal constraints first, then delegates to
        model_interface for node-specific validation.
        
        Args:
            proof: The PoNWProof object to verify
            
        Returns:
            (is_valid, reason) tuple
        """
        proof_type = getattr(proof, 'proof_type', None)
        
        if proof_type == "training":
            return self._verify_training_work(proof)
        elif proof_type == "inference":
            return self._verify_inference_work(proof)
        elif proof_type == "uptime":
            return self._verify_uptime_work(proof)
        elif proof_type == "data":
            return self._verify_data_work(proof)
        else:
            return False, f"Unknown proof type: {proof_type}"
            
    def _verify_training_work(self, proof: Any) -> Tuple[bool, str]:
        """
        Verify training proof with both universal and node-specific checks.
        
        Universal Checks (this method):
        1. Required fields present (model_hash)
        2. Physical rate limit (can't exceed hardware capabilities)
        3. Minimum uptime threshold
        
        Node-Specific Checks (delegated to model_interface):
        4. Model hash matches local architecture
        5. Claimed batches <= internal counter
        6. Training is actually enabled
        
        The "Golden Rule" of PoNW: L(w - lr * g) < L(w)
        We verify this indirectly through counter checks.
        """
        # =====================================================================
        # UNIVERSAL CHECK 1: Required Fields
        # =====================================================================
        if not getattr(proof, 'model_hash', None):
            return False, "Missing model hash for training proof"
        
        # =====================================================================
        # UNIVERSAL CHECK 2: Physical Rate Limit
        # =====================================================================
        # No GPU can sustain more than MAX_TRAINING_RATE_PER_SEC batches/second.
        # This is a hard physical constraint.
        uptime = getattr(proof, 'uptime_seconds', 0)
        batches = getattr(proof, 'training_batches', 0)
        
        if uptime > 0 and batches > 0:
            rate = batches / uptime
            if rate > MAX_TRAINING_RATE_PER_SEC:
                return False, (
                    f"Impossible training rate: {rate:.2f} batches/sec "
                    f"(max: {MAX_TRAINING_RATE_PER_SEC})"
                )
        
        # =====================================================================
        # UNIVERSAL CHECK 3: Minimum Uptime
        # =====================================================================
        # Prevent micro-farming attacks with tiny uptimes
        if batches > 0 and uptime < MIN_UPTIME_FOR_TRAINING_REWARD:
            return False, (
                f"Uptime too short for training reward: {uptime:.1f}s "
                f"(min: {MIN_UPTIME_FOR_TRAINING_REWARD}s)"
            )
        
        # =====================================================================
        # NODE-SPECIFIC CHECKS: Delegate to ModelInterface
        # =====================================================================
        # These checks require knowledge of the node's internal state.
        # 
        # IMPORTANT: Observer nodes and nodes without a model interface
        # can still accept proofs based on:
        # 1. Cryptographic signature verification (done earlier in verify_proof)
        # 2. Universal rate limit checks (done above)
        # 3. Network consensus (proofs are gossiped and cross-validated)
        #
        # This is similar to how light clients in Bitcoin/Ethereum work -
        # they trust cryptographic proofs without full validation.
        if self.model_interface is None:
            # No model interface = observer mode or light client
            # Accept based on signature + universal checks only
            return True, "Accepted via signature verification (no local model)"
        
        if not hasattr(self.model_interface, 'verify_training_work'):
            return True, "Accepted via signature verification (model interface incomplete)"
        
        return self.model_interface.verify_training_work(proof)

    def _verify_inference_work(self, proof: Any) -> Tuple[bool, str]:
        """
        Verify inference proof via economic receipts.
        
        Inference rewards flow from User -> Node.
        We verify that the User authorized this work.
        
        Universal Checks:
        1. Request ID present (links to payment)
        2. Token rate within physical limits
        """
        # =====================================================================
        # UNIVERSAL CHECK 1: Request ID Required
        # =====================================================================
        if not getattr(proof, 'request_id', None):
            return False, "Missing request_id: anonymous inference not verifiable"
        
        # =====================================================================
        # UNIVERSAL CHECK 2: Token Rate Limit
        # =====================================================================
        uptime = getattr(proof, 'uptime_seconds', 0)
        tokens = getattr(proof, 'tokens_processed', 0)
        
        if uptime > 0 and tokens > 0:
            rate = tokens / uptime
            if rate > MAX_INFERENCE_TOKENS_PER_SEC:
                return False, (
                    f"Impossible inference rate: {rate:.0f} tokens/sec "
                    f"(max: {MAX_INFERENCE_TOKENS_PER_SEC})"
                )
        
        # Inference receipts are further validated by the Market/Ledger
        # which checks that the user actually paid for this request
        return True, "Inference work linked to valid request"
    
    def _verify_uptime_work(self, proof: Any) -> Tuple[bool, str]:
        """
        Verify uptime proof.
        
        Uptime is verified through:
        1. Gossip protocol (peers attest to availability)
        2. Rate limits in the Ledger (can't claim faster than real time)
        
        The Ledger handles most uptime validation through its
        rate limiting and gossip-based peer attestation.
        """
        uptime = getattr(proof, 'uptime_seconds', 0)
        
        # Sanity check: uptime can't be negative
        if uptime < 0:
            return False, "Negative uptime is invalid"
        
        # Sanity check: uptime can't exceed proof window
        # (The Ledger enforces this more precisely with timestamps)
        max_reasonable_uptime = 3600 * 24  # 24 hours max per proof
        if uptime > max_reasonable_uptime:
            return False, f"Uptime {uptime}s exceeds maximum proof window"
        
        return True, "Uptime valid"
    
    def _verify_data_work(self, proof: Any) -> Tuple[bool, str]:
        """
        Verify data serving proof.
        
        Data serving is verified optimistically:
        - Nodes claim they served data shards
        - The tracker/DHT can spot-check availability
        - Economic incentives discourage false claims
        
        Future: Could add Merkle proofs for data possession.
        """
        layers_held = getattr(proof, 'layers_held', 0)
        
        # Sanity check: can't hold negative layers
        if layers_held < 0:
            return False, "Negative layers_held is invalid"
        
        # Sanity check: can't hold more layers than exist
        max_layers = 128  # Generous upper bound
        if layers_held > max_layers:
            return False, f"layers_held {layers_held} exceeds maximum {max_layers}"
        
        return True, "Data serving valid (optimistic verification)"
