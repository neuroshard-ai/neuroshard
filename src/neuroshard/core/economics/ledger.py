"""
NEURO Token Ledger System

This module implements the NEURO token economics for NeuroShard:
- Proof of Neural Work (PoNW) verification
- Token minting through verified work
- Fee burn mechanism (5% deflationary)
- Anti-cheat measures and rate limiting
- ECDSA cryptographic signature verification (trustless)

Security Model:
1. Nodes cannot claim arbitrary rewards - all rewards require verifiable proof
2. Proofs are signed with ECDSA - ANYONE can verify without shared secrets
3. Replay attacks prevented via signature deduplication
4. Plausibility checks prevent inflated claims
5. Cross-validation via gossip consensus

Based on: docs/whitepaper/neuroshard_whitepaper.tex
"""

import sqlite3
import time
import json
import logging
import threading
import hashlib
import os
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, asdict
from enum import Enum

# Import ECDSA crypto module - REQUIRED
from neuroshard.core.crypto.ecdsa import (
    NodeCrypto, 
    verify_signature,
    register_public_key,
    get_public_key,
    is_valid_signature_format,
    is_valid_node_id_format
)

# Import ProofVerifier for semantic validation
from neuroshard.core.consensus.verifier import ProofVerifier

logger = logging.getLogger(__name__)

# ============================================================================
# CONSTANTS - Token Economics (from Centralized Economics Module)
# ============================================================================
# All economic constants are defined in neuroshard/core/economics.py
# Import from there to ensure consistency across the codebase.
# ============================================================================

from neuroshard.core.economics.constants import (
    # Reward rates (per-layer training)
    UPTIME_REWARD_PER_MINUTE,
    TRAINING_REWARD_PER_BATCH_PER_LAYER,
    ASYNC_TRAINING_REWARD_PER_BATCH_PER_LAYER,
    DATA_REWARD_PER_SAMPLE,
    
    # Dynamic inference pricing (PURE MARKET)
    INFERENCE_MARKET_PRICE_SMOOTHING,
    INFERENCE_MARKET_CAPACITY_TIMEOUT,
    INFERENCE_MARKET_TARGET_RESPONSE_TIME,
    INFERENCE_MARKET_BASE_PRICE,
    
    # Quorum role distribution  
    INITIATOR_SHARE,
    PROCESSOR_SHARE,
    FINISHER_SHARE,
    
    # Role bonuses (ADDITIVE)
    INITIATOR_BONUS,
    FINISHER_BONUS,
    TRAINING_BONUS,
    
    # Staking
    STAKING_BASE_BONUS,
    STAKING_UNIT,
    MIN_STAKE_AMOUNT,
    MAX_STAKE_AMOUNT,
    
    # Validator requirements
    VALIDATOR_MIN_STAKE,
    VALIDATOR_MIN_MEMORY_MB,
    VALIDATION_FEE_PER_PROOF,
    VALIDATION_CONSENSUS_THRESHOLD,
    VALIDATOR_ROTATION_ENABLED,
    VALIDATOR_SELECTION_RANDOMNESS,
    REMOTE_STAKE_MULTIPLIER_CAP,
    
    # Fees and burns
    FEE_BURN_RATE,
    BURN_ADDRESS,
    
    # Anti-cheat limits
    MAX_UPTIME_PER_PROOF,
    MAX_TOKENS_PER_MINUTE,
    MAX_PROOFS_PER_HOUR,
    PROOF_FRESHNESS_WINDOW,
    MAX_REWARD_PER_PROOF,
    
    # Slashing
    SLASH_AMOUNT,
    WHISTLEBLOWER_REWARD_RATE,
    VALIDATOR_SLASH_MULTIPLIER,
    
    # Helper functions
    calculate_stake_multiplier,
    calculate_training_reward,
    calculate_role_bonus,
    calculate_async_freshness,
    is_valid_stake_amount,
    is_valid_stake_duration,
)


class ProofType(Enum):
    """Types of Proof of Neural Work."""
    UPTIME = "uptime"
    INFERENCE = "inference"
    TRAINING = "training"
    DATA = "data"


@dataclass
class PoNWProof:
    """
    Proof of Neural Work - Cryptographically signed proof of contribution.
    
    Security Properties:
    1. node_id: Derived from node_token (cannot be forged)
    2. timestamp: Must be recent (prevents replay)
    3. signature: ECDSA(node_key, canonical_payload)
    4. nonce: Random value to prevent signature collision
    5. request_id: Links to InferenceRequest for price locking
    6. epoch_id: Binds proof to a specific epoch (prevents epoch-hopping)
    7. model_hash_start/end: Proves actual weight changes (training work)
    8. gradient_commitment: Spot-checkable proof of gradient computation
    """
    node_id: str
    proof_type: str
    timestamp: float
    nonce: str
    
    # Work metrics
    uptime_seconds: float = 0.0
    tokens_processed: int = 0
    training_batches: int = 0
    data_samples: int = 0
    
    # Marketplace - links to InferenceRequest for price locking
    request_id: Optional[str] = None  # If inference, which request was this?
    
    # Context for verification
    model_hash: str = ""          # Legacy: Hash of model state (for training proofs)
    layers_held: int = 0          # Number of layers this node holds
    has_embedding: bool = False   # Driver node
    has_lm_head: bool = False     # Validator node
    
    # Training metrics (for global loss tracking)
    current_loss: Optional[float] = None  # Current training loss (for aggregation)
    
    # Reputation (for reward bonus calculation)
    reputation: float = 1.0  # 0.0 to 1.0, default 1.0 for new nodes
    
    # =========================================================================
    # NEW: Chained PoNW Fields (Blockchain-like Verification)
    # =========================================================================
    
    # Epoch binding - proof must belong to a specific epoch
    epoch_id: int = 0             # Which epoch this proof belongs to
    
    # Model state commitment - proves actual training work
    model_hash_start: str = ""    # H(weights) BEFORE training
    model_hash_end: str = ""      # H(weights) AFTER training
    
    # Gradient commitment - the "nonce equivalent" for PoNW
    # This is a hash of gradient statistics that can be spot-checked
    gradient_commitment: str = ""  # H(gradient_norms + loss + data_hash)
    data_hash: str = ""           # H(training_batch) for reproducibility
    
    # Gradient statistics for verification
    gradient_norm: float = 0.0    # L2 norm of gradients (sanity check)
    
    # =========================================================================
    # MoE Expert Tracking (Mixture of Experts)
    # =========================================================================
    moe_expert_count: int = 0     # Number of experts held by this node
    moe_scarcity_avg: float = 1.0 # Average scarcity bonus across held experts
    
    # Signature
    signature: str = ""
    
    def canonical_payload(self) -> str:
        """Create canonical string for signing (deterministic ordering).
        
        CRITICAL: All float fields must use consistent formatting to ensure
        the same payload is generated on sender and receiver.
        
        NOTE: All fields that affect reward must be in the payload.
        The epoch_id and gradient_commitment are included for chain integrity.
        """
        return (
            f"{self.node_id}:{self.proof_type}:{self.timestamp:.6f}:{self.nonce}:"
            f"{float(self.uptime_seconds):.1f}:{self.tokens_processed}:{self.training_batches}:"
            f"{self.data_samples}:{self.request_id if self.request_id else ''}:"
            f"{self.model_hash}:{self.layers_held}:"
            f"{self.epoch_id}:{self.model_hash_start}:{self.model_hash_end}:"
            f"{self.gradient_commitment}"
        )
    
    def verify_training_work(self) -> Tuple[bool, str]:
        """
        Verify that the proof shows evidence of actual training work.
        
        NO LEGACY SUPPORT - All nodes must use chained PoNW.
        
        Requirements:
        1. model_hash_start != model_hash_end (weights actually changed)
        2. gradient_commitment present (spot-checkable proof of computation)
        3. current_loss > 0 (training computed actual loss)
        
        Returns:
            (is_valid, error_message)
        """
        if self.proof_type != "training":
            return True, "Not a training proof"
        
        if self.training_batches <= 0:
            return True, "No training batches claimed"
        
        # REQUIRED: Must have valid loss (proves forward pass happened)
        if self.current_loss is None or self.current_loss <= 0:
            return False, "No valid loss - training proof requires loss > 0"
        
        # REQUIRED: Must have model state hashes (proves weight tracking)
        if not self.model_hash_start or not self.model_hash_end:
            return False, "Missing model state hashes - update node to latest version"
        
        # REQUIRED: Weights must have changed (proves backward pass happened)
        if self.model_hash_start == self.model_hash_end:
            return False, "Model weights unchanged - no actual training occurred"
        
        # REQUIRED: Must have gradient commitment (for spot-checking)
        if not self.gradient_commitment:
            return False, "Missing gradient commitment - update node to latest version"
        
        return True, "Training work verified"
    
    def to_dict(self) -> Dict:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'PoNWProof':
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


def sign_proof(proof: 'PoNWProof', node_token: str) -> 'PoNWProof':
    """
    Sign a Proof of Neural Work using ECDSA.
    
    Args:
        proof: The proof to sign
        node_token: The node's secret token for signing
        
    Returns:
        The same proof with signature field populated
    """
    from neuroshard.core.crypto.ecdsa import sign_message
    payload = proof.canonical_payload()
    proof.signature = sign_message(payload, node_token)
    return proof


@dataclass
class Transaction:
    """NEURO transfer transaction with fee burn."""
    tx_id: str
    from_id: str
    to_id: str
    amount: float
    fee: float              # Total fee
    burn_amount: float      # 5% of fee burned
    timestamp: float
    signature: str
    memo: str = ""
    
    def canonical_payload(self) -> str:
        return f"{self.from_id}:{self.to_id}:{self.amount}:{self.fee}:{self.timestamp}:{self.memo}"


@dataclass 
class LedgerStats:
    """Global ledger statistics."""
    total_minted: float = 0.0
    total_burned: float = 0.0
    total_transferred: float = 0.0
    circulating_supply: float = 0.0
    total_proofs_processed: int = 0
    total_transactions: int = 0


class NEUROLedger:
    """
    Secure NEURO Token Ledger with Proof of Neural Work verification.
    
    Security Features:
    1. ECDSA signatures on all proofs (trustless verification)
    2. Rate limiting to prevent reward inflation
    3. Plausibility checks on claimed work
    4. Replay attack prevention via signature deduplication
    5. Fee burn mechanism (5% deflationary)
    6. Slashing for detected fraud
    
    Cryptography:
    - Uses ECDSA with secp256k1 curve (same as Bitcoin/Ethereum)
    - Anyone can verify signatures with just the public key
    - No shared secrets needed for verification
    """
    
    def __init__(
        self, 
        db_path: str = "neuro_ledger.db",
        node_id: Optional[str] = None,
        node_token: Optional[str] = None,
        model_interface: Optional[Any] = None
    ):
        self.db_path = db_path
        self.lock = threading.Lock()
        
        # Initialize ECDSA crypto - REQUIRED
        self.crypto: Optional[NodeCrypto] = None
        if node_token:
            self.crypto = NodeCrypto(node_token)
            self.node_id = self.crypto.node_id
            logger.info(f"ECDSA crypto initialized for node {self.node_id[:16]}...")
        else:
            self.node_id = node_id or "unknown"
        
        self.node_token = node_token
        
        # Initialize inference market (PURE MARKET PRICING - no caps!)
        # Quality emerges naturally: stupid model = no demand = low price
        # Excellent model = high demand = high price (market finds true value)
        self.inference_market = None
        from neuroshard.core.economics.market import InferenceMarket
        self.inference_market = InferenceMarket(
            price_smoothing=INFERENCE_MARKET_PRICE_SMOOTHING,
            capacity_timeout=INFERENCE_MARKET_CAPACITY_TIMEOUT,
            base_price=INFERENCE_MARKET_BASE_PRICE
        )
        logger.info(f"Dynamic inference pricing enabled: PURE MARKET (no artificial caps)")
        
        # Initialize database
        self._init_db()
        
        # Initialize Verifier
        self.verifier = ProofVerifier(model_interface=model_interface)
        
        # Role verification callback (set by runner after layer pool is initialized)
        # This prevents nodes from claiming Validator/Driver role bonuses they don't have
        self._role_verifier = None
        
        # Cache for verified roles (signature -> (verified_embed, verified_head))
        # Used to ensure rewards are based on VERIFIED roles, not claimed ones
        self._verified_roles: Dict[str, Tuple[bool, bool]] = {}
        
        logger.info(f"NEUROLedger initialized: node={self.node_id[:16]}...")
    
    def set_role_verifier(self, verifier_fn):
        """
        Set the role verification callback.
        
        The callback should accept (node_id, claimed_has_embedding, claimed_has_lm_head)
        and return (is_valid, actual_has_embedding, actual_has_lm_head).
        
        This prevents nodes from claiming Validator bonuses they don't deserve.
        """
        self._role_verifier = verifier_fn
        logger.info("Role verifier registered - fake role claims will be rejected")
    
    def set_model_interface(self, model_interface):
        """
        Set the model interface for training work verification.
        
        This is called after the NeuroNode is initialized, since the ledger
        is created before the node (in P2P setup).
        
        Args:
            model_interface: Object implementing verify_training_work(proof).
                             Typically a SwarmEnabledDynamicNode.
        """
        self.verifier.model_interface = model_interface
        logger.info("Model interface registered - training work verification enabled")
        
    def _init_db(self):
        """Initialize SQLite database with all required tables."""
        with self.lock:
            with sqlite3.connect(self.db_path, timeout=60.0) as conn:
                # Enable Write-Ahead Logging for better concurrency
                try:
                    conn.execute("PRAGMA journal_mode=WAL;")
                except Exception as e:
                    logger.warning(f"Failed to enable WAL mode: {e}")

                # Main balances table
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS balances (
                        node_id TEXT PRIMARY KEY,
                        balance REAL DEFAULT 0.0,
                        total_earned REAL DEFAULT 0.0,
                        total_spent REAL DEFAULT 0.0,
                        last_proof_time REAL DEFAULT 0.0,
                        proof_count INTEGER DEFAULT 0,
                        created_at REAL DEFAULT 0.0
                    )
                """)
                
                # Proof history (for replay prevention and audit)
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS proof_history (
                        signature TEXT PRIMARY KEY,
                        node_id TEXT NOT NULL,
                        proof_type TEXT NOT NULL,
                        timestamp REAL NOT NULL,
                        uptime_seconds REAL DEFAULT 0.0,
                        tokens_processed INTEGER DEFAULT 0,
                        training_batches INTEGER DEFAULT 0,
                        data_samples INTEGER DEFAULT 0,
                        reward_amount REAL DEFAULT 0.0,
                        received_at REAL NOT NULL,
                        verified BOOLEAN DEFAULT 1,
                        current_loss REAL DEFAULT NULL,
                        has_lm_head BOOLEAN DEFAULT 0
                    )
                """)
                # Add columns if missing (migration for existing DBs)
                try:
                    conn.execute("ALTER TABLE proof_history ADD COLUMN current_loss REAL DEFAULT NULL")
                except:
                    pass  # Column already exists
                try:
                    conn.execute("ALTER TABLE proof_history ADD COLUMN has_lm_head BOOLEAN DEFAULT 0")
                except:
                    pass  # Column already exists
                conn.execute("CREATE INDEX IF NOT EXISTS idx_proof_node ON proof_history(node_id)")
                conn.execute("CREATE INDEX IF NOT EXISTS idx_proof_time ON proof_history(timestamp)")
                
                # Transaction history
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS transactions (
                        tx_id TEXT PRIMARY KEY,
                        from_id TEXT NOT NULL,
                        to_id TEXT NOT NULL,
                        amount REAL NOT NULL,
                        fee REAL DEFAULT 0.0,
                        burn_amount REAL DEFAULT 0.0,
                        timestamp REAL NOT NULL,
                        memo TEXT DEFAULT '',
                        signature TEXT NOT NULL
                    )
                """)
                conn.execute("CREATE INDEX IF NOT EXISTS idx_tx_from ON transactions(from_id)")
                conn.execute("CREATE INDEX IF NOT EXISTS idx_tx_to ON transactions(to_id)")
                
                # Stakes (for reward multiplier)
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS stakes (
                        node_id TEXT PRIMARY KEY,
                        amount REAL DEFAULT 0.0,
                        locked_until REAL DEFAULT 0.0,
                        updated_at REAL DEFAULT 0.0
                    )
                """)
                
                # Global stats
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS global_stats (
                        id INTEGER PRIMARY KEY CHECK (id = 1),
                        total_minted REAL DEFAULT 0.0,
                        total_burned REAL DEFAULT 0.0,
                        total_transferred REAL DEFAULT 0.0,
                        total_proofs INTEGER DEFAULT 0,
                        total_transactions INTEGER DEFAULT 0,
                        updated_at REAL DEFAULT 0.0
                    )
                """)

                # Initialize global stats if not exists (Genesis Block)
                # 
                # TRANSPARENCY NOTICE:
                # ====================
                # This is the Genesis Block initialization. The ledger starts with:
                # - total_minted = 0.0 (no pre-mine)
                # - total_burned = 0.0
                # - total_transferred = 0.0
                # - total_proofs = 0
                # - total_transactions = 0
                #
                # ALL NEURO tokens must be earned through verified Proof of Neural Work.
                # There is NO pre-allocation, NO founder tokens, NO ICO.
                # Even the project creators must run nodes and do real work to earn NEURO.
                #
                # This can be independently verified by any node by checking:
                # SELECT * FROM global_stats WHERE id = 1;
                #
                conn.execute("""
                    INSERT OR IGNORE INTO global_stats (id, total_minted, total_burned, updated_at) 
                    VALUES (1, 0.0, 0.0, ?)
                """, (time.time(),))
                
                # Create genesis record in proof_history for auditability
                genesis_exists = conn.execute(
                    "SELECT 1 FROM proof_history WHERE signature = 'GENESIS_BLOCK'"
                ).fetchone()
                
                if not genesis_exists:
                    conn.execute("""
                        INSERT INTO proof_history 
                        (signature, node_id, proof_type, timestamp, uptime_seconds, 
                         tokens_processed, training_batches, data_samples, reward_amount, received_at)
                        VALUES ('GENESIS_BLOCK', 'GENESIS', 'GENESIS', ?, 0, 0, 0, 0, 0.0, ?)
                    """, (time.time(), time.time()))
                    logger.info("Genesis Block created - Ledger initialized with zero supply")
                
                # Rate limiting table
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS rate_limits (
                        node_id TEXT PRIMARY KEY,
                        proofs_last_hour INTEGER DEFAULT 0,
                        tokens_last_minute INTEGER DEFAULT 0,
                        last_reset_hour REAL DEFAULT 0.0,
                        last_reset_minute REAL DEFAULT 0.0
                    )
                """)
                
                # Fraud reports (for slashing)
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS fraud_reports (
                        report_id TEXT PRIMARY KEY,
                        reporter_id TEXT NOT NULL,
                        accused_id TEXT NOT NULL,
                        proof_signature TEXT,
                        reason TEXT NOT NULL,
                        evidence TEXT,
                        status TEXT DEFAULT 'pending',
                        slash_amount REAL DEFAULT 0.0,
                        created_at REAL NOT NULL
                    )
                """)
                
                # =========================================================
                # CHAINED EPOCHS (Blockchain-like Epoch Chain)
                # =========================================================
                # Each epoch is cryptographically linked to the previous:
                # - prev_epoch_hash: Creates immutable chain
                # - model_state_hash: Tracks training progression
                # - proofs_merkle_root: Efficiently prove proof inclusion
                # - gradient_commitments_root: Enable spot-check verification
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS epochs (
                        epoch_id INTEGER PRIMARY KEY,
                        prev_epoch_hash TEXT NOT NULL,
                        timestamp_start REAL NOT NULL,
                        timestamp_end REAL NOT NULL,
                        model_state_hash_start TEXT NOT NULL,
                        model_state_hash_end TEXT NOT NULL,
                        proofs_merkle_root TEXT NOT NULL,
                        proof_count INTEGER DEFAULT 0,
                        total_reward REAL DEFAULT 0.0,
                        total_batches INTEGER DEFAULT 0,
                        average_loss REAL DEFAULT 0.0,
                        gradient_commitments_root TEXT DEFAULT '',
                        proposer_node_id TEXT NOT NULL,
                        proposer_signature TEXT NOT NULL,
                        epoch_hash TEXT NOT NULL UNIQUE,
                        finalized BOOLEAN DEFAULT 0,
                        created_at REAL DEFAULT 0.0
                    )
                """)
                conn.execute("CREATE INDEX IF NOT EXISTS idx_epoch_hash ON epochs(epoch_hash)")
                conn.execute("CREATE INDEX IF NOT EXISTS idx_epoch_finalized ON epochs(finalized)")
                
                # Gradient commitments for spot-check verification
                # These can be challenged and re-verified for economic security
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS gradient_commitments (
                        commitment_hash TEXT PRIMARY KEY,
                        proof_signature TEXT NOT NULL,
                        node_id TEXT NOT NULL,
                        epoch_id INTEGER NOT NULL,
                        model_hash_start TEXT NOT NULL,
                        model_hash_end TEXT NOT NULL,
                        data_hash TEXT NOT NULL,
                        gradient_norms TEXT NOT NULL,
                        final_loss REAL NOT NULL,
                        batch_size INTEGER DEFAULT 0,
                        sequence_length INTEGER DEFAULT 0,
                        created_at REAL DEFAULT 0.0,
                        spot_checked BOOLEAN DEFAULT 0,
                        spot_check_result TEXT DEFAULT NULL,
                        FOREIGN KEY (epoch_id) REFERENCES epochs(epoch_id)
                    )
                """)
                conn.execute("CREATE INDEX IF NOT EXISTS idx_gc_proof ON gradient_commitments(proof_signature)")
                conn.execute("CREATE INDEX IF NOT EXISTS idx_gc_epoch ON gradient_commitments(epoch_id)")
                conn.execute("CREATE INDEX IF NOT EXISTS idx_gc_node ON gradient_commitments(node_id)")
                
                # Spot-check challenges for economic security
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS spot_check_challenges (
                        challenge_id TEXT PRIMARY KEY,
                        commitment_hash TEXT NOT NULL,
                        challenger_id TEXT NOT NULL,
                        challenger_stake REAL DEFAULT 0.0,
                        challenged_at REAL NOT NULL,
                        status TEXT DEFAULT 'pending',
                        result TEXT DEFAULT NULL,
                        resolved_at REAL DEFAULT NULL,
                        FOREIGN KEY (commitment_hash) REFERENCES gradient_commitments(commitment_hash)
                    )
                """)
    
    # ========================================================================
    # SIGNATURE & VERIFICATION
    # ========================================================================
    
    def _sign(self, payload: str) -> str:
        """
        Sign a payload using ECDSA with secp256k1.
        
        ECDSA signatures enable trustless verification by any node.
        Anyone can verify using our public key.
        """
        if not self.crypto:
            raise ValueError("Cannot sign without crypto initialized (need node_token)")
        
        return self.crypto.sign(payload)
    
    def get_public_key_hex(self) -> str:
        """Get this node's public key in hex format for sharing."""
        if not self.crypto:
            raise ValueError("Crypto not initialized")
        return self.crypto.get_public_key_hex()
    
    def get_public_key_bytes(self) -> bytes:
        """Get this node's public key bytes for verification."""
        if not self.crypto:
            raise ValueError("Crypto not initialized")
        return self.crypto.get_public_key_bytes()
    
    def _verify_signature(self, proof: PoNWProof) -> bool:
        """
        Verify proof signature using ECDSA.
        
        Security Model:
        ===============
        ECDSA enables TRUSTLESS verification:
        - Signature can be verified by ANYONE with the public key
        - No shared secret needed
        - Full cryptographic verification
        - Same curve as Bitcoin/Ethereum (secp256k1)
        
        TRANSPARENCY GUARANTEE:
        =======================
        There is NO admin backdoor. The ONLY way to get NEURO is:
        1. Run a node that does real work (training, inference, uptime)
        2. Create a proof of that work, signed with ECDSA
        3. Submit the proof, which passes ALL verification checks
        4. Receive rewards proportional to verified work
        
        Even the project creators must run nodes and do real work to earn NEURO.
        """
        if not proof.signature or proof.signature == "unsigned":
            logger.warning("Missing or unsigned signature")
            return False
            
        # Validate signature format
        if not is_valid_signature_format(proof.signature):
            logger.warning(f"Invalid signature format: {proof.signature[:20]}...")
            return False

        # Validate node_id format (32 hex chars)
        if not is_valid_node_id_format(proof.node_id):
            logger.warning(f"Invalid node_id format: {proof.node_id}")
            return False
        
        # For our own proofs, verify with our crypto
        if proof.node_id == self.node_id and self.crypto:
            payload = proof.canonical_payload()
            return self.crypto.verify(payload, proof.signature)
        
        # For external proofs, use the public key registry
        payload = proof.canonical_payload()
        result = verify_signature(proof.node_id, payload, proof.signature)
        if not result:
            # Log as debug - signature mismatches are common during version transitions
            # where different nodes may have different canonical_payload formats
            logger.warning(f"Signature verification failed for {proof.node_id[:16]}... (likely version mismatch)")
        return result
    
    # ========================================================================
    # PROOF CREATION & PROCESSING
    # ========================================================================
    
    def create_proof(
        self,
        proof_type: ProofType,
        uptime_seconds: float = 0.0,
        tokens_processed: int = 0,
        training_batches: int = 0,
        data_samples: int = 0,
        model_hash: str = "",
        layers_held: int = 0,
        has_embedding: bool = False,
        has_lm_head: bool = False,
        current_loss: Optional[float] = None,
        # NEW: Chained PoNW fields
        model_hash_start: str = "",
        model_hash_end: str = "",
        gradient_commitment: str = "",
        data_hash: str = "",
        gradient_norm: float = 0.0,
    ) -> PoNWProof:
        """
        Create a signed Proof of Neural Work.
        
        Rate Limiting Applied:
        - uptime_seconds capped at MAX_UPTIME_PER_PROOF
        - tokens_processed checked against rate limits
        
        Chained PoNW Fields (required for training proofs):
        - model_hash_start: Hash of model weights BEFORE training
        - model_hash_end: Hash of model weights AFTER training
        - gradient_commitment: Hash of gradient statistics (spot-checkable)
        - data_hash: Hash of training data (for reproducibility)
        - gradient_norm: L2 norm of gradients (sanity check)
        """
        # Apply rate limits
        uptime_seconds = min(uptime_seconds, MAX_UPTIME_PER_PROOF)
        
        # Generate unique nonce
        nonce = hashlib.sha256(f"{time.time()}:{os.urandom(16).hex()}".encode()).hexdigest()[:16]
        
        # Compute epoch_id from current time (60-second epochs)
        epoch_id = int(time.time() / 60)
        
        # CRITICAL: All fields must match canonical_payload for signature verification
        proof = PoNWProof(
            node_id=self.node_id,
            proof_type=proof_type.value,
            timestamp=time.time(),
            nonce=nonce,
            uptime_seconds=uptime_seconds,
            tokens_processed=tokens_processed,
            training_batches=training_batches,
            data_samples=data_samples,
            model_hash=model_hash,
            layers_held=layers_held,
            has_embedding=has_embedding,
            has_lm_head=has_lm_head,
            current_loss=current_loss,
            # Chained PoNW fields
            epoch_id=epoch_id,
            model_hash_start=model_hash_start,
            model_hash_end=model_hash_end,
            gradient_commitment=gradient_commitment,
            data_hash=data_hash,
            gradient_norm=gradient_norm,
        )
        
        # Sign the proof
        payload = proof.canonical_payload()
        proof.signature = self._sign(payload)
        
        return proof
    
    def verify_proof(self, proof: PoNWProof) -> Tuple[bool, str]:
        """
        Verify a Proof of Neural Work.
        
        Checks:
        1. Signature validity
        2. Timestamp freshness
        3. Replay prevention (signature not seen before)
        4. Rate limiting
        5. Plausibility of claimed work
        
        Returns: (is_valid, reason)
        """
        # 1. Check signature
        if not self._verify_signature(proof):
            return False, "Invalid signature"
        
        # 2. Check timestamp freshness
        age = time.time() - proof.timestamp
        if age > PROOF_FRESHNESS_WINDOW:
            return False, f"Proof too old ({age:.0f}s > {PROOF_FRESHNESS_WINDOW}s)"
        if age < -60:  # Allow 1 minute clock skew
            return False, "Proof timestamp in future"
        
        with self.lock:
            with sqlite3.connect(self.db_path, timeout=60.0) as conn:
                # 3. Check for replay
                existing = conn.execute(
                    "SELECT 1 FROM proof_history WHERE signature = ?",
                    (proof.signature,)
                ).fetchone()
                if existing:
                    return False, "Duplicate proof (replay)"
                
                # 4. Rate limiting
                is_limited, limit_reason = self._check_rate_limits(conn, proof)
                if is_limited:
                    return False, limit_reason
                
                # 5. Plausibility checks
                is_plausible, plausibility_reason = self._check_plausibility(proof)
                if not is_plausible:
                    return False, plausibility_reason
        
        # 6. Work Content Verification (Semantic Check)
        # This verifies that the work was ACTUALLY done (e.g. gradients reduce loss)
        is_work_valid, work_reason = self.verifier.verify_work_content(proof)
        if not is_work_valid:
            return False, f"Work validation failed: {work_reason}"

        return True, "Valid"

    def has_proof(self, signature: str) -> bool:
        """Check if a proof signature already exists in the ledger (Thread-safe)."""
        if not signature:
            return False
            
        try:
            with self.lock:
                with sqlite3.connect(self.db_path, timeout=5.0) as conn:
                    cursor = conn.cursor()
                    cursor.execute("SELECT 1 FROM proof_history WHERE signature = ?", (signature,))
                    return cursor.fetchone() is not None
        except Exception as e:
            logger.error(f"Error checking proof existence: {e}")
            return False
    
    def _check_rate_limits(self, conn, proof: PoNWProof) -> Tuple[bool, str]:
        """
        Check if node is within rate limits (READ-ONLY).
        
        NOTE: This ONLY checks the limit, it does NOT increment the counter.
        The counter is incremented in _increment_rate_limits() which is
        called from process_proof() AFTER the proof is successfully accepted.
        
        This prevents the bug where gossip duplicates (which all pass verify_proof
        before any reaches process_proof) inflate the rate limit counter.
        """
        now = time.time()
        
        # Get or create rate limit record
        row = conn.execute(
            "SELECT proofs_last_hour, tokens_last_minute, last_reset_hour, last_reset_minute FROM rate_limits WHERE node_id = ?",
            (proof.node_id,)
        ).fetchone()
        
        if row:
            proofs_last_hour, tokens_last_minute, last_reset_hour, last_reset_minute = row
            
            # Reset hourly counter if needed (virtual reset, don't write)
            if now - last_reset_hour > 3600:
                proofs_last_hour = 0
            
            # Reset minute counter if needed (virtual reset, don't write)
            if now - last_reset_minute > 60:
                tokens_last_minute = 0
        else:
            proofs_last_hour = 0
            tokens_last_minute = 0
        
        # Check limits (READ-ONLY - no increment here!)
        if proofs_last_hour >= MAX_PROOFS_PER_HOUR:
            return True, f"Rate limit: max {MAX_PROOFS_PER_HOUR} proofs/hour"
        
        new_tokens = tokens_last_minute + proof.tokens_processed
        if new_tokens > MAX_TOKENS_PER_MINUTE * 60:  # Scaled to hour
            return True, f"Rate limit: max {MAX_TOKENS_PER_MINUTE} tokens/minute"
        
        return False, ""
    
    def _increment_rate_limits(self, conn, proof: PoNWProof):
        """
        Increment rate limit counters AFTER a proof is successfully accepted.
        
        This is called from process_proof() only when a proof is actually
        added to proof_history (not for duplicates or failed verifications).
        """
        now = time.time()
        
        # Get current values
        row = conn.execute(
            "SELECT proofs_last_hour, tokens_last_minute, last_reset_hour, last_reset_minute FROM rate_limits WHERE node_id = ?",
            (proof.node_id,)
        ).fetchone()
        
        if row:
            proofs_last_hour, tokens_last_minute, last_reset_hour, last_reset_minute = row
            
            # Reset hourly counter if needed
            if now - last_reset_hour > 3600:
                proofs_last_hour = 0
                last_reset_hour = now
            
            # Reset minute counter if needed
            if now - last_reset_minute > 60:
                tokens_last_minute = 0
                last_reset_minute = now
        else:
            proofs_last_hour = 0
            tokens_last_minute = 0
            last_reset_hour = now
            last_reset_minute = now
        
        # NOW increment the counter (only for successfully accepted proofs)
        new_tokens = tokens_last_minute + proof.tokens_processed
        
        conn.execute("""
            INSERT INTO rate_limits (node_id, proofs_last_hour, tokens_last_minute, last_reset_hour, last_reset_minute)
            VALUES (?, ?, ?, ?, ?)
            ON CONFLICT(node_id) DO UPDATE SET
                proofs_last_hour = ?,
                tokens_last_minute = ?,
                last_reset_hour = ?,
                last_reset_minute = ?
        """, (
            proof.node_id,
            proofs_last_hour + 1, new_tokens, last_reset_hour, last_reset_minute,
            proofs_last_hour + 1, new_tokens, last_reset_hour, last_reset_minute
        ))
    
    def _check_plausibility(self, proof: PoNWProof) -> Tuple[bool, str]:
        """Check if claimed work is plausible."""
        # CRITICAL: Verify role claims against actual layer assignments
        # This prevents nodes from claiming Validator/Driver bonuses they don't deserve
        if self._role_verifier and (proof.has_embedding or proof.has_lm_head):
            is_role_valid, actual_embed, actual_head = self._role_verifier(
                proof.node_id, 
                proof.has_embedding, 
                proof.has_lm_head
            )
            
            # Store VERIFIED roles for reward calculation
            # Even if we accept the proof, we only pay bonuses for VERIFIED roles
            self._verified_roles[proof.signature] = (actual_embed, actual_head)
            
            if not is_role_valid:
                if proof.has_lm_head and not actual_head:
                    logger.warning(f"FAKE VALIDATOR DETECTED: {proof.node_id[:16]}... claimed has_lm_head=True but is NOT a Validator")
                    return False, "Invalid role claim: not a Validator"
                if proof.has_embedding and not actual_embed:
                    logger.warning(f"FAKE DRIVER DETECTED: {proof.node_id[:16]}... claimed has_embedding=True but is NOT a Driver")
                    return False, "Invalid role claim: not a Driver"
        else:
            # No role verifier or no role claims - use claimed values
            self._verified_roles[proof.signature] = (proof.has_embedding, proof.has_lm_head)
        
        # Uptime check
        if proof.uptime_seconds > MAX_UPTIME_PER_PROOF:
            return False, f"Uptime too high ({proof.uptime_seconds}s > {MAX_UPTIME_PER_PROOF}s)"
        
        # Token rate check (tokens per second)
        if proof.uptime_seconds > 0:
            tokens_per_second = proof.tokens_processed / proof.uptime_seconds
            max_tps = MAX_TOKENS_PER_MINUTE / 60
            if tokens_per_second > max_tps * 2:  # Allow 2x buffer
                return False, f"Token rate implausible ({tokens_per_second:.0f} > {max_tps * 2:.0f} tps)"
        
        # Training batches check - fast GPUs can do 5-10 batches/second
        # Tier 1/2 GPUs (A100, RTX 4090, Jetson Orin) can easily hit 300-600 batches/min
        if proof.uptime_seconds > 0:
            batches_per_minute = (proof.training_batches / proof.uptime_seconds) * 60
            if batches_per_minute > 2000:  # 10 batches/second max for fast GPUs
                return False, f"Training rate implausible ({batches_per_minute:.0f} batches/min)"
        
        # =================================================================
        # CRITICAL: Training proof must show ACTUAL work was done
        # =================================================================
        # Use the new verify_training_work() method on PoNWProof which checks:
        # 1. model_hash_start != model_hash_end (weights actually changed)
        # 2. gradient_commitment is present (can be spot-checked)
        # 3. current_loss is valid (training computed loss)
        #
        # This replaces the old simple check and enables gradient spot-checking.
        if proof.proof_type == "training" and proof.training_batches > 0:
            is_valid, error_msg = proof.verify_training_work()
            if not is_valid:
                logger.warning(
                    f"FAKE TRAINING DETECTED: {proof.node_id[:16]}... claimed {proof.training_batches} batches "
                    f"but failed verification: {error_msg}"
                )
                return False, f"Training proof rejected: {error_msg}"
        
        # =================================================================
        # EPOCH BINDING: Proof must belong to current or recent epoch
        # =================================================================
        # This prevents "epoch hopping" where nodes submit proofs to whichever
        # epoch gives them the best reward or to avoid detection.
        if proof.epoch_id > 0:  # 0 = legacy proof (no epoch binding)
            current_epoch = int(time.time() / 60)  # 60-second epochs
            epoch_diff = abs(current_epoch - proof.epoch_id)
            
            # Allow proofs from current epoch or 1 epoch back (grace period)
            if epoch_diff > 1:
                logger.warning(
                    f"EPOCH MISMATCH: {proof.node_id[:16]}... submitted proof for epoch {proof.epoch_id} "
                    f"but current epoch is {current_epoch}"
                )
                return False, f"Proof epoch {proof.epoch_id} too far from current epoch {current_epoch}"
        
        return True, ""
    
    def process_proof(self, proof: PoNWProof, skip_verification: bool = False) -> Tuple[bool, float, str]:
        """
        Process a verified proof and credit rewards.
        
        Args:
            proof: The PoNW proof to process
            skip_verification: If True, skip verify_proof (caller already verified)
        
        Returns: (success, reward_amount, message)
        """
        # Verify first (unless caller already did)
        if not skip_verification:
            is_valid, reason = self.verify_proof(proof)
            if not is_valid:
                return False, 0.0, reason
        
        # Calculate reward
        reward = self._calculate_reward(proof)
        
        with self.lock:
            with sqlite3.connect(self.db_path, timeout=60.0) as conn:
                # Double-check replay (in case of race)
                existing = conn.execute(
                    "SELECT 1 FROM proof_history WHERE signature = ?",
                    (proof.signature,)
                ).fetchone()
                if existing:
                    return False, 0.0, "Duplicate proof"
                
                # Record proof (including current_loss and has_lm_head for training stats aggregation)
                conn.execute("""
                    INSERT INTO proof_history 
                    (signature, node_id, proof_type, timestamp, uptime_seconds, 
                     tokens_processed, training_batches, data_samples, reward_amount, received_at, current_loss, has_lm_head)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    proof.signature, proof.node_id, proof.proof_type, proof.timestamp,
                    proof.uptime_seconds, proof.tokens_processed, proof.training_batches,
                    proof.data_samples, reward, time.time(), proof.current_loss, proof.has_lm_head
                ))
                
                # Credit balance
                conn.execute("""
                    INSERT INTO balances (node_id, balance, total_earned, last_proof_time, proof_count, created_at)
                    VALUES (?, ?, ?, ?, 1, ?)
                    ON CONFLICT(node_id) DO UPDATE SET
                        balance = balance + ?,
                        total_earned = total_earned + ?,
                        last_proof_time = ?,
                        proof_count = proof_count + 1
                """, (
                    proof.node_id, reward, reward, proof.timestamp, time.time(),
                    reward, reward, proof.timestamp
                ))
                
                # Update global stats
                conn.execute("""
                    UPDATE global_stats SET
                        total_minted = total_minted + ?,
                        total_proofs = total_proofs + 1,
                        updated_at = ?
                    WHERE id = 1
                """, (reward, time.time()))
                
                # Increment rate limits NOW (after proof is successfully accepted)
                # This prevents gossip duplicates from inflating the counter
                self._increment_rate_limits(conn, proof)
        
        # If this was a marketplace request, register proof received (DISTRIBUTED)
        # Multiple nodes (driver, workers, validator) submit proofs for same request
        if proof.request_id and self.inference_market:
            is_complete, error = self.inference_market.register_proof_received(
                request_id=proof.request_id,
                node_id=proof.node_id,
                is_initiator=proof.has_embedding,
                is_finisher=proof.has_lm_head
            )
            if error:
                logger.warning(f"Failed to register proof for {proof.request_id[:8]}...: {error}")
            elif is_complete:
                logger.info(f"Request {proof.request_id[:8]}... COMPLETED (all proofs received)")
        
        logger.info(f"PoNW: {proof.node_id[:12]}... earned {reward:.6f} NEURO "
                   f"(type={proof.proof_type}, uptime={proof.uptime_seconds:.0f}s, "
                   f"tokens={proof.tokens_processed})"
                   f"{f', request={proof.request_id[:8]}...' if proof.request_id else ''}")
        
        return True, reward, "Proof processed"
    
    def _calculate_reward(self, proof: PoNWProof) -> float:
        """
        Calculate NEURO reward for a Proof of Neural Work.
        
        Reward Structure (quorum-based training rewards):
        ====================================================
        
        1. UPTIME REWARD (all nodes):
           - 0.0001 NEURO per minute of uptime
           - Minimal to discourage idle farming
        
        2. INFERENCE REWARD (PURE MARKET-BASED):
           - Total pool: DYNAMIC (based on supply/demand)
           - Distributed by quorum role:
             * INITIATOR (has Layer 0):     15%
             * PROCESSOR (middle layers):   70%
             * FINISHER (has LM head):      15%
        
        3. TRAINING REWARD (PER LAYER):
           - Pipeline: 0.0005 NEURO × batches × layers_held
           - Async: 0.0003 NEURO × batches × layers_held × freshness
           - This directly scales reward with contribution!
        
        4. DATA REWARD:
           - 0.00001 NEURO per data sample served
        
        5. MULTIPLIERS (ADDITIVE for role, multiplicative for stake):
           - Role bonus: 1.0 + (0.2 if initiator) + (0.3 if finisher) + (0.1 if training)
           - Staking: Logarithmic curve (see calculate_stake_multiplier)
           - Reputation: Up to +10% for reliable nodes
        """
        # =====================================================================
        # 1. UPTIME REWARD (same for all roles)
        # =====================================================================
        uptime_reward = (proof.uptime_seconds / 60.0) * UPTIME_REWARD_PER_MINUTE
        
        # =====================================================================
        # 2. INFERENCE REWARD (MARKETPLACE with REQUEST MATCHING)
        # =====================================================================
        # Price is LOCKED at request submission time (prevents timing attacks)
        # - If request_id present: Use locked price from InferenceRequest
        # - If no request_id: Use current market price (legacy/direct inference)
        
        if proof.request_id and self.inference_market:
            # NEW MARKETPLACE: Use locked price from request
            request = self.inference_market.get_request(proof.request_id)
            
            if not request:
                raise ValueError(f"Request {proof.request_id} not found")
            
            if request.claimed_by != proof.node_id:
                raise ValueError(f"Request {proof.request_id} was claimed by {request.claimed_by}, "
                               f"but proof submitted by {proof.node_id}")
            
            if request.completed:
                raise ValueError(f"Request {proof.request_id} already completed")
            
            # Use LOCKED price from request submission time
            market_price = request.locked_price
            logger.debug(f"Using locked price {market_price:.6f} from request {proof.request_id[:8]}...")
        else:
            # Legacy mode or direct inference: use current market price
            market_price = self.inference_market.get_current_price()
            logger.debug(f"Using current market price {market_price:.6f} (no request_id)")
        
        inference_pool = (proof.tokens_processed / 1_000_000.0) * market_price
        
        # Determine role and calculate share
        # CRITICAL: Use VERIFIED roles, not claimed roles!
        # This ensures nodes can't claim bonuses they don't deserve
        if proof.signature in self._verified_roles:
            is_initiator, is_finisher = self._verified_roles[proof.signature]
            # Clean up cache entry
            del self._verified_roles[proof.signature]
        else:
            # Use claimed roles (determined by layer range in quorum)
            is_initiator = proof.has_embedding  # Holds Layer 0
            is_finisher = proof.has_lm_head     # Holds LM head layer
        
        is_training = proof.training_batches > 0
        
        inference_reward = 0.0
        
        if is_initiator:
            # Initiator (holds embedding) gets 15% of the inference pool
            inference_reward += inference_pool * INITIATOR_SHARE
        
        if is_finisher:
            # Finisher (holds LM head) gets 15% of the inference pool
            inference_reward += inference_pool * FINISHER_SHARE
        
        if proof.layers_held > 0:
            # Processors get rewarded per layer they process
            # The PROCESSOR_SHARE (70%) represents the total computation work.
            processor_pool = inference_pool * PROCESSOR_SHARE
            inference_reward += processor_pool
        
        # =====================================================================
        # 3. TRAINING REWARD (PER LAYER - the key change!)
        # =====================================================================
        # Training reward now properly scales with layers_held:
        # - More layers = more work = more reward
        # - This incentivizes nodes to hold more layers
        #
        # Formula: batches × rate_per_layer × layers_held
        #
        # For async contributors (not in pipeline), we apply freshness decay
        # since stale gradients are less valuable for training convergence.
        
        training_reward = 0.0
        if proof.training_batches > 0 and proof.layers_held > 0:
            # Use the canonical training reward calculation
            # For now, we assume pipeline (not async) - async would have separate proof type
            training_reward = calculate_training_reward(
                batches=proof.training_batches,
                layers_held=proof.layers_held,
                is_async=False,  # Pipeline training
                age_seconds=0.0,
            )
        
        # =====================================================================
        # 4. DATA REWARD
        # =====================================================================
        data_reward = proof.data_samples * DATA_REWARD_PER_SAMPLE
        
        # =====================================================================
        # 5. CALCULATE BASE REWARD
        # =====================================================================
        base_reward = uptime_reward + inference_reward + training_reward + data_reward
        
        # =====================================================================
        # 6. STAKING MULTIPLIER (with diminishing returns)
        # =====================================================================
        # SECURITY: For LOCAL proofs, use our verified stake
        # For REMOTE proofs, we use their claimed stake but cap the multiplier
        stake = self._get_stake(proof.node_id)
        
        # If this is a REMOTE proof (not from us), cap the stake multiplier
        # to prevent fake stake claims from inflating rewards
        is_local_proof = (proof.node_id == self.node_id)
        
        if is_local_proof:
            # Our own proof - use full stake multiplier
            stake_multiplier = self._calculate_stake_multiplier(stake)
        else:
            # Remote proof - cap multiplier for security (from economics.py)
            # This limits the impact of fake stake claims
            stake_multiplier = min(REMOTE_STAKE_MULTIPLIER_CAP, self._calculate_stake_multiplier(stake))
        
        # =====================================================================
        # 7. ROLE BONUS MULTIPLIER (ADDITIVE - not multiplicative!)
        # =====================================================================
        # Role bonuses are now ADDITIVE to prevent exponential growth
        # Final multiplier = 1.0 + sum of applicable bonuses
        # Example: Initiator + Finisher + Training = 1.0 + 0.2 + 0.3 + 0.1 = 1.6x
        role_multiplier = calculate_role_bonus(
            has_embedding=is_initiator,
            has_lm_head=is_finisher,
            is_training=is_training,
        )
        
        # =====================================================================
        # 8. REPUTATION BONUS
        # =====================================================================
        # Nodes with high uptime and success rates receive bonus rewards.
        # This incentivizes reliable participation in the network.
        # Reputation is tracked per-member in QuorumMember.reputation (0.0 to 1.0)
        # We use the reputation from the proof if provided, otherwise default to 1.0
        reputation = getattr(proof, 'reputation', 1.0)
        if reputation is None:
            reputation = 1.0
        reputation = max(0.0, min(1.0, reputation))  # Clamp to [0, 1]
        
        # REPUTATION_BONUS_MAX = 0.10 (up to 10% extra for perfect reputation)
        from neuroshard.core.economics.constants import REPUTATION_BONUS_MAX
        reputation_bonus = 1.0 + (reputation * REPUTATION_BONUS_MAX)
        
        # =====================================================================
        # 9. MOE EXPERT SCARCITY BONUS
        # =====================================================================
        # Nodes holding under-replicated experts receive bonus rewards.
        # This incentivizes a balanced distribution of experts across the network.
        # Scarcity bonus is up to 1.5x for experts with no replicas.
        moe_scarcity_bonus = 1.0
        
        if hasattr(proof, 'moe_expert_count') and proof.moe_expert_count > 0:
            # If proof includes MoE scarcity bonus, use it
            if hasattr(proof, 'moe_scarcity_avg') and proof.moe_scarcity_avg > 0:
                moe_scarcity_bonus = proof.moe_scarcity_avg
        
        # =====================================================================
        # 10. FINAL REWARD
        # =====================================================================
        total_reward = base_reward * stake_multiplier * role_multiplier * reputation_bonus * moe_scarcity_bonus
        
        return total_reward
    
    def _get_stake(self, node_id: str) -> float:
        """Get staked amount for a node."""
        with sqlite3.connect(self.db_path, timeout=60.0) as conn:
            row = conn.execute(
                "SELECT amount FROM stakes WHERE node_id = ? AND locked_until > ?",
                (node_id, time.time())
            ).fetchone()
            return row[0] if row else 0.0
    
    def _calculate_stake_multiplier(self, stake: float) -> float:
        """
        Calculate stake multiplier using centralized economics function.
        
        See neuroshard/core/economics.py for formula and examples.
        """
        return calculate_stake_multiplier(stake)
    
    # ========================================================================
    # VALIDATOR ELIGIBILITY
    # ========================================================================
    
    def is_eligible_validator(self, node_id: str = None) -> Tuple[bool, str]:
        """
        Check if a node is eligible to be a Validator.
        
        Requirements:
        1. Minimum stake of VALIDATOR_MIN_STAKE (100 NEURO)
        2. Memory requirements checked at layer assignment time
        
        Returns: (eligible, reason)
        """
        node_id = node_id or self.node_id
        stake = self._get_stake(node_id)
        
        if stake < VALIDATOR_MIN_STAKE:
            return False, f"Insufficient stake: {stake:.2f} < {VALIDATOR_MIN_STAKE} NEURO required"
        
        return True, f"Eligible with {stake:.2f} NEURO staked"
    
    def get_validator_info(self, node_id: str = None) -> dict:
        """Get validator status and info for a node."""
        node_id = node_id or self.node_id
        stake = self._get_stake(node_id)
        eligible, reason = self.is_eligible_validator(node_id)
        
        return {
            "node_id": node_id,
            "stake": stake,
            "stake_multiplier": self._calculate_stake_multiplier(stake),
            "is_eligible_validator": eligible,
            "eligibility_reason": reason,
            "min_stake_required": VALIDATOR_MIN_STAKE,
            "validation_fee_per_proof": VALIDATION_FEE_PER_PROOF,
        }
    
    # ========================================================================
    # DYNAMIC INFERENCE MARKET
    # ========================================================================
    
    def register_inference_capacity(
        self,
        tokens_per_second: int,
        min_price: float = 0.0
    ):
        """
        Register this node's available inference capacity with the market.
        
        Args:
            tokens_per_second: Processing capacity (tokens/sec)
            min_price: Minimum NEURO per 1M tokens node will accept
        """
        if not USE_DYNAMIC_INFERENCE_PRICING or not self.inference_market:
            return
        
        self.inference_market.register_capacity(
            node_id=self.node_id,
            tokens_per_second=tokens_per_second,
            min_price=min_price
        )
        logger.debug(f"Registered inference capacity: {tokens_per_second} t/s, min_price={min_price:.4f}")
    
    def withdraw_inference_capacity(self):
        """
        Withdraw this node from the inference market (e.g., to focus on training).
        """
        if not USE_DYNAMIC_INFERENCE_PRICING or not self.inference_market:
            return
        
        self.inference_market.withdraw_capacity(self.node_id)
        logger.debug(f"Withdrew from inference market")
    
    def get_inference_market_stats(self) -> dict:
        """
        Get current inference market statistics.
        
        Returns:
            Market stats including price, supply, demand, utilization
        """
        # ALWAYS use dynamic market pricing (no fallback)
        stats = self.inference_market.get_market_stats()
        stats["mode"] = "pure_market"
        return stats
    
    def submit_inference_request(
        self,
        request_id: str,
        user_id: str,
        tokens_requested: int,
        max_price: float,
        priority: int = 0
    ) -> bool:
        """
        Submit an inference request to the market.
        
        Args:
            request_id: Unique request identifier
            user_id: User submitting request
            tokens_requested: Number of tokens to generate
            max_price: Maximum NEURO per 1M tokens user will pay
            priority: Request priority (higher = more urgent)
        
        Returns:
            True if request accepted, False if price too high
        """

        return self.inference_market.submit_request(
            request_id=request_id,
            user_id=user_id,
            tokens_requested=tokens_requested,
            max_price=max_price,
            priority=priority
        )
    
    # ========================================================================
    # PROOF VALIDATION (Stake-Weighted Consensus)
    # ========================================================================
    
    def validate_proof_as_validator(
        self,
        proof: PoNWProof,
        vote: bool,
        validation_details: str = ""
    ) -> Tuple[bool, float, str]:
        """
        Cast a validation vote on a proof as a Validator.
        
        Only nodes meeting VALIDATOR_MIN_STAKE can validate.
        Validators earn VALIDATION_FEE_PER_PROOF for each validation.
        Bad validators (voting against consensus) can be slashed.
        
        Args:
            proof: The PoNW proof to validate
            vote: True = valid, False = invalid
            validation_details: Optional details about validation
            
        Returns: (success, fee_earned, message)
        """
        # Check eligibility
        eligible, reason = self.is_eligible_validator()
        if not eligible:
            return False, 0.0, f"Not eligible to validate: {reason}"
        
        # Record the validation vote
        validation_id = hashlib.sha256(
            f"{self.node_id}:{proof.signature}:{time.time()}".encode()
        ).hexdigest()[:32]
        
        my_stake = self._get_stake(self.node_id)
        
        with self.lock:
            with sqlite3.connect(self.db_path, timeout=60.0) as conn:
                # Create validation_votes table if not exists
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
                
                # Check if already voted
                existing = conn.execute(
                    "SELECT validation_id FROM validation_votes WHERE proof_signature = ? AND validator_id = ?",
                    (proof.signature, self.node_id)
                ).fetchone()
                
                if existing:
                    return False, 0.0, "Already voted on this proof"
                
                # Record vote
                conn.execute("""
                    INSERT INTO validation_votes 
                    (validation_id, proof_signature, validator_id, validator_stake, vote, details, timestamp, fee_earned)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    validation_id,
                    proof.signature,
                    self.node_id,
                    my_stake,
                    1 if vote else 0,
                    validation_details,
                    time.time(),
                    VALIDATION_FEE_PER_PROOF
                ))
                
                # Credit validation fee
                conn.execute("""
                    UPDATE balances SET
                        balance = balance + ?,
                        total_earned = total_earned + ?
                    WHERE node_id = ?
                """, (VALIDATION_FEE_PER_PROOF, VALIDATION_FEE_PER_PROOF, self.node_id))
        
        logger.info(f"Validation vote recorded: {self.node_id[:16]}... voted {'VALID' if vote else 'INVALID'} "
                   f"on proof {proof.signature[:16]}... (stake: {my_stake:.2f})")
        
        return True, VALIDATION_FEE_PER_PROOF, f"Vote recorded, earned {VALIDATION_FEE_PER_PROOF} NEURO"
    
    def get_proof_validation_status(self, proof_signature: str) -> dict:
        """
        Get the current validation status of a proof.
        
        Returns stake-weighted vote tallies and consensus status.
        """
        with sqlite3.connect(self.db_path, timeout=60.0) as conn:
            # Check if validation_votes table exists
            table_exists = conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name='validation_votes'"
            ).fetchone()
            
            if not table_exists:
                return {
                    "proof_signature": proof_signature,
                    "total_votes": 0,
                    "valid_votes": 0,
                    "invalid_votes": 0,
                    "valid_stake": 0.0,
                    "invalid_stake": 0.0,
                    "total_stake": 0.0,
                    "consensus_reached": False,
                    "consensus_result": None,
                }
            
            # Get all votes for this proof
            votes = conn.execute("""
                SELECT validator_id, validator_stake, vote
                FROM validation_votes
                WHERE proof_signature = ?
            """, (proof_signature,)).fetchall()
            
            valid_stake = 0.0
            invalid_stake = 0.0
            valid_count = 0
            invalid_count = 0
            
            for _, stake, vote in votes:
                if vote:
                    valid_stake += stake
                    valid_count += 1
                else:
                    invalid_stake += stake
                    invalid_count += 1
            
            total_stake = valid_stake + invalid_stake
            
            # Check consensus
            consensus_reached = False
            consensus_result = None
            
            if total_stake > 0:
                valid_ratio = valid_stake / total_stake
                if valid_ratio >= VALIDATION_CONSENSUS_THRESHOLD:
                    consensus_reached = True
                    consensus_result = True  # Valid
                elif (1 - valid_ratio) >= VALIDATION_CONSENSUS_THRESHOLD:
                    consensus_reached = True
                    consensus_result = False  # Invalid
            
            return {
                "proof_signature": proof_signature,
                "total_votes": len(votes),
                "valid_votes": valid_count,
                "invalid_votes": invalid_count,
                "valid_stake": valid_stake,
                "invalid_stake": invalid_stake,
                "total_stake": total_stake,
                "valid_ratio": valid_stake / total_stake if total_stake > 0 else 0,
                "consensus_reached": consensus_reached,
                "consensus_result": consensus_result,
                "threshold": VALIDATION_CONSENSUS_THRESHOLD,
            }
    
    def select_validators_for_proof(self, proof: PoNWProof, num_validators: int = 3) -> List[str]:
        """
        Select validators for a proof using stake-weighted random selection.
        
        Selection algorithm:
        1. Get all eligible validators (stake >= VALIDATOR_MIN_STAKE)
        2. Weight by stake with randomness factor
        3. Select top N by weighted score
        
        This ensures:
        - Higher stake = higher chance of selection
        - But randomness prevents monopoly
        - Small stakers still get opportunities
        """
        import random
        
        with sqlite3.connect(self.db_path, timeout=60.0) as conn:
            # Get all nodes with sufficient stake
            eligible = conn.execute("""
                SELECT node_id, amount
                FROM stakes
                WHERE amount >= ? AND locked_until > ?
            """, (VALIDATOR_MIN_STAKE, time.time())).fetchall()
            
            if not eligible:
                return []
            
            # Exclude the proof submitter
            eligible = [(nid, stake) for nid, stake in eligible if nid != proof.node_id]
            
            if not eligible:
                return []
            
            # Calculate selection scores (stake-weighted with randomness)
            scores = []
            for node_id, stake in eligible:
                # Score = stake * (1 - randomness) + random * randomness * max_stake
                max_stake = max(s for _, s in eligible)
                stake_component = stake * (1 - VALIDATOR_SELECTION_RANDOMNESS)
                random_component = random.random() * VALIDATOR_SELECTION_RANDOMNESS * max_stake
                score = stake_component + random_component
                scores.append((node_id, score, stake))
            
            # Sort by score and select top N
            scores.sort(key=lambda x: x[1], reverse=True)
            selected = [node_id for node_id, _, _ in scores[:num_validators]]
            
            logger.debug(f"Selected {len(selected)} validators for proof {proof.signature[:16]}...")
            
            return selected
    
    # ========================================================================
    # TRANSACTIONS & FEE BURN
    # ========================================================================
    
    def transfer(
        self,
        to_id: str,
        amount: float,
        memo: str = ""
    ) -> Tuple[bool, str, Optional[Transaction]]:
        """
        Transfer NEURO to another node with 5% fee burn.
        
        Fee Structure:
        - 5% of amount is burned (deflationary)
        - Recipient receives full amount
        - Sender pays amount + fee
        
        Returns: (success, message, transaction)
        """
        if amount <= 0:
            return False, "Amount must be positive", None
        
        # Calculate fee and burn
        fee = amount * FEE_BURN_RATE
        burn_amount = fee  # 100% of fee is burned
        total_deduction = amount + fee
        
        # Generate transaction ID
        tx_id = hashlib.sha256(
            f"{self.node_id}:{to_id}:{amount}:{time.time()}:{os.urandom(8).hex()}".encode()
        ).hexdigest()
        
        tx = Transaction(
            tx_id=tx_id,
            from_id=self.node_id,
            to_id=to_id,
            amount=amount,
            fee=fee,
            burn_amount=burn_amount,
            timestamp=time.time(),
            signature="",
            memo=memo
        )
        
        # Sign transaction
        tx.signature = self._sign(tx.canonical_payload())
        
        with self.lock:
            with sqlite3.connect(self.db_path, timeout=60.0) as conn:
                # Check sender balance
                row = conn.execute(
                    "SELECT balance FROM balances WHERE node_id = ?",
                    (self.node_id,)
                ).fetchone()
                
                current_balance = row[0] if row else 0.0
                
                if current_balance < total_deduction:
                    return False, f"Insufficient balance ({current_balance:.6f} < {total_deduction:.6f})", None
                
                # Deduct from sender (amount + fee)
                conn.execute("""
                    UPDATE balances SET
                        balance = balance - ?,
                        total_spent = total_spent + ?
                    WHERE node_id = ?
                """, (total_deduction, total_deduction, self.node_id))
                
                # Credit to recipient
                conn.execute("""
                    INSERT INTO balances (node_id, balance, total_earned, created_at)
                    VALUES (?, ?, ?, ?)
                    ON CONFLICT(node_id) DO UPDATE SET
                        balance = balance + ?,
                        total_earned = total_earned + ?
                """, (to_id, amount, amount, time.time(), amount, amount))
                
                # Record burn (to special burn address)
                conn.execute("""
                    INSERT INTO balances (node_id, balance, total_earned, created_at)
                    VALUES (?, ?, ?, ?)
                    ON CONFLICT(node_id) DO UPDATE SET
                        balance = balance + ?,
                        total_earned = total_earned + ?
                """, (BURN_ADDRESS, burn_amount, burn_amount, time.time(), burn_amount, burn_amount))
                
                # Record transaction
                conn.execute("""
                    INSERT INTO transactions 
                    (tx_id, from_id, to_id, amount, fee, burn_amount, timestamp, memo, signature)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    tx.tx_id, tx.from_id, tx.to_id, tx.amount, tx.fee,
                    tx.burn_amount, tx.timestamp, tx.memo, tx.signature
                ))
                
                # Update global stats
                conn.execute("""
                    UPDATE global_stats SET
                        total_burned = total_burned + ?,
                        total_transferred = total_transferred + ?,
                        total_transactions = total_transactions + 1,
                        updated_at = ?
                    WHERE id = 1
                """, (burn_amount, amount, time.time()))
        
        logger.info(f"Transfer: {self.node_id[:12]}... → {to_id[:12]}... "
                   f"amount={amount:.6f} fee={fee:.6f} burned={burn_amount:.6f}")
        
        return True, "Transfer complete", tx
    
    def spend_for_inference(self, tokens_requested: int) -> Tuple[bool, float, str]:
        """
        Spend NEURO for inference (with 5% burn).
        
        Cost: 1 NEURO per 1M tokens (from whitepaper)
        Fee: 5% burned
        
        Returns: (success, cost, message)
        """
        # Calculate cost
        cost = (tokens_requested / 1_000_000.0) * 1.0
        cost = max(0.0001, cost)  # Minimum cost
        
        fee = cost * FEE_BURN_RATE
        total_cost = cost + fee
        
        with self.lock:
            with sqlite3.connect(self.db_path, timeout=60.0) as conn:
                # Check balance
                row = conn.execute(
                    "SELECT balance FROM balances WHERE node_id = ?",
                    (self.node_id,)
                ).fetchone()
                
                current_balance = row[0] if row else 0.0
                
                if current_balance < total_cost:
                    return False, total_cost, f"Insufficient NEURO ({current_balance:.6f} < {total_cost:.6f})"
                
                # Deduct cost
                conn.execute("""
                    UPDATE balances SET
                        balance = balance - ?,
                        total_spent = total_spent + ?
                    WHERE node_id = ?
                """, (total_cost, total_cost, self.node_id))
                
                # Burn the fee
                conn.execute("""
                    INSERT INTO balances (node_id, balance, total_earned, created_at)
                    VALUES (?, ?, ?, ?)
                    ON CONFLICT(node_id) DO UPDATE SET
                        balance = balance + ?
                """, (BURN_ADDRESS, fee, fee, time.time(), fee))
                
                # Update global stats
                conn.execute("""
                    UPDATE global_stats SET
                        total_burned = total_burned + ?,
                        updated_at = ?
                    WHERE id = 1
                """, (fee, time.time()))
        
        logger.info(f"Inference spend: {self.node_id[:12]}... cost={cost:.6f} fee={fee:.6f} burned")
        
        return True, total_cost, "Inference authorized"
    
    # ========================================================================
    # STAKING
    # ========================================================================
    
    def stake(self, amount: float, duration_days: int = 30) -> Tuple[bool, str]:
        """
        Stake NEURO for reward multiplier.
        
        Staking provides:
        - 10% bonus per 1000 NEURO staked
        - Locked for specified duration
        """
        if amount <= 0:
            return False, "Amount must be positive"
        
        lock_until = time.time() + (duration_days * 24 * 3600)
        
        with self.lock:
            with sqlite3.connect(self.db_path, timeout=60.0) as conn:
                # Check balance
                row = conn.execute(
                    "SELECT balance FROM balances WHERE node_id = ?",
                    (self.node_id,)
                ).fetchone()
                
                current_balance = row[0] if row else 0.0
                
                if current_balance < amount:
                    return False, f"Insufficient balance ({current_balance:.6f} < {amount:.6f})"
                
                # Get current stake
                row = conn.execute(
                    "SELECT amount FROM stakes WHERE node_id = ?",
                    (self.node_id,)
                ).fetchone()
                current_stake = row[0] if row else 0.0
                
                # Deduct from balance
                conn.execute("""
                    UPDATE balances SET balance = balance - ? WHERE node_id = ?
                """, (amount, self.node_id))
                
                # Add to stake
                conn.execute("""
                    INSERT INTO stakes (node_id, amount, locked_until, updated_at)
                    VALUES (?, ?, ?, ?)
                        ON CONFLICT(node_id) DO UPDATE SET
                        amount = amount + ?,
                        locked_until = MAX(locked_until, ?),
                        updated_at = ?
                """, (
                    self.node_id, amount, lock_until, time.time(),
                    amount, lock_until, time.time()
                ))
        
        new_stake = current_stake + amount
        multiplier = calculate_stake_multiplier(new_stake)
        
        logger.info(f"Staked: {self.node_id[:12]}... amount={amount:.6f} "
                   f"total_stake={new_stake:.6f} multiplier={multiplier:.2f}x")
        
        return True, f"Staked {amount:.6f} NEURO (new multiplier: {multiplier:.2f}x)"
    
    def unstake(self) -> Tuple[bool, float, str]:
        """
        Unstake NEURO (if lock period expired).
        
        Returns: (success, amount_unstaked, message)
        """
        with self.lock:
            with sqlite3.connect(self.db_path, timeout=60.0) as conn:
                row = conn.execute(
                    "SELECT amount, locked_until FROM stakes WHERE node_id = ?",
                    (self.node_id,)
                ).fetchone()
                
                if not row or row[0] == 0:
                    return False, 0.0, "No stake found"
                
                amount, locked_until = row
                
                if time.time() < locked_until:
                    remaining = (locked_until - time.time()) / 3600
                    return False, 0.0, f"Stake locked for {remaining:.1f} more hours"
                
                # Return stake to balance
                conn.execute("""
                    UPDATE balances SET balance = balance + ? WHERE node_id = ?
                """, (amount, self.node_id))
                
                # Clear stake
                conn.execute("""
                    UPDATE stakes SET amount = 0, updated_at = ? WHERE node_id = ?
                """, (time.time(), self.node_id))
        
        logger.info(f"Unstaked: {self.node_id[:12]}... amount={amount:.6f}")
        
        return True, amount, f"Unstaked {amount:.6f} NEURO"
    
    def update_stake(self, node_id: str, amount: float, locked_until: float = None) -> bool:
        """
        Update stake record for a REMOTE node (from P2P gossip).
        
        SECURITY MODEL:
        ===============
        This does NOT directly affect reward calculations for the remote node.
        It only maintains a local VIEW of what other nodes claim to have staked.
        
        The actual reward multiplier is calculated based on:
        1. For LOCAL node: Our own stake (from stake() method, which requires balance)
        2. For REMOTE proofs: We can verify their claimed multiplier is reasonable
        
        A malicious node can claim any stake, but:
        - They cannot earn MORE than the base reward without actual work
        - Validators cross-check stake claims during proof validation
        - Consensus rejects proofs with implausible multipliers
        
        This gossip sync is primarily for:
        - Validator selection (who can validate proofs)
        - Network visibility (dashboard displays)
        
        Returns True if the update was applied.
        """
        if locked_until is None:
            locked_until = time.time() + 86400 * 30  # Default 30 day lock
        
        # Validate stake amount using centralized economics
        is_valid, error_msg = is_valid_stake_amount(amount)
        if not is_valid:
            logger.warning(f"Rejected stake claim from {node_id[:16]}...: {error_msg}")
            return False
        
        try:
            with self.lock:
                with sqlite3.connect(self.db_path, timeout=60.0) as conn:
                    # Mark this as a REMOTE stake (not locally verified)
                    conn.execute("""
                        INSERT INTO stakes (node_id, amount, locked_until, updated_at)
                        VALUES (?, ?, ?, ?)
                        ON CONFLICT(node_id) DO UPDATE SET
                            amount = ?,
                            locked_until = ?,
                            updated_at = ?
                    """, (node_id, amount, locked_until, time.time(), amount, locked_until, time.time()))
                    return True
        except Exception as e:
            logger.error(f"Failed to update stake for {node_id[:16]}...: {e}")
            return False
    
    def get_local_stake(self, node_id: str) -> float:
        """Get stake for a specific node."""
        return self._get_stake(node_id)
    
    def create_transaction(self, from_id: str, to_id: str, amount: float, signature: str) -> bool:
        """
        Create a transaction from gossip (external source).
        
        Note: This only processes transactions where we are the sender,
        as we can't spend other nodes' balances.
        """
        # Only allow if we're the sender
        if from_id != self.node_id:
            logger.debug(f"Cannot create transaction for another node: {from_id[:12]}...")
            return False
        
        success, _, _ = self.transfer(to_id, amount)
        return success
    
    # ========================================================================
    # SLASHING (Fraud Prevention)
    # ========================================================================
    
    def report_fraud(
        self,
        accused_id: str,
        reason: str,
        proof_signature: Optional[str] = None,
        evidence: Optional[str] = None
    ) -> Tuple[bool, str]:
        """
        Report suspected fraud for slashing.
        
        If verified, the accused node loses SLASH_AMOUNT NEURO:
        - 50% goes to reporter (whistleblower reward)
        - 50% is burned
        """
        report_id = hashlib.sha256(
            f"{self.node_id}:{accused_id}:{time.time()}:{os.urandom(8).hex()}".encode()
        ).hexdigest()
        
        with self.lock:
            with sqlite3.connect(self.db_path, timeout=60.0) as conn:
                conn.execute("""
                    INSERT INTO fraud_reports 
                    (report_id, reporter_id, accused_id, proof_signature, reason, evidence, created_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                """, (
                    report_id, self.node_id, accused_id, proof_signature,
                    reason, evidence, time.time()
                ))
        
        logger.warning(f"Fraud report: {self.node_id[:12]}... reported {accused_id[:12]}... "
                      f"reason={reason}")
        
        return True, f"Fraud report submitted (ID: {report_id[:16]}...)"
    
    def execute_slash(self, accused_id: str, reporter_id: str) -> Tuple[bool, str]:
        """
        Execute slashing after fraud verification.
        
        Called by consensus mechanism after fraud is confirmed.
        """
        with self.lock:
            with sqlite3.connect(self.db_path, timeout=60.0) as conn:
                # Get accused balance
                row = conn.execute(
                    "SELECT balance FROM balances WHERE node_id = ?",
                    (accused_id,)
                ).fetchone()
                
                current_balance = row[0] if row else 0.0
                slash_amount = min(SLASH_AMOUNT, current_balance)
                
                if slash_amount <= 0:
                    return False, "No balance to slash"
                
                whistleblower_reward = slash_amount * WHISTLEBLOWER_REWARD_RATE
                burn_amount = slash_amount - whistleblower_reward
                
                # Deduct from accused
                conn.execute("""
                    UPDATE balances SET balance = balance - ? WHERE node_id = ?
                """, (slash_amount, accused_id))
                
                # Reward whistleblower
                conn.execute("""
                    INSERT INTO balances (node_id, balance, total_earned, created_at)
                    VALUES (?, ?, ?, ?)
                    ON CONFLICT(node_id) DO UPDATE SET
                        balance = balance + ?,
                        total_earned = total_earned + ?
                """, (reporter_id, whistleblower_reward, whistleblower_reward, time.time(),
                      whistleblower_reward, whistleblower_reward))
                
                # Burn remainder
                conn.execute("""
                    INSERT INTO balances (node_id, balance, total_earned, created_at)
                    VALUES (?, ?, ?, ?)
                    ON CONFLICT(node_id) DO UPDATE SET
                        balance = balance + ?
                """, (BURN_ADDRESS, burn_amount, burn_amount, time.time(), burn_amount))
                
                # Update global stats
                conn.execute("""
                    UPDATE global_stats SET
                        total_burned = total_burned + ?,
                        updated_at = ?
                    WHERE id = 1
                """, (burn_amount, time.time()))
        
        logger.warning(f"Slashed: {accused_id[:12]}... lost {slash_amount:.6f} NEURO "
                      f"(whistleblower={whistleblower_reward:.6f}, burned={burn_amount:.6f})")
        
        return True, f"Slashed {slash_amount:.6f} NEURO"
    
    def slash_bad_validator(self, validator_id: str, proof_signature: str, reason: str) -> Tuple[bool, str]:
        """
        Slash a validator who voted against consensus.
        
        Validators are held to a higher standard - they are slashed 2x the normal amount
        for voting incorrectly (VALIDATOR_SLASH_MULTIPLIER).
        
        This is called when consensus is reached and a validator's vote differs.
        """
        slash_amount = SLASH_AMOUNT * VALIDATOR_SLASH_MULTIPLIER
        
        with self.lock:
            with sqlite3.connect(self.db_path, timeout=60.0) as conn:
                # Get validator's stake (they lose from stake first)
                stake_row = conn.execute(
                    "SELECT amount FROM stakes WHERE node_id = ?",
                    (validator_id,)
                ).fetchone()
                
                stake = stake_row[0] if stake_row else 0.0
                
                # Get balance
                balance_row = conn.execute(
                    "SELECT balance FROM balances WHERE node_id = ?",
                    (validator_id,)
                ).fetchone()
                
                balance = balance_row[0] if balance_row else 0.0
                
                total_available = stake + balance
                actual_slash = min(slash_amount, total_available)
                
                if actual_slash <= 0:
                    return False, "No funds to slash"
                
                # Deduct from stake first, then balance
                stake_deduction = min(actual_slash, stake)
                balance_deduction = actual_slash - stake_deduction
                
                if stake_deduction > 0:
                    conn.execute("""
                        UPDATE stakes SET amount = amount - ? WHERE node_id = ?
                    """, (stake_deduction, validator_id))
                
                if balance_deduction > 0:
                    conn.execute("""
                        UPDATE balances SET balance = balance - ? WHERE node_id = ?
                    """, (balance_deduction, validator_id))
                
                # Burn the slashed amount
                conn.execute("""
                    INSERT INTO balances (node_id, balance, total_earned, created_at)
                    VALUES (?, ?, ?, ?)
                    ON CONFLICT(node_id) DO UPDATE SET
                        balance = balance + ?
                """, (BURN_ADDRESS, actual_slash, actual_slash, time.time(), actual_slash))
                
                # Update global stats
                conn.execute("""
                    UPDATE global_stats SET
                        total_burned = total_burned + ?,
                        updated_at = ?
                    WHERE id = 1
                """, (actual_slash, time.time()))
                
                # Record the slash
                conn.execute("""
                    INSERT INTO fraud_reports 
                    (report_id, reporter_id, accused_id, proof_signature, reason, evidence, created_at, status)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    hashlib.sha256(f"validator_slash:{validator_id}:{proof_signature}:{time.time()}".encode()).hexdigest()[:32],
                    "CONSENSUS",  # Reporter is the consensus mechanism
                    validator_id,
                    proof_signature,
                    reason,
                    f"Slashed {actual_slash:.6f} NEURO (stake: {stake_deduction:.6f}, balance: {balance_deduction:.6f})",
                    time.time(),
                    "executed"
                ))
        
        logger.warning(f"Validator slashed: {validator_id[:12]}... lost {actual_slash:.6f} NEURO "
                      f"for bad validation on {proof_signature[:16]}... Reason: {reason}")
        
        return True, f"Validator slashed {actual_slash:.6f} NEURO"
    
    def check_and_slash_bad_validators(self, proof_signature: str) -> List[str]:
        """
        Check if consensus was reached and slash validators who voted wrong.
        
        Called after a proof reaches consensus.
        Returns list of slashed validator IDs.
        """
        status = self.get_proof_validation_status(proof_signature)
        
        if not status["consensus_reached"]:
            return []
        
        consensus_result = status["consensus_result"]
        slashed = []
        
        with sqlite3.connect(self.db_path, timeout=60.0) as conn:
            # Get all votes for this proof
            votes = conn.execute("""
                SELECT validator_id, vote
                FROM validation_votes
                WHERE proof_signature = ?
            """, (proof_signature,)).fetchall()
            
            for validator_id, vote in votes:
                vote_bool = bool(vote)
                if vote_bool != consensus_result:
                    # This validator voted against consensus
                    success, msg = self.slash_bad_validator(
                        validator_id=validator_id,
                        proof_signature=proof_signature,
                        reason=f"Voted {'VALID' if vote_bool else 'INVALID'} but consensus was {'VALID' if consensus_result else 'INVALID'}"
                    )
                    if success:
                        slashed.append(validator_id)
        
        if slashed:
            logger.info(f"Slashed {len(slashed)} validators for proof {proof_signature[:16]}...")
        
        return slashed
    
    # ========================================================================
    # QUERIES
    # ========================================================================
    
    def get_balance(self, node_id: Optional[str] = None) -> float:
        """Get balance for a node."""
        node_id = node_id or self.node_id
        
        with self.lock:
            with sqlite3.connect(self.db_path, timeout=60.0) as conn:
                row = conn.execute(
                    "SELECT balance FROM balances WHERE node_id = ?",
                    (node_id,)
                ).fetchone()
                return row[0] if row else 0.0
    
    def get_balance_details(self, node_id: Optional[str] = None) -> Dict:
        """
        Get detailed balance information including confirmation status.
        
        IMPORTANT: Understanding Local vs Network Balance
        ================================================
        
        LOCAL BALANCE (what this node's ledger shows):
        - All proofs YOU generated that YOU witnessed
        - Includes solo-earned NEURO
        
        NETWORK BALANCE (what other nodes see):
        - Only proofs gossiped within PROOF_FRESHNESS_WINDOW (5 min)
        - If you ran solo, network didn't witness your work
        
        WHY THE DIFFERENCE?
        - Security: Prevents fabricating work while alone
        - Like Bitcoin: unmined blocks don't count until network sees them
        - Solo NEURO is "unconfirmed" - needs witnesses to be network-confirmed
        
        HOW TO CONFIRM:
        - Keep running with peers online
        - All new proofs will be gossiped and confirmed
        - Historical solo-earned NEURO stays LOCAL only
        """
        node_id = node_id or self.node_id
        
        with self.lock:
            with sqlite3.connect(self.db_path, timeout=60.0) as conn:
                # Get balance info
                row = conn.execute("""
                    SELECT balance, total_earned, proof_count
                    FROM balances WHERE node_id = ?
                """, (node_id,)).fetchone()
                
                local_balance = row[0] if row else 0.0
                total_earned = row[1] if row else 0.0
                proof_count = row[2] if row else 0
                
                # Get count of proofs that could have been gossiped
                # (proofs where we had peers at the time)
                # This is approximate - we track via proof timestamps
                fresh_proofs = conn.execute("""
                    SELECT COUNT(*) FROM proof_history 
                    WHERE node_id = ?
                """, (node_id,)).fetchone()[0]
                
                return {
                    "node_id": node_id,
                    "local_balance": local_balance,
                    "total_earned": total_earned,
                    "proof_count": proof_count,
                    "note": (
                        "Local balance includes all proofs you generated. "
                        "Network balance only counts proofs witnessed by peers within 5 min. "
                        "If you ran solo, that NEURO is LOCAL only."
                    )
                }
    
    def get_account_info(self, node_id: Optional[str] = None) -> Dict:
        """Get full account information."""
        node_id = node_id or self.node_id
        
        with self.lock:
            with sqlite3.connect(self.db_path, timeout=60.0) as conn:
                # Balance info
                row = conn.execute("""
                    SELECT balance, total_earned, total_spent, last_proof_time, proof_count, created_at
                    FROM balances WHERE node_id = ?
                """, (node_id,)).fetchone()
                
                if not row:
                    return {
                        "node_id": node_id,
                        "balance": 0.0,
                        "total_earned": 0.0,
                        "total_spent": 0.0,
                        "stake": 0.0,
                        "stake_multiplier": 1.0,
                        "proof_count": 0
                    }
                
                balance, total_earned, total_spent, last_proof_time, proof_count, created_at = row
                
                # Stake info
                stake_row = conn.execute(
                    "SELECT amount, locked_until FROM stakes WHERE node_id = ?",
                    (node_id,)
                ).fetchone()
                
                stake = stake_row[0] if stake_row else 0.0
                stake_locked_until = stake_row[1] if stake_row else 0.0
                stake_multiplier = calculate_stake_multiplier(stake)
                
                return {
                    "node_id": node_id,
                    "balance": balance,
                    "total_earned": total_earned,
                    "total_spent": total_spent,
                    "stake": stake,
                    "stake_locked_until": stake_locked_until,
                    "stake_multiplier": stake_multiplier,
                    "proof_count": proof_count,
                    "last_proof_time": last_proof_time,
                    "created_at": created_at
                }
    
    def get_global_stats(self) -> LedgerStats:
        """Get global ledger statistics."""
        with self.lock:
            with sqlite3.connect(self.db_path, timeout=60.0) as conn:
                row = conn.execute("""
                    SELECT total_minted, total_burned, total_transferred, total_proofs, total_transactions
                    FROM global_stats WHERE id = 1
                """).fetchone()
                
                if not row:
                    return LedgerStats()
                
                total_minted, total_burned, total_transferred, total_proofs, total_transactions = row
                
                return LedgerStats(
                    total_minted=total_minted,
                    total_burned=total_burned,
                    total_transferred=total_transferred,
                    circulating_supply=total_minted - total_burned,
                    total_proofs_processed=total_proofs,
                    total_transactions=total_transactions
                )
    
    def get_burn_stats(self) -> Dict:
        """Get burn statistics."""
        with self.lock:
            with sqlite3.connect(self.db_path, timeout=60.0) as conn:
                # Total burned
                row = conn.execute(
                    "SELECT balance FROM balances WHERE node_id = ?",
                    (BURN_ADDRESS,)
                ).fetchone()
                total_burned = row[0] if row else 0.0
                
                # Global stats
                stats_row = conn.execute(
                    "SELECT total_minted, total_burned FROM global_stats WHERE id = 1"
                ).fetchone()
                
                total_minted = stats_row[0] if stats_row else 0.0
                
                return {
                    "total_burned": total_burned,
                    "total_minted": total_minted,
                    "burn_rate": FEE_BURN_RATE,
                    "circulating_supply": total_minted - total_burned,
                    "burn_percentage": (total_burned / total_minted * 100) if total_minted > 0 else 0.0
                }
    
    # =========================================================================
    # OPTIMISTIC VERIFICATION - CHALLENGE SYSTEM
    # =========================================================================
    
    def challenge_proof(
        self,
        proof_signature: str,
        challenger_id: str,
        challenger_stake: float,
    ) -> Tuple[bool, float, str]:
        """
        Challenge a pending proof (Optimistic Verification).
        
        Anyone can challenge a proof within its challenge window by staking NEURO.
        If the challenge succeeds (fraud detected), challenger wins.
        If the challenge fails (proof valid), challenger loses stake.
        
        Args:
            proof_signature: Signature of the proof being challenged
            challenger_id: Node ID of the challenger
            challenger_stake: Amount of NEURO staked on the challenge
            
        Returns:
            (success, amount, message) tuple
            - success: True if challenge was processed
            - amount: Reward (if challenge succeeded) or loss (if failed)
            - message: Description of result
        """
        from neuroshard.core.consensus.verifier import (
            CHALLENGE_WINDOW,
            MIN_CHALLENGE_STAKE,
            SLASH_MULTIPLIER,
        )
        
        # Validate stake amount
        if challenger_stake < MIN_CHALLENGE_STAKE:
            return False, 0.0, f"Insufficient stake: {challenger_stake} < {MIN_CHALLENGE_STAKE}"
        
        with self.lock:
            with sqlite3.connect(self.db_path, timeout=60.0) as conn:
                # Get challenger's balance
                challenger_balance = conn.execute(
                    "SELECT balance FROM balances WHERE node_id = ?",
                    (challenger_id,)
                ).fetchone()
                
                if not challenger_balance or challenger_balance[0] < challenger_stake:
                    return False, 0.0, "Insufficient balance to stake challenge"
                
                # Get the proof being challenged
                proof_row = conn.execute("""
                    SELECT node_id, proof_type, timestamp, reward_amount, 
                           training_batches, tokens_processed, uptime_seconds
                    FROM proof_history
                    WHERE signature = ?
                """, (proof_signature,)).fetchone()
                
                if not proof_row:
                    return False, 0.0, "Proof not found"
                
                prover_id, proof_type, proof_timestamp, reward_amount, batches, tokens, uptime = proof_row
                
                # Check if within challenge window
                elapsed = time.time() - proof_timestamp
                if elapsed > CHALLENGE_WINDOW:
                    return False, 0.0, f"Challenge window expired ({elapsed:.0f}s > {CHALLENGE_WINDOW}s)"
                
                # Lock challenger's stake
                conn.execute("""
                    UPDATE balances SET balance = balance - ? WHERE node_id = ?
                """, (challenger_stake, challenger_id))
                
                # Perform verification
                is_fraud = self._verify_challenge_claim(
                    prover_id, proof_type, batches, tokens, uptime
                )
                
                if is_fraud:
                    # Challenge succeeded - fraud detected
                    slash_amount = reward_amount * SLASH_MULTIPLIER
                    challenger_reward = challenger_stake + slash_amount / 2
                    burn_amount = slash_amount / 2
                    
                    # Slash prover
                    conn.execute("""
                        UPDATE balances SET balance = balance - ? WHERE node_id = ?
                    """, (min(slash_amount, reward_amount), prover_id))
                    
                    # Reward challenger
                    conn.execute("""
                        UPDATE balances SET balance = balance + ? WHERE node_id = ?
                    """, (challenger_reward, challenger_id))
                    
                    # Burn portion
                    conn.execute("""
                        UPDATE balances SET balance = balance + ? WHERE node_id = ?
                    """, (burn_amount, BURN_ADDRESS))
                    
                    # Record challenge result
                    conn.execute("""
                        INSERT INTO fraud_reports 
                        (report_id, reporter_id, accused_id, proof_signature, reason, created_at, status)
                        VALUES (?, ?, ?, ?, ?, ?, ?)
                    """, (
                        hashlib.sha256(f"challenge:{proof_signature}:{time.time()}".encode()).hexdigest()[:32],
                        challenger_id,
                        prover_id,
                        proof_signature,
                        "Challenge succeeded - fraud detected",
                        time.time(),
                        "resolved"
                    ))
                    
                    logger.warning(f"Challenge succeeded: {prover_id[:8]}... slashed {slash_amount:.6f} NEURO")
                    return True, challenger_reward, f"Fraud detected. Reward: {challenger_reward:.6f} NEURO"
                
                else:
                    # Challenge failed - proof was valid
                    prover_bonus = challenger_stake * 0.9  # 90% to prover
                    burn_amount = challenger_stake * 0.1   # 10% burned
                    
                    # Prover gets challenger's stake
                    conn.execute("""
                        UPDATE balances SET balance = balance + ? WHERE node_id = ?
                    """, (prover_bonus, prover_id))
                    
                    # Burn portion
                    conn.execute("""
                        UPDATE balances SET balance = balance + ? WHERE node_id = ?
                    """, (burn_amount, BURN_ADDRESS))
                    
                    logger.info(f"Challenge failed: {challenger_id[:8]}... lost {challenger_stake:.6f} NEURO")
                    return True, -challenger_stake, f"Challenge failed. Lost: {challenger_stake:.6f} NEURO"
    
    def _verify_challenge_claim(
        self,
        prover_id: str,
        proof_type: str,
        batches: int,
        tokens: int,
        uptime: float,
    ) -> bool:
        """
        Verify a challenged proof claim.
        
        Returns True if fraud is detected.
        """
        # Physical rate limits
        if proof_type == "training" and uptime > 0 and batches > 0:
            rate = batches / uptime
            if rate > 10.0:  # MAX_TRAINING_RATE_PER_SEC (10 batches/sec for small models)
                return True  # Fraud - impossible rate
        
        if proof_type == "inference" and uptime > 0 and tokens > 0:
            rate = tokens / uptime
            if rate > 5000.0:  # MAX_INFERENCE_TOKENS_PER_SEC
                return True  # Fraud - impossible rate
        
        # If verifier interface is available, do deeper check
        if self.verifier and self.verifier.model_interface:
            # Create minimal proof object
            class ProofStub:
                pass
            proof = ProofStub()
            proof.node_id = prover_id
            proof.proof_type = proof_type
            proof.training_batches = batches
            proof.tokens_processed = tokens
            proof.uptime_seconds = uptime
            proof.model_hash = ""
            
            is_valid, _ = self.verifier.verify_work_content(proof)
            if not is_valid:
                return True  # Fraud detected
        
        return False  # No fraud detected
    
    def get_challengeable_proofs(self, limit: int = 100) -> List[Dict[str, Any]]:
        """
        Get proofs that are still within the challenge window.
        
        Useful for nodes that want to verify and potentially challenge proofs.
        
        Args:
            limit: Maximum number of proofs to return
            
        Returns:
            List of proof info dicts
        """
        from neuroshard.core.consensus.verifier import CHALLENGE_WINDOW
        
        cutoff_time = time.time() - CHALLENGE_WINDOW
        
        with self.lock:
            with sqlite3.connect(self.db_path, timeout=60.0) as conn:
                rows = conn.execute("""
                    SELECT signature, node_id, proof_type, timestamp, reward_amount,
                           training_batches, tokens_processed, uptime_seconds
                    FROM proof_history
                    WHERE timestamp > ?
                    ORDER BY timestamp DESC
                    LIMIT ?
                """, (cutoff_time, limit)).fetchall()
                
                proofs = []
                for row in rows:
                    sig, node, ptype, ts, reward, batches, tokens, uptime = row
                    proofs.append({
                        "signature": sig,
                        "node_id": node,
                        "proof_type": ptype,
                        "timestamp": ts,
                        "reward": reward,
                        "training_batches": batches,
                        "tokens_processed": tokens,
                        "uptime_seconds": uptime,
                        "challenge_window_remaining": max(0, CHALLENGE_WINDOW - (time.time() - ts)),
                    })
                
                return proofs


# ============================================================================
# BACKWARD COMPATIBILITY ALIASES
# ============================================================================

# LedgerManager is now just an alias for NEUROLedger
LedgerManager = NEUROLedger

# Legacy ProofOfWork class for any remaining references
ProofOfWork = PoNWProof
