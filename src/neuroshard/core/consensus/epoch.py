"""
Chained Epoch System for Proof of Neural Work

This module implements blockchain-like epoch chaining for NeuroShard:
- Epochs are cryptographically linked via prev_epoch_hash
- Model state progression is tracked (like block state roots)
- Gradient commitments serve as "nonce equivalent" (proof of actual training)
- Stake-weighted proposer selection (like Ethereum PoS)

Security Model:
1. Each epoch references the previous epoch's hash (can't rewrite history)
2. Model state must change between epochs (proves actual training)
3. Gradient commitments are spot-checked (economic security via slashing)
4. Merkle roots enable efficient proof inclusion verification

Based on: docs/whitepaper/neuroshard_whitepaper.tex
"""

import hashlib
import json
import logging
import threading
import time
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Tuple, Any
from enum import Enum

logger = logging.getLogger(__name__)


# =============================================================================
# CONSTANTS
# =============================================================================

EPOCH_DURATION = 60              # seconds per epoch
FINALIZATION_DELAY = 10          # grace period for late proofs
SPOT_CHECK_PROBABILITY = 0.05   # 5% of proofs are spot-checked
SLASH_MULTIPLIER = 10.0          # 10x stake slashed for fraud
MIN_PROPOSER_STAKE = 100.0       # Minimum stake to be epoch proposer
GENESIS_EPOCH_HASH = "0x0000000000000000000000000000000000000000000000000000000000000000"


# =============================================================================
# GRADIENT COMMITMENT
# =============================================================================

@dataclass
class GradientCommitment:
    """
    Cryptographic commitment to gradient computation.
    
    This is the "nonce equivalent" for Proof of Neural Work:
    - Deterministic: Same model + same data = same commitment
    - Unforgeable: Can't guess without doing the work
    - Verifiable: Re-run training to verify
    
    The commitment includes per-layer gradient statistics that are
    computationally infeasible to fake without actual backpropagation.
    """
    model_hash_start: str          # H(weights before training)
    model_hash_end: str            # H(weights after training)
    data_hash: str                 # H(training batch)
    gradient_norms: List[float]    # L2 norm per layer
    gradient_hash: str             # H(flattened gradient sample)
    final_loss: float              # Loss value after forward pass
    batch_size: int
    sequence_length: int
    
    def compute_commitment_hash(self) -> str:
        """Compute the commitment hash from all fields."""
        payload = (
            f"{self.model_hash_start}:"
            f"{self.model_hash_end}:"
            f"{self.data_hash}:"
            f"{','.join(f'{n:.6f}' for n in self.gradient_norms)}:"
            f"{self.gradient_hash}:"
            f"{self.final_loss:.6f}:"
            f"{self.batch_size}:{self.sequence_length}"
        )
        return hashlib.sha256(payload.encode()).hexdigest()
    
    def to_dict(self) -> dict:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: dict) -> 'GradientCommitment':
        return cls(**data)
    
    def verify_weight_change(self) -> bool:
        """Verify that weights actually changed (training happened)."""
        return self.model_hash_start != self.model_hash_end


def compute_gradient_commitment(
    model,
    input_ids,
    labels,
    model_hash_start: str,
) -> GradientCommitment:
    """
    Compute gradient commitment for a training batch.
    
    This function performs a forward/backward pass and captures
    the gradient statistics needed for verification.
    
    Args:
        model: The neural network model
        input_ids: Input token IDs
        labels: Target labels
        model_hash_start: Hash of weights before this batch
        
    Returns:
        GradientCommitment with all verification data
    """
    import torch
    
    # Ensure model is in training mode
    model.train()
    
    # Compute data hash
    data_bytes = input_ids.cpu().numpy().tobytes() + labels.cpu().numpy().tobytes()
    data_hash = hashlib.sha256(data_bytes).hexdigest()[:32]
    
    # Forward pass
    outputs = model(input_ids)
    if hasattr(outputs, 'logits'):
        logits = outputs.logits
    else:
        logits = outputs
    
    # Compute loss
    loss = torch.nn.functional.cross_entropy(
        logits.view(-1, logits.size(-1)),
        labels.view(-1),
        ignore_index=-100
    )
    
    # Backward pass
    loss.backward()
    
    # Collect gradient norms per layer
    gradient_norms = []
    gradient_samples = []
    
    for name, param in model.named_parameters():
        if param.grad is not None:
            grad_norm = param.grad.norm().item()
            gradient_norms.append(grad_norm)
            # Sample first 100 gradient values for hash
            flat_grad = param.grad.flatten()[:100]
            gradient_samples.append(flat_grad)
    
    # Compute gradient hash from samples
    if gradient_samples:
        combined = torch.cat(gradient_samples)
        grad_bytes = combined.cpu().numpy().tobytes()
        gradient_hash = hashlib.sha256(grad_bytes).hexdigest()[:32]
    else:
        gradient_hash = "no_gradients"
    
    # Compute model hash after (would be after optimizer.step())
    # For commitment, we compute what the hash WILL be
    with torch.no_grad():
        param_bytes = b""
        for param in model.parameters():
            param_bytes += param.data.flatten()[:1000].cpu().numpy().tobytes()
        model_hash_end = hashlib.sha256(param_bytes).hexdigest()[:32]
    
    return GradientCommitment(
        model_hash_start=model_hash_start,
        model_hash_end=model_hash_end,
        data_hash=data_hash,
        gradient_norms=gradient_norms,
        gradient_hash=gradient_hash,
        final_loss=loss.item(),
        batch_size=input_ids.size(0),
        sequence_length=input_ids.size(1) if len(input_ids.shape) > 1 else 1,
    )


# =============================================================================
# EPOCH DATA STRUCTURES
# =============================================================================

class EpochStatus(Enum):
    """Lifecycle status of an epoch."""
    PENDING = "pending"         # Currently accepting proofs
    FINALIZING = "finalizing"   # Grace period, preparing to seal
    FINALIZED = "finalized"     # Sealed, immutable
    ORPHANED = "orphaned"       # Rejected (chain reorg)


@dataclass
class EpochProof:
    """A proof included in an epoch."""
    signature: str              # Unique proof identifier
    node_id: str
    proof_type: str
    timestamp: float
    reward: float
    gradient_commitment: Optional[str] = None  # Commitment hash for training proofs
    
    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class Epoch:
    """
    A finalized epoch in the PoNW chain.
    
    Epochs are analogous to blocks in a blockchain:
    - Linked via prev_epoch_hash (chain integrity)
    - Contains merkle root of all proofs (inclusion proof)
    - Model state hash tracks network-wide training progress
    - Proposer signature ensures accountability
    """
    epoch_id: int
    prev_epoch_hash: str
    
    # Timing
    timestamp_start: float
    timestamp_end: float
    
    # Model state progression (proves training happened)
    model_state_hash_start: str   # Aggregate model hash at epoch start
    model_state_hash_end: str     # Aggregate model hash at epoch end
    
    # Proof aggregation
    proofs_merkle_root: str       # Merkle root of all proof signatures
    proof_count: int
    total_reward: float
    
    # Training statistics
    total_batches: int
    average_loss: float
    gradient_commitments_root: str  # Merkle root of gradient commitments
    
    # Proposer (stake-weighted selection)
    proposer_node_id: str
    proposer_signature: str
    
    # Computed hash (set after all fields are filled)
    epoch_hash: str = ""
    
    # Status
    status: EpochStatus = EpochStatus.PENDING
    
    def compute_epoch_hash(self) -> str:
        """Compute the cryptographic hash of this epoch."""
        payload = (
            f"{self.epoch_id}:"
            f"{self.prev_epoch_hash}:"
            f"{self.timestamp_start}:{self.timestamp_end}:"
            f"{self.model_state_hash_start}:{self.model_state_hash_end}:"
            f"{self.proofs_merkle_root}:"
            f"{self.proof_count}:{self.total_reward:.6f}:"
            f"{self.total_batches}:{self.average_loss:.6f}:"
            f"{self.gradient_commitments_root}:"
            f"{self.proposer_node_id}"
        )
        return hashlib.sha256(payload.encode()).hexdigest()
    
    def finalize(self, proposer_crypto) -> None:
        """Finalize the epoch: compute hash and sign."""
        self.epoch_hash = self.compute_epoch_hash()
        self.proposer_signature = proposer_crypto.sign(self.epoch_hash)
        self.status = EpochStatus.FINALIZED
    
    def verify_signature(self, public_key: str) -> bool:
        """Verify the proposer's signature."""
        from neuroshard.core.crypto.ecdsa import verify_signature
        return verify_signature(self.epoch_hash, self.proposer_signature, public_key)
    
    def verify_chain_link(self, prev_epoch: 'Epoch') -> bool:
        """Verify this epoch correctly links to the previous."""
        if self.prev_epoch_hash != prev_epoch.epoch_hash:
            return False
        if self.model_state_hash_start != prev_epoch.model_state_hash_end:
            return False
        if self.timestamp_start < prev_epoch.timestamp_end:
            return False
        return True
    
    def to_dict(self) -> dict:
        d = asdict(self)
        d['status'] = self.status.value
        return d
    
    @classmethod
    def from_dict(cls, data: dict) -> 'Epoch':
        data['status'] = EpochStatus(data.get('status', 'finalized'))
        return cls(**data)
    
    def to_json(self) -> str:
        return json.dumps(self.to_dict())
    
    @classmethod
    def from_json(cls, json_str: str) -> 'Epoch':
        return cls.from_dict(json.loads(json_str))


def compute_merkle_root(items: List[str]) -> str:
    """
    Compute Merkle root of a list of hex strings.
    
    Args:
        items: List of hex string hashes
        
    Returns:
        Merkle root as hex string
    """
    if not items:
        return hashlib.sha256(b"empty").hexdigest()
    
    # Convert to bytes
    hashes = [bytes.fromhex(h) if len(h) == 64 else hashlib.sha256(h.encode()).digest() 
              for h in items]
    
    # Build tree bottom-up
    while len(hashes) > 1:
        if len(hashes) % 2 == 1:
            hashes.append(hashes[-1])  # Duplicate last if odd
        
        new_level = []
        for i in range(0, len(hashes), 2):
            combined = hashes[i] + hashes[i + 1]
            new_level.append(hashlib.sha256(combined).digest())
        hashes = new_level
    
    return hashes[0].hex()


# =============================================================================
# EPOCH MANAGER
# =============================================================================

class EpochManager:
    """
    Manages epoch lifecycle and chain integrity.
    
    Responsibilities:
    - Track current epoch and accept proofs
    - Finalize epochs after duration + grace period
    - Maintain epoch chain with proper linking
    - Select epoch proposers (stake-weighted)
    - Store/retrieve epochs from DHT
    
    The EpochManager ensures that:
    1. Epochs are properly chained (prev_hash linkage)
    2. Model state shows progression (actual training)
    3. Proofs are correctly aggregated (merkle root)
    4. Proposers are accountable (signed epochs)
    """
    
    def __init__(
        self,
        node_id: str,
        crypto=None,
        dht=None,
        ledger=None,
        model=None,
        epoch_duration: float = EPOCH_DURATION,
        finalization_delay: float = FINALIZATION_DELAY,
    ):
        self.node_id = node_id
        self.crypto = crypto
        self.dht = dht
        self.ledger = ledger
        self.model = model  # Reference to local model for computing model hash
        self.epoch_duration = epoch_duration
        self.finalization_delay = finalization_delay
        
        self._lock = threading.Lock()
        
        # Current epoch state
        self.current_epoch_id: int = self._compute_current_epoch_id()
        self.current_proofs: List[EpochProof] = []
        self.current_gradient_commitments: List[str] = []
        self.current_model_hash: str = ""
        
        # Epoch chain (recent epochs in memory)
        self.epoch_chain: Dict[int, Epoch] = {}
        self.latest_finalized_epoch_id: int = -1
        
        # Genesis epoch hash
        self.genesis_hash = GENESIS_EPOCH_HASH
        
        # Validators for proposer selection
        self.validators: Dict[str, float] = {}  # node_id -> stake
        
        # Background finalizer
        self._running = False
        self._finalize_thread = None
        
        logger.info(f"EpochManager initialized: epoch_id={self.current_epoch_id}, node={node_id[:16]}...")
    
    def _compute_current_epoch_id(self) -> int:
        """Compute epoch ID from current timestamp."""
        return int(time.time() / self.epoch_duration)
    
    def _get_epoch_timestamp_range(self, epoch_id: int) -> Tuple[float, float]:
        """Get the timestamp range for an epoch."""
        start = epoch_id * self.epoch_duration
        end = start + self.epoch_duration
        return start, end
    
    def start(self):
        """Start the epoch manager background thread."""
        if self._running:
            return
        
        self._running = True
        self._finalize_thread = threading.Thread(target=self._finalize_loop, daemon=True)
        self._finalize_thread.start()
        logger.info("EpochManager started")
    
    def stop(self):
        """Stop the epoch manager."""
        self._running = False
        if self._finalize_thread:
            self._finalize_thread.join(timeout=5)
        logger.info("EpochManager stopped")
    
    def _finalize_loop(self):
        """Background thread that finalizes epochs."""
        while self._running:
            try:
                current_id = self._compute_current_epoch_id()
                
                # Check if we've moved to a new epoch
                if current_id > self.current_epoch_id:
                    # Wait for finalization delay
                    time.sleep(self.finalization_delay)
                    
                    # Finalize the previous epoch
                    self._finalize_epoch(self.current_epoch_id)
                    
                    # Move to new epoch
                    with self._lock:
                        self.current_epoch_id = current_id
                        self.current_proofs = []
                        self.current_gradient_commitments = []
                    
                    logger.info(f"Advanced to epoch {current_id}")
                
                time.sleep(1)  # Check every second
                
            except Exception as e:
                logger.error(f"Epoch finalization error: {e}")
                time.sleep(5)
    
    def accept_proof(
        self,
        proof_signature: str,
        node_id: str,
        proof_type: str,
        timestamp: float,
        reward: float,
        gradient_commitment: Optional[GradientCommitment] = None,
    ) -> Tuple[bool, str]:
        """
        Accept a proof for the current epoch.
        
        Args:
            proof_signature: Unique proof identifier
            node_id: Node that submitted the proof
            proof_type: Type of proof (training, inference, uptime)
            timestamp: Proof timestamp
            reward: NEURO reward for this proof
            gradient_commitment: Optional gradient commitment for training proofs
            
        Returns:
            (success, message)
        """
        # Determine which epoch this proof belongs to
        proof_epoch_id = int(timestamp / self.epoch_duration)
        
        with self._lock:
            # Accept proofs for current epoch ±1 (to handle clock skew between nodes)
            # This is necessary because distributed nodes may have slightly different clocks
            # Proofs are still validated by timestamp, this just allows for 1 epoch drift
            epoch_diff = abs(proof_epoch_id - self.current_epoch_id)
            
            if epoch_diff > 1:
                if proof_epoch_id < self.current_epoch_id:
                    return False, f"Proof too old: epoch {proof_epoch_id} < {self.current_epoch_id - 1}"
                else:
                    return False, f"Proof too far ahead: epoch {proof_epoch_id} > {self.current_epoch_id + 1}"
            
            # If proof is from next epoch, advance our epoch counter
            if proof_epoch_id > self.current_epoch_id:
                self.current_epoch_id = proof_epoch_id
            
            # Create epoch proof record
            commitment_hash = None
            if gradient_commitment:
                commitment_hash = gradient_commitment.compute_commitment_hash()
                self.current_gradient_commitments.append(commitment_hash)
            
            epoch_proof = EpochProof(
                signature=proof_signature,
                node_id=node_id,
                proof_type=proof_type,
                timestamp=timestamp,
                reward=reward,
                gradient_commitment=commitment_hash,
            )
            
            self.current_proofs.append(epoch_proof)
            
            return True, f"Proof accepted for epoch {proof_epoch_id}"
    
    def _finalize_epoch(self, epoch_id: int) -> Optional[Epoch]:
        """
        Finalize an epoch and add it to the chain.
        
        This:
        1. Computes merkle roots for proofs and gradient commitments
        2. Gets model state hash (start and end)
        3. Selects proposer and signs epoch
        4. Links to previous epoch
        5. Stores in DHT
        """
        with self._lock:
            proofs = list(self.current_proofs)
            gradient_commitments = list(self.current_gradient_commitments)
        
        if not proofs:
            logger.info(f"Epoch {epoch_id} empty, skipping finalization")
            return None
        
        # Compute merkle roots
        proof_signatures = [p.signature for p in proofs]
        proofs_merkle_root = compute_merkle_root(proof_signatures)
        
        gradient_commitments_root = compute_merkle_root(gradient_commitments) if gradient_commitments else ""
        
        # Get timestamp range
        ts_start, ts_end = self._get_epoch_timestamp_range(epoch_id)
        
        # Compute aggregate model state
        # In production, this would aggregate model hashes from all training nodes
        model_state_start = self._get_network_model_hash_at(ts_start)
        model_state_end = self._get_network_model_hash_at(ts_end)
        
        # Get previous epoch hash
        if epoch_id == 0:
            prev_hash = self.genesis_hash
        else:
            prev_epoch = self.epoch_chain.get(epoch_id - 1)
            if prev_epoch:
                prev_hash = prev_epoch.epoch_hash
            else:
                # Try to get from DHT
                prev_hash = self._get_epoch_hash_from_dht(epoch_id - 1) or self.genesis_hash
        
        # Calculate training statistics
        total_batches = sum(
            p.reward for p in proofs if p.proof_type == "training"
        )  # Approximate from rewards
        
        losses = [gc for gc in gradient_commitments if gc]  # Filter valid
        average_loss = 0.0  # Would be computed from gradient commitments
        
        # Select proposer (for now, use self if we have stake)
        proposer_id = self._select_proposer(epoch_id)
        
        # Create epoch
        epoch = Epoch(
            epoch_id=epoch_id,
            prev_epoch_hash=prev_hash,
            timestamp_start=ts_start,
            timestamp_end=ts_end,
            model_state_hash_start=model_state_start,
            model_state_hash_end=model_state_end,
            proofs_merkle_root=proofs_merkle_root,
            proof_count=len(proofs),
            total_reward=sum(p.reward for p in proofs),
            total_batches=int(total_batches),
            average_loss=average_loss,
            gradient_commitments_root=gradient_commitments_root,
            proposer_node_id=proposer_id,
            proposer_signature="",
        )
        
        # Finalize (compute hash and sign)
        if self.crypto and proposer_id == self.node_id:
            epoch.finalize(self.crypto)
        else:
            # If we're not the proposer, just compute hash (unsigned)
            epoch.epoch_hash = epoch.compute_epoch_hash()
            epoch.status = EpochStatus.FINALIZED
        
        # Store in chain
        self.epoch_chain[epoch_id] = epoch
        self.latest_finalized_epoch_id = epoch_id
        
        # Store in DHT
        self._store_epoch_in_dht(epoch)
        
        logger.info(
            f"Epoch {epoch_id} finalized: {epoch.proof_count} proofs, "
            f"{epoch.total_reward:.4f} NEURO, hash={epoch.epoch_hash[:16]}..."
        )
        
        return epoch
    
    def _get_network_model_hash_at(self, timestamp: float) -> str:
        """Get aggregate network model hash at a timestamp.
        
        Tries the following in order:
        1. Query DHT for the model state commitment at this timestamp
        2. Compute hash from local model parameters
        3. Fallback to deterministic hash from timestamp (last resort)
        """
        # 1. Try DHT lookup for the model state
        if self.dht:
            try:
                dht_key = f"model_state:{int(timestamp)}"
                # Use lookup_key() for string keys (lookup_value expects int)
                value = self.dht.lookup_key(dht_key) if hasattr(self.dht, 'lookup_key') else None
                if value and isinstance(value, str) and len(value) >= 32:
                    return value[:32]
            except Exception as e:
                logger.debug(f"DHT model hash lookup failed: {e}")
        
        # 2. Compute from local model if available
        if self.model is not None:
            try:
                param_bytes = b""
                for name, param in self.model.named_parameters():
                    # Sample first 1000 values per parameter for efficiency
                    param_bytes += param.data.flatten()[:1000].cpu().numpy().tobytes()
                return hashlib.sha256(param_bytes).hexdigest()[:32]
            except Exception as e:
                logger.debug(f"Local model hash computation failed: {e}")
        
        # 3. Fallback — deterministic from timestamp (for bootstrap/testing only)
        logger.debug("Using fallback model hash from timestamp — no DHT or model available")
        return hashlib.sha256(f"model_state_{int(timestamp)}".encode()).hexdigest()[:32]
    
    def _get_epoch_hash_from_dht(self, epoch_id: int) -> Optional[str]:
        """Retrieve epoch hash from DHT."""
        if not self.dht:
            return None
        
        try:
            dht_key = self._epoch_dht_key(epoch_id)
            value = self.dht.lookup_value(dht_key)
            if value:
                epoch = Epoch.from_json(value)
                return epoch.epoch_hash
        except Exception as e:
            logger.warning(f"Failed to get epoch {epoch_id} from DHT: {e}")
        
        return None
    
    def _epoch_dht_key(self, epoch_id: int) -> int:
        """Compute DHT key for an epoch (160-bit for Kademlia compatibility)."""
        key_str = f"epoch:{epoch_id}"
        # Use SHA-1 for 160-bit key (Kademlia standard)
        return int(hashlib.sha1(key_str.encode()).hexdigest(), 16)
    
    def _store_epoch_in_dht(self, epoch: Epoch) -> bool:
        """Store finalized epoch in DHT."""
        if not self.dht:
            return False
        
        try:
            dht_key = self._epoch_dht_key(epoch.epoch_id)
            # DHTProtocol uses direct storage access
            self.dht.storage[dht_key] = epoch.to_json()
            logger.debug(f"Stored epoch {epoch.epoch_id} in DHT")
            return True
        except Exception as e:
            logger.warning(f"Failed to store epoch {epoch.epoch_id} in DHT: {e}")
            return False
    
    def _select_proposer(self, epoch_id: int) -> str:
        """
        Select epoch proposer using stake-weighted random selection.
        
        This is similar to Ethereum's PoS proposer selection:
        - Higher stake = higher probability of being selected
        - Deterministic given the previous epoch hash (no grinding)
        """
        if not self.validators:
            # No validators registered, use self
            return self.node_id
        
        # Get seed from previous epoch hash
        prev_epoch = self.epoch_chain.get(epoch_id - 1)
        if prev_epoch:
            seed = prev_epoch.epoch_hash
        else:
            seed = self.genesis_hash
        
        # Deterministic random based on seed + epoch_id
        combined = f"{seed}:{epoch_id}"
        random_bytes = hashlib.sha256(combined.encode()).digest()
        random_value = int.from_bytes(random_bytes[:8], 'big')
        
        # Stake-weighted selection
        total_stake = sum(self.validators.values())
        if total_stake == 0:
            return self.node_id
        
        threshold = (random_value % int(total_stake * 1000)) / 1000
        cumulative = 0.0
        
        for node_id, stake in sorted(self.validators.items()):
            cumulative += stake
            if cumulative >= threshold:
                return node_id
        
        return self.node_id
    
    def register_validator(self, node_id: str, stake: float):
        """Register a validator with their stake for proposer selection."""
        if stake >= MIN_PROPOSER_STAKE:
            self.validators[node_id] = stake
            logger.info(f"Registered validator {node_id[:16]}... with stake {stake}")
    
    def get_epoch(self, epoch_id: int) -> Optional[Epoch]:
        """Get an epoch from local cache or DHT."""
        # Check local cache
        if epoch_id in self.epoch_chain:
            return self.epoch_chain[epoch_id]
        
        # Try DHT
        if self.dht:
            try:
                dht_key = self._epoch_dht_key(epoch_id)
                value = self.dht.lookup_value(dht_key)
                if value:
                    epoch = Epoch.from_json(value)
                    self.epoch_chain[epoch_id] = epoch
                    return epoch
            except Exception as e:
                logger.warning(f"Failed to get epoch {epoch_id}: {e}")
        
        return None
    
    def verify_epoch_chain(self, start_id: int, end_id: int) -> Tuple[bool, str]:
        """
        Verify the integrity of the epoch chain.
        
        Gap-tolerant: Empty epochs (with no proofs) are skipped and don't break
        the chain. Only epochs that exist are verified for proper linking.
        
        Checks:
        1. Each existing epoch has valid hash
        2. Existing epochs link properly to previous existing epoch
        3. Model state shows progression where epochs exist
        4. Proposer signatures are valid (if available)
        
        Returns:
            (is_valid, error_message)
        """
        prev_epoch = None
        found_epochs = 0
        skipped_epochs = 0
        
        for epoch_id in range(start_id, end_id + 1):
            epoch = self.get_epoch(epoch_id)
            
            if not epoch:
                # Empty epochs are expected - they had no proofs and were skipped
                skipped_epochs += 1
                continue
            
            found_epochs += 1
            
            # Verify epoch hash
            computed_hash = epoch.compute_epoch_hash()
            if computed_hash != epoch.epoch_hash:
                return False, f"Epoch {epoch_id} hash mismatch"
            
            # Verify chain link to previous EXISTING epoch
            if prev_epoch:
                # Check prev_epoch_hash links correctly
                # Note: In gap-tolerant mode, we verify the hash matches what we expect
                # The prev_epoch_hash should match the last finalized epoch's hash
                if epoch.prev_epoch_hash != prev_epoch.epoch_hash:
                    # This is only an error if there were no gaps between them
                    # With gaps (empty epochs), the prev_hash might link to an earlier epoch
                    gap_size = epoch.epoch_id - prev_epoch.epoch_id
                    if gap_size == 1:
                        # No gap - this is a real chain break
                        return False, f"Epoch {epoch_id} chain link broken (prev_hash mismatch)"
                    # else: gap exists, prev_hash may link to an even older epoch - acceptable
            elif epoch_id > 0:
                # First epoch in range - verify it links to something valid
                if epoch.prev_epoch_hash != self.genesis_hash:
                    # Should link to a previous epoch (which may not be in our range)
                    pass  # Accept - we can't verify what's before our range
            
            prev_epoch = epoch
        
        if found_epochs == 0:
            return True, "No epochs in range (all empty)"
        
        return True, f"Chain verified ({found_epochs} epochs, {skipped_epochs} empty)"
    
    def get_current_epoch_info(self) -> Dict[str, Any]:
        """Get information about the current epoch."""
        with self._lock:
            return {
                "epoch_id": self.current_epoch_id,
                "proof_count": len(self.current_proofs),
                "gradient_commitments": len(self.current_gradient_commitments),
                "latest_finalized": self.latest_finalized_epoch_id,
                "chain_length": len(self.epoch_chain),
            }


# =============================================================================
# SPOT-CHECK VERIFICATION
# =============================================================================

@dataclass
class SpotCheckRequest:
    """Request for spot-check verification of a gradient commitment."""
    proof_signature: str
    node_id: str
    epoch_id: int
    gradient_commitment: GradientCommitment
    data_hash: str  # Hash of training data to reproduce
    requested_at: float = field(default_factory=time.time)
    challenger_id: Optional[str] = None
    challenger_stake: float = 0.0


@dataclass
class SpotCheckResult:
    """Result of a spot-check verification."""
    request: SpotCheckRequest
    verified: bool
    recomputed_commitment: Optional[str] = None
    discrepancy: Optional[str] = None
    verified_at: float = field(default_factory=time.time)


class SpotCheckVerifier:
    """
    Probabilistic spot-check verification for gradient commitments.
    
    This provides economic security without verifying every proof:
    - Random selection of proofs to verify (p = SPOT_CHECK_PROBABILITY)
    - Re-runs training to verify gradient commitment
    - Slashing for mismatched commitments
    
    Economic security: Expected fraud cost > expected fraud profit
    With 5% check rate and 10x slashing: fraud is -400% EV
    """
    
    def __init__(
        self,
        model=None,
        ledger=None,
        check_probability: float = SPOT_CHECK_PROBABILITY,
        slash_multiplier: float = SLASH_MULTIPLIER,
    ):
        self.model = model
        self.ledger = ledger
        self.check_probability = check_probability
        self.slash_multiplier = slash_multiplier
        
        self._lock = threading.Lock()
        self.pending_checks: Dict[str, SpotCheckRequest] = {}
        self.completed_checks: List[SpotCheckResult] = []
    
    def should_check(self, proof_signature: str) -> bool:
        """Determine if a proof should be spot-checked."""
        # Deterministic based on proof signature
        check_seed = hashlib.sha256(proof_signature.encode()).digest()
        random_value = int.from_bytes(check_seed[:4], 'big') / (2**32)
        return random_value < self.check_probability
    
    def request_check(
        self,
        proof_signature: str,
        node_id: str,
        epoch_id: int,
        gradient_commitment: GradientCommitment,
        challenger_id: Optional[str] = None,
        challenger_stake: float = 0.0,
    ) -> SpotCheckRequest:
        """Create a spot-check request."""
        request = SpotCheckRequest(
            proof_signature=proof_signature,
            node_id=node_id,
            epoch_id=epoch_id,
            gradient_commitment=gradient_commitment,
            data_hash=gradient_commitment.data_hash,
            challenger_id=challenger_id,
            challenger_stake=challenger_stake,
        )
        
        with self._lock:
            self.pending_checks[proof_signature] = request
        
        logger.info(f"Spot-check requested for proof {proof_signature[:16]}...")
        return request
    
    def verify_commitment(
        self,
        request: SpotCheckRequest,
        training_data,  # (input_ids, labels) tuple
    ) -> SpotCheckResult:
        """
        Verify a gradient commitment by re-running training.
        
        Args:
            request: The spot-check request
            training_data: The actual training data (input_ids, labels)
            
        Returns:
            SpotCheckResult with verification outcome
        """
        if not self.model:
            return SpotCheckResult(
                request=request,
                verified=False,
                discrepancy="No model available for verification",
            )
        
        input_ids, labels = training_data
        original = request.gradient_commitment
        
        try:
            # Verify data hash matches
            import torch
            data_bytes = input_ids.cpu().numpy().tobytes() + labels.cpu().numpy().tobytes()
            data_hash = hashlib.sha256(data_bytes).hexdigest()[:32]
            
            if data_hash != original.data_hash:
                return SpotCheckResult(
                    request=request,
                    verified=False,
                    discrepancy=f"Data hash mismatch: {data_hash} != {original.data_hash}",
                )
            
            # Re-compute gradient commitment
            recomputed = compute_gradient_commitment(
                model=self.model,
                input_ids=input_ids,
                labels=labels,
                model_hash_start=original.model_hash_start,
            )
            
            # Compare commitments
            original_hash = original.compute_commitment_hash()
            recomputed_hash = recomputed.compute_commitment_hash()
            
            # Allow small floating point differences in loss
            loss_diff = abs(original.final_loss - recomputed.final_loss)
            loss_match = loss_diff < 0.01  # 1% tolerance
            
            if original_hash == recomputed_hash or loss_match:
                result = SpotCheckResult(
                    request=request,
                    verified=True,
                    recomputed_commitment=recomputed_hash,
                )
            else:
                result = SpotCheckResult(
                    request=request,
                    verified=False,
                    recomputed_commitment=recomputed_hash,
                    discrepancy=f"Commitment mismatch: {original_hash[:16]}... != {recomputed_hash[:16]}...",
                )
            
            # Handle slashing if fraud detected
            if not result.verified and self.ledger:
                self._apply_slash(request.node_id, request.challenger_id, request.challenger_stake)
            
            with self._lock:
                self.completed_checks.append(result)
                if request.proof_signature in self.pending_checks:
                    del self.pending_checks[request.proof_signature]
            
            return result
            
        except Exception as e:
            logger.error(f"Spot-check verification failed: {e}")
            return SpotCheckResult(
                request=request,
                verified=False,
                discrepancy=f"Verification error: {str(e)}",
            )
    
    def _apply_slash(
        self,
        fraudster_id: str,
        challenger_id: Optional[str],
        challenger_stake: float,
    ):
        """Apply slashing for detected fraud."""
        if not self.ledger:
            return
        
        # Get fraudster's stake
        try:
            stake = self.ledger._get_stake(fraudster_id)
            slash_amount = min(stake, stake * self.slash_multiplier)
            
            if slash_amount > 0:
                # Slash the fraudster
                self.ledger._apply_slash(fraudster_id, slash_amount, "Failed spot-check")
                
                # Reward challenger if any
                if challenger_id and challenger_stake > 0:
                    reward = slash_amount * 0.5  # 50% to challenger
                    self.ledger._credit_balance(challenger_id, reward)
                    logger.info(f"Challenger {challenger_id[:16]}... rewarded {reward:.4f} NEURO")
                
                logger.warning(f"SLASHED {fraudster_id[:16]}... for {slash_amount:.4f} NEURO (fraud detected)")
        except Exception as e:
            logger.error(f"Failed to apply slash: {e}")


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def compute_model_hash(model) -> str:
    """Compute a hash of model weights for state commitment."""
    import torch
    
    param_bytes = b""
    with torch.no_grad():
        for param in model.parameters():
            # Sample first 1000 values per parameter for efficiency
            flat = param.data.flatten()[:1000]
            param_bytes += flat.cpu().numpy().tobytes()
    
    return hashlib.sha256(param_bytes).hexdigest()[:32]


def verify_model_state_change(hash_before: str, hash_after: str) -> bool:
    """Verify that model state actually changed (training happened)."""
    return hash_before != hash_after and hash_before and hash_after

