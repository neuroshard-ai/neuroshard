"""
Proof of Neural Work Tests

Tests for the optimistic verification system:
- Pipeline proof generation
- Cohort sync proof generation
- Optimistic acceptance
- Challenge system and slashing

Uses real cryptographic operations for realistic testing.
"""

import pytest
import time
import hashlib
import json
import torch
from unittest.mock import MagicMock, patch
from typing import Dict, Any


# =============================================================================
# PIPELINE PROOF TESTS
# =============================================================================

class TestPipelineProofs:
    """Tests for pipeline proof generation and validation."""
    
    def test_pipeline_proof_creation(self):
        """Test creating a pipeline proof."""
        from neuroshard.core.consensus.verifier import PipelineProof
        
        proof = PipelineProof(
            node_id="node-123",
            quorum_id="quorum-456",
            layer_range=(0, 4),
            batch_id="batch-789",
            step_id=1,
            input_activation_hash=hashlib.sha256(b"input").digest(),
            output_activation_hash=hashlib.sha256(b"output").digest(),
            gradient_merkle_root=hashlib.sha256(b"gradients").digest(),
            prev_node_id="prev-node",
            next_node_id="next-node",
            speed_tier="tier2",
            contribution_mode="pipeline",
        )
        
        assert proof.node_id == "node-123"
        assert proof.quorum_id == "quorum-456"
        assert proof.layer_range == (0, 4)
        assert proof.timestamp > 0
    
    def test_pipeline_proof_serialization(self):
        """Test serializing and deserializing pipeline proofs."""
        from neuroshard.core.consensus.verifier import PipelineProof
        
        proof = PipelineProof(
            node_id="node-1",
            quorum_id="quorum-1",
            layer_range=(4, 8),
            batch_id="batch-1",
            step_id=5,
            input_activation_hash=b"input_hash",
            output_activation_hash=b"output_hash",
            gradient_merkle_root=b"grad_root",
            prev_node_id="prev",
            next_node_id="next",
        )
        
        # Serialize
        data = proof.to_dict()
        
        assert 'node_id' in data
        assert 'layer_range' in data
        
        # Deserialize
        restored = PipelineProof.from_dict(data)
        
        assert restored.node_id == proof.node_id
        assert restored.layer_range == proof.layer_range
        assert restored.step_id == proof.step_id
    
    def test_pipeline_proof_signing(self):
        """Test signing pipeline proofs."""
        from neuroshard.core.consensus.verifier import PipelineProof
        from neuroshard.core.crypto.ecdsa import (
            derive_keypair_from_token, ecdsa_sign, ecdsa_verify
        )
        
        # Create key pair from a test token
        token = "test-node-token-123"
        keypair = derive_keypair_from_token(token)
        
        proof = PipelineProof(
            node_id=keypair.node_id[:16],
            quorum_id="quorum-1",
            layer_range=(0, 4),
            batch_id="batch-1",
            step_id=1,
            input_activation_hash=b"in",
            output_activation_hash=b"out",
            gradient_merkle_root=b"grad",
            prev_node_id="",
            next_node_id="next",
        )
        
        # Create data to sign (as string)
        proof_data = json.dumps(proof.to_dict(), sort_keys=True)
        
        # Sign using the crypto module functions
        signature = ecdsa_sign(proof_data, keypair.private_key_bytes)
        proof.signature = signature
        
        assert len(proof.signature) > 0
        
        # Verify
        is_valid = ecdsa_verify(proof_data, signature, keypair.public_key_bytes)
        assert is_valid
    
    def test_activation_hash_computation(self, simple_model):
        """Test computing activation hashes."""
        simple_model.eval()
        
        input_ids = torch.randint(0, simple_model.vocab_size, (2, 32))
        
        with torch.no_grad():
            x = simple_model.embedding(input_ids)
            
            # Hash activation
            activation_bytes = x.numpy().tobytes()
            activation_hash = hashlib.sha256(activation_bytes).digest()
            
            # Process through one layer
            x = simple_model.layers[0](x)
            
            # Hash output
            output_bytes = x.numpy().tobytes()
            output_hash = hashlib.sha256(output_bytes).digest()
        
        # Hashes should be different
        assert activation_hash != output_hash


# =============================================================================
# COHORT SYNC PROOF TESTS
# =============================================================================

class TestCohortSyncProofs:
    """Tests for cohort synchronization proof generation."""
    
    def test_cohort_sync_proof_creation(self):
        """Test creating a cohort sync proof."""
        from neuroshard.core.consensus.verifier import CohortSyncProof
        
        proof = CohortSyncProof(
            node_id="node-1",
            layer_range=(0, 4),
            sync_round=5,
            batches_processed=100,
            pseudo_gradient_hash=hashlib.sha256(b"pseudo").digest(),
            cohort_members=["node-1", "node-2", "node-3"],
            aggregated_gradient_hash=hashlib.sha256(b"agg").digest(),
            initial_weights_hash=hashlib.sha256(b"init").digest(),
            final_weights_hash=hashlib.sha256(b"final").digest(),
            quorum_id="quorum-1",
        )
        
        assert proof.sync_round == 5
        assert proof.batches_processed == 100
        assert len(proof.cohort_members) == 3
    
    def test_cohort_sync_proof_serialization(self):
        """Test serializing cohort sync proofs."""
        from neuroshard.core.consensus.verifier import CohortSyncProof
        
        proof = CohortSyncProof(
            node_id="node-1",
            layer_range=(4, 8),
            sync_round=10,
            batches_processed=500,
            pseudo_gradient_hash=b"pseudo",
            cohort_members=["a", "b"],
            aggregated_gradient_hash=b"agg",
            initial_weights_hash=b"init",
            final_weights_hash=b"final",
        )
        
        data = proof.to_dict()
        restored = CohortSyncProof.from_dict(data)
        
        assert restored.sync_round == 10
        assert restored.batches_processed == 500
    
    def test_pseudo_gradient_hash(self, simple_model):
        """Test hashing pseudo-gradients for proof."""
        from neuroshard.core.swarm.diloco import DiLoCoTrainer, DiLoCoConfig
        
        config = DiLoCoConfig(inner_steps=5)
        trainer = DiLoCoTrainer(model=simple_model, config=config)
        
        trainer.start_inner_loop()
        
        # Simulate training
        with torch.no_grad():
            for param in simple_model.parameters():
                param.add_(torch.randn_like(param) * 0.01)
        
        # Compute pseudo-gradients
        pseudo_grads = trainer.compute_pseudo_gradient()
        
        # Hash them
        combined = b""
        for name in sorted(pseudo_grads.keys()):
            combined += pseudo_grads[name].numpy().tobytes()
        
        grad_hash = hashlib.sha256(combined).digest()
        
        assert len(grad_hash) == 32


# =============================================================================
# OPTIMISTIC ACCEPT TESTS
# =============================================================================

class TestOptimisticAccept:
    """Tests for optimistic proof acceptance."""
    
    def test_verifier_initialization(self):
        """Test ProofVerifier initialization."""
        from neuroshard.core.consensus.verifier import ProofVerifier
        
        verifier = ProofVerifier(optimistic=True, network_size=10)
        
        assert verifier.optimistic
        assert verifier.network_size == 10
    
    def test_proof_immediate_acceptance(self):
        """Test that proofs can be accepted optimistically."""
        from neuroshard.core.consensus.verifier import ProofVerifier, PendingProof
        
        verifier = ProofVerifier(optimistic=True)
        
        # Accept proof optimistically
        proof_data = {"node_id": "test-node", "work": 100}
        proof_id = "proof-123"
        reward = 10.0
        
        accepted, message, pending = verifier.accept_proof_optimistic(
            proof=proof_data,
            proof_id=proof_id,
            reward=reward,
        )
        
        assert accepted
        assert pending is not None
        assert pending.reward_credited == reward
    
    def test_challenge_window_constant(self):
        """Test challenge window constant is defined."""
        from neuroshard.core.consensus.verifier import CHALLENGE_WINDOW
        
        # Challenge window should be a reasonable time (e.g., 10 minutes = 600 seconds)
        assert CHALLENGE_WINDOW > 0
        assert CHALLENGE_WINDOW <= 3600  # Max 1 hour


# =============================================================================
# CHALLENGE SYSTEM TESTS
# =============================================================================

class TestChallengeSystem:
    """Tests for the challenge and slashing system."""
    
    def test_challenge_proof(self):
        """Test challenging an accepted proof."""
        from neuroshard.core.consensus.verifier import ProofVerifier
        
        verifier = ProofVerifier(optimistic=True)
        
        # First accept a proof
        proof_data = {"node_id": "suspect", "work": 1000}
        proof_id = "challengeable-proof"
        
        verifier.accept_proof_optimistic(
            proof=proof_data,
            proof_id=proof_id,
            reward=10.0,
        )
        
        # Challenge it
        success, message, result = verifier.challenge_proof(
            proof_id=proof_id,
            challenger_id="challenger",
            challenger_stake=100.0,
        )
        
        # Challenge should be processed
        assert message  # Should have a message
    
    def test_slashing_constants(self):
        """Test slashing constants are defined."""
        from neuroshard.core.economics.constants import SLASH_AMOUNT
        
        assert SLASH_AMOUNT > 0
    
    def test_invalid_proof_detection(self):
        """Test detecting obviously invalid proofs."""
        from neuroshard.core.consensus.verifier import PipelineProof
        
        # Create proof with mismatched hashes (suspicious)
        proof = PipelineProof(
            node_id="malicious-node",
            quorum_id="quorum-1",
            layer_range=(0, 4),
            batch_id="batch-invalid",
            step_id=1,
            input_activation_hash=b"same",
            output_activation_hash=b"same",  # Suspicious - same as input!
            gradient_merkle_root=b"grad",
            prev_node_id="",
            next_node_id="",
        )
        
        # Basic validation - same input/output hash is suspicious
        is_suspicious = (
            proof.input_activation_hash == proof.output_activation_hash
        )
        
        assert is_suspicious


# =============================================================================
# REWARD CALCULATION TESTS
# =============================================================================

class TestRewardCalculation:
    """Tests for PoNW reward calculations."""
    
    def test_pipeline_role_shares(self):
        """Test reward share distribution by pipeline role."""
        from neuroshard.core.economics.constants import (
            INITIATOR_SHARE, PROCESSOR_SHARE, FINISHER_SHARE
        )
        
        # Shares should sum to 1
        total = INITIATOR_SHARE + PROCESSOR_SHARE + FINISHER_SHARE
        assert abs(total - 1.0) < 0.01
    
    def test_initiator_bonus_multiplier(self):
        """Test initiator bonus is ADDITIVE (not multiplicative)."""
        from neuroshard.core.economics.constants import INITIATOR_BONUS, calculate_role_bonus
        
        # INITIATOR_BONUS is 0.2 (+20% as an ADDITIVE bonus)
        assert INITIATOR_BONUS == 0.2
        
        # Using calculate_role_bonus, the multiplier is 1.0 + bonus
        multiplier = calculate_role_bonus(has_embedding=True)
        assert multiplier == 1.2  # 1.0 + 0.2
        
        base_reward = 100.0
        initiator_reward = base_reward * multiplier
        
        # Should be 120% of base
        assert initiator_reward == 120.0
    
    def test_finisher_bonus_multiplier(self):
        """Test finisher bonus is ADDITIVE (not multiplicative)."""
        from neuroshard.core.economics.constants import FINISHER_BONUS, calculate_role_bonus
        
        # FINISHER_BONUS is 0.3 (+30% as an ADDITIVE bonus)
        assert FINISHER_BONUS == 0.3
        
        # Using calculate_role_bonus, the multiplier is 1.0 + bonus
        multiplier = calculate_role_bonus(has_lm_head=True)
        assert multiplier == 1.3  # 1.0 + 0.3
        
        base_reward = 100.0
        finisher_reward = base_reward * multiplier
        
        # Should be 130% of base
        assert finisher_reward == 130.0
    
    def test_layer_bonus_calculation(self):
        """Test layer scaling via per-layer training reward rate."""
        from neuroshard.core.economics.constants import (
            TRAINING_REWARD_PER_BATCH_PER_LAYER,
            calculate_training_reward,
        )
        
        # Layer scaling is now done via per-layer rate, not a bonus multiplier
        # LAYER_BONUS is deprecated (0.0) - scaling is direct via:
        # reward = batches × TRAINING_REWARD_PER_BATCH_PER_LAYER × layers_held
        
        batches = 100
        layers = 4
        
        reward = calculate_training_reward(batches, layers)
        expected = batches * TRAINING_REWARD_PER_BATCH_PER_LAYER * layers
        
        # Reward should scale linearly with layers
        assert abs(reward - expected) < 0.0001
        assert abs(reward - 0.2) < 0.0001  # 100 × 0.0005 × 4 = 0.2
    
    def test_sqrt_n_batch_weight(self):
        """Test sqrt(n) weighting for batch count."""
        import math
        
        def calculate_weight(batches: int, freshness: float = 1.0) -> float:
            return math.sqrt(batches) * freshness
        
        w_100 = calculate_weight(100)
        w_400 = calculate_weight(400)
        
        # sqrt(400) / sqrt(100) = 2
        assert abs(w_400 / w_100 - 2.0) < 0.01
    
    def test_scarcity_bonus(self):
        """Test scarcity bonus calculation."""
        from neuroshard.core.economics.constants import compute_scarcity_bonus
        
        # Under-replicated layer should get bonus
        bonus = compute_scarcity_bonus(
            layer_id=5,
            replicas=1,
            network_size=100,
        )
        
        # Bonus should be positive for under-replicated layers
        assert bonus > 1.0


# =============================================================================
# INTEGRATION TESTS
# =============================================================================

class TestPoNWIntegration:
    """Integration tests for complete PoNW flow."""
    
    def test_training_proof_generation(self, simple_model, mock_genesis_loader):
        """Test generating proof during training."""
        from neuroshard.core.consensus.verifier import PipelineProof
        
        # Simulate training batch
        batch = mock_genesis_loader.get_batch()
        
        with torch.no_grad():
            x = simple_model.embedding(batch['input_ids'])
            input_hash = hashlib.sha256(x.numpy().tobytes()).digest()
            
            for layer in simple_model.layers:
                x = layer(x)
            
            output_hash = hashlib.sha256(x.numpy().tobytes()).digest()
        
        # Create proof
        proof = PipelineProof(
            node_id="training-node",
            quorum_id="train-quorum",
            layer_range=(0, 4),
            batch_id=f"train-{time.time()}",
            step_id=1,
            input_activation_hash=input_hash,
            output_activation_hash=output_hash,
            gradient_merkle_root=hashlib.sha256(b"grads").digest(),
            prev_node_id="",
            next_node_id="",
            contribution_mode="pipeline",
        )
        
        # Verify proof structure
        assert proof.input_activation_hash != proof.output_activation_hash
        assert proof.layer_range == (0, 4)
    
    def test_sync_proof_generation(self, simple_model):
        """Test generating cohort sync proof."""
        from neuroshard.core.consensus.verifier import CohortSyncProof
        from neuroshard.core.swarm.diloco import DiLoCoTrainer, DiLoCoConfig
        
        # Setup trainer
        config = DiLoCoConfig(inner_steps=5)
        trainer = DiLoCoTrainer(model=simple_model, config=config)
        
        # Record initial weights
        initial_hash = hashlib.sha256(
            b"".join(p.data.numpy().tobytes() for p in simple_model.parameters())
        ).digest()
        
        trainer.start_inner_loop()
        
        # Train
        with torch.no_grad():
            for param in simple_model.parameters():
                param.add_(torch.randn_like(param) * 0.01)
        
        # Compute pseudo-gradients
        pseudo_grads = trainer.compute_pseudo_gradient()
        grad_bytes = b"".join(g.numpy().tobytes() for g in pseudo_grads.values())
        pseudo_hash = hashlib.sha256(grad_bytes).digest()
        
        # Record final weights
        final_hash = hashlib.sha256(
            b"".join(p.data.numpy().tobytes() for p in simple_model.parameters())
        ).digest()
        
        # Create sync proof
        sync_proof = CohortSyncProof(
            node_id="sync-node",
            layer_range=(0, 4),
            sync_round=1,
            batches_processed=100,
            pseudo_gradient_hash=pseudo_hash,
            cohort_members=["sync-node", "peer-1", "peer-2"],
            aggregated_gradient_hash=pseudo_hash,
            initial_weights_hash=initial_hash,
            final_weights_hash=final_hash,
            quorum_id="sync-quorum",
        )
        
        assert sync_proof.batches_processed == 100
        assert len(sync_proof.cohort_members) == 3
    
    def test_optimistic_acceptance_flow(self):
        """Test complete optimistic acceptance flow."""
        from neuroshard.core.consensus.verifier import ProofVerifier
        
        verifier = ProofVerifier(optimistic=True)
        
        # 1. Submit and accept proof
        proof_data = {"node_id": "worker", "batches": 100}
        proof_id = f"proof-{time.time()}"
        reward = 50.0
        
        accepted, message, pending = verifier.accept_proof_optimistic(
            proof=proof_data,
            proof_id=proof_id,
            reward=reward,
        )
        
        assert accepted
        assert pending.reward_credited == reward
        
        # 2. Proof should be pending during challenge window
        assert verifier._pending_proofs.get_proof(proof_id) is not None
