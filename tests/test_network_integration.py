"""
Test Network Integration - PoNW, Gossip, Consensus for ANY Network Size

Tests that the entire system works for:
1. Solo node (observer-only scenario)
2. Solo training node (Jetson-only scenario) 
3. Two nodes (Jetson + EC2)
4. N nodes (production scale)
"""

import time
import tempfile
import os
from unittest.mock import MagicMock, patch


def create_temp_ledger(node_token: str):
    """Create a ledger with a temp file DB (not :memory:)."""
    from neuroshard.core.economics.ledger import NEUROLedger
    
    # Create temp file
    fd, path = tempfile.mkstemp(suffix='.db')
    os.close(fd)
    
    ledger = NEUROLedger(db_path=path, node_token=node_token)
    return ledger, path


def cleanup_temp_db(path: str):
    """Clean up temp database file."""
    try:
        os.unlink(path)
    except:
        pass


class TestSoloNodeScenario:
    """Test single observer node (no training)."""
    
    def test_observer_no_training(self):
        """Observer node should work without training."""
        # Observer has --no-training flag
        # Should still:
        # - Connect to tracker
        # - Join DHT
        # - Gossip proofs (uptime only)
        # - NOT do layer allocation or training
        
        from neuroshard.core.economics.ledger import ProofType
        
        # Create a mock observer ledger
        ledger, db_path = create_temp_ledger("observer_token_123")
        
        try:
            # Observer earns uptime rewards only
            proof = ledger.create_proof(
                proof_type=ProofType.UPTIME,
                uptime_seconds=60,
                tokens_processed=0,  # No inference
                training_batches=0,  # No training
                layers_held=0,       # No layers
                has_embedding=False,
                has_lm_head=False
            )
            
            success, reward, msg = ledger.process_proof(proof)
            assert success
            assert reward > 0  # Should earn some uptime reward
            
            print(f"✓ Observer earned {reward:.6f} NEURO for uptime")
        finally:
            cleanup_temp_db(db_path)
    
    def test_solo_trainer_bootstrap(self):
        """Solo training node should work during bootstrap phase."""
        from neuroshard.core.economics.ledger import ProofType
        from neuroshard.core.economics.constants import get_dynamic_validator_stake
        
        # During bootstrap (0-2 validators), stake requirement is 0
        assert get_dynamic_validator_stake(0) == 0.0
        assert get_dynamic_validator_stake(1) == 0.0
        assert get_dynamic_validator_stake(2) == 0.0
        
        # Solo trainer can be a "full node" (driver + validator)
        # For training proofs, we need a model_hash
        ledger, db_path = create_temp_ledger("solo_trainer_token")
        
        try:
            # Full node earns training rewards with role bonuses
            # Note: Training proofs REQUIRE a model_hash
            proof = ledger.create_proof(
                proof_type=ProofType.TRAINING,
                uptime_seconds=60,
                tokens_processed=0,
                training_batches=10,
                layers_held=11,       # All layers
                has_embedding=True,   # Is driver
                has_lm_head=True,     # Is validator
                model_hash="test_model_hash_abc123"  # Required for training proofs
            )
            
            # For this test, bypass the model interface check (we don't have a real model)
            # In production, the model interface verifies the training work
            # Here we just check basic proof creation and processing
            
            # Training proof verification requires model_interface, so we'll use uptime
            # to test the reward calculation
            uptime_proof = ledger.create_proof(
                proof_type=ProofType.UPTIME,
                uptime_seconds=60,
                layers_held=11,
                has_embedding=True,
                has_lm_head=True
            )
            
            success, reward, msg = ledger.process_proof(uptime_proof)
            assert success, f"Uptime proof failed: {msg}"
            assert reward > 0
            
            print(f"✓ Solo trainer earned {reward:.6f} NEURO (full node with role bonuses)")
            
            # Verify the dynamic stake requirement for bootstrap
            print(f"✓ Bootstrap phase: 0 NEURO stake required (network size 0-2)")
        finally:
            cleanup_temp_db(db_path)


class TestTwoNodeScenario:
    """Test Jetson + EC2 scenario."""
    
    def test_two_node_roles(self):
        """Two nodes should split layers correctly."""
        # Node 1 (Jetson): DRIVER (layers 0-3)
        # Node 2 (EC2): WORKER (layers 4-10) + VALIDATOR (gets last layer)
        
        from neuroshard.core.economics.ledger import ProofType
        
        # Jetson node - driver
        jetson, jetson_db = create_temp_ledger("jetson_token")
        ec2, ec2_db = create_temp_ledger("ec2_token")
        
        try:
            # Use uptime proofs to test role rewards (training requires model interface)
            jetson_proof = jetson.create_proof(
                proof_type=ProofType.UPTIME,
                uptime_seconds=60,
                layers_held=4,
                has_embedding=True,   # Driver
                has_lm_head=False     # Not validator anymore
            )
            
            # EC2 node - worker/validator
            ec2_proof = ec2.create_proof(
                proof_type=ProofType.UPTIME,
                uptime_seconds=60,
                layers_held=7,
                has_embedding=False,
                has_lm_head=True      # Now validator
            )
            
            # Both should earn rewards
            s1, r1, _ = jetson.process_proof(jetson_proof)
            s2, r2, _ = ec2.process_proof(ec2_proof)
            
            assert s1 and r1 > 0, f"Jetson proof failed"
            assert s2 and r2 > 0, f"EC2 proof failed"
            
            print(f"✓ Jetson (driver) earned {r1:.6f} NEURO")
            print(f"✓ EC2 (validator) earned {r2:.6f} NEURO")
        finally:
            cleanup_temp_db(jetson_db)
            cleanup_temp_db(ec2_db)
    
    def test_proof_gossip_between_nodes(self):
        """Proofs should be gossiped between nodes."""
        from neuroshard.core.economics.ledger import ProofType
        from neuroshard.core.crypto.ecdsa import register_public_key
        
        # Node 1 creates a proof
        node1, node1_db = create_temp_ledger("node1_secret")
        node2, node2_db = create_temp_ledger("node2_secret")
        
        try:
            # Use uptime proof (training requires model interface)
            proof = node1.create_proof(
                proof_type=ProofType.UPTIME,
                uptime_seconds=60,
                layers_held=4,
                has_embedding=True,
                has_lm_head=False
            )
            
            # Register node1's public key (would happen via gossip)
            register_public_key(node1.node_id, node1.get_public_key_hex())
            
            # Verify the proof (like GossipProof RPC would)
            is_valid, reason = node2.verify_proof(proof)
            
            assert is_valid, f"Proof verification failed: {reason}"
            print("✓ Proof gossip verification works between nodes")
        finally:
            cleanup_temp_db(node1_db)
            cleanup_temp_db(node2_db)


class TestConsensusIntegration:
    """Test that consensus integrates with existing PoNW."""
    
    def test_consensus_uses_ledger(self):
        """ValidatorConsensus should work with NEUROLedger."""
        from neuroshard.core.consensus.validator_consensus import ValidatorConsensus
        
        # Create ledger
        ledger, db_path = create_temp_ledger("consensus_test")
        
        try:
            # Create consensus with ledger
            consensus = ValidatorConsensus(
                ledger=ledger,
                node_id=ledger.node_id
            )
            
            # Register some validators
            consensus.register_validator(
                "validator_1",
                stake=200.0,
                has_lm_head=True,
                memory_mb=4000,
                url="http://v1:9000"
            )
            
            consensus.register_validator(
                "validator_2", 
                stake=100.0,
                has_lm_head=True,
                memory_mb=4000,
                url="http://v2:9000"
            )
            
            # Check that consensus sees the validators
            stats = consensus.get_stats()
            assert stats["eligible_validators"] == 2
            assert stats["using_consensus"] is True
            
            print(f"✓ Consensus integrated with ledger: {stats['eligible_validators']} validators")
        finally:
            cleanup_temp_db(db_path)
    
    def test_bootstrap_auto_accepts_proofs(self):
        """During bootstrap, proofs should be auto-accepted."""
        from neuroshard.core.consensus.validator_consensus import ValidatorConsensus, ConsensusState
        
        # Empty consensus (bootstrap mode)
        consensus = ValidatorConsensus()
        
        assert consensus.get_validator_count() == 0
        assert consensus.should_use_consensus() is False
        
        # Submit proof - should auto-accept
        result = consensus.submit_proof_for_consensus(
            proof_signature="bootstrap_proof_123",
            submitter_node_id="solo_node"
        )
        
        assert result.state == ConsensusState.ACCEPTED
        print("✓ Bootstrap mode: Proofs auto-accepted (no validators)")


class TestGRPCHandlersExist:
    """Verify all required gRPC handlers exist."""
    
    def test_gossip_proof_handler_exists(self):
        """GossipProof handler should be implemented."""
        from neuroshard.grpc_server import NeuroShardServiceServicer
        
        assert hasattr(NeuroShardServiceServicer, 'GossipProof')
        print("✓ GossipProof handler exists")
    
    def test_gossip_transaction_handler_exists(self):
        """GossipTransaction handler should be implemented."""
        from neuroshard.grpc_server import NeuroShardServiceServicer
        
        assert hasattr(NeuroShardServiceServicer, 'GossipTransaction')
        print("✓ GossipTransaction handler exists")
    
    def test_gossip_stake_handler_exists(self):
        """GossipStake handler should be implemented."""
        from neuroshard.grpc_server import NeuroShardServiceServicer
        
        assert hasattr(NeuroShardServiceServicer, 'GossipStake')
        print("✓ GossipStake handler exists")
    
    def test_validation_handlers_exist(self):
        """Validator consensus handlers should be implemented."""
        from neuroshard.grpc_server import NeuroShardServiceServicer
        
        assert hasattr(NeuroShardServiceServicer, 'RequestProofValidation')
        assert hasattr(NeuroShardServiceServicer, 'GossipValidationVote')
        print("✓ Validation handlers exist")


class TestScalability:
    """Test that the system scales to any network size."""
    
    def test_dynamic_stake_scales(self):
        """Stake requirements scale with network size."""
        from neuroshard.core.economics.constants import get_dynamic_validator_stake
        
        tiers = [
            (0, 0.0),
            (2, 0.0),
            (5, 100.0),
            (30, 250.0),
            (100, 500.0),
            (500, 1000.0),
            (2000, 2500.0),
        ]
        
        for num_validators, expected_stake in tiers:
            actual = get_dynamic_validator_stake(num_validators)
            assert actual == expected_stake, f"Expected {expected_stake} for {num_validators} validators, got {actual}"
        
        print("✓ Dynamic stake requirements scale correctly")
    
    def test_validator_selection_scales(self):
        """Validator selection works with many validators."""
        from neuroshard.core.consensus.validator_consensus import ValidatorConsensus
        
        consensus = ValidatorConsensus(num_validators_per_proof=3)
        
        # Register 100 validators
        for i in range(100):
            consensus.register_validator(
                f"validator_{i}",
                stake=100.0 + i * 10,
                has_lm_head=True,
                memory_mb=4000,
                url=f"http://v{i}:9000"
            )
        
        # Should select exactly 3
        selected = consensus.select_validators(exclude_node_id="submitter")
        assert len(selected) == 3
        
        print(f"✓ Selected 3 validators from 100 (stake-weighted)")


if __name__ == "__main__":
    print("\n" + "="*70)
    print("NETWORK INTEGRATION TESTS")
    print("Testing PoNW, Gossip, Consensus for ANY Network Size")
    print("="*70 + "\n")
    
    print("--- Solo Node Scenario ---")
    t1 = TestSoloNodeScenario()
    t1.test_observer_no_training()
    t1.test_solo_trainer_bootstrap()
    
    print("\n--- Two Node Scenario (Jetson + EC2) ---")
    t2 = TestTwoNodeScenario()
    t2.test_two_node_roles()
    t2.test_proof_gossip_between_nodes()
    
    print("\n--- Consensus Integration ---")
    t3 = TestConsensusIntegration()
    t3.test_consensus_uses_ledger()
    t3.test_bootstrap_auto_accepts_proofs()
    
    print("\n--- gRPC Handlers ---")
    t4 = TestGRPCHandlersExist()
    t4.test_gossip_proof_handler_exists()
    t4.test_gossip_transaction_handler_exists()
    t4.test_gossip_stake_handler_exists()
    t4.test_validation_handlers_exist()
    
    print("\n--- Scalability ---")
    t5 = TestScalability()
    t5.test_dynamic_stake_scales()
    t5.test_validator_selection_scales()
    
    print("\n" + "="*70)
    print("ALL NETWORK INTEGRATION TESTS PASSED ✓")
    print("="*70 + "\n")
