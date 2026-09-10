"""
End-to-End System Test

Verifies the complete flow for NeuroShard:
1. DHT - Layer announcements and discovery
2. Layer Allocation - How nodes get assigned layers
3. Gossip - PoNW proof propagation
4. Consensus - Validator proof verification
"""

import time
import hashlib
import tempfile
import os
from unittest.mock import MagicMock


def create_test_pool():
    """Create a layer pool with test architecture."""
    from neuroshard.core.model.dynamic import DynamicLayerPool
    pool = DynamicLayerPool()
    pool.vocab_capacity = 50000
    # Let the pool auto-calculate architecture
    pool._auto_recalculate_architecture()
    return pool


class TestLayerAllocationRoles:
    """Test layer allocation for different roles."""
    
    def test_driver_gets_embedding(self):
        """DRIVER should get embedding layer."""
        pool = create_test_pool()
        
        # First node becomes DRIVER
        assigned = pool.register_node(
            node_id="first_node",
            node_url="http://first:8000",
            grpc_addr="first:9000",
            available_memory_mb=8000,
            enable_training=True
        )
        
        assert pool.embedding_holder == "first_node"
        assert 0 in assigned
        print(f"✓ DRIVER gets embedding layer (layers: {assigned})")
    
    def test_full_node_gets_all(self):
        """Full node with enough memory gets ALL layers."""
        pool = create_test_pool()
        
        # Node with ALL layers becomes full node (DRIVER + VALIDATOR)
        assigned = pool.register_node(
            node_id="full_node",
            node_url="http://full:8000",
            grpc_addr="full:9000",
            available_memory_mb=16000,  # Lots of memory
            enable_training=True
        )
        
        assert pool.embedding_holder == "full_node"
        assert pool.lm_head_holder == "full_node"
        
        print(f"✓ Full node gets embedding AND LM head (layers: {assigned})")
    
    def test_observer_no_training_becomes_worker(self):
        """Observer (--no-training) cannot become DRIVER."""
        pool = create_test_pool()
        
        # Observer without training enabled
        assigned = pool.register_node(
            node_id="observer_node",
            node_url="http://observer:8000",
            grpc_addr="observer:9000",
            available_memory_mb=2000,
            enable_training=False  # Key: No training!
        )
        
        # Observer should NOT get embedding (can't be DRIVER)
        assert pool.embedding_holder != "observer_node"
        
        print(f"✓ Observer with no training gets layers: {assigned} (no embedding)")


class TestPipelineRouting:
    """Test multi-hop pipeline routing via DHT."""
    
    def test_get_next_hop_finds_peer(self):
        """get_next_hop should find peer holding next layer."""
        from neuroshard.core.network.p2p import P2PManager
        
        # Create P2P manager
        manager = P2PManager(
            my_url="http://node1:8000",
            shard_range="0-3",
            tracker_url="http://tracker:3000",
            node_token="test_token"
        )
        
        # Add peer to known_peers
        manager.known_peers["http://node2:8000"] = {
            "shard_range": "4-10",
            "last_seen": time.time()
        }
        
        # Find next hop for layer 4 (after our layer 3)
        next_hop = manager.get_next_hop(4)
        
        assert next_hop is not None
        print(f"✓ get_next_hop found: {next_hop}")
    
    def test_get_next_hop_filters_non_training_peers(self):
        """get_next_hop with for_training=True should skip non-training peers."""
        from neuroshard.core.network.p2p import P2PManager
        
        # Create P2P manager
        manager = P2PManager(
            my_url="http://driver:8000",
            shard_range="0-3",
            tracker_url="http://tracker:3000",
            node_token="test_token"
        )
        
        # Add observer (training disabled) - should be skipped
        manager.known_peers["http://observer:8000"] = {
            "shard_range": "4-10",
            "training_enabled": False,  # Observer with --no-training
            "last_seen": time.time()
        }
        
        # Add trainer (training enabled) - should be found
        manager.known_peers["http://trainer:8000"] = {
            "shard_range": "4-10",
            "training_enabled": True,
            "last_seen": time.time()
        }
        
        # Without for_training, should find either
        next_hop_any = manager.get_next_hop(4, for_training=False)
        assert next_hop_any is not None
        
        # With for_training=True, should only find trainer
        next_hop_training = manager.get_next_hop(4, for_training=True)
        assert next_hop_training == "http://trainer:8000"
        
        print("✓ get_next_hop correctly filters non-training peers")
    
    def test_observer_excluded_from_training_pipeline(self):
        """Observer nodes (--no-training) should be excluded from training pipeline."""
        from neuroshard.core.network.p2p import P2PManager
        
        # Create P2P manager for driver
        manager = P2PManager(
            my_url="http://driver:8000",
            shard_range="0-3",
            tracker_url="http://tracker:3000",
            node_token="test_token"
        )
        
        # Only add observer (training disabled)
        manager.known_peers["http://observer:8000"] = {
            "shard_range": "4-10",
            "training_enabled": False,
            "last_seen": time.time()
        }
        
        # With for_training=True, should NOT find the observer
        next_hop = manager.get_next_hop(4, for_training=True)
        
        assert next_hop is None  # No training-capable peer found
        print("✓ Observer correctly excluded from training pipeline")
    
    def test_pipeline_route_ordering(self):
        """Pipeline route should be correctly ordered by layer."""
        from neuroshard.core.model.dynamic import DynamicLayerPool, LayerAssignment
        
        pool = create_test_pool()
        
        # Manually set up a 3-node pipeline
        pool.layer_assignments = {
            0: [LayerAssignment(0, "driver", "http://driver:8000", "driver:9000")],
            1: [LayerAssignment(1, "driver", "http://driver:8000", "driver:9000")],
            2: [LayerAssignment(2, "driver", "http://driver:8000", "driver:9000")],
            3: [LayerAssignment(3, "worker", "http://worker:8000", "worker:9000")],
            4: [LayerAssignment(4, "worker", "http://worker:8000", "worker:9000")],
            5: [LayerAssignment(5, "worker", "http://worker:8000", "worker:9000")],
            6: [LayerAssignment(6, "validator", "http://validator:8000", "validator:9000")],
            7: [LayerAssignment(7, "validator", "http://validator:8000", "validator:9000")],
            8: [LayerAssignment(8, "validator", "http://validator:8000", "validator:9000")],
            9: [LayerAssignment(9, "validator", "http://validator:8000", "validator:9000")],
            10: [LayerAssignment(10, "validator", "http://validator:8000", "validator:9000")],
        }
        pool.current_num_layers = 11
        
        route = pool.get_pipeline_route()
        
        # Route should have entries for each layer
        assert len(route) == 11
        assert route[0] == "http://driver:8000"
        assert route[5] == "http://worker:8000"
        assert route[10] == "http://validator:8000"
        
        print(f"✓ Pipeline route ordered correctly: 3 nodes")


class TestGossipProofFlow:
    """Test PoNW proof gossip flow."""
    
    def test_proof_creation_and_signature(self):
        """Proofs should be correctly signed with ECDSA."""
        from neuroshard.core.economics.ledger import NEUROLedger, ProofType
        
        fd, db_path = tempfile.mkstemp(suffix='.db')
        os.close(fd)
        
        try:
            ledger = NEUROLedger(db_path=db_path, node_token="test_token")
            
            # Create proof
            proof = ledger.create_proof(
                proof_type=ProofType.UPTIME,
                uptime_seconds=60,
                layers_held=5,
                has_embedding=True,
                has_lm_head=False
            )
            
            # Proof should have signature
            assert proof.signature
            assert len(proof.signature) > 0
            
            # Verify the signature
            is_valid, reason = ledger.verify_proof(proof)
            assert is_valid, f"Proof verification failed: {reason}"
            
            print("✓ Proof created and signed correctly")
        finally:
            os.unlink(db_path)
    
    def test_all_gossip_handlers_exist(self):
        """All gossip gRPC handlers should exist."""
        from neuroshard.grpc_server import NeuroShardServiceServicer
        
        handlers = ['GossipProof', 'GossipTransaction', 'GossipStake', 
                    'GossipGradient', 'GossipValidationVote']
        
        for h in handlers:
            assert hasattr(NeuroShardServiceServicer, h), f"Missing handler: {h}"
        
        print(f"✓ All {len(handlers)} gossip handlers implemented")


class TestValidatorConsensusFlow:
    """Test validator consensus for proof verification."""
    
    def test_validators_selected_for_proof(self):
        """Validators should be selected for proof verification."""
        from neuroshard.core.consensus.validator_consensus import ValidatorConsensus
        
        consensus = ValidatorConsensus(num_validators_per_proof=3)
        
        # Register validators
        for i in range(5):
            consensus.register_validator(
                f"validator_{i}",
                stake=100.0 + i * 50,
                has_lm_head=True,
                memory_mb=4000,
                url=f"http://v{i}:9000"
            )
        
        # Select validators
        selected = consensus.select_validators(exclude_node_id="submitter")
        
        assert len(selected) == 3
        print(f"✓ Selected {len(selected)} validators for proof verification")
    
    def test_consensus_vote_flow(self):
        """Votes should affect consensus result."""
        from neuroshard.core.consensus.validator_consensus import ValidatorConsensus, ConsensusState
        
        consensus = ValidatorConsensus()
        
        # Register 2 validators
        consensus.register_validator("v1", stake=500, has_lm_head=True, memory_mb=4000, url="http://v1:9000")
        consensus.register_validator("v2", stake=200, has_lm_head=True, memory_mb=4000, url="http://v2:9000")
        
        # Submit proof
        consensus.submit_proof_for_consensus("proof_123", "submitter")
        
        # Cast votes (500/700 = 71.4% valid > 66%)
        consensus.cast_vote("proof_123", vote=False, validator_id="v2", stake=200)
        consensus.cast_vote("proof_123", vote=True, validator_id="v1", stake=500)
        
        result = consensus.get_consensus_status("proof_123")
        
        assert result.consensus_reached
        assert result.consensus_result == True  # ACCEPTED
        
        print("✓ Consensus voting works correctly")


class TestCompleteFlow:
    """Test the complete end-to-end flow."""
    
    def test_solo_node_complete_flow(self):
        """Solo node should work end-to-end."""
        from neuroshard.core.economics.ledger import NEUROLedger, ProofType
        
        fd, db_path = tempfile.mkstemp(suffix='.db')
        os.close(fd)
        
        try:
            # 1. Create layer pool
            pool = create_test_pool()
            
            # 2. Register as full node
            assigned = pool.register_node(
                node_id="solo_node",
                node_url="http://solo:8000",
                grpc_addr="solo:9000",
                available_memory_mb=16000,
                enable_training=True
            )
            
            # 3. Create ledger for PoNW
            ledger = NEUROLedger(db_path=db_path, node_token="solo_token")
            
            # 4. Generate and process proof
            proof = ledger.create_proof(
                proof_type=ProofType.UPTIME,
                uptime_seconds=60,
                layers_held=len(assigned),
                has_embedding=pool.embedding_holder == "solo_node",
                has_lm_head=pool.lm_head_holder == "solo_node"
            )
            
            success, reward, msg = ledger.process_proof(proof)
            
            # 5. Verify everything worked
            assert pool.embedding_holder == "solo_node"
            assert pool.lm_head_holder == "solo_node"
            assert success
            assert reward > 0
            
            print("✓ Solo node complete flow:")
            print(f"  - Layers: {len(assigned)}")
            print(f"  - Role: DRIVER + VALIDATOR")
            print(f"  - Reward: {reward:.6f} NEURO")
        finally:
            os.unlink(db_path)
    
    def test_two_node_pipeline_flow(self):
        """Two nodes should form a pipeline correctly."""
        pool = create_test_pool()
        
        # Node 1: First node gets all layers
        node1_assigned = pool.register_node(
            node_id="node1",
            node_url="http://node1:8000",
            grpc_addr="node1:9000",
            available_memory_mb=8000,
            enable_training=True
        )
        
        # Node 2: Second node provides redundancy
        node2_assigned = pool.register_node(
            node_id="node2",
            node_url="http://node2:8000",
            grpc_addr="node2:9000",
            available_memory_mb=8000,
            enable_training=True
        )
        
        # Both nodes should have layer 0 (redundancy)
        assert 0 in node1_assigned
        assert 0 in node2_assigned
        
        # Both nodes should have all layers (full redundancy)
        assert len(node1_assigned) == 11
        assert len(node2_assigned) == 11
        
        print("✓ Two-node pipeline with full redundancy:")
        print(f"  - Node1: layers {node1_assigned}")
        print(f"  - Node2: layers {node2_assigned} (redundancy)")


if __name__ == "__main__":
    print("\n" + "="*70)
    print("END-TO-END SYSTEM TESTS")
    print("="*70 + "\n")
    
    print("--- Layer Allocation Roles ---")
    t2 = TestLayerAllocationRoles()
    t2.test_driver_gets_embedding()
    t2.test_full_node_gets_all()
    t2.test_observer_no_training_becomes_worker()
    
    print("\n--- Pipeline Routing ---")
    t3 = TestPipelineRouting()
    t3.test_get_next_hop_finds_peer()
    t3.test_get_next_hop_filters_non_training_peers()
    t3.test_observer_excluded_from_training_pipeline()
    t3.test_pipeline_route_ordering()
    
    print("\n--- Gossip Proof Flow ---")
    t4 = TestGossipProofFlow()
    t4.test_proof_creation_and_signature()
    t4.test_all_gossip_handlers_exist()
    
    print("\n--- Validator Consensus ---")
    t5 = TestValidatorConsensusFlow()
    t5.test_validators_selected_for_proof()
    t5.test_consensus_vote_flow()
    
    print("\n--- Complete Flows ---")
    t6 = TestCompleteFlow()
    t6.test_solo_node_complete_flow()
    t6.test_two_node_pipeline_flow()
    
    print("\n" + "="*70)
    print("ALL END-TO-END TESTS PASSED ✓")
    print("="*70 + "\n")
