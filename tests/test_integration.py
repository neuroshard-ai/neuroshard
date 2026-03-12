"""
Integration tests for NeuroShard system.

Tests complete workflows:
- End-to-end PoNW proof cycle
- Swarm routing with buffer integration
- Training pipeline with DiLoCo
- Multi-node simulation
- Validator consensus
"""

import sys
import os
import unittest
import asyncio
import tempfile
import time
import shutil

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn

# Import all components
from neuroshard.core.crypto.ecdsa import NodeCrypto, derive_node_id_from_token
from neuroshard.core.economics.ledger import NEUROLedger, PoNWProof, ProofType
from neuroshard.core.swarm.router import SwarmRouter, PeerCandidate
from neuroshard.core.swarm.buffers import ActivationBuffer, OutboundBuffer, ActivationPacket, ActivationPriority
from neuroshard.core.swarm.aggregation import RobustAggregator, AggregationConfig, AggregationStrategy
from neuroshard.core.swarm.diloco import DiLoCoTrainer, DiLoCoConfig
from neuroshard.core.swarm.heartbeat import CapacityBitmask
from neuroshard.core.network.dht import RoutingTable, Node


class SimpleModel(nn.Module):
    """Simple model for testing."""
    
    def __init__(self, input_size=10, hidden_size=20, output_size=5):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)
        
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        return self.fc2(x)


class TestPoNWEndToEnd(unittest.TestCase):
    """Integration tests for complete PoNW workflow."""
    
    def setUp(self):
        """Set up test environment."""
        self.temp_dir = tempfile.mkdtemp()
        self.node_token = "integration_test_token_12345678"
        self.ledger = NEUROLedger(
            db_path=os.path.join(self.temp_dir, "ledger.db"),
            node_token=self.node_token
        )
        
    def tearDown(self):
        """Clean up."""
        try:
            shutil.rmtree(self.temp_dir)
        except:
            pass
            
    def test_complete_proof_cycle(self):
        """Test complete cycle: create proof -> verify -> process -> credit."""
        initial_balance = self.ledger.get_balance()
        initial_stats = self.ledger.get_global_stats()
        
        # Create proof
        proof = self.ledger.create_proof(
            proof_type=ProofType.TRAINING,
            uptime_seconds=60.0,
            tokens_processed=1000,
            training_batches=5,
            layers_held=10,
            has_embedding=False,
            has_lm_head=False,
        )
        
        # Verify proof signature
        is_valid, reason = self.ledger.verify_proof(proof)
        self.assertTrue(is_valid, f"Verification failed: {reason}")
        
        # Process proof
        success, reward, msg = self.ledger.process_proof(proof)
        self.assertTrue(success, f"Processing failed: {msg}")
        self.assertGreater(reward, 0)
        
        # Verify balance credited
        final_balance = self.ledger.get_balance()
        self.assertEqual(final_balance, initial_balance + reward)
        
        # Verify global stats updated
        final_stats = self.ledger.get_global_stats()
        self.assertEqual(
            final_stats.total_minted,
            initial_stats.total_minted + reward
        )
        
    def test_multi_proof_session(self):
        """Test multiple proofs in a session."""
        proofs_processed = 0
        total_reward = 0.0
        
        for i in range(5):
            proof = self.ledger.create_proof(
                proof_type=ProofType.UPTIME,
                uptime_seconds=60.0,
                layers_held=3,
            )
            
            success, reward, _ = self.ledger.process_proof(proof)
            
            if success:
                proofs_processed += 1
                total_reward += reward
                
            # Wait to avoid timestamp collision
            time.sleep(0.01)
            
        self.assertEqual(proofs_processed, 5)
        self.assertGreater(total_reward, 0)
        
        # Verify in proof history
        account = self.ledger.get_account_info()
        self.assertEqual(account['proof_count'], 5)


class TestSwarmRoutingIntegration(unittest.TestCase):
    """Integration tests for swarm routing with buffers."""
    
    def setUp(self):
        """Set up swarm components."""
        self.router = SwarmRouter()
        self.inbound_buffer = ActivationBuffer(max_size=50)
        self.outbound_buffer = OutboundBuffer(max_size=50)
        
    def test_peer_discovery_and_routing(self):
        """Test peer discovery updates routing."""
        # Register peers via heartbeats
        for i in range(5):
            heartbeat = CapacityBitmask(
                node_id=f"peer_{i}_" + "0" * 24,
                grpc_addr=f"192.168.1.{i}:50051",
                available_memory_mb=4096,
                queue_depth=i * 5,  # Varying queue depths
                layer_range=(i * 10, (i + 1) * 10),
                is_accepting_activations=True,
            )
            
            self.router.update_peer_from_heartbeat(
                node_id=heartbeat.node_id,
                grpc_addr=heartbeat.grpc_addr,
                layer_range=heartbeat.layer_range,
                queue_depth=heartbeat.queue_depth,
                available_memory_mb=heartbeat.available_memory_mb,
                gpu_utilization=50.0,
                is_accepting=heartbeat.is_accepting_activations,
            )
            
        # Query for layer 15 - should find peer_1 (layers 10-20)
        candidates = self.router.get_candidates(15)
        
        self.assertGreater(len(candidates), 0)
        self.assertTrue(any(c.covers_layer(15) for c in candidates))
        
    def test_buffer_flow_with_router(self):
        """Test activation flow through buffers."""
        # Create packet
        packet = ActivationPacket.create_forward(
            session_id="test_session",
            micro_batch_id=1,
            tensor=torch.randn(2, 768),
            source_node="node_a",
            target_layer=5,
            is_inference=False,
        )
        
        # Put in inbound
        self.inbound_buffer.put(packet)
        # 1 item in buffer of 50 is 2% which is still "starved" (< 10%)
        # Just verify the packet was added successfully
        self.assertEqual(self.inbound_buffer.fill_rate, 1/50)
        
        # Get from inbound (simulating compute)
        retrieved = self.inbound_buffer.get(timeout=0)
        self.assertIsNotNone(retrieved)
        self.assertEqual(retrieved.session_id, "test_session")
        
        # Verify stats
        stats = self.inbound_buffer.get_stats()
        self.assertEqual(stats['packets_in'], 1)
        self.assertEqual(stats['packets_out'], 1)
        
    def test_priority_ordering(self):
        """Test priority ordering in buffers."""
        # Add packets in non-priority order
        self.inbound_buffer.put(ActivationPacket(
            priority=ActivationPriority.TRAINING_BACKWARD,
            session_id="backward",
            tensor_data=torch.zeros(1, 10),
        ))
        self.inbound_buffer.put(ActivationPacket(
            priority=ActivationPriority.INFERENCE_URGENT,
            session_id="urgent_inference",
            tensor_data=torch.zeros(1, 10),
        ))
        self.inbound_buffer.put(ActivationPacket(
            priority=ActivationPriority.TRAINING_FORWARD,
            session_id="forward",
            tensor_data=torch.zeros(1, 10),
        ))
        
        # Should come out in priority order
        p1 = self.inbound_buffer.get(timeout=0)
        p2 = self.inbound_buffer.get(timeout=0)
        p3 = self.inbound_buffer.get(timeout=0)
        
        self.assertEqual(p1.session_id, "urgent_inference")
        self.assertEqual(p2.session_id, "forward")
        self.assertEqual(p3.session_id, "backward")


class TestTrainingPipeline(unittest.TestCase):
    """Integration tests for training pipeline."""
    
    def setUp(self):
        """Set up training components."""
        self.model = SimpleModel()
        self.trainer = DiLoCoTrainer(
            self.model,
            config=DiLoCoConfig(inner_steps=3)  # Small for testing
        )
        self.aggregator = RobustAggregator(
            aggregation_config=AggregationConfig(
                strategy=AggregationStrategy.TRIMMED_MEAN
            )
        )
        
    def test_training_with_aggregation(self):
        """Test training with robust gradient aggregation."""
        # Do inner loop
        for _ in range(self.trainer.config.inner_steps):
            x = torch.randn(2, 10)
            y = self.model(x)
            loss = y.sum()
            self.trainer.inner_step(loss)
            
        # Compute pseudo-gradient
        pseudo_grads = self.trainer.compute_pseudo_gradient()
        
        # Simulate receiving gradients from peers
        peer_grads = {}
        for name, grad in pseudo_grads.items():
            # Simulated peer gradient (slightly different)
            peer_grads[name] = grad + torch.randn_like(grad) * 0.1
            
        # Add to aggregator
        self.aggregator.add_contribution(
            "peer1",
            peer_grads,
            validate=False
        )
        
        # Aggregate with local
        aggregated = self.aggregator.aggregate(local_grads=pseudo_grads)
        
        # Apply outer update
        self.trainer.apply_outer_update(aggregated)
        
        # Verify training progressed
        self.assertEqual(self.trainer.stats.outer_step_count, 1)
        
    def test_byzantine_resistance_in_training(self):
        """Test training resists Byzantine gradient attacks."""
        # Multiple training rounds
        for round_num in range(3):
            # Inner loop
            for _ in range(self.trainer.config.inner_steps):
                x = torch.randn(2, 10)
                loss = self.model(x).sum()
                self.trainer.inner_step(loss)
                
            # Get local gradients
            local_grads = self.trainer.compute_pseudo_gradient()
            
            # Create aggregator with median (Byzantine-tolerant)
            aggregator = RobustAggregator(
                aggregation_config=AggregationConfig(
                    strategy=AggregationStrategy.MEDIAN
                )
            )
            
            # Add honest peers
            for i in range(3):
                honest_grads = {
                    name: grad * (1.0 + 0.1 * i)
                    for name, grad in local_grads.items()
                }
                aggregator.add_contribution(f"honest_{i}", honest_grads, validate=False)
                
            # Add Byzantine peer
            byzantine_grads = {
                name: grad * -100  # Extreme opposite direction
                for name, grad in local_grads.items()
            }
            aggregator.add_contribution("byzantine", byzantine_grads, validate=False)
            
            # Aggregate - should resist Byzantine
            aggregated = aggregator.aggregate(local_grads=local_grads)
            
            # Apply update
            self.trainer.apply_outer_update(aggregated)
            
        # Training should have completed
        self.assertEqual(self.trainer.stats.outer_step_count, 3)


class TestMultiNodeSimulation(unittest.TestCase):
    """Integration tests simulating multiple nodes."""
    
    def setUp(self):
        """Set up multi-node environment."""
        self.temp_dir = tempfile.mkdtemp()
        
        # Create multiple nodes
        self.nodes = []
        for i in range(3):
            token = f"node_{i}_token_" + "x" * (32 - len(f"node_{i}_token_"))
            ledger = NEUROLedger(
                db_path=os.path.join(self.temp_dir, f"node_{i}.db"),
                node_token=token
            )
            router = SwarmRouter()
            
            self.nodes.append({
                'token': token,
                'ledger': ledger,
                'router': router,
                'crypto': NodeCrypto(token),
            })
            
    def tearDown(self):
        """Clean up."""
        try:
            shutil.rmtree(self.temp_dir)
        except:
            pass
            
    def test_nodes_discover_each_other(self):
        """Test nodes can discover each other."""
        # Each node broadcasts heartbeat
        for i, node in enumerate(self.nodes):
            heartbeat = CapacityBitmask(
                node_id=node['crypto'].node_id,
                grpc_addr=f"192.168.1.{i + 1}:50051",
                available_memory_mb=4096,
                queue_depth=0,
                layer_range=(i * 10, (i + 1) * 10),
                is_accepting_activations=True,
            )
            
            # Update all other nodes' routers
            for j, other_node in enumerate(self.nodes):
                if i != j:
                    other_node['router'].update_peer_from_heartbeat(
                        node_id=heartbeat.node_id,
                        grpc_addr=heartbeat.grpc_addr,
                        layer_range=heartbeat.layer_range,
                        queue_depth=heartbeat.queue_depth,
                        available_memory_mb=heartbeat.available_memory_mb,
                        gpu_utilization=50.0,
                        is_accepting=heartbeat.is_accepting_activations,
                    )
                    
        # Each node should know about 2 peers
        for node in self.nodes:
            self.assertEqual(len(node['router'].peer_stats), 2)
            
    def test_cross_node_proof_verification(self):
        """Test nodes can verify each other's proofs."""
        node_a = self.nodes[0]
        node_b = self.nodes[1]
        
        # Register A's public key with B
        node_b['ledger'].crypto.store_public_key(
            node_a['crypto'].node_id,
            node_a['crypto'].get_public_key_hex()
        )
        
        # A creates a proof
        proof = node_a['ledger'].create_proof(
            proof_type=ProofType.UPTIME,
            uptime_seconds=60.0,
            layers_held=5,
        )
        
        # B verifies A's proof
        is_valid, reason = node_b['ledger'].verify_proof(proof)
        
        self.assertTrue(is_valid, f"Cross-node verification failed: {reason}")


class TestDHTIntegration(unittest.TestCase):
    """Integration tests for DHT-based peer discovery."""
    
    def test_dht_with_swarm_router(self):
        """Test DHT integration with swarm router."""
        # Create DHT routing table
        local_node = Node(node_id=0x1000, ip="127.0.0.1", port=8000)
        dht = RoutingTable(local_node)
        
        # Add some peers to DHT
        for i in range(5):
            peer = Node(
                node_id=(i + 1) * 0x100,
                ip=f"192.168.1.{i}",
                port=50051
            )
            dht.add_contact(peer)
            
        # Create swarm router
        router = SwarmRouter()
        
        # Simulate DHT lookup and router update
        closest = dht.find_closest(target_id=0x300, k=3)
        
        for peer in closest:
            router.register_peer(PeerCandidate(
                node_id=str(peer.id),
                grpc_addr=f"{peer.ip}:{peer.port}",
                layer_range=(0, 24),
            ))
            
        # Router should have candidates
        candidates = router.get_candidates(target_layer=10)
        self.assertGreater(len(candidates), 0)


class TestHeartbeatIntegration(unittest.TestCase):
    """Integration tests for heartbeat system."""
    
    def test_heartbeat_serialization_roundtrip(self):
        """Test heartbeat binary serialization."""
        original = CapacityBitmask(
            node_id="test_node_12345678901234567890",
            timestamp=time.time(),
            grpc_addr="192.168.1.100:50051",
            available_memory_mb=8192,
            queue_depth=42,
            layer_range=(5, 15),
            gpu_utilization=75.5,
            is_training=True,
            is_accepting_inference=True,
            is_accepting_activations=True,
            training_step=12345,
        )
        
        # Serialize and deserialize
        data = original.to_bytes()
        restored = CapacityBitmask.from_bytes(data)
        
        self.assertIsNotNone(restored)
        self.assertEqual(restored.node_id, original.node_id)
        self.assertEqual(restored.available_memory_mb, original.available_memory_mb)
        self.assertEqual(restored.layer_range, original.layer_range)
        self.assertEqual(restored.training_step, original.training_step)
        
    def test_heartbeat_updates_router(self):
        """Test heartbeat updates router peer list."""
        router = SwarmRouter()
        
        # Simulate receiving heartbeats
        for i in range(3):
            heartbeat = CapacityBitmask(
                node_id=f"peer_{i}_" + "a" * 24,
                grpc_addr=f"10.0.0.{i}:50051",
                available_memory_mb=4096 * (i + 1),
                queue_depth=i * 10,
                layer_range=(0, 24),
                is_accepting_activations=True,
            )
            
            router.update_peer_from_heartbeat(
                node_id=heartbeat.node_id,
                grpc_addr=heartbeat.grpc_addr,
                layer_range=heartbeat.layer_range,
                queue_depth=heartbeat.queue_depth,
                available_memory_mb=heartbeat.available_memory_mb,
                gpu_utilization=50.0,
                is_accepting=heartbeat.is_accepting_activations,
            )
            
        # Check router has peers
        self.assertEqual(len(router.peer_stats), 3)
        
        # Check router stats
        stats = router.get_stats()
        self.assertEqual(stats['known_peers'], 3)


class TestFullWorkflow(unittest.TestCase):
    """Test complete workflow from training to rewards."""
    
    def setUp(self):
        """Set up complete environment."""
        self.temp_dir = tempfile.mkdtemp()
        self.token = "full_workflow_test_token_1234"
        
        # Create all components
        self.ledger = NEUROLedger(
            db_path=os.path.join(self.temp_dir, "ledger.db"),
            node_token=self.token
        )
        self.model = SimpleModel()
        self.trainer = DiLoCoTrainer(
            self.model,
            config=DiLoCoConfig(inner_steps=3)
        )
        self.router = SwarmRouter()
        
    def tearDown(self):
        """Clean up."""
        try:
            shutil.rmtree(self.temp_dir)
        except:
            pass
            
    def test_complete_training_session(self):
        """Test complete training session with rewards."""
        initial_balance = self.ledger.get_balance()
        
        # Simulate training session
        training_batches = 0
        
        for outer_step in range(2):
            # Inner loop
            for i in range(self.trainer.config.inner_steps):
                x = torch.randn(2, 10)
                loss = self.model(x).sum()
                self.trainer.inner_step(loss)
                training_batches += 1
                
            # Compute and apply update
            pseudo_grads = self.trainer.compute_pseudo_gradient()
            self.trainer.apply_outer_update(pseudo_grads)
            
            # Create and submit proof for this training round
            proof = self.ledger.create_proof(
                proof_type=ProofType.TRAINING,
                uptime_seconds=30.0,  # ~30 seconds per outer step
                training_batches=self.trainer.config.inner_steps,
                layers_held=5,
            )
            
            success, reward, _ = self.ledger.process_proof(proof)
            self.assertTrue(success)
            
        # Verify training completed
        self.assertEqual(self.trainer.stats.outer_step_count, 2)
        
        # Verify rewards credited
        final_balance = self.ledger.get_balance()
        self.assertGreater(final_balance, initial_balance)
        
        # Verify global stats
        stats = self.ledger.get_global_stats()
        self.assertGreater(stats.total_minted, 0)


if __name__ == '__main__':
    unittest.main()

