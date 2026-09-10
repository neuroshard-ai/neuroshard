"""
Unit tests for Swarm Architecture Components

Tests:
- SwarmRouter (multipath routing, failover)
- ActivationBuffer (priority queue, soft overflow)
- OutboundBuffer (async send, pressure checking)
- SwarmHeartbeat (capacity broadcast)
- ComputeEngine (soft overflow handling)
- NATTraversal (STUN client, hole punching)
"""

import asyncio
import struct
import time
import unittest
from unittest.mock import Mock, MagicMock, AsyncMock, patch
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Test imports
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    

class TestPeerCandidate(unittest.TestCase):
    """Tests for PeerCandidate scoring."""
    
    def setUp(self):
        from neuroshard.core.swarm.router import PeerCandidate
        self.PeerCandidate = PeerCandidate
        
    def test_score_default_weights(self):
        """Test default scoring (40% latency, 60% queue)."""
        peer = self.PeerCandidate(
            node_id="test1",
            grpc_addr="localhost:50051",
            layer_range=(0, 10),
            latency_ms=100,  # 100/500 = 0.2 normalized
            queue_depth=50,   # 50/100 = 0.5 normalized
        )
        
        # Expected: 0.4 * 0.2 + 0.6 * 0.5 = 0.08 + 0.3 = 0.38
        score = peer.score()
        self.assertAlmostEqual(score, 0.38, places=2)
        
    def test_score_with_failures(self):
        """Test score penalty for failures."""
        peer = self.PeerCandidate(
            node_id="test2",
            grpc_addr="localhost:50052",
            layer_range=(0, 10),
            latency_ms=100,
            queue_depth=0,
            consecutive_failures=2,  # 0.1 * 2^2 = 0.4 penalty
        )
        
        base_score = 0.4 * 0.2 + 0.6 * 0.0  # 0.08
        penalty = 0.1 * (2 ** 2)  # 0.4
        expected = base_score + penalty
        
        score = peer.score()
        self.assertAlmostEqual(score, expected, places=2)
        
    def test_covers_layer(self):
        """Test layer range checking."""
        peer = self.PeerCandidate(
            node_id="test3",
            grpc_addr="localhost:50053",
            layer_range=(5, 15),
        )
        
        self.assertFalse(peer.covers_layer(4))
        self.assertTrue(peer.covers_layer(5))
        self.assertTrue(peer.covers_layer(10))
        self.assertTrue(peer.covers_layer(14))
        self.assertFalse(peer.covers_layer(15))
        
    def test_success_rate(self):
        """Test success rate calculation."""
        peer = self.PeerCandidate(
            node_id="test4",
            grpc_addr="localhost:50054",
            layer_range=(0, 10),
        )
        
        # Initial state: assume 100% success
        self.assertEqual(peer.success_rate, 1.0)
        
        # Record some results
        peer.record_success(50.0)
        peer.record_success(60.0)
        peer.record_failure()
        
        self.assertEqual(peer.total_requests, 3)
        self.assertEqual(peer.successful_requests, 2)
        self.assertAlmostEqual(peer.success_rate, 2/3, places=2)


class TestActivationBuffer(unittest.TestCase):
    """Tests for ActivationBuffer."""
    
    def setUp(self):
        from neuroshard.core.swarm.buffers import (
            ActivationBuffer,
            ActivationPacket,
            ActivationPriority,
        )
        self.ActivationBuffer = ActivationBuffer
        self.ActivationPacket = ActivationPacket
        self.ActivationPriority = ActivationPriority
        
    def _create_packet(self, priority, micro_batch_id=0):
        """Helper to create test packet."""
        if TORCH_AVAILABLE:
            tensor = torch.zeros(1, 10)
        else:
            tensor = None
            
        return self.ActivationPacket(
            priority=priority,
            sequence_num=micro_batch_id,  # Use micro_batch_id for unique ordering
            session_id="test",
            micro_batch_id=micro_batch_id,
            tensor_data=tensor,
            source_node="node1",
            target_layer=0,
        )
        
    def test_priority_ordering(self):
        """Test packets are dequeued by priority."""
        buffer = self.ActivationBuffer(max_size=10)
        
        # Add in non-priority order
        buffer.put(self._create_packet(self.ActivationPriority.TRAINING_BACKWARD))
        buffer.put(self._create_packet(self.ActivationPriority.INFERENCE_URGENT))
        buffer.put(self._create_packet(self.ActivationPriority.TRAINING_FORWARD))
        
        # Should come out in priority order
        p1 = buffer.get(timeout=0)
        p2 = buffer.get(timeout=0)
        p3 = buffer.get(timeout=0)
        
        self.assertEqual(p1.priority, self.ActivationPriority.INFERENCE_URGENT)
        self.assertEqual(p2.priority, self.ActivationPriority.TRAINING_FORWARD)
        self.assertEqual(p3.priority, self.ActivationPriority.TRAINING_BACKWARD)
        
    def test_fill_rate(self):
        """Test fill rate calculation."""
        buffer = self.ActivationBuffer(max_size=10)
        
        self.assertEqual(buffer.fill_rate, 0.0)
        
        for i in range(5):
            buffer.put(self._create_packet(10, micro_batch_id=i))
            
        self.assertEqual(buffer.fill_rate, 0.5)
        
        for i in range(5, 10):
            buffer.put(self._create_packet(10, micro_batch_id=i))
            
        self.assertEqual(buffer.fill_rate, 1.0)
        
    def test_is_starved(self):
        """Test starvation detection."""
        buffer = self.ActivationBuffer(max_size=100)
        
        # Empty buffer is starved
        self.assertTrue(buffer.is_starved)
        
        # Add 5 packets (5%)
        for i in range(5):
            buffer.put(self._create_packet(10, i))
            
        # Still starved (< 10%)
        self.assertTrue(buffer.is_starved)
        
        # Add more
        for i in range(5, 15):
            buffer.put(self._create_packet(10, i))
            
        # No longer starved (15%)
        self.assertFalse(buffer.is_starved)
        
    def test_is_backpressured(self):
        """Test backpressure detection."""
        buffer = self.ActivationBuffer(max_size=10)
        
        # Empty buffer is not backpressured
        self.assertFalse(buffer.is_backpressured)
        
        # Fill to 90%
        for i in range(9):
            buffer.put(self._create_packet(10, i))
            
        # Now backpressured
        self.assertTrue(buffer.is_backpressured)
        
    def test_put_nowait_full(self):
        """Test non-blocking put when full."""
        buffer = self.ActivationBuffer(max_size=2)
        
        buffer.put(self._create_packet(10, 0))
        buffer.put(self._create_packet(10, 1))
        
        # Buffer is full, put_nowait should return False
        result = buffer.put_nowait(self._create_packet(10, 2))
        self.assertFalse(result)
        
    def test_get_nowait_empty(self):
        """Test non-blocking get when empty."""
        buffer = self.ActivationBuffer(max_size=10)
        
        # Empty buffer, get_nowait should return None
        result = buffer.get_nowait()
        self.assertIsNone(result)
        
    def test_statistics(self):
        """Test statistics tracking."""
        buffer = self.ActivationBuffer(max_size=10)
        
        # Add and remove some packets
        for i in range(5):
            buffer.put(self._create_packet(10, i))
            
        for i in range(3):
            buffer.get(timeout=0)
            
        stats = buffer.get_stats()
        
        self.assertEqual(stats["packets_in"], 5)
        self.assertEqual(stats["packets_out"], 3)
        self.assertEqual(stats["queue_size"], 2)


class TestOutboundBuffer(unittest.TestCase):
    """Tests for OutboundBuffer."""
    
    def setUp(self):
        from neuroshard.core.swarm.buffers import OutboundBuffer, ActivationPacket
        self.OutboundBuffer = OutboundBuffer
        self.ActivationPacket = ActivationPacket
        
    def _create_packet(self, micro_batch_id=0):
        """Helper to create test packet."""
        if TORCH_AVAILABLE:
            tensor = torch.zeros(1, 10)
        else:
            tensor = None
            
        return self.ActivationPacket(
            priority=10,
            sequence_num=micro_batch_id,
            session_id="test",
            micro_batch_id=micro_batch_id,
            tensor_data=tensor,
            source_node="node1",
            target_layer=0,
        )
        
    def test_check_pressure_ok(self):
        """Test pressure check when buffer is not full."""
        buffer = self.OutboundBuffer(max_size=10)
        
        self.assertEqual(buffer.check_pressure(), "ok")
        
    def test_check_pressure_soft_overflow(self):
        """Test pressure check at soft overflow threshold."""
        buffer = self.OutboundBuffer(max_size=10)
        
        # Fill to 90%
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        for i in range(9):
            loop.run_until_complete(buffer.put(self._create_packet(i)))
            
        self.assertEqual(buffer.check_pressure(), "soft_overflow")
        
        loop.close()
        
    def test_check_pressure_hard_overflow(self):
        """Test pressure check at hard overflow threshold."""
        buffer = self.OutboundBuffer(max_size=10)
        
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        # Fill to 100%
        for i in range(10):
            loop.run_until_complete(buffer.put(self._create_packet(i)))
            
        self.assertEqual(buffer.check_pressure(), "hard_overflow")
        
        loop.close()


class TestCapacityBitmask(unittest.TestCase):
    """Tests for CapacityBitmask serialization."""
    
    def setUp(self):
        from neuroshard.core.swarm.heartbeat import CapacityBitmask
        self.CapacityBitmask = CapacityBitmask
        
    def test_serialization_roundtrip(self):
        """Test binary serialization and deserialization."""
        original = self.CapacityBitmask(
            node_id="test_node_123",
            timestamp=time.time(),
            grpc_addr="192.168.1.100:50051",
            available_memory_mb=8192,
            queue_depth=42,
            layer_range=(5, 15),
            gpu_utilization=75.5,
            network_saturation=30.0,
            is_training=True,
            is_accepting_inference=False,
            is_accepting_activations=True,
            training_step=12345,
        )
        
        # Serialize
        data = original.to_bytes()
        
        # Deserialize
        restored = self.CapacityBitmask.from_bytes(data)
        
        self.assertIsNotNone(restored)
        self.assertEqual(restored.node_id, "test_node_123")
        self.assertEqual(restored.grpc_addr, "192.168.1.100:50051")
        self.assertEqual(restored.available_memory_mb, 8192)
        self.assertEqual(restored.queue_depth, 42)
        self.assertEqual(restored.layer_range, (5, 15))
        self.assertEqual(int(restored.gpu_utilization), 75)  # uint8 truncation
        self.assertEqual(restored.is_training, True)
        self.assertEqual(restored.is_accepting_inference, False)
        self.assertEqual(restored.is_accepting_activations, True)
        self.assertEqual(restored.training_step, 12345)
        
    def test_json_roundtrip(self):
        """Test JSON serialization and deserialization."""
        original = self.CapacityBitmask(
            node_id="json_test",
            grpc_addr="localhost:50051",
            available_memory_mb=4096,
            queue_depth=10,
            layer_range=(0, 24),
        )
        
        json_str = original.to_json()
        restored = self.CapacityBitmask.from_json(json_str)
        
        self.assertIsNotNone(restored)
        self.assertEqual(restored.node_id, "json_test")
        self.assertEqual(restored.available_memory_mb, 4096)
        self.assertEqual(restored.layer_range, (0, 24))
        
    def test_invalid_magic_number(self):
        """Test that invalid magic number is rejected."""
        # Create invalid data with wrong magic number
        invalid_data = struct.pack('>I', 0xDEADBEEF) + b'\x00' * 100
        
        result = self.CapacityBitmask.from_bytes(invalid_data)
        self.assertIsNone(result)


class TestNATTraversal(unittest.TestCase):
    """Tests for NAT traversal components."""
    
    def setUp(self):
        from neuroshard.core.network.nat_traversal import (
            NATType, 
            PeerConnectivity,
            STUNClient,
        )
        self.NATType = NATType
        self.PeerConnectivity = PeerConnectivity
        self.STUNClient = STUNClient
        
    def test_nat_type_enum(self):
        """Test NAT type enumeration."""
        self.assertEqual(self.NATType.SYMMETRIC.value, "symmetric")
        self.assertEqual(self.NATType.FULL_CONE.value, "full_cone")
        
    def test_peer_connectivity_best_addr(self):
        """Test best address selection."""
        # With public address
        peer = self.PeerConnectivity(
            peer_id="test1",
            public_addr=("1.2.3.4", 5000),
            private_addrs=[("192.168.1.1", 5000)],
        )
        self.assertEqual(peer.best_addr, ("1.2.3.4", 5000))
        
        # Without public address
        peer2 = self.PeerConnectivity(
            peer_id="test2",
            private_addrs=[("192.168.1.1", 5000)],
        )
        self.assertEqual(peer2.best_addr, ("192.168.1.1", 5000))
        
        # No addresses
        peer3 = self.PeerConnectivity(peer_id="test3")
        self.assertIsNone(peer3.best_addr)
        
    def test_stun_client_create_request(self):
        """Test STUN binding request creation."""
        client = self.STUNClient()
        
        request, tid = client._create_binding_request()
        
        # Check request format
        self.assertEqual(len(request), 20)  # Header only, no attributes
        self.assertEqual(len(tid), 12)  # Transaction ID is 12 bytes
        
        # Unpack and verify
        msg_type, msg_len, cookie = struct.unpack('>HHI', request[:8])
        self.assertEqual(msg_type, 0x0001)  # Binding Request
        self.assertEqual(msg_len, 0)  # No attributes
        self.assertEqual(cookie, 0x2112A442)  # Magic cookie


class TestSwarmRouter(unittest.TestCase):
    """Tests for SwarmRouter."""
    
    def setUp(self):
        from neuroshard.core.swarm.router import SwarmRouter, PeerCandidate
        self.SwarmRouter = SwarmRouter
        self.PeerCandidate = PeerCandidate
        
    def test_get_candidates_from_cache(self):
        """Test candidate retrieval from local cache."""
        router = self.SwarmRouter()
        
        # Add peers to cache
        peer1 = self.PeerCandidate(
            node_id="peer1",
            grpc_addr="localhost:50051",
            layer_range=(0, 10),
            latency_ms=50,
            queue_depth=10,
        )
        peer2 = self.PeerCandidate(
            node_id="peer2",
            grpc_addr="localhost:50052",
            layer_range=(0, 10),
            latency_ms=100,
            queue_depth=20,
        )
        
        router.register_peer(peer1)
        router.register_peer(peer2)
        
        # Get candidates for layer 5
        candidates = router.get_candidates(5)
        
        self.assertEqual(len(candidates), 2)
        # peer1 should be first (better score)
        self.assertEqual(candidates[0].node_id, "peer1")
        
    def test_get_candidates_layer_filtering(self):
        """Test that only peers covering the layer are returned."""
        router = self.SwarmRouter()
        
        # Add peers with different layer ranges
        router.register_peer(self.PeerCandidate(
            node_id="peer1",
            grpc_addr="localhost:50051",
            layer_range=(0, 10),
        ))
        router.register_peer(self.PeerCandidate(
            node_id="peer2",
            grpc_addr="localhost:50052",
            layer_range=(10, 20),
        ))
        
        # Layer 5 should only return peer1
        candidates = router.get_candidates(5)
        self.assertEqual(len(candidates), 1)
        self.assertEqual(candidates[0].node_id, "peer1")
        
        # Layer 15 should only return peer2
        candidates = router.get_candidates(15)
        self.assertEqual(len(candidates), 1)
        self.assertEqual(candidates[0].node_id, "peer2")
        
    def test_update_peer_from_heartbeat(self):
        """Test peer update from heartbeat."""
        router = self.SwarmRouter()
        
        router.update_peer_from_heartbeat(
            node_id="new_peer",
            grpc_addr="192.168.1.1:50051",
            layer_range=(5, 15),
            queue_depth=25,
            available_memory_mb=4096,
            gpu_utilization=50.0,
            is_accepting=True,
        )
        
        self.assertIn("new_peer", router.peer_stats)
        peer = router.peer_stats["new_peer"]
        self.assertEqual(peer.queue_depth, 25)
        self.assertEqual(peer.available_memory_mb, 4096)


class TestComputeEngineStepOutcome(unittest.TestCase):
    """Tests for ComputeEngine step outcomes."""
    
    def setUp(self):
        from neuroshard.core.swarm.compute import StepOutcome, ComputeStats
        self.StepOutcome = StepOutcome
        self.ComputeStats = ComputeStats
        
    def test_step_outcomes(self):
        """Test step outcome enum values."""
        self.assertEqual(self.StepOutcome.SENT.value, "sent")
        self.assertEqual(self.StepOutcome.LOCAL_ONLY.value, "local_only")
        self.assertEqual(self.StepOutcome.DROPPED.value, "dropped")
        
    def test_compute_stats(self):
        """Test compute statistics tracking."""
        stats = self.ComputeStats()
        
        stats.total_steps = 100
        stats.local_only_steps = 5
        stats.dropped_steps = 2
        stats.total_compute_time_ms = 5000
        
        self.assertAlmostEqual(stats.local_only_rate, 0.05)
        self.assertAlmostEqual(stats.drop_rate, 0.02)
        self.assertAlmostEqual(stats.avg_compute_time_ms, 50.0)


class TestIntegration(unittest.TestCase):
    """Integration tests for swarm components working together."""
    
    @unittest.skipUnless(TORCH_AVAILABLE, "PyTorch not available")
    def test_buffer_flow(self):
        """Test packet flow through inbound -> compute -> outbound."""
        from neuroshard.core.swarm.buffers import (
            ActivationBuffer,
            OutboundBuffer,
            ActivationPacket,
            ActivationPriority,
        )
        
        inbound = ActivationBuffer(max_size=10)
        outbound = OutboundBuffer(max_size=10)
        
        # Create packet
        packet = ActivationPacket(
            priority=ActivationPriority.TRAINING_FORWARD,
            sequence_num=1,
            session_id="test_session",
            micro_batch_id=1,
            tensor_data=torch.randn(2, 768),
            source_node="node_a",
            target_layer=5,
        )
        
        # Put in inbound
        inbound.put(packet)
        self.assertEqual(inbound.fill_rate, 0.1)
        
        # Get from inbound
        retrieved = inbound.get(timeout=0)
        self.assertIsNotNone(retrieved)
        self.assertEqual(retrieved.session_id, "test_session")
        
        # Put in outbound
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        loop.run_until_complete(outbound.put(retrieved))
        loop.close()
        
        self.assertEqual(outbound.fill_rate, 0.1)
        
    def test_heartbeat_router_integration(self):
        """Test heartbeat service updating router."""
        from neuroshard.core.swarm.router import SwarmRouter
        from neuroshard.core.swarm.heartbeat import CapacityBitmask
        
        router = SwarmRouter()
        
        # Simulate receiving heartbeat
        heartbeat = CapacityBitmask(
            node_id="remote_peer",
            grpc_addr="10.0.0.5:50051",
            available_memory_mb=8192,
            queue_depth=15,
            layer_range=(0, 12),
            gpu_utilization=60.0,
            is_accepting_activations=True,
        )
        
        # Update router from heartbeat
        router.update_peer_from_heartbeat(
            node_id=heartbeat.node_id,
            grpc_addr=heartbeat.grpc_addr,
            layer_range=heartbeat.layer_range,
            queue_depth=heartbeat.queue_depth,
            available_memory_mb=heartbeat.available_memory_mb,
            gpu_utilization=heartbeat.gpu_utilization,
            is_accepting=heartbeat.is_accepting_activations,
        )
        
        # Router should now have this peer
        candidates = router.get_candidates(5)
        self.assertEqual(len(candidates), 1)
        self.assertEqual(candidates[0].node_id, "remote_peer")


if __name__ == '__main__':
    unittest.main()

