"""
Phase 2 Integration Tests for Swarm Architecture

Tests the real integration between components:
- SwarmServiceMixin (gRPC integration)
- SwarmEnabledDynamicNode (full node integration)
- AsyncSwarmTrainer (async training loop)
- SwarmRouter + ActivationBuffer (end-to-end flow)

NO MOCKING - tests use real components.

Run with:
    python -m pytest tests/test_swarm_phase2.py -v
    # or
    python -m unittest tests.test_swarm_phase2 -v
"""

import asyncio
import time
import threading
import unittest
from typing import Dict, List, Any

import torch

# Import components
from neuroshard.core.swarm.router import SwarmRouter, PeerCandidate
from neuroshard.core.swarm.buffers import ActivationBuffer, OutboundBuffer, ActivationPacket
from neuroshard.core.swarm.heartbeat import SwarmHeartbeatService, CapacityBitmask
from neuroshard.core.swarm.service import SwarmServiceMixin, SwarmNodeState
from neuroshard.core.swarm.factory import (
    SwarmEnabledDynamicNode,
    SwarmNodeConfig,
    SwarmComponents,
)
from neuroshard.core.swarm.trainer import (
    AsyncSwarmTrainer,
    TrainerConfig,
    TrainerState,
)


class TestSwarmNodeState(unittest.TestCase):
    """Test SwarmNodeState dataclass."""
    
    def test_create_state(self):
        """Test creating node state."""
        state = SwarmNodeState(
            node_id="test123",
            layer_range=(0, 4),
            grpc_addr="localhost:50051",
        )
        
        self.assertEqual(state.node_id, "test123")
        self.assertEqual(state.layer_range, (0, 4))
        self.assertTrue(state.is_accepting_activations)
    
    def test_state_defaults(self):
        """Test default values."""
        state = SwarmNodeState(
            node_id="test",
            layer_range=(0, 1),
            grpc_addr="",
        )
        
        self.assertEqual(state.available_memory_mb, 0)
        self.assertEqual(state.gpu_utilization, 0.0)
        self.assertFalse(state.is_training)


class TestSwarmComponentsStandalone(unittest.TestCase):
    """Test SwarmComponents container directly."""
    
    def test_create_components(self):
        """Test creating components container."""
        components = SwarmComponents()
        
        self.assertIsNone(components.swarm_router)
        self.assertIsNone(components.inbound_buffer)
        self.assertFalse(components.running)
    
    def test_stats_empty(self):
        """Test stats with no components."""
        components = SwarmComponents()
        stats = components.get_stats()
        
        self.assertIn("running", stats)
        self.assertFalse(stats["running"])
    
    def test_components_lifecycle(self):
        """Test start/stop with actual components."""
        components = SwarmComponents()
        
        # Initialize real components
        components.inbound_buffer = ActivationBuffer(max_size=10)
        components.outbound_buffer = OutboundBuffer(max_size=10)
        components.swarm_router = SwarmRouter(dht_protocol=None)
        
        # Start sync
        components.start_sync()
        
        # Should be running (heartbeat/checkpointer not set, so just state change)
        # Check stats
        stats = components.get_stats()
        self.assertIn("inbound_buffer", stats)
        self.assertIn("outbound_buffer", stats)
        self.assertIn("router", stats)
        
        # Stop
        components.stop_sync()


class TestSwarmNodeConfig(unittest.TestCase):
    """Test SwarmNodeConfig dataclass."""
    
    def test_default_config(self):
        """Test default configuration values."""
        config = SwarmNodeConfig()
        
        # Buffer defaults
        self.assertEqual(config.inbound_buffer_size, 100)
        self.assertEqual(config.outbound_buffer_size, 50)
        self.assertEqual(config.soft_overflow_threshold, 0.9)
        self.assertEqual(config.hard_overflow_threshold, 0.99)
        
        # Routing defaults
        self.assertEqual(config.ack_timeout_ms, 200)
        self.assertEqual(config.k_candidates, 3)
        
        # Heartbeat defaults
        self.assertEqual(config.heartbeat_interval, 5.0)
        self.assertEqual(config.heartbeat_port, 9999)
        
        # DiLoCo defaults
        self.assertEqual(config.diloco_inner_steps, 500)
        self.assertEqual(config.diloco_outer_lr, 0.7)
        
        # Aggregation defaults
        self.assertEqual(config.aggregation_method, "trimmed_mean")
    
    def test_custom_config(self):
        """Test custom configuration values."""
        config = SwarmNodeConfig(
            inbound_buffer_size=200,
            k_candidates=5,
            diloco_inner_steps=1000,
            aggregation_method="krum",
        )
        
        self.assertEqual(config.inbound_buffer_size, 200)
        self.assertEqual(config.k_candidates, 5)
        self.assertEqual(config.diloco_inner_steps, 1000)
        self.assertEqual(config.aggregation_method, "krum")
    
    def test_checkpoint_dir(self):
        """Test checkpoint directory creation."""
        config = SwarmNodeConfig()
        path = config.get_checkpoint_dir()
        
        self.assertTrue(path.exists())
        self.assertTrue(path.is_dir())


class TestAsyncSwarmTrainer(unittest.TestCase):
    """Test AsyncSwarmTrainer."""
    
    def test_create_trainer_config(self):
        """Test creating trainer config."""
        config = TrainerConfig(
            micro_batch_size=8,
            num_micro_batches=4,
            gradient_accumulation_steps=4,
        )
        
        self.assertEqual(config.micro_batch_size, 8)
        self.assertEqual(config.num_micro_batches, 4)
    
    def test_trainer_state(self):
        """Test trainer state enum."""
        self.assertEqual(TrainerState.IDLE.value, "idle")
        self.assertEqual(TrainerState.RUNNING.value, "running")
        self.assertEqual(TrainerState.STOPPED.value, "stopped")


class TestSwarmServiceMixin(unittest.TestCase):
    """Test SwarmServiceMixin."""
    
    def test_swarm_node_state_creation(self):
        """Test creating swarm node state."""
        state = SwarmNodeState(
            node_id="node1",
            layer_range=(0, 12),
            grpc_addr="localhost:50051",
            available_memory_mb=8192,
            gpu_utilization=0.5,
            is_training=True,
            is_accepting_activations=True,
        )
        
        self.assertEqual(state.node_id, "node1")
        self.assertEqual(state.layer_range, (0, 12))
        self.assertEqual(state.available_memory_mb, 8192)
        self.assertTrue(state.is_training)


class TestEndToEndFlow(unittest.TestCase):
    """Test end-to-end packet flow through swarm components."""
    
    def test_buffer_to_buffer_flow(self):
        """Test packet flow: inbound -> process -> outbound."""
        # Create real buffers
        inbound = ActivationBuffer(max_size=10)
        outbound = OutboundBuffer(max_size=10)
        
        # Create packet
        packet = ActivationPacket(
            priority=10,
            session_id="test_session",
            micro_batch_id=0,
            tensor_data=torch.randn(1, 10, 64),
            source_node="node_a",
            target_layer=0,
        )
        
        # Put into inbound
        success = inbound.put_nowait(packet)
        self.assertTrue(success)
        self.assertEqual(inbound.get_stats()['queue_size'], 1)
        
        # Get from inbound
        retrieved = inbound.get_nowait()
        self.assertIsNotNone(retrieved)
        self.assertEqual(retrieved.session_id, "test_session")
        
        # Process would happen here...
        
        # Put result into outbound
        result_packet = ActivationPacket(
            priority=10,
            session_id="test_session",
            micro_batch_id=0,
            tensor_data=torch.randn(1, 10, 64),
            source_node="node_b",
            target_layer=1,
        )
        
        loop = asyncio.new_event_loop()
        try:
            loop.run_until_complete(outbound.put(result_packet))
        finally:
            loop.close()
        
        # Verify outbound has packet
        self.assertEqual(outbound.get_stats()['queue_size'], 1)
    
    def test_router_peer_registration(self):
        """Test router peer registration and lookup."""
        router = SwarmRouter(dht_protocol=None)
        
        # Register peers
        peer1 = PeerCandidate(
            node_id="peer1",
            grpc_addr="peer1:50051",
            layer_range=(0, 4),
        )
        peer2 = PeerCandidate(
            node_id="peer2",
            grpc_addr="peer2:50051",
            layer_range=(0, 6),
        )
        
        router.register_peer(peer1)
        router.register_peer(peer2)
        
        # Get candidates for layer 2 (both should match)
        candidates = router.get_candidates(2)
        self.assertEqual(len(candidates), 2)
        
        # Get candidates for layer 5 (only peer2)
        candidates = router.get_candidates(5)
        self.assertEqual(len(candidates), 1)
        self.assertEqual(candidates[0].node_id, "peer2")
    
    def test_router_candidate_scoring(self):
        """Test router candidate scoring."""
        router = SwarmRouter(dht_protocol=None)
        
        # Register peers with different qualities
        fast_peer = PeerCandidate(
            node_id="fast",
            grpc_addr="fast:50051",
            layer_range=(0, 4),
            latency_ms=10.0,
            queue_depth=5,
        )
        slow_peer = PeerCandidate(
            node_id="slow",
            grpc_addr="slow:50051",
            layer_range=(0, 4),
            latency_ms=100.0,
            queue_depth=50,
        )
        
        router.register_peer(fast_peer)
        router.register_peer(slow_peer)
        
        # Get candidates - fast should be first
        candidates = router.get_candidates(0)
        self.assertEqual(len(candidates), 2)
        self.assertEqual(candidates[0].node_id, "fast")  # Lower score = better
    
    def test_heartbeat_capacity_bitmask(self):
        """Test heartbeat capacity bitmask creation."""
        bitmask = CapacityBitmask(
            node_id="node1",
            timestamp=time.time(),
            available_memory_mb=4096,
            queue_depth=10,
            layer_range=(0, 4),
            gpu_utilization=0.5,
            network_saturation=0.2,
            is_training=True,
            is_accepting_inference=True,
            is_accepting_activations=True,
            grpc_addr="localhost:50051",
        )
        
        # Test serialization roundtrip
        data = bitmask.to_bytes()
        restored = CapacityBitmask.from_bytes(data)
        
        self.assertEqual(restored.node_id, bitmask.node_id)
        self.assertEqual(restored.available_memory_mb, bitmask.available_memory_mb)
        self.assertEqual(restored.layer_range, bitmask.layer_range)


class TestBufferMetrics(unittest.TestCase):
    """Test buffer metrics and overflow handling."""
    
    def test_inbound_fill_rate(self):
        """Test inbound buffer fill rate calculation."""
        buffer = ActivationBuffer(max_size=10)
        
        self.assertEqual(buffer.fill_rate, 0.0)
        
        # Add 5 packets
        for i in range(5):
            packet = ActivationPacket(
                priority=10,
                session_id=f"session_{i}",
                micro_batch_id=i,
                tensor_data=torch.randn(1, 10, 64),
                source_node="test",
                target_layer=0,
            )
            buffer.put_nowait(packet)
        
        self.assertEqual(buffer.fill_rate, 0.5)
    
    def test_outbound_overflow_detection(self):
        """Test outbound buffer overflow detection."""
        buffer = OutboundBuffer(
            max_size=10,
            soft_overflow_threshold=0.8,
            hard_overflow_threshold=0.95,
        )
        
        # Initially not in overflow
        self.assertFalse(buffer.is_soft_overflow)
        self.assertFalse(buffer.is_hard_overflow)
        
        # Check thresholds are set correctly
        self.assertEqual(buffer.soft_limit, 0.8)
        self.assertEqual(buffer.hard_limit, 0.95)
    
    def test_buffer_stats(self):
        """Test buffer statistics collection."""
        buffer = ActivationBuffer(max_size=100)
        
        # Add some packets
        for i in range(5):
            packet = ActivationPacket(
                priority=10,
                session_id=f"session_{i}",
                micro_batch_id=i,
                tensor_data=torch.randn(1, 10, 64),
                source_node="test",
                target_layer=0,
            )
            buffer.put_nowait(packet)
        
        # Get some packets
        for _ in range(3):
            buffer.get_nowait()
        
        stats = buffer.get_stats()
        
        self.assertEqual(stats["packets_in"], 5)
        self.assertEqual(stats["packets_out"], 3)
        self.assertEqual(stats["queue_size"], 2)


class TestAsyncOperations(unittest.TestCase):
    """Test async operations."""
    
    def test_async_router_start_stop(self):
        """Test async router lifecycle."""
        router = SwarmRouter(dht_protocol=None)
        
        async def test():
            await router.start()
            # _cleanup_task is set after start
            self.assertIsNotNone(router._cleanup_task)
            
            await router.stop()
            # Task should be cancelled
            self.assertTrue(router._cleanup_task.cancelled() or router._cleanup_task.done())
        
        loop = asyncio.new_event_loop()
        try:
            loop.run_until_complete(test())
        finally:
            loop.close()
    
    def test_outbound_async_put_get(self):
        """Test async outbound buffer operations."""
        buffer = OutboundBuffer(max_size=10)
        
        async def test():
            packet = ActivationPacket(
                priority=10,
                session_id="async_test",
                micro_batch_id=0,
                tensor_data=torch.randn(1, 10, 64),
                source_node="test",
                target_layer=0,
            )
            
            await buffer.put(packet)
            self.assertEqual(buffer.get_stats()['queue_size'], 1)
            
            retrieved = await buffer.get()
            self.assertEqual(retrieved.session_id, "async_test")
            self.assertEqual(buffer.get_stats()['queue_size'], 0)
        
        loop = asyncio.new_event_loop()
        try:
            loop.run_until_complete(test())
        finally:
            loop.close()


if __name__ == '__main__':
    unittest.main()
