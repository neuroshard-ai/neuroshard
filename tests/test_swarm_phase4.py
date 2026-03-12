"""
Phase 4 Integration Tests - Swarm Runner Integration

Tests for:
1. SwarmNodeConfig configuration
2. SwarmComponents container
3. SwarmLogger structured logging
4. SwarmEnabledDynamicNode (requires real DynamicNeuroNode)

NO MOCKING - tests use real components.

Run with:
    python -m pytest tests/test_swarm_phase4.py -v
    # or
    python -m unittest tests.test_swarm_phase4 -v
"""

import unittest
import torch
import time
import threading
import logging
from pathlib import Path
import tempfile
import os

# Suppress logging during tests
logging.getLogger('neuroshard').setLevel(logging.WARNING)


class TestSwarmNodeConfig(unittest.TestCase):
    """Test SwarmNodeConfig dataclass."""
    
    def test_default_config(self):
        """Test default configuration values."""
        from neuroshard.core.swarm.factory import SwarmNodeConfig
        
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
        self.assertEqual(config.diloco_outer_momentum, 0.9)
        
        # Checkpointing defaults
        self.assertEqual(config.checkpoint_interval, 120)
        self.assertEqual(config.max_checkpoints, 5)
        
        # Aggregation defaults
        self.assertEqual(config.aggregation_method, "trimmed_mean")
        self.assertEqual(config.krum_f, 0)
        self.assertEqual(config.trimmed_mean_beta, 0.1)
        
        # Validation defaults
        self.assertEqual(config.cosine_threshold, 0.5)
        self.assertEqual(config.magnitude_ratio_threshold, 10.0)
    
    def test_custom_config(self):
        """Test custom configuration values."""
        from neuroshard.core.swarm.factory import SwarmNodeConfig
        
        config = SwarmNodeConfig(
            diloco_inner_steps=1000,
            k_candidates=5,
            aggregation_method="krum",
            checkpoint_interval=60,
        )
        
        self.assertEqual(config.diloco_inner_steps, 1000)
        self.assertEqual(config.k_candidates, 5)
        self.assertEqual(config.aggregation_method, "krum")
        self.assertEqual(config.checkpoint_interval, 60)
    
    def test_checkpoint_dir_creation(self):
        """Test checkpoint directory path resolution."""
        from neuroshard.core.swarm.factory import SwarmNodeConfig
        
        with tempfile.TemporaryDirectory() as tmpdir:
            config = SwarmNodeConfig(checkpoint_dir=os.path.join(tmpdir, "checkpoints"))
            
            path = config.get_checkpoint_dir()
            
            self.assertTrue(path.exists())
            self.assertTrue(path.is_dir())
    
    def test_default_checkpoint_dir(self):
        """Test default checkpoint directory uses home."""
        from neuroshard.core.swarm.factory import SwarmNodeConfig
        
        config = SwarmNodeConfig()
        path = config.get_checkpoint_dir()
        
        # Should be under ~/.neuroshard/checkpoints
        self.assertTrue(str(path).endswith("checkpoints"))
        self.assertTrue(path.exists())


class TestSwarmComponents(unittest.TestCase):
    """Test SwarmComponents container."""
    
    def test_create_empty_components(self):
        """Test creating empty components container."""
        from neuroshard.core.swarm.factory import SwarmComponents
        
        components = SwarmComponents()
        
        self.assertIsNone(components.swarm_router)
        self.assertIsNone(components.inbound_buffer)
        self.assertIsNone(components.outbound_buffer)
        self.assertIsNone(components.heartbeat_service)
        self.assertIsNone(components.compute_engine)
        self.assertIsNone(components.diloco_trainer)
        self.assertFalse(components.running)
    
    def test_components_stats_empty(self):
        """Test stats with no components."""
        from neuroshard.core.swarm.factory import SwarmComponents
        
        components = SwarmComponents()
        stats = components.get_stats()
        
        self.assertIn("running", stats)
        self.assertFalse(stats["running"])
        self.assertNotIn("inbound_buffer", stats)
    
    def test_components_with_buffers(self):
        """Test components with real buffers."""
        from neuroshard.core.swarm.factory import SwarmComponents
        from neuroshard.core.swarm.buffers import ActivationBuffer, OutboundBuffer
        
        components = SwarmComponents()
        components.inbound_buffer = ActivationBuffer(max_size=50)
        components.outbound_buffer = OutboundBuffer(max_size=25)
        
        stats = components.get_stats()
        
        self.assertIn("inbound_buffer", stats)
        self.assertIn("outbound_buffer", stats)
        self.assertEqual(stats["inbound_buffer"]["max_size"], 50)
        self.assertEqual(stats["outbound_buffer"]["max_size"], 25)
    
    def test_components_start_stop(self):
        """Test start/stop lifecycle."""
        from neuroshard.core.swarm.factory import SwarmComponents
        from neuroshard.core.swarm.buffers import ActivationBuffer, OutboundBuffer
        from neuroshard.core.swarm.router import SwarmRouter
        
        components = SwarmComponents()
        components.inbound_buffer = ActivationBuffer(max_size=10)
        components.outbound_buffer = OutboundBuffer(max_size=10)
        components.swarm_router = SwarmRouter(dht_protocol=None)
        
        # Start sync (no heartbeat/checkpointer)
        components.start_sync()
        
        # Stop
        components.stop_sync()


class TestSwarmLogger(unittest.TestCase):
    """Test SwarmLogger structured logging."""
    
    def test_create_logger(self):
        """Test creating swarm logger."""
        from neuroshard.core.swarm.logger import SwarmLogger, NodeRole
        
        logger = SwarmLogger(
            node_id="test123",
            role=NodeRole.WORKER,
        )
        
        self.assertEqual(logger.node_id, "test123")
        self.assertEqual(logger.role, NodeRole.WORKER)
    
    def test_logger_roles(self):
        """Test all logger roles."""
        from neuroshard.core.swarm.logger import NodeRole
        
        self.assertEqual(NodeRole.DRIVER.value, "DRIVER")
        self.assertEqual(NodeRole.WORKER.value, "WORKER")
        self.assertEqual(NodeRole.VALIDATOR.value, "VALIDATOR")
    
    def test_log_categories(self):
        """Test log categories."""
        from neuroshard.core.swarm.logger import LogCategory
        
        self.assertIn("swarm", [c.value for c in LogCategory])
        self.assertIn("diloco", [c.value for c in LogCategory])
        self.assertIn("training", [c.value for c in LogCategory])
    
    def test_logger_training_log(self):
        """Test logger training step logging."""
        from neuroshard.core.swarm.logger import SwarmLogger, NodeRole
        
        logger = SwarmLogger(
            node_id="test",
            role=NodeRole.WORKER,
        )
        
        # Log some training steps
        logger.log_training_step(round=1, loss=0.5, tokens=100, duration_ms=10)
        logger.log_training_step(round=2, loss=0.4, tokens=100, duration_ms=10)
    
    def test_get_swarm_logger(self):
        """Test global logger getter."""
        from neuroshard.core.swarm.logger import get_swarm_logger, init_swarm_logger, NodeRole
        
        # Initialize global logger
        init_swarm_logger("test_node", NodeRole.DRIVER)
        
        logger = get_swarm_logger()
        self.assertIsNotNone(logger)


class TestActivationBufferIntegration(unittest.TestCase):
    """Test activation buffer in integration scenarios."""
    
    def test_priority_ordering(self):
        """Test packets are processed by priority."""
        from neuroshard.core.swarm.buffers import ActivationBuffer, ActivationPacket, ActivationPriority
        
        buffer = ActivationBuffer(max_size=10)
        
        # Add packets with different priorities
        low_priority = ActivationPacket(
            priority=ActivationPriority.TRAINING_BACKWARD,
            session_id="low",
            micro_batch_id=0,
            tensor_data=torch.randn(1, 10),
            source_node="test",
            target_layer=0,
        )
        
        high_priority = ActivationPacket(
            priority=ActivationPriority.INFERENCE_URGENT,
            session_id="high",
            micro_batch_id=0,
            tensor_data=torch.randn(1, 10),
            source_node="test",
            target_layer=0,
        )
        
        buffer.put_nowait(low_priority)
        buffer.put_nowait(high_priority)
        
        # High priority should come out first
        first = buffer.get_nowait()
        self.assertEqual(first.session_id, "high")
        
        second = buffer.get_nowait()
        self.assertEqual(second.session_id, "low")
    
    def test_backpressure_detection(self):
        """Test backpressure detection at high fill rate."""
        from neuroshard.core.swarm.buffers import ActivationBuffer, ActivationPacket
        
        buffer = ActivationBuffer(max_size=10)
        
        # Fill to 90%
        for i in range(9):
            packet = ActivationPacket(
                priority=10,
                session_id=f"session_{i}",
                micro_batch_id=i,
                tensor_data=torch.randn(1, 10),
                source_node="test",
                target_layer=0,
            )
            buffer.put_nowait(packet)
        
        self.assertTrue(buffer.is_backpressured)
        self.assertFalse(buffer.is_starved)
    
    def test_starvation_detection(self):
        """Test starvation detection when empty."""
        from neuroshard.core.swarm.buffers import ActivationBuffer
        
        buffer = ActivationBuffer(max_size=10)
        
        self.assertTrue(buffer.is_starved)
        self.assertFalse(buffer.is_backpressured)


class TestSwarmRouterIntegration(unittest.TestCase):
    """Test swarm router integration scenarios."""
    
    def test_multi_peer_layer_coverage(self):
        """Test routing with multiple peers covering same layers."""
        from neuroshard.core.swarm.router import SwarmRouter, PeerCandidate
        
        router = SwarmRouter(dht_protocol=None)
        
        # Register multiple peers for same layer range
        for i in range(3):
            peer = PeerCandidate(
                node_id=f"peer_{i}",
                grpc_addr=f"peer{i}:50051",
                layer_range=(0, 6),
                latency_ms=10.0 + i * 5,  # Different latencies
                queue_depth=i * 2,
            )
            router.register_peer(peer)
        
        # Get candidates for layer 3
        candidates = router.get_candidates(3)
        
        # All 3 peers should be returned (they all cover layer 3)
        self.assertEqual(len(candidates), 3)
        
        # All expected peers should be in the list
        peer_ids = {c.node_id for c in candidates}
        self.assertEqual(peer_ids, {"peer_0", "peer_1", "peer_2"})
    
    def test_layer_filtering(self):
        """Test peers are filtered by layer coverage."""
        from neuroshard.core.swarm.router import SwarmRouter, PeerCandidate
        
        router = SwarmRouter(dht_protocol=None)
        
        # Peer covering layers 0-4
        peer1 = PeerCandidate(
            node_id="peer_low",
            grpc_addr="peer_low:50051",
            layer_range=(0, 4),
        )
        
        # Peer covering layers 4-8
        peer2 = PeerCandidate(
            node_id="peer_high",
            grpc_addr="peer_high:50051",
            layer_range=(4, 8),
        )
        
        router.register_peer(peer1)
        router.register_peer(peer2)
        
        # Layer 2 - only peer_low
        candidates = router.get_candidates(2)
        self.assertEqual(len(candidates), 1)
        self.assertEqual(candidates[0].node_id, "peer_low")
        
        # Layer 6 - only peer_high
        candidates = router.get_candidates(6)
        self.assertEqual(len(candidates), 1)
        self.assertEqual(candidates[0].node_id, "peer_high")


class TestDiLoCoConfig(unittest.TestCase):
    """Test DiLoCo configuration through SwarmNodeConfig."""
    
    def test_diloco_settings(self):
        """Test DiLoCo settings in config."""
        from neuroshard.core.swarm.factory import SwarmNodeConfig
        
        config = SwarmNodeConfig(
            diloco_inner_steps=1000,
            diloco_outer_lr=0.5,
            diloco_outer_momentum=0.8,
        )
        
        self.assertEqual(config.diloco_inner_steps, 1000)
        self.assertEqual(config.diloco_outer_lr, 0.5)
        self.assertEqual(config.diloco_outer_momentum, 0.8)
    
    def test_aggregation_settings(self):
        """Test aggregation method settings."""
        from neuroshard.core.swarm.factory import SwarmNodeConfig
        
        config = SwarmNodeConfig(
            aggregation_method="krum",
            krum_f=2,
        )
        
        self.assertEqual(config.aggregation_method, "krum")
        self.assertEqual(config.krum_f, 2)


class TestOutboundBufferOverflow(unittest.TestCase):
    """Test outbound buffer overflow scenarios."""
    
    def test_soft_overflow_threshold(self):
        """Test soft overflow threshold configuration."""
        from neuroshard.core.swarm.buffers import OutboundBuffer
        
        buffer = OutboundBuffer(
            max_size=10,
            soft_overflow_threshold=0.7,
            hard_overflow_threshold=0.9,
        )
        
        self.assertEqual(buffer.soft_limit, 0.7)
        self.assertEqual(buffer.hard_limit, 0.9)
    
    def test_pressure_check(self):
        """Test pressure check returns correct status."""
        from neuroshard.core.swarm.buffers import OutboundBuffer
        
        buffer = OutboundBuffer(max_size=10)
        
        # Empty buffer - OK
        self.assertEqual(buffer.check_pressure(), "ok")
        self.assertFalse(buffer.is_soft_overflow)
        self.assertFalse(buffer.is_hard_overflow)


class TestSwarmStatsIntegration(unittest.TestCase):
    """Test integrated stats collection."""
    
    def test_components_combined_stats(self):
        """Test combined stats from multiple components."""
        from neuroshard.core.swarm.factory import SwarmComponents
        from neuroshard.core.swarm.buffers import ActivationBuffer, OutboundBuffer
        from neuroshard.core.swarm.router import SwarmRouter
        
        components = SwarmComponents()
        components.inbound_buffer = ActivationBuffer(max_size=100)
        components.outbound_buffer = OutboundBuffer(max_size=50)
        components.swarm_router = SwarmRouter(dht_protocol=None)
        
        stats = components.get_stats()
        
        # Should have stats from all components
        self.assertIn("inbound_buffer", stats)
        self.assertIn("outbound_buffer", stats)
        self.assertIn("router", stats)
        
        # Check nested stats
        self.assertIn("fill_rate", stats["inbound_buffer"])
        self.assertIn("queue_size", stats["outbound_buffer"])


if __name__ == '__main__':
    unittest.main()
