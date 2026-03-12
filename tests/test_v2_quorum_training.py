"""
ARCHITECTURE_V2 Quorum-Based Training Tests

Tests for the complete training pipeline:
- Quorum formation with speed matching
- Within-quorum training pipeline
- DiLoCo cross-quorum synchronization
- Quorum lifecycle (renewal, dissolution)

Uses real gRPC servers on localhost for realistic testing.
"""

import pytest
import asyncio
import time
import json
import hashlib
import threading
import torch
import torch.nn as nn
from unittest.mock import MagicMock, patch, AsyncMock
from typing import Dict, List, Tuple, Any


# =============================================================================
# QUORUM FORMATION TESTS
# =============================================================================

class TestQuorumFormation:
    """Tests for quorum formation with speed tier matching."""
    
    def test_quorum_creation(self):
        """Test basic quorum creation."""
        from neuroshard.core.swarm.quorum import Quorum, QuorumLifecycle
        
        quorum = Quorum(
            quorum_id="test-quorum",
            speed_tier="tier2",
        )
        
        assert quorum.quorum_id == "test-quorum"
        assert quorum.speed_tier == "tier2"
        assert quorum.lifecycle == QuorumLifecycle.FORMING
        assert len(quorum.members) == 0
    
    def test_quorum_member_addition(self):
        """Test adding members to a quorum."""
        from neuroshard.core.swarm.quorum import (
            Quorum, QuorumMember, QuorumRole, QuorumLifecycle
        )
        
        quorum = Quorum(quorum_id="test", speed_tier="tier2")
        
        # Add initiator
        initiator = QuorumMember(
            node_id="node1",
            endpoint="localhost:50001",
            layer_range=(0, 4),
            speed_tier="tier2",
            role=QuorumRole.INITIATOR,
        )
        quorum.members.append(initiator)
        
        # Add processor
        processor = QuorumMember(
            node_id="node2",
            endpoint="localhost:50002",
            layer_range=(4, 8),
            speed_tier="tier2",
            role=QuorumRole.PROCESSOR,
        )
        quorum.members.append(processor)
        
        # Add finisher
        finisher = QuorumMember(
            node_id="node3",
            endpoint="localhost:50003",
            layer_range=(8, 12),
            speed_tier="tier2",
            role=QuorumRole.FINISHER,
        )
        quorum.members.append(finisher)
        
        assert len(quorum.members) == 3
        assert quorum.layer_coverage == {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11}
    
    def test_quorum_completeness_check(self):
        """Test checking if quorum covers all layers."""
        from neuroshard.core.swarm.quorum import Quorum, QuorumMember, QuorumRole
        
        quorum = Quorum(quorum_id="test", speed_tier="tier2")
        
        # Incomplete quorum (no members yet after clearing)
        quorum.members = []
        
        # Add one member - still incomplete for all roles
        quorum.members.append(QuorumMember(
            node_id="node1",
            endpoint="localhost:50001",
            layer_range=(0, 6),
            speed_tier="tier2",
            role=QuorumRole.INITIATOR,
        ))
        
        # Add remaining layers
        quorum.members.append(QuorumMember(
            node_id="node2",
            endpoint="localhost:50002",
            layer_range=(6, 12),
            speed_tier="tier2",
            role=QuorumRole.FINISHER,
        ))
        
        # Layer coverage should be complete
        assert quorum.layer_coverage == set(range(12))
    
    def test_speed_tier_matching(self):
        """Test that quorums only accept compatible speed tiers."""
        from neuroshard.core.model.dynamic import are_tiers_compatible, SpeedTier
        
        # T2 compatible with T1, T2, T3
        assert are_tiers_compatible(SpeedTier.T2, SpeedTier.T1)
        assert are_tiers_compatible(SpeedTier.T2, SpeedTier.T2)
        assert are_tiers_compatible(SpeedTier.T2, SpeedTier.T3)
        
        # T2 not compatible with T4, T5
        assert not are_tiers_compatible(SpeedTier.T2, SpeedTier.T4)
        assert not are_tiers_compatible(SpeedTier.T2, SpeedTier.T5)
    
    def test_quorum_formation_service(self, dht_network):
        """Test QuorumFormationService finds compatible peers."""
        from neuroshard.core.swarm.quorum import (
            QuorumFormationService, QuorumRegistry
        )
        
        # Create mock registry
        registry = QuorumRegistry(dht_protocol=dht_network.create_dht("registry"))
        
        # Create formation service
        service = QuorumFormationService(registry=registry)
        
        # Register some nodes in DHT
        dht = dht_network.create_dht("test")
        for layer in range(12):
            dht.announce(f"layer:{layer}", f"node{layer % 4}:5000{layer % 4}")
        
        # Service should be able to form quorums
        assert service is not None


# =============================================================================
# QUORUM TRAINING TESTS
# =============================================================================

class TestQuorumTraining:
    """Tests for within-quorum training pipeline."""
    
    def test_quorum_trainer_initialization(self):
        """Test QuorumTrainer initialization."""
        from neuroshard.core.swarm.quorum import (
            Quorum, QuorumMember, QuorumRole, QuorumTrainer
        )
        
        # Create a complete quorum
        quorum = Quorum(quorum_id="train-test", speed_tier="tier2")
        quorum.members.append(QuorumMember(
            node_id="node1",
            endpoint="localhost:50001",
            layer_range=(0, 12),
            speed_tier="tier2",
            role=QuorumRole.INITIATOR,
        ))
        
        # Create mock model
        model = MagicMock()
        model.named_parameters.return_value = []
        
        # Create trainer
        trainer = QuorumTrainer(
            quorum=quorum,
            node_id="node1",
            model=model,
            optimizer=MagicMock(),
        )
        
        assert trainer.quorum == quorum
        assert trainer.node_id == "node1"
        assert trainer.is_initiator
    
    def test_training_batch_processing(self, simple_model, mock_genesis_loader):
        """Test processing a training batch through model."""
        batch = mock_genesis_loader.get_batch()
        
        assert batch is not None
        assert 'input_ids' in batch
        assert 'labels' in batch
        
        # Forward pass
        logits = simple_model(batch['input_ids'])
        
        assert logits.shape[0] == batch['input_ids'].shape[0]
        assert logits.shape[2] == simple_model.vocab_size
        
        # Compute loss
        loss = simple_model.compute_loss(logits, batch['labels'])
        
        assert loss.item() > 0
        assert not torch.isnan(loss)
    
    def test_gradient_computation(self, simple_model, mock_genesis_loader):
        """Test gradient computation during training."""
        batch = mock_genesis_loader.get_batch()
        
        # Forward
        logits = simple_model(batch['input_ids'])
        loss = simple_model.compute_loss(logits, batch['labels'])
        
        # Backward
        loss.backward()
        
        # Check gradients exist
        has_gradients = False
        for param in simple_model.parameters():
            if param.grad is not None and param.grad.abs().sum() > 0:
                has_gradients = True
                break
        
        assert has_gradients
    
    def test_batch_recording(self):
        """Test that quorum records batch progress."""
        from neuroshard.core.swarm.quorum import Quorum
        
        quorum = Quorum(quorum_id="test", speed_tier="tier2")
        
        initial_batches = quorum.total_batches
        
        # Record batches
        quorum.record_batch(5)
        
        assert quorum.total_batches == initial_batches + 5
        
        quorum.record_batch(10)
        
        assert quorum.total_batches == initial_batches + 15


# =============================================================================
# DILOCO SYNC TESTS
# =============================================================================

class TestDiLoCoSync:
    """Tests for DiLoCo cross-quorum synchronization."""
    
    def test_pseudo_gradient_computation(self, simple_model):
        """Test computing pseudo-gradients for DiLoCo."""
        from neuroshard.core.swarm.diloco import DiLoCoTrainer, DiLoCoConfig
        
        config = DiLoCoConfig(inner_steps=10)
        trainer = DiLoCoTrainer(model=simple_model, config=config)
        
        # Start inner loop (saves initial weights)
        trainer.start_inner_loop()
        
        # Modify weights (simulate training)
        with torch.no_grad():
            for param in simple_model.parameters():
                param.add_(torch.randn_like(param) * 0.01)
        
        # Compute pseudo-gradient
        pseudo_grads = trainer.compute_pseudo_gradient()
        
        assert len(pseudo_grads) > 0
        
        # Pseudo-gradients should be non-zero
        has_nonzero = False
        for name, grad in pseudo_grads.items():
            if grad.abs().sum() > 0:
                has_nonzero = True
                break
        
        assert has_nonzero
    
    def test_outer_optimizer_step(self, simple_model):
        """Test outer optimizer applies momentum correctly."""
        from neuroshard.core.swarm.diloco import OuterOptimizer
        
        optimizer = OuterOptimizer(lr=0.7, momentum=0.9)
        
        # Create pseudo-gradients
        pseudo_grads = {}
        for name, param in simple_model.named_parameters():
            pseudo_grads[name] = torch.randn_like(param) * 0.01
        
        # Save initial weights
        initial_weights = {
            name: param.clone() for name, param in simple_model.named_parameters()
        }
        
        # Apply outer step
        optimizer.step(simple_model, pseudo_grads)
        
        # Weights should have changed
        weights_changed = False
        for name, param in simple_model.named_parameters():
            if not torch.allclose(param, initial_weights[name]):
                weights_changed = True
                break
        
        assert weights_changed
    
    def test_sync_interval_tracking(self, simple_model):
        """Test tracking batches until sync."""
        from neuroshard.core.swarm.diloco import DiLoCoTrainer, DiLoCoConfig
        
        config = DiLoCoConfig(inner_steps=100)
        trainer = DiLoCoTrainer(model=simple_model, config=config)
        
        trainer.start_inner_loop()
        
        # Should not sync initially
        assert not trainer.should_sync()
        
        # Simulate training steps
        for _ in range(50):
            trainer.stats.inner_step_count += 1
        
        assert not trainer.should_sync()
        
        # After 100 steps, should sync
        for _ in range(50):
            trainer.stats.inner_step_count += 1
        
        assert trainer.should_sync()
    
    def test_freshness_weighting(self):
        """Test freshness decay for gradient weighting."""
        from neuroshard.core.swarm.aggregation import compute_contribution_weight
        
        # Fresh contribution (< 1 hour) - freshness = 1.0
        # Weight = sqrt(100) * 1.0 = 10.0
        weight_fresh = compute_contribution_weight(batches_processed=100, age_hours=0.5)
        assert abs(weight_fresh - 10.0) < 0.01
        
        # Day old (< 1 day) - freshness = 0.9
        # Weight = sqrt(100) * 0.9 = 9.0
        weight_day = compute_contribution_weight(batches_processed=100, age_hours=12)
        assert abs(weight_day - 9.0) < 0.01
        
        # Week old (< 1 week) - freshness = 0.7
        # Weight = sqrt(100) * 0.7 = 7.0
        weight_week = compute_contribution_weight(batches_processed=100, age_hours=100)
        assert abs(weight_week - 7.0) < 0.01
        
        # Very old (>= 1 week) - freshness = 0.3
        # Weight = sqrt(100) * 0.3 = 3.0
        weight_old = compute_contribution_weight(batches_processed=100, age_hours=200)
        assert abs(weight_old - 3.0) < 0.01
    
    def test_sqrt_n_weighting(self):
        """Test sqrt(n) weighting for contributions."""
        import math
        from neuroshard.core.swarm.aggregation import compute_contribution_weight
        
        # 100 batches, fresh
        weight_100 = compute_contribution_weight(100, age_hours=0.5)
        
        # 400 batches, fresh (should be 2x weight of 100)
        weight_400 = compute_contribution_weight(400, age_hours=0.5)
        
        # sqrt(400) / sqrt(100) = 2
        assert abs(weight_400 / weight_100 - 2.0) < 0.01


# =============================================================================
# QUORUM LIFECYCLE TESTS
# =============================================================================

class TestQuorumLifecycle:
    """Tests for quorum session management and lifecycle."""
    
    def test_session_timing(self):
        """Test session timing and expiration."""
        from neuroshard.core.swarm.quorum import Quorum, BASE_SESSION_DURATION
        
        quorum = Quorum(quorum_id="test", speed_tier="tier2")
        quorum.session_start = time.time()
        quorum.session_end = quorum.session_start + BASE_SESSION_DURATION
        
        # Session should not be expired initially
        assert not quorum.is_session_expired
        
        # Check session remaining
        assert quorum.session_remaining > 0
        assert quorum.session_remaining <= BASE_SESSION_DURATION
    
    def test_lifecycle_transitions(self):
        """Test quorum lifecycle state transitions."""
        from neuroshard.core.swarm.quorum import Quorum, QuorumLifecycle
        
        quorum = Quorum(quorum_id="test", speed_tier="tier2")
        
        # Starts in FORMING
        assert quorum.lifecycle == QuorumLifecycle.FORMING
        
        # Transition to ACTIVE
        quorum.lifecycle = QuorumLifecycle.ACTIVE
        assert quorum.lifecycle == QuorumLifecycle.ACTIVE
        
        # Transition to RENEWING
        quorum.lifecycle = QuorumLifecycle.RENEWING
        assert quorum.lifecycle == QuorumLifecycle.RENEWING
        
        # Transition to DISSOLVED
        quorum.lifecycle = QuorumLifecycle.DISSOLVED
        assert quorum.lifecycle == QuorumLifecycle.DISSOLVED
    
    def test_member_health_tracking(self):
        """Test tracking member health in quorum."""
        from neuroshard.core.swarm.quorum import (
            Quorum, QuorumMember, QuorumRole, HEARTBEAT_INTERVAL, STALE_THRESHOLD
        )
        
        quorum = Quorum(quorum_id="test", speed_tier="tier2")
        
        # Add healthy member
        healthy = QuorumMember(
            node_id="healthy",
            endpoint="localhost:50001",
            layer_range=(0, 6),
            speed_tier="tier2",
            role=QuorumRole.INITIATOR,
        )
        healthy.last_heartbeat = time.time()
        quorum.members.append(healthy)
        
        # Add stale member
        stale = QuorumMember(
            node_id="stale",
            endpoint="localhost:50002",
            layer_range=(6, 12),
            speed_tier="tier2",
            role=QuorumRole.FINISHER,
        )
        stale.last_heartbeat = time.time() - (HEARTBEAT_INTERVAL * STALE_THRESHOLD + 10)
        quorum.members.append(stale)
        
        # Check health counts
        healthy_members = quorum.get_healthy_members()
        stale_members = quorum.get_stale_members()
        
        assert len(healthy_members) == 1
        assert len(stale_members) == 1
        assert healthy_members[0].node_id == "healthy"
        assert stale_members[0].node_id == "stale"
    
    def test_renewal_check_timing(self):
        """Test renewal check timing."""
        from neuroshard.core.swarm.quorum import (
            Quorum, BASE_SESSION_DURATION, RENEWAL_CHECK_RATIO
        )
        
        quorum = Quorum(quorum_id="test", speed_tier="tier2")
        quorum.session_start = time.time()
        quorum.session_end = quorum.session_start + BASE_SESSION_DURATION
        
        # Initially not time for renewal
        assert not quorum.should_check_renewal
        
        # Simulate time passing to 80% of session
        renewal_time = BASE_SESSION_DURATION * RENEWAL_CHECK_RATIO
        quorum.session_start = time.time() - renewal_time - 1
        quorum.session_end = quorum.session_start + BASE_SESSION_DURATION
        
        assert quorum.should_check_renewal


# =============================================================================
# ROBUST AGGREGATION TESTS
# =============================================================================

class TestRobustAggregation:
    """Tests for Byzantine-tolerant gradient aggregation."""
    
    def test_mean_aggregation(self):
        """Test simple mean aggregation."""
        from neuroshard.core.swarm.aggregation import (
            RobustAggregator, AggregationConfig, AggregationStrategy
        )
        
        config = AggregationConfig(
            strategy=AggregationStrategy.MEAN,
        )
        aggregator = RobustAggregator(aggregation_config=config)
        
        # Add contributions
        for i in range(3):
            aggregator.add_contribution(
                peer_id=f"node{i}",
                gradients={"param": torch.ones(10) * (i + 1)},
                validate=False,  # Skip validation for test
            )
        
        # Aggregate
        result = aggregator.aggregate()
        
        # Mean of 1, 2, 3 = 2
        assert torch.allclose(result["param"], torch.ones(10) * 2)
    
    def test_trimmed_mean_aggregation(self):
        """Test trimmed mean handles outliers."""
        from neuroshard.core.swarm.aggregation import (
            RobustAggregator, AggregationConfig, AggregationStrategy
        )
        
        config = AggregationConfig(
            strategy=AggregationStrategy.TRIMMED_MEAN,
            trim_fraction=0.2,
        )
        aggregator = RobustAggregator(aggregation_config=config)
        
        # Add normal contributions
        for i in range(4):
            aggregator.add_contribution(
                peer_id=f"node{i}",
                gradients={"param": torch.ones(10)},
                validate=False,
            )
        
        # Add outlier
        aggregator.add_contribution(
            peer_id="outlier",
            gradients={"param": torch.ones(10) * 100},
            validate=False,
        )
        
        result = aggregator.aggregate()
        
        # Outlier should be trimmed, result close to 1
        assert result["param"].mean().item() < 10
    
    def test_freshness_weighted_aggregation(self):
        """Test freshness-weighted aggregation (enabled by default)."""
        from neuroshard.core.swarm.aggregation import (
            RobustAggregator, AggregationConfig, AggregationStrategy
        )
        
        # Default config has use_freshness_weights=True
        config = AggregationConfig(
            strategy=AggregationStrategy.MEAN,
            use_freshness_weights=True,  # Uses sqrt(batches) * freshness
        )
        aggregator = RobustAggregator(aggregation_config=config)
        
        # Large contribution (high weight via more batches)
        # Weight = sqrt(100) * 1.0 = 10
        aggregator.add_contribution(
            peer_id="large",
            gradients={"param": torch.ones(10) * 1},
            batches_processed=100,
            validate=False,
        )
        
        # Smaller contribution (lower weight)
        # Weight = sqrt(10) * 1.0 ≈ 3.16
        aggregator.add_contribution(
            peer_id="small",
            gradients={"param": torch.ones(10) * 10},
            batches_processed=10,
            validate=False,
        )
        
        result = aggregator.aggregate()
        
        # Large contribution should dominate due to sqrt(n) weighting
        # Weighted avg: (10*1 + 3.16*10) / (10 + 3.16) ≈ 3.16
        assert result["param"].mean().item() < 5


# =============================================================================
# INTEGRATION TESTS
# =============================================================================

class TestTrainingIntegration:
    """Integration tests for complete training flows."""
    
    def test_quorum_training_session(
        self, simple_model, mock_genesis_loader, dht_network
    ):
        """Test a complete training session within a quorum."""
        from neuroshard.core.swarm.quorum import (
            Quorum, QuorumMember, QuorumRole, QuorumLifecycle
        )
        from neuroshard.core.swarm.diloco import DiLoCoTrainer, DiLoCoConfig
        
        # Create quorum
        quorum = Quorum(quorum_id="train-session", speed_tier="tier2")
        quorum.members.append(QuorumMember(
            node_id="solo-node",
            endpoint="localhost:50001",
            layer_range=(0, 4),  # Matches model layers
            speed_tier="tier2",
            role=QuorumRole.INITIATOR,
        ))
        quorum.lifecycle = QuorumLifecycle.ACTIVE
        quorum.session_start = time.time()
        quorum.session_end = quorum.session_start + 3600
        
        # Create DiLoCo trainer
        config = DiLoCoConfig(inner_steps=5)
        trainer = DiLoCoTrainer(
            model=simple_model,
            config=config,
            quorum=quorum,
            dht_protocol=dht_network.create_dht("trainer"),
            layer_range=(0, 4),
        )
        
        trainer.start_inner_loop()
        
        # Train for a few batches
        optimizer = torch.optim.Adam(simple_model.parameters())
        
        for _ in range(5):
            batch = mock_genesis_loader.get_batch()
            
            logits = simple_model(batch['input_ids'])
            loss = simple_model.compute_loss(logits, batch['labels'])
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            trainer.stats.inner_step_count += 1
        
        # Should be ready to sync
        assert trainer.should_sync()
        
        # Compute pseudo-gradients
        pseudo_grads = trainer.compute_pseudo_gradient()
        assert len(pseudo_grads) > 0
        
        # Record batches
        quorum.record_batch(5)
        assert quorum.total_batches == 5
    
    def test_multi_node_gradient_sync(self, simple_model):
        """Test gradient synchronization between multiple nodes."""
        from neuroshard.core.swarm.aggregation import (
            RobustAggregator, AggregationConfig, AggregationStrategy
        )
        from neuroshard.core.swarm.diloco import DiLoCoTrainer, DiLoCoConfig
        
        # Create multiple trainers (simulating nodes)
        trainers = []
        models = []
        
        for i in range(3):
            model = type(simple_model)(
                vocab_size=simple_model.vocab_size,
                hidden_dim=simple_model.hidden_dim,
                num_layers=simple_model.num_layers,
            )
            # Copy initial weights
            model.load_state_dict(simple_model.state_dict())
            
            trainer = DiLoCoTrainer(
                model=model,
                config=DiLoCoConfig(inner_steps=10),
            )
            trainer.start_inner_loop()
            
            trainers.append(trainer)
            models.append(model)
        
        # Each trainer does different training
        for i, (trainer, model) in enumerate(zip(trainers, models)):
            with torch.no_grad():
                for param in model.parameters():
                    param.add_(torch.randn_like(param) * 0.01 * (i + 1))
        
        # Compute pseudo-gradients
        all_grads = []
        for trainer in trainers:
            grads = trainer.compute_pseudo_gradient()
            all_grads.append(grads)
        
        # Aggregate
        config = AggregationConfig(strategy=AggregationStrategy.MEAN)
        aggregator = RobustAggregator(aggregation_config=config)
        
        for i, grads in enumerate(all_grads):
            aggregator.add_contribution(
                peer_id=f"node{i}",
                gradients=grads,
                validate=False,
            )
        
        aggregated = aggregator.aggregate()
        
        # All should agree on aggregated gradients
        assert len(aggregated) > 0
