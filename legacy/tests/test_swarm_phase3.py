"""
Phase 3 Tests for Swarm Architecture - DiLoCo & Robust Aggregation

Tests the following components:
- DiLoCoTrainer: Inner/outer loop training
- SpeculativeCheckpointer: Hot snapshots and recovery
- RobustAggregator: Byzantine-tolerant gradient aggregation
- GradientValidator: Statistical gradient validation

Run with:
    python -m pytest tests/test_swarm_phase3.py -v
    # or
    python -m unittest tests.test_swarm_phase3 -v
"""

import asyncio
import os
import shutil
import tempfile
import time
import unittest
from pathlib import Path
from typing import Dict, List
from unittest.mock import Mock, MagicMock, patch

import torch
import torch.nn as nn

# Import Phase 3 components
from neuroshard.core.swarm.diloco import (
    DiLoCoTrainer,
    DiLoCoConfig,
    DiLoCoPhase,
    OuterOptimizer,
    create_diloco_trainer,
)
from neuroshard.core.swarm.checkpoint import (
    SpeculativeCheckpointer,
    CheckpointConfig,
    CheckpointType,
    create_checkpointer,
)
from neuroshard.core.swarm.aggregation import (
    RobustAggregator,
    GradientValidator,
    AggregationStrategy,
    AggregationConfig,
    ValidationConfig,
    GradientContribution,
    create_robust_aggregator,
)


class SimpleModel(nn.Module):
    """Simple model for testing."""
    
    def __init__(self, input_dim=64, hidden_dim=128, output_dim=32):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.my_layer_ids = [0, 1]
    
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        return self.fc2(x)


class TestOuterOptimizer(unittest.TestCase):
    """Test OuterOptimizer (Nesterov momentum)."""
    
    def test_create_optimizer(self):
        """Test creating outer optimizer."""
        opt = OuterOptimizer(lr=0.7, momentum=0.9)
        
        self.assertEqual(opt.lr, 0.7)
        self.assertEqual(opt.momentum, 0.9)
        self.assertEqual(len(opt.velocity), 0)
    
    def test_step_updates_params(self):
        """Test optimizer step updates model parameters."""
        model = SimpleModel()
        opt = OuterOptimizer(lr=0.1, momentum=0.9)
        
        # Get initial params
        initial_fc1 = model.fc1.weight.data.clone()
        
        # Create pseudo-gradient
        pseudo_grads = {
            'fc1.weight': torch.randn_like(model.fc1.weight),
            'fc1.bias': torch.randn_like(model.fc1.bias),
        }
        
        # Step
        opt.step(model, pseudo_grads)
        
        # Params should have changed
        self.assertFalse(torch.equal(initial_fc1, model.fc1.weight.data))
    
    def test_momentum_accumulates(self):
        """Test momentum accumulates across steps."""
        model = SimpleModel()
        opt = OuterOptimizer(lr=0.1, momentum=0.9)
        
        pseudo_grads = {'fc1.weight': torch.ones_like(model.fc1.weight)}
        
        # First step - initializes velocity
        opt.step(model, pseudo_grads)
        self.assertIn('fc1.weight', opt.velocity)
        
        v1 = opt.velocity['fc1.weight'].clone()
        
        # Second step - accumulates
        opt.step(model, pseudo_grads)
        v2 = opt.velocity['fc1.weight']
        
        # Velocity should be larger (momentum accumulation)
        self.assertGreater(v2.norm().item(), v1.norm().item() * 0.5)
    
    def test_state_dict(self):
        """Test state dict save/load."""
        opt1 = OuterOptimizer(lr=0.7, momentum=0.9)
        opt1.velocity['test'] = torch.randn(10)
        
        state = opt1.state_dict()
        
        opt2 = OuterOptimizer()
        opt2.load_state_dict(state)
        
        self.assertEqual(opt2.lr, 0.7)
        self.assertTrue(torch.equal(opt1.velocity['test'], opt2.velocity['test']))


class TestDiLoCoTrainer(unittest.TestCase):
    """Test DiLoCoTrainer."""
    
    def setUp(self):
        self.model = SimpleModel()
    
    def test_create_trainer(self):
        """Test creating DiLoCo trainer."""
        trainer = DiLoCoTrainer(self.model)
        
        self.assertEqual(trainer.phase, DiLoCoPhase.IDLE)
        self.assertIsNotNone(trainer.inner_optimizer)
        self.assertIsNotNone(trainer.outer_optimizer)
    
    def test_create_with_config(self):
        """Test creating with custom config."""
        config = DiLoCoConfig(
            inner_steps=100,
            outer_lr=0.5,
            outer_momentum=0.95,
        )
        
        trainer = DiLoCoTrainer(self.model, config=config)
        
        self.assertEqual(trainer.config.inner_steps, 100)
        self.assertEqual(trainer.config.outer_lr, 0.5)
    
    def test_start_inner_loop(self):
        """Test starting inner loop saves weights."""
        trainer = DiLoCoTrainer(self.model)
        
        trainer.start_inner_loop()
        
        self.assertEqual(trainer.phase, DiLoCoPhase.INNER_LOOP)
        self.assertGreater(len(trainer.initial_weights), 0)
        
        # Initial weights should match current
        for name, param in self.model.named_parameters():
            if name in trainer.initial_weights:
                self.assertTrue(torch.equal(
                    param.data,
                    trainer.initial_weights[name]
                ))
    
    def test_inner_step(self):
        """Test inner optimization step."""
        trainer = DiLoCoTrainer(self.model)
        
        # Forward pass
        x = torch.randn(4, 64)
        output = self.model(x)
        loss = output.mean()
        
        # Inner step
        loss_value = trainer.inner_step(loss)
        
        self.assertIsInstance(loss_value, float)
        self.assertEqual(trainer.stats.inner_step_count, 1)
        self.assertEqual(trainer.phase, DiLoCoPhase.INNER_LOOP)
    
    def test_should_sync(self):
        """Test should_sync detection."""
        config = DiLoCoConfig(inner_steps=3)
        trainer = DiLoCoTrainer(self.model, config=config)
        
        # Run inner steps
        for _ in range(3):
            x = torch.randn(4, 64)
            loss = self.model(x).mean()
            trainer.inner_step(loss)
        
        self.assertTrue(trainer.should_sync())
    
    def test_compute_pseudo_gradient(self):
        """Test pseudo-gradient computation."""
        trainer = DiLoCoTrainer(self.model)
        trainer.start_inner_loop()
        
        # Modify model (simulating training)
        for param in self.model.parameters():
            param.data.add_(0.1)
        
        # Compute pseudo-gradient
        pseudo_grads = trainer.compute_pseudo_gradient()
        
        self.assertGreater(len(pseudo_grads), 0)
        
        # Pseudo-grad should be initial - current (direction of improvement)
        for name, grad in pseudo_grads.items():
            expected = trainer.initial_weights[name] - dict(self.model.named_parameters())[name].data
            self.assertTrue(torch.allclose(grad, expected))
    
    def test_apply_outer_update(self):
        """Test applying outer update."""
        trainer = DiLoCoTrainer(self.model)
        trainer.start_inner_loop()
        
        initial_params = {
            name: param.data.clone()
            for name, param in self.model.named_parameters()
        }
        
        # Create pseudo-gradients
        pseudo_grads = {
            name: torch.randn_like(param)
            for name, param in self.model.named_parameters()
        }
        
        # Apply outer update
        trainer.apply_outer_update(pseudo_grads)
        
        # Params should have changed
        for name, param in self.model.named_parameters():
            self.assertFalse(torch.equal(param.data, initial_params[name]))
        
        # Stats should be updated
        self.assertEqual(trainer.stats.outer_step_count, 1)
    
    def test_full_training_cycle(self):
        """Test full inner loop -> sync -> outer step cycle."""
        config = DiLoCoConfig(inner_steps=5, gradient_accumulation=1)
        trainer = DiLoCoTrainer(self.model, config=config)
        
        # Run inner loop
        for step in range(5):
            x = torch.randn(4, 64)
            loss = self.model(x).mean()
            trainer.inner_step(loss)
        
        self.assertTrue(trainer.should_sync())
        
        # Compute and apply
        pseudo_grads = trainer.compute_pseudo_gradient()
        trainer.apply_outer_update(pseudo_grads)
        
        # Should have reset for new inner loop
        self.assertEqual(trainer.stats.inner_step_count, 0)
        self.assertEqual(trainer.stats.outer_step_count, 1)
    
    def test_state_dict(self):
        """Test state dict save/load."""
        trainer1 = DiLoCoTrainer(self.model)
        trainer1.start_inner_loop()
        trainer1.stats.outer_step_count = 5
        
        state = trainer1.state_dict()
        
        model2 = SimpleModel()
        trainer2 = DiLoCoTrainer(model2)
        trainer2.load_state_dict(state)
        
        self.assertEqual(trainer2.stats.outer_step_count, 5)
    
    def test_get_stats(self):
        """Test getting training stats."""
        trainer = DiLoCoTrainer(self.model)
        
        stats = trainer.get_stats()
        
        self.assertIn('phase', stats)
        self.assertIn('inner_step_count', stats)
        self.assertIn('outer_step_count', stats)
    
    def test_factory_function(self):
        """Test create_diloco_trainer factory."""
        trainer = create_diloco_trainer(
            self.model,
            inner_steps=100,
            outer_lr=0.5,
        )
        
        self.assertIsInstance(trainer, DiLoCoTrainer)
        self.assertEqual(trainer.config.inner_steps, 100)


class TestGradientValidator(unittest.TestCase):
    """Test GradientValidator."""
    
    def test_create_validator(self):
        """Test creating validator."""
        validator = GradientValidator()
        self.assertIsNotNone(validator.config)
    
    def test_validate_similar_gradients(self):
        """Test validating similar gradients passes."""
        validator = GradientValidator()
        
        reference = {'param': torch.randn(10, 10)}
        # Similar gradient (add small noise)
        submitted = {'param': reference['param'] + 0.01 * torch.randn(10, 10)}
        
        is_valid, reason = validator.validate(submitted, reference)
        
        self.assertTrue(is_valid)
        self.assertEqual(validator.validations_passed, 1)
    
    def test_reject_opposite_direction(self):
        """Test rejecting gradient in opposite direction."""
        config = ValidationConfig(min_cosine_similarity=0.5)
        validator = GradientValidator(config)
        
        reference = {'param': torch.ones(10, 10)}
        # Opposite direction
        submitted = {'param': -torch.ones(10, 10)}
        
        is_valid, reason = validator.validate(submitted, reference)
        
        self.assertFalse(is_valid)
        self.assertIn('cosine', reason.lower())
    
    def test_reject_large_magnitude(self):
        """Test rejecting gradient with large magnitude."""
        config = ValidationConfig(max_magnitude_ratio=5.0)
        validator = GradientValidator(config)
        
        reference = {'param': torch.ones(10, 10)}
        # Much larger magnitude
        submitted = {'param': 100 * torch.ones(10, 10)}
        
        is_valid, reason = validator.validate(submitted, reference)
        
        self.assertFalse(is_valid)
        self.assertIn('magnitude', reason.lower())
    
    def test_trust_score_check(self):
        """Test trust score threshold."""
        config = ValidationConfig(min_trust_score=0.5)
        validator = GradientValidator(config)
        
        reference = {'param': torch.randn(10, 10)}
        submitted = {'param': reference['param'].clone()}
        
        is_valid, reason = validator.validate(submitted, reference, peer_trust=0.3)
        
        self.assertFalse(is_valid)
        self.assertIn('trust', reason.lower())
    
    def test_get_stats(self):
        """Test getting validation stats."""
        validator = GradientValidator()
        
        # Run some validations
        ref = {'param': torch.randn(10)}
        validator.validate({'param': ref['param'].clone()}, ref)
        validator.validate({'param': -ref['param']}, ref)
        
        stats = validator.get_stats()
        
        self.assertIn('validations_performed', stats)
        self.assertIn('success_rate', stats)


class TestRobustAggregator(unittest.TestCase):
    """Test RobustAggregator."""
    
    def test_create_aggregator(self):
        """Test creating aggregator."""
        agg = RobustAggregator()
        self.assertIsNotNone(agg.agg_config)
        self.assertIsNotNone(agg.validator)
    
    def test_add_contribution(self):
        """Test adding gradient contribution."""
        agg = RobustAggregator()
        
        grads = {'param': torch.randn(10)}
        accepted, reason = agg.add_contribution(
            peer_id="peer1",
            gradients=grads,
            validate=False,
        )
        
        self.assertTrue(accepted)
        self.assertEqual(len(agg.contributions), 1)
    
    def test_mean_aggregation(self):
        """Test simple mean aggregation."""
        config = AggregationConfig(strategy=AggregationStrategy.MEAN)
        agg = RobustAggregator(aggregation_config=config)
        
        # Add contributions
        agg.add_contribution("peer1", {'param': torch.tensor([1.0, 2.0])}, validate=False)
        agg.add_contribution("peer2", {'param': torch.tensor([3.0, 4.0])}, validate=False)
        
        result = agg.aggregate()
        
        expected = torch.tensor([2.0, 3.0])
        self.assertTrue(torch.allclose(result['param'], expected))
    
    def test_median_aggregation(self):
        """Test median aggregation."""
        config = AggregationConfig(strategy=AggregationStrategy.MEDIAN)
        agg = RobustAggregator(aggregation_config=config)
        
        # Add contributions with outlier
        agg.add_contribution("peer1", {'param': torch.tensor([1.0])}, validate=False)
        agg.add_contribution("peer2", {'param': torch.tensor([2.0])}, validate=False)
        agg.add_contribution("peer3", {'param': torch.tensor([100.0])}, validate=False)  # Outlier
        
        result = agg.aggregate()
        
        # Median should be 2.0, not affected by outlier
        self.assertEqual(result['param'].item(), 2.0)
    
    def test_trimmed_mean_aggregation(self):
        """Test trimmed mean aggregation."""
        config = AggregationConfig(
            strategy=AggregationStrategy.TRIMMED_MEAN,
            trim_fraction=0.2,  # Trim 20% from each end
        )
        agg = RobustAggregator(aggregation_config=config)
        
        # Add contributions
        for i in range(5):
            agg.add_contribution(f"peer{i}", {'param': torch.tensor([float(i)])}, validate=False)
        
        result = agg.aggregate()
        
        # After trimming top/bottom, should average middle values
        # Values: 0, 1, 2, 3, 4 -> trim 0 and 4 -> avg(1, 2, 3) = 2.0
        self.assertAlmostEqual(result['param'].item(), 2.0, places=1)
    
    def test_krum_aggregation(self):
        """Test Krum aggregation selects closest to majority."""
        config = AggregationConfig(
            strategy=AggregationStrategy.KRUM,
            num_byzantine=1,
        )
        agg = RobustAggregator(aggregation_config=config)
        
        # Add similar contributions and one outlier
        agg.add_contribution("peer1", {'param': torch.tensor([1.0, 1.0])}, validate=False)
        agg.add_contribution("peer2", {'param': torch.tensor([1.1, 0.9])}, validate=False)
        agg.add_contribution("peer3", {'param': torch.tensor([0.9, 1.1])}, validate=False)
        agg.add_contribution("peer4", {'param': torch.tensor([100.0, 100.0])}, validate=False)  # Byzantine
        
        result = agg.aggregate()
        
        # Krum should select one of the close-together gradients, not the outlier
        self.assertLess(result['param'].norm().item(), 10.0)
    
    def test_validation_rejects_bad_gradients(self):
        """Test that validation rejects bad gradients."""
        val_config = ValidationConfig(min_cosine_similarity=0.9)
        agg = RobustAggregator(validation_config=val_config)
        
        reference = {'param': torch.ones(10)}
        
        # Good gradient
        accepted1, _ = agg.add_contribution(
            "peer1",
            {'param': torch.ones(10) * 1.1},
            reference_grads=reference,
        )
        
        # Bad gradient (opposite direction)
        accepted2, _ = agg.add_contribution(
            "peer2",
            {'param': -torch.ones(10)},
            reference_grads=reference,
        )
        
        self.assertTrue(accepted1)
        self.assertFalse(accepted2)
        self.assertEqual(len(agg.contributions), 1)
    
    def test_aggregate_with_local(self):
        """Test aggregation including local gradients."""
        config = AggregationConfig(strategy=AggregationStrategy.MEAN)
        agg = RobustAggregator(aggregation_config=config)
        
        agg.add_contribution("peer1", {'param': torch.tensor([1.0])}, validate=False)
        
        local_grads = {'param': torch.tensor([3.0])}
        result = agg.aggregate(local_grads=local_grads)
        
        # Average of 1.0 and 3.0
        self.assertEqual(result['param'].item(), 2.0)
    
    def test_clear_contributions(self):
        """Test clearing contributions."""
        agg = RobustAggregator()
        
        agg.add_contribution("peer1", {'param': torch.randn(10)}, validate=False)
        self.assertEqual(len(agg.contributions), 1)
        
        agg.clear()
        self.assertEqual(len(agg.contributions), 0)
    
    def test_get_stats(self):
        """Test getting aggregator stats."""
        agg = RobustAggregator()
        
        stats = agg.get_stats()
        
        self.assertIn('aggregations_performed', stats)
        self.assertIn('validation_stats', stats)
    
    def test_factory_function(self):
        """Test create_robust_aggregator factory."""
        agg = create_robust_aggregator(
            strategy="median",
            min_cosine_similarity=0.5,
        )
        
        self.assertIsInstance(agg, RobustAggregator)
        self.assertEqual(agg.agg_config.strategy, AggregationStrategy.MEDIAN)


class TestSpeculativeCheckpointer(unittest.TestCase):
    """Test SpeculativeCheckpointer."""
    
    def setUp(self):
        self.model = SimpleModel()
        self.optimizer = torch.optim.AdamW(self.model.parameters())
        self.temp_dir = tempfile.mkdtemp()
    
    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_create_checkpointer(self):
        """Test creating checkpointer."""
        cp = SpeculativeCheckpointer(
            model=self.model,
            optimizer=self.optimizer,
            config=CheckpointConfig(checkpoint_dir=self.temp_dir),
            node_id="test_node",
        )
        
        self.assertFalse(cp.running)
        self.assertEqual(cp.node_id, "test_node")
    
    def test_force_snapshot(self):
        """Test forcing a hot snapshot."""
        cp = SpeculativeCheckpointer(
            model=self.model,
            optimizer=self.optimizer,
            config=CheckpointConfig(checkpoint_dir=self.temp_dir),
            node_id="test_node",
        )
        
        metadata = cp.force_snapshot()
        
        self.assertIsNotNone(metadata)
        self.assertEqual(metadata.checkpoint_type, CheckpointType.HOT)
        self.assertTrue(Path(metadata.file_path).exists())
        self.assertEqual(cp.snapshots_saved, 1)
    
    def test_load_checkpoint(self):
        """Test loading a checkpoint."""
        cp = SpeculativeCheckpointer(
            model=self.model,
            optimizer=self.optimizer,
            config=CheckpointConfig(checkpoint_dir=self.temp_dir),
            node_id="test_node",
        )
        
        # Save snapshot
        metadata = cp.force_snapshot()
        
        # Load it back
        checkpoint = cp.load_checkpoint(metadata.file_path)
        
        self.assertIsNotNone(checkpoint)
        self.assertIn('model_state_dict', checkpoint)
        self.assertIn('optimizer_state_dict', checkpoint)
    
    def test_restore_from_checkpoint(self):
        """Test restoring model from checkpoint."""
        cp = SpeculativeCheckpointer(
            model=self.model,
            optimizer=self.optimizer,
            config=CheckpointConfig(checkpoint_dir=self.temp_dir),
            node_id="test_node",
        )
        
        # Set training state
        cp.update_training_state(training_step=100, outer_step=5)
        
        # Save snapshot
        metadata = cp.force_snapshot()
        
        # Modify model
        for param in self.model.parameters():
            param.data.fill_(0)
        
        # Load and restore
        checkpoint = cp.load_checkpoint(metadata.file_path)
        success = cp.restore_from_checkpoint(checkpoint)
        
        self.assertTrue(success)
        self.assertEqual(cp.training_step, 100)
        self.assertEqual(cp.recoveries_performed, 1)
    
    def test_cleanup_old_snapshots(self):
        """Test cleanup keeps only max snapshots."""
        config = CheckpointConfig(
            checkpoint_dir=self.temp_dir,
            max_hot_snapshots=2,
        )
        cp = SpeculativeCheckpointer(
            model=self.model,
            optimizer=self.optimizer,
            config=config,
            node_id="test_node",
        )
        
        # Create 4 snapshots
        for _ in range(4):
            cp.force_snapshot()
            time.sleep(0.1)  # Small delay for different timestamps
        
        # Should only keep 2
        self.assertEqual(len(cp.hot_snapshots), 2)
    
    def test_update_training_state(self):
        """Test updating training state."""
        cp = SpeculativeCheckpointer(
            model=self.model,
            optimizer=self.optimizer,
            config=CheckpointConfig(checkpoint_dir=self.temp_dir),
        )
        
        cp.update_training_state(training_step=50, outer_step=2, inner_step=10)
        
        self.assertEqual(cp.training_step, 50)
        self.assertEqual(cp.outer_step, 2)
        self.assertEqual(cp.inner_step, 10)
    
    def test_get_stats(self):
        """Test getting checkpointer stats."""
        cp = SpeculativeCheckpointer(
            model=self.model,
            optimizer=self.optimizer,
            config=CheckpointConfig(checkpoint_dir=self.temp_dir),
        )
        
        stats = cp.get_stats()
        
        self.assertIn('running', stats)
        self.assertIn('snapshots_saved', stats)
        self.assertIn('hot_snapshot_count', stats)
    
    def test_with_diloco_trainer(self):
        """Test checkpointing with DiLoCo trainer state."""
        trainer = DiLoCoTrainer(self.model)
        trainer.start_inner_loop()
        trainer.stats.outer_step_count = 3
        
        cp = SpeculativeCheckpointer(
            model=self.model,
            optimizer=self.optimizer,
            diloco_trainer=trainer,
            config=CheckpointConfig(checkpoint_dir=self.temp_dir),
            node_id="test_node",
        )
        
        # Save snapshot
        metadata = cp.force_snapshot()
        
        # Load checkpoint
        checkpoint = cp.load_checkpoint(metadata.file_path)
        
        self.assertIn('diloco_state', checkpoint)
        self.assertEqual(
            checkpoint['diloco_state']['stats']['outer_step_count'],
            3
        )
    
    def test_factory_function(self):
        """Test create_checkpointer factory."""
        cp = create_checkpointer(
            model=self.model,
            optimizer=self.optimizer,
            checkpoint_dir=self.temp_dir,
            snapshot_interval=60.0,
            node_id="factory_node",
        )
        
        self.assertIsInstance(cp, SpeculativeCheckpointer)
        self.assertEqual(cp.config.snapshot_interval, 60.0)


class TestIntegration(unittest.TestCase):
    """Integration tests for Phase 3 components."""
    
    def test_diloco_with_robust_aggregation(self):
        """Test DiLoCo training with robust aggregation."""
        model = SimpleModel()
        trainer = DiLoCoTrainer(model, config=DiLoCoConfig(inner_steps=3))
        aggregator = create_robust_aggregator(strategy="trimmed_mean")
        
        # Run inner loop
        for _ in range(3):
            x = torch.randn(4, 64)
            loss = model(x).mean()
            trainer.inner_step(loss)
        
        # Compute pseudo-gradient
        local_grads = trainer.compute_pseudo_gradient()
        
        # Simulate peer contributions
        peer_grads = {
            name: grad + 0.1 * torch.randn_like(grad)
            for name, grad in local_grads.items()
        }
        aggregator.add_contribution("peer1", peer_grads, validate=False)
        
        # Aggregate
        aggregated = aggregator.aggregate(local_grads=local_grads)
        
        # Apply
        trainer.apply_outer_update(aggregated)
        
        self.assertEqual(trainer.stats.outer_step_count, 1)
    
    def test_full_training_with_checkpointing(self):
        """Test full training cycle with checkpointing."""
        temp_dir = tempfile.mkdtemp()
        
        try:
            model = SimpleModel()
            optimizer = torch.optim.AdamW(model.parameters())
            
            trainer = DiLoCoTrainer(model, config=DiLoCoConfig(inner_steps=3))
            checkpointer = SpeculativeCheckpointer(
                model=model,
                optimizer=optimizer,
                diloco_trainer=trainer,
                config=CheckpointConfig(checkpoint_dir=temp_dir),
            )
            
            # Run some training
            for outer_step in range(2):
                for _ in range(3):
                    x = torch.randn(4, 64)
                    loss = model(x).mean()
                    trainer.inner_step(loss)
                
                pseudo_grads = trainer.compute_pseudo_gradient()
                trainer.apply_outer_update(pseudo_grads)
                
                # Update checkpointer state
                checkpointer.update_training_state(
                    training_step=outer_step * 3 + 3,
                    outer_step=outer_step + 1,
                )
            
            # Save checkpoint
            metadata = checkpointer.force_snapshot()
            
            # Verify
            checkpoint = checkpointer.load_checkpoint(metadata.file_path)
            self.assertEqual(checkpoint['diloco_state']['stats']['outer_step_count'], 2)
            
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)
    
    def test_gradient_validation_pipeline(self):
        """Test gradient validation in aggregation pipeline."""
        model = SimpleModel()
        trainer = DiLoCoTrainer(model, config=DiLoCoConfig(inner_steps=3))
        
        # Get local reference
        for _ in range(3):
            x = torch.randn(4, 64)
            trainer.inner_step(model(x).mean())
        local_grads = trainer.compute_pseudo_gradient()
        
        # Create aggregator with lenient validation for testing
        val_config = ValidationConfig(min_cosine_similarity=0.3)
        aggregator = RobustAggregator(validation_config=val_config)
        
        # Good peer gradient (scale local by small factor to maintain direction)
        good_grads = {k: v * 1.1 for k, v in local_grads.items()}
        accepted1, _ = aggregator.add_contribution(
            "good_peer", good_grads, reference_grads=local_grads
        )
        
        # Bad peer gradient (opposite direction)
        bad_grads = {k: -v for k, v in local_grads.items()}
        accepted2, _ = aggregator.add_contribution(
            "bad_peer", bad_grads, reference_grads=local_grads
        )
        
        self.assertTrue(accepted1)
        self.assertFalse(accepted2)


if __name__ == '__main__':
    unittest.main()

