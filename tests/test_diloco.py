"""
Unit tests for DiLoCo (Distributed Low-Communication) Training.

Tests:
- DiLoCo trainer lifecycle (inner/outer loop)
- Pseudo-gradient computation
- Outer optimizer (Nesterov momentum)
- State checkpointing
- Sync callback integration
"""

import sys
import os
import unittest
import asyncio
import time
from unittest.mock import Mock, AsyncMock, patch

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn

from neuroshard.core.swarm.diloco import (
    DiLoCoTrainer,
    DiLoCoConfig,
    DiLoCoPhase,
    DiLoCoStats,
    OuterOptimizer,
    DiLoCoGossipProtocol,
    create_diloco_trainer,
)


class SimpleModel(nn.Module):
    """Simple model for testing."""
    
    def __init__(self, input_size=10, hidden_size=20, output_size=5):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)
        
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        return self.fc2(x)


class TestDiLoCoConfig(unittest.TestCase):
    """Tests for DiLoCo configuration."""
    
    def test_default_config(self):
        """Test default configuration values."""
        config = DiLoCoConfig()
        
        self.assertEqual(config.inner_steps, 500)
        self.assertEqual(config.outer_lr, 0.7)
        self.assertEqual(config.outer_momentum, 0.9)
        self.assertEqual(config.inner_lr, 1e-4)
        
    def test_custom_config(self):
        """Test custom configuration."""
        config = DiLoCoConfig(
            inner_steps=100,
            outer_lr=0.5,
            inner_lr=1e-3,
        )
        
        self.assertEqual(config.inner_steps, 100)
        self.assertEqual(config.outer_lr, 0.5)
        self.assertEqual(config.inner_lr, 1e-3)


class TestOuterOptimizer(unittest.TestCase):
    """Tests for the outer optimizer with Nesterov momentum."""
    
    def setUp(self):
        """Set up test model."""
        self.model = SimpleModel()
        self.optimizer = OuterOptimizer(lr=0.7, momentum=0.9)
        
    def test_initial_step(self):
        """Test first optimizer step."""
        # Create pseudo-gradients
        pseudo_grads = {}
        for name, param in self.model.named_parameters():
            pseudo_grads[name] = torch.randn_like(param)
            
        # Store initial weights
        initial_weights = {
            name: param.clone() for name, param in self.model.named_parameters()
        }
        
        # Apply outer step
        self.optimizer.step(self.model, pseudo_grads)
        
        # Weights should have changed
        for name, param in self.model.named_parameters():
            self.assertFalse(
                torch.equal(param.data, initial_weights[name]),
                f"Parameter {name} didn't change"
            )
            
    def test_momentum_accumulation(self):
        """Test that momentum accumulates across steps."""
        # Create consistent pseudo-gradients
        pseudo_grads = {}
        for name, param in self.model.named_parameters():
            pseudo_grads[name] = torch.ones_like(param) * 0.1
            
        # First step
        self.optimizer.step(self.model, pseudo_grads)
        
        # Check velocity was initialized
        self.assertEqual(len(self.optimizer.velocity), len(pseudo_grads))
        
        # Second step - momentum should amplify
        weights_after_first = {
            name: param.clone() for name, param in self.model.named_parameters()
        }
        
        self.optimizer.step(self.model, pseudo_grads)
        
        # Weights should have moved further
        for name, param in self.model.named_parameters():
            if name in pseudo_grads:
                # Should have moved in the gradient direction
                self.assertNotEqual(
                    param.data.sum().item(),
                    weights_after_first[name].sum().item()
                )
                
    def test_state_dict(self):
        """Test saving and loading state."""
        # Do a step to create velocity
        pseudo_grads = {
            name: torch.randn_like(param)
            for name, param in self.model.named_parameters()
        }
        self.optimizer.step(self.model, pseudo_grads)
        
        # Save state
        state = self.optimizer.state_dict()
        
        self.assertEqual(state['lr'], 0.7)
        self.assertEqual(state['momentum'], 0.9)
        self.assertIn('velocity', state)
        
        # Create new optimizer and load state
        new_optimizer = OuterOptimizer()
        new_optimizer.load_state_dict(state)
        
        self.assertEqual(new_optimizer.lr, 0.7)
        self.assertEqual(new_optimizer.momentum, 0.9)
        self.assertEqual(len(new_optimizer.velocity), len(pseudo_grads))


class TestDiLoCoTrainer(unittest.TestCase):
    """Tests for DiLoCoTrainer."""
    
    def setUp(self):
        """Set up trainer."""
        self.model = SimpleModel()
        self.config = DiLoCoConfig(inner_steps=5)  # Small for testing
        self.trainer = DiLoCoTrainer(
            model=self.model,
            config=self.config,
        )
        
    def test_initial_state(self):
        """Test initial trainer state."""
        self.assertEqual(self.trainer.phase, DiLoCoPhase.IDLE)
        self.assertEqual(self.trainer.stats.inner_step_count, 0)
        self.assertEqual(self.trainer.stats.outer_step_count, 0)
        
    def test_start_inner_loop(self):
        """Test starting an inner loop."""
        self.trainer.start_inner_loop()
        
        self.assertEqual(self.trainer.phase, DiLoCoPhase.INNER_LOOP)
        self.assertEqual(len(self.trainer.initial_weights), len(list(self.model.parameters())))
        
    def test_inner_step(self):
        """Test inner training step."""
        # Create dummy loss
        x = torch.randn(2, 10)
        y = self.model(x)
        loss = y.sum()
        
        # Execute inner step
        loss_value = self.trainer.inner_step(loss)
        
        self.assertEqual(self.trainer.phase, DiLoCoPhase.INNER_LOOP)
        self.assertEqual(self.trainer.stats.inner_step_count, 1)
        self.assertIsInstance(loss_value, float)
        
    def test_should_sync(self):
        """Test sync trigger detection."""
        # Initially should not sync
        self.assertFalse(self.trainer.should_sync())
        
        # Do enough inner steps
        for _ in range(self.config.inner_steps):
            x = torch.randn(2, 10)
            loss = self.model(x).sum()
            self.trainer.inner_step(loss)
            
        # Should now trigger sync
        self.assertTrue(self.trainer.should_sync())
        
    def test_compute_pseudo_gradient(self):
        """Test pseudo-gradient computation."""
        # Start inner loop (saves initial weights)
        self.trainer.start_inner_loop()
        
        # Do some training
        for _ in range(3):
            x = torch.randn(2, 10)
            loss = self.model(x).sum()
            self.trainer.inner_step(loss)
            
        # Compute pseudo-gradient
        pseudo_grads = self.trainer.compute_pseudo_gradient()
        
        self.assertEqual(self.trainer.phase, DiLoCoPhase.COMPUTING_DELTA)
        self.assertEqual(len(pseudo_grads), len(self.trainer.initial_weights))
        
        # Each pseudo-grad should be: initial - current
        for name, grad in pseudo_grads.items():
            self.assertEqual(grad.shape, self.trainer.initial_weights[name].shape)
            
    def test_apply_outer_update_with_local_grads(self):
        """Test outer update with local gradients only."""
        # Complete inner loop
        self.trainer.start_inner_loop()
        for _ in range(3):
            x = torch.randn(2, 10)
            loss = self.model(x).sum()
            self.trainer.inner_step(loss)
            
        # Apply outer update (without sync - uses local grads)
        self.trainer.apply_outer_update(None)
        
        self.assertEqual(self.trainer.stats.outer_step_count, 1)
        self.assertEqual(self.trainer.stats.local_only_outer_steps, 1)
        
        # Should start new inner loop
        self.assertEqual(self.trainer.phase, DiLoCoPhase.INNER_LOOP)
        self.assertEqual(self.trainer.stats.inner_step_count, 0)
        
    def test_apply_outer_update_with_aggregated_grads(self):
        """Test outer update with aggregated gradients."""
        self.trainer.start_inner_loop()
        for _ in range(3):
            x = torch.randn(2, 10)
            loss = self.model(x).sum()
            self.trainer.inner_step(loss)
            
        # Create mock aggregated gradients
        aggregated = self.trainer.compute_pseudo_gradient()
        
        # Apply outer update
        self.trainer.apply_outer_update(aggregated)
        
        self.assertEqual(self.trainer.stats.outer_step_count, 1)
        self.assertEqual(self.trainer.stats.local_only_outer_steps, 0)
        
    def test_full_training_cycle(self):
        """Test complete training cycle."""
        # Do multiple outer steps
        for outer_step in range(3):
            # Inner loop
            for _ in range(self.config.inner_steps):
                x = torch.randn(2, 10)
                loss = self.model(x).sum()
                self.trainer.inner_step(loss)
                
            # Should trigger sync
            self.assertTrue(self.trainer.should_sync())
            
            # Compute and apply
            pseudo_grads = self.trainer.compute_pseudo_gradient()
            self.trainer.apply_outer_update(pseudo_grads)
            
        self.assertEqual(self.trainer.stats.outer_step_count, 3)
        self.assertEqual(
            self.trainer.stats.total_inner_steps,
            3 * self.config.inner_steps
        )
        
    def test_gradient_accumulation(self):
        """Test gradient accumulation."""
        config = DiLoCoConfig(inner_steps=5, gradient_accumulation=2)
        trainer = DiLoCoTrainer(self.model, config=config)
        
        trainer.start_inner_loop()
        
        # First accumulation step - no optimizer step yet
        x = torch.randn(2, 10)
        loss = self.model(x).sum()
        trainer.inner_step(loss)
        
        self.assertEqual(trainer.stats.inner_step_count, 0)
        
        # Second accumulation step - now optimizer steps
        loss2 = self.model(torch.randn(2, 10)).sum()
        trainer.inner_step(loss2)
        
        self.assertEqual(trainer.stats.inner_step_count, 1)
        
    def test_get_stats(self):
        """Test statistics retrieval."""
        self.trainer.start_inner_loop()
        
        x = torch.randn(2, 10)
        loss = self.model(x).sum()
        self.trainer.inner_step(loss)
        
        stats = self.trainer.get_stats()
        
        self.assertEqual(stats['phase'], 'inner_loop')
        self.assertEqual(stats['inner_step_count'], 1)
        self.assertIn('avg_inner_loss', stats)
        
    def test_state_dict_and_load(self):
        """Test saving and loading trainer state."""
        # Do some training
        self.trainer.start_inner_loop()
        for _ in range(3):
            x = torch.randn(2, 10)
            loss = self.model(x).sum()
            self.trainer.inner_step(loss)
            
        # Save state
        state = self.trainer.state_dict()
        
        self.assertIn('config', state)
        self.assertIn('inner_optimizer', state)
        self.assertIn('outer_optimizer', state)
        self.assertIn('stats', state)
        
        # Create new trainer and load
        new_trainer = DiLoCoTrainer(self.model, config=self.config)
        new_trainer.load_state_dict(state)
        
        self.assertEqual(
            new_trainer.stats.inner_step_count,
            self.trainer.stats.inner_step_count
        )
        
    def test_sync_callback(self):
        """Test sync callback integration."""
        callback_called = []
        
        async def mock_sync(pseudo_grads):
            callback_called.append(True)
            return pseudo_grads  # Return same grads
            
        self.trainer.set_sync_callback(mock_sync)
        
        # Complete inner loop
        self.trainer.start_inner_loop()
        for _ in range(3):
            x = torch.randn(2, 10)
            loss = self.model(x).sum()
            self.trainer.inner_step(loss)
            
        # Run async sync
        pseudo_grads = self.trainer.compute_pseudo_gradient()
        
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        result = loop.run_until_complete(self.trainer.sync_with_peers(pseudo_grads))
        loop.close()
        
        self.assertEqual(len(callback_called), 1)
        self.assertIsNotNone(result)


class TestDiLoCoGossipProtocol(unittest.TestCase):
    """Tests for DiLoCo gossip protocol."""
    
    def setUp(self):
        """Set up gossip protocol."""
        self.protocol = DiLoCoGossipProtocol(
            node_id="test_node",
            min_peers=1,
            timeout=5.0,
        )
        
    def test_receive_contribution(self):
        """Test receiving contributions from peers."""
        grads = {"fc1.weight": torch.randn(20, 10)}
        
        self.protocol.receive_contribution("peer1", grads)
        
        self.assertIn("peer1", self.protocol.pending_contributions)
        
    def test_aggregate_contributions(self):
        """Test aggregating contributions."""
        # Create contributions
        contributions = {
            "peer1": {"weight": torch.ones(5)},
            "peer2": {"weight": torch.ones(5) * 3},
        }
        
        aggregated = self.protocol._aggregate_contributions(contributions)
        
        # Should be average: (1 + 3) / 2 = 2
        self.assertTrue(
            torch.allclose(aggregated["weight"], torch.ones(5) * 2)
        )
        
    def test_aggregate_empty(self):
        """Test aggregating empty contributions."""
        result = self.protocol._aggregate_contributions({})
        self.assertEqual(result, {})


class TestFactoryFunction(unittest.TestCase):
    """Tests for factory function."""
    
    def test_create_diloco_trainer(self):
        """Test factory function creates trainer correctly."""
        model = SimpleModel()
        
        trainer = create_diloco_trainer(
            model=model,
            inner_steps=100,
            outer_lr=0.5,
            inner_lr=1e-3,
        )
        
        self.assertIsInstance(trainer, DiLoCoTrainer)
        self.assertEqual(trainer.config.inner_steps, 100)
        self.assertEqual(trainer.config.outer_lr, 0.5)
        self.assertEqual(trainer.config.inner_lr, 1e-3)


class TestDiLoCoStats(unittest.TestCase):
    """Tests for DiLoCo statistics tracking."""
    
    def test_avg_inner_loss(self):
        """Test average inner loss calculation."""
        stats = DiLoCoStats()
        
        # Initially zero
        self.assertEqual(stats.avg_inner_loss, 0.0)
        
        # Add some losses
        stats.inner_loss_sum = 10.0
        stats.inner_loss_count = 5
        
        self.assertEqual(stats.avg_inner_loss, 2.0)
        
    def test_reset_inner_stats(self):
        """Test resetting inner stats."""
        stats = DiLoCoStats()
        stats.inner_loss_sum = 10.0
        stats.inner_loss_count = 5
        stats.inner_step_count = 10
        
        stats.reset_inner_stats()
        
        self.assertEqual(stats.inner_loss_sum, 0.0)
        self.assertEqual(stats.inner_loss_count, 0)
        self.assertEqual(stats.inner_step_count, 0)


if __name__ == '__main__':
    unittest.main()

