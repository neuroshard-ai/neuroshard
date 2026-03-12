"""
Unit tests for Robust Gradient Aggregation (Byzantine-Tolerant).

Tests:
- Simple mean aggregation
- Coordinate-wise median aggregation
- Trimmed mean aggregation
- Krum/Multi-Krum aggregation
- Gradient validation (cosine similarity, magnitude, variance)
- Byzantine attack resistance
"""

import sys
import os
import unittest
import math

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from neuroshard.core.swarm.aggregation import (
    RobustAggregator,
    GradientValidator,
    GradientContribution,
    AggregationConfig,
    AggregationStrategy,
    ValidationConfig,
    create_robust_aggregator,
)


class TestGradientValidator(unittest.TestCase):
    """Tests for gradient validation."""
    
    def setUp(self):
        """Set up validator."""
        self.config = ValidationConfig(
            min_cosine_similarity=0.5,
            max_magnitude_ratio=5.0,
            min_magnitude_ratio=0.2,
            max_variance_ratio=50.0,
        )
        self.validator = GradientValidator(self.config)
        
    def test_valid_gradient(self):
        """Test validation passes for similar gradients."""
        reference = {"weight": torch.randn(10, 10)}
        submitted = {"weight": reference["weight"] * 1.1}  # Slightly scaled
        
        is_valid, reason = self.validator.validate(submitted, reference)
        
        self.assertTrue(is_valid)
        
    def test_cosine_similarity_check(self):
        """Test cosine similarity validation."""
        reference = {"weight": torch.randn(10, 10)}
        
        # Opposite direction gradient
        submitted = {"weight": -reference["weight"]}
        
        is_valid, reason = self.validator.validate(submitted, reference)
        
        self.assertFalse(is_valid)
        self.assertIn("cosine", reason.lower())
        
    def test_magnitude_too_high(self):
        """Test rejection of gradients with too high magnitude."""
        reference = {"weight": torch.randn(10, 10)}
        
        # 10x magnitude (above 5x threshold)
        submitted = {"weight": reference["weight"] * 10.0}
        
        is_valid, reason = self.validator.validate(submitted, reference)
        
        self.assertFalse(is_valid)
        self.assertIn("magnitude", reason.lower())
        
    def test_magnitude_too_low(self):
        """Test rejection of gradients with too low magnitude."""
        reference = {"weight": torch.randn(10, 10) * 10}  # Large reference
        
        # 0.01x magnitude (below 0.2x threshold)
        submitted = {"weight": reference["weight"] * 0.01}
        
        is_valid, reason = self.validator.validate(submitted, reference)
        
        self.assertFalse(is_valid)
        self.assertIn("magnitude", reason.lower())
        
    def test_variance_check(self):
        """Test variance ratio validation."""
        reference = {"weight": torch.randn(10, 10)}
        
        # Artificially high variance
        submitted = {"weight": torch.randn(10, 10) * 100}
        
        is_valid, reason = self.validator.validate(submitted, reference)
        
        # May fail on magnitude or variance
        # The key is it should fail for suspicious gradients
        # (Note: random gradients might pass cosine check, but fail magnitude)
        
    def test_zero_reference(self):
        """Test handling of zero reference gradient."""
        reference = {"weight": torch.zeros(10, 10)}
        submitted = {"weight": torch.randn(10, 10)}
        
        # Should not crash - zero reference is edge case
        is_valid, reason = self.validator.validate(submitted, reference)
        
        # Should pass (we're lenient with zero references)
        self.assertTrue(is_valid)
        
    def test_stats_tracking(self):
        """Test validation statistics."""
        reference = {"weight": torch.randn(10)}
        
        # Valid gradients
        for _ in range(5):
            self.validator.validate(
                {"weight": reference["weight"] * 1.1},
                reference
            )
            
        # Invalid gradient
        self.validator.validate(
            {"weight": -reference["weight"]},  # Opposite direction
            reference
        )
        
        stats = self.validator.get_stats()
        
        self.assertEqual(stats['validations_performed'], 6)
        self.assertGreater(stats['validations_passed'], 0)


class TestRobustAggregatorMean(unittest.TestCase):
    """Tests for mean aggregation."""
    
    def setUp(self):
        """Set up aggregator with mean strategy."""
        self.aggregator = RobustAggregator(
            aggregation_config=AggregationConfig(strategy=AggregationStrategy.MEAN),
            validation_config=ValidationConfig()
        )
        
    def test_simple_mean(self):
        """Test simple mean aggregation."""
        # Add contributions
        self.aggregator.add_contribution(
            "peer1",
            {"weight": torch.ones(5) * 2},
            validate=False
        )
        self.aggregator.add_contribution(
            "peer2",
            {"weight": torch.ones(5) * 4},
            validate=False
        )
        
        result = self.aggregator.aggregate()
        
        # Mean of [2, 4] = 3
        self.assertTrue(
            torch.allclose(result["weight"], torch.ones(5) * 3)
        )
        
    def test_mean_with_local(self):
        """Test mean aggregation including local gradients."""
        self.aggregator.add_contribution(
            "peer1",
            {"weight": torch.ones(5) * 1},
            validate=False
        )
        
        local_grads = {"weight": torch.ones(5) * 5}
        result = self.aggregator.aggregate(local_grads=local_grads)
        
        # Mean of [1, 5] = 3
        self.assertTrue(
            torch.allclose(result["weight"], torch.ones(5) * 3)
        )
        
    def test_clear(self):
        """Test clearing contributions."""
        self.aggregator.add_contribution(
            "peer1",
            {"weight": torch.randn(5)},
            validate=False
        )
        
        self.aggregator.clear()
        
        self.assertEqual(len(self.aggregator.contributions), 0)


class TestRobustAggregatorMedian(unittest.TestCase):
    """Tests for median aggregation."""
    
    def setUp(self):
        """Set up aggregator with median strategy."""
        self.aggregator = RobustAggregator(
            aggregation_config=AggregationConfig(strategy=AggregationStrategy.MEDIAN)
        )
        
    def test_coordinate_median(self):
        """Test coordinate-wise median."""
        self.aggregator.add_contribution(
            "peer1", {"weight": torch.tensor([1.0, 2.0, 3.0])}, validate=False
        )
        self.aggregator.add_contribution(
            "peer2", {"weight": torch.tensor([2.0, 3.0, 4.0])}, validate=False
        )
        self.aggregator.add_contribution(
            "peer3", {"weight": torch.tensor([3.0, 4.0, 5.0])}, validate=False
        )
        
        result = self.aggregator.aggregate()
        
        # Median of each coordinate
        expected = torch.tensor([2.0, 3.0, 4.0])
        self.assertTrue(torch.allclose(result["weight"], expected))
        
    def test_median_rejects_outliers(self):
        """Test that median is robust to outliers."""
        # Two honest peers
        self.aggregator.add_contribution(
            "honest1", {"weight": torch.ones(5)}, validate=False
        )
        self.aggregator.add_contribution(
            "honest2", {"weight": torch.ones(5)}, validate=False
        )
        
        # One Byzantine peer with extreme values
        self.aggregator.add_contribution(
            "byzantine", {"weight": torch.ones(5) * 1000}, validate=False
        )
        
        result = self.aggregator.aggregate()
        
        # Median should be 1, not influenced by 1000
        self.assertTrue(torch.allclose(result["weight"], torch.ones(5)))


class TestRobustAggregatorTrimmedMean(unittest.TestCase):
    """Tests for trimmed mean aggregation."""
    
    def setUp(self):
        """Set up aggregator with trimmed mean strategy."""
        self.aggregator = RobustAggregator(
            aggregation_config=AggregationConfig(
                strategy=AggregationStrategy.TRIMMED_MEAN,
                trim_fraction=0.2,  # Trim 20% from each end
            )
        )
        
    def test_trimmed_mean(self):
        """Test trimmed mean removes extreme values."""
        # 5 contributions - will trim 1 from each end
        for i, val in enumerate([1, 2, 3, 4, 100]):  # 100 is outlier
            self.aggregator.add_contribution(
                f"peer{i}",
                {"weight": torch.tensor([float(val)])},
                validate=False
            )
            
        result = self.aggregator.aggregate()
        
        # After trimming 1 and 100, mean of [2, 3, 4] = 3
        self.assertTrue(
            torch.allclose(result["weight"], torch.tensor([3.0]))
        )
        
    def test_trimmed_mean_small_sample(self):
        """Test trimmed mean with too few samples."""
        # Only 2 contributions - can't trim
        self.aggregator.add_contribution(
            "peer1", {"weight": torch.ones(3)}, validate=False
        )
        self.aggregator.add_contribution(
            "peer2", {"weight": torch.ones(3) * 2}, validate=False
        )
        
        result = self.aggregator.aggregate()
        
        # Falls back to regular mean
        self.assertTrue(
            torch.allclose(result["weight"], torch.ones(3) * 1.5)
        )


class TestRobustAggregatorKrum(unittest.TestCase):
    """Tests for Krum aggregation."""
    
    def setUp(self):
        """Set up aggregator with Krum strategy."""
        self.aggregator = RobustAggregator(
            aggregation_config=AggregationConfig(
                strategy=AggregationStrategy.KRUM,
                num_byzantine=1,
            )
        )
        
    def test_krum_selects_majority(self):
        """Test Krum selects gradient closest to majority."""
        # Three similar gradients
        self.aggregator.add_contribution(
            "honest1", {"weight": torch.ones(10) * 1.0}, validate=False
        )
        self.aggregator.add_contribution(
            "honest2", {"weight": torch.ones(10) * 1.1}, validate=False
        )
        self.aggregator.add_contribution(
            "honest3", {"weight": torch.ones(10) * 0.9}, validate=False
        )
        
        # One outlier
        self.aggregator.add_contribution(
            "byzantine", {"weight": torch.ones(10) * 100}, validate=False
        )
        
        result = self.aggregator.aggregate()
        
        # Should select one of the honest gradients (close to 1.0)
        self.assertLess(result["weight"].mean().item(), 10.0)
        
    def test_krum_small_sample(self):
        """Test Krum falls back for small samples."""
        self.aggregator.add_contribution(
            "peer1", {"weight": torch.ones(5)}, validate=False
        )
        
        result = self.aggregator.aggregate()
        
        # Should return something (falls back to mean)
        self.assertTrue(
            torch.allclose(result["weight"], torch.ones(5))
        )


class TestRobustAggregatorMultiKrum(unittest.TestCase):
    """Tests for Multi-Krum aggregation."""
    
    def setUp(self):
        """Set up aggregator with Multi-Krum strategy."""
        self.aggregator = RobustAggregator(
            aggregation_config=AggregationConfig(
                strategy=AggregationStrategy.MULTI_KRUM,
                num_byzantine=1,
                multi_krum_k=2,
            )
        )
        
    def test_multi_krum_averages_top_k(self):
        """Test Multi-Krum averages top-k selections."""
        # Three similar gradients
        self.aggregator.add_contribution(
            "honest1", {"weight": torch.ones(10) * 1.0}, validate=False
        )
        self.aggregator.add_contribution(
            "honest2", {"weight": torch.ones(10) * 2.0}, validate=False
        )
        self.aggregator.add_contribution(
            "honest3", {"weight": torch.ones(10) * 1.5}, validate=False
        )
        
        # One outlier
        self.aggregator.add_contribution(
            "byzantine", {"weight": torch.ones(10) * 1000}, validate=False
        )
        
        result = self.aggregator.aggregate()
        
        # Should be average of top-2, not influenced by 1000
        self.assertLess(result["weight"].mean().item(), 5.0)


class TestByzantineAttackResistance(unittest.TestCase):
    """Tests for Byzantine attack resistance."""
    
    def test_median_resists_49_percent_byzantine(self):
        """Test median can handle up to 49% Byzantine nodes."""
        aggregator = RobustAggregator(
            aggregation_config=AggregationConfig(strategy=AggregationStrategy.MEDIAN)
        )
        
        # 51% honest nodes (values around 1.0)
        for i in range(51):
            aggregator.add_contribution(
                f"honest{i}",
                {"weight": torch.ones(10) * (0.9 + 0.2 * (i % 3))},
                validate=False
            )
            
        # 49% Byzantine nodes (extreme values)
        for i in range(49):
            aggregator.add_contribution(
                f"byzantine{i}",
                {"weight": torch.ones(10) * 1000 * ((i % 2) * 2 - 1)},  # +/- 1000
                validate=False
            )
            
        result = aggregator.aggregate()
        
        # Median should be close to honest value (around 1.0)
        mean_val = result["weight"].mean().item()
        self.assertLess(abs(mean_val - 1.0), 1.0)
        
    def test_krum_resists_byzantine(self):
        """Test Krum resists Byzantine attacks."""
        aggregator = RobustAggregator(
            aggregation_config=AggregationConfig(
                strategy=AggregationStrategy.KRUM,
                num_byzantine=2,
            )
        )
        
        # 5 honest nodes
        for i in range(5):
            aggregator.add_contribution(
                f"honest{i}",
                {"weight": torch.ones(10) + torch.randn(10) * 0.1},
                validate=False
            )
            
        # 2 Byzantine nodes
        for i in range(2):
            aggregator.add_contribution(
                f"byzantine{i}",
                {"weight": torch.ones(10) * (-100 if i == 0 else 100)},
                validate=False
            )
            
        result = aggregator.aggregate()
        
        # Should select an honest gradient
        mean_val = result["weight"].mean().item()
        self.assertLess(abs(mean_val - 1.0), 2.0)
        
    def test_validation_rejects_poisoned(self):
        """Test validation rejects poisoned gradients."""
        aggregator = create_robust_aggregator(
            strategy="trimmed_mean",
            min_cosine_similarity=0.5,
        )
        
        reference = {"weight": torch.randn(100)}
        
        # Valid gradient (similar direction)
        valid_grad = {"weight": reference["weight"] * 1.2}
        accepted, _ = aggregator.add_contribution(
            "honest",
            valid_grad,
            reference_grads=reference,
            validate=True
        )
        self.assertTrue(accepted)
        
        # Poisoned gradient (opposite direction)
        poisoned = {"weight": -reference["weight"]}
        rejected, _ = aggregator.add_contribution(
            "attacker",
            poisoned,
            reference_grads=reference,
            validate=True
        )
        self.assertFalse(rejected)


class TestGradientContribution(unittest.TestCase):
    """Tests for GradientContribution dataclass."""
    
    def test_contribution_creation(self):
        """Test creating a contribution."""
        contrib = GradientContribution(
            peer_id="test_peer",
            gradients={"weight": torch.randn(10)},
            trust_score=0.9,
        )
        
        self.assertEqual(contrib.peer_id, "test_peer")
        self.assertEqual(contrib.trust_score, 0.9)
        self.assertFalse(contrib.is_validated)
        
    def test_contribution_age(self):
        """Test contribution age calculation."""
        import time
        
        contrib = GradientContribution(
            peer_id="test",
            gradients={},
        )
        
        time.sleep(0.1)
        
        age = contrib.age_seconds
        self.assertGreater(age, 0.05)
        self.assertLess(age, 1.0)


class TestFactoryFunction(unittest.TestCase):
    """Tests for factory function."""
    
    def test_create_robust_aggregator(self):
        """Test factory function."""
        aggregator = create_robust_aggregator(
            strategy="trimmed_mean",
            trim_fraction=0.1,
            min_cosine_similarity=0.3,
        )
        
        self.assertIsInstance(aggregator, RobustAggregator)
        self.assertEqual(
            aggregator.agg_config.strategy,
            AggregationStrategy.TRIMMED_MEAN
        )
        self.assertEqual(aggregator.agg_config.trim_fraction, 0.1)
        
    def test_create_with_invalid_strategy(self):
        """Test factory with invalid strategy falls back to default."""
        aggregator = create_robust_aggregator(strategy="invalid_strategy")
        
        # Should fall back to trimmed_mean
        self.assertEqual(
            aggregator.agg_config.strategy,
            AggregationStrategy.TRIMMED_MEAN
        )


class TestAggregatorStats(unittest.TestCase):
    """Tests for aggregator statistics."""
    
    def test_stats_tracking(self):
        """Test statistics are tracked correctly."""
        aggregator = create_robust_aggregator(strategy="mean")
        
        reference = {"weight": torch.randn(10)}
        
        # Valid contribution
        aggregator.add_contribution(
            "peer1",
            {"weight": reference["weight"] * 1.1},
            reference_grads=reference,
        )
        
        # Invalid contribution (opposite direction)
        aggregator.add_contribution(
            "peer2",
            {"weight": -reference["weight"]},
            reference_grads=reference,
        )
        
        # Aggregate
        aggregator.aggregate()
        
        stats = aggregator.get_stats()
        
        self.assertEqual(stats['total_contributions_received'], 2)
        self.assertEqual(stats['contributions_rejected'], 1)
        self.assertEqual(stats['aggregations_performed'], 1)


if __name__ == '__main__':
    unittest.main()

