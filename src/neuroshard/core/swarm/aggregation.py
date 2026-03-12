"""
Robust Gradient Aggregation - Byzantine-Tolerant Gradient Verification

Implements robust aggregation for DiLoCo pseudo-gradients:
- Statistical validation of gradients
- Byzantine-tolerant aggregation (Krum, Median, Trimmed Mean)
- Cosine similarity verification
- Gradient magnitude checking

Key Insight: "Since DiLoCo syncs less often, bad gradients are more damaging.
Enhanced verification required."

Supports multiple aggregation strategies:
1. Simple Mean: Fast but vulnerable to Byzantine nodes
2. Coordinate-wise Median: Robust to outliers
3. Trimmed Mean: Removes top/bottom percentiles
4. Krum: Selects gradients closest to majority
5. Multi-Krum: Weighted combination of top-k

Usage:
    aggregator = RobustAggregator(strategy="trimmed_mean", trim_fraction=0.1)
    
    # Validate incoming gradients
    is_valid, reason = aggregator.validate_gradient(peer_grad, local_grad)
    
    if is_valid:
        aggregator.add_contribution(peer_id, peer_grad)
    
    # Get aggregated result
    aggregated = aggregator.aggregate()
"""

import logging
import math
import threading
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple
from enum import Enum
from collections import defaultdict

import torch
import torch.nn.functional as F

logger = logging.getLogger(__name__)


class AggregationStrategy(Enum):
    """Strategy for aggregating gradients."""
    MEAN = "mean"                    # Simple average
    MEDIAN = "median"                # Coordinate-wise median
    TRIMMED_MEAN = "trimmed_mean"    # Remove top/bottom percentiles
    KRUM = "krum"                    # Select closest to majority
    MULTI_KRUM = "multi_krum"        # Weighted top-k
    GEOMETRIC_MEDIAN = "geometric_median"  # L2 geometric median


@dataclass
class ValidationConfig:
    """Configuration for gradient validation."""
    # Cosine similarity
    min_cosine_similarity: float = 0.3    # Minimum alignment with local gradient
    
    # Magnitude
    max_magnitude_ratio: float = 10.0     # Max ratio to local gradient norm
    min_magnitude_ratio: float = 0.1      # Min ratio to local gradient norm
    
    # Variance
    max_variance_ratio: float = 100.0     # Max variance ratio
    
    # Statistical
    zscore_threshold: float = 3.0         # Max z-score for outlier detection
    
    # Trust
    require_signature: bool = False        # Require cryptographic signature
    min_trust_score: float = 0.0          # Minimum trust score for peer


@dataclass
class AggregationConfig:
    """
    Configuration for aggregation.
    
    - use_freshness_weights: Enable sqrt(n) * freshness weighting
      This gives fair influence based on batches processed while
      preventing large nodes from dominating.
    """
    strategy: AggregationStrategy = AggregationStrategy.TRIMMED_MEAN
    
    # Trimmed mean settings
    trim_fraction: float = 0.1            # Fraction to trim from each end
    
    # Krum settings  
    num_byzantine: int = 0                # Expected number of Byzantine nodes
    multi_krum_k: int = 0                 # Number of gradients to select (0 = auto)
    
    # Geometric median settings
    max_iterations: int = 100             # Max iterations for convergence
    tolerance: float = 1e-6               # Convergence tolerance
    
    # Weighting
    use_trust_weights: bool = False       # Weight by peer trust scores
    use_freshness_weights: bool = True    # Weight by sqrt(batches) * freshness


def compute_contribution_weight(batches_processed: int, age_hours: float) -> float:
    """
    Compute contribution weight using sqrt(n) * freshness formula.
    
    Weight = sqrt(batches_processed) * freshness
    
    Freshness decay:
    - < 1 hour:  1.0
    - < 1 day:   0.9
    - < 1 week:  0.7
    - >= 1 week: 0.3
    
    Args:
        batches_processed: Number of batches in this contribution
        age_hours: Age of contribution in hours
        
    Returns:
        Weight value
    """
    # Freshness decay
    if age_hours < 1:
        freshness = 1.0
    elif age_hours < 24:  # < 1 day
        freshness = 0.9
    elif age_hours < 168:  # < 1 week
        freshness = 0.7
    else:
        freshness = 0.3
    
    # sqrt(n) weighting
    return math.sqrt(batches_processed) * freshness


@dataclass
class GradientContribution:
    """
    A gradient contribution from a peer.
    
    Addition:
    - batches_processed: Number of batches this contribution represents
    - weight: Computed as sqrt(batches) * freshness
    
    This gives fair influence based on actual work done while
    preventing large nodes from dominating.
    """
    peer_id: str
    gradients: Dict[str, torch.Tensor]
    timestamp: float = field(default_factory=time.time)
    trust_score: float = 1.0
    signature: Optional[str] = None
    
    # Batch count for sqrt(n) weighting
    batches_processed: int = 1
    
    # Validation results
    is_validated: bool = False
    validation_reason: str = ""
    
    @property
    def age_seconds(self) -> float:
        return time.time() - self.timestamp
    
    @property
    def age_hours(self) -> float:
        """Get age in hours for freshness calculation."""
        return self.age_seconds / 3600.0
    
    @property
    def freshness(self) -> float:
        """
        Compute freshness factor based on age.
        
        Decay schedule:
        - < 1 hour:  1.0 (fresh)
        - < 1 day:   0.9
        - < 1 week:  0.7
        - >= 1 week: 0.3 (stale)
        """
        age = self.age_hours
        if age < 1:
            return 1.0
        elif age < 24:  # < 1 day
            return 0.9
        elif age < 168:  # < 1 week
            return 0.7
        else:
            return 0.3
    
    @property
    def weight(self) -> float:
        """
        Compute contribution weight.
        
        Weight = sqrt(batches_processed) * freshness
        
        This provides fair influence:
        - sqrt(n) prevents large nodes from dominating
        - Freshness prioritizes recent contributions
        """
        return math.sqrt(self.batches_processed) * self.freshness


class GradientValidator:
    """
    Validates incoming gradients against local reference.
    
    Performs multiple checks:
    1. Cosine similarity (direction alignment)
    2. Magnitude ratio (scale)
    3. Variance ratio (distribution)
    4. Z-score outlier detection
    """
    
    def __init__(self, config: Optional[ValidationConfig] = None):
        self.config = config or ValidationConfig()
        
        # Stats
        self.validations_performed = 0
        self.validations_passed = 0
        self.validations_failed = 0
        self.failure_reasons: Dict[str, int] = defaultdict(int)
        
    def validate(
        self,
        submitted_grads: Dict[str, torch.Tensor],
        reference_grads: Dict[str, torch.Tensor],
        peer_trust: float = 1.0,
    ) -> Tuple[bool, str]:
        """
        Validate submitted gradients against reference.
        
        Args:
            submitted_grads: Gradients from peer
            reference_grads: Local reference gradients
            peer_trust: Trust score of submitting peer
            
        Returns:
            (is_valid, reason) tuple
        """
        self.validations_performed += 1
        
        # Check trust score
        if peer_trust < self.config.min_trust_score:
            self._record_failure("low_trust")
            return False, f"Trust score {peer_trust} below minimum {self.config.min_trust_score}"
        
        # Validate each parameter
        for name, submitted in submitted_grads.items():
            if name not in reference_grads:
                continue
            
            reference = reference_grads[name]
            
            # Check 1: Cosine similarity (direction)
            is_valid, reason = self._check_cosine_similarity(submitted, reference, name)
            if not is_valid:
                self._record_failure("cosine_similarity")
                return False, reason
            
            # Check 2: Magnitude ratio (scale)
            is_valid, reason = self._check_magnitude(submitted, reference, name)
            if not is_valid:
                self._record_failure("magnitude")
                return False, reason
            
            # Check 3: Variance ratio (distribution)
            is_valid, reason = self._check_variance(submitted, reference, name)
            if not is_valid:
                self._record_failure("variance")
                return False, reason
        
        self.validations_passed += 1
        return True, "Validation passed"
    
    def _check_cosine_similarity(
        self,
        submitted: torch.Tensor,
        reference: torch.Tensor,
        param_name: str,
    ) -> Tuple[bool, str]:
        """Check cosine similarity between gradients."""
        # Flatten for comparison
        submitted_flat = submitted.flatten()
        reference_flat = reference.flatten()
        
        # Handle zero vectors
        submitted_norm = submitted_flat.norm()
        reference_norm = reference_flat.norm()
        
        if submitted_norm == 0 or reference_norm == 0:
            # Zero gradient - suspicious but might be valid
            if submitted_norm == 0 and reference_norm == 0:
                return True, "Both zero"
            return True, "One zero vector - allowing"
        
        # Compute cosine similarity
        cosine_sim = F.cosine_similarity(
            submitted_flat.unsqueeze(0),
            reference_flat.unsqueeze(0)
        ).item()
        
        if cosine_sim < self.config.min_cosine_similarity:
            return False, (
                f"Cosine similarity {cosine_sim:.3f} below threshold "
                f"{self.config.min_cosine_similarity} for {param_name}"
            )
        
        return True, f"cosine={cosine_sim:.3f}"
    
    def _check_magnitude(
        self,
        submitted: torch.Tensor,
        reference: torch.Tensor,
        param_name: str,
    ) -> Tuple[bool, str]:
        """Check magnitude ratio of gradients."""
        submitted_norm = submitted.norm().item()
        reference_norm = reference.norm().item()
        
        if reference_norm == 0:
            return True, "Reference norm is zero"
        
        ratio = submitted_norm / reference_norm
        
        if ratio > self.config.max_magnitude_ratio:
            return False, (
                f"Magnitude ratio {ratio:.2f} exceeds max "
                f"{self.config.max_magnitude_ratio} for {param_name}"
            )
        
        if ratio < self.config.min_magnitude_ratio:
            return False, (
                f"Magnitude ratio {ratio:.2f} below min "
                f"{self.config.min_magnitude_ratio} for {param_name}"
            )
        
        return True, f"magnitude_ratio={ratio:.2f}"
    
    def _check_variance(
        self,
        submitted: torch.Tensor,
        reference: torch.Tensor,
        param_name: str,
    ) -> Tuple[bool, str]:
        """Check variance ratio of gradients."""
        submitted_var = submitted.var().item()
        reference_var = reference.var().item()
        
        if reference_var == 0:
            return True, "Reference variance is zero"
        
        ratio = submitted_var / reference_var
        
        if ratio > self.config.max_variance_ratio:
            return False, (
                f"Variance ratio {ratio:.2f} exceeds max "
                f"{self.config.max_variance_ratio} for {param_name}"
            )
        
        return True, f"variance_ratio={ratio:.2f}"
    
    def _record_failure(self, reason: str):
        """Record a validation failure."""
        self.validations_failed += 1
        self.failure_reasons[reason] += 1
    
    def get_stats(self) -> Dict[str, Any]:
        """Get validation statistics."""
        success_rate = (
            self.validations_passed / self.validations_performed
            if self.validations_performed > 0 else 0.0
        )
        return {
            'validations_performed': self.validations_performed,
            'validations_passed': self.validations_passed,
            'validations_failed': self.validations_failed,
            'success_rate': success_rate,
            'failure_reasons': dict(self.failure_reasons),
        }


class RobustAggregator:
    """
    Byzantine-tolerant gradient aggregator.
    
    Supports multiple aggregation strategies for robustness
    against malicious or faulty nodes.
    """
    
    def __init__(
        self,
        aggregation_config: Optional[AggregationConfig] = None,
        validation_config: Optional[ValidationConfig] = None,
    ):
        self.agg_config = aggregation_config or AggregationConfig()
        self.validator = GradientValidator(validation_config)
        
        # Contributions
        self.contributions: List[GradientContribution] = []
        self._lock = threading.Lock()
        
        # Stats
        self.aggregations_performed = 0
        self.total_contributions_received = 0
        self.contributions_rejected = 0
    
    def clear(self):
        """Clear all contributions."""
        with self._lock:
            self.contributions.clear()
    
    def add_contribution(
        self,
        peer_id: str,
        gradients: Dict[str, torch.Tensor],
        reference_grads: Optional[Dict[str, torch.Tensor]] = None,
        trust_score: float = 1.0,
        validate: bool = True,
        batches_processed: int = 1,
    ) -> Tuple[bool, str]:
        """
        Add a gradient contribution from a peer.
        
        Args:
            peer_id: ID of contributing peer
            gradients: Gradient tensors from peer
            reference_grads: Local reference for validation
            trust_score: Trust score of peer
            validate: Whether to validate before adding
            batches_processed: Number of batches this contribution represents
        
        Returns:
            (accepted, reason) tuple
        """
        self.total_contributions_received += 1
        
        # Create contribution
        contribution = GradientContribution(
            peer_id=peer_id,
            gradients=gradients,
            trust_score=trust_score,
            batches_processed=batches_processed,
        )
        
        # Validate if reference provided
        if validate and reference_grads is not None:
            is_valid, reason = self.validator.validate(
                gradients,
                reference_grads,
                trust_score
            )
            
            contribution.is_validated = True
            contribution.validation_reason = reason
            
            if not is_valid:
                self.contributions_rejected += 1
                logger.warning(f"Rejected gradient from {peer_id}: {reason}")
                return False, reason
        
        with self._lock:
            self.contributions.append(contribution)
        return True, "Accepted"
    
    def aggregate(
        self,
        local_grads: Optional[Dict[str, torch.Tensor]] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Aggregate all contributions using configured strategy.
        
        Args:
            local_grads: Optional local gradients to include
            
        Returns:
            Aggregated gradients
        """
        if not self.contributions and local_grads is None:
            return {}
        
        # Snapshot contributions under lock, then release for heavy computation
        with self._lock:
            all_contributions = list(self.contributions)
        if local_grads is not None:
            all_contributions.append(GradientContribution(
                peer_id="local",
                gradients=local_grads,
                trust_score=1.0,
                is_validated=True,
            ))
        
        # Select aggregation method
        strategy = self.agg_config.strategy
        
        if strategy == AggregationStrategy.MEAN:
            result = self._aggregate_mean(all_contributions)
        elif strategy == AggregationStrategy.MEDIAN:
            result = self._aggregate_median(all_contributions)
        elif strategy == AggregationStrategy.TRIMMED_MEAN:
            result = self._aggregate_trimmed_mean(all_contributions)
        elif strategy == AggregationStrategy.KRUM:
            result = self._aggregate_krum(all_contributions)
        elif strategy == AggregationStrategy.MULTI_KRUM:
            result = self._aggregate_multi_krum(all_contributions)
        elif strategy == AggregationStrategy.GEOMETRIC_MEDIAN:
            result = self._aggregate_geometric_median(all_contributions)
        else:
            result = self._aggregate_mean(all_contributions)
        
        self.aggregations_performed += 1
        self.clear()  # Thread-safe clear after aggregation
        
        return result
    
    def _aggregate_mean(
        self,
        contributions: List[GradientContribution],
    ) -> Dict[str, torch.Tensor]:
        """
        Mean aggregation with optional sqrt(n) * freshness weighting.
        
        Uses sqrt(batches_processed) * freshness as weights.
        This gives fair influence based on actual work while preventing
        large nodes from dominating.
        """
        if not contributions:
            return {}
        
        # Get all parameter names
        param_names = set()
        for c in contributions:
            param_names.update(c.gradients.keys())
        
        # Average each parameter
        result = {}
        for name in param_names:
            # Filter contributions with this parameter
            relevant = [c for c in contributions if name in c.gradients]
            tensors = [c.gradients[name] for c in relevant]
            
            if tensors:
                # Compute weights
                if self.agg_config.use_trust_weights:
                    # Trust-based weighting
                    weights = torch.tensor([c.trust_score for c in relevant])
                elif self.agg_config.use_freshness_weights:
                    # sqrt(n) * freshness weighting
                    weights = torch.tensor([c.weight for c in relevant])
                else:
                    # Equal weighting
                    weights = torch.ones(len(relevant))
                
                # Normalize weights
                weights = weights / weights.sum()
                
                # Weighted sum
                result[name] = sum(
                    w.item() * t for w, t in zip(weights, tensors)
                )
        
        return result
    
    def _aggregate_median(
        self,
        contributions: List[GradientContribution],
    ) -> Dict[str, torch.Tensor]:
        """Coordinate-wise median aggregation."""
        if not contributions:
            return {}
        
        param_names = set()
        for c in contributions:
            param_names.update(c.gradients.keys())
        
        result = {}
        for name in param_names:
            tensors = [
                c.gradients[name] for c in contributions
                if name in c.gradients
            ]
            if tensors:
                stacked = torch.stack(tensors)
                result[name] = stacked.median(dim=0)[0]
        
        return result
    
    def _aggregate_trimmed_mean(
        self,
        contributions: List[GradientContribution],
    ) -> Dict[str, torch.Tensor]:
        """Trimmed mean aggregation (removes top/bottom percentiles)."""
        if not contributions:
            return {}
        
        trim_fraction = self.agg_config.trim_fraction
        n = len(contributions)
        
        # Number to trim from each end
        trim_count = int(n * trim_fraction)
        if 2 * trim_count >= n:
            trim_count = max(0, n // 2 - 1)
        
        param_names = set()
        for c in contributions:
            param_names.update(c.gradients.keys())
        
        result = {}
        for name in param_names:
            tensors = [
                c.gradients[name] for c in contributions
                if name in c.gradients
            ]
            if tensors:
                stacked = torch.stack(tensors)  # [n, ...]
                
                if trim_count > 0 and len(tensors) > 2 * trim_count:
                    # Sort along first dimension and trim
                    sorted_tensors = stacked.sort(dim=0)[0]
                    trimmed = sorted_tensors[trim_count:-trim_count]
                    result[name] = trimmed.mean(dim=0)
                else:
                    result[name] = stacked.mean(dim=0)
        
        return result
    
    def _aggregate_krum(
        self,
        contributions: List[GradientContribution],
    ) -> Dict[str, torch.Tensor]:
        """
        Krum aggregation - select gradient closest to majority.
        
        Assumes at most f Byzantine nodes out of n.
        Selects the gradient with smallest sum of distances to 
        its n - f - 2 closest neighbors.
        """
        if not contributions:
            return {}
        
        n = len(contributions)
        f = min(self.agg_config.num_byzantine, n - 2)
        
        if n <= 2:
            return self._aggregate_mean(contributions)
        
        # Compute pairwise distances
        distances = self._compute_pairwise_distances(contributions)
        
        # For each gradient, sum distances to n - f - 2 closest
        scores = []
        keep = n - f - 2
        
        for i in range(n):
            # Get distances from i to all others
            dists = distances[i]
            # Sort and sum closest (excluding self which is 0)
            sorted_dists = sorted(dists)
            score = sum(sorted_dists[1:keep+1])  # Skip self (index 0)
            scores.append(score)
        
        # Select gradient with minimum score
        best_idx = min(range(n), key=lambda i: scores[i])
        
        return contributions[best_idx].gradients
    
    def _aggregate_multi_krum(
        self,
        contributions: List[GradientContribution],
    ) -> Dict[str, torch.Tensor]:
        """
        Multi-Krum aggregation - average of top-k Krum selections.
        """
        if not contributions:
            return {}
        
        n = len(contributions)
        k = self.agg_config.multi_krum_k
        if k <= 0 or k >= n:
            k = max(1, n - self.agg_config.num_byzantine)
        
        # Compute Krum scores
        f = min(self.agg_config.num_byzantine, n - 2)
        distances = self._compute_pairwise_distances(contributions)
        
        scores = []
        keep = max(1, n - f - 2)
        
        for i in range(n):
            dists = distances[i]
            sorted_dists = sorted(dists)
            score = sum(sorted_dists[1:keep+1])
            scores.append((score, i))
        
        # Select top-k by score (lower is better)
        scores.sort(key=lambda x: x[0])
        selected_indices = [idx for _, idx in scores[:k]]
        
        # Average selected gradients
        selected = [contributions[i] for i in selected_indices]
        return self._aggregate_mean(selected)
    
    def _aggregate_geometric_median(
        self,
        contributions: List[GradientContribution],
    ) -> Dict[str, torch.Tensor]:
        """
        Geometric median aggregation via Weiszfeld algorithm.
        
        More robust than coordinate-wise median.
        """
        if not contributions:
            return {}
        
        # Start with mean as initial estimate
        result = self._aggregate_mean(contributions)
        
        # Iterative refinement
        for iteration in range(self.agg_config.max_iterations):
            prev_result = {k: v.clone() for k, v in result.items()}
            
            for name in result.keys():
                tensors = [
                    c.gradients[name] for c in contributions
                    if name in c.gradients
                ]
                if not tensors:
                    continue
                
                # Compute weighted update
                current = result[name]
                weights = []
                weighted_sum = torch.zeros_like(current)
                
                for t in tensors:
                    dist = (t - current).norm().item()
                    if dist > 1e-10:
                        w = 1.0 / dist
                        weights.append(w)
                        weighted_sum += w * t
                    else:
                        # Point is at current estimate
                        weights.append(1e10)
                        weighted_sum += 1e10 * t
                
                total_weight = sum(weights)
                if total_weight > 0:
                    result[name] = weighted_sum / total_weight
            
            # Check convergence
            total_change = sum(
                (result[k] - prev_result[k]).norm().item()
                for k in result.keys()
            )
            if total_change < self.agg_config.tolerance:
                break
        
        return result
    
    def _compute_pairwise_distances(
        self,
        contributions: List[GradientContribution],
    ) -> List[List[float]]:
        """Compute pairwise L2 distances between all gradients."""
        n = len(contributions)
        distances = [[0.0] * n for _ in range(n)]
        
        for i in range(n):
            for j in range(i + 1, n):
                # Sum squared distances across all parameters
                total_dist = 0.0
                for name in contributions[i].gradients.keys():
                    if name in contributions[j].gradients:
                        diff = contributions[i].gradients[name] - contributions[j].gradients[name]
                        total_dist += diff.norm().item() ** 2
                
                dist = math.sqrt(total_dist)
                distances[i][j] = dist
                distances[j][i] = dist
        
        return distances
    
    def get_stats(self) -> Dict[str, Any]:
        """Get aggregator statistics."""
        return {
            'aggregations_performed': self.aggregations_performed,
            'total_contributions_received': self.total_contributions_received,
            'contributions_rejected': self.contributions_rejected,
            'current_contributions': len(self.contributions),
            'validation_stats': self.validator.get_stats(),
        }


# ==================== FACTORY FUNCTIONS ====================

def create_robust_aggregator(
    strategy: str = "trimmed_mean",
    trim_fraction: float = 0.1,
    num_byzantine: int = 0,
    min_cosine_similarity: float = 0.3,
    **kwargs,
) -> RobustAggregator:
    """
    Factory function to create a robust aggregator.
    
    Args:
        strategy: Aggregation strategy name
        trim_fraction: Fraction to trim (for trimmed_mean)
        num_byzantine: Expected Byzantine nodes (for Krum)
        min_cosine_similarity: Minimum gradient alignment
        **kwargs: Additional config options
        
    Returns:
        Configured RobustAggregator
    """
    try:
        agg_strategy = AggregationStrategy(strategy)
    except ValueError:
        agg_strategy = AggregationStrategy.TRIMMED_MEAN
    
    agg_config = AggregationConfig(
        strategy=agg_strategy,
        trim_fraction=trim_fraction,
        num_byzantine=num_byzantine,
    )
    
    val_config = ValidationConfig(
        min_cosine_similarity=min_cosine_similarity,
    )
    
    return RobustAggregator(
        aggregation_config=agg_config,
        validation_config=val_config,
    )
