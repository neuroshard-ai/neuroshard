"""
Dynamic Architecture Scaler - Core of NeuroShard's Unlimited Scaling

This module calculates optimal model architecture (width + depth) based on
total network capacity, following empirical scaling laws from GPT-3 and Chinchilla.

Key Principles:
1. Width grows faster than depth (empirically more efficient)
2. Both dimensions scale with network size
3. Architecture updates are gradual and automated
4. No fixed model sizes - purely capacity-driven
"""

import math
import logging
from typing import Dict, Optional, Tuple
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class ModelArchitecture:
    """Configuration for a dynamically-sized model."""
    hidden_dim: int
    intermediate_dim: int
    num_layers: int
    num_heads: int
    num_kv_heads: int
    vocab_size: int = 32000  # Initial size - expands automatically as tokenizer grows (unlimited)
    max_seq_len: int = 2048
    dropout: float = 0.0
    rope_theta: float = 10000.0
    
    def __post_init__(self):
        """Validate architecture constraints."""
        assert self.hidden_dim % self.num_heads == 0, \
            f"hidden_dim ({self.hidden_dim}) must be divisible by num_heads ({self.num_heads})"
        assert self.num_heads % self.num_kv_heads == 0, \
            f"num_heads ({self.num_heads}) must be divisible by num_kv_heads ({self.num_kv_heads})"
        assert self.hidden_dim > 0 and self.num_layers > 0, \
            "Dimensions must be positive"
        # head_dim must be even for Rotary Position Embeddings (RoPE)
        head_dim = self.hidden_dim // self.num_heads
        assert head_dim % 2 == 0, \
            f"head_dim ({head_dim}) must be even for RoPE. Adjust hidden_dim or num_heads."
    
    def estimate_params(self) -> int:
        """Estimate total parameters in this architecture."""
        # Embedding
        embed_params = self.vocab_size * self.hidden_dim
        
        # Per-layer params (attention + FFN + norms)
        # Attention: Q, K, V, O projections
        attn_params = 4 * self.hidden_dim * self.hidden_dim
        # FFN: gate, up, down (SwiGLU)
        ffn_params = 3 * self.hidden_dim * self.intermediate_dim
        # Norms (RMSNorm has 1 param per dim)
        norm_params = 2 * self.hidden_dim
        
        layer_params = attn_params + ffn_params + norm_params
        
        # LM head (tied with embedding, so don't count twice)
        # Final norm
        final_norm_params = self.hidden_dim
        
        total = embed_params + (layer_params * self.num_layers) + final_norm_params
        return total
    
    def estimate_memory_mb(self) -> float:
        """
        Estimate memory needed for this architecture.
        Includes weights + gradients + optimizer states (Adam).
        """
        params = self.estimate_params()
        # 4 bytes per param (FP32) × 4 (weights + grads + Adam m + v)
        return (params * 4 * 4) / (1024 * 1024)
    
    def to_dict(self) -> Dict:
        """Serialize for checkpointing."""
        return {
            "hidden_dim": self.hidden_dim,
            "intermediate_dim": self.intermediate_dim,
            "num_layers": self.num_layers,
            "num_heads": self.num_heads,
            "num_kv_heads": self.num_kv_heads,
            "vocab_size": self.vocab_size,
            "max_seq_len": self.max_seq_len,
            "dropout": self.dropout,
            "rope_theta": self.rope_theta,
        }
    
    @classmethod
    def from_dict(cls, data: Dict):
        """Deserialize from checkpoint."""
        return cls(**data)


def calculate_optimal_architecture(
    total_network_memory_mb: float,
    utilization_factor: float = 0.6
) -> ModelArchitecture:
    """
    Calculate optimal model architecture based on total network capacity.
    
    Follows empirical scaling laws:
    - Chinchilla: Optimal compute scales as params^1.0 × tokens^1.0
    - GPT-3: Width grows faster than depth for efficiency
    - Empirical: depth ∝ params^0.4, width ∝ params^0.6
    
    Args:
        total_network_memory_mb: Total available memory across all nodes
        utilization_factor: Fraction of memory to use (default 0.6 for safety)
    
    Returns:
        ModelArchitecture optimized for this capacity
    """
    # STABILITY FIX: Round memory to nearest 500MB tier to prevent architecture
    # changes due to small memory fluctuations between runs.
    # This ensures checkpoints remain compatible across restarts.
    MEMORY_TIER_MB = 500
    rounded_memory_mb = math.floor(total_network_memory_mb / MEMORY_TIER_MB) * MEMORY_TIER_MB
    # Ensure at least 500MB (minimum tier)
    rounded_memory_mb = max(MEMORY_TIER_MB, rounded_memory_mb)
    
    # Calculate parameter budget
    # Memory per param: 16 bytes (FP32 weight + grad + 2× Adam states)
    usable_memory_mb = rounded_memory_mb * utilization_factor
    max_params = int((usable_memory_mb * 1024 * 1024) / 16)
    params_millions = max_params / 1e6
    
    logger.info(f"Network capacity: {total_network_memory_mb:.0f}MB (rounded to {rounded_memory_mb}MB tier) → {params_millions:.0f}M params budget")
    
    # Scaling formulas based on parameter count
    # These are empirically derived from successful LLM architectures
    
    if params_millions < 50:  # < 50M params (tiny network)
        # Example: 1-5 nodes with 2GB each
        # Architecture: Similar to GPT-2 Small
        hidden_dim = 384
        num_layers = max(6, int(12 * (params_millions / 50) ** 0.5))
        num_heads = 6
        
    elif params_millions < 150:  # 50M - 150M params (small network)
        # Example: 5-20 nodes
        # Architecture: GPT-2 Medium scale
        hidden_dim = 512
        num_layers = max(8, int(16 * (params_millions / 150) ** 0.5))
        num_heads = 8
        
    elif params_millions < 500:  # 150M - 500M params (medium network)
        # Example: 20-50 nodes
        # Architecture: GPT-2 Large → GPT-2 XL
        hidden_dim = int(640 + 384 * ((params_millions - 150) / 350))
        num_layers = max(12, int(24 * (params_millions / 500) ** 0.5))
        num_heads = max(8, hidden_dim // 64)
        
    elif params_millions < 2000:  # 500M - 2B params (large network)
        # Example: 50-200 nodes
        # Architecture: GPT-3 Small → Medium
        hidden_dim = int(1024 * (params_millions / 500) ** 0.6)
        num_layers = max(18, int(32 * (params_millions / 2000) ** 0.4))
        num_heads = max(12, hidden_dim // 64)
        
    elif params_millions < 10000:  # 2B - 10B params (very large)
        # Example: 200-1000 nodes
        # Architecture: GPT-3 Medium → Large
        hidden_dim = int(1536 * (params_millions / 2000) ** 0.6)
        num_layers = max(24, int(48 * (params_millions / 10000) ** 0.4))
        num_heads = max(16, hidden_dim // 64)
        
    elif params_millions < 100000:  # 10B - 100B params (frontier)
        # Example: 1000-10000 nodes
        # Architecture: GPT-3 XL → GPT-4 scale
        hidden_dim = int(2048 * (params_millions / 10000) ** 0.6)
        num_layers = max(32, int(64 * (params_millions / 100000) ** 0.35))
        num_heads = max(20, hidden_dim // 96)
        
    else:  # > 100B params (mega-scale)
        # Example: 10000+ nodes
        # Architecture: Beyond GPT-4
        hidden_dim = int(4096 * (params_millions / 100000) ** 0.5)
        num_layers = max(48, int(80 * (params_millions / 100000) ** 0.3))
        num_heads = max(32, hidden_dim // 128)
    
    # Ensure hidden_dim is divisible by num_heads AND head_dim is even (for RoPE)
    # head_dim = hidden_dim // num_heads must be even for rotary embeddings
    head_dim = hidden_dim // num_heads
    if head_dim % 2 != 0:
        head_dim = ((head_dim + 1) // 2) * 2  # Round up to nearest even
    hidden_dim = head_dim * num_heads
    
    # GQA: Use 1/3 to 1/4 the number of KV heads
    # IMPORTANT: num_heads must be divisible by num_kv_heads
    num_kv_heads = max(1, num_heads // 3)
    # Round to ensure divisibility
    while num_heads % num_kv_heads != 0:
        num_kv_heads -= 1
        if num_kv_heads < 1:
            num_kv_heads = 1
            break
    
    # SwiGLU intermediate dimension (standard: 8/3 × hidden for gated FFN)
    intermediate_dim = int(hidden_dim * 8 / 3)
    # Round to nearest multiple of 64 for efficiency
    intermediate_dim = ((intermediate_dim + 63) // 64) * 64
    
    arch = ModelArchitecture(
        hidden_dim=hidden_dim,
        intermediate_dim=intermediate_dim,
        num_layers=num_layers,
        num_heads=num_heads,
        num_kv_heads=num_kv_heads,
    )
    
    # Verify we're within budget
    actual_memory = arch.estimate_memory_mb()
    if actual_memory > usable_memory_mb:
        # Scale down slightly
        scale_factor = math.sqrt(usable_memory_mb / actual_memory)
        hidden_dim = int(hidden_dim * scale_factor)
        # Ensure head_dim is even (for RoPE)
        head_dim = hidden_dim // num_heads
        if head_dim % 2 != 0:
            head_dim = (head_dim // 2) * 2  # Round down to nearest even
        hidden_dim = head_dim * num_heads
        
        arch = ModelArchitecture(
            hidden_dim=hidden_dim,
            intermediate_dim=int(hidden_dim * 8 / 3),
            num_layers=num_layers,
            num_heads=num_heads,
            num_kv_heads=num_kv_heads,
        )
    
    logger.info(f"Calculated architecture: {arch.num_layers}L × {arch.hidden_dim}H "
                f"({arch.estimate_params()/1e6:.0f}M params, {arch.estimate_memory_mb():.0f}MB)")
    
    return arch


def should_upgrade_architecture(
    current: ModelArchitecture,
    new: ModelArchitecture,
    min_improvement: float = 1.3
) -> Tuple[bool, str]:
    """
    Determine if architecture upgrade is worthwhile.
    
    Args:
        current: Current architecture
        new: Proposed new architecture
        min_improvement: Minimum parameter ratio to trigger upgrade (default 1.3 = 30% improvement)
    
    Returns:
        (should_upgrade, reason)
    """
    current_params = current.estimate_params()
    new_params = new.estimate_params()
    improvement_ratio = new_params / current_params
    
    if improvement_ratio < min_improvement:
        return False, f"Improvement {improvement_ratio:.2f}x < threshold {min_improvement}x"
    
    # Additional checks: ensure balanced scaling
    width_ratio = new.hidden_dim / current.hidden_dim
    depth_ratio = new.num_layers / current.num_layers
    
    if width_ratio < 1.0 or depth_ratio < 1.0:
        return False, "Architecture would shrink (not allowed)"
    
    # Warn if extremely imbalanced
    if width_ratio > 2.0 and depth_ratio < 1.1:
        logger.warning(f"Width growing much faster than depth ({width_ratio:.1f}x vs {depth_ratio:.1f}x)")
    
    reason = (f"Upgrade justified: {current_params/1e6:.0f}M → {new_params/1e6:.0f}M params "
              f"({improvement_ratio:.2f}x, width {width_ratio:.2f}x, depth {depth_ratio:.2f}x)")
    
    return True, reason


def estimate_memory_per_layer(arch: ModelArchitecture) -> float:
    """
    Estimate memory needed per layer (useful for layer assignment).
    
    Returns memory in MB.
    """
    # Per-layer params
    attn_params = 4 * arch.hidden_dim * arch.hidden_dim
    ffn_params = 3 * arch.hidden_dim * arch.intermediate_dim
    norm_params = 2 * arch.hidden_dim
    
    layer_params = attn_params + ffn_params + norm_params
    
    # Memory: params × 16 bytes (weights + grads + Adam states)
    return (layer_params * 16) / (1024 * 1024)


def estimate_embedding_memory_mb(hidden_dim: int, vocab_capacity: int) -> float:
    """
    Estimate memory for embedding and LM head based on vocab capacity.
    
    Args:
        hidden_dim: Model hidden dimension
        vocab_capacity: Current vocabulary capacity (NOT tokenizer vocab_size)
    
    Returns:
        Memory in MB for embedding + lm_head (weights + gradients + optimizer states)
    """
    # Embedding: vocab_capacity × hidden_dim params
    # LM head: vocab_capacity × hidden_dim params (not tied)
    # Total: 2 × vocab_capacity × hidden_dim
    embed_params = 2 * vocab_capacity * hidden_dim
    
    # Memory: params × 16 bytes (weights 4B + grads 4B + Adam m 4B + Adam v 4B)
    return (embed_params * 16) / (1024 * 1024)


def calculate_layer_assignment(
    available_memory_mb: float,
    arch: ModelArchitecture,
    safety_factor: float = 0.6,
    vocab_capacity: int = 32000,
    training_mode: bool = True,
    needs_embedding: bool = True,
    device_type: str = "cpu"
) -> int:
    """
    Calculate how many layers a node can hold.
    
    COMPUTE-BALANCED ASSIGNMENT:
    We calculate based on BOTH memory AND compute throughput.
    A GPU can process layers 10-50x faster than CPU, so we adjust accordingly.
    
    This ensures pipeline stages are balanced for similar processing time.
    
    Args:
        available_memory_mb: Node's available memory
        arch: Current network architecture
        safety_factor: Use only 60% of memory for safety (GPU) or 50% (CPU)
        vocab_capacity: Current vocab capacity for embedding/LM head (dynamic!)
        training_mode: If True, reserve more memory for gradients/optimizer
        needs_embedding: If True, reserve memory for embedding/LM head (Driver/Validator only!)
                        Workers (middle layers) don't need embedding and can hold MORE layers!
        device_type: "cuda" or "cpu" - affects compute throughput factor
    
    Returns:
        Number of layers this node can hold (balanced for compute throughput)
    """
    usable_memory = available_memory_mb * safety_factor
    memory_per_layer = estimate_memory_per_layer(arch)
    
    # DYNAMIC: Calculate actual embedding/LM head memory based on vocab capacity
    # This is critical for dynamic vocabulary - as vocab grows, less memory for layers!
    # NOTE: Only Drivers (Layer 0) and Validators (Last Layer) need embedding memory!
    # Workers (middle layers) skip this and can hold MORE layers!
    if needs_embedding:
        embedding_memory = estimate_embedding_memory_mb(arch.hidden_dim, vocab_capacity)
    else:
        embedding_memory = 0  # Workers don't need embedding/LM head!
    
    # TRAINING vs INFERENCE memory overhead
    # Training needs: forward activations + backward gradients + optimizer peak
    # With gradient checkpointing (always on for >16 layers), activations are minimal
    if training_mode:
        # Reserve 20% for training overhead (checkpointed activations, optimizer peaks)
        # Reduced from 35% since gradient checkpointing is now always enabled
        training_overhead_ratio = 0.20
    else:
        # Inference only needs 5% buffer
        training_overhead_ratio = 0.05
    
    training_overhead = usable_memory * training_overhead_ratio
    usable_for_layers = usable_memory - embedding_memory - training_overhead
    
    if usable_for_layers <= 0:
        # Not enough memory even for embedding - return minimum
        logger.warning(f"[MEMORY] Very limited memory: {usable_memory:.0f}MB usable, "
                      f"{embedding_memory:.0f}MB for vocab, {training_overhead:.0f}MB overhead")
        return 1
    
    memory_layers = max(1, int(usable_for_layers / memory_per_layer))
    
    # COMPUTE-BALANCED ASSIGNMENT
    # GPU is roughly 10-50x faster than CPU for neural network forward/backward
    # We limit layers based on compute throughput to balance the pipeline
    # This prevents slow nodes from becoming bottlenecks
    if device_type == "cuda":
        # GPU can process layers fast - use full memory capacity
        compute_factor = 1.0
        # But cap at a reasonable amount for pipeline balance
        compute_max_layers = memory_layers  # GPU: memory-limited
    else:
        # CPU is slow - limit to just a few layers regardless of memory
        # A CPU processing 3-4 layers takes similar time as GPU processing 30-40
        compute_factor = 0.15  # CPU gets ~15% of its memory capacity
        # Hard cap to prevent CPU bottleneck even if lots of memory
        compute_max_layers = min(4, memory_layers)  # CPU: max 4 layers
    
    # Apply compute factor
    compute_balanced_layers = max(1, int(memory_layers * compute_factor))
    max_layers = min(compute_balanced_layers, compute_max_layers)
    
    # Log the calculation for transparency
    role = "Driver/Validator" if needs_embedding else "Worker"
    logger.debug(f"[MEMORY] Layer calc ({role}): {available_memory_mb:.0f}MB × {safety_factor} = {usable_memory:.0f}MB usable")
    if needs_embedding:
        logger.debug(f"[MEMORY]   - Embedding ({vocab_capacity:,} vocab): {embedding_memory:.0f}MB")
    else:
        logger.debug(f"[MEMORY]   - No embedding needed (Worker)")
    logger.debug(f"[MEMORY]   - Training overhead ({training_overhead_ratio*100:.0f}%): {training_overhead:.0f}MB")
    logger.debug(f"[MEMORY]   - Memory allows: {memory_layers} layers")
    logger.debug(f"[MEMORY]   - Compute balanced ({device_type}): {max_layers} layers (factor={compute_factor})")
    
    return max_layers

