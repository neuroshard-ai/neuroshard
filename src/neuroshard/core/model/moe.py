"""
Mixture of Experts (MoE) for NeuroShard

Sparse MoE layers with distributed expert support. Scales from
1 node (all experts local) to thousands of nodes (experts distributed).

Components:
- MoEConfig: Configuration dataclass
- MoERouter: Token-to-expert routing (top-k selection)
- Expert: Single FFN expert (SwiGLU)
- DistributedMoELayer: Full MoE layer with local/remote expert support
- MoEDecoderLayer: Transformer layer with attention + MoE FFN

Scaling:
- 1 node: All experts local, no remote calls
- N nodes: Experts distributed, remote calls when needed
- Memory-based assignment: Small nodes hold fewer experts

Router Sync:
- Routers synced via DiLoCo (part of named_parameters)
- Small weights (~6KB per layer)
- Gradients computed locally
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import logging
from typing import Optional, Tuple, Dict, List, Any
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


# =============================================================================
# MoE CONFIGURATION
# =============================================================================

@dataclass
class MoEConfig:
    """Configuration for Mixture of Experts layers."""
    
    # Expert architecture
    num_experts: int = 8                    # Total experts per layer
    num_experts_per_token: int = 2          # Top-k routing (k experts per token)
    expert_capacity_factor: float = 1.25    # Capacity buffer (1.25 = 25% extra)
    
    # Router configuration
    router_jitter_noise: float = 0.01       # Noise for exploration during training
    router_z_loss_coef: float = 0.001       # Router z-loss coefficient
    router_aux_loss_coef: float = 0.01      # Load balancing loss coefficient
    
    # Expert dimensions (inherited from main config)
    hidden_dim: int = 768
    intermediate_dim: int = 2048
    
    # Distributed settings
    local_experts: List[int] = field(default_factory=list)  # Which experts this node holds
    enable_expert_parallelism: bool = True   # Enable distributed expert routing
    expert_dropout: float = 0.0              # Dropout on expert outputs
    
    # Token dropping
    drop_tokens: bool = True                 # Drop tokens that exceed capacity
    pad_to_capacity: bool = False            # Pad batches to capacity
    
    @property
    def expert_capacity(self) -> int:
        """Tokens per expert (with capacity factor)."""
        # For batch_size * seq_len tokens, each expert handles:
        # (tokens * top_k / num_experts) * capacity_factor
        # This is computed dynamically per forward pass
        return -1  # Computed dynamically


# =============================================================================
# ROUTER
# =============================================================================

class MoERouter(nn.Module):
    """
    Token-to-expert routing network.
    
    Implements Top-K routing with auxiliary losses for load balancing.
    The router is a simple linear layer that produces expert probabilities.
    """
    
    def __init__(self, config: MoEConfig):
        super().__init__()
        self.config = config
        self.num_experts = config.num_experts
        self.top_k = config.num_experts_per_token
        
        # Router is just a linear layer
        self.gate = nn.Linear(config.hidden_dim, config.num_experts, bias=False)
        
        # Initialize with small values for stable training
        nn.init.kaiming_uniform_(self.gate.weight, a=math.sqrt(5))
    
    def forward(
        self, 
        hidden_states: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Route tokens to experts.
        
        Args:
            hidden_states: [batch_size, seq_len, hidden_dim]
            
        Returns:
            router_probs: [batch_size, seq_len, num_experts] - Full routing probabilities
            routing_weights: [batch_size, seq_len, top_k] - Weights for selected experts
            selected_experts: [batch_size, seq_len, top_k] - Indices of selected experts
            aux_losses: Dict with 'load_balancing_loss' and 'router_z_loss'
        """
        batch_size, seq_len, hidden_dim = hidden_states.shape
        
        # Flatten for routing
        hidden_flat = hidden_states.view(-1, hidden_dim)  # [B*S, H]
        
        # Compute router logits
        router_logits = self.gate(hidden_flat)  # [B*S, num_experts]
        
        # Add jitter noise during training for exploration
        if self.training and self.config.router_jitter_noise > 0:
            noise = torch.randn_like(router_logits) * self.config.router_jitter_noise
            router_logits = router_logits + noise
        
        # Compute router probabilities (for load balancing loss)
        router_probs = F.softmax(router_logits, dim=-1)  # [B*S, num_experts]
        
        # Top-k selection
        routing_weights, selected_experts = torch.topk(
            router_probs, self.top_k, dim=-1
        )  # Both: [B*S, top_k]
        
        # Renormalize routing weights to sum to 1
        routing_weights = routing_weights / routing_weights.sum(dim=-1, keepdim=True)
        
        # Compute auxiliary losses
        aux_losses = self._compute_aux_losses(router_logits, router_probs, selected_experts)
        
        # Reshape outputs
        router_probs = router_probs.view(batch_size, seq_len, self.num_experts)
        routing_weights = routing_weights.view(batch_size, seq_len, self.top_k)
        selected_experts = selected_experts.view(batch_size, seq_len, self.top_k)
        
        return router_probs, routing_weights, selected_experts, aux_losses
    
    def _compute_aux_losses(
        self,
        router_logits: torch.Tensor,
        router_probs: torch.Tensor,
        selected_experts: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """Compute auxiliary losses for training stability."""
        
        losses = {}
        
        # 1. Load Balancing Loss (encourages uniform expert utilization)
        # Penalizes when experts have very different utilization
        if self.config.router_aux_loss_coef > 0:
            # Fraction of tokens routed to each expert
            num_tokens = router_probs.shape[0]
            expert_mask = F.one_hot(selected_experts, self.num_experts)  # [B*S, top_k, num_experts]
            expert_mask = expert_mask.sum(dim=1)  # [B*S, num_experts] - count per expert
            tokens_per_expert = expert_mask.float().mean(dim=0)  # [num_experts]
            
            # Router probability assigned to each expert
            router_prob_per_expert = router_probs.mean(dim=0)  # [num_experts]
            
            # Load balancing loss = num_experts * sum(tokens_frac * prob_frac)
            # This encourages tokens_frac ≈ prob_frac ≈ 1/num_experts
            load_balancing_loss = (
                self.num_experts * 
                (tokens_per_expert * router_prob_per_expert).sum()
            )
            losses['load_balancing_loss'] = load_balancing_loss * self.config.router_aux_loss_coef
        else:
            losses['load_balancing_loss'] = torch.tensor(0.0, device=router_logits.device)
        
        # 2. Router Z-Loss (prevents router logits from growing too large)
        if self.config.router_z_loss_coef > 0:
            router_z_loss = torch.logsumexp(router_logits, dim=-1).pow(2).mean()
            losses['router_z_loss'] = router_z_loss * self.config.router_z_loss_coef
        else:
            losses['router_z_loss'] = torch.tensor(0.0, device=router_logits.device)
        
        return losses


# =============================================================================
# EXPERT LAYER (SwiGLU FFN)
# =============================================================================

class Expert(nn.Module):
    """
    Single expert network (SwiGLU FFN).
    
    This is the same as NeuroMLP but isolated for MoE distribution.
    Each node holds a subset of experts.
    """
    
    def __init__(self, hidden_dim: int, intermediate_dim: int):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.intermediate_dim = intermediate_dim
        
        # SwiGLU: gate * silu(up)
        self.gate_proj = nn.Linear(hidden_dim, intermediate_dim, bias=False)
        self.up_proj = nn.Linear(hidden_dim, intermediate_dim, bias=False)
        self.down_proj = nn.Linear(intermediate_dim, hidden_dim, bias=False)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through expert."""
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))
    
    @property
    def num_params(self) -> int:
        """Number of parameters in this expert."""
        return 3 * self.hidden_dim * self.intermediate_dim


# =============================================================================
# DISTRIBUTED MOE LAYER
# =============================================================================

class DistributedMoELayer(nn.Module):
    """
    Mixture of Experts layer with distributed expert support.
    
    This layer can operate in two modes:
    1. LOCAL: All experts are on this node (for small networks)
    2. DISTRIBUTED: Some experts are remote (for large networks)
    
    The router always runs locally. Expert computation is either
    local or forwarded via gRPC to the node holding that expert.
    """
    
    def __init__(
        self,
        config: MoEConfig,
        layer_idx: int,
        expert_forward_fn: Optional[callable] = None
    ):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.expert_forward_fn = expert_forward_fn  # For remote expert calls
        
        # Router (always local)
        self.router = MoERouter(config)
        
        # Local experts (only those assigned to this node)
        self.local_experts: nn.ModuleDict = nn.ModuleDict()
        self._local_expert_ids: List[int] = []
        
        # Expert dropout
        self.expert_dropout = nn.Dropout(config.expert_dropout)
        
        # Statistics tracking
        self.register_buffer('expert_counts', torch.zeros(config.num_experts))
        self.register_buffer('total_tokens', torch.tensor(0))
        self._missing_expert_count = 0  # Track missing expert calls
    
    def initialize_local_experts(self, expert_ids: List[int]):
        """
        Initialize experts that this node will hold.
        
        Args:
            expert_ids: List of expert indices to hold locally
        """
        self._local_expert_ids = expert_ids
        for expert_id in expert_ids:
            self.local_experts[str(expert_id)] = Expert(
                self.config.hidden_dim,
                self.config.intermediate_dim
            )
        logger.info(f"[MoE] Layer {self.layer_idx}: Initialized {len(expert_ids)} local experts: {expert_ids}")
    
    def is_expert_local(self, expert_id: int) -> bool:
        """Check if an expert is held locally."""
        return str(expert_id) in self.local_experts
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        remote_expert_forward: Optional[callable] = None
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        MoE forward pass with distributed expert support.
        
        Args:
            hidden_states: [batch_size, seq_len, hidden_dim]
            remote_expert_forward: Optional async function for remote experts
            
        Returns:
            output: [batch_size, seq_len, hidden_dim]
            aux_data: Dict with routing stats and losses
        """
        batch_size, seq_len, hidden_dim = hidden_states.shape
        device = hidden_states.device
        
        # 1. Route tokens to experts
        router_probs, routing_weights, selected_experts, aux_losses = self.router(hidden_states)
        # routing_weights: [B, S, top_k]
        # selected_experts: [B, S, top_k]
        
        # 2. Compute expert capacity
        num_tokens = batch_size * seq_len
        tokens_per_expert = num_tokens * self.config.num_experts_per_token / self.config.num_experts
        expert_capacity = int(tokens_per_expert * self.config.expert_capacity_factor)
        expert_capacity = max(expert_capacity, self.config.num_experts_per_token)
        
        # 3. Initialize output tensor
        output = torch.zeros_like(hidden_states)
        
        # 4. Process experts with local-first optimization
        # Strategy: Process all local experts first, then batch remote calls
        # This minimizes blocking on network I/O
        
        # Collect work items: (expert_id, mask, input, weights)
        work_items: List[Tuple[int, torch.Tensor, torch.Tensor, torch.Tensor]] = []
        
        for k in range(self.config.num_experts_per_token):
            expert_indices = selected_experts[:, :, k]  # [B, S]
            expert_weights = routing_weights[:, :, k]   # [B, S]
            
            for expert_id in range(self.config.num_experts):
                mask = (expert_indices == expert_id)  # [B, S]
                if not mask.any():
                    continue
                
                expert_input = hidden_states[mask]  # [num_tokens_for_expert, H]
                
                # Apply capacity limit if needed
                if self.config.drop_tokens and len(expert_input) > expert_capacity:
                    expert_input = expert_input[:expert_capacity]
                    indices = mask.nonzero(as_tuple=True)
                    kept_mask = torch.zeros_like(mask)
                    for i in range(min(expert_capacity, len(indices[0]))):
                        kept_mask[indices[0][i], indices[1][i]] = True
                    mask = kept_mask
                
                weights = expert_weights[mask].unsqueeze(-1)  # [num_tokens, 1]
                work_items.append((expert_id, mask, expert_input, weights))
        
        # Separate local and remote work
        local_work = [(eid, m, inp, w) for eid, m, inp, w in work_items if self.is_expert_local(eid)]
        remote_work = [(eid, m, inp, w) for eid, m, inp, w in work_items if not self.is_expert_local(eid)]
        
        # Helper: accumulate expert output into the output tensor using proper indexing.
        # Boolean indexing (output[mask]) creates a COPY in PyTorch, so in-place += on
        # the copy never writes back. We use nonzero flat indices instead.
        output_flat = output.view(-1, hidden_dim)
        
        def _accumulate_expert_output(mask, expert_output, weights):
            """Write expert output into the flat output tensor via explicit indices."""
            indices = mask.nonzero(as_tuple=False)  # [N, 2] for [batch, seq]
            n = min(expert_output.shape[0], indices.shape[0])
            if n == 0:
                return
            flat_idx = indices[:n, 0] * seq_len + indices[:n, 1]
            output_flat[flat_idx] += weights[:n] * expert_output[:n]
        
        # Process LOCAL experts first (zero network latency)
        for expert_id, mask, expert_input, weights in local_work:
            expert_output = self.local_experts[str(expert_id)](expert_input)
            expert_output = self.expert_dropout(expert_output)
            
            if expert_output.shape[0] < weights.shape[0]:
                weights = weights[:expert_output.shape[0]]
            _accumulate_expert_output(mask, expert_output, weights)
            
            if self.training:
                self.expert_counts[expert_id] += expert_output.shape[0]
                self.total_tokens += expert_output.shape[0]
        
        # Process REMOTE experts
        # TODO: Future optimization - use ThreadPoolExecutor for parallel gRPC calls
        # For now, process sequentially but after all local work is done
        for expert_id, mask, expert_input, weights in remote_work:
            if remote_expert_forward is not None:
                try:
                    expert_output = remote_expert_forward(
                        self.layer_idx, expert_id, expert_input
                    )
                except Exception as e:
                    logger.debug(f"[MoE] Remote expert {expert_id} call failed: {e}")
                    # Zero contribution instead of pass-through to avoid corrupting
                    # the model with unprocessed hidden states
                    expert_output = torch.zeros_like(expert_input)
                    self._missing_expert_count += 1
            else:
                if not hasattr(self, '_missing_logged'):
                    logger.warning(f"[MoE] Expert {expert_id} not available locally and no "
                                   f"remote forward function provided — using zero contribution")
                    self._missing_logged = True
                # Zero contribution: mathematically safe, avoids corrupting the model
                expert_output = torch.zeros_like(expert_input)
                self._missing_expert_count += 1
            
            expert_output = self.expert_dropout(expert_output)
            
            if expert_output.shape[0] < weights.shape[0]:
                weights = weights[:expert_output.shape[0]]
            _accumulate_expert_output(mask, expert_output, weights)
            
            if self.training:
                self.expert_counts[expert_id] += expert_output.shape[0]
                self.total_tokens += expert_output.shape[0]
        
        # 5. Compile auxiliary data
        aux_data = {
            'aux_losses': aux_losses,
            'expert_capacity': expert_capacity,
            'local_experts': self._local_expert_ids,
            'router_probs': router_probs,
        }
        
        return output, aux_data
    
    def get_expert_utilization(self) -> Dict[int, float]:
        """Get utilization statistics for each expert."""
        if self.total_tokens.item() == 0:
            return {i: 0.0 for i in range(self.config.num_experts)}
        
        utilization = {}
        for i in range(self.config.num_experts):
            utilization[i] = self.expert_counts[i].item() / self.total_tokens.item()
        return utilization
    
    def get_layer_stats(self) -> Dict[str, Any]:
        """Get comprehensive layer statistics."""
        return {
            "layer_idx": self.layer_idx,
            "num_experts": self.config.num_experts,
            "local_experts": self._local_expert_ids,
            "num_local": len(self._local_expert_ids),
            "total_tokens": self.total_tokens.item(),
            "missing_expert_calls": self._missing_expert_count,
            "utilization": self.get_expert_utilization(),
        }
    
    def reset_statistics(self):
        """Reset expert utilization statistics."""
        self.expert_counts.zero_()
        self.total_tokens.zero_()
        self._missing_expert_count = 0
    
    def get_router_stats(self) -> Dict[str, Any]:
        """
        Get router statistics for monitoring and debugging.
        
        Router weights are synced via DiLoCo automatically because:
        1. self.router is an nn.Module attribute
        2. DiLoCo uses model.named_parameters() which includes router.gate
        3. Pseudo-gradients include router weight updates
        
        Returns:
            Dict with router weight stats and sync info
        """
        router_weight = self.router.gate.weight
        return {
            "router_weight_norm": router_weight.data.norm().item(),
            "router_weight_mean": router_weight.data.mean().item(),
            "router_weight_std": router_weight.data.std().item(),
            "router_shape": list(router_weight.shape),
            "router_size_bytes": router_weight.numel() * 4,  # float32
            "router_requires_grad": router_weight.requires_grad,
            "num_experts": self.config.num_experts,
            "local_experts": self._local_expert_ids,
        }
    
    def get_router_parameter_name(self) -> str:
        """Get the parameter name for the router weights (for DiLoCo tracking)."""
        return "router.gate.weight"


# =============================================================================
# MOE DECODER LAYER (Attention + MoE FFN)
# =============================================================================

class MoEDecoderLayer(nn.Module):
    """
    Transformer decoder layer with MoE FFN.
    
    Structure:
    - Self Attention (standard)
    - MoE FFN (replaces dense FFN)
    
    This is a drop-in replacement for NeuroDecoderLayer.
    """
    
    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        num_kv_heads: int,
        intermediate_dim: int,
        max_seq_len: int,
        layer_idx: int,
        moe_config: MoEConfig,
        rope_theta: float = 10000.0,
        attention_dropout: float = 0.0,
    ):
        super().__init__()
        self.layer_idx = layer_idx
        
        # Import attention components from llm.py
        from neuroshard.core.model.llm import NeuroAttention, RMSNorm, NeuroLLMConfig
        
        # Create a config for attention
        attn_config = NeuroLLMConfig(
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            num_kv_heads=num_kv_heads,
            intermediate_dim=intermediate_dim,
            max_seq_len=max_seq_len,
            rope_theta=rope_theta,
            attention_dropout=attention_dropout,
        )
        
        # Self attention (standard)
        self.self_attn = NeuroAttention(attn_config, layer_idx)
        
        # MoE FFN (replaces dense FFN)
        self.moe = DistributedMoELayer(moe_config, layer_idx)
        
        # Layer norms
        self.input_layernorm = RMSNorm(hidden_dim)
        self.post_attention_layernorm = RMSNorm(hidden_dim)
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache: bool = False,
        remote_expert_forward: Optional[callable] = None,
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]], Dict]:
        """
        Forward pass through MoE decoder layer.
        
        Returns:
            hidden_states: Updated hidden states
            present_key_value: KV cache (if use_cache)
            moe_aux_data: MoE routing statistics and losses
        """
        # Self attention with residual
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states, present_key_value = self.self_attn(
            hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            use_cache=use_cache,
        )
        hidden_states = residual + hidden_states
        
        # MoE FFN with residual
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states, moe_aux_data = self.moe(
            hidden_states,
            remote_expert_forward=remote_expert_forward
        )
        hidden_states = residual + hidden_states
        
        return hidden_states, present_key_value, moe_aux_data


# =============================================================================
# EXPERT ASSIGNMENT UTILITIES
# =============================================================================

@dataclass
class ExpertAssignment:
    """Tracks which node holds which expert."""
    layer_id: int
    expert_id: int
    node_id: str
    node_url: str
    grpc_addr: str
    capacity: int = 0  # Tokens/second this expert can handle
    utilization: float = 0.0  # Current utilization (0-1)


class ExpertRegistry:
    """
    Registry for expert locations across the network.
    
    Used for:
    1. Tracking which experts exist and where
    2. Load balancing (route to least loaded)
    3. Scarcity rewards (under-replicated experts earn more)
    """
    
    def __init__(self):
        # {layer_id: {expert_id: [ExpertAssignment, ...]}}
        self.assignments: Dict[int, Dict[int, List[ExpertAssignment]]] = {}
        self.target_replicas: int = 2  # Target replicas per expert
    
    def register_expert(self, assignment: ExpertAssignment):
        """Register an expert assignment."""
        if assignment.layer_id not in self.assignments:
            self.assignments[assignment.layer_id] = {}
        if assignment.expert_id not in self.assignments[assignment.layer_id]:
            self.assignments[assignment.layer_id][assignment.expert_id] = []
        
        # Check if this node already has this expert
        existing = [a for a in self.assignments[assignment.layer_id][assignment.expert_id]
                   if a.node_id == assignment.node_id]
        if existing:
            # Update existing
            idx = self.assignments[assignment.layer_id][assignment.expert_id].index(existing[0])
            self.assignments[assignment.layer_id][assignment.expert_id][idx] = assignment
        else:
            self.assignments[assignment.layer_id][assignment.expert_id].append(assignment)
    
    def get_expert_holders(self, layer_id: int, expert_id: int) -> List[ExpertAssignment]:
        """Get all nodes holding a specific expert."""
        return self.assignments.get(layer_id, {}).get(expert_id, [])
    
    def get_least_loaded_holder(self, layer_id: int, expert_id: int) -> Optional[ExpertAssignment]:
        """Get the least loaded holder for an expert."""
        holders = self.get_expert_holders(layer_id, expert_id)
        if not holders:
            return None
        return min(holders, key=lambda x: x.utilization)
    
    def get_scarcity_bonus(self, layer_id: int, expert_id: int) -> float:
        """
        Calculate scarcity bonus for an expert.
        
        Under-replicated experts earn more to incentivize nodes to hold them.
        """
        holders = self.get_expert_holders(layer_id, expert_id)
        num_replicas = len(holders)
        
        if num_replicas >= self.target_replicas:
            return 1.0  # No bonus
        
        # Bonus scales with scarcity
        # 0 replicas: 1.5x bonus (dangerous - need holders!)
        # 1 replica: 1.25x bonus
        scarcity_factor = (self.target_replicas - num_replicas) / self.target_replicas
        return 1.0 + 0.5 * scarcity_factor
    
    def get_layer_experts_status(self, layer_id: int, num_experts: int) -> Dict[int, Dict]:
        """Get status of all experts in a layer."""
        status = {}
        for expert_id in range(num_experts):
            holders = self.get_expert_holders(layer_id, expert_id)
            status[expert_id] = {
                'num_replicas': len(holders),
                'scarcity_bonus': self.get_scarcity_bonus(layer_id, expert_id),
                'holders': [h.node_id[:8] for h in holders],
            }
        return status


# =============================================================================
# MOE GRADIENT HANDLING
# =============================================================================

def compute_moe_loss(
    main_loss: torch.Tensor,
    moe_aux_losses: List[Dict[str, torch.Tensor]]
) -> torch.Tensor:
    """
    Combine main loss with MoE auxiliary losses.
    
    Args:
        main_loss: Main cross-entropy loss
        moe_aux_losses: List of aux losses from each MoE layer
        
    Returns:
        Combined loss
    """
    total_loss = main_loss
    
    for layer_aux in moe_aux_losses:
        total_loss = total_loss + layer_aux.get('load_balancing_loss', 0)
        total_loss = total_loss + layer_aux.get('router_z_loss', 0)
    
    return total_loss


# =============================================================================
# FACTORY FUNCTION
# =============================================================================

def create_moe_config(
    hidden_dim: int,
    intermediate_dim: int,
    num_experts: int = 8,
    top_k: int = 2,
    local_expert_ids: Optional[List[int]] = None,
) -> MoEConfig:
    """
    Create MoE configuration.
    
    Args:
        hidden_dim: Model hidden dimension
        intermediate_dim: FFN intermediate dimension
        num_experts: Total number of experts per layer
        top_k: Number of experts per token
        local_expert_ids: Which experts this node holds (None = all)
        
    Returns:
        MoEConfig instance
    """
    if local_expert_ids is None:
        local_expert_ids = list(range(num_experts))
    
    return MoEConfig(
        hidden_dim=hidden_dim,
        intermediate_dim=intermediate_dim,
        num_experts=num_experts,
        num_experts_per_token=top_k,
        local_experts=local_expert_ids,
    )

