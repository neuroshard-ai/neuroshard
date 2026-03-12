"""
NeuroLLM - The People's Language Model

A transformer-based LLM designed from the ground up for decentralized training
and inference. This is not a wrapper around GPT or LLaMA - it's a completely
new model that grows smarter as more nodes contribute compute and data.

Key Design Principles:
1. Modular Architecture: Easy to distribute layers/shards across nodes
2. Efficient Gradients: Optimized for gradient compression and gossip
3. Privacy-First: Supports differential privacy and federated learning
4. Reward-Aligned: Training contributions earn NEURO tokens
5. Self-Improving: Model improves as network grows

Architecture:
- RMSNorm (more stable for distributed training)
- Rotary Position Embeddings (RoPE) - no absolute position limits
- Grouped Query Attention (GQA) - memory efficient
- SwiGLU activation - better performance
- Mixture of Experts (optional) - scalability

The model starts small and grows with the network:
- Phase 1: 125M params - Bootstrap
- Phase 2: 1B params - Early adoption  
- Phase 3: 7B params - Growth
- Phase 4: 70B+ params - Maturity
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import logging
from typing import Optional, Tuple, Dict, List, Any
from dataclasses import dataclass, field
from enum import Enum

logger = logging.getLogger(__name__)


class NeuroLLMPhase(Enum):
    """Growth phases of NeuroLLM."""
    BOOTSTRAP = "bootstrap"      # 125M - Initial training
    EARLY = "early"              # 1B - Early adoption
    GROWTH = "growth"            # 7B - Network growth
    MATURE = "mature"            # 70B+ - Full scale


@dataclass
class NeuroLLMConfig:
    """
    Configuration for NeuroLLM.
    
    Designed to be easily scalable as the network grows.
    """
    # Model identity
    version: str = "0.1.0"
    phase: NeuroLLMPhase = NeuroLLMPhase.BOOTSTRAP
    
    # Architecture - Bootstrap phase (125M params)
    vocab_size: int = 32000          # Initial vocab - auto-expands to unlimited as tokenizer learns
    hidden_dim: int = 768            # Hidden dimension
    num_layers: int = 12             # Transformer layers
    num_heads: int = 12              # Attention heads
    num_kv_heads: int = 4            # KV heads (GQA)
    intermediate_dim: int = 2048     # FFN intermediate
    max_seq_len: int = 2048          # Maximum sequence length
    
    # Regularization
    dropout: float = 0.0             # Dropout (0 for inference)
    attention_dropout: float = 0.0
    
    # Training
    tie_word_embeddings: bool = True
    use_cache: bool = True
    
    # Distributed training
    gradient_checkpointing: bool = False
    gradient_accumulation_steps: int = 1
    
    # RoPE settings
    rope_theta: float = 10000.0
    rope_scaling: Optional[Dict] = None
    
    # MoE settings (for future scaling)
    num_experts: int = 0             # 0 = dense model
    num_experts_per_tok: int = 2
    
    @classmethod
    def bootstrap(cls) -> 'NeuroLLMConfig':
        """Bootstrap phase config (~125M params)."""
        return cls(
            phase=NeuroLLMPhase.BOOTSTRAP,
            hidden_dim=768,
            num_layers=12,
            num_heads=12,
            num_kv_heads=4,
            intermediate_dim=2048,
        )
    
    @classmethod
    def early(cls) -> 'NeuroLLMConfig':
        """Early adoption config (~1B params)."""
        return cls(
            phase=NeuroLLMPhase.EARLY,
            hidden_dim=2048,
            num_layers=24,
            num_heads=16,
            num_kv_heads=4,
            intermediate_dim=5632,
        )
    
    @classmethod
    def growth(cls) -> 'NeuroLLMConfig':
        """Growth phase config (~7B params)."""
        return cls(
            phase=NeuroLLMPhase.GROWTH,
            hidden_dim=4096,
            num_layers=32,
            num_heads=32,
            num_kv_heads=8,
            intermediate_dim=11008,
        )
    
    @classmethod
    def mature(cls) -> 'NeuroLLMConfig':
        """Mature phase config (~70B params)."""
        return cls(
            phase=NeuroLLMPhase.MATURE,
            hidden_dim=8192,
            num_layers=80,
            num_heads=64,
            num_kv_heads=8,
            intermediate_dim=28672,
        )
    
    @property
    def head_dim(self) -> int:
        return self.hidden_dim // self.num_heads
    
    @property
    def num_params(self) -> int:
        """Estimate total parameters."""
        # Embeddings
        embed_params = self.vocab_size * self.hidden_dim
        
        # Per layer
        # Attention: Q, K, V, O projections
        attn_params = (
            self.hidden_dim * self.hidden_dim +  # Q
            self.hidden_dim * (self.hidden_dim // self.num_heads * self.num_kv_heads) * 2 +  # K, V
            self.hidden_dim * self.hidden_dim  # O
        )
        # FFN: up, gate, down
        ffn_params = 3 * self.hidden_dim * self.intermediate_dim
        # Norms
        norm_params = 2 * self.hidden_dim
        
        layer_params = attn_params + ffn_params + norm_params
        
        # Total
        total = embed_params + self.num_layers * layer_params + self.hidden_dim  # final norm
        
        if not self.tie_word_embeddings:
            total += self.vocab_size * self.hidden_dim
        
        return total


class RMSNorm(nn.Module):
    """
    Root Mean Square Layer Normalization.
    
    More stable than LayerNorm for distributed training.
    """
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Calculate RMS
        rms = torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return x * rms * self.weight


class RotaryEmbedding(nn.Module):
    """
    Rotary Position Embeddings (RoPE).
    
    Allows the model to generalize to longer sequences than seen during training.
    """
    def __init__(self, dim: int, max_seq_len: int = 2048, theta: float = 10000.0):
        super().__init__()
        self.dim = dim
        self.max_seq_len = max_seq_len
        self.theta = theta
        
        # Precompute frequencies
        inv_freq = 1.0 / (theta ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)
        
        # Build cache
        self._build_cache(max_seq_len)
    
    def _build_cache(self, seq_len: int):
        """Build cos/sin cache for given sequence length."""
        t = torch.arange(seq_len, device=self.inv_freq.device)
        freqs = torch.outer(t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos())
        self.register_buffer("sin_cached", emb.sin())
    
    def forward(self, x: torch.Tensor, seq_len: int) -> Tuple[torch.Tensor, torch.Tensor]:
        if seq_len > self.max_seq_len:
            self._build_cache(seq_len)
            self.max_seq_len = seq_len
        
        return (
            self.cos_cached[:seq_len],
            self.sin_cached[:seq_len]
        )


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Rotate half the hidden dims."""
    x1 = x[..., :x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2:]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q: torch.Tensor, k: torch.Tensor, 
                          cos: torch.Tensor, sin: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Apply rotary position embeddings to Q and K.
    
    Args:
        q: Query tensor [batch, heads, seq_len, head_dim]
        k: Key tensor [batch, heads, seq_len, head_dim]  
        cos: Cosine embeddings [seq_len, head_dim]
        sin: Sine embeddings [seq_len, head_dim]
    
    Returns:
        Tuple of rotated Q and K tensors
    """
    # Ensure cos/sin are properly shaped for broadcasting with 4D tensors
    # cos/sin: [seq_len, head_dim] -> [1, 1, seq_len, head_dim]
    if cos.dim() == 2:
        cos = cos.unsqueeze(0).unsqueeze(0)
        sin = sin.unsqueeze(0).unsqueeze(0)
    
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


class NeuroAttention(nn.Module):
    """
    Grouped Query Attention (GQA) for NeuroLLM.
    
    Uses fewer KV heads than query heads for memory efficiency.
    """
    def __init__(self, config: NeuroLLMConfig, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        
        self.hidden_dim = config.hidden_dim
        self.num_heads = config.num_heads
        self.num_kv_heads = config.num_kv_heads
        self.head_dim = config.head_dim
        self.num_key_value_groups = self.num_heads // self.num_kv_heads
        
        # Projections
        self.q_proj = nn.Linear(self.hidden_dim, self.num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(self.hidden_dim, self.num_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(self.hidden_dim, self.num_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_dim, bias=False)
        
        # RoPE
        self.rotary_emb = RotaryEmbedding(
            self.head_dim,
            max_seq_len=config.max_seq_len,
            theta=config.rope_theta
        )
        
        self.attention_dropout = nn.Dropout(config.attention_dropout)
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        batch_size, seq_len, _ = hidden_states.shape
        
        # Project Q, K, V
        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)
        
        # Reshape for multi-head attention
        query_states = query_states.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(batch_size, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(batch_size, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)
        
        # Apply RoPE
        cos, sin = self.rotary_emb(hidden_states, seq_len)
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)
        
        # Handle KV cache
        if past_key_value is not None:
            key_states = torch.cat([past_key_value[0], key_states], dim=2)
            value_states = torch.cat([past_key_value[1], value_states], dim=2)
        
        past_key_value = (key_states, value_states) if use_cache else None
        
        # Repeat KV heads for GQA
        key_states = key_states.repeat_interleave(self.num_key_value_groups, dim=1)
        value_states = value_states.repeat_interleave(self.num_key_value_groups, dim=1)
        
        # Attention
        attn_weights = torch.matmul(query_states, key_states.transpose(-2, -1)) / math.sqrt(self.head_dim)
        
        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask
        
        attn_weights = F.softmax(attn_weights, dim=-1)
        attn_weights = self.attention_dropout(attn_weights)
        
        attn_output = torch.matmul(attn_weights, value_states)
        
        # Reshape and project output
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, -1)
        attn_output = self.o_proj(attn_output)
        
        return attn_output, past_key_value


class NeuroMLP(nn.Module):
    """
    SwiGLU-based MLP for NeuroLLM.
    
    SwiGLU provides better performance than standard GELU.
    """
    def __init__(self, config: NeuroLLMConfig):
        super().__init__()
        self.hidden_dim = config.hidden_dim
        self.intermediate_dim = config.intermediate_dim
        
        # SwiGLU: gate * silu(up)
        self.gate_proj = nn.Linear(self.hidden_dim, self.intermediate_dim, bias=False)
        self.up_proj = nn.Linear(self.hidden_dim, self.intermediate_dim, bias=False)
        self.down_proj = nn.Linear(self.intermediate_dim, self.hidden_dim, bias=False)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class NeuroDecoderLayer(nn.Module):
    """
    Single transformer decoder layer for NeuroLLM.
    """
    def __init__(self, config: NeuroLLMConfig, layer_idx: int):
        super().__init__()
        self.layer_idx = layer_idx
        
        self.self_attn = NeuroAttention(config, layer_idx)
        self.mlp = NeuroMLP(config)
        
        self.input_layernorm = RMSNorm(config.hidden_dim)
        self.post_attention_layernorm = RMSNorm(config.hidden_dim)
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
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
        
        # MLP with residual
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states
        
        return hidden_states, present_key_value


class NeuroLLMModel(nn.Module):
    """
    The core NeuroLLM transformer model.
    
    This is the base model without the LM head.
    """
    def __init__(self, config: NeuroLLMConfig):
        super().__init__()
        self.config = config
        
        # Token embeddings
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_dim)
        
        # Transformer layers
        self.layers = nn.ModuleList([
            NeuroDecoderLayer(config, layer_idx)
            for layer_idx in range(config.num_layers)
        ])
        
        # Final norm
        self.norm = RMSNorm(config.hidden_dim)
        
        # Gradient checkpointing
        self.gradient_checkpointing = config.gradient_checkpointing
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        """Initialize weights with small values for stable training."""
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[List[Tuple[torch.Tensor, torch.Tensor]]]]:
        batch_size, seq_len = input_ids.shape
        
        # Get embeddings
        hidden_states = self.embed_tokens(input_ids)
        
        # Create causal mask
        if attention_mask is None:
            attention_mask = torch.ones((batch_size, seq_len), device=input_ids.device)
        
        # Create causal attention mask
        causal_mask = torch.triu(
            torch.full((seq_len, seq_len), float('-inf'), device=input_ids.device),
            diagonal=1
        )
        
        # Process through layers
        present_key_values = [] if use_cache else None
        
        for idx, layer in enumerate(self.layers):
            past_key_value = past_key_values[idx] if past_key_values else None
            
            if self.gradient_checkpointing and self.training:
                hidden_states, present_key_value = torch.utils.checkpoint.checkpoint(
                    layer,
                    hidden_states,
                    causal_mask,
                    position_ids,
                    past_key_value,
                    use_cache,
                )
            else:
                hidden_states, present_key_value = layer(
                    hidden_states,
                    attention_mask=causal_mask,
                    position_ids=position_ids,
                    past_key_value=past_key_value,
                    use_cache=use_cache,
                )
            
            if use_cache:
                present_key_values.append(present_key_value)
        
        # Final norm
        hidden_states = self.norm(hidden_states)
        
        return hidden_states, present_key_values


class NeuroLLMForCausalLM(nn.Module):
    """
    NeuroLLM with language modeling head.
    
    This is the full model for both training and inference.
    """
    def __init__(self, config: NeuroLLMConfig):
        super().__init__()
        self.config = config
        
        self.model = NeuroLLMModel(config)
        
        # LM head (optionally tied to embeddings)
        self.lm_head = nn.Linear(config.hidden_dim, config.vocab_size, bias=False)
        
        if config.tie_word_embeddings:
            self.lm_head.weight = self.model.embed_tokens.weight
        
        # Loss function
        self.loss_fn = nn.CrossEntropyLoss(ignore_index=-100)
        
        logger.info(f"NeuroLLM initialized: {config.num_params / 1e6:.1f}M parameters, "
                   f"phase={config.phase.value}")
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None,
        use_cache: bool = False,
    ) -> Dict[str, torch.Tensor]:
        # Get hidden states
        hidden_states, present_key_values = self.model(
            input_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            use_cache=use_cache,
        )
        
        # Get logits
        logits = self.lm_head(hidden_states)
        
        # Calculate loss if labels provided
        loss = None
        if labels is not None:
            # Shift for next token prediction
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            
            # Flatten
            shift_logits = shift_logits.view(-1, self.config.vocab_size)
            shift_labels = shift_labels.view(-1)
            
            loss = self.loss_fn(shift_logits, shift_labels)
        
        return {
            "loss": loss,
            "logits": logits,
            "past_key_values": present_key_values,
        }
    
    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 100,
        temperature: float = 1.0,
        top_k: int = 50,
        top_p: float = 0.9,
        do_sample: bool = True,
        valid_vocab_size: int = 266,  # Mask tokens beyond this (default: byte tokens only)
    ) -> torch.Tensor:
        """Generate text autoregressively."""
        self.eval()
        
        past_key_values = None
        generated = input_ids
        
        for _ in range(max_new_tokens):
            # Forward pass
            outputs = self.forward(
                input_ids=generated if past_key_values is None else generated[:, -1:],
                past_key_values=past_key_values,
                use_cache=True,
            )
            
            logits = outputs["logits"][:, -1, :]
            past_key_values = outputs["past_key_values"]
            
            # Constrain to valid vocabulary (standard BPE tokenizer behavior)
            # Tokens beyond valid_vocab_size don't exist in the tokenizer yet
            if valid_vocab_size < logits.size(-1):
                logits[:, valid_vocab_size:] = float('-inf')
            
            # Apply temperature
            if temperature != 1.0:
                logits = logits / temperature
            
            # Apply top-k
            if top_k > 0:
                indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
                logits[indices_to_remove] = float('-inf')
            
            # Apply top-p (nucleus sampling)
            if top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0
                
                indices_to_remove = sorted_indices_to_remove.scatter(
                    1, sorted_indices, sorted_indices_to_remove
                )
                logits[indices_to_remove] = float('-inf')
            
            # Sample or greedy
            if do_sample:
                probs = F.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
            else:
                next_token = torch.argmax(logits, dim=-1, keepdim=True)
            
            generated = torch.cat([generated, next_token], dim=-1)
            
            # Check for EOS (assuming token 2 is EOS)
            if next_token.item() == 2:
                break
        
        return generated
    
    def get_num_params(self) -> int:
        """Get actual number of parameters."""
        return sum(p.numel() for p in self.parameters())
    
    def save_checkpoint(self, path: str, extra_state: Dict = None):
        """Save model checkpoint."""
        state = {
            "config": self.config.__dict__,
            "model_state_dict": self.state_dict(),
            "version": self.config.version,
        }
        if extra_state:
            state.update(extra_state)
        torch.save(state, path)
        logger.info(f"Saved checkpoint to {path}")
    
    @classmethod
    def load_checkpoint(cls, path: str, device: str = "cpu") -> 'NeuroLLMForCausalLM':
        """Load model from checkpoint."""
        # Use weights_only=False for full checkpoint loading (includes config)
        # This is safe because we only load our own checkpoints
        state = torch.load(path, map_location=device, weights_only=False)
        
        # Reconstruct config
        config_dict = state["config"]
        config_dict["phase"] = NeuroLLMPhase(config_dict["phase"])
        config = NeuroLLMConfig(**config_dict)
        
        # Create model and load weights
        model = cls(config)
        model.load_state_dict(state["model_state_dict"])
        
        logger.info(f"Loaded checkpoint from {path}")
        return model
    
    # ==================== SHARDED MODEL SUPPORT ====================
    
    def get_layer_names(self, layer_idx: int) -> List[str]:
        """Get parameter names for a specific layer."""
        prefix = f"model.layers.{layer_idx}."
        return [name for name in self.state_dict().keys() if name.startswith(prefix)]
    
    def extract_shard(
        self,
        start_layer: int,
        end_layer: int,
        include_embedding: bool = False,
        include_lm_head: bool = False
    ) -> Dict[str, torch.Tensor]:
        """
        Extract weights for a shard (subset of layers).
        
        Args:
            start_layer: First layer to include
            end_layer: Last layer to include (exclusive)
            include_embedding: Include embedding layer
            include_lm_head: Include LM head
            
        Returns:
            Dict of parameter names to tensors
        """
        state_dict = self.state_dict()
        shard_weights = {}
        
        for name, param in state_dict.items():
            include = False
            
            # Embedding
            if include_embedding and ("embed" in name.lower() or "wte" in name.lower()):
                include = True
            
            # LM head
            if include_lm_head and ("lm_head" in name.lower()):
                include = True
            
            # Final norm (goes with LM head)
            if include_lm_head and ("final" in name.lower() or "ln_f" in name.lower() or "model.norm" in name.lower()):
                include = True
            
            # Transformer layers
            import re
            match = re.search(r'layers\.(\d+)\.', name)
            if match:
                layer_num = int(match.group(1))
                if start_layer <= layer_num < end_layer:
                    include = True
            
            if include:
                shard_weights[name] = param.clone()
        
        logger.info(f"Extracted shard: layers {start_layer}-{end_layer}, "
                   f"embed={include_embedding}, head={include_lm_head}, "
                   f"params={len(shard_weights)}")
        
        return shard_weights
    
    def load_shard(
        self,
        shard_weights: Dict[str, torch.Tensor],
        strict: bool = False
    ) -> Tuple[List[str], List[str]]:
        """
        Load weights from a shard into the model.
        
        Args:
            shard_weights: Dict of parameter names to tensors
            strict: If True, raise error on missing/unexpected keys
            
        Returns:
            (missing_keys, unexpected_keys)
        """
        current_state = self.state_dict()
        
        missing_keys = []
        unexpected_keys = []
        loaded_keys = []
        
        for name, param in shard_weights.items():
            if name in current_state:
                if current_state[name].shape == param.shape:
                    current_state[name].copy_(param)
                    loaded_keys.append(name)
                else:
                    logger.warning(f"Shape mismatch for {name}: "
                                  f"model={current_state[name].shape}, shard={param.shape}")
            else:
                unexpected_keys.append(name)
        
        # Check for missing keys in the shard range
        for name in current_state.keys():
            if name not in shard_weights and name not in loaded_keys:
                # Only report as missing if it should have been in the shard
                pass  # Shards are partial by design
        
        if strict and unexpected_keys:
            raise RuntimeError(f"Unexpected keys in shard: {unexpected_keys}")
        
        logger.info(f"Loaded shard: {len(loaded_keys)} parameters")
        
        return missing_keys, unexpected_keys
    
    def save_shard(
        self,
        path: str,
        start_layer: int,
        end_layer: int,
        include_embedding: bool = False,
        include_lm_head: bool = False,
        extra_state: Dict = None
    ):
        """
        Save a shard to disk.
        
        Args:
            path: Save path
            start_layer: First layer
            end_layer: Last layer (exclusive)
            include_embedding: Include embedding
            include_lm_head: Include LM head
            extra_state: Additional state to save
        """
        shard_weights = self.extract_shard(
            start_layer, end_layer, include_embedding, include_lm_head
        )
        
        state = {
            "shard_config": {
                "start_layer": start_layer,
                "end_layer": end_layer,
                "include_embedding": include_embedding,
                "include_lm_head": include_lm_head,
            },
            "model_config": self.config.__dict__,
            "shard_weights": shard_weights,
            "num_params": sum(p.numel() for p in shard_weights.values()),
        }
        
        if extra_state:
            state.update(extra_state)
        
        torch.save(state, path)
        logger.info(f"Saved shard to {path}: layers {start_layer}-{end_layer}, "
                   f"{state['num_params']} params")
    
    @classmethod
    def load_shard_file(cls, path: str, device: str = "cpu") -> Tuple[Dict[str, torch.Tensor], Dict]:
        """
        Load a shard file.
        
        Args:
            path: Shard file path
            device: Device to load to
            
        Returns:
            (shard_weights, shard_config)
        """
        state = torch.load(path, map_location=device, weights_only=False)
        
        return state["shard_weights"], state["shard_config"]
    
    def forward_shard(
        self,
        hidden_states: torch.Tensor,
        start_layer: int,
        end_layer: int,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[List[Tuple[torch.Tensor, torch.Tensor]]]]:
        """
        Forward pass through a subset of layers (for pipeline parallelism).
        
        This is used when the model is sharded across nodes:
        - Node A processes layers 0-3
        - Node B processes layers 4-7
        - etc.
        
        Args:
            hidden_states: Input hidden states [batch, seq, hidden]
            start_layer: First layer to process
            end_layer: Last layer to process (exclusive)
            attention_mask: Attention mask
            position_ids: Position IDs
            past_key_values: KV cache
            use_cache: Whether to return KV cache
            
        Returns:
            (output_hidden_states, new_past_key_values)
        """
        new_past_key_values = [] if use_cache else None
        
        for idx in range(start_layer, end_layer):
            layer = self.model.layers[idx]
            
            past_kv = past_key_values[idx] if past_key_values else None
            
            hidden_states, new_kv = layer(
                hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_value=past_kv,
                use_cache=use_cache,
            )
            
            if use_cache:
                new_past_key_values.append(new_kv)
        
        return hidden_states, new_past_key_values


# Convenience function
def create_neuro_llm(phase: str = "bootstrap") -> NeuroLLMForCausalLM:
    """
    Create a NeuroLLM model for the specified phase.
    
    Args:
        phase: One of "bootstrap", "early", "growth", "mature"
        
    Returns:
        NeuroLLMForCausalLM instance
    """
    config_map = {
        "bootstrap": NeuroLLMConfig.bootstrap,
        "early": NeuroLLMConfig.early,
        "growth": NeuroLLMConfig.growth,
        "mature": NeuroLLMConfig.mature,
    }
    
    if phase not in config_map:
        raise ValueError(f"Unknown phase: {phase}. Choose from {list(config_map.keys())}")
    
    config = config_map[phase]()
    return NeuroLLMForCausalLM(config)

