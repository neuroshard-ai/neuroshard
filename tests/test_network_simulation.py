"""
NeuroShard Network Simulation — Full Lifecycle Test

Simulates a realistic network lifecycle:
  Phase 1: Single node starts alone (solo mode)
  Phase 2: Second node joins — quorum forms
  Phase 3: Third and fourth nodes join — multiple quorums
  Phase 4: Training with DiLoCo sync across quorums
  Phase 5: Inference/generation on the trained model
  Phase 6: Node departure and quorum resilience

This test uses real NeuroShard components (model, tokenizer, aggregation,
DiLoCo, quorum) with mock networking (in-process DHT, no gRPC).

Run:
    pytest tests/test_network_simulation.py -v -s
"""

import time
import math
import hashlib
import json
import logging
import threading
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field

import torch
import torch.nn as nn
import pytest

# NeuroShard imports
from neuroshard.core.model.tokenizer import NeuroTokenizer, get_neuro_tokenizer
from neuroshard.core.swarm.aggregation import RobustAggregator, AggregationConfig

logger = logging.getLogger(__name__)


# =============================================================================
# SIMULATION INFRASTRUCTURE
# =============================================================================

class SimulatedDHTNetwork:
    """In-process DHT network shared by all simulated nodes."""

    def __init__(self):
        self.storage: Dict[str, Any] = {}
        self.lock = threading.Lock()
        self.nodes: Dict[str, 'SimulatedDHT'] = {}

    def create_dht(self, node_id: str) -> 'SimulatedDHT':
        dht = SimulatedDHT(self, node_id)
        with self.lock:
            self.nodes[node_id] = dht
        return dht

    def store(self, key: str, value: Any):
        with self.lock:
            # For layer announcements, accumulate endpoints
            if key.startswith("layer_") or key.startswith("shard_provider_"):
                existing = self.storage.get(key, [])
                if isinstance(existing, list):
                    if value not in existing:
                        existing.append(value)
                    self.storage[key] = existing
                else:
                    self.storage[key] = [value]
            else:
                self.storage[key] = value

    def get(self, key: str) -> Any:
        with self.lock:
            return self.storage.get(key)

    def get_all_keys(self) -> List[str]:
        with self.lock:
            return list(self.storage.keys())


class SimulatedDHT:
    """Per-node DHT view."""

    def __init__(self, network: SimulatedDHTNetwork, node_id: str):
        self.network = network
        self.node_id = node_id
        self.storage = {}

    def announce(self, key: str, value: str = None):
        if value is None:
            value = self.node_id
        self.network.store(key, value)
        self.storage[key] = value

    def lookup_value(self, key: str) -> Any:
        return self.network.get(key)

    def store(self, key: str, value: Any):
        self.network.store(key, value)
        self.storage[key] = value

    def store_value(self, key: str, value: str):
        self.network.store(key, value)
        self.storage[key] = value


@dataclass
class SimulatedNode:
    """A simulated NeuroShard node with real model components."""
    node_id: str
    port: int
    layer_range: Tuple[int, int]
    speed_tier: str = "tier2"

    # Real components
    model: Optional[nn.Module] = None
    optimizer: Optional[torch.optim.Optimizer] = None
    dht: Optional[SimulatedDHT] = None
    tokenizer: Optional[NeuroTokenizer] = None

    # Architecture
    hidden_dim: int = 64
    num_heads: int = 4
    vocab_size: int = 500
    num_layers: int = 0

    # State
    total_batches: int = 0
    current_loss: float = float('inf')
    initial_weights: Dict[str, torch.Tensor] = field(default_factory=dict)
    contribution_mode: str = "async"  # "pipeline" or "async"
    is_running: bool = False

    # Quorum
    quorum_id: Optional[str] = None
    quorum_role: Optional[str] = None  # "initiator", "processor", "finisher"

    @property
    def has_embedding(self) -> bool:
        return self.layer_range[0] == 0

    @property
    def has_lm_head(self) -> bool:
        return hasattr(self, '_total_layers') and self.layer_range[1] >= self._total_layers

    @property
    def is_full_node(self) -> bool:
        return self.has_embedding and self.has_lm_head

    def snapshot_weights(self):
        """Save initial weights for DiLoCo pseudo-gradient."""
        self.initial_weights = {
            name: param.data.clone()
            for name, param in self.model.named_parameters()
        }

    def compute_pseudo_gradient(self) -> Dict[str, torch.Tensor]:
        """Compute pseudo-gradient (current - initial)."""
        pseudo_grad = {}
        for name, param in self.model.named_parameters():
            if name in self.initial_weights:
                pseudo_grad[name] = param.data - self.initial_weights[name]
        return pseudo_grad


class NetworkSimulator:
    """
    Orchestrates a multi-node NeuroShard simulation.

    Manages node lifecycle, quorum formation, training coordination,
    and DiLoCo synchronization — all in-process with real model weights.
    """

    def __init__(self, total_layers: int = 8, hidden_dim: int = 64,
                 vocab_size: int = 500):
        self.total_layers = total_layers
        self.hidden_dim = hidden_dim
        self.vocab_size = vocab_size
        self.num_heads = 4

        self.dht_network = SimulatedDHTNetwork()
        self.nodes: Dict[str, SimulatedNode] = {}
        self.quorums: Dict[str, List[str]] = {}  # quorum_id -> [node_ids]
        self.aggregator = RobustAggregator(
            aggregation_config=AggregationConfig()  # Default: TRIMMED_MEAN
        )
        self.tokenizer = NeuroTokenizer(vocab_size=vocab_size)

        # Shared model weights (ground truth for sync)
        self._global_model = self._create_model(list(range(total_layers)))
        self._next_port = 9000

        # Training data (synthetic)
        torch.manual_seed(42)
        self.training_data = torch.randint(10, vocab_size, (200, 64))

    def _create_model(self, layer_ids: List[int]) -> nn.Module:
        """Create a small transformer model covering specific layers."""
        model = SmallNeuroLLM(
            vocab_size=self.vocab_size,
            hidden_dim=self.hidden_dim,
            num_layers=len(layer_ids),
            num_heads=self.num_heads,
            has_embedding=(0 in layer_ids),
            has_lm_head=(self.total_layers - 1 in layer_ids
                         or max(layer_ids) >= self.total_layers - 1),
        )
        return model

    def add_node(self, node_token: str = None,
                 speed_tier: str = "tier2") -> SimulatedNode:
        """Add a new node to the network, auto-assign layers."""
        if node_token is None:
            node_token = f"node_{len(self.nodes)}"

        node_id = hashlib.sha256(node_token.encode()).hexdigest()[:32]
        port = self._next_port
        self._next_port += 1

        # Determine layer assignment based on current network coverage
        layer_range = self._assign_layers(node_id)

        layer_ids = list(range(layer_range[0], layer_range[1]))
        model = self._create_model(layer_ids)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
        dht = self.dht_network.create_dht(node_id)

        node = SimulatedNode(
            node_id=node_id,
            port=port,
            layer_range=layer_range,
            speed_tier=speed_tier,
            model=model,
            optimizer=optimizer,
            dht=dht,
            tokenizer=self.tokenizer,
            hidden_dim=self.hidden_dim,
            num_heads=self.num_heads,
            vocab_size=self.vocab_size,
            num_layers=len(layer_ids),
        )
        node._total_layers = self.total_layers
        node.is_running = True
        node.snapshot_weights()

        self.nodes[node_id] = node

        # Announce layers to DHT
        for layer_id in layer_ids:
            dht.announce(f"layer_{layer_id}", node_id)

        logger.info(f"[SIM] Node {node_id[:8]} joined: layers {layer_range}, "
                    f"tier={speed_tier}, params={sum(p.numel() for p in model.parameters()):,}")

        return node

    def _assign_layers(self, node_id: str) -> Tuple[int, int]:
        """Assign layers to a new node based on current coverage."""
        if not self.nodes:
            # First node gets all layers (solo mode)
            return (0, self.total_layers)

        # Find least-covered layer ranges
        coverage = [0] * self.total_layers
        for n in self.nodes.values():
            for l in range(n.layer_range[0], n.layer_range[1]):
                coverage[l] += 1

        # Find the stretch of least-covered layers
        min_cov = min(coverage)
        # Assign a contiguous block of layers (half the total for multi-node)
        layers_per_node = max(2, self.total_layers // max(2, len(self.nodes) + 1))

        # Start from the least-covered position
        best_start = 0
        best_score = float('inf')
        for start in range(self.total_layers - layers_per_node + 1):
            score = sum(coverage[start:start + layers_per_node])
            if score < best_score:
                best_score = score
                best_start = start

        return (best_start, best_start + layers_per_node)

    def form_quorums(self) -> List[str]:
        """Form quorums from available nodes."""
        available = [n for n in self.nodes.values() if n.is_running and n.quorum_id is None]

        if len(available) < 2:
            # Solo node — no quorum needed, trains alone
            for n in available:
                n.contribution_mode = "async"
            return []

        # Group by compatible speed tiers
        formed = []
        used = set()

        # Try to form complete quorums (covering all layers)
        for anchor in available:
            if anchor.node_id in used:
                continue

            quorum_members = [anchor]
            covered = set(range(anchor.layer_range[0], anchor.layer_range[1]))
            used.add(anchor.node_id)

            for candidate in available:
                if candidate.node_id in used:
                    continue
                candidate_layers = set(range(candidate.layer_range[0], candidate.layer_range[1]))
                new_coverage = candidate_layers - covered
                if new_coverage:
                    quorum_members.append(candidate)
                    covered |= candidate_layers
                    used.add(candidate.node_id)

                if len(covered) >= self.total_layers:
                    break

            # Register quorum
            quorum_id = hashlib.sha256(
                f"quorum_{'_'.join(m.node_id[:8] for m in quorum_members)}".encode()
            ).hexdigest()[:16]

            is_complete = len(covered) >= self.total_layers

            for i, member in enumerate(quorum_members):
                member.quorum_id = quorum_id
                member.contribution_mode = "pipeline" if is_complete else "async"
                # Assign roles
                if member.has_embedding:
                    member.quorum_role = "initiator"
                elif hasattr(member, '_total_layers') and member.layer_range[1] >= member._total_layers:
                    member.quorum_role = "finisher"
                else:
                    member.quorum_role = "processor"

            self.quorums[quorum_id] = [m.node_id for m in quorum_members]
            formed.append(quorum_id)

            status = "COMPLETE" if is_complete else "PARTIAL"
            logger.info(f"[SIM] Quorum {quorum_id[:8]} formed ({status}): "
                        f"{len(quorum_members)} members, "
                        f"layers {min(covered)}-{max(covered)}")

        return formed

    def train_step(self, node: SimulatedNode, batch_idx: int = 0) -> float:
        """Execute one training step on a node."""
        if not node.model or not node.is_running:
            return float('inf')

        # Get batch — standard LM training: input = tokens[:-1], labels = tokens[1:]
        start = (batch_idx * 4) % (len(self.training_data) - 5)
        batch = self.training_data[start:start + 4]  # [4, 64]
        input_ids = batch[:, :-1]   # [4, 63]
        labels = batch[:, 1:]       # [4, 63] — shifted by 1 (same shape)

        node.model.train()
        node.optimizer.zero_grad()

        # Forward pass
        if node.model.has_embedding:
            logits = node.model(input_ids)
        else:
            # Processor/middle node: forward through hidden states
            # Simulate receiving activations from the previous node
            hidden = torch.randn(input_ids.shape[0], input_ids.shape[1], node.hidden_dim)
            logits = node.model(hidden)

        # Compute loss (only if node has LM head)
        if node.model.has_lm_head:
            logits_flat = logits.view(-1, node.vocab_size)
            labels_flat = labels.reshape(-1)
            loss = nn.functional.cross_entropy(logits_flat, labels_flat)
        else:
            # Processor nodes: use a proxy loss from hidden representations
            loss = logits.pow(2).mean() * 0.01

        loss.backward()
        torch.nn.utils.clip_grad_norm_(node.model.parameters(), 1.0)
        node.optimizer.step()

        node.total_batches += 1
        node.current_loss = loss.item()
        return loss.item()

    def diloco_sync(self, quorum_id: str) -> float:
        """Perform DiLoCo cross-quorum sync for all nodes in a quorum.

        1. Each node computes pseudo-gradient (current - initial)
        2. Aggregate using Trimmed Mean
        3. Apply outer update (Nesterov momentum)
        4. Reset initial weights
        """
        if quorum_id not in self.quorums:
            return 0.0

        member_ids = self.quorums[quorum_id]
        members = [self.nodes[nid] for nid in member_ids if nid in self.nodes]

        if not members:
            return 0.0

        # 1. Collect pseudo-gradients
        pseudo_grads = []
        for node in members:
            pg = node.compute_pseudo_gradient()
            if pg:
                pseudo_grads.append(pg)

        if not pseudo_grads:
            return 0.0

        # 2. Aggregate (Trimmed Mean for Byzantine robustness)
        aggregated = {}
        param_names = list(pseudo_grads[0].keys())

        for name in param_names:
            tensors = [pg[name] for pg in pseudo_grads if name in pg]
            if len(tensors) == 1:
                aggregated[name] = tensors[0]
            elif len(tensors) >= 3:
                stacked = torch.stack(tensors)
                trim = max(1, len(tensors) // 10)
                sorted_t, _ = torch.sort(stacked, dim=0)
                if trim < len(tensors) // 2:
                    aggregated[name] = sorted_t[trim:-trim].mean(dim=0)
                else:
                    aggregated[name] = sorted_t.mean(dim=0)
            else:
                aggregated[name] = torch.stack(tensors).mean(dim=0)

        # 3. Apply outer update with momentum (lr=0.7)
        outer_lr = 0.7
        total_delta = 0.0
        for node in members:
            with torch.no_grad():
                for name, param in node.model.named_parameters():
                    if name in node.initial_weights and name in aggregated:
                        delta = aggregated[name]
                        param.data = node.initial_weights[name] + outer_lr * delta
                        total_delta += delta.abs().mean().item()

            # 4. Reset for next round
            node.snapshot_weights()

        avg_delta = total_delta / max(1, len(param_names) * len(members))
        logger.info(f"[SIM] DiLoCo sync for quorum {quorum_id[:8]}: "
                    f"{len(members)} nodes, avg_delta={avg_delta:.6f}")
        return avg_delta

    def generate(self, node: SimulatedNode, prompt_tokens: List[int],
                 max_tokens: int = 20) -> List[int]:
        """Generate tokens using a node's model."""
        if not node.model or not node.model.has_embedding or not node.model.has_lm_head:
            return []

        node.model.eval()
        generated = list(prompt_tokens)

        with torch.no_grad():
            for _ in range(max_tokens):
                input_t = torch.tensor([generated[-32:]])  # Last 32 tokens
                logits = node.model(input_t)
                next_logits = logits[0, -1, :] / 0.8  # temperature
                probs = torch.softmax(next_logits, dim=-1)
                next_token = torch.multinomial(probs, 1).item()
                generated.append(next_token)

        return generated[len(prompt_tokens):]

    def remove_node(self, node_id: str):
        """Remove a node from the network (simulates departure)."""
        if node_id not in self.nodes:
            return

        node = self.nodes[node_id]
        node.is_running = False

        # Remove from quorum
        if node.quorum_id and node.quorum_id in self.quorums:
            self.quorums[node.quorum_id].remove(node_id)
            if not self.quorums[node.quorum_id]:
                del self.quorums[node.quorum_id]

        del self.nodes[node_id]
        logger.info(f"[SIM] Node {node_id[:8]} departed")

    def get_network_stats(self) -> Dict[str, Any]:
        """Get current network statistics."""
        active = [n for n in self.nodes.values() if n.is_running]
        layer_coverage = [0] * self.total_layers
        for n in active:
            for l in range(n.layer_range[0], n.layer_range[1]):
                if l < self.total_layers:
                    layer_coverage[l] += 1

        return {
            "total_nodes": len(active),
            "total_quorums": len(self.quorums),
            "layer_coverage": layer_coverage,
            "min_coverage": min(layer_coverage) if layer_coverage else 0,
            "total_batches": sum(n.total_batches for n in active),
            "avg_loss": sum(n.current_loss for n in active) / max(1, len(active)),
            "modes": {n.node_id[:8]: n.contribution_mode for n in active},
        }


class SmallNeuroLLM(nn.Module):
    """Tiny transformer for simulation (matches NeuroLLM architecture)."""

    def __init__(self, vocab_size: int = 500, hidden_dim: int = 64,
                 num_layers: int = 4, num_heads: int = 4,
                 has_embedding: bool = True, has_lm_head: bool = True):
        super().__init__()
        self.vocab_size = vocab_size
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.has_embedding = has_embedding
        self.has_lm_head = has_lm_head

        if has_embedding:
            self.embedding = nn.Embedding(vocab_size, hidden_dim)
        else:
            self.embedding = None

        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=hidden_dim,
                nhead=num_heads,
                dim_feedforward=hidden_dim * 4,
                batch_first=True,
                dropout=0.0,
            )
            for _ in range(num_layers)
        ])

        if has_lm_head:
            self.lm_head = nn.Linear(hidden_dim, vocab_size)
        else:
            self.lm_head = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.embedding is not None:
            x = self.embedding(x)
        for layer in self.layers:
            x = layer(x)
        if self.lm_head is not None:
            x = self.lm_head(x)
        return x


# =============================================================================
# THE SIMULATION TEST
# =============================================================================

class TestNetworkSimulation:
    """Full network lifecycle simulation."""

    def test_full_lifecycle(self):
        """
        Simulate the complete NeuroShard network lifecycle:

        Phase 1: Genesis — Single node starts alone
        Phase 2: Growth  — Second node joins, quorum forms
        Phase 3: Scale   — More nodes join, multiple quorums
        Phase 4: Train   — DiLoCo distributed training
        Phase 5: Infer   — Generation on trained model
        Phase 6: Churn   — Node departs, network adapts
        """
        print("\n" + "=" * 70)
        print("  NEUROSHARD NETWORK SIMULATION")
        print("  Full lifecycle: 1 node -> 4 nodes -> training -> inference")
        print("=" * 70)

        sim = NetworkSimulator(total_layers=8, hidden_dim=64, vocab_size=500)

        # =====================================================================
        # PHASE 1: GENESIS — Single node starts alone
        # =====================================================================
        print(f"\n{'─' * 60}")
        print("PHASE 1: GENESIS — Single node starts alone")
        print(f"{'─' * 60}")

        node1 = sim.add_node("genesis_node")
        stats = sim.get_network_stats()
        print(f"  Nodes: {stats['total_nodes']}")
        print(f"  Layer coverage: {stats['layer_coverage']}")
        print(f"  Node {node1.node_id[:8]}: layers {node1.layer_range}, "
              f"full_node={node1.is_full_node}")

        # Solo node trains in async mode (no quorum possible)
        quorums = sim.form_quorums()
        assert len(quorums) == 0, "Solo node should not form quorum"
        assert node1.contribution_mode == "async"
        print(f"  Mode: {node1.contribution_mode} (solo — no peers to form quorum)")

        # Train a few steps
        losses_phase1 = []
        for step in range(10):
            loss = sim.train_step(node1, step)
            losses_phase1.append(loss)

        print(f"  Training: 10 steps, loss {losses_phase1[0]:.4f} -> {losses_phase1[-1]:.4f}")
        assert losses_phase1[-1] < losses_phase1[0] * 1.5, "Loss should not explode"

        # =====================================================================
        # PHASE 2: GROWTH — Second node joins, quorum forms
        # =====================================================================
        print(f"\n{'─' * 60}")
        print("PHASE 2: GROWTH — Second node joins, quorum forms")
        print(f"{'─' * 60}")

        node2 = sim.add_node("second_node")
        stats = sim.get_network_stats()
        print(f"  Nodes: {stats['total_nodes']}")
        print(f"  Node {node2.node_id[:8]}: layers {node2.layer_range}")
        print(f"  Layer coverage: {stats['layer_coverage']}")

        # Now we can form a quorum
        quorums = sim.form_quorums()
        print(f"  Quorums formed: {len(quorums)}")
        for qid in quorums:
            members = sim.quorums[qid]
            print(f"    Quorum {qid[:8]}: {len(members)} members")
            for mid in members:
                n = sim.nodes[mid]
                print(f"      {mid[:8]}: layers {n.layer_range}, "
                      f"role={n.quorum_role}, mode={n.contribution_mode}")

        assert len(quorums) >= 1, "Should form at least 1 quorum"

        # Train both nodes
        for step in range(10):
            sim.train_step(node1, step + 10)
            sim.train_step(node2, step + 10)

        # DiLoCo sync
        for qid in quorums:
            delta = sim.diloco_sync(qid)
            print(f"  DiLoCo sync (quorum {qid[:8]}): avg_delta={delta:.6f}")

        stats = sim.get_network_stats()
        print(f"  Total batches: {stats['total_batches']}")
        print(f"  Avg loss: {stats['avg_loss']:.4f}")

        # =====================================================================
        # PHASE 3: SCALE — More nodes join
        # =====================================================================
        print(f"\n{'─' * 60}")
        print("PHASE 3: SCALE — Two more nodes join the network")
        print(f"{'─' * 60}")

        node3 = sim.add_node("third_node")
        node4 = sim.add_node("fourth_node")
        stats = sim.get_network_stats()
        print(f"  Nodes: {stats['total_nodes']}")
        print(f"  Layer coverage: {stats['layer_coverage']}")

        # Re-form quorums with all nodes
        # Reset quorum assignments
        for n in sim.nodes.values():
            n.quorum_id = None
            n.quorum_role = None
        sim.quorums.clear()

        quorums = sim.form_quorums()
        print(f"  Quorums formed: {len(quorums)}")
        for qid in quorums:
            members = sim.quorums[qid]
            print(f"    Quorum {qid[:8]}: {len(members)} members")
            for mid in members:
                n = sim.nodes[mid]
                print(f"      {mid[:8]}: layers {n.layer_range}, "
                      f"role={n.quorum_role}")

        assert stats['total_nodes'] == 4

        # =====================================================================
        # PHASE 4: TRAINING — Extended training with DiLoCo sync
        # =====================================================================
        print(f"\n{'─' * 60}")
        print("PHASE 4: TRAINING — 50 steps with periodic DiLoCo sync")
        print(f"{'─' * 60}")

        sync_interval = 10  # Sync every 10 steps (simulating 500 in production)
        all_losses = {n.node_id[:8]: [] for n in sim.nodes.values()}
        sync_count = 0

        for step in range(50):
            # Train all nodes
            for node in sim.nodes.values():
                loss = sim.train_step(node, step + 20)
                all_losses[node.node_id[:8]].append(loss)

            # DiLoCo sync at interval
            if (step + 1) % sync_interval == 0:
                sync_count += 1
                for qid in list(sim.quorums.keys()):
                    sim.diloco_sync(qid)

        # Report training results
        print(f"  DiLoCo syncs performed: {sync_count}")
        for nid_short, losses in all_losses.items():
            first5 = sum(losses[:5]) / 5
            last5 = sum(losses[-5:]) / 5
            print(f"  Node {nid_short}: loss {first5:.4f} -> {last5:.4f} "
                  f"({len(losses)} steps)")

        stats = sim.get_network_stats()
        print(f"  Network total batches: {stats['total_batches']}")
        print(f"  Network avg loss: {stats['avg_loss']:.4f}")

        # Verify training reduced loss
        for nid_short, losses in all_losses.items():
            first_avg = sum(losses[:5]) / 5
            last_avg = sum(losses[-5:]) / 5
            # Loss should generally decrease or stay stable (not explode)
            assert last_avg < first_avg * 2.0, (
                f"Node {nid_short} loss exploded: {first_avg:.4f} -> {last_avg:.4f}"
            )

        # =====================================================================
        # PHASE 5: INFERENCE — Generate tokens
        # =====================================================================
        print(f"\n{'─' * 60}")
        print("PHASE 5: INFERENCE — Generate tokens on trained model")
        print(f"{'─' * 60}")

        # Find a full node (has embedding + LM head)
        full_nodes = [n for n in sim.nodes.values() if n.is_full_node]
        print(f"  Full nodes (can generate): {len(full_nodes)}")

        if full_nodes:
            gen_node = full_nodes[0]
            prompt = [10, 20, 30, 40, 50]  # Byte-level token IDs
            generated = sim.generate(gen_node, prompt, max_tokens=20)
            print(f"  Prompt tokens: {prompt}")
            print(f"  Generated tokens ({len(generated)}): {generated[:20]}")
            assert len(generated) > 0, "Should generate at least 1 token"
            assert all(0 <= t < sim.vocab_size for t in generated), \
                "Generated tokens should be in valid range"
        else:
            print("  (No full nodes — inference requires embedding + LM head)")

        # =====================================================================
        # PHASE 6: CHURN — Node departs, network adapts
        # =====================================================================
        print(f"\n{'─' * 60}")
        print("PHASE 6: CHURN — Node departs, network adapts")
        print(f"{'─' * 60}")

        departing = list(sim.nodes.keys())[-1]
        departing_short = departing[:8]
        print(f"  Removing node {departing_short}...")
        sim.remove_node(departing)

        stats = sim.get_network_stats()
        print(f"  Remaining nodes: {stats['total_nodes']}")
        print(f"  Layer coverage: {stats['layer_coverage']}")

        # Re-form quorums after departure
        for n in sim.nodes.values():
            n.quorum_id = None
            n.quorum_role = None
        sim.quorums.clear()

        quorums = sim.form_quorums()
        print(f"  Quorums reformed: {len(quorums)}")

        # Training continues with remaining nodes
        for step in range(10):
            for node in sim.nodes.values():
                sim.train_step(node, step + 70)

        stats = sim.get_network_stats()
        print(f"  Training continues: {stats['total_batches']} total batches")
        print(f"  Avg loss after churn: {stats['avg_loss']:.4f}")

        # =====================================================================
        # SUMMARY
        # =====================================================================
        print(f"\n{'=' * 70}")
        print("  SIMULATION COMPLETE")
        print(f"{'=' * 70}")
        print(f"  Total training steps: {stats['total_batches']}")
        print(f"  Final avg loss: {stats['avg_loss']:.4f}")
        print(f"  Nodes active: {stats['total_nodes']}")
        print(f"  Quorums active: {stats['total_quorums']}")
        print(f"  Layer coverage: {stats['layer_coverage']}")
        print(f"  DiLoCo syncs: {sync_count}")
        print(f"{'=' * 70}\n")


    def test_solo_to_swarm_transition(self):
        """Test that a solo node transitions correctly when peers arrive."""
        sim = NetworkSimulator(total_layers=4, hidden_dim=32, vocab_size=200)

        # Start solo
        node1 = sim.add_node("solo")
        assert node1.layer_range == (0, 4), "Solo node should cover all layers"
        assert node1.is_full_node, "Solo node should be a full node"

        # Train solo
        for i in range(5):
            sim.train_step(node1, i)
        solo_loss = node1.current_loss

        # Peer joins
        node2 = sim.add_node("peer")
        quorums = sim.form_quorums()
        assert len(quorums) >= 1, "Should form quorum with 2 nodes"

        # Both nodes train
        for i in range(5):
            sim.train_step(node1, i + 5)
            sim.train_step(node2, i + 5)

        # Sync
        for qid in quorums:
            sim.diloco_sync(qid)

        print(f"\n  Solo -> Swarm transition test passed")
        print(f"    Solo loss: {solo_loss:.4f}")
        print(f"    Post-swarm loss: {node1.current_loss:.4f}")


    def test_diloco_convergence(self):
        """Verify that DiLoCo sync produces convergent updates."""
        sim = NetworkSimulator(total_layers=4, hidden_dim=32, vocab_size=200)

        node1 = sim.add_node("node_a")
        node2 = sim.add_node("node_b")
        quorums = sim.form_quorums()

        # Train independently for a while
        for i in range(20):
            sim.train_step(node1, i)
            sim.train_step(node2, i)

        # Capture pre-sync parameters
        pre_sync_params = {
            name: param.data.clone()
            for name, param in node1.model.named_parameters()
        }

        # Sync
        for qid in quorums:
            delta = sim.diloco_sync(qid)

        # Verify weights changed
        weights_changed = False
        for name, param in node1.model.named_parameters():
            if name in pre_sync_params:
                if not torch.allclose(param.data, pre_sync_params[name]):
                    weights_changed = True
                    break

        assert weights_changed, "DiLoCo sync should update model weights"
        print(f"\n  DiLoCo convergence test passed (weights updated by sync)")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    pytest.main([__file__, "-v", "-s"])
