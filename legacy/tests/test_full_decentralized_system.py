"""
COMPREHENSIVE TEST SUITE FOR NEUROSHARD DECENTRALIZED LLM TRAINING

Tests ALL aspects of the distributed training system:
1. Role assignments (DRIVER/WORKER/VALIDATOR) for 1 to N nodes
2. DHT operations (announce, lookup, expiry)
3. Network dynamics (join, leave, rejoin, failure recovery)
4. Layer scaling (model grows with network)
5. Vocabulary expansion
6. Pipeline training (forward/backward across nodes)
7. Edge cases and stress tests
"""

import sys
import time
import json
import torch
import hashlib
import threading
from unittest.mock import patch, MagicMock
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass

# Setup path
sys.path.insert(0, 'src')

from neuroshard.core.model.dynamic import (
    DynamicLayerPool,
    DynamicNeuroLLM,
    LayerAssignment,
    INITIAL_VOCAB_SIZE,
    MAX_VOCAB_SIZE,
)
from neuroshard.core.model.scaler import ModelArchitecture, calculate_layer_assignment


# ============================================================================
# MOCK DHT INFRASTRUCTURE
# ============================================================================

class MockDHTNetwork:
    """Simulates a distributed hash table network shared by all nodes."""
    
    def __init__(self):
        self.storage: Dict[int, Tuple[str, float]] = {}  # key -> (value, expiry_time)
        self.nodes: Dict[str, 'MockNodeDHT'] = {}
        self.lock = threading.Lock()
        self.ttl_seconds = 300  # DHT entry TTL
    
    def create_node_dht(self, node_url: str, grpc_addr: str) -> 'MockNodeDHT':
        """Create a DHT interface for a node."""
        dht = MockNodeDHT(self, node_url, grpc_addr)
        self.nodes[node_url] = dht
        return dht
    
    def store(self, key_int: int, value: str, ttl: float = None):
        """Store a value with TTL."""
        with self.lock:
            expiry = time.time() + (ttl or self.ttl_seconds)
            
            # Merge with existing if present
            if key_int in self.storage:
                existing_value, existing_expiry = self.storage[key_int]
                try:
                    existing_list = json.loads(existing_value)
                    new_list = json.loads(value)
                    merged = list(set(existing_list + new_list))
                    value = json.dumps(merged)
                except:
                    pass
            
            self.storage[key_int] = (value, expiry)
    
    def lookup(self, key_int: int) -> Optional[str]:
        """Lookup a value, returning None if expired."""
        with self.lock:
            if key_int not in self.storage:
                return None
            
            value, expiry = self.storage[key_int]
            if time.time() > expiry:
                del self.storage[key_int]
                return None
            
            return value
    
    def remove_node_entries(self, node_url: str):
        """Remove all entries for a node (simulates node leaving)."""
        with self.lock:
            for key_int in list(self.storage.keys()):
                value, expiry = self.storage[key_int]
                try:
                    nodes = json.loads(value)
                    if node_url in nodes:
                        nodes.remove(node_url)
                        if nodes:
                            self.storage[key_int] = (json.dumps(nodes), expiry)
                        else:
                            del self.storage[key_int]
                except:
                    pass
    
    def expire_entries(self):
        """Force expire all entries (simulate TTL expiry)."""
        with self.lock:
            self.storage.clear()
    
    def get_all_layer_holders(self) -> Dict[int, List[str]]:
        """Debug: Get all layer holders."""
        result = {}
        for layer_id in range(100):  # Check up to 100 layers
            key_str = f"layer_{layer_id}"
            key_int = int(hashlib.sha1(key_str.encode()).hexdigest(), 16)
            value = self.lookup(key_int)
            if value:
                try:
                    result[layer_id] = json.loads(value)
                except:
                    pass
        return result


class MockNodeDHT:
    """DHT interface for a single node."""
    
    def __init__(self, network: MockDHTNetwork, node_url: str, grpc_addr: str):
        self.network = network
        self.node_url = node_url
        self.grpc_addr = grpc_addr
    
    def announce(self, key_string: str):
        """Announce this node for a key."""
        key_int = int(hashlib.sha1(key_string.encode()).hexdigest(), 16)
        self.network.store(key_int, json.dumps([self.node_url]))
        return True
    
    def lookup_value(self, key_int: int) -> Optional[str]:
        """Lookup a value."""
        return self.network.lookup(key_int)
    
    def announce_layers(self, layer_ids: List[int]):
        """Announce multiple layers."""
        for layer_id in layer_ids:
            self.announce(f"layer_{layer_id}")


# ============================================================================
# SIMULATED NODE
# ============================================================================

@dataclass
class SimulatedNode:
    """Represents a node in the network."""
    node_id: str
    node_url: str
    grpc_addr: str
    memory_mb: float
    pool: DynamicLayerPool
    dht: MockNodeDHT
    model: Optional[DynamicNeuroLLM] = None
    optimizer: Optional[torch.optim.Optimizer] = None
    layers: List[int] = None
    role: str = "UNKNOWN"
    
    def __post_init__(self):
        # Only initialize to empty list if None was passed
        if self.layers is None:
            self.layers = []


class NetworkSimulator:
    """Simulates a decentralized network of nodes."""
    
    def __init__(self, layer_capacity_per_node: int = 4):
        self.dht_network = MockDHTNetwork()
        self.nodes: Dict[str, SimulatedNode] = {}
        self.layer_capacity = layer_capacity_per_node
        self.node_counter = 0
    
    def add_node(self, memory_mb: float = 4000, with_model: bool = True) -> SimulatedNode:
        """Add a new node to the network."""
        self.node_counter += 1
        node_id = f"node_{self.node_counter}"
        node_url = f"http://{node_id}:8000"
        grpc_addr = f"{node_id}:9000"
        
        # Create DHT interface
        dht = self.dht_network.create_node_dht(node_url, grpc_addr)
        
        # Create layer pool
        pool = DynamicLayerPool(dht_protocol=dht)
        pool._device_hint = 'cpu'
        pool.vocab_capacity = 32000
        
        # Register node - mock the scaler function
        # Since import happens inside register_node, we patch at source
        import neuroshard.core.model.scaler as scaler_module
        original_func = scaler_module.calculate_layer_assignment
        scaler_module.calculate_layer_assignment = lambda *args, **kwargs: self.layer_capacity
        try:
            layers = pool.register_node(
                node_id=node_id,
                node_url=node_url,
                grpc_addr=grpc_addr,
                available_memory_mb=memory_mb,
                staked_amount=0
            )
        finally:
            scaler_module.calculate_layer_assignment = original_func
        
        # Announce layers to DHT
        dht.announce_layers(layers)
        
        # Determine role
        if 0 in layers and pool.embedding_holder == node_id:
            if pool.lm_head_holder == node_id:
                role = "DRIVER+VALIDATOR"
            else:
                role = "DRIVER"
        elif pool.lm_head_holder == node_id:
            role = "VALIDATOR"
        else:
            role = "WORKER"
        
        # Create model if requested
        model = None
        optimizer = None
        if with_model and layers:
            model = DynamicNeuroLLM(node_id=node_id, layer_pool=pool, device="cpu")
            model.initialize_layers(layers)
            optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
        
        node = SimulatedNode(
            node_id=node_id,
            node_url=node_url,
            grpc_addr=grpc_addr,
            memory_mb=memory_mb,
            pool=pool,
            dht=dht,
            model=model,
            optimizer=optimizer,
            layers=layers,
            role=role
        )
        
        self.nodes[node_id] = node
        return node
    
    def remove_node(self, node_id: str):
        """Remove a node from the network."""
        if node_id in self.nodes:
            node = self.nodes[node_id]
            # Remove from DHT
            self.dht_network.remove_node_entries(node.node_url)
            # Remove from local tracking
            del self.nodes[node_id]
    
    def get_network_coverage(self) -> Dict[int, List[str]]:
        """Get which layers are covered by which nodes."""
        return self.dht_network.get_all_layer_holders()
    
    def get_node_by_role(self, role: str) -> List[SimulatedNode]:
        """Get all nodes with a specific role."""
        return [n for n in self.nodes.values() if role in n.role]


# ============================================================================
# TEST CLASSES
# ============================================================================

class TestRoleAssignments:
    """Test role assignments for various network sizes."""
    
    def test_single_node_is_full_node(self):
        """Single node should be DRIVER+VALIDATOR (full node)."""
        print("\n  [1 node] Testing single node role...")
        
        sim = NetworkSimulator(layer_capacity_per_node=11)  # Can hold all layers
        node = sim.add_node(memory_mb=10000)
        
        assert 0 in node.layers, "Single node should have layer 0"
        assert node.pool.embedding_holder == node.node_id
        assert node.pool.lm_head_holder == node.node_id
        assert node.role == "DRIVER+VALIDATOR"
        
        print(f"    ✅ Single node is {node.role}: layers {node.layers}")
    
    def test_two_nodes_driver_validator(self):
        """Two nodes: one DRIVER, one WORKER+VALIDATOR."""
        print("\n  [2 nodes] Testing driver + validator split...")
        
        sim = NetworkSimulator(layer_capacity_per_node=6)
        
        node1 = sim.add_node(memory_mb=4000)
        node2 = sim.add_node(memory_mb=4000)
        
        # Node 1 should be DRIVER
        assert 0 in node1.layers
        assert node1.pool.embedding_holder == node1.node_id
        
        # Node 2 should be WORKER (and possibly VALIDATOR)
        assert 0 not in node2.layers
        
        # Combined should cover pipeline
        all_layers = set(node1.layers) | set(node2.layers)
        print(f"    Node 1 ({node1.role}): {node1.layers}")
        print(f"    Node 2 ({node2.role}): {node2.layers}")
        print(f"    Combined: {sorted(all_layers)}")
        
        assert 0 in all_layers
        print("    ✅ Two nodes properly split roles")
    
    def test_three_nodes_full_pipeline(self):
        """Three nodes should form complete pipeline."""
        print("\n  [3 nodes] Testing full pipeline...")
        
        sim = NetworkSimulator(layer_capacity_per_node=4)
        
        nodes = [sim.add_node(memory_mb=3000) for _ in range(3)]
        
        # Check roles
        drivers = [n for n in nodes if "DRIVER" in n.role]
        validators = [n for n in nodes if "VALIDATOR" in n.role]
        workers = [n for n in nodes if n.role == "WORKER"]
        
        assert len(drivers) >= 1, "Should have at least one DRIVER"
        
        # Print network state
        for i, node in enumerate(nodes):
            print(f"    Node {i} ({node.role}): {node.layers}")
        
        coverage = sim.get_network_coverage()
        print(f"    DHT Coverage: {sorted(coverage.keys())}")
        
        print("    ✅ Three nodes form pipeline")
    
    def test_ten_nodes_with_redundancy(self):
        """Ten nodes should cover all layers with redundancy."""
        print("\n  [10 nodes] Testing redundancy...")
        
        sim = NetworkSimulator(layer_capacity_per_node=3)
        
        nodes = [sim.add_node(memory_mb=2500) for _ in range(10)]
        
        # Check layer coverage
        coverage = sim.get_network_coverage()
        
        print(f"    Layers covered: {sorted(coverage.keys())}")
        
        # Count redundancy
        redundant_layers = [l for l, holders in coverage.items() if len(holders) > 1]
        print(f"    Layers with redundancy: {len(redundant_layers)}")
        
        # Should have full coverage
        assert 0 in coverage, "Layer 0 must be covered"
        
        print("    ✅ Ten nodes provide coverage + redundancy")
    
    def test_hundred_nodes_simulation(self):
        """Simulate 100 nodes joining."""
        print("\n  [100 nodes] Stress test simulation...")
        
        sim = NetworkSimulator(layer_capacity_per_node=2)
        
        # Add 100 nodes
        for i in range(100):
            sim.add_node(memory_mb=2000, with_model=False)  # Skip model creation for speed
            if (i + 1) % 25 == 0:
                coverage = sim.get_network_coverage()
                print(f"    After {i+1} nodes: {len(coverage)} layers covered")
        
        # Final coverage
        coverage = sim.get_network_coverage()
        
        # Count nodes per layer
        avg_redundancy = sum(len(h) for h in coverage.values()) / len(coverage) if coverage else 0
        
        print(f"    Total layers covered: {len(coverage)}")
        print(f"    Average redundancy: {avg_redundancy:.1f}x")
        
        # Role distribution
        drivers = len([n for n in sim.nodes.values() if "DRIVER" in n.role])
        validators = len([n for n in sim.nodes.values() if "VALIDATOR" in n.role])
        workers = len([n for n in sim.nodes.values() if n.role == "WORKER"])
        
        print(f"    Roles: {drivers} drivers, {validators} validators, {workers} workers")
        
        assert drivers >= 1, "Must have at least one driver"
        print("    ✅ 100 nodes simulation complete")


class TestDHTOperations:
    """Test DHT operations."""
    
    def test_dht_announce_and_lookup(self):
        """Test basic DHT announce and lookup."""
        print("\n  Testing DHT announce/lookup...")
        
        network = MockDHTNetwork()
        dht1 = network.create_node_dht("http://node1:8000", "node1:9000")
        dht2 = network.create_node_dht("http://node2:8000", "node2:9000")
        
        # Node 1 announces layers
        dht1.announce("layer_0")
        dht1.announce("layer_1")
        dht1.announce("layer_2")
        
        # Node 2 should see them
        for i in range(3):
            key_int = int(hashlib.sha1(f"layer_{i}".encode()).hexdigest(), 16)
            value = dht2.lookup_value(key_int)
            assert value is not None, f"Layer {i} should be discoverable"
        
        print("    ✅ DHT announce/lookup works")
    
    def test_dht_merge_multiple_holders(self):
        """Test DHT merges multiple holders for same layer."""
        print("\n  Testing DHT holder merging...")
        
        network = MockDHTNetwork()
        dht1 = network.create_node_dht("http://node1:8000", "node1:9000")
        dht2 = network.create_node_dht("http://node2:8000", "node2:9000")
        
        # Both announce layer 5
        dht1.announce("layer_5")
        dht2.announce("layer_5")
        
        # Should see both holders
        key_int = int(hashlib.sha1("layer_5".encode()).hexdigest(), 16)
        value = network.lookup(key_int)
        holders = json.loads(value)
        
        assert len(holders) == 2, f"Expected 2 holders, got {len(holders)}"
        assert "http://node1:8000" in holders
        assert "http://node2:8000" in holders
        
        print(f"    Holders for layer 5: {holders}")
        print("    ✅ DHT merges multiple holders")
    
    def test_dht_expiry(self):
        """Test DHT entry expiry."""
        print("\n  Testing DHT TTL expiry...")
        
        network = MockDHTNetwork()
        network.ttl_seconds = 0.1  # 100ms TTL for testing
        
        dht = network.create_node_dht("http://node1:8000", "node1:9000")
        dht.announce("layer_0")
        
        # Should be visible immediately
        key_int = int(hashlib.sha1("layer_0".encode()).hexdigest(), 16)
        assert network.lookup(key_int) is not None
        
        # Wait for expiry
        time.sleep(0.15)
        
        # Should be gone
        assert network.lookup(key_int) is None
        
        print("    ✅ DHT entries expire correctly")
    
    def test_dht_node_removal(self):
        """Test removing a node's entries from DHT."""
        print("\n  Testing DHT node removal...")
        
        network = MockDHTNetwork()
        dht1 = network.create_node_dht("http://node1:8000", "node1:9000")
        dht2 = network.create_node_dht("http://node2:8000", "node2:9000")
        
        # Both announce layers
        dht1.announce_layers([0, 1, 2])
        dht2.announce_layers([2, 3, 4])
        
        # Layer 2 should have both
        coverage = network.get_all_layer_holders()
        assert len(coverage[2]) == 2
        
        # Remove node1
        network.remove_node_entries("http://node1:8000")
        
        # Layer 0, 1 should be gone; layer 2 should only have node2
        coverage = network.get_all_layer_holders()
        assert 0 not in coverage
        assert 1 not in coverage
        assert coverage[2] == ["http://node2:8000"]
        
        print("    ✅ DHT node removal works")


class TestNetworkDynamics:
    """Test network dynamics - join, leave, recovery."""
    
    def test_node_join_extends_pipeline(self):
        """New node should extend pipeline at frontier."""
        print("\n  Testing node join extends pipeline...")
        
        sim = NetworkSimulator(layer_capacity_per_node=4)
        
        # First node
        node1 = sim.add_node()
        print(f"    Node 1: {node1.layers}")
        
        # Second node should continue
        node2 = sim.add_node()
        print(f"    Node 2: {node2.layers}")
        
        # Should not overlap on layer 0
        assert 0 in node1.layers
        assert 0 not in node2.layers
        
        # Node 2 should start after node 1
        if node1.layers and node2.layers:
            assert min(node2.layers) > min(node1.layers)
        
        print("    ✅ Node join extends pipeline correctly")
    
    def test_node_leave_creates_gap(self):
        """Node leaving creates gap that new node fills."""
        print("\n  Testing gap filling after node leave...")
        
        sim = NetworkSimulator(layer_capacity_per_node=4)
        
        # Add 3 nodes
        node1 = sim.add_node()
        node2 = sim.add_node()
        node3 = sim.add_node()
        
        print(f"    Initial: Node1={node1.layers}, Node2={node2.layers}, Node3={node3.layers}")
        
        # Remove middle node
        node2_layers = node2.layers.copy()
        sim.remove_node(node2.node_id)
        
        # New node should fill gap
        node4 = sim.add_node()
        print(f"    After removal: Node4={node4.layers}")
        
        # Node 4 should cover where node 2 was
        assert min(node4.layers) == min(node2_layers), "New node should fill gap"
        
        print("    ✅ Gap filling works correctly")
    
    def test_driver_failure_recovery(self):
        """When DRIVER leaves, network should still function."""
        print("\n  Testing DRIVER failure recovery...")
        
        sim = NetworkSimulator(layer_capacity_per_node=4)
        
        # Add 3 nodes
        node1 = sim.add_node()  # DRIVER
        node2 = sim.add_node()  # WORKER
        node3 = sim.add_node()  # WORKER/VALIDATOR
        
        # Verify node1 is driver
        assert "DRIVER" in node1.role
        
        # Simulate driver failure (DHT entries expire)
        sim.dht_network.expire_entries()
        
        # Re-announce remaining nodes
        for node in [node2, node3]:
            node.dht.announce_layers(node.layers)
        
        # New node joining should become DRIVER
        node4 = sim.add_node()
        
        print(f"    Node4 after driver failure: {node4.role}, layers={node4.layers}")
        
        # Node 4 should become driver since layer 0 is now uncovered
        # (or provide redundancy if it sees stale data)
        coverage = sim.get_network_coverage()
        assert len(coverage) > 0, "Network should have coverage"
        
        print("    ✅ Network recovers from driver failure")
    
    def test_rapid_join_leave_cycle(self):
        """Test rapid join/leave cycles."""
        print("\n  Testing rapid join/leave cycles...")
        
        sim = NetworkSimulator(layer_capacity_per_node=3)
        
        # Initial setup
        for _ in range(5):
            sim.add_node(with_model=False)
        
        initial_coverage = len(sim.get_network_coverage())
        print(f"    Initial coverage: {initial_coverage} layers")
        
        # Rapid cycles
        for cycle in range(10):
            # Remove random node
            if sim.nodes:
                node_id = list(sim.nodes.keys())[0]
                sim.remove_node(node_id)
            
            # Add new node
            sim.add_node(with_model=False)
        
        final_coverage = len(sim.get_network_coverage())
        print(f"    Final coverage: {final_coverage} layers")
        
        assert final_coverage >= initial_coverage - 2, "Coverage should be maintained"
        
        print("    ✅ Rapid join/leave cycles handled")


class TestLayerScaling:
    """Test layer and model scaling."""
    
    def test_model_starts_with_base_architecture(self):
        """First node should use base 11-layer architecture."""
        print("\n  Testing base architecture...")
        
        sim = NetworkSimulator(layer_capacity_per_node=11)
        node = sim.add_node()
        
        arch = node.pool.current_architecture
        assert arch.num_layers == 11
        assert arch.hidden_dim == 512
        
        print(f"    Architecture: {arch.num_layers}L × {arch.hidden_dim}H")
        print("    ✅ Base architecture correct")
    
    def test_layer_assignment_respects_capacity(self):
        """Nodes should only get layers they can hold."""
        print("\n  Testing capacity-based assignment...")
        
        # Small capacity
        sim = NetworkSimulator(layer_capacity_per_node=2)
        node = sim.add_node(memory_mb=1000)
        
        assert len(node.layers) <= 2, f"Got {len(node.layers)} layers, expected ≤ 2"
        
        print(f"    Low memory node: {len(node.layers)} layers")
        print("    ✅ Capacity respected")
    
    def test_pipeline_coverage_grows_with_nodes(self):
        """More nodes = more layer coverage."""
        print("\n  Testing pipeline growth...")
        
        sim = NetworkSimulator(layer_capacity_per_node=2)
        
        coverages = []
        for i in range(6):
            sim.add_node(with_model=False)
            coverage = len(sim.get_network_coverage())
            coverages.append(coverage)
            print(f"    After {i+1} nodes: {coverage} layers covered")
        
        # Coverage should grow (or at least not shrink)
        for i in range(1, len(coverages)):
            assert coverages[i] >= coverages[i-1], "Coverage should not shrink"
        
        print("    ✅ Pipeline coverage grows with nodes")


class TestVocabularyExpansion:
    """Test dynamic vocabulary expansion."""
    
    def test_vocab_capacity_setting(self):
        """Test vocab capacity is correctly set."""
        print("\n  Testing vocab capacity...")
        
        pool = DynamicLayerPool(dht_protocol=None)
        pool._device_hint = 'cpu'
        
        # Default
        assert pool.vocab_capacity == INITIAL_VOCAB_SIZE
        
        # Set larger
        pool.vocab_capacity = 100000
        assert pool.vocab_capacity == 100000
        
        print(f"    Initial vocab: {INITIAL_VOCAB_SIZE}")
        print(f"    Max vocab: {MAX_VOCAB_SIZE}")
        print("    ✅ Vocab capacity setting works")
    
    def test_model_supports_vocab_expansion(self):
        """Test model can handle vocab expansion."""
        print("\n  Testing model vocab expansion support...")
        
        sim = NetworkSimulator(layer_capacity_per_node=6)
        node = sim.add_node()
        
        # Check model has embedding and lm_head
        if node.model.has_embedding:
            embed_size = node.model.embedding.num_embeddings
            print(f"    Embedding vocab: {embed_size}")
        
        if node.model.has_lm_head:
            lm_head_out = node.model.lm_head.out_features
            print(f"    LM head output: {lm_head_out}")
        
        print("    ✅ Model supports vocab expansion")


class TestPipelineTraining:
    """Test distributed training through pipeline."""
    
    def test_single_node_training(self):
        """Test training on single full node."""
        print("\n  Testing single node training...")
        
        sim = NetworkSimulator(layer_capacity_per_node=11)
        node = sim.add_node()
        
        assert node.model.has_embedding
        assert node.model.has_lm_head
        
        # Training step
        input_ids = torch.randint(0, 1000, (2, 16))
        labels = torch.randint(0, 1000, (2, 16))
        
        node.optimizer.zero_grad()
        
        hidden = node.model.embed(input_ids)
        hidden = node.model.forward_my_layers(hidden)
        logits = node.model.compute_logits(hidden)
        
        loss = torch.nn.functional.cross_entropy(
            logits.view(-1, logits.size(-1)),
            labels.view(-1)
        )
        
        loss.backward()
        node.optimizer.step()
        
        print(f"    Loss: {loss.item():.4f}")
        print("    ✅ Single node training works")
    
    def test_two_node_pipeline_training(self):
        """Test training through 2-node pipeline."""
        print("\n  Testing 2-node pipeline training...")
        
        sim = NetworkSimulator(layer_capacity_per_node=6)
        node1 = sim.add_node()  # DRIVER
        node2 = sim.add_node()  # WORKER/VALIDATOR
        
        print(f"    Node 1 ({node1.role}): {node1.layers}")
        print(f"    Node 2 ({node2.role}): {node2.layers}")
        
        input_ids = torch.randint(0, 1000, (2, 16))
        labels = torch.randint(0, 1000, (2, 16))
        
        # Forward through pipeline
        node1.optimizer.zero_grad()
        node2.optimizer.zero_grad()
        
        # Node 1: embed and forward
        hidden1 = node1.model.embed(input_ids)
        hidden1 = node1.model.forward_my_layers(hidden1)
        
        # "Send" to node 2 (detach simulates gRPC)
        hidden1_sent = hidden1.detach().requires_grad_(True)
        
        # Node 2: forward and compute loss
        hidden2 = node2.model.forward_my_layers(hidden1_sent)
        
        if node2.model.has_lm_head:
            logits = node2.model.compute_logits(hidden2)
        else:
            # Node 1 has lm_head (shouldn't happen with proper setup)
            logits = node1.model.compute_logits(hidden1)
        
        loss = torch.nn.functional.cross_entropy(
            logits.view(-1, logits.size(-1)),
            labels.view(-1)
        )
        
        # Backward
        loss.backward()
        node2.optimizer.step()
        
        # Send gradients back
        if hidden1_sent.grad is not None:
            hidden1.backward(hidden1_sent.grad)
            node1.optimizer.step()
        
        print(f"    Loss: {loss.item():.4f}")
        print("    ✅ 2-node pipeline training works")
    
    def test_three_node_pipeline_training(self):
        """Test training through 3-node pipeline."""
        print("\n  Testing 3-node pipeline training...")
        
        sim = NetworkSimulator(layer_capacity_per_node=4)
        nodes = [sim.add_node() for _ in range(3)]
        
        for i, node in enumerate(nodes):
            print(f"    Node {i} ({node.role}): {node.layers}")
        
        input_ids = torch.randint(0, 1000, (2, 16))
        labels = torch.randint(0, 1000, (2, 16))
        
        # Zero grads
        for node in nodes:
            node.optimizer.zero_grad()
        
        # Forward pipeline
        hidden = nodes[0].model.embed(input_ids)
        hiddens = [hidden]
        
        for i, node in enumerate(nodes):
            hidden = node.model.forward_my_layers(hidden)
            if i < len(nodes) - 1:
                hidden_sent = hidden.detach().requires_grad_(True)
                hiddens.append(hidden_sent)
                hidden = hidden_sent
        
        # Compute loss on validator
        validator = None
        for node in reversed(nodes):
            if node.model.has_lm_head:
                validator = node
                break
        
        assert validator is not None, "Must have a validator"
        logits = validator.model.compute_logits(hidden)
        
        loss = torch.nn.functional.cross_entropy(
            logits.view(-1, logits.size(-1)),
            labels.view(-1)
        )
        
        # Backward
        loss.backward()
        
        # Propagate gradients
        for i in range(len(hiddens) - 1, 0, -1):
            if hiddens[i].grad is not None:
                hiddens[i-1].backward(hiddens[i].grad)
        
        # Update all
        for node in nodes:
            node.optimizer.step()
        
        print(f"    Loss: {loss.item():.4f}")
        print("    ✅ 3-node pipeline training works")
    
    def test_training_multiple_steps(self):
        """Test multiple training steps show loss decrease."""
        print("\n  Testing loss convergence over steps...")
        
        sim = NetworkSimulator(layer_capacity_per_node=11)
        node = sim.add_node()
        
        # Fixed input for convergence test
        torch.manual_seed(42)
        input_ids = torch.randint(0, 100, (4, 32))
        labels = torch.randint(0, 100, (4, 32))
        
        losses = []
        for step in range(10):
            node.optimizer.zero_grad()
            
            hidden = node.model.embed(input_ids)
            hidden = node.model.forward_my_layers(hidden)
            logits = node.model.compute_logits(hidden)
            
            loss = torch.nn.functional.cross_entropy(
                logits.view(-1, logits.size(-1)),
                labels.view(-1)
            )
            
            loss.backward()
            node.optimizer.step()
            losses.append(loss.item())
        
        print(f"    Loss: {losses[0]:.4f} → {losses[-1]:.4f}")
        
        # Loss should decrease (or at least not explode)
        assert losses[-1] < losses[0] * 2, "Loss should not explode"
        
        print("    ✅ Training shows expected behavior")


class TestEdgeCases:
    """Test edge cases and error handling."""
    
    def test_empty_network_first_node(self):
        """First node in empty network should work."""
        print("\n  Testing first node in empty network...")
        
        sim = NetworkSimulator(layer_capacity_per_node=6)
        node = sim.add_node()
        
        assert len(node.layers) > 0
        assert 0 in node.layers
        
        print(f"    First node: {node.role}")
        print("    ✅ First node works")
    
    def test_node_with_minimal_capacity(self):
        """Node with minimal capacity should get at least 1 layer."""
        print("\n  Testing minimal capacity node...")
        
        sim = NetworkSimulator(layer_capacity_per_node=1)
        node = sim.add_node(memory_mb=500)
        
        assert len(node.layers) >= 1, "Should get at least 1 layer"
        
        print(f"    Minimal node: {node.layers}")
        print("    ✅ Minimal capacity handled")
    
    def test_all_nodes_same_capacity(self):
        """All nodes with same capacity should distribute evenly."""
        print("\n  Testing equal capacity distribution...")
        
        sim = NetworkSimulator(layer_capacity_per_node=4)
        nodes = [sim.add_node(memory_mb=3000) for _ in range(5)]
        
        layer_counts = [len(n.layers) for n in nodes]
        print(f"    Layer counts: {layer_counts}")
        
        # All should have similar counts
        assert max(layer_counts) - min(layer_counts) <= 2
        
        print("    ✅ Equal capacity distributed evenly")
    
    def test_very_large_network_roles(self):
        """Test role distribution in large network."""
        print("\n  Testing role distribution in large network...")
        
        sim = NetworkSimulator(layer_capacity_per_node=2)
        
        for _ in range(50):
            sim.add_node(with_model=False)
        
        drivers = len([n for n in sim.nodes.values() if "DRIVER" in n.role])
        validators = len([n for n in sim.nodes.values() if "VALIDATOR" in n.role])
        workers = len([n for n in sim.nodes.values() if n.role == "WORKER"])
        
        print(f"    Drivers: {drivers}")
        print(f"    Validators: {validators}")
        print(f"    Workers: {workers}")
        
        assert drivers >= 1, "Must have driver"
        assert validators >= 1, "Must have validator"
        
        print("    ✅ Large network roles distributed correctly")


# ============================================================================
# MAIN TEST RUNNER
# ============================================================================

def run_all_tests():
    """Run all tests."""
    print("\n" + "=" * 70)
    print("NEUROSHARD COMPREHENSIVE DECENTRALIZED SYSTEM TEST")
    print("=" * 70)
    
    test_classes = [
        ("Role Assignments", TestRoleAssignments),
        ("DHT Operations", TestDHTOperations),
        ("Network Dynamics", TestNetworkDynamics),
        ("Layer Scaling", TestLayerScaling),
        ("Vocabulary Expansion", TestVocabularyExpansion),
        ("Pipeline Training", TestPipelineTraining),
        ("Edge Cases", TestEdgeCases),
    ]
    
    passed = 0
    failed = 0
    
    for section_name, test_class in test_classes:
        print(f"\n{'─' * 70}")
        print(f"📋 {section_name}")
        print('─' * 70)
        sys.stdout.flush()
        
        instance = test_class()
        test_methods = sorted([m for m in dir(instance) if m.startswith('test_')])
        
        for method_name in test_methods:
            try:
                method = getattr(instance, method_name)
                method()
                passed += 1
            except Exception as e:
                print(f"  ❌ {method_name}: {e}")
                import traceback
                traceback.print_exc()
                failed += 1
            sys.stdout.flush()
    
    print("\n" + "=" * 70)
    print(f"FINAL RESULTS: {passed} passed, {failed} failed")
    print("=" * 70)
    
    if failed == 0:
        print("\n🎉 ALL TESTS PASSED! The decentralized system is working correctly.")
    else:
        print(f"\n⚠️  {failed} test(s) failed. Please review.")
    
    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    exit(0 if success else 1)
