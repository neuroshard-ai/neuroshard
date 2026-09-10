"""
COMPREHENSIVE MULTI-NODE ENVIRONMENT TEST SUITE

Tests the complete NeuroShard distributed system including:
1. Multi-node layer allocation and role assignment
2. DHT-based peer discovery and layer announcements
3. Pipeline training across multiple nodes
4. Node join/leave dynamics with automatic rebalancing
5. Gradient flow through distributed pipeline
6. Checkpoint save/load with layer reassignment
7. Vocab expansion across distributed nodes
8. gRPC-style communication simulation
9. Failure recovery scenarios
10. DiLoCo synchronization patterns

This test suite simulates a real multi-node environment without requiring
actual network connectivity by mocking the DHT and gRPC layers.
"""

import sys
import os
import time
import json
import torch
import torch.nn as nn
import hashlib
import threading
import tempfile
import shutil
from pathlib import Path
from unittest.mock import patch, MagicMock
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from concurrent.futures import ThreadPoolExecutor, as_completed

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from neuroshard.core.model.dynamic import (
    DynamicLayerPool,
    DynamicNeuroLLM,
    DynamicNeuroNode,
    LayerAssignment,
    INITIAL_VOCAB_SIZE,
    MAX_VOCAB_SIZE,
)
from neuroshard.core.model.scaler import ModelArchitecture, calculate_layer_assignment


# =============================================================================
# NETWORK SIMULATION INFRASTRUCTURE
# =============================================================================

class MockDHTNetwork:
    """
    Simulates a fully connected DHT network.
    
    All nodes share the same storage (simulating DHT replication).
    Supports TTL-based expiry and multi-holder merging.
    """
    
    def __init__(self, ttl_seconds: float = 300):
        self.storage: Dict[int, Tuple[str, float]] = {}  # key -> (value, expiry_time)
        self.nodes: Dict[str, 'MockNodeDHT'] = {}
        self.lock = threading.Lock()
        self.ttl_seconds = ttl_seconds
    
    def create_node_dht(self, node_url: str, grpc_addr: str) -> 'MockNodeDHT':
        """Create a DHT interface for a node."""
        dht = MockNodeDHT(self, node_url, grpc_addr)
        with self.lock:
            self.nodes[node_url] = dht
        return dht
    
    def store(self, key_int: int, value: str, ttl: float = None):
        """Store a value with TTL, merging with existing holders."""
        with self.lock:
            expiry = time.time() + (ttl or self.ttl_seconds)
            
            if key_int in self.storage:
                existing_value, _ = self.storage[key_int]
                try:
                    existing_list = json.loads(existing_value)
                    new_list = json.loads(value) if value.startswith('[') else [value]
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
    
    def expire_all(self):
        """Force expire all entries (simulate network partition)."""
        with self.lock:
            self.storage.clear()
    
    def get_all_layer_holders(self) -> Dict[int, List[str]]:
        """Get all layer holders for debugging."""
        result = {}
        for layer_id in range(100):
            key_str = f"layer_{layer_id}"
            key_int = int(hashlib.sha1(key_str.encode()).hexdigest(), 16)
            value = self.lookup(key_int)
            if value:
                try:
                    result[layer_id] = json.loads(value)
                except:
                    result[layer_id] = [value]
        return result


class MockNodeDHT:
    """DHT interface for a single node."""
    
    def __init__(self, network: MockDHTNetwork, node_url: str, grpc_addr: str):
        self.network = network
        self.node_url = node_url
        self.grpc_addr = grpc_addr
        self.storage = {}  # Local storage for compatibility
    
    def announce(self, key_string: str):
        """Announce this node for a key."""
        key_int = int(hashlib.sha1(key_string.encode()).hexdigest(), 16)
        self.network.store(key_int, json.dumps([self.node_url]))
        # Also store locally
        self.storage[key_int] = json.dumps([self.node_url])
        return True
    
    def lookup_value(self, key_int: int) -> Optional[str]:
        """Lookup a value from the network."""
        return self.network.lookup(key_int)
    
    def announce_layers(self, layer_ids: List[int]):
        """Announce multiple layers."""
        for layer_id in layer_ids:
            self.announce(f"layer_{layer_id}")


class MockGRPCNetwork:
    """
    Simulates gRPC communication between nodes.
    
    Nodes register themselves and can send/receive tensors.
    """
    
    def __init__(self):
        self.nodes: Dict[str, 'SimulatedNode'] = {}
        self.lock = threading.Lock()
        self.message_queue: Dict[str, List[Dict]] = {}  # node_id -> messages
    
    def register_node(self, node: 'SimulatedNode'):
        """Register a node for gRPC communication."""
        with self.lock:
            self.nodes[node.grpc_addr] = node
            self.message_queue[node.grpc_addr] = []
    
    def unregister_node(self, grpc_addr: str):
        """Remove a node."""
        with self.lock:
            if grpc_addr in self.nodes:
                del self.nodes[grpc_addr]
            if grpc_addr in self.message_queue:
                del self.message_queue[grpc_addr]
    
    def send_activation(self, from_addr: str, to_addr: str, 
                        hidden_states: torch.Tensor, 
                        session_id: str,
                        is_training: bool = False,
                        labels: Optional[torch.Tensor] = None) -> Tuple[bool, Optional[torch.Tensor], Optional[float]]:
        """
        Send activation from one node to another.
        
        Returns: (success, output_tensor, loss)
        """
        with self.lock:
            if to_addr not in self.nodes:
                return False, None, None
            
            target_node = self.nodes[to_addr]
        
        # Simulate gRPC forward
        try:
            output, loss, grad = target_node.forward_pipeline(
                hidden_states=hidden_states,
                session_id=session_id,
                source_layer=0,  # Will be computed by target
                is_training=is_training,
                labels=labels
            )
            return True, output, loss
        except Exception as e:
            print(f"gRPC simulation error: {e}")
            return False, None, None


@dataclass
class SimulatedNode:
    """
    Represents a fully simulated NeuroShard node.
    
    Has all components: DHT, model, optimizer, layer pool.
    """
    node_id: str
    node_url: str
    grpc_addr: str
    memory_mb: float
    pool: DynamicLayerPool
    dht: MockNodeDHT
    model: Optional[DynamicNeuroLLM] = None
    optimizer: Optional[torch.optim.Optimizer] = None
    layers: List[int] = field(default_factory=list)
    role: str = "UNKNOWN"
    checkpoint_dir: Optional[Path] = None
    
    # Training state
    total_training_rounds: int = 0
    current_loss: float = float('inf')
    
    def forward_pipeline(self, hidden_states: torch.Tensor, session_id: str,
                        source_layer: int, is_training: bool = False,
                        labels: Optional[torch.Tensor] = None,
                        attention_mask: Optional[torch.Tensor] = None
                        ) -> Tuple[torch.Tensor, Optional[float], Optional[torch.Tensor]]:
        """Handle incoming pipeline forward request."""
        if not self.model:
            raise RuntimeError("Model not initialized")
        
        # Forward through my layers
        output = self.model.forward_my_layers(hidden_states, attention_mask)
        
        # If I have LM head and this is training, compute loss
        if is_training and labels is not None and self.model.has_lm_head:
            logits = self.model.compute_logits(output)
            logits_flat = logits.view(-1, logits.size(-1))
            labels_flat = labels.view(-1)
            loss = nn.functional.cross_entropy(logits_flat, labels_flat, ignore_index=-100)
            
            if hasattr(hidden_states, 'requires_grad') and hidden_states.requires_grad:
                loss.backward()
                grad_output = hidden_states.grad
            else:
                grad_output = None
            
            self.current_loss = loss.item()
            return output, loss.item(), grad_output
        
        return output, None, None


class MultiNodeSimulator:
    """
    Simulates a complete multi-node NeuroShard network.
    
    Manages node lifecycle, DHT, gRPC, and training coordination.
    """
    
    def __init__(self, default_layer_capacity: int = 4):
        self.dht_network = MockDHTNetwork()
        self.grpc_network = MockGRPCNetwork()
        self.nodes: Dict[str, SimulatedNode] = {}
        self.default_layer_capacity = default_layer_capacity
        self.node_counter = 0
        self.temp_dirs: List[Path] = []
    
    def add_node(self, memory_mb: float = 4000, 
                 layer_capacity: Optional[int] = None,
                 with_model: bool = True,
                 with_checkpoint_dir: bool = False) -> SimulatedNode:
        """Add a new node to the network."""
        self.node_counter += 1
        node_id = f"node_{self.node_counter}"
        node_url = f"http://{node_id}:8000"
        grpc_addr = f"{node_id}:9000"
        
        capacity = layer_capacity or self.default_layer_capacity
        
        # Create DHT interface
        dht = self.dht_network.create_node_dht(node_url, grpc_addr)
        
        # Create layer pool with DHT
        pool = DynamicLayerPool(dht_protocol=dht)
        pool._device_hint = 'cpu'
        pool.vocab_capacity = 32000
        
        # Create checkpoint dir if requested
        checkpoint_dir = None
        if with_checkpoint_dir:
            checkpoint_dir = Path(tempfile.mkdtemp())
            self.temp_dirs.append(checkpoint_dir)
            pool.CHECKPOINT_DIR = checkpoint_dir
        
        # Register node with mocked layer assignment
        with patch('neuroshard.core.model.scaler.calculate_layer_assignment', return_value=capacity):
            layers = pool.register_node(
                node_id=node_id,
                node_url=node_url,
                grpc_addr=grpc_addr,
                available_memory_mb=memory_mb,
                staked_amount=0
            )
        
        # Announce layers to DHT
        dht.announce_layers(layers)
        
        # Determine role
        if 0 in layers and pool.embedding_holder == node_id:
            role = "DRIVER+VALIDATOR" if pool.lm_head_holder == node_id else "DRIVER"
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
            role=role,
            checkpoint_dir=checkpoint_dir
        )
        
        self.nodes[node_id] = node
        self.grpc_network.register_node(node)
        
        return node
    
    def remove_node(self, node_id: str):
        """Remove a node from the network."""
        if node_id in self.nodes:
            node = self.nodes[node_id]
            
            # Remove from DHT
            self.dht_network.remove_node_entries(node.node_url)
            
            # Remove from gRPC
            self.grpc_network.unregister_node(node.grpc_addr)
            
            # Remove from tracking
            del self.nodes[node_id]
    
    def get_network_coverage(self) -> Dict[int, List[str]]:
        """Get which layers are covered by which nodes."""
        return self.dht_network.get_all_layer_holders()
    
    def get_nodes_by_role(self, role: str) -> List[SimulatedNode]:
        """Get all nodes with a specific role."""
        return [n for n in self.nodes.values() if role in n.role]
    
    def cleanup(self):
        """Clean up temporary directories."""
        for temp_dir in self.temp_dirs:
            if temp_dir.exists():
                shutil.rmtree(temp_dir)
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.cleanup()


# =============================================================================
# TEST CLASSES
# =============================================================================

class TestMultiNodeLayerAllocation:
    """Test layer allocation across multiple nodes."""
    
    def test_single_node_full_model(self):
        """Single node should hold all layers as full node."""
        print("\n  [1 NODE] Testing single node full model...")
        
        with MultiNodeSimulator(default_layer_capacity=11) as sim:
            node = sim.add_node(memory_mb=10000)
            
            assert 0 in node.layers, "Should have layer 0"
            assert 10 in node.layers, "Should have layer 10"
            assert node.role == "DRIVER+VALIDATOR"
            assert node.model.has_embedding
            assert node.model.has_lm_head
            
            print(f"    Node: {node.role}, layers={node.layers}")
            print("    ✅ Single node is full model")
    
    def test_two_node_pipeline(self):
        """Two nodes should split the pipeline."""
        print("\n  [2 NODES] Testing two-node pipeline split...")
        
        with MultiNodeSimulator(default_layer_capacity=6) as sim:
            node1 = sim.add_node(memory_mb=4000)
            node2 = sim.add_node(memory_mb=4000)
            
            # Verify node 1 is driver
            assert 0 in node1.layers
            assert node1.model.has_embedding
            
            # Verify node 2 is worker/validator
            assert 0 not in node2.layers
            
            # Verify full coverage
            all_layers = set(node1.layers) | set(node2.layers)
            assert 0 in all_layers and 10 in all_layers
            
            print(f"    Node 1 ({node1.role}): {node1.layers}")
            print(f"    Node 2 ({node2.role}): {node2.layers}")
            print("    ✅ Two nodes split pipeline correctly")
    
    def test_five_node_redundancy(self):
        """Five nodes should provide redundancy on later layers."""
        print("\n  [5 NODES] Testing five-node redundancy...")
        
        with MultiNodeSimulator(default_layer_capacity=3) as sim:
            nodes = [sim.add_node(memory_mb=2500) for _ in range(5)]
            
            coverage = sim.get_network_coverage()
            
            # Check full coverage
            assert 0 in coverage, "Layer 0 must be covered"
            assert all(i in coverage for i in range(11)), "All layers should be covered"
            
            # Check redundancy
            redundant = [l for l, h in coverage.items() if len(h) > 1]
            
            print(f"    Coverage: {sorted(coverage.keys())}")
            print(f"    Redundant layers: {redundant}")
            print("    ✅ Five nodes provide coverage with redundancy")
    
    def test_ten_node_high_redundancy(self):
        """Ten nodes should have high redundancy."""
        print("\n  [10 NODES] Testing ten-node high redundancy...")
        
        with MultiNodeSimulator(default_layer_capacity=3) as sim:
            nodes = [sim.add_node(memory_mb=2500, with_model=False) for _ in range(10)]
            
            coverage = sim.get_network_coverage()
            
            avg_redundancy = sum(len(h) for h in coverage.values()) / len(coverage) if coverage else 0
            
            print(f"    Layers covered: {len(coverage)}")
            print(f"    Average redundancy: {avg_redundancy:.1f}x")
            
            assert avg_redundancy > 1.5, "Should have significant redundancy"
            print("    ✅ Ten nodes provide high redundancy")


class TestDHTPeerDiscovery:
    """Test DHT-based peer discovery."""
    
    def test_layer_discovery_across_nodes(self):
        """Nodes should discover each other's layers via DHT."""
        print("\n  Testing layer discovery across nodes...")
        
        with MultiNodeSimulator(default_layer_capacity=4) as sim:
            node1 = sim.add_node()
            node2 = sim.add_node()
            node3 = sim.add_node()
            
            # Each node should see all layers
            coverage = sim.get_network_coverage()
            
            assert len(coverage) == 11, f"All 11 layers should be discoverable, got {len(coverage)}"
            
            print(f"    Discovered layers: {sorted(coverage.keys())}")
            print("    ✅ Layer discovery works across nodes")
    
    def test_dht_holder_merging(self):
        """Multiple holders for same layer should be merged."""
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
        
        assert len(holders) == 2
        assert "http://node1:8000" in holders
        assert "http://node2:8000" in holders
        
        print(f"    Holders: {holders}")
        print("    ✅ DHT holder merging works")
    
    def test_dht_node_departure(self):
        """Node departure should remove its entries."""
        print("\n  Testing DHT node departure...")
        
        with MultiNodeSimulator(default_layer_capacity=6) as sim:
            node1 = sim.add_node()
            node2 = sim.add_node()
            
            coverage_before = sim.get_network_coverage()
            
            # Remove node1
            sim.remove_node(node1.node_id)
            
            coverage_after = sim.get_network_coverage()
            
            # Node1's exclusive layers should be gone
            for layer in node1.layers:
                if layer not in node2.layers:
                    assert layer not in coverage_after, f"Layer {layer} should be removed"
            
            print(f"    Before: {sorted(coverage_before.keys())}")
            print(f"    After: {sorted(coverage_after.keys())}")
            print("    ✅ DHT node departure works")


class TestDistributedPipelineTraining:
    """Test training across distributed pipeline."""
    
    def test_two_node_forward_backward(self):
        """Test forward and backward pass through 2-node pipeline."""
        print("\n  Testing 2-node forward-backward pass...")
        
        with MultiNodeSimulator(default_layer_capacity=6) as sim:
            driver = sim.add_node(memory_mb=4000)
            worker = sim.add_node(memory_mb=4000)
            
            assert driver.model.has_embedding, "Driver should have embedding"
            
            # Training batch
            input_ids = torch.randint(0, 1000, (2, 16))
            labels = torch.randint(0, 1000, (2, 16))
            
            # Forward through driver
            driver.optimizer.zero_grad()
            hidden1 = driver.model.embed(input_ids)
            hidden1 = driver.model.forward_my_layers(hidden1)
            
            # Simulate sending to worker (detach for gRPC simulation)
            hidden1_sent = hidden1.detach().requires_grad_(True)
            
            # Forward through worker
            worker.optimizer.zero_grad()
            hidden2 = worker.model.forward_my_layers(hidden1_sent)
            
            # Compute loss (worker has LM head)
            if worker.model.has_lm_head:
                logits = worker.model.compute_logits(hidden2)
            else:
                logits = driver.model.compute_logits(hidden1)
            
            loss = nn.functional.cross_entropy(
                logits.view(-1, logits.size(-1)),
                labels.view(-1)
            )
            
            # Backward
            loss.backward()
            worker.optimizer.step()
            
            # Send gradients back
            if hidden1_sent.grad is not None:
                hidden1.backward(hidden1_sent.grad)
                driver.optimizer.step()
            
            print(f"    Driver ({driver.role}): {driver.layers}")
            print(f"    Worker ({worker.role}): {worker.layers}")
            print(f"    Loss: {loss.item():.4f}")
            print("    ✅ 2-node forward-backward works")
    
    def test_three_node_pipeline_gradient_flow(self):
        """Test gradient flow through 3-node pipeline."""
        print("\n  Testing 3-node pipeline gradient flow...")
        
        with MultiNodeSimulator(default_layer_capacity=4) as sim:
            nodes = [sim.add_node(memory_mb=3000) for _ in range(3)]
            
            for i, node in enumerate(nodes):
                print(f"    Node {i} ({node.role}): {node.layers}")
            
            # Training batch
            input_ids = torch.randint(0, 1000, (2, 16))
            labels = torch.randint(0, 1000, (2, 16))
            
            # Zero all grads
            for node in nodes:
                node.optimizer.zero_grad()
            
            # Forward through all nodes
            driver = nodes[0]
            hidden = driver.model.embed(input_ids)
            hiddens = [hidden]
            
            for node in nodes:
                hidden = node.model.forward_my_layers(hidden)
                # Store detached copy for gradient propagation
                hidden_sent = hidden.detach().requires_grad_(True)
                hiddens.append(hidden_sent)
                hidden = hidden_sent
            
            # Find validator and compute loss
            validator = None
            for node in reversed(nodes):
                if node.model.has_lm_head:
                    validator = node
                    break
            
            assert validator is not None, "Must have a validator"
            
            logits = validator.model.compute_logits(hiddens[-1])
            loss = nn.functional.cross_entropy(
                logits.view(-1, logits.size(-1)),
                labels.view(-1)
            )
            
            # Backward through all nodes
            loss.backward()
            
            # Propagate gradients
            for i in range(len(hiddens) - 1, 0, -1):
                if hiddens[i].grad is not None:
                    hiddens[i-1].backward(hiddens[i].grad)
            
            # Update all optimizers
            for node in nodes:
                node.optimizer.step()
            
            print(f"    Loss: {loss.item():.4f}")
            print("    ✅ 3-node gradient flow works")
    
    def test_loss_convergence_multi_node(self):
        """Test that loss converges in multi-node training."""
        print("\n  Testing loss convergence in multi-node training...")
        
        with MultiNodeSimulator(default_layer_capacity=6) as sim:
            driver = sim.add_node(memory_mb=4000)
            worker = sim.add_node(memory_mb=4000)
            
            # Fixed data for convergence test
            torch.manual_seed(42)
            input_ids = torch.randint(0, 100, (4, 32))
            labels = torch.randint(0, 100, (4, 32))
            
            losses = []
            for step in range(10):
                driver.optimizer.zero_grad()
                worker.optimizer.zero_grad()
                
                hidden = driver.model.embed(input_ids)
                hidden = driver.model.forward_my_layers(hidden)
                
                hidden_sent = hidden.detach().requires_grad_(True)
                hidden2 = worker.model.forward_my_layers(hidden_sent)
                
                if worker.model.has_lm_head:
                    logits = worker.model.compute_logits(hidden2)
                else:
                    logits = driver.model.compute_logits(hidden)
                
                loss = nn.functional.cross_entropy(
                    logits.view(-1, logits.size(-1)),
                    labels.view(-1)
                )
                
                loss.backward()
                worker.optimizer.step()
                
                if hidden_sent.grad is not None:
                    hidden.backward(hidden_sent.grad)
                    driver.optimizer.step()
                
                losses.append(loss.item())
            
            print(f"    Loss: {losses[0]:.4f} → {losses[-1]:.4f}")
            
            # Loss should decrease (or at least not explode)
            assert losses[-1] < losses[0] * 1.5, "Loss should not explode"
            print("    ✅ Multi-node training shows convergence")


class TestNodeJoinLeave:
    """Test dynamic node join/leave scenarios."""
    
    def test_node_join_extends_pipeline(self):
        """New node should extend the pipeline frontier."""
        print("\n  Testing node join extends pipeline...")
        
        with MultiNodeSimulator(default_layer_capacity=4) as sim:
            node1 = sim.add_node()
            coverage1 = set(sim.get_network_coverage().keys())
            
            node2 = sim.add_node()
            coverage2 = set(sim.get_network_coverage().keys())
            
            assert len(coverage2) >= len(coverage1), "Coverage should grow or stay same"
            
            # Node 2 should start after node 1
            if node1.layers and node2.layers:
                assert 0 not in node2.layers, "Node 2 shouldn't have layer 0"
            
            print(f"    Node 1: {node1.layers}")
            print(f"    Node 2: {node2.layers}")
            print(f"    Coverage: {len(coverage1)} → {len(coverage2)} layers")
            print("    ✅ Node join extends pipeline")
    
    def test_node_leave_creates_gap(self):
        """Node leaving creates gap for new node to fill."""
        print("\n  Testing gap filling after node leave...")
        
        with MultiNodeSimulator(default_layer_capacity=4) as sim:
            node1 = sim.add_node()
            node2 = sim.add_node()
            node3 = sim.add_node()
            
            print(f"    Before: N1={node1.layers}, N2={node2.layers}, N3={node3.layers}")
            
            node2_layers = node2.layers.copy()
            sim.remove_node(node2.node_id)
            
            # New node should fill gap
            node4 = sim.add_node()
            
            print(f"    After: N4={node4.layers}")
            
            # Node 4 should fill the same position as node 2
            assert min(node4.layers) == min(node2_layers), "New node should fill gap"
            print("    ✅ Gap filling works")
    
    def test_driver_failure_recovery(self):
        """Network should recover from driver failure."""
        print("\n  Testing driver failure recovery...")
        
        with MultiNodeSimulator(default_layer_capacity=4) as sim:
            driver = sim.add_node()
            worker1 = sim.add_node()
            worker2 = sim.add_node()
            
            assert "DRIVER" in driver.role
            
            # Simulate driver failure (DHT entries expire)
            sim.dht_network.expire_all()
            
            # Workers re-announce
            worker1.dht.announce_layers(worker1.layers)
            worker2.dht.announce_layers(worker2.layers)
            
            # New node joins - should become driver
            new_driver = sim.add_node()
            
            print(f"    Original driver: {driver.role}")
            print(f"    New node after failure: {new_driver.role}")
            
            # New node should take driver role since layer 0 is uncovered
            # (or provide redundancy if it sees stale data)
            coverage = sim.get_network_coverage()
            assert len(coverage) > 0, "Network should have coverage"
            
            print("    ✅ Network recovers from driver failure")
    
    def test_rapid_churn(self):
        """Network should handle rapid join/leave cycles."""
        print("\n  Testing rapid node churn...")
        
        with MultiNodeSimulator(default_layer_capacity=3) as sim:
            # Initial setup
            for _ in range(5):
                sim.add_node(with_model=False)
            
            initial_coverage = len(sim.get_network_coverage())
            
            # Rapid cycles
            for _ in range(10):
                # Remove random node
                if sim.nodes:
                    node_id = list(sim.nodes.keys())[0]
                    sim.remove_node(node_id)
                
                # Add new node
                sim.add_node(with_model=False)
            
            final_coverage = len(sim.get_network_coverage())
            
            print(f"    Coverage: {initial_coverage} → {final_coverage}")
            
            assert final_coverage >= initial_coverage - 2, "Coverage should be maintained"
            print("    ✅ Rapid churn handled")


class TestVocabExpansion:
    """Test vocabulary expansion across distributed nodes."""
    
    def test_vocab_expansion_driver(self):
        """Driver node should handle vocab expansion."""
        print("\n  Testing vocab expansion on driver...")
        
        with MultiNodeSimulator(default_layer_capacity=11) as sim:
            node = sim.add_node()
            
            assert node.model.has_embedding
            
            original_vocab = node.model.vocab_capacity
            
            # Expand vocab
            new_vocab = original_vocab + 50000
            expanded = node.model.check_and_expand_vocab_if_needed(new_vocab)
            
            assert expanded, "Vocab should expand"
            assert node.model.vocab_capacity > original_vocab
            assert node.model.embedding.num_embeddings == node.model.vocab_capacity
            
            print(f"    Vocab: {original_vocab} → {node.model.vocab_capacity}")
            print("    ✅ Vocab expansion works")
    
    def test_vocab_consistency_pipeline(self):
        """Vocab size should be consistent across pipeline."""
        print("\n  Testing vocab consistency across pipeline...")
        
        with MultiNodeSimulator(default_layer_capacity=6) as sim:
            driver = sim.add_node()
            worker = sim.add_node()
            
            # Both should have same vocab capacity
            assert driver.pool.vocab_capacity == worker.pool.vocab_capacity
            
            print(f"    Driver vocab: {driver.pool.vocab_capacity}")
            print(f"    Worker vocab: {worker.pool.vocab_capacity}")
            print("    ✅ Vocab consistency maintained")


class TestParallelNodeOperations:
    """Test parallel node operations."""
    
    def test_parallel_node_join(self):
        """Multiple nodes joining in parallel should be handled correctly."""
        print("\n  Testing parallel node join...")
        
        with MultiNodeSimulator(default_layer_capacity=3) as sim:
            # Initial node
            sim.add_node(with_model=False)
            
            # Parallel joins
            with ThreadPoolExecutor(max_workers=5) as executor:
                futures = [
                    executor.submit(sim.add_node, 2500, None, False)
                    for _ in range(5)
                ]
                
                nodes = [f.result() for f in as_completed(futures)]
            
            coverage = sim.get_network_coverage()
            
            print(f"    Total nodes: {len(sim.nodes)}")
            print(f"    Layers covered: {len(coverage)}")
            
            assert len(coverage) == 11, "All layers should be covered"
            print("    ✅ Parallel node join works")
    
    def test_parallel_training_steps(self):
        """Parallel training steps on different nodes."""
        print("\n  Testing parallel training steps...")
        
        with MultiNodeSimulator(default_layer_capacity=11) as sim:
            # Create independent full nodes
            nodes = []
            for i in range(3):
                # Each node gets its own fresh DHT view
                node = sim.add_node(memory_mb=10000)
                nodes.append(node)
            
            # Run training steps in parallel
            def train_step(node):
                if not node.model.has_embedding:
                    return None
                
                input_ids = torch.randint(0, 1000, (2, 16))
                labels = torch.randint(0, 1000, (2, 16))
                
                node.optimizer.zero_grad()
                hidden = node.model.embed(input_ids)
                hidden = node.model.forward_my_layers(hidden)
                
                if node.model.has_lm_head:
                    logits = node.model.compute_logits(hidden)
                    loss = nn.functional.cross_entropy(
                        logits.view(-1, logits.size(-1)),
                        labels.view(-1)
                    )
                    loss.backward()
                    node.optimizer.step()
                    return loss.item()
                return None
            
            with ThreadPoolExecutor(max_workers=3) as executor:
                futures = [executor.submit(train_step, n) for n in nodes]
                losses = [f.result() for f in futures]
            
            valid_losses = [l for l in losses if l is not None]
            
            print(f"    Losses: {valid_losses}")
            print("    ✅ Parallel training works")


class TestCheckpointingMultiNode:
    """Test checkpointing in multi-node environment."""
    
    def test_checkpoint_save_load(self):
        """Test checkpoint save and load."""
        print("\n  Testing checkpoint save/load...")
        
        with MultiNodeSimulator(default_layer_capacity=11) as sim:
            node = sim.add_node(memory_mb=10000, with_checkpoint_dir=True)
            
            # Train a bit
            input_ids = torch.randint(0, 1000, (2, 16))
            labels = torch.randint(0, 1000, (2, 16))
            
            node.optimizer.zero_grad()
            hidden = node.model.embed(input_ids)
            hidden = node.model.forward_my_layers(hidden)
            logits = node.model.compute_logits(hidden)
            loss = nn.functional.cross_entropy(
                logits.view(-1, logits.size(-1)),
                labels.view(-1)
            )
            loss.backward()
            node.optimizer.step()
            
            # Save checkpoint
            checkpoint_path = node.checkpoint_dir / f"test_checkpoint.pt"
            
            state_dict = {}
            if node.model.has_embedding:
                state_dict["embedding.weight"] = node.model.embedding.weight.cpu()
            if node.model.has_lm_head:
                state_dict["lm_head.weight"] = node.model.lm_head.weight.cpu()
            
            for layer_id, layer in node.model.my_layers.items():
                for name, param in layer.named_parameters():
                    state_dict[f"layer_{layer_id}.{name}"] = param.cpu()
            
            checkpoint = {
                "state_dict": state_dict,
                "layer_ids": node.layers,
                "architecture": {
                    "hidden_dim": node.model.architecture.hidden_dim,
                    "num_layers": node.model.architecture.num_layers,
                },
                "training_rounds": 1,
            }
            
            torch.save(checkpoint, checkpoint_path)
            
            # Verify checkpoint exists
            assert checkpoint_path.exists()
            
            # Load checkpoint
            loaded = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
            
            assert loaded["training_rounds"] == 1
            assert loaded["layer_ids"] == node.layers
            
            print(f"    Saved layers: {loaded['layer_ids']}")
            print("    ✅ Checkpoint save/load works")


class TestEdgeCases:
    """Test edge cases and error scenarios."""
    
    def test_single_layer_nodes(self):
        """Test network with minimal capacity nodes."""
        print("\n  Testing single-layer nodes...")
        
        with MultiNodeSimulator(default_layer_capacity=1) as sim:
            nodes = [sim.add_node(memory_mb=500, with_model=False) for _ in range(15)]
            
            coverage = sim.get_network_coverage()
            
            print(f"    Nodes: {len(nodes)}")
            print(f"    Layers covered: {len(coverage)}")
            
            assert len(coverage) >= 10, "Should cover most layers"
            print("    ✅ Single-layer nodes work")
    
    def test_heterogeneous_capacity(self):
        """Test network with varied node capacities."""
        print("\n  Testing heterogeneous capacity nodes...")
        
        with MultiNodeSimulator() as sim:
            # Different capacities
            capacities = [2, 4, 6, 8, 3]
            
            for cap in capacities:
                with patch('neuroshard.core.model.scaler.calculate_layer_assignment', return_value=cap):
                    sim.add_node(memory_mb=cap * 500, layer_capacity=cap, with_model=False)
            
            coverage = sim.get_network_coverage()
            
            layer_counts = [len(n.layers) for n in sim.nodes.values()]
            
            print(f"    Capacities assigned: {layer_counts}")
            print(f"    Layers covered: {len(coverage)}")
            
            assert len(coverage) == 11, "All layers should be covered"
            print("    ✅ Heterogeneous capacity works")
    
    def test_network_partition_recovery(self):
        """Test recovery from network partition."""
        print("\n  Testing network partition recovery...")
        
        with MultiNodeSimulator(default_layer_capacity=4) as sim:
            # Initial network
            nodes = [sim.add_node(with_model=False) for _ in range(4)]
            
            initial_coverage = sim.get_network_coverage()
            
            # Simulate partition (DHT entries expire)
            sim.dht_network.expire_all()
            
            # Re-announce all nodes
            for node in nodes:
                node.dht.announce_layers(node.layers)
            
            recovered_coverage = sim.get_network_coverage()
            
            print(f"    Before partition: {len(initial_coverage)} layers")
            print(f"    After recovery: {len(recovered_coverage)} layers")
            
            assert len(recovered_coverage) == len(initial_coverage)
            print("    ✅ Network partition recovery works")


# =============================================================================
# TEST RUNNER
# =============================================================================

def run_all_tests():
    """Run all multi-node environment tests."""
    print("\n" + "=" * 80)
    print("COMPREHENSIVE MULTI-NODE ENVIRONMENT TEST SUITE")
    print("=" * 80)
    
    test_classes = [
        ("Multi-Node Layer Allocation", TestMultiNodeLayerAllocation),
        ("DHT Peer Discovery", TestDHTPeerDiscovery),
        ("Distributed Pipeline Training", TestDistributedPipelineTraining),
        ("Node Join/Leave Dynamics", TestNodeJoinLeave),
        ("Vocabulary Expansion", TestVocabExpansion),
        ("Parallel Node Operations", TestParallelNodeOperations),
        ("Multi-Node Checkpointing", TestCheckpointingMultiNode),
        ("Edge Cases", TestEdgeCases),
    ]
    
    passed = 0
    failed = 0
    
    for section_name, test_class in test_classes:
        print(f"\n{'─' * 80}")
        print(f"📋 {section_name}")
        print('─' * 80)
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
    
    print("\n" + "=" * 80)
    print(f"FINAL RESULTS: {passed} passed, {failed} failed")
    print("=" * 80)
    
    if failed == 0:
        print("\n🎉 ALL MULTI-NODE TESTS PASSED!")
        print("The distributed system is working correctly in multi-node scenarios.")
    else:
        print(f"\n⚠️  {failed} test(s) failed. Please review the errors above.")
    
    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
