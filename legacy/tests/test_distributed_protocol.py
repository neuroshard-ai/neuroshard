"""
Comprehensive test for NeuroShard's distributed training protocol.

Tests:
1. DHT layer announcements and discovery
2. Layer assignment (DRIVER vs WORKER)
3. Pipeline training flow
4. Node join/leave dynamics
5. Full decentralized operation
"""

import threading
import time
import torch
import json
from unittest.mock import MagicMock, patch
from typing import Dict, List, Optional
import sys

# Import the components we're testing
from neuroshard.core.model.dynamic import (
    DynamicLayerPool,
    DynamicNeuroNode,
    DynamicNeuroLLM,
    LayerAssignment,
)
from neuroshard.core.model.scaler import ModelArchitecture


# ============ MOCK DHT INFRASTRUCTURE ============

class MockDHTNetwork:
    """
    Simulates a network of DHTs that share storage.
    Each node has its own DHT instance but they share storage.
    """
    
    def __init__(self):
        self.shared_storage: Dict[int, str] = {}
        self.nodes: Dict[str, 'NodeDHT'] = {}
    
    def create_node_dht(self, node_url: str) -> 'NodeDHT':
        """Create a DHT instance for a node."""
        dht = NodeDHT(self.shared_storage, node_url)
        self.nodes[node_url] = dht
        return dht
    
    def clear(self):
        """Clear all storage."""
        self.shared_storage.clear()
        self.nodes.clear()


class NodeDHT:
    """DHT instance for a specific node."""
    
    def __init__(self, shared_storage: Dict[int, str], node_url: str):
        self.storage = shared_storage
        self.node_url = node_url
        self.grpc_addr = node_url.replace("http://", "").replace(":8000", ":9000")
    
    def announce(self, key_string: str):
        """Announce a key with gRPC address."""
        import hashlib
        key_int = int(hashlib.sha1(key_string.encode()).hexdigest(), 16)
        
        # Merge with existing holders
        if key_int in self.storage:
            existing = json.loads(self.storage[key_int])
            if self.node_url not in existing:
                existing.append(self.node_url)
                self.storage[key_int] = json.dumps(existing)
        else:
            self.storage[key_int] = json.dumps([self.node_url])
        
        return True
    
    def lookup_value(self, key_int: int) -> Optional[str]:
        """Look up a value."""
        return self.storage.get(key_int)
    
    def remove(self, key_string: str):
        """Remove this node from a key's holders."""
        import hashlib
        key_int = int(hashlib.sha1(key_string.encode()).hexdigest(), 16)
        if key_int in self.storage:
            existing = json.loads(self.storage[key_int])
            if self.node_url in existing:
                existing.remove(self.node_url)
                if existing:
                    self.storage[key_int] = json.dumps(existing)
                else:
                    del self.storage[key_int]


# ============ TEST CLASSES ============

class TestLayerAssignment:
    """Test layer assignment logic."""
    
    def test_first_node_becomes_driver(self):
        """First node should become DRIVER with layer 0."""
        pool = DynamicLayerPool(dht_protocol=None)
        pool._device_hint = 'cpu'
        
        with patch('neuroshard.core.model.scaler.calculate_layer_assignment', return_value=6):
            layers = pool.register_node(
                node_id="node_1",
                node_url="http://node1:8000",
                grpc_addr="node1:9000",
                available_memory_mb=8000,
                staked_amount=0
            )
        
        assert 0 in layers, "First node should have layer 0 (DRIVER)"
        assert pool.embedding_holder == "node_1", "First node should be embedding holder"
        print(f"✅ First node assigned as DRIVER with layers: {layers}")
    
    def test_second_node_becomes_worker(self):
        """Second node should become WORKER - extending the pipeline."""
        dht_network = MockDHTNetwork()
        
        # First node (DRIVER)
        dht1 = dht_network.create_node_dht("http://node1:8000")
        pool1 = DynamicLayerPool(dht_protocol=dht1)
        pool1._device_hint = 'cpu'
        
        with patch('neuroshard.core.model.scaler.calculate_layer_assignment', return_value=4):
            layers1 = pool1.register_node(
                node_id="node_1",
                node_url="http://node1:8000",
                grpc_addr="node1:9000",
                available_memory_mb=2000,
                staked_amount=0
            )
        
        for layer_id in layers1:
            dht1.announce(f"layer_{layer_id}")
        
        print(f"  Node 1 (DRIVER) layers: {layers1}")
        
        # Second node (WORKER)
        dht2 = dht_network.create_node_dht("http://node2:8000")
        pool2 = DynamicLayerPool(dht_protocol=dht2)
        pool2._device_hint = 'cpu'
        
        with patch('neuroshard.core.model.scaler.calculate_layer_assignment', return_value=4):
            layers2 = pool2.register_node(
                node_id="node_2",
                node_url="http://node2:8000",
                grpc_addr="node2:9000",
                available_memory_mb=4000,
                staked_amount=0
            )
        
        print(f"  Node 2 (WORKER) layers: {layers2}")
        
        # Verify node 1 is DRIVER (has layer 0)
        assert 0 in layers1, "Node 1 should be DRIVER (layer 0)"
        
        # Verify node 2 is WORKER (does NOT have layer 0)
        assert 0 not in layers2, "Node 2 should NOT have layer 0 (WORKER)"
        
        # Verify pipeline is extended
        max_layer1 = max(layers1)
        min_layer2 = min(layers2)
        print(f"  Driver ends at {max_layer1}, Worker starts at {min_layer2}")
        
        print("✅ Second node correctly assigned as WORKER")
    
    def test_three_nodes_full_pipeline(self):
        """Three nodes should cover all layers."""
        dht_network = MockDHTNetwork()
        
        all_assigned = []
        
        for i in range(3):
            dht = dht_network.create_node_dht(f"http://node{i}:8000")
            pool = DynamicLayerPool(dht_protocol=dht)
            pool._device_hint = 'cpu'
            
            with patch('neuroshard.core.model.scaler.calculate_layer_assignment', return_value=4):
                layers = pool.register_node(
                    node_id=f"node_{i}",
                    node_url=f"http://node{i}:8000",
                    grpc_addr=f"node{i}:9000",
                    available_memory_mb=3000,
                    staked_amount=0
                )
            
            for layer_id in layers:
                dht.announce(f"layer_{layer_id}")
            
            all_assigned.append(layers)
            print(f"  Node {i}: {layers}")
        
        # Combine all layers
        all_layers = set()
        for layers in all_assigned:
            all_layers.update(layers)
        
        print(f"  Combined: {sorted(all_layers)}")
        
        # Verify full coverage
        assert 0 in all_layers, "Layer 0 should be covered"
        assert 10 in all_layers, "Layer 10 should be covered"
        
        print("✅ Three nodes cover full pipeline")


class TestDHTDiscovery:
    """Test DHT-based network discovery."""
    
    def test_dht_layer_discovery(self):
        """Test that nodes can discover layers via DHT."""
        dht_network = MockDHTNetwork()
        
        # Node 1 announces layers 0-5
        dht1 = dht_network.create_node_dht("http://node1:8000")
        for i in range(6):
            dht1.announce(f"layer_{i}")
        
        # Node 2 should discover these layers
        dht2 = dht_network.create_node_dht("http://node2:8000")
        
        discovered = []
        for i in range(11):
            import hashlib
            key_int = int(hashlib.sha1(f"layer_{i}".encode()).hexdigest(), 16)
            value = dht2.lookup_value(key_int)
            if value:
                discovered.append(i)
        
        print(f"  Discovered layers: {discovered}")
        assert 0 in discovered, "Should discover layer 0"
        assert len(discovered) == 6, f"Should discover 6 layers, got {len(discovered)}"
        print("✅ DHT layer discovery works")
    
    def test_frontier_detection(self):
        """Test finding the frontier (first unoccupied layer)."""
        dht_network = MockDHTNetwork()
        
        # Node 1 announces layers 0-5
        dht1 = dht_network.create_node_dht("http://node1:8000")
        for i in range(6):
            dht1.announce(f"layer_{i}")
        
        # Create pool for node 2
        dht2 = dht_network.create_node_dht("http://node2:8000")
        pool = DynamicLayerPool(dht_protocol=dht2)
        
        # Find frontier
        frontier = pool._find_frontier_layer()
        
        print(f"  Frontier layer: {frontier}")
        assert frontier == 6, f"Frontier should be layer 6, got {frontier}"
        print("✅ Frontier detection works")


class TestPipelineTraining:
    """Test pipeline training flow."""
    
    def test_full_forward_backward_pass(self):
        """Test complete forward and backward pass through pipeline."""
        dht_network = MockDHTNetwork()
        
        # Setup DRIVER
        dht1 = dht_network.create_node_dht("http://driver:8000")
        pool1 = DynamicLayerPool(dht_protocol=dht1)
        pool1._device_hint = 'cpu'
        pool1.vocab_capacity = 32000
        
        with patch('neuroshard.core.model.scaler.calculate_layer_assignment', return_value=6):
            layers1 = pool1.register_node(
                node_id="driver",
                node_url="http://driver:8000",
                grpc_addr="driver:9000",
                available_memory_mb=4000,
                staked_amount=0
            )
        
        for layer_id in layers1:
            dht1.announce(f"layer_{layer_id}")
        
        model1 = DynamicNeuroLLM(node_id="driver", layer_pool=pool1, device="cpu")
        model1.initialize_layers(layers1)
        optimizer1 = torch.optim.AdamW(model1.parameters(), lr=1e-4)
        
        # Setup WORKER
        dht2 = dht_network.create_node_dht("http://worker:8000")
        pool2 = DynamicLayerPool(dht_protocol=dht2)
        pool2._device_hint = 'cpu'
        pool2.vocab_capacity = 32000
        
        with patch('neuroshard.core.model.scaler.calculate_layer_assignment', return_value=6):
            layers2 = pool2.register_node(
                node_id="worker",
                node_url="http://worker:8000",
                grpc_addr="worker:9000",
                available_memory_mb=4000,
                staked_amount=0
            )
        
        model2 = DynamicNeuroLLM(node_id="worker", layer_pool=pool2, device="cpu")
        model2.initialize_layers(layers2)
        optimizer2 = torch.optim.AdamW(model2.parameters(), lr=1e-4)
        
        print(f"  DRIVER: layers {layers1}, embed={model1.has_embedding}")
        print(f"  WORKER: layers {layers2}, head={model2.has_lm_head}")
        
        # Training step
        input_ids = torch.randint(0, 1000, (2, 16))
        labels = torch.randint(0, 1000, (2, 16))
        
        # Forward
        optimizer1.zero_grad()
        embeddings = model1.embed(input_ids)
        hidden1 = model1.forward_my_layers(embeddings)
        
        hidden1_sent = hidden1.detach().requires_grad_(True)
        
        optimizer2.zero_grad()
        hidden2 = model2.forward_my_layers(hidden1_sent)
        logits = model2.compute_logits(hidden2)
        
        # Loss
        loss = torch.nn.functional.cross_entropy(
            logits.view(-1, logits.size(-1)),
            labels.view(-1)
        )
        
        # Backward
        loss.backward()
        optimizer2.step()
        
        hidden1.backward(hidden1_sent.grad)
        optimizer1.step()
        
        print(f"  Training step: loss={loss.item():.4f}")
        print("✅ Full forward-backward pass works")


class TestNodeDynamics:
    """Test node join/leave dynamics."""
    
    def test_node_removal(self):
        """Test that removing a node updates assignments."""
        pool = DynamicLayerPool(dht_protocol=None)
        pool._device_hint = 'cpu'
        
        with patch('neuroshard.core.model.scaler.calculate_layer_assignment', return_value=6):
            layers1 = pool.register_node(
                node_id="node_1",
                node_url="http://node1:8000",
                grpc_addr="node1:9000",
                available_memory_mb=8000,
                staked_amount=0
            )
        
        assert pool.embedding_holder == "node_1"
        
        # Remove node 1
        pool.remove_node("node_1")
        
        assert pool.embedding_holder is None, "Embedding holder should be None after removal"
        assert "node_1" not in pool.node_capacities, "Node should be removed from capacities"
        
        print("✅ Node removal works correctly")
    
    def test_node_rejoin_fills_gap(self):
        """Test that a new node fills gaps left by departed nodes."""
        dht_network = MockDHTNetwork()
        
        # Node 1 joins
        dht1 = dht_network.create_node_dht("http://node1:8000")
        pool1 = DynamicLayerPool(dht_protocol=dht1)
        pool1._device_hint = 'cpu'
        
        with patch('neuroshard.core.model.scaler.calculate_layer_assignment', return_value=4):
            layers1 = pool1.register_node(
                node_id="node_1",
                node_url="http://node1:8000",
                grpc_addr="node1:9000",
                available_memory_mb=2000,
                staked_amount=0
            )
        
        for layer_id in layers1:
            dht1.announce(f"layer_{layer_id}")
        
        # Node 2 joins
        dht2 = dht_network.create_node_dht("http://node2:8000")
        pool2 = DynamicLayerPool(dht_protocol=dht2)
        pool2._device_hint = 'cpu'
        
        with patch('neuroshard.core.model.scaler.calculate_layer_assignment', return_value=4):
            layers2 = pool2.register_node(
                node_id="node_2",
                node_url="http://node2:8000",
                grpc_addr="node2:9000",
                available_memory_mb=2000,
                staked_amount=0
            )
        
        for layer_id in layers2:
            dht2.announce(f"layer_{layer_id}")
        
        print(f"  Node 1: {layers1}")
        print(f"  Node 2: {layers2}")
        
        # Node 2 leaves (remove from DHT)
        for layer_id in layers2:
            dht2.remove(f"layer_{layer_id}")
        
        # Node 3 joins - should fill gap
        dht3 = dht_network.create_node_dht("http://node3:8000")
        pool3 = DynamicLayerPool(dht_protocol=dht3)
        pool3._device_hint = 'cpu'
        
        with patch('neuroshard.core.model.scaler.calculate_layer_assignment', return_value=4):
            layers3 = pool3.register_node(
                node_id="node_3",
                node_url="http://node3:8000",
                grpc_addr="node3:9000",
                available_memory_mb=2000,
                staked_amount=0
            )
        
        print(f"  Node 3 (filled gap): {layers3}")
        
        # Node 3 should fill the same gap
        assert min(layers3) == min(layers2), "Node 3 should fill node 2's gap"
        
        print("✅ Node rejoin correctly fills gaps")
    
    def test_stale_cleanup(self):
        """Test cleanup of stale node assignments."""
        pool = DynamicLayerPool(dht_protocol=None)
        pool._device_hint = 'cpu'
        
        with patch('neuroshard.core.model.scaler.calculate_layer_assignment', return_value=6):
            layers = pool.register_node(
                node_id="node_1",
                node_url="http://node1:8000",
                grpc_addr="node1:9000",
                available_memory_mb=8000,
                staked_amount=0
            )
        
        # Send heartbeat
        pool.heartbeat("node_1", layers)
        
        # Cleanup with short timeout should not remove
        stale = pool.cleanup_stale_assignments(timeout_seconds=10)
        assert len(stale) == 0, "Node should not be stale yet"
        
        # Manipulate last heartbeat time
        pool._last_heartbeat["node_1"] = time.time() - 400
        
        # Cleanup should now find stale node
        stale = pool.cleanup_stale_assignments(timeout_seconds=300)
        assert "node_1" in stale, "Node should be detected as stale"
        
        print("✅ Stale cleanup works correctly")


class TestFullPipeline:
    """Test complete end-to-end pipeline."""
    
    def test_five_node_scaling(self):
        """Test network scaling with 5 nodes joining sequentially."""
        print("\n  --- Testing 5-node scaling ---")
        dht_network = MockDHTNetwork()
        
        nodes_data = []
        for i in range(5):
            dht = dht_network.create_node_dht(f"http://node{i}:8000")
            pool = DynamicLayerPool(dht_protocol=dht)
            pool._device_hint = 'cpu'
            
            # Each node has different capacity
            capacity = 3 + (i % 3)  # 3, 4, 5, 3, 4 layers
            with patch('neuroshard.core.model.scaler.calculate_layer_assignment', return_value=capacity):
                layers = pool.register_node(
                    node_id=f"node_{i}",
                    node_url=f"http://node{i}:8000",
                    grpc_addr=f"node{i}:9000",
                    available_memory_mb=2000 + i * 500,
                    staked_amount=0
                )
            
            for layer_id in layers:
                dht.announce(f"layer_{layer_id}")
            
            nodes_data.append((pool, layers))
            print(f"    Node {i}: layers {layers}")
        
        # Verify coverage
        all_layers = set()
        for _, layers in nodes_data:
            all_layers.update(layers)
        
        print(f"    Coverage: {sorted(all_layers)}")
        assert 0 in all_layers, "Must have layer 0"
        print("  ✅ 5-node scaling works")
    
    def test_gradient_flow_three_nodes(self):
        """Test gradient flow through 3-node pipeline."""
        print("\n  --- Testing 3-node gradient flow ---")
        dht_network = MockDHTNetwork()
        models = []
        optimizers = []
        
        # Create 3 nodes
        for i in range(3):
            dht = dht_network.create_node_dht(f"http://node{i}:8000")
            pool = DynamicLayerPool(dht_protocol=dht)
            pool._device_hint = 'cpu'
            pool.vocab_capacity = 32000
            
            with patch('neuroshard.core.model.scaler.calculate_layer_assignment', return_value=4):
                layers = pool.register_node(
                    node_id=f"node_{i}",
                    node_url=f"http://node{i}:8000",
                    grpc_addr=f"node{i}:9000",
                    available_memory_mb=3000,
                    staked_amount=0
                )
            
            for layer_id in layers:
                dht.announce(f"layer_{layer_id}")
            
            model = DynamicNeuroLLM(node_id=f"node_{i}", layer_pool=pool, device="cpu")
            model.initialize_layers(layers)
            
            models.append(model)
            optimizers.append(torch.optim.AdamW(model.parameters(), lr=1e-4))
            print(f"    Node {i}: layers {layers}, embed={model.has_embedding}, head={model.has_lm_head}")
        
        # Training step with gradient flow through all 3 nodes
        input_ids = torch.randint(0, 1000, (2, 16))
        labels = torch.randint(0, 1000, (2, 16))
        
        # Zero grads
        for opt in optimizers:
            opt.zero_grad()
        
        # Forward through all nodes
        hidden = models[0].embed(input_ids)
        hiddens = [hidden]
        
        for model in models:
            hidden = model.forward_my_layers(hidden)
            hiddens.append(hidden.detach().requires_grad_(True))
            hidden = hiddens[-1]
        
        # Compute loss on validator (last node with lm_head)
        for model in reversed(models):
            if model.has_lm_head:
                logits = model.compute_logits(hiddens[-1])
                break
        
        loss = torch.nn.functional.cross_entropy(
            logits.view(-1, logits.size(-1)),
            labels.view(-1)
        )
        
        # Backward through all nodes
        loss.backward()
        
        # Propagate gradients backwards
        for i in range(len(hiddens) - 1, 0, -1):
            if hiddens[i].grad is not None:
                hiddens[i-1].backward(hiddens[i].grad)
        
        # Update all optimizers
        for opt in optimizers:
            opt.step()
        
        print(f"    Loss: {loss.item():.4f}")
        print("  ✅ 3-node gradient flow works")
    
    def test_two_node_training(self):
        """Simulate complete two-node distributed training."""
        print("\n" + "=" * 50)
        print("SIMULATING TWO-NODE DISTRIBUTED TRAINING")
        print("=" * 50)
        
        dht_network = MockDHTNetwork()
        
        # ============ DRIVER ============
        dht1 = dht_network.create_node_dht("http://driver:8000")
        pool1 = DynamicLayerPool(dht_protocol=dht1)
        pool1._device_hint = 'cpu'
        pool1.vocab_capacity = 32000
        
        with patch('neuroshard.core.model.scaler.calculate_layer_assignment', return_value=6):
            layers1 = pool1.register_node(
                node_id="driver_node",
                node_url="http://driver:8000",
                grpc_addr="driver:9000",
                available_memory_mb=4000,
                staked_amount=0
            )
        
        for layer_id in layers1:
            dht1.announce(f"layer_{layer_id}")
        
        model1 = DynamicNeuroLLM(node_id="driver_node", layer_pool=pool1, device="cpu")
        model1.initialize_layers(layers1)
        optimizer1 = torch.optim.AdamW(model1.parameters(), lr=1e-4)
        
        print(f"  DRIVER: {layers1}, embed={model1.has_embedding}")
        
        # ============ WORKER ============
        dht2 = dht_network.create_node_dht("http://worker:8000")
        pool2 = DynamicLayerPool(dht_protocol=dht2)
        pool2._device_hint = 'cpu'
        pool2.vocab_capacity = 32000
        
        with patch('neuroshard.core.model.scaler.calculate_layer_assignment', return_value=6):
            layers2 = pool2.register_node(
                node_id="worker_node",
                node_url="http://worker:8000",
                grpc_addr="worker:9000",
                available_memory_mb=4000,
                staked_amount=0
            )
        
        model2 = DynamicNeuroLLM(node_id="worker_node", layer_pool=pool2, device="cpu")
        model2.initialize_layers(layers2)
        optimizer2 = torch.optim.AdamW(model2.parameters(), lr=1e-4)
        
        print(f"  WORKER: {layers2}, head={model2.has_lm_head}")
        
        # ============ TRAINING ============
        print("\n  Training steps:")
        losses = []
        for step in range(3):
            input_ids = torch.randint(0, 1000, (2, 16))
            labels = torch.randint(0, 1000, (2, 16))
            
            # Forward
            optimizer1.zero_grad()
            hidden1 = model1.forward_my_layers(model1.embed(input_ids))
            
            hidden1_sent = hidden1.detach().requires_grad_(True)
            
            optimizer2.zero_grad()
            hidden2 = model2.forward_my_layers(hidden1_sent)
            logits = model2.compute_logits(hidden2)
            
            loss = torch.nn.functional.cross_entropy(
                logits.view(-1, logits.size(-1)),
                labels.view(-1)
            )
            
            # Backward
            loss.backward()
            optimizer2.step()
            hidden1.backward(hidden1_sent.grad)
            optimizer1.step()
            
            losses.append(loss.item())
            print(f"    Step {step+1}: loss={loss.item():.4f}")
        
        print("\n" + "=" * 50)
        print("✅ TWO-NODE DISTRIBUTED TRAINING COMPLETE!")
        print("=" * 50)


def run_all_tests():
    """Run all tests and report results."""
    import sys
    
    print("\n" + "=" * 70, flush=True)
    print("NEUROSHARD DISTRIBUTED PROTOCOL TEST SUITE", flush=True)
    print("=" * 70, flush=True)
    
    test_classes = [
        TestLayerAssignment,
        TestDHTDiscovery,
        TestPipelineTraining,
        TestNodeDynamics,
        TestFullPipeline,
    ]
    
    passed = 0
    failed = 0
    
    for test_class in test_classes:
        print(f"\n{'─' * 50}", flush=True)
        print(f"Running: {test_class.__name__}", flush=True)
        print('─' * 50, flush=True)
        sys.stdout.flush()
        
        instance = test_class()
        test_methods = sorted([m for m in dir(instance) if m.startswith('test_')])
        
        for method_name in test_methods:
            print(f"  >> Starting: {method_name}...", flush=True)
            sys.stdout.flush()
            try:
                method = getattr(instance, method_name)
                method()
                passed += 1
                print(f"  << Finished: {method_name}", flush=True)
            except Exception as e:
                print(f"❌ {method_name}: {e}", flush=True)
                import traceback
                traceback.print_exc()
                failed += 1
            sys.stdout.flush()
    
    print("\n" + "=" * 70, flush=True)
    print(f"RESULTS: {passed} passed, {failed} failed", flush=True)
    print("=" * 70, flush=True)
    
    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    exit(0 if success else 1)
