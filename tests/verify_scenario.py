import unittest
import threading
import time
import shutil
import tempfile
from unittest.mock import MagicMock, patch
import torch

from neuroshard.core.model.dynamic import DynamicLayerPool, LayerAssignment

class MockDHT:
    def __init__(self):
        self.storage = {}
        
    def lookup_value(self, key):
        return self.storage.get(key)
        
    def announce(self, key_str):
        # Simulate storing "node_id" or similar
        # For simplicity, we just mark it as existing
        pass

class TestObserverJetsonScenario(unittest.TestCase):
    def setUp(self):
        self.test_dir = tempfile.mkdtemp()
        
    def tearDown(self):
        shutil.rmtree(self.test_dir)

    def test_observer_does_not_block_driver_role(self):
        """
        Simulate the exact scenario:
        1. Observer node joins (no training) -> Should NOT take Layer 0
        2. Jetson node joins (training enabled) -> Should take Layer 0 (DRIVER)
        """
        print("\n=== Testing Observer + Jetson Scenario ===")
        
        # Shared DHT state simulation
        dht_storage = {}
        
        def mock_discover(layer_id):
            key_str = f"layer_{layer_id}"
            if key_str in dht_storage:
                return [dht_storage[key_str]]
            return []
            
        # --- 1. Start OBSERVER Node (No Training) ---
        print("\n[1] Starting Observer Node (no-training)...")
        observer_pool = DynamicLayerPool()
        # Mock DHT discovery
        observer_pool._discover_layer_holders_from_dht = MagicMock(side_effect=mock_discover)
        
        # Mock assigning layer to update shared "DHT"
        def observer_assign(layer_id, node_id, *args):
            # Only update local state in base class
            if layer_id not in observer_pool.layer_assignments:
                observer_pool.layer_assignments[layer_id] = []
            observer_pool.layer_assignments[layer_id].append(
                LayerAssignment(layer_id, node_id, "url", "grpc")
            )
            # Update shared DHT
            dht_storage[f"layer_{layer_id}"] = node_id
            
        observer_pool._assign_layer = MagicMock(side_effect=observer_assign)
        
        # Register Observer: 2GB RAM, Training=False
        observer_layers = observer_pool.register_node(
            node_id="observer_node",
            node_url="http://observer:8000",
            grpc_addr="observer:9000",
            available_memory_mb=2000,
            enable_training=False 
        )
        
        print(f"Observer Layers: {observer_layers}")
        
        # VERIFICATION 1: Observer should NOT have Layer 0
        self.assertNotIn(0, observer_layers, "Observer (no-training) incorrectly took Layer 0!")
        self.assertTrue(len(observer_layers) > 0, "Observer should have assigned some layers")
        self.assertEqual(observer_layers[0], 1, "Observer should start at Layer 1")
        
        # --- 2. Start JETSON Node (Training Enabled) ---
        print("\n[2] Starting Jetson Node (training enabled)...")
        jetson_pool = DynamicLayerPool()
        jetson_pool._discover_layer_holders_from_dht = MagicMock(side_effect=mock_discover)
        
        def jetson_assign(layer_id, node_id, *args):
            if layer_id not in jetson_pool.layer_assignments:
                jetson_pool.layer_assignments[layer_id] = []
            jetson_pool.layer_assignments[layer_id].append(
                LayerAssignment(layer_id, node_id, "url", "grpc")
            )
            dht_storage[f"layer_{layer_id}"] = node_id
            
        jetson_pool._assign_layer = MagicMock(side_effect=jetson_assign)
        
        # Register Jetson: 10GB RAM, Training=True
        # It should see Layer 0 is empty (because Observer skipped it)
        jetson_layers = jetson_pool.register_node(
            node_id="jetson_node",
            node_url="http://jetson:8000",
            grpc_addr="jetson:9000",
            available_memory_mb=10000,
            enable_training=True
        )
        
        print(f"Jetson Layers: {jetson_layers}")
        
        # VERIFICATION 2: Jetson MUST take Layer 0 (Driver)
        self.assertIn(0, jetson_layers, "Jetson should have taken Layer 0 (DRIVER role)")
        self.assertEqual(jetson_pool.embedding_holder, "jetson_node", "Jetson should be the embedding holder")
        
        print("\n✅ Test Passed: Observer skipped Layer 0, allowing Jetson to become Driver.")

if __name__ == '__main__':
    unittest.main()
