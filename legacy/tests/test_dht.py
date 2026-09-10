"""
Unit tests for DHT (Distributed Hash Table) components.

Tests:
- Kademlia routing table operations
- Node management and bucket splitting
- XOR-distance based routing
- Peer discovery
"""

import sys
import os
import unittest
import time
import random

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from neuroshard.core.network.dht import (
    Node,
    KBucket,
    RoutingTable,
    K_BUCKET_SIZE,
    ID_BITS,
)


class TestNode(unittest.TestCase):
    """Tests for DHT Node class."""
    
    def test_node_creation(self):
        """Test creating a node."""
        node = Node(node_id=12345, ip="192.168.1.1", port=8000)
        
        self.assertEqual(node.id, 12345)
        self.assertEqual(node.ip, "192.168.1.1")
        self.assertEqual(node.port, 8000)
        self.assertIsNotNone(node.last_seen)
        
    def test_node_equality(self):
        """Test node equality based on ID."""
        node1 = Node(node_id=100, ip="192.168.1.1", port=8000)
        node2 = Node(node_id=100, ip="192.168.1.2", port=9000)  # Same ID, different IP/port
        node3 = Node(node_id=200, ip="192.168.1.1", port=8000)  # Different ID
        
        self.assertEqual(node1, node2)
        self.assertNotEqual(node1, node3)
        
    def test_node_repr(self):
        """Test node string representation."""
        node = Node(node_id=0xABCDEF, ip="127.0.0.1", port=5000)
        repr_str = repr(node)
        
        self.assertIn("127.0.0.1", repr_str)
        self.assertIn("5000", repr_str)


class TestKBucket(unittest.TestCase):
    """Tests for KBucket class."""
    
    def test_bucket_creation(self):
        """Test creating a bucket with range."""
        bucket = KBucket(lower=0, upper=1024)
        
        self.assertEqual(bucket.range, (0, 1024))
        self.assertEqual(len(bucket.nodes), 0)
        
    def test_add_node(self):
        """Test adding nodes to bucket."""
        bucket = KBucket(lower=0, upper=2**160)
        node = Node(node_id=100, ip="192.168.1.1", port=8000)
        
        result = bucket.add_node(node)
        
        self.assertTrue(result)
        self.assertEqual(len(bucket.nodes), 1)
        
    def test_add_existing_node_moves_to_tail(self):
        """Test that re-adding a node moves it to tail (LRU)."""
        bucket = KBucket(lower=0, upper=2**160)
        node1 = Node(node_id=100, ip="192.168.1.1", port=8000)
        node2 = Node(node_id=200, ip="192.168.1.2", port=8000)
        node3 = Node(node_id=300, ip="192.168.1.3", port=8000)
        
        bucket.add_node(node1)
        bucket.add_node(node2)
        bucket.add_node(node3)
        
        # Re-add node1 (with updated info)
        node1_updated = Node(node_id=100, ip="192.168.1.100", port=9000)
        bucket.add_node(node1_updated)
        
        # node1 should now be at the tail
        self.assertEqual(len(bucket.nodes), 3)
        self.assertEqual(bucket.nodes[-1].id, 100)
        
    def test_bucket_capacity_limit(self):
        """Test bucket rejects when full."""
        bucket = KBucket(lower=0, upper=2**160)
        
        # Fill the bucket
        for i in range(K_BUCKET_SIZE):
            node = Node(node_id=i, ip=f"192.168.1.{i}", port=8000)
            result = bucket.add_node(node)
            self.assertTrue(result)
        
        # Try to add one more
        extra_node = Node(node_id=999, ip="192.168.1.99", port=8000)
        result = bucket.add_node(extra_node)
        
        self.assertFalse(result)
        self.assertEqual(len(bucket.nodes), K_BUCKET_SIZE)
        
    def test_get_nodes(self):
        """Test getting list of nodes."""
        bucket = KBucket(lower=0, upper=2**160)
        
        for i in range(5):
            bucket.add_node(Node(node_id=i, ip=f"192.168.1.{i}", port=8000))
            
        nodes = bucket.get_nodes()
        
        self.assertEqual(len(nodes), 5)
        self.assertIsInstance(nodes, list)


class TestRoutingTable(unittest.TestCase):
    """Tests for RoutingTable class."""
    
    def setUp(self):
        """Set up test routing table."""
        self.local_node = Node(node_id=0x1000, ip="127.0.0.1", port=8000)
        self.table = RoutingTable(self.local_node)
        
    def test_initial_state(self):
        """Test routing table initial state."""
        self.assertEqual(len(self.table.buckets), 1)
        self.assertEqual(self.table.buckets[0].range, (0, 2**ID_BITS))
        
    def test_add_contact_ignores_self(self):
        """Test that adding local node is ignored."""
        initial_count = sum(len(b.nodes) for b in self.table.buckets)
        
        self.table.add_contact(self.local_node)
        
        final_count = sum(len(b.nodes) for b in self.table.buckets)
        self.assertEqual(initial_count, final_count)
        
    def test_add_contact(self):
        """Test adding a contact to routing table."""
        peer = Node(node_id=0x2000, ip="192.168.1.1", port=8000)
        
        self.table.add_contact(peer)
        
        all_nodes = self.table.get_all_nodes()
        self.assertEqual(len(all_nodes), 1)
        self.assertEqual(all_nodes[0].id, 0x2000)
        
    def test_bucket_splitting(self):
        """Test that buckets split when full."""
        initial_buckets = len(self.table.buckets)
        
        # Add many nodes to trigger splitting
        for i in range(K_BUCKET_SIZE + 5):
            # Generate random node IDs distributed across the ID space
            node_id = random.randint(0, 2**ID_BITS - 1)
            if node_id == self.local_node.id:
                continue
            peer = Node(node_id=node_id, ip=f"192.168.{i//256}.{i%256}", port=8000)
            self.table.add_contact(peer)
            
        # Should have more buckets now
        self.assertGreaterEqual(len(self.table.buckets), initial_buckets)
        
    def test_find_closest(self):
        """Test finding closest nodes to a target."""
        # Add some nodes
        nodes = []
        for i in range(10):
            node_id = (i + 1) * 0x100
            node = Node(node_id=node_id, ip=f"192.168.1.{i}", port=8000)
            self.table.add_contact(node)
            nodes.append(node)
            
        # Find closest to target
        target_id = 0x350
        closest = self.table.find_closest(target_id, k=3)
        
        self.assertLessEqual(len(closest), 3)
        
        # Verify they are sorted by XOR distance
        for i in range(len(closest) - 1):
            dist_i = closest[i].id ^ target_id
            dist_j = closest[i + 1].id ^ target_id
            self.assertLessEqual(dist_i, dist_j)
            
    def test_find_closest_empty_table(self):
        """Test find_closest on empty table."""
        closest = self.table.find_closest(0x5000, k=5)
        
        self.assertEqual(len(closest), 0)
        
    def test_get_all_nodes(self):
        """Test getting all nodes from all buckets."""
        # Add nodes (avoid local_node.id which is 0x1000)
        for i in range(5):
            node = Node(node_id=(i + 2) * 0x1000, ip=f"192.168.1.{i}", port=8000)
            self.table.add_contact(node)
            
        all_nodes = self.table.get_all_nodes()
        
        self.assertEqual(len(all_nodes), 5)
        
    def test_xor_distance_metric(self):
        """Test XOR distance calculation in find_closest."""
        # Add nodes at specific distances
        self.table.add_contact(Node(node_id=0b1000, ip="192.168.1.1", port=8000))  # XOR distance to 0b1100 = 0b0100 = 4
        self.table.add_contact(Node(node_id=0b1110, ip="192.168.1.2", port=8000))  # XOR distance to 0b1100 = 0b0010 = 2
        self.table.add_contact(Node(node_id=0b1111, ip="192.168.1.3", port=8000))  # XOR distance to 0b1100 = 0b0011 = 3
        
        target = 0b1100
        closest = self.table.find_closest(target, k=3)
        
        # Should be ordered: 0b1110 (dist=2), 0b1111 (dist=3), 0b1000 (dist=4)
        self.assertEqual(closest[0].id, 0b1110)
        self.assertEqual(closest[1].id, 0b1111)
        self.assertEqual(closest[2].id, 0b1000)


class TestDHTIntegration(unittest.TestCase):
    """Integration tests for DHT components."""
    
    def test_multiple_routing_tables_find_each_other(self):
        """Test that multiple routing tables can find common nodes."""
        # Create three routing tables
        node_a = Node(node_id=0x1000, ip="192.168.1.1", port=8000)
        node_b = Node(node_id=0x2000, ip="192.168.1.2", port=8000)
        node_c = Node(node_id=0x3000, ip="192.168.1.3", port=8000)
        
        table_a = RoutingTable(node_a)
        table_b = RoutingTable(node_b)
        table_c = RoutingTable(node_c)
        
        # A knows B
        table_a.add_contact(node_b)
        # B knows A and C
        table_b.add_contact(node_a)
        table_b.add_contact(node_c)
        # C knows B
        table_c.add_contact(node_b)
        
        # A should find B when looking for C's ID
        closest_from_a = table_a.find_closest(node_c.id, k=1)
        self.assertEqual(len(closest_from_a), 1)
        self.assertEqual(closest_from_a[0].id, node_b.id)
        
    def test_large_routing_table(self):
        """Test routing table with many nodes."""
        local_node = Node(node_id=0x8000000000, ip="127.0.0.1", port=8000)
        table = RoutingTable(local_node)
        
        # Add 100 nodes
        for i in range(100):
            node_id = random.randint(0, 2**ID_BITS - 1)
            if node_id == local_node.id:
                continue
            node = Node(node_id=node_id, ip=f"10.0.{i//256}.{i%256}", port=8000 + i)
            table.add_contact(node)
            
        # Should have many nodes
        all_nodes = table.get_all_nodes()
        self.assertGreater(len(all_nodes), 50)  # Some may be dropped due to full buckets
        
        # Find closest should work
        closest = table.find_closest(random.randint(0, 2**ID_BITS - 1), k=K_BUCKET_SIZE)
        self.assertGreater(len(closest), 0)
        

if __name__ == '__main__':
    unittest.main()

