
import unittest
import time
import shutil
import os
from neuroshard.core.economics.ledger import LedgerManager, ProofType
from neuroshard.core.network.dht import RoutingTable, Node, ID_BITS

class TestDecentralization(unittest.TestCase):
    def setUp(self):
        self.test_db_1 = "test_ledger_1.db"
        self.test_db_2 = "test_ledger_2.db"
        
    def tearDown(self):
        if os.path.exists(self.test_db_1):
            os.remove(self.test_db_1)
        if os.path.exists(self.test_db_2):
            os.remove(self.test_db_2)

    def test_ledger_exchange(self):
        """Verify two nodes can verify each other's proofs."""
        # Node 1
        l1 = LedgerManager(self.test_db_1, node_id="node1", node_token="secret1")
        # Node 2
        l2 = LedgerManager(self.test_db_2, node_id="node2", node_token="secret2")
        
        # Node 1 generates proof
        proof1 = l1.create_proof(proof_type=ProofType.UPTIME, uptime_seconds=60)
        self.assertTrue(proof1.signature != "")
        
        # Node 2 receives and verifies proof from Node 1
        # Note: In this simple PoC we verify signature by just checking it exists/hashed 
        # because we didn't implement public key distribution yet. 
        # In real system, L2 needs L1's public key.
        
        # Simulate L2 processing L1's proof
        success = l2.process_proof(proof1)
        self.assertTrue(success)
        
        # Check balance update
        bal = l2.get_balance("node1")
        self.assertAlmostEqual(bal, 10.0) # 10 credits per min
        
        # Replay attack check
        success_replay = l2.process_proof(proof1)
        self.assertFalse(success_replay)

    def test_dht_splitting(self):
        """Verify K-Bucket splits correctly."""
        local_node = Node(0, "localhost", 8000)
        rt = RoutingTable(local_node)
        
        # Add many nodes to trigger split
        # K=20. We add 30 nodes close to local_node (ID 0)
        # These should trigger a split of the first bucket (0-MAX)
        
        for i in range(1, 30):
            n = Node(i, "1.2.3.4", 9000+i)
            rt.add_contact(n)
            
        # We expect more than 1 bucket now
        self.assertTrue(len(rt.buckets) > 1)
        
        # Verify nodes are preserved
        total_nodes = sum(len(b.nodes) for b in rt.buckets)
        self.assertEqual(total_nodes, 29)

if __name__ == '__main__':
    unittest.main()

