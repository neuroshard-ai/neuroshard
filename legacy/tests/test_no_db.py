import unittest
import os
import shutil
from neuroshard.core.economics.ledger import LedgerManager, ProofOfWork

class TestDecentralizedArchitecture(unittest.TestCase):
    """Verify the system works without a central Postgres DB."""
    
    def setUp(self):
        self.db_path = "test_no_central_db.db"
        
    def tearDown(self):
        if os.path.exists(self.db_path):
            os.remove(self.db_path)
            
    def test_independent_ledger(self):
        """Ensure a node can track credits independently of any central server."""
        ledger = LedgerManager(self.db_path, node_id="test_node", node_token="secret")
        
        # Simulate receiving a proof from a peer
        proof = ProofOfWork(node_id="peer_node", timestamp=1234567890, uptime=60, signature="valid_sig")
        # Manually set signature to avoid hash check failure in this unit test which mocks the logic
        # In real impl, we'd sign it.
        # For this test, we assume the ledger accepts it if valid format (LedgerManager.verify_proof logic)
        
        # Actually, let's use the helper to generate a valid signed proof
        proof = ledger.create_proof(uptime_seconds=60)
        proof.node_id = "peer_node" # Pretend it's from someone else
        
        success = ledger.process_proof(proof)
        self.assertTrue(success, "Ledger should accept valid proof without central DB connection")
        
        balance = ledger.get_balance("peer_node")
        self.assertEqual(balance, 10.0, "Credits should be 10.0 after 1 minute of uptime")

if __name__ == '__main__':
    unittest.main()

