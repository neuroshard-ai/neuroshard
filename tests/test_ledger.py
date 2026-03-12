"""
Unit tests for NEURO Ledger and Economics System.

Tests:
- Proof of Neural Work (PoNW) creation and verification
- ECDSA signature verification
- Reward calculation with multipliers
- Staking and stake multipliers (diminishing returns)
- Transfers and fee burn
- Slashing mechanisms
- Validator eligibility
"""

import sys
import os
import unittest
import time
import tempfile
import hashlib

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from neuroshard.core.economics.ledger import (
    NEUROLedger,
    PoNWProof,
    ProofType,
    Transaction,
    LedgerStats,
    sign_proof,
)
from neuroshard.core.economics.constants import (
    UPTIME_REWARD_PER_MINUTE,
    TRAINING_REWARD_PER_BATCH_PER_LAYER,
    INITIATOR_BONUS,
    FINISHER_BONUS,
    VALIDATOR_MIN_STAKE,
    SLASH_AMOUNT,
    FEE_BURN_RATE,
    calculate_stake_multiplier,
)


class TestPoNWProof(unittest.TestCase):
    """Tests for PoNW proof creation and serialization."""
    
    def test_proof_creation(self):
        """Test creating a PoNW proof."""
        proof = PoNWProof(
            node_id="abc123def456789012345678901234",
            proof_type="uptime",
            timestamp=time.time(),
            nonce="randomnonce",
            uptime_seconds=60.0,
            tokens_processed=1000,
            training_batches=5,
            layers_held=10,
        )
        
        self.assertEqual(proof.proof_type, "uptime")
        self.assertEqual(proof.uptime_seconds, 60.0)
        self.assertEqual(proof.tokens_processed, 1000)
        self.assertEqual(proof.layers_held, 10)
        
    def test_canonical_payload(self):
        """Test that canonical payload is deterministic."""
        proof = PoNWProof(
            node_id="abc123def456789012345678901234",
            proof_type="training",
            timestamp=1234567890.123456,
            nonce="testnonce",
            uptime_seconds=120.0,
            tokens_processed=5000,
            training_batches=10,
            data_samples=0,
            model_hash="",
            layers_held=5,
        )
        
        payload1 = proof.canonical_payload()
        payload2 = proof.canonical_payload()
        
        self.assertEqual(payload1, payload2)
        self.assertIn("abc123def456789012345678901234", payload1)
        self.assertIn("training", payload1)
        
    def test_proof_to_dict(self):
        """Test proof serialization to dict."""
        proof = PoNWProof(
            node_id="testnode12345678901234567890123",
            proof_type="inference",
            timestamp=time.time(),
            nonce="abc",
            tokens_processed=100,
        )
        
        data = proof.to_dict()
        
        self.assertIsInstance(data, dict)
        self.assertEqual(data["node_id"], "testnode12345678901234567890123")
        self.assertEqual(data["proof_type"], "inference")
        
    def test_proof_from_dict(self):
        """Test proof deserialization from dict."""
        data = {
            "node_id": "testnode12345678901234567890123",
            "proof_type": "uptime",
            "timestamp": 1234567890.0,
            "nonce": "xyz",
            "uptime_seconds": 60.0,
            "tokens_processed": 0,
            "training_batches": 0,
            "data_samples": 0,
            "model_hash": "",
            "layers_held": 5,
            "has_embedding": True,
            "has_lm_head": False,
            "signature": "somesignature",
        }
        
        proof = PoNWProof.from_dict(data)
        
        self.assertEqual(proof.node_id, "testnode12345678901234567890123")
        self.assertEqual(proof.uptime_seconds, 60.0)
        self.assertTrue(proof.has_embedding)


class TestNEUROLedger(unittest.TestCase):
    """Tests for NEUROLedger class."""
    
    def setUp(self):
        """Set up test ledger with temporary database."""
        self.temp_dir = tempfile.mkdtemp()
        self.db_path = os.path.join(self.temp_dir, "test_ledger.db")
        self.node_token = "test_token_for_ledger_tests_12345"
        self.ledger = NEUROLedger(
            db_path=self.db_path,
            node_token=self.node_token
        )
        
    def tearDown(self):
        """Clean up temporary database."""
        import shutil
        try:
            shutil.rmtree(self.temp_dir)
        except:
            pass
            
    def test_ledger_initialization(self):
        """Test ledger initializes correctly."""
        self.assertIsNotNone(self.ledger.node_id)
        self.assertEqual(len(self.ledger.node_id), 32)  # 32 hex chars
        
    def test_genesis_block(self):
        """Test ledger starts with zero supply (genesis block)."""
        stats = self.ledger.get_global_stats()
        
        self.assertEqual(stats.total_minted, 0.0)
        self.assertEqual(stats.total_burned, 0.0)
        
    def test_create_proof(self):
        """Test creating a signed proof."""
        proof = self.ledger.create_proof(
            proof_type=ProofType.UPTIME,
            uptime_seconds=60.0,
            tokens_processed=0,
            layers_held=5,
        )
        
        self.assertEqual(proof.node_id, self.ledger.node_id)
        self.assertIsNotNone(proof.signature)
        self.assertNotEqual(proof.signature, "")
        
    def test_verify_own_proof(self):
        """Test verifying our own proof."""
        proof = self.ledger.create_proof(
            proof_type=ProofType.UPTIME,
            uptime_seconds=60.0,
            layers_held=3,
        )
        
        is_valid, reason = self.ledger.verify_proof(proof)
        
        self.assertTrue(is_valid, f"Verification failed: {reason}")
        
    def test_verify_proof_detects_tampering(self):
        """Test that tampered proof is rejected."""
        proof = self.ledger.create_proof(
            proof_type=ProofType.UPTIME,
            uptime_seconds=60.0,
            layers_held=3,
        )
        
        # Tamper with the proof
        proof.uptime_seconds = 3600.0  # Changed from 60
        
        is_valid, reason = self.ledger.verify_proof(proof)
        
        self.assertFalse(is_valid)
        self.assertIn("signature", reason.lower())
        
    def test_verify_proof_rejects_stale(self):
        """Test that old proofs are rejected."""
        proof = self.ledger.create_proof(
            proof_type=ProofType.UPTIME,
            uptime_seconds=60.0,
        )
        
        # Make the proof old
        proof.timestamp = time.time() - 400  # 6+ minutes old
        
        # Re-sign with old timestamp
        proof = sign_proof(proof, self.node_token)
        
        is_valid, reason = self.ledger.verify_proof(proof)
        
        self.assertFalse(is_valid)
        self.assertIn("old", reason.lower())
        
    def test_verify_proof_rejects_replay(self):
        """Test that replay attacks are prevented."""
        proof = self.ledger.create_proof(
            proof_type=ProofType.UPTIME,
            uptime_seconds=60.0,
        )
        
        # Process once
        success1, reward1, msg1 = self.ledger.process_proof(proof)
        self.assertTrue(success1)
        
        # Try to replay
        success2, reward2, msg2 = self.ledger.process_proof(proof)
        
        self.assertFalse(success2)
        self.assertIn("duplicate", msg2.lower())
        
    def test_process_proof_credits_reward(self):
        """Test that processing a proof credits the reward."""
        initial_balance = self.ledger.get_balance()
        
        proof = self.ledger.create_proof(
            proof_type=ProofType.UPTIME,
            uptime_seconds=60.0,
            layers_held=5,
        )
        
        success, reward, msg = self.ledger.process_proof(proof)
        
        self.assertTrue(success)
        self.assertGreater(reward, 0)
        
        final_balance = self.ledger.get_balance()
        self.assertEqual(final_balance, initial_balance + reward)
        
    def test_reward_calculation_uptime(self):
        """Test uptime reward calculation."""
        proof = self.ledger.create_proof(
            proof_type=ProofType.UPTIME,
            uptime_seconds=60.0,
            layers_held=0,
        )
        
        success, reward, _ = self.ledger.process_proof(proof)
        
        self.assertTrue(success)
        # Uptime reward should be approximately UPTIME_REWARD_PER_MINUTE for 60s
        expected = UPTIME_REWARD_PER_MINUTE
        self.assertAlmostEqual(reward, expected, delta=0.001)
        
    def test_reward_calculation_training(self):
        """Test training reward calculation with per-layer rewards."""
        # Test the reward calculation directly (bypass verification)
        # Full verification requires model interface which isn't available in unit tests
        proof = self.ledger.create_proof(
            proof_type=ProofType.TRAINING,
            uptime_seconds=60.0,
            training_batches=10,
            layers_held=5,
            model_hash="test_hash_12345",  # Required for training proofs
        )
        
        # Test reward calculation directly
        reward = self.ledger._calculate_reward(proof)
        
        # Training reward should be per-layer: 10 batches × 0.0005 × 5 layers = 0.025 base
        # Plus uptime: 60s / 60 × 0.0001 = 0.0001
        # With training bonus (1.1x via role_bonus additive): (0.025 + 0.0001) × 1.1
        # Expected: ~0.02761 NEURO (before reputation bonus of 1.1x)
        self.assertGreater(reward, UPTIME_REWARD_PER_MINUTE)  # More than just uptime
        self.assertGreater(reward, 0.02)  # Should have substantial training reward
        
    def test_driver_bonus(self):
        """Test Driver role bonus (has_embedding=True)."""
        # Without driver bonus
        proof1 = self.ledger.create_proof(
            proof_type=ProofType.UPTIME,
            uptime_seconds=60.0,
            has_embedding=False,
            has_lm_head=False,
            layers_held=5,
        )
        success1, reward1, _ = self.ledger.process_proof(proof1)
        
        # Create new ledger to avoid replay detection
        ledger2 = NEUROLedger(
            db_path=os.path.join(self.temp_dir, "ledger2.db"),
            node_token=self.node_token
        )
        
        # With driver bonus
        proof2 = ledger2.create_proof(
            proof_type=ProofType.UPTIME,
            uptime_seconds=60.0,
            has_embedding=True,
            has_lm_head=False,
            layers_held=5,
        )
        success2, reward2, _ = ledger2.process_proof(proof2)
        
        self.assertTrue(success1)
        self.assertTrue(success2)
        self.assertGreater(reward2, reward1)
        
    def test_validator_bonus(self):
        """Test Validator role bonus (has_lm_head=True)."""
        # Without validator bonus
        proof1 = self.ledger.create_proof(
            proof_type=ProofType.UPTIME,
            uptime_seconds=60.0,
            has_embedding=False,
            has_lm_head=False,
            layers_held=5,
        )
        success1, reward1, _ = self.ledger.process_proof(proof1)
        
        # Create new ledger
        ledger2 = NEUROLedger(
            db_path=os.path.join(self.temp_dir, "ledger3.db"),
            node_token=self.node_token
        )
        
        # With validator bonus
        proof2 = ledger2.create_proof(
            proof_type=ProofType.UPTIME,
            uptime_seconds=60.0,
            has_embedding=False,
            has_lm_head=True,
            layers_held=5,
        )
        success2, reward2, _ = ledger2.process_proof(proof2)
        
        self.assertTrue(success1)
        self.assertTrue(success2)
        self.assertGreater(reward2, reward1)


class TestStaking(unittest.TestCase):
    """Tests for staking functionality."""
    
    def setUp(self):
        """Set up test ledger."""
        self.temp_dir = tempfile.mkdtemp()
        self.db_path = os.path.join(self.temp_dir, "stake_test.db")
        self.node_token = "stake_test_token_12345678901234"
        self.ledger = NEUROLedger(
            db_path=self.db_path,
            node_token=self.node_token
        )
        
        # Give ourselves some balance to stake
        self._credit_balance(1000.0)
        
    def _credit_balance(self, amount: float):
        """Helper to credit balance for testing."""
        import sqlite3
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT INTO balances (node_id, balance, total_earned, created_at)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(node_id) DO UPDATE SET balance = balance + ?
            """, (self.ledger.node_id, amount, amount, time.time(), amount))
            
    def tearDown(self):
        """Clean up."""
        import shutil
        try:
            shutil.rmtree(self.temp_dir)
        except:
            pass
            
    def test_stake(self):
        """Test staking NEURO."""
        initial_balance = self.ledger.get_balance()
        
        success, msg = self.ledger.stake(100.0, duration_days=30)
        
        self.assertTrue(success)
        
        final_balance = self.ledger.get_balance()
        self.assertEqual(final_balance, initial_balance - 100.0)
        
    def test_stake_insufficient_balance(self):
        """Test staking more than balance fails."""
        success, msg = self.ledger.stake(10000.0, duration_days=30)
        
        self.assertFalse(success)
        self.assertIn("insufficient", msg.lower())
        
    def test_stake_multiplier_diminishing_returns(self):
        """Test that stake multiplier has diminishing returns."""
        # Small stake
        mult_1k = calculate_stake_multiplier(1000)
        
        # Large stake
        mult_100k = calculate_stake_multiplier(100000)
        
        # 100x more stake should NOT give 100x more multiplier
        ratio = mult_100k / mult_1k
        self.assertLess(ratio, 3.0)  # Much less than 100x
        
    def test_stake_multiplier_values(self):
        """Test specific stake multiplier values."""
        # 0 stake = 1.0 multiplier
        self.assertEqual(calculate_stake_multiplier(0), 1.0)
        
        # 1000 stake should give ~1.1x
        mult_1k = calculate_stake_multiplier(1000)
        self.assertGreater(mult_1k, 1.09)
        self.assertLess(mult_1k, 1.15)
        
        # Higher stakes have diminishing returns
        mult_10k = calculate_stake_multiplier(10000)
        mult_50k = calculate_stake_multiplier(50000)
        
        # Each 10x increase in stake gives progressively less benefit
        gain_1k_to_10k = mult_10k - mult_1k
        gain_10k_to_50k = mult_50k - mult_10k
        
        # 50x increase should give less benefit per token than 10x increase
        per_token_1k_10k = gain_1k_to_10k / 9000
        per_token_10k_50k = gain_10k_to_50k / 40000
        
        self.assertGreater(per_token_1k_10k, per_token_10k_50k)
        
    def test_unstake_locked(self):
        """Test cannot unstake while locked."""
        self.ledger.stake(100.0, duration_days=30)
        
        success, amount, msg = self.ledger.unstake()
        
        self.assertFalse(success)
        self.assertIn("locked", msg.lower())
        
    def test_validator_eligibility(self):
        """Test validator eligibility based on stake."""
        # Initially not eligible
        eligible, reason = self.ledger.is_eligible_validator()
        self.assertFalse(eligible)
        
        # Stake enough to be eligible
        self.ledger.stake(VALIDATOR_MIN_STAKE, duration_days=30)
        
        eligible, reason = self.ledger.is_eligible_validator()
        self.assertTrue(eligible)


class TestTransfers(unittest.TestCase):
    """Tests for transfer functionality."""
    
    def setUp(self):
        """Set up test ledgers."""
        self.temp_dir = tempfile.mkdtemp()
        
        self.sender_ledger = NEUROLedger(
            db_path=os.path.join(self.temp_dir, "sender.db"),
            node_token="sender_token_12345678901234567"
        )
        
        # Credit sender with balance
        self._credit_balance(self.sender_ledger, 1000.0)
        
    def _credit_balance(self, ledger, amount: float):
        """Helper to credit balance."""
        import sqlite3
        with sqlite3.connect(ledger.db_path) as conn:
            conn.execute("""
                INSERT INTO balances (node_id, balance, total_earned, created_at)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(node_id) DO UPDATE SET balance = balance + ?
            """, (ledger.node_id, amount, amount, time.time(), amount))
            
    def tearDown(self):
        """Clean up."""
        import shutil
        try:
            shutil.rmtree(self.temp_dir)
        except:
            pass
            
    def test_transfer(self):
        """Test basic transfer."""
        recipient_id = "recipient123456789012345678901"
        initial_balance = self.sender_ledger.get_balance()
        
        success, msg, tx = self.sender_ledger.transfer(
            to_id=recipient_id,
            amount=100.0,
            memo="Test transfer"
        )
        
        self.assertTrue(success)
        self.assertIsNotNone(tx)
        self.assertEqual(tx.amount, 100.0)
        
        # Check fee burn
        expected_fee = 100.0 * FEE_BURN_RATE
        self.assertAlmostEqual(tx.burn_amount, expected_fee, places=6)
        
        # Check sender balance
        final_balance = self.sender_ledger.get_balance()
        expected_balance = initial_balance - 100.0 - expected_fee
        self.assertAlmostEqual(final_balance, expected_balance, places=6)
        
    def test_transfer_insufficient_balance(self):
        """Test transfer with insufficient balance fails."""
        success, msg, tx = self.sender_ledger.transfer(
            to_id="recipient123456789012345678901",
            amount=10000.0
        )
        
        self.assertFalse(success)
        self.assertIsNone(tx)
        self.assertIn("insufficient", msg.lower())
        
    def test_transfer_negative_amount(self):
        """Test transfer with negative amount fails."""
        success, msg, tx = self.sender_ledger.transfer(
            to_id="recipient123456789012345678901",
            amount=-100.0
        )
        
        self.assertFalse(success)
        
    def test_fee_burn_updates_global_stats(self):
        """Test that fee burn updates global burn stats."""
        stats_before = self.sender_ledger.get_global_stats()
        
        self.sender_ledger.transfer(
            to_id="recipient123456789012345678901",
            amount=100.0
        )
        
        stats_after = self.sender_ledger.get_global_stats()
        
        expected_burn = 100.0 * FEE_BURN_RATE
        self.assertAlmostEqual(
            stats_after.total_burned - stats_before.total_burned,
            expected_burn,
            places=6
        )


class TestSlashing(unittest.TestCase):
    """Tests for slashing functionality."""
    
    def setUp(self):
        """Set up test ledger."""
        self.temp_dir = tempfile.mkdtemp()
        self.db_path = os.path.join(self.temp_dir, "slash_test.db")
        self.ledger = NEUROLedger(
            db_path=self.db_path,
            node_token="slash_test_token_123456789012"
        )
        
    def _credit_balance(self, node_id: str, amount: float):
        """Helper to credit balance."""
        import sqlite3
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT INTO balances (node_id, balance, total_earned, created_at)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(node_id) DO UPDATE SET balance = balance + ?
            """, (node_id, amount, amount, time.time(), amount))
            
    def tearDown(self):
        """Clean up."""
        import shutil
        try:
            shutil.rmtree(self.temp_dir)
        except:
            pass
            
    def test_report_fraud(self):
        """Test reporting fraud."""
        success, msg = self.ledger.report_fraud(
            accused_id="malicious_node_12345678901234",
            reason="Submitted poisoned gradients",
            evidence="Statistical analysis shows gradient deviation"
        )
        
        self.assertTrue(success)
        self.assertIn("report", msg.lower())
        
    def test_execute_slash(self):
        """Test executing a slash."""
        accused_id = "accused_node_123456789012345"
        reporter_id = "reporter_node_12345678901234"
        
        # Give accused some balance
        self._credit_balance(accused_id, 100.0)
        
        success, msg = self.ledger.execute_slash(accused_id, reporter_id)
        
        self.assertTrue(success)
        
        # Check accused balance decreased
        accused_balance = self.ledger.get_balance(accused_id)
        self.assertLess(accused_balance, 100.0)
        
        # Check whistleblower got reward
        reporter_balance = self.ledger.get_balance(reporter_id)
        self.assertGreater(reporter_balance, 0)
        
    def test_slash_no_balance(self):
        """Test slashing node with no balance."""
        success, msg = self.ledger.execute_slash(
            accused_id="empty_node_1234567890123456",
            reporter_id="reporter_node_12345678901234"
        )
        
        self.assertFalse(success)
        self.assertIn("no balance", msg.lower())


class TestAccountInfo(unittest.TestCase):
    """Tests for account information queries."""
    
    def setUp(self):
        """Set up test ledger."""
        self.temp_dir = tempfile.mkdtemp()
        self.ledger = NEUROLedger(
            db_path=os.path.join(self.temp_dir, "account_test.db"),
            node_token="account_test_token_1234567890"
        )
        
    def tearDown(self):
        """Clean up."""
        import shutil
        try:
            shutil.rmtree(self.temp_dir)
        except:
            pass
            
    def test_get_balance_new_account(self):
        """Test getting balance for new account."""
        balance = self.ledger.get_balance()
        self.assertEqual(balance, 0.0)
        
    def test_get_account_info(self):
        """Test getting full account info."""
        info = self.ledger.get_account_info()
        
        self.assertEqual(info["node_id"], self.ledger.node_id)
        self.assertEqual(info["balance"], 0.0)
        self.assertEqual(info["stake"], 0.0)
        self.assertEqual(info["stake_multiplier"], 1.0)
        
    def test_get_burn_stats(self):
        """Test getting burn statistics."""
        stats = self.ledger.get_burn_stats()
        
        self.assertEqual(stats["total_burned"], 0.0)
        self.assertEqual(stats["burn_rate"], FEE_BURN_RATE)


if __name__ == '__main__':
    unittest.main()

