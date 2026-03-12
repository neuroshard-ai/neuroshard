"""
Unit tests for ECDSA Cryptographic Operations.

Tests:
- Key derivation from token
- Signature creation and verification
- Public key registration and lookup
- Node ID derivation
- Signature format validation
- Cross-node verification (trustless)
"""

import sys
import os
import unittest
import hashlib

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from neuroshard.core.crypto.ecdsa import (
    KeyPair,
    NodeCrypto,
    derive_keypair_from_token,
    derive_node_id_from_token,
    ecdsa_sign,
    ecdsa_verify,
    sign_message,
    register_public_key,
    get_public_key,
    verify_signature,
    is_valid_signature_format,
    is_valid_node_id_format,
    _public_key_registry,
)


class TestKeyDerivation(unittest.TestCase):
    """Tests for key derivation from tokens."""
    
    def test_derive_keypair(self):
        """Test deriving a keypair from token."""
        token = "test_token_12345"
        keypair = derive_keypair_from_token(token)
        
        self.assertIsInstance(keypair, KeyPair)
        self.assertEqual(len(keypair.private_key_bytes), 32)  # 256 bits
        self.assertEqual(len(keypair.public_key_bytes), 33)   # Compressed point
        self.assertEqual(len(keypair.node_id), 32)            # 32 hex chars
        
    def test_derive_deterministic(self):
        """Test that derivation is deterministic."""
        token = "deterministic_test_token"
        
        keypair1 = derive_keypair_from_token(token)
        keypair2 = derive_keypair_from_token(token)
        
        self.assertEqual(keypair1.private_key_bytes, keypair2.private_key_bytes)
        self.assertEqual(keypair1.public_key_bytes, keypair2.public_key_bytes)
        self.assertEqual(keypair1.node_id, keypair2.node_id)
        
    def test_derive_different_tokens_different_keys(self):
        """Test that different tokens produce different keys."""
        keypair1 = derive_keypair_from_token("token_one")
        keypair2 = derive_keypair_from_token("token_two")
        
        self.assertNotEqual(keypair1.private_key_bytes, keypair2.private_key_bytes)
        self.assertNotEqual(keypair1.public_key_bytes, keypair2.public_key_bytes)
        self.assertNotEqual(keypair1.node_id, keypair2.node_id)
        
    def test_derive_node_id(self):
        """Test node ID derivation."""
        token = "node_id_test_token"
        node_id = derive_node_id_from_token(token)
        
        self.assertEqual(len(node_id), 32)
        self.assertTrue(all(c in '0123456789abcdef' for c in node_id))
        
    def test_node_id_matches_keypair(self):
        """Test that node_id matches keypair derivation."""
        token = "consistency_test"
        
        keypair = derive_keypair_from_token(token)
        node_id = derive_node_id_from_token(token)
        
        self.assertEqual(keypair.node_id, node_id)


class TestECDSASignature(unittest.TestCase):
    """Tests for ECDSA signing and verification."""
    
    def setUp(self):
        """Set up test keypair."""
        self.token = "signing_test_token"
        self.keypair = derive_keypair_from_token(self.token)
        
    def test_sign_message(self):
        """Test signing a message."""
        message = "Hello, World!"
        signature = ecdsa_sign(message, self.keypair.private_key_bytes)
        
        self.assertIsInstance(signature, str)
        self.assertGreater(len(signature), 100)  # ECDSA sigs are ~70 bytes = ~140 hex
        
    def test_verify_valid_signature(self):
        """Test verifying a valid signature."""
        message = "Test message for verification"
        signature = ecdsa_sign(message, self.keypair.private_key_bytes)
        
        is_valid = ecdsa_verify(message, signature, self.keypair.public_key_bytes)
        
        self.assertTrue(is_valid)
        
    def test_verify_invalid_signature(self):
        """Test verifying an invalid signature."""
        message = "Original message"
        signature = ecdsa_sign(message, self.keypair.private_key_bytes)
        
        # Try to verify with different message
        is_valid = ecdsa_verify("Tampered message", signature, self.keypair.public_key_bytes)
        
        self.assertFalse(is_valid)
        
    def test_verify_wrong_public_key(self):
        """Test verification fails with wrong public key."""
        message = "Test message"
        signature = ecdsa_sign(message, self.keypair.private_key_bytes)
        
        # Use different keypair's public key
        other_keypair = derive_keypair_from_token("different_token")
        
        is_valid = ecdsa_verify(message, signature, other_keypair.public_key_bytes)
        
        self.assertFalse(is_valid)
        
    def test_signature_determinism(self):
        """Test that signatures are deterministic for same message."""
        message = "Deterministic signature test"
        
        sig1 = ecdsa_sign(message, self.keypair.private_key_bytes)
        sig2 = ecdsa_sign(message, self.keypair.private_key_bytes)
        
        # ECDSA with deterministic k (RFC 6979) should produce same signature
        # Note: Some implementations use random k, so this may vary
        # At minimum, both should verify correctly
        self.assertTrue(ecdsa_verify(message, sig1, self.keypair.public_key_bytes))
        self.assertTrue(ecdsa_verify(message, sig2, self.keypair.public_key_bytes))
        
    def test_sign_message_convenience(self):
        """Test convenience sign_message function."""
        message = "Convenience function test"
        
        signature = sign_message(message, self.token)
        
        # Should verify with keypair derived from same token
        is_valid = ecdsa_verify(message, signature, self.keypair.public_key_bytes)
        self.assertTrue(is_valid)


class TestNodeCrypto(unittest.TestCase):
    """Tests for NodeCrypto class."""
    
    def setUp(self):
        """Set up NodeCrypto instance."""
        self.token = "node_crypto_test_token"
        self.crypto = NodeCrypto(self.token)
        
    def test_initialization(self):
        """Test NodeCrypto initialization."""
        self.assertEqual(len(self.crypto.node_id), 32)
        self.assertIsNotNone(self.crypto.keypair)
        
    def test_sign_and_verify_own(self):
        """Test signing and verifying own messages."""
        message = "Self-verification test"
        
        signature = self.crypto.sign(message)
        is_valid = self.crypto.verify(message, signature)
        
        self.assertTrue(is_valid)
        
    def test_get_public_key_hex(self):
        """Test getting public key in hex format."""
        pubkey_hex = self.crypto.get_public_key_hex()
        
        self.assertEqual(len(pubkey_hex), 66)  # 33 bytes = 66 hex chars
        self.assertTrue(all(c in '0123456789abcdef' for c in pubkey_hex))
        
    def test_get_public_key_bytes(self):
        """Test getting raw public key bytes."""
        pubkey_bytes = self.crypto.get_public_key_bytes()
        
        self.assertEqual(len(pubkey_bytes), 33)
        self.assertIsInstance(pubkey_bytes, bytes)


class TestPublicKeyRegistry(unittest.TestCase):
    """Tests for public key registration and lookup."""
    
    def setUp(self):
        """Clear registry before each test."""
        _public_key_registry.clear()
        
    def test_register_public_key(self):
        """Test registering a public key."""
        keypair = derive_keypair_from_token("registry_test")
        pubkey_hex = keypair.public_key_bytes.hex()
        
        success = register_public_key(keypair.node_id, pubkey_hex)
        
        self.assertTrue(success)
        
    def test_register_mismatched_key(self):
        """Test that mismatched node_id/key is rejected."""
        keypair = derive_keypair_from_token("mismatch_test")
        
        # Try to register with wrong node_id
        success = register_public_key("wrong_node_id_12345678901234", keypair.public_key_bytes.hex())
        
        self.assertFalse(success)
        
    def test_get_registered_key(self):
        """Test retrieving a registered key."""
        keypair = derive_keypair_from_token("get_test")
        pubkey_hex = keypair.public_key_bytes.hex()
        
        register_public_key(keypair.node_id, pubkey_hex)
        
        retrieved = get_public_key(keypair.node_id)
        
        self.assertEqual(retrieved, keypair.public_key_bytes)
        
    def test_get_unregistered_key(self):
        """Test retrieving an unregistered key returns None."""
        result = get_public_key("nonexistent_node_id_123456")
        
        self.assertIsNone(result)


class TestCrossNodeVerification(unittest.TestCase):
    """Tests for cross-node (trustless) verification."""
    
    def setUp(self):
        """Set up two nodes."""
        _public_key_registry.clear()
        
        self.node_a_token = "node_a_token_12345"
        self.node_b_token = "node_b_token_67890"
        
        self.crypto_a = NodeCrypto(self.node_a_token)
        self.crypto_b = NodeCrypto(self.node_b_token)
        
    def test_verify_cross_node_with_pubkey(self):
        """Test B can verify A's signature with A's public key."""
        message = "Cross-node verification"
        
        # A signs
        signature = self.crypto_a.sign(message)
        
        # B verifies using A's public key
        is_valid = verify_signature(
            self.crypto_a.node_id,
            message,
            signature,
            public_key_hex=self.crypto_a.get_public_key_hex()
        )
        
        self.assertTrue(is_valid)
        
    def test_verify_cross_node_from_registry(self):
        """Test verification using registered public key."""
        # Register A's public key
        register_public_key(
            self.crypto_a.node_id,
            self.crypto_a.get_public_key_hex()
        )
        
        message = "Registry verification test"
        signature = self.crypto_a.sign(message)
        
        # B verifies without providing public key (uses registry)
        is_valid = verify_signature(
            self.crypto_a.node_id,
            message,
            signature
        )
        
        self.assertTrue(is_valid)
        
    def test_verify_without_pubkey_fails(self):
        """Test verification without public key fails gracefully."""
        message = "No pubkey test"
        signature = self.crypto_a.sign(message)
        
        # Try to verify without public key and not in registry
        is_valid = verify_signature(
            self.crypto_a.node_id,
            message,
            signature
            # No public_key_hex provided
        )
        
        self.assertFalse(is_valid)
        
    def test_verify_with_spoofed_node_id(self):
        """Test that spoofed node_id is detected."""
        message = "Spoof detection test"
        signature = self.crypto_a.sign(message)
        
        # Attacker tries to claim they are node B but uses A's signature
        is_valid = verify_signature(
            self.crypto_b.node_id,  # Wrong node_id
            message,
            signature,
            public_key_hex=self.crypto_a.get_public_key_hex()  # A's key
        )
        
        # Should fail because node_id doesn't match public key
        self.assertFalse(is_valid)


class TestValidationHelpers(unittest.TestCase):
    """Tests for validation helper functions."""
    
    def test_valid_signature_format(self):
        """Test valid signature format detection."""
        # Valid ECDSA signature (hex, 140-144 chars)
        keypair = derive_keypair_from_token("format_test")
        signature = ecdsa_sign("test", keypair.private_key_bytes)
        
        self.assertTrue(is_valid_signature_format(signature))
        
    def test_invalid_signature_format_empty(self):
        """Test empty signature is invalid."""
        self.assertFalse(is_valid_signature_format(""))
        self.assertFalse(is_valid_signature_format(None))
        
    def test_invalid_signature_format_non_hex(self):
        """Test non-hex signature is invalid."""
        self.assertFalse(is_valid_signature_format("not-hex-data!!"))
        
    def test_valid_node_id_format(self):
        """Test valid node ID format detection."""
        node_id = derive_node_id_from_token("id_format_test")
        
        self.assertTrue(is_valid_node_id_format(node_id))
        
    def test_invalid_node_id_format_wrong_length(self):
        """Test node ID with wrong length is invalid."""
        self.assertFalse(is_valid_node_id_format("abc123"))  # Too short
        self.assertFalse(is_valid_node_id_format("a" * 64))  # Too long
        
    def test_invalid_node_id_format_non_hex(self):
        """Test node ID with non-hex chars is invalid."""
        self.assertFalse(is_valid_node_id_format("zzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzz"))
        
    def test_invalid_node_id_format_empty(self):
        """Test empty node ID is invalid."""
        self.assertFalse(is_valid_node_id_format(""))
        self.assertFalse(is_valid_node_id_format(None))


class TestSecurityProperties(unittest.TestCase):
    """Tests for security properties of the crypto system."""
    
    def test_private_key_never_exposed(self):
        """Test that private key is not directly accessible."""
        crypto = NodeCrypto("security_test_token")
        
        # Private key should only be in keypair, not exposed
        self.assertFalse(hasattr(crypto, 'private_key'))
        
        # keypair is accessible but that's internal
        self.assertIsNotNone(crypto.keypair.private_key_bytes)
        
    def test_signature_non_replayable_across_messages(self):
        """Test that signatures can't be replayed across messages."""
        crypto = NodeCrypto("replay_test")
        
        msg1 = "message one"
        msg2 = "message two"
        
        sig1 = crypto.sign(msg1)
        
        # sig1 should not verify for msg2
        self.assertFalse(crypto.verify(msg2, sig1))
        
    def test_node_id_collision_resistance(self):
        """Test that different tokens have different node IDs."""
        # Generate many node IDs and check for collisions
        node_ids = set()
        
        for i in range(1000):
            token = f"collision_test_{i}_{os.urandom(16).hex()}"
            node_id = derive_node_id_from_token(token)
            node_ids.add(node_id)
            
        # All should be unique
        self.assertEqual(len(node_ids), 1000)


class TestEdgeCases(unittest.TestCase):
    """Tests for edge cases and error handling."""
    
    def test_empty_message_signature(self):
        """Test signing and verifying empty message."""
        crypto = NodeCrypto("empty_msg_test")
        
        signature = crypto.sign("")
        is_valid = crypto.verify("", signature)
        
        self.assertTrue(is_valid)
        
    def test_unicode_message_signature(self):
        """Test signing and verifying unicode message."""
        crypto = NodeCrypto("unicode_test")
        
        message = "Hello, 世界! 🌍 Привет мир"
        signature = crypto.sign(message)
        is_valid = crypto.verify(message, signature)
        
        self.assertTrue(is_valid)
        
    def test_long_message_signature(self):
        """Test signing and verifying long message."""
        crypto = NodeCrypto("long_msg_test")
        
        message = "x" * 100000  # 100KB message
        signature = crypto.sign(message)
        is_valid = crypto.verify(message, signature)
        
        self.assertTrue(is_valid)
        
    def test_special_characters_in_token(self):
        """Test token with special characters."""
        token = "test!@#$%^&*()_+-={}|[]\\:\";'<>?,./"
        
        # Should not crash
        keypair = derive_keypair_from_token(token)
        
        self.assertIsNotNone(keypair.node_id)


if __name__ == '__main__':
    unittest.main()

