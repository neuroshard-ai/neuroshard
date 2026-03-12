"""
NEURO Cryptographic Module - ECDSA Only

This module provides cryptographic primitives for the NEURO token system
using ECDSA (Elliptic Curve Digital Signature Algorithm) with secp256k1.

Security Model:
===============
ECDSA allows ANYONE to verify a signature without knowing the private key.
This enables TRUE TRUSTLESS verification in the P2P network.

Key Components:
1. Private Key: 32-byte secret derived from node_token
2. Public Key: 33-byte compressed point on secp256k1 curve
3. Node ID: First 32 hex chars of SHA256(public_key)
4. Signature: DER-encoded ECDSA signature

Why ECDSA over HMAC:
- HMAC requires shared secret - only the signer can verify
- ECDSA uses public/private key pair - ANYONE can verify with public key
- This enables trustless P2P verification without central authority
"""

import hashlib
import os
import logging
from typing import Tuple, Optional
from dataclasses import dataclass

logger = logging.getLogger(__name__)

# Import cryptography library for ECDSA - REQUIRED
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import ec
from cryptography.hazmat.backends import default_backend
from cryptography.exceptions import InvalidSignature

ECDSA_AVAILABLE = True


@dataclass
class KeyPair:
    """ECDSA key pair for signing and verification."""
    private_key_bytes: bytes  # 32 bytes
    public_key_bytes: bytes   # 33 bytes (compressed)
    node_id: str              # 32 hex chars (from public key hash)


def derive_keypair_from_token(node_token: str) -> KeyPair:
    """
    Derive an ECDSA key pair from a node token.
    
    The private key is derived deterministically from the token,
    ensuring the same token always produces the same key pair.
    
    Args:
        node_token: The node's secret token (from registration)
        
    Returns:
        KeyPair with private key, public key, and derived node_id
        
    Raises:
        ValueError: If key derivation fails
    """
    # Derive 32-byte private key from token
    private_key_bytes = hashlib.sha256(node_token.encode()).digest()
    
    # Create ECDSA private key from bytes
    private_key = ec.derive_private_key(
        int.from_bytes(private_key_bytes, 'big'),
        ec.SECP256K1(),
        default_backend()
    )
    
    # Get public key in compressed format
    public_key = private_key.public_key()
    public_key_bytes = public_key.public_bytes(
        encoding=serialization.Encoding.X962,
        format=serialization.PublicFormat.CompressedPoint
    )
    
    # Derive node_id from public key (first 32 hex chars of SHA256)
    node_id = hashlib.sha256(public_key_bytes).hexdigest()[:32]
    
    return KeyPair(
        private_key_bytes=private_key_bytes,
        public_key_bytes=public_key_bytes,
        node_id=node_id
    )


def ecdsa_sign(payload: str, private_key_bytes: bytes) -> str:
    """
    Sign a payload using ECDSA with secp256k1 curve.
    
    Args:
        payload: The message to sign
        private_key_bytes: 32-byte private key
        
    Returns:
        Hex-encoded DER signature
    """
    # Recreate private key from bytes
    private_key = ec.derive_private_key(
        int.from_bytes(private_key_bytes, 'big'),
        ec.SECP256K1(),
        default_backend()
    )
    
    # Sign the payload
    signature = private_key.sign(
        payload.encode(),
        ec.ECDSA(hashes.SHA256())
    )
    
    # Return hex-encoded signature
    return signature.hex()


def ecdsa_verify(payload: str, signature_hex: str, public_key_bytes: bytes) -> bool:
    """
    Verify an ECDSA signature.
    
    This is the key function that enables TRUSTLESS verification.
    Anyone can verify a signature using only the public key.
    
    Args:
        payload: The original message
        signature_hex: Hex-encoded DER signature
        public_key_bytes: 33-byte compressed public key
        
    Returns:
        True if signature is valid
    """
    try:
        # Decode signature
        signature = bytes.fromhex(signature_hex)
        
        # Recreate public key from bytes
        public_key = ec.EllipticCurvePublicKey.from_encoded_point(
            ec.SECP256K1(),
            public_key_bytes
        )
        
        # Verify signature
        public_key.verify(
            signature,
            payload.encode(),
            ec.ECDSA(hashes.SHA256())
        )
        
        return True
    except InvalidSignature:
        logger.debug(f"ECDSA signature verification failed")
        return False
    except Exception as e:
        logger.warning(f"ECDSA verification error: {e}")
        return False


def derive_node_id_from_token(token: str) -> str:
    """
    Derive a deterministic node ID from a token.
    
    node_id = SHA256(public_key)[:32]
    
    This ensures the same token always produces the same node_id.
    """
    keypair = derive_keypair_from_token(token)
    return keypair.node_id


def sign_message(payload: str, node_token: str) -> str:
    """
    Sign a message using the node's ECDSA private key.
    
    Convenience wrapper around ecdsa_sign that derives the key from token.
    
    Args:
        payload: The message to sign
        node_token: The node's secret token
        
    Returns:
        Hex-encoded ECDSA signature
    """
    keypair = derive_keypair_from_token(node_token)
    return ecdsa_sign(payload, keypair.private_key_bytes)


class NodeCrypto:
    """
    Cryptographic operations for a node using ECDSA.
    
    Provides signing and verification using secp256k1 curve.
    """
    
    def __init__(self, node_token: str):
        """
        Initialize crypto with node token.
        
        Args:
            node_token: The node's secret token
            
        Raises:
            ValueError: If key derivation fails
        """
        self.node_token = node_token
        self.keypair = derive_keypair_from_token(node_token)
        self.node_id = self.keypair.node_id
        
        logger.info(f"ECDSA initialized for node {self.node_id[:16]}...")
    
    def sign(self, payload: str) -> str:
        """
        Sign a payload using ECDSA.
        
        Args:
            payload: The message to sign
            
        Returns:
            Hex-encoded ECDSA signature
        """
        return ecdsa_sign(payload, self.keypair.private_key_bytes)
    
    def verify(self, payload: str, signature: str, public_key_bytes: Optional[bytes] = None) -> bool:
        """
        Verify a signature.
        
        Args:
            payload: The original message
            signature: Hex-encoded ECDSA signature
            public_key_bytes: Public key to verify against (defaults to our own)
            
        Returns:
            True if signature is valid
        """
        pk = public_key_bytes or self.keypair.public_key_bytes
        return ecdsa_verify(payload, signature, pk)
    
    def get_public_key_hex(self) -> str:
        """Get the public key in hex format for sharing."""
        return self.keypair.public_key_bytes.hex()
    
    def get_public_key_bytes(self) -> bytes:
        """Get the raw public key bytes."""
        return self.keypair.public_key_bytes
    
    def store_public_key(self, node_id: str, public_key_hex: str) -> bool:
        """
        Store a public key for another node (for verification).
        
        This is called when receiving proofs/stakes from other nodes
        that include their public key.
        
        Args:
            node_id: The node's ID
            public_key_hex: Hex-encoded public key
            
        Returns:
            True if stored successfully
        """
        return register_public_key(node_id, public_key_hex)


# Registry of known public keys (node_id -> public_key_bytes)
# This is populated via DHT gossip
_public_key_registry: dict = {}


def register_public_key(node_id: str, public_key_hex: str) -> bool:
    """
    Register a node's public key for future verification.
    
    Called when receiving public key announcements via gossip.
    
    Args:
        node_id: The node's ID (should match SHA256(public_key)[:32])
        public_key_hex: Hex-encoded compressed public key
        
    Returns:
        True if registration successful
    """
    try:
        public_key_bytes = bytes.fromhex(public_key_hex)
        
        # Verify the public key matches the claimed node_id
        expected_node_id = hashlib.sha256(public_key_bytes).hexdigest()[:32]
        if expected_node_id != node_id:
            logger.warning(f"Public key mismatch for {node_id[:16]}... "
                          f"(expected {expected_node_id[:16]}...)")
            return False
        
        _public_key_registry[node_id] = public_key_bytes
        logger.debug(f"Registered public key for {node_id[:16]}...")
        return True
    except Exception as e:
        logger.error(f"Failed to register public key: {e}")
        return False


def get_public_key(node_id: str) -> Optional[bytes]:
    """Get a registered public key for a node."""
    return _public_key_registry.get(node_id)


def verify_signature(node_id: str, payload: str, signature: str, public_key_hex: str = None) -> bool:
    """
    Verify a signature from any node.
    
    This is the key function for trustless verification in the P2P network.
    
    SECURITY: We NEVER accept unverified signatures. If we don't have the 
    public key, the signature is REJECTED. The sender must include their
    public key in the message for verification.
    
    Args:
        node_id: The claimed signer's node ID
        payload: The original message
        signature: Hex-encoded ECDSA signature
        public_key_hex: Optional public key (if not in registry)
        
    Returns:
        True if signature is valid, False otherwise
    """
    # Try to get public key from registry first
    public_key = get_public_key(node_id)
    
    # If not in registry, try the provided public key
    if not public_key and public_key_hex:
        try:
            public_key = bytes.fromhex(public_key_hex)
            
            # CRITICAL: Verify node_id matches the public key!
            # node_id should be SHA256(public_key)[:32]
            expected_node_id = hashlib.sha256(public_key).hexdigest()[:32]
            if expected_node_id != node_id:
                logger.warning(f"Public key mismatch! Claimed {node_id[:16]}... but key gives {expected_node_id[:16]}...")
                return False
            
            # Register for future use
            register_public_key(node_id, public_key_hex)
        except Exception as e:
            logger.debug(f"Failed to parse provided public key: {e}")
    
    if not public_key:
        # SECURITY: No public key = REJECT
        # We cannot verify without the public key
        logger.warning(f"Signature rejected: No public key for {node_id[:16]}...")
        return False
    
    return ecdsa_verify(payload, signature, public_key)


def is_valid_signature_format(signature: str) -> bool:
    """
    Check if a signature has valid ECDSA format.
    
    ECDSA DER signatures are typically 70-72 bytes (140-144 hex chars).
    """
    if not signature:
        return False
    
    # Must be hex
    if not all(c in '0123456789abcdef' for c in signature.lower()):
        return False
    
    # ECDSA DER signatures are typically 70-72 bytes
    # In hex that's 140-144 characters
    sig_len = len(signature)
    if sig_len < 128 or sig_len > 150:
        logger.debug(f"Unusual signature length: {sig_len}")
        # Still allow - some valid signatures can vary
    
    return True


def is_valid_node_id_format(node_id: str) -> bool:
    """
    Check if a node ID has valid format.
    
    Node IDs are 32 hex characters (128 bits).
    """
    if not node_id:
        return False
    
    if len(node_id) != 32:
        return False
    
    if not all(c in '0123456789abcdef' for c in node_id.lower()):
        return False
    
    return True
