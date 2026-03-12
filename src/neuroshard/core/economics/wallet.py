"""
Wallet Utilities - BIP39 Mnemonic and Key Derivation
Similar to MetaMask: Users control their own private keys via seed phrases.
"""

from mnemonic import Mnemonic
import hashlib
from cryptography.hazmat.primitives.asymmetric import ec
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.backends import default_backend


class WalletManager:
    """Manages BIP39 mnemonic seed phrases and key derivation"""
    
    def __init__(self):
        self.mnemo = Mnemonic("english")
    
    def generate_mnemonic(self, strength: int = 128) -> str:
        """
        Generate a new BIP39 mnemonic seed phrase.
        
        Args:
            strength: Entropy strength in bits (128 = 12 words, 256 = 24 words)
        
        Returns:
            12-word mnemonic seed phrase
        """
        return self.mnemo.generate(strength=strength)
    
    def validate_mnemonic(self, mnemonic: str) -> bool:
        """
        Validate a BIP39 mnemonic seed phrase.
        
        Args:
            mnemonic: The seed phrase to validate
        
        Returns:
            True if valid, False otherwise
        """
        return self.mnemo.check(mnemonic)
    
    def mnemonic_to_token(self, mnemonic: str) -> str:
        """
        Derive a node token from a mnemonic seed phrase.
        
        Args:
            mnemonic: The BIP39 seed phrase
        
        Returns:
            Deterministic node token (hex string)
        """
        if not self.validate_mnemonic(mnemonic):
            raise ValueError("Invalid mnemonic seed phrase")
        
        # Convert mnemonic to seed (deterministic)
        seed = self.mnemo.to_seed(mnemonic, passphrase="")
        
        # Use first 32 bytes of seed as token
        # This ensures the token is deterministic from the mnemonic
        token = seed[:32].hex()
        
        return token
    
    def token_to_node_id(self, token: str) -> str:
        """
        Derive the public node_id from a token using ECDSA.
        
        This matches the derivation in neuroshard/core/crypto.py:
        1. private_key = SHA256(token)
        2. public_key = ECDSA_derive(private_key) on secp256k1
        3. node_id = SHA256(public_key)[:32]
        
        Args:
            token: The node token (hex string or regular string)
        
        Returns:
            Public node_id (32-char hex string)
        """
        # Derive private key from token
        private_key_bytes = hashlib.sha256(token.encode()).digest()
        
        # Create ECDSA private key on secp256k1 curve
        private_key = ec.derive_private_key(
            int.from_bytes(private_key_bytes, 'big'),
            ec.SECP256K1(),
            default_backend()
        )
        
        # Get compressed public key
        public_key = private_key.public_key()
        public_key_bytes = public_key.public_bytes(
            encoding=serialization.Encoding.X962,
            format=serialization.PublicFormat.CompressedPoint
        )
        
        # node_id = SHA256(public_key)[:32]
        node_id = hashlib.sha256(public_key_bytes).hexdigest()[:32]
        
        return node_id
    
    def create_wallet(self) -> dict:
        """
        Create a complete new wallet with mnemonic, token, and node_id.
        
        Returns:
            {
                'mnemonic': '12-word seed phrase',
                'token': 'node token (PRIVATE - derived from mnemonic)',
                'node_id': 'public wallet address',
                'wallet_id': 'short display ID'
            }
        """
        # Generate new mnemonic
        mnemonic = self.generate_mnemonic()
        
        # Derive token from mnemonic
        token = self.mnemonic_to_token(mnemonic)
        
        # Derive public node_id from token
        node_id = self.token_to_node_id(token)
        
        # Wallet ID is first 16 chars of node_id (for display)
        wallet_id = node_id[:16]
        
        return {
            'mnemonic': mnemonic,
            'token': token,
            'node_id': node_id,
            'wallet_id': wallet_id
        }
    
    def recover_wallet(self, mnemonic: str) -> dict:
        """
        Recover wallet from mnemonic seed phrase.
        
        Args:
            mnemonic: The 12-word seed phrase
        
        Returns:
            {
                'token': 'node token (PRIVATE)',
                'node_id': 'public wallet address',
                'wallet_id': 'short display ID'
            }
        """
        if not self.validate_mnemonic(mnemonic):
            raise ValueError("Invalid mnemonic seed phrase")
        
        # Derive token from mnemonic
        token = self.mnemonic_to_token(mnemonic)
        
        # Derive public node_id from token
        node_id = self.token_to_node_id(token)
        
        # Wallet ID is first 16 chars of node_id (for display)
        wallet_id = node_id[:16]
        
        return {
            'token': token,
            'node_id': node_id,
            'wallet_id': wallet_id
        }


# Global instance
wallet_manager = WalletManager()

