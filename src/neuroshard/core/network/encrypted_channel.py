"""
Encrypted Prompt Channel for Privacy-Preserving Inference

Users send encrypted prompts DIRECTLY to driver nodes (not via marketplace).
This ensures only the chosen driver can read the prompt.

For simplicity, we use symmetric encryption (shared secret via request_id).
In production, use asymmetric (driver's public key).
"""

import base64
import hashlib
import logging
from typing import Optional, Dict
from dataclasses import dataclass
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
import time

logger = logging.getLogger(__name__)


@dataclass
class EncryptedPrompt:
    """An encrypted prompt sent to a driver node."""
    request_id: str
    encrypted_data: str  # Base64 encoded
    timestamp: float
    user_id: str


class PromptEncryption:
    """Handles encryption/decryption of prompts for privacy."""
    
    @staticmethod
    def derive_key(request_id: str, salt: bytes = b'neuroshard_v1') -> bytes:
        """
        Derive encryption key from request_id.
        
        In production: Use driver's public key (asymmetric encryption).
        For now: Derive symmetric key from request_id (both sides know it).
        """
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=salt,
            iterations=100000,
        )
        key = base64.urlsafe_b64encode(kdf.derive(request_id.encode()))
        return key
    
    @staticmethod
    def encrypt_prompt(prompt: str, request_id: str) -> str:
        """
        Encrypt a prompt for sending to driver.
        
        Args:
            prompt: The plaintext prompt
            request_id: The request ID (used to derive key)
        
        Returns:
            Base64 encoded encrypted data
        """
        key = PromptEncryption.derive_key(request_id)
        f = Fernet(key)
        encrypted = f.encrypt(prompt.encode('utf-8'))
        return base64.b64encode(encrypted).decode('utf-8')
    
    @staticmethod
    def decrypt_prompt(encrypted_data: str, request_id: str) -> str:
        """
        Decrypt a prompt received from user.
        
        Args:
            encrypted_data: Base64 encoded encrypted prompt
            request_id: The request ID (used to derive key)
        
        Returns:
            Plaintext prompt
        """
        key = PromptEncryption.derive_key(request_id)
        f = Fernet(key)
        encrypted = base64.b64decode(encrypted_data.encode('utf-8'))
        decrypted = f.decrypt(encrypted)
        return decrypted.decode('utf-8')


class PromptQueue:
    """
    Queue for encrypted prompts waiting to be processed by driver.
    
    Driver nodes maintain this queue - prompts are sent directly from users.
    """
    
    def __init__(self, max_size: int = 100):
        self.prompts: Dict[str, EncryptedPrompt] = {}
        self.max_size = max_size
        self.timeout = 300  # 5 minutes
    
    def add_prompt(self, prompt: EncryptedPrompt) -> bool:
        """
        Add encrypted prompt to queue.
        
        Returns True if added, False if queue full.
        """
        if len(self.prompts) >= self.max_size:
            # Cleanup old prompts first
            self.cleanup_old_prompts()
            
            if len(self.prompts) >= self.max_size:
                logger.warning(f"Prompt queue full ({self.max_size}), rejecting new prompt")
                return False
        
        self.prompts[prompt.request_id] = prompt
        logger.info(f"Added encrypted prompt for request {prompt.request_id[:8]}... (queue size: {len(self.prompts)})")
        return True
    
    def get_prompt(self, request_id: str) -> Optional[EncryptedPrompt]:
        """Get and remove prompt from queue."""
        prompt = self.prompts.pop(request_id, None)
        if prompt:
            logger.debug(f"Retrieved prompt for request {request_id[:8]}...")
        return prompt
    
    def has_prompt(self, request_id: str) -> bool:
        """Check if prompt exists for request."""
        return request_id in self.prompts
    
    def cleanup_old_prompts(self):
        """Remove prompts older than timeout."""
        now = time.time()
        to_remove = [
            req_id for req_id, prompt in self.prompts.items()
            if (now - prompt.timestamp) > self.timeout
        ]
        
        for req_id in to_remove:
            del self.prompts[req_id]
            logger.info(f"Removed old prompt for request {req_id[:8]}... (timeout)")

