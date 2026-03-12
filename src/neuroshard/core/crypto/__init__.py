# neuroshard/core/crypto/__init__.py
"""
Cryptography components for NeuroShard.

- ecdsa: ECDSA signing and verification
"""

__all__ = [
    'sign_message',
    'verify_signature',
    'generate_keypair',
    'is_valid_node_id_format',
]

def __getattr__(name):
    """Lazy loading of submodules."""
    if name in ('sign_message', 'verify_signature', 'generate_keypair', 'is_valid_node_id_format'):
        from neuroshard.core.crypto import ecdsa
        return getattr(ecdsa, name)
    raise AttributeError(f"module 'neuroshard.core.crypto' has no attribute '{name}'")
