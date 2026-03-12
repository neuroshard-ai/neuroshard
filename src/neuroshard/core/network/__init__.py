# neuroshard/core/network/__init__.py
"""
Network components for NeuroShard.

- p2p: P2PManager
- p2p_data: P2PDataManager  
- dht: DHT, Node, RoutingTable
- dht_protocol: DHTProtocol
- dht_service: DHTService, DHTServiceMixin
- nat: NATTraverser
- nat_traversal: NATTraversalManager, STUNClient
- connection_pool: ConnectionPool, get_channel
- encrypted_channel: EncryptedPrompt
"""

# Lazy imports to avoid circular dependencies
__all__ = [
    'P2PManager',
    'P2PDataManager',
    'DHT',
    'DHTProtocol',
    'DHTService',
    'NATTraverser',
    'NATTraversalManager',
    'STUNClient',
    'ConnectionPool',
    'get_channel',
]

def __getattr__(name):
    """Lazy loading of submodules."""
    if name == 'P2PManager':
        from neuroshard.core.network.p2p import P2PManager
        return P2PManager
    elif name == 'P2PDataManager':
        from neuroshard.core.network.p2p_data import P2PDataManager
        return P2PDataManager
    elif name in ('DHT', 'Node', 'RoutingTable', 'ID_BITS'):
        from neuroshard.core.network import dht
        return getattr(dht, name)
    elif name == 'DHTProtocol':
        from neuroshard.core.network.dht_protocol import DHTProtocol
        return DHTProtocol
    elif name in ('DHTService', 'DHTServiceMixin'):
        from neuroshard.core.network import dht_service
        return getattr(dht_service, name)
    elif name == 'NATTraverser':
        from neuroshard.core.network.nat import NATTraverser
        return NATTraverser
    elif name in ('NATTraversalManager', 'STUNClient', 'NATType', 'PeerConnectivity'):
        from neuroshard.core.network import nat_traversal
        return getattr(nat_traversal, name)
    elif name in ('ConnectionPool', 'get_channel'):
        from neuroshard.core.network import connection_pool
        return getattr(connection_pool, name)
    raise AttributeError(f"module 'neuroshard.core.network' has no attribute '{name}'")
