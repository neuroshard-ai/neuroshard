"""
Shared pytest fixtures for test suite.

Provides:
- gRPC server factory for spinning up test nodes
- Multi-node cluster management
- Mock genesis data loader
- Test model creation utilities
"""

import pytest
import asyncio
import socket
import threading
import time
import os
import sys
import tempfile
import hashlib
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from contextlib import contextmanager
from concurrent import futures

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import torch
import torch.nn as nn
import grpc

from neuroshard.protos import neuroshard_pb2, neuroshard_pb2_grpc


# =============================================================================
# PORT ALLOCATION
# =============================================================================

_port_lock = threading.Lock()
_next_port = 50000


def get_free_port() -> int:
    """Get a free port for testing."""
    global _next_port
    with _port_lock:
        port = _next_port
        _next_port += 1
        # Also check if port is actually free
        while not _is_port_free(port):
            port = _next_port
            _next_port += 1
        return port


def _is_port_free(port: int) -> bool:
    """Check if a port is available."""
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.bind(('localhost', port))
        sock.close()
        return True
    except OSError:
        return False


# =============================================================================
# TEST NODE
# =============================================================================

@dataclass
class TestNode:
    """A test node with gRPC server and model."""
    node_id: str
    grpc_port: int
    http_port: int
    layer_range: Tuple[int, int]
    speed_tier: str = "tier2"
    
    # Components (set after creation)
    model: Any = None
    server: Any = None
    dht: Any = None
    ledger: Any = None
    quorum: Any = None
    
    # State
    running: bool = False
    endpoint: str = ""
    
    def __post_init__(self):
        self.endpoint = f"localhost:{self.grpc_port}"
    
    def start(self):
        """Start the gRPC server."""
        if self.server and not self.running:
            self.server.start()
            self.running = True
    
    def stop(self):
        """Stop the gRPC server."""
        if self.server and self.running:
            self.server.stop(grace=1)
            self.running = False
    
    def __enter__(self):
        self.start()
        return self
    
    def __exit__(self, *args):
        self.stop()


# =============================================================================
# MOCK DHT NETWORK
# =============================================================================

class MockDHTNetwork:
    """Simulates a DHT network for testing."""
    
    def __init__(self):
        self.storage: Dict[str, str] = {}
        self.nodes: Dict[str, 'MockDHT'] = {}
        self.lock = threading.Lock()
    
    def create_dht(self, node_id: str) -> 'MockDHT':
        """Create a DHT instance for a node."""
        dht = MockDHT(self, node_id)
        with self.lock:
            self.nodes[node_id] = dht
        return dht
    
    def store(self, key: str, value: str):
        """Store a value in the network."""
        with self.lock:
            self.storage[key] = value
    
    def get(self, key: str) -> Optional[str]:
        """Get a value from the network."""
        with self.lock:
            return self.storage.get(key)
    
    def get_all_keys(self) -> List[str]:
        """Get all keys in the network."""
        with self.lock:
            return list(self.storage.keys())


class MockDHT:
    """DHT interface for a single node."""
    
    def __init__(self, network: MockDHTNetwork, node_id: str):
        self.network = network
        self.node_id = node_id
        self.storage = {}  # Local cache
    
    def announce(self, key: str, value: str = None):
        """Announce presence for a key."""
        if value is None:
            value = self.node_id
        self.network.store(key, value)
        self.storage[key] = value
    
    def lookup_value(self, key: str) -> Optional[str]:
        """Look up a value by key."""
        return self.network.get(key)
    
    def store_value(self, key: str, value: str):
        """Store a value."""
        self.network.store(key, value)
        self.storage[key] = value
    
    def get_local(self, key: str) -> Optional[str]:
        """Get from local cache."""
        return self.storage.get(key)


# =============================================================================
# MOCK GENESIS LOADER
# =============================================================================

class MockGenesisLoader:
    """Provides synthetic training data for tests."""
    
    def __init__(
        self,
        vocab_size: int = 1000,
        seq_length: int = 64,
        batch_size: int = 4,
        num_samples: int = 100,
    ):
        self.vocab_size = vocab_size
        self.seq_length = seq_length
        self.batch_size = batch_size
        self.num_samples = num_samples
        self._index = 0
        
        # Pre-generate data
        torch.manual_seed(42)
        self.data = torch.randint(0, vocab_size, (num_samples, seq_length))
    
    def get_batch(self) -> Optional[Dict[str, torch.Tensor]]:
        """Get next batch of training data."""
        if self._index >= self.num_samples:
            self._index = 0  # Loop
        
        end_idx = min(self._index + self.batch_size, self.num_samples)
        batch_data = self.data[self._index:end_idx]
        self._index = end_idx
        
        if len(batch_data) == 0:
            return None
        
        return {
            'input_ids': batch_data,
            'labels': batch_data.clone(),  # For LM training
            'attention_mask': torch.ones_like(batch_data),
        }
    
    def reset(self):
        """Reset to beginning."""
        self._index = 0


# =============================================================================
# SIMPLE TEST MODEL
# =============================================================================

class SimpleTestModel(nn.Module):
    """A simple transformer-like model for testing."""
    
    def __init__(
        self,
        vocab_size: int = 1000,
        hidden_dim: int = 64,
        num_layers: int = 4,
        num_heads: int = 4,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        # Embedding
        self.embedding = nn.Embedding(vocab_size, hidden_dim)
        
        # Transformer layers
        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=hidden_dim,
                nhead=num_heads,
                dim_feedforward=hidden_dim * 4,
                batch_first=True,
            )
            for _ in range(num_layers)
        ])
        
        # LM head
        self.lm_head = nn.Linear(hidden_dim, vocab_size)
    
    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        x = self.embedding(input_ids)
        for layer in self.layers:
            x = layer(x)
        return self.lm_head(x)
    
    def forward_layers(self, x: torch.Tensor, start: int, end: int) -> torch.Tensor:
        """Forward through specific layers only."""
        for i in range(start, min(end, len(self.layers))):
            x = self.layers[i](x)
        return x
    
    def compute_loss(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """Compute cross-entropy loss."""
        return nn.functional.cross_entropy(
            logits.view(-1, self.vocab_size),
            labels.view(-1),
        )


# =============================================================================
# GRPC SERVICE IMPLEMENTATION FOR TESTS
# =============================================================================

class TestNeuroShardServicer(neuroshard_pb2_grpc.NeuroShardServiceServicer):
    """gRPC servicer for test nodes."""
    
    def __init__(self, node: TestNode):
        self.node = node
        self.received_proofs = []
        self.received_heartbeats = []
        self.received_activations = []
    
    def GossipProof(self, request, context):
        """Handle proof gossip."""
        self.received_proofs.append(request)
        return neuroshard_pb2.GossipProofResponse(
            accepted=True,
        )
    
    def SwarmForward(self, request, context):
        """Handle activation forwarding."""
        self.received_activations.append(request)
        return neuroshard_pb2.SwarmForwardResponse(
            request_id=request.request_id,
            success=True,
        )
    
    def GetSwarmStatus(self, request, context):
        """Return node status."""
        return neuroshard_pb2.SwarmStatusResponse(
            node_id=self.node.node_id,
            layer_start=self.node.layer_range[0],
            layer_end=self.node.layer_range[1],
            is_accepting_activations=True,
        )
    
    def UpdatePeerCapacity(self, request, context):
        """Handle peer capacity update (heartbeat equivalent)."""
        self.received_heartbeats.append(request)
        return neuroshard_pb2.UpdatePeerCapacityResponse(
            success=True,
        )


# =============================================================================
# FIXTURES
# =============================================================================

@pytest.fixture
def dht_network():
    """Create a shared mock DHT network."""
    return MockDHTNetwork()


@pytest.fixture
def mock_genesis_loader():
    """Create a mock genesis data loader."""
    return MockGenesisLoader()


@pytest.fixture
def simple_model():
    """Create a simple test model."""
    return SimpleTestModel()


@pytest.fixture
def grpc_server_factory():
    """Factory for creating gRPC test servers."""
    servers = []
    
    def create_server(node: TestNode) -> grpc.Server:
        """Create and configure a gRPC server for a test node."""
        server = grpc.server(futures.ThreadPoolExecutor(max_workers=4))
        servicer = TestNeuroShardServicer(node)
        neuroshard_pb2_grpc.add_NeuroShardServiceServicer_to_server(servicer, server)
        server.add_insecure_port(f'[::]:{node.grpc_port}')
        node.server = server
        node.servicer = servicer
        servers.append(server)
        return server
    
    yield create_server
    
    # Cleanup
    for server in servers:
        server.stop(grace=0)


@pytest.fixture
def test_node_factory(grpc_server_factory, dht_network):
    """Factory for creating complete test nodes."""
    nodes = []
    
    def create_node(
        layer_range: Tuple[int, int] = (0, 4),
        speed_tier: str = "tier2",
        start: bool = True,
    ) -> TestNode:
        """Create a test node with all components."""
        node_id = hashlib.sha256(f"node_{get_free_port()}".encode()).hexdigest()[:16]
        grpc_port = get_free_port()
        http_port = get_free_port()
        
        node = TestNode(
            node_id=node_id,
            grpc_port=grpc_port,
            http_port=http_port,
            layer_range=layer_range,
            speed_tier=speed_tier,
        )
        
        # Create components
        node.model = SimpleTestModel(num_layers=layer_range[1] - layer_range[0])
        node.dht = dht_network.create_dht(node_id)
        
        # Create gRPC server
        grpc_server_factory(node)
        
        if start:
            node.start()
        
        nodes.append(node)
        return node
    
    yield create_node
    
    # Cleanup
    for node in nodes:
        node.stop()


@pytest.fixture
def multi_node_cluster(test_node_factory):
    """Create a cluster of coordinated test nodes covering all layers."""
    
    def create_cluster(
        num_nodes: int = 4,
        total_layers: int = 12,
        speed_tier: str = "tier2",
    ) -> List[TestNode]:
        """Create a cluster where nodes cover all layers."""
        nodes = []
        layers_per_node = total_layers // num_nodes
        
        for i in range(num_nodes):
            start_layer = i * layers_per_node
            end_layer = start_layer + layers_per_node
            if i == num_nodes - 1:
                end_layer = total_layers  # Last node gets remaining
            
            node = test_node_factory(
                layer_range=(start_layer, end_layer),
                speed_tier=speed_tier,
            )
            nodes.append(node)
        
        # Register nodes in each other's DHT
        for node in nodes:
            for layer in range(node.layer_range[0], node.layer_range[1]):
                node.dht.announce(f"layer:{layer}", node.endpoint)
        
        return nodes
    
    return create_cluster


@pytest.fixture
def event_loop():
    """Create an event loop for async tests."""
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()


# =============================================================================
# UTILITY FIXTURES
# =============================================================================

@pytest.fixture
def temp_db_path():
    """Create a temporary database path."""
    with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as f:
        path = f.name
    yield path
    try:
        os.unlink(path)
    except:
        pass


@pytest.fixture
def temp_dir():
    """Create a temporary directory."""
    import tempfile
    import shutil
    path = tempfile.mkdtemp()
    yield path
    try:
        shutil.rmtree(path)
    except:
        pass
