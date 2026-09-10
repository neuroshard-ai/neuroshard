"""
Decentralization and P2P Network Tests for NeuroShard.

These tests verify that the distributed system properly handles:
- Multi-node peer discovery and convergence
- P2P gossip propagation (proofs, transactions, stakes)
- DHT replication across nodes
- Network partitioning and healing
- Byzantine nodes in distributed settings
- DiLoCo gradient gossip synchronization
"""

import sys
import os
import unittest
import asyncio
import tempfile
import time
import hashlib
import threading
import shutil
from typing import Dict, List, Any
from dataclasses import dataclass, field
from collections import defaultdict

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from neuroshard.core.crypto.ecdsa import NodeCrypto, derive_node_id_from_token
from neuroshard.core.economics.ledger import NEUROLedger, PoNWProof, ProofType
from neuroshard.core.swarm.router import SwarmRouter, PeerCandidate
from neuroshard.core.swarm.aggregation import RobustAggregator, AggregationConfig, AggregationStrategy
from neuroshard.core.swarm.diloco import DiLoCoGossipProtocol
from neuroshard.core.swarm.heartbeat import CapacityBitmask
from neuroshard.core.network.dht import RoutingTable, Node, KBucket


# ==================== SIMULATED NETWORK INFRASTRUCTURE ====================

@dataclass
class SimulatedMessage:
    """Message in the simulated network."""
    msg_type: str
    sender: str
    payload: Dict[str, Any]
    timestamp: float = field(default_factory=time.time)


class SimulatedNetwork:
    """
    Simulated P2P network for testing decentralization.
    
    Provides:
    - Message passing between virtual nodes
    - Network partition simulation
    - Latency simulation
    - Message loss simulation
    """
    
    def __init__(self, latency_ms: float = 0, loss_rate: float = 0.0):
        self.nodes: Dict[str, 'VirtualNode'] = {}
        self.message_queue: List[SimulatedMessage] = []
        self.partitions: List[set] = []  # Sets of node IDs that can communicate
        self.latency_ms = latency_ms
        self.loss_rate = loss_rate
        self._lock = threading.Lock()
        self.delivered_messages: Dict[str, List[SimulatedMessage]] = defaultdict(list)
        
    def add_node(self, node: 'VirtualNode'):
        """Add a node to the network."""
        with self._lock:
            self.nodes[node.node_id] = node
            self.delivered_messages[node.node_id] = []
            
    def remove_node(self, node_id: str):
        """Remove a node from the network."""
        with self._lock:
            if node_id in self.nodes:
                del self.nodes[node_id]
                
    def partition(self, group_a: set, group_b: set):
        """Create a network partition between two groups."""
        self.partitions = [group_a, group_b]
        
    def heal_partition(self):
        """Remove network partitions."""
        self.partitions = []
        
    def can_communicate(self, sender: str, receiver: str) -> bool:
        """Check if two nodes can communicate given partitions."""
        if not self.partitions:
            return True
        for partition in self.partitions:
            if sender in partition and receiver in partition:
                return True
        return False
        
    def broadcast(self, sender: str, msg_type: str, payload: Dict[str, Any]):
        """Broadcast a message to all reachable nodes."""
        with self._lock:
            for node_id, node in self.nodes.items():
                if node_id != sender and self.can_communicate(sender, node_id):
                    # Simulate message loss
                    if self.loss_rate > 0 and hash(f"{time.time()}{node_id}") % 100 < self.loss_rate * 100:
                        continue
                    msg = SimulatedMessage(msg_type, sender, payload)
                    self.delivered_messages[node_id].append(msg)
                    node.receive_message(msg)
                    
    def send(self, sender: str, receiver: str, msg_type: str, payload: Dict[str, Any]) -> bool:
        """Send a message to a specific node."""
        if not self.can_communicate(sender, receiver):
            return False
        with self._lock:
            if receiver in self.nodes:
                msg = SimulatedMessage(msg_type, sender, payload)
                self.delivered_messages[receiver].append(msg)
                self.nodes[receiver].receive_message(msg)
                return True
        return False
        
    def get_message_count(self, node_id: str) -> int:
        """Get count of messages delivered to a node."""
        return len(self.delivered_messages.get(node_id, []))


class VirtualNode:
    """
    Virtual node for decentralization testing.
    
    Simulates a NeuroShard node with:
    - Ledger for PoNW
    - DHT routing table
    - Swarm router
    - Gossip capabilities
    """
    
    def __init__(
        self,
        node_id: str,
        network: SimulatedNetwork,
        temp_dir: str,
    ):
        self.node_id = node_id
        self.network = network
        self.token = f"token_{node_id}_" + "x" * max(0, 32 - len(f"token_{node_id}_"))
        
        # Create components
        self.crypto = NodeCrypto(self.token)
        self.ledger = NEUROLedger(
            db_path=os.path.join(temp_dir, f"ledger_{node_id}.db"),
            node_token=self.token
        )
        
        # DHT node ID derived from token
        dht_node_id = int(hashlib.sha256(self.token.encode()).hexdigest()[:40], 16) % (2**160)
        self.dht_node = Node(dht_node_id, "127.0.0.1", 8000 + hash(node_id) % 1000)
        self.dht = RoutingTable(self.dht_node)
        
        # Swarm router
        self.router = SwarmRouter()
        
        # Known peers for gossip
        self.known_peers: set = set()
        
        # Received proofs
        self.received_proofs: List[PoNWProof] = []
        
        # DHT storage
        self.dht_storage: Dict[str, Any] = {}
        
        # Message handlers
        self.message_handlers = {
            'heartbeat': self._handle_heartbeat,
            'proof': self._handle_proof,
            'dht_store': self._handle_dht_store,
            'dht_lookup': self._handle_dht_lookup,
            'stake_update': self._handle_stake_update,
        }
        
        # Register with network
        network.add_node(self)
        
    def receive_message(self, msg: SimulatedMessage):
        """Handle incoming message."""
        handler = self.message_handlers.get(msg.msg_type)
        if handler:
            handler(msg)
            
    def _handle_heartbeat(self, msg: SimulatedMessage):
        """Handle heartbeat from peer."""
        self.known_peers.add(msg.sender)
        
        # Update DHT
        payload = msg.payload
        if 'dht_node_id' in payload:
            peer_dht_node = Node(
                payload['dht_node_id'],
                payload.get('ip', '127.0.0.1'),
                payload.get('port', 8000)
            )
            self.dht.add_contact(peer_dht_node)
            
        # Update router
        if 'layer_range' in payload:
            self.router.update_peer_from_heartbeat(
                node_id=msg.sender,
                grpc_addr=payload.get('grpc_addr', '127.0.0.1:50051'),
                layer_range=payload['layer_range'],
                queue_depth=payload.get('queue_depth', 0),
                available_memory_mb=payload.get('memory_mb', 4096),
                gpu_utilization=payload.get('gpu_util', 50.0),
                is_accepting=True,
            )
            
    def _handle_proof(self, msg: SimulatedMessage):
        """Handle proof from peer."""
        proof_data = msg.payload
        
        # Reconstruct proof
        proof = PoNWProof(
            node_id=proof_data['node_id'],
            timestamp=proof_data['timestamp'],
            proof_type=proof_data['proof_type'],
            nonce=proof_data['nonce'],
            signature=proof_data['signature'],
            uptime_seconds=proof_data.get('uptime_seconds', 0),
            tokens_processed=proof_data.get('tokens_processed', 0),
            training_batches=proof_data.get('training_batches', 0),
            data_samples=proof_data.get('data_samples', 0),
            model_hash=proof_data.get('model_hash', ''),
            layers_held=proof_data.get('layers_held', 0),
            has_embedding=proof_data.get('has_embedding', False),
            has_lm_head=proof_data.get('has_lm_head', False),
        )
        
        self.received_proofs.append(proof)
        
        # Store sender's public key for verification
        if 'public_key' in proof_data:
            self.ledger.crypto.store_public_key(
                proof.node_id,
                proof_data['public_key']
            )
            
    def _handle_dht_store(self, msg: SimulatedMessage):
        """Handle DHT store request."""
        key = msg.payload['key']
        value = msg.payload['value']
        self.dht_storage[key] = value
        
    def _handle_dht_lookup(self, msg: SimulatedMessage):
        """Handle DHT lookup - respond with value if we have it."""
        key = msg.payload['key']
        if key in self.dht_storage:
            self.network.send(
                self.node_id,
                msg.sender,
                'dht_response',
                {'key': key, 'value': self.dht_storage[key]}
            )
            
    def _handle_stake_update(self, msg: SimulatedMessage):
        """Handle stake gossip from peer."""
        # Track peer stakes for validation
        pass
        
    def broadcast_heartbeat(self, layer_range: tuple = (0, 24)):
        """Broadcast heartbeat to network."""
        self.network.broadcast(
            self.node_id,
            'heartbeat',
            {
                'node_id': self.node_id,
                'dht_node_id': self.dht_node.id,
                'ip': self.dht_node.ip,
                'port': self.dht_node.port,
                'layer_range': layer_range,
                'grpc_addr': f'127.0.0.1:{50051 + hash(self.node_id) % 1000}',
                'memory_mb': 8192,
                'gpu_util': 50.0,
            }
        )
        
    def broadcast_proof(self, proof: PoNWProof):
        """Broadcast proof to network."""
        self.network.broadcast(
            self.node_id,
            'proof',
            {
                'node_id': proof.node_id,
                'timestamp': proof.timestamp,
                'proof_type': proof.proof_type,
                'nonce': proof.nonce,
                'signature': proof.signature,
                'uptime_seconds': proof.uptime_seconds,
                'tokens_processed': proof.tokens_processed,
                'training_batches': proof.training_batches,
                'model_hash': proof.model_hash,  # Required for signature verification
                'layers_held': proof.layers_held,
                'has_embedding': proof.has_embedding,
                'has_lm_head': proof.has_lm_head,
                'data_samples': proof.data_samples,
                'public_key': self.crypto.get_public_key_hex(),
            }
        )
        
    def store_in_dht(self, key: str, value: Any):
        """Store a value in DHT (replicated to closest nodes)."""
        # Store locally
        self.dht_storage[key] = value
        
        # Broadcast to all peers (simplified k-replicas)
        self.network.broadcast(
            self.node_id,
            'dht_store',
            {'key': key, 'value': value}
        )


# ==================== DECENTRALIZATION TESTS ====================

class TestMultiNodeGossipPropagation(unittest.TestCase):
    """Test gossip propagation across multiple nodes."""
    
    def setUp(self):
        """Set up multi-node network."""
        self.temp_dir = tempfile.mkdtemp()
        self.network = SimulatedNetwork()
        
        # Create 5 nodes
        self.nodes = []
        for i in range(5):
            node = VirtualNode(f"node_{i}", self.network, self.temp_dir)
            self.nodes.append(node)
            
    def tearDown(self):
        """Clean up."""
        try:
            shutil.rmtree(self.temp_dir)
        except:
            pass
            
    def test_heartbeat_propagates_to_all_nodes(self):
        """Test that heartbeat reaches all nodes."""
        # Node 0 broadcasts heartbeat
        self.nodes[0].broadcast_heartbeat(layer_range=(0, 10))
        
        # All other nodes should receive it
        for i in range(1, 5):
            self.assertIn(
                "node_0",
                self.nodes[i].known_peers,
                f"Node {i} should know about node_0"
            )
            
    def test_full_network_discovery(self):
        """Test all nodes discover each other."""
        # Each node broadcasts heartbeat
        for node in self.nodes:
            node.broadcast_heartbeat()
            
        # Each node should know all other nodes
        for i, node in enumerate(self.nodes):
            expected_peers = {f"node_{j}" for j in range(5) if j != i}
            self.assertEqual(
                node.known_peers,
                expected_peers,
                f"Node {i} should know all peers"
            )
            
    def test_proof_gossip_reaches_all_peers(self):
        """Test that a proof gossips to all nodes."""
        # Register public keys across nodes
        for i, node in enumerate(self.nodes):
            for j, other in enumerate(self.nodes):
                if i != j:
                    other.ledger.crypto.store_public_key(
                        node.crypto.node_id,
                        node.crypto.get_public_key_hex()
                    )
                    
        # Node 0 creates and broadcasts proof
        proof = self.nodes[0].ledger.create_proof(
            proof_type=ProofType.TRAINING,
            uptime_seconds=60,
            training_batches=10,
        )
        self.nodes[0].broadcast_proof(proof)
        
        # All other nodes should receive it
        for i in range(1, 5):
            self.assertEqual(
                len(self.nodes[i].received_proofs), 1,
                f"Node {i} should have received the proof"
            )
            
            # Verify the proof
            received = self.nodes[i].received_proofs[0]
            is_valid, reason = self.nodes[i].ledger.verify_proof(received)
            self.assertTrue(is_valid, f"Proof verification failed: {reason}")
            
    def test_dht_replication(self):
        """Test DHT values replicate across nodes."""
        # First, have nodes discover each other
        for node in self.nodes:
            node.broadcast_heartbeat()
            
        # Node 0 stores a value
        self.nodes[0].store_in_dht("layer_5", "192.168.1.1:50051")
        
        # All nodes should have the value
        for node in self.nodes:
            self.assertIn("layer_5", node.dht_storage)
            self.assertEqual(node.dht_storage["layer_5"], "192.168.1.1:50051")


class TestNetworkPartitioning(unittest.TestCase):
    """Test network behavior under partitioning."""
    
    def setUp(self):
        """Set up network with potential partitions."""
        self.temp_dir = tempfile.mkdtemp()
        self.network = SimulatedNetwork()
        
        # Create 6 nodes (for even split)
        self.nodes = []
        for i in range(6):
            node = VirtualNode(f"node_{i}", self.network, self.temp_dir)
            self.nodes.append(node)
            
    def tearDown(self):
        """Clean up."""
        try:
            shutil.rmtree(self.temp_dir)
        except:
            pass
            
    def test_partition_blocks_communication(self):
        """Test that partitioned nodes cannot communicate."""
        # Create partition: [0,1,2] and [3,4,5]
        group_a = {"node_0", "node_1", "node_2"}
        group_b = {"node_3", "node_4", "node_5"}
        self.network.partition(group_a, group_b)
        
        # Node 0 broadcasts
        self.nodes[0].broadcast_heartbeat()
        
        # Only nodes in same partition should receive
        self.assertIn("node_0", self.nodes[1].known_peers)
        self.assertIn("node_0", self.nodes[2].known_peers)
        
        # Nodes in other partition should NOT receive
        self.assertNotIn("node_0", self.nodes[3].known_peers)
        self.assertNotIn("node_0", self.nodes[4].known_peers)
        self.assertNotIn("node_0", self.nodes[5].known_peers)
        
    def test_within_partition_works(self):
        """Test that nodes within same partition can still operate."""
        # Create partition
        group_a = {"node_0", "node_1", "node_2"}
        group_b = {"node_3", "node_4", "node_5"}
        self.network.partition(group_a, group_b)
        
        # Nodes in group A discover each other
        for i in range(3):
            self.nodes[i].broadcast_heartbeat(layer_range=(i*10, (i+1)*10))
            
        # Each node in A should know the other 2
        for i in range(3):
            self.assertEqual(len(self.nodes[i].known_peers), 2)
            
        # Nodes in group B also discover each other
        for i in range(3, 6):
            self.nodes[i].broadcast_heartbeat(layer_range=(i*10, (i+1)*10))
            
        for i in range(3, 6):
            self.assertEqual(len(self.nodes[i].known_peers), 2)
            
    def test_partition_healing(self):
        """Test that healing partition restores communication."""
        # Create partition
        group_a = {"node_0", "node_1", "node_2"}
        group_b = {"node_3", "node_4", "node_5"}
        self.network.partition(group_a, group_b)
        
        # Nodes discover within partitions
        for node in self.nodes:
            node.broadcast_heartbeat()
            
        # Verify partition exists
        self.assertNotIn("node_0", self.nodes[5].known_peers)
        
        # Heal partition
        self.network.heal_partition()
        
        # Node 0 broadcasts again
        self.nodes[0].broadcast_heartbeat()
        
        # Now node 5 should receive it
        self.assertIn("node_0", self.nodes[5].known_peers)
        
    def test_proof_during_partition(self):
        """Test proof verification works within partition."""
        # Create partition
        group_a = {"node_0", "node_1"}
        group_b = {"node_2", "node_3", "node_4", "node_5"}
        self.network.partition(group_a, group_b)
        
        # Register keys within partition A
        self.nodes[1].ledger.crypto.store_public_key(
            self.nodes[0].crypto.node_id,
            self.nodes[0].crypto.get_public_key_hex()
        )
        
        # Node 0 creates proof and broadcasts
        proof = self.nodes[0].ledger.create_proof(
            proof_type=ProofType.UPTIME,
            uptime_seconds=60,
        )
        self.nodes[0].broadcast_proof(proof)
        
        # Node 1 should receive and verify
        self.assertEqual(len(self.nodes[1].received_proofs), 1)
        is_valid, _ = self.nodes[1].ledger.verify_proof(
            self.nodes[1].received_proofs[0]
        )
        self.assertTrue(is_valid)
        
        # Node 3 should NOT have received it
        self.assertEqual(len(self.nodes[3].received_proofs), 0)


class TestDHTDecentralization(unittest.TestCase):
    """Test DHT decentralization properties."""
    
    def setUp(self):
        """Set up DHT network."""
        self.temp_dir = tempfile.mkdtemp()
        self.network = SimulatedNetwork()
        
        # Create 8 nodes with distributed DHT IDs
        self.nodes = []
        for i in range(8):
            node = VirtualNode(f"node_{i}", self.network, self.temp_dir)
            self.nodes.append(node)
            
        # Have all nodes discover each other
        for node in self.nodes:
            node.broadcast_heartbeat()
            
    def tearDown(self):
        """Clean up."""
        try:
            shutil.rmtree(self.temp_dir)
        except:
            pass
            
    def test_dht_routing_convergence(self):
        """Test DHT routing tables converge with network size."""
        # Each node should have added others to DHT
        for node in self.nodes:
            all_dht_nodes = node.dht.get_all_nodes()
            # Should know about most other nodes
            self.assertGreater(len(all_dht_nodes), 0)
            
    def test_dht_closest_lookup(self):
        """Test DHT finds closest nodes."""
        # Pick a random target ID
        target_id = 0x12345678
        
        # Each node finds closest
        for node in self.nodes:
            closest = node.dht.find_closest(target_id, k=3)
            
            # Should return up to k nodes
            self.assertLessEqual(len(closest), 3)
            
            # Results should be sorted by distance
            if len(closest) > 1:
                for i in range(len(closest) - 1):
                    dist_a = closest[i].id ^ target_id
                    dist_b = closest[i+1].id ^ target_id
                    self.assertLessEqual(dist_a, dist_b)
                    
    def test_dht_value_availability(self):
        """Test DHT values remain available with node churn."""
        # Store value from node 0
        self.nodes[0].store_in_dht("important_key", "important_value")
        
        # Verify all nodes have it
        for node in self.nodes:
            self.assertIn("important_key", node.dht_storage)
            
        # Remove node 0 from network
        self.network.remove_node("node_0")
        
        # Value should still be available from other nodes
        remaining_count = sum(
            1 for node in self.nodes[1:]
            if "important_key" in node.dht_storage
        )
        self.assertEqual(remaining_count, 7)  # All remaining nodes have it


class TestByzantineResistance(unittest.TestCase):
    """Test Byzantine fault tolerance in distributed network."""
    
    def setUp(self):
        """Set up network with some Byzantine nodes."""
        self.temp_dir = tempfile.mkdtemp()
        self.network = SimulatedNetwork()
        
        # Create 7 nodes (can tolerate 2 Byzantine with 3f+1)
        self.nodes = []
        for i in range(7):
            node = VirtualNode(f"node_{i}", self.network, self.temp_dir)
            self.nodes.append(node)
            
        # Register public keys
        for i, node in enumerate(self.nodes):
            for j, other in enumerate(self.nodes):
                if i != j:
                    other.ledger.crypto.store_public_key(
                        node.crypto.node_id,
                        node.crypto.get_public_key_hex()
                    )
                    
    def tearDown(self):
        """Clean up."""
        try:
            shutil.rmtree(self.temp_dir)
        except:
            pass
            
    def test_honest_majority_aggregation(self):
        """Test gradient aggregation with Byzantine minority."""
        # Create aggregator with trimmed mean
        aggregator = RobustAggregator(
            aggregation_config=AggregationConfig(
                strategy=AggregationStrategy.TRIMMED_MEAN,
                trim_fraction=0.25,  # Trim 25% from each end
            )
        )
        
        # 5 honest nodes contribute similar gradients
        base_grad = {'weight': torch.randn(10, 10)}
        for i in range(5):
            honest_grad = {
                'weight': base_grad['weight'] + torch.randn(10, 10) * 0.1
            }
            aggregator.add_contribution(f"honest_{i}", honest_grad, validate=False)
            
        # 2 Byzantine nodes contribute extreme gradients
        for i in range(2):
            byzantine_grad = {
                'weight': base_grad['weight'] * 1000  # Extreme values
            }
            aggregator.add_contribution(f"byzantine_{i}", byzantine_grad, validate=False)
            
        # Aggregate
        result = aggregator.aggregate(local_grads=base_grad)
        
        # Result should be close to honest gradients, not Byzantine
        diff = (result['weight'] - base_grad['weight']).abs().mean()
        self.assertLess(diff, 1.0)  # Should be close to original
        
        # Verify it's NOT dominated by Byzantine values
        byzantine_diff = (result['weight'] - base_grad['weight'] * 1000).abs().mean()
        self.assertGreater(byzantine_diff, diff)  # Should be far from Byzantine
        
    def test_invalid_proof_rejected(self):
        """Test that forged proofs are rejected."""
        # Node 1 creates a valid proof
        valid_proof = self.nodes[1].ledger.create_proof(
            proof_type=ProofType.TRAINING,
            uptime_seconds=60,
            training_batches=10,
        )
        
        # Node 2 tries to forge by modifying the proof
        forged_proof = PoNWProof(
            node_id=valid_proof.node_id,
            timestamp=valid_proof.timestamp,
            proof_type=valid_proof.proof_type,
            nonce=valid_proof.nonce,
            signature=valid_proof.signature,  # Keep original signature
            uptime_seconds=valid_proof.uptime_seconds,
            tokens_processed=99999,  # Forge higher token count
            training_batches=valid_proof.training_batches,
        )
        
        # Other nodes should reject the forged proof
        is_valid, reason = self.nodes[3].ledger.verify_proof(forged_proof)
        self.assertFalse(is_valid)
        self.assertIn("signature", reason.lower())
        
    def test_replay_attack_rejected(self):
        """Test that replay attacks are detected."""
        # Node 0 creates proof
        proof = self.nodes[0].ledger.create_proof(
            proof_type=ProofType.UPTIME,
            uptime_seconds=60,
        )
        
        # Process proof first time - should succeed
        success1, reward1, _ = self.nodes[0].ledger.process_proof(proof)
        self.assertTrue(success1)
        
        # Try to replay same proof - should fail
        success2, reward2, msg = self.nodes[0].ledger.process_proof(proof)
        self.assertFalse(success2)


class TestGossipProtocol(unittest.TestCase):
    """Test DiLoCo gossip protocol synchronization."""
    
    def test_gradient_collection(self):
        """Test collecting gradients from multiple peers."""
        gossip = DiLoCoGossipProtocol(
            node_id="test_node",
            min_peers=2,
            timeout=1.0,
        )
        
        # Simulate receiving contributions from peers
        peer1_grads = {'fc1.weight': torch.randn(10, 10)}
        peer2_grads = {'fc1.weight': torch.randn(10, 10)}
        
        gossip.receive_contribution("peer_1", peer1_grads)
        gossip.receive_contribution("peer_2", peer2_grads)
        
        # Verify contributions stored
        self.assertEqual(len(gossip.pending_contributions), 2)
        
    def test_gradient_aggregation(self):
        """Test aggregating gradients correctly."""
        gossip = DiLoCoGossipProtocol(
            node_id="test_node",
            min_peers=2,
        )
        
        # Create known gradients
        grad1 = torch.ones(5, 5)
        grad2 = torch.ones(5, 5) * 3
        
        contributions = {
            'peer_1': {'weight': grad1},
            'peer_2': {'weight': grad2},
        }
        
        # Aggregate
        result = gossip._aggregate_contributions(contributions)
        
        # Should be average
        expected = (grad1 + grad2) / 2
        self.assertTrue(torch.allclose(result['weight'], expected))


class TestNetworkConvergence(unittest.TestCase):
    """Test network convergence properties."""
    
    def setUp(self):
        """Set up growing network."""
        self.temp_dir = tempfile.mkdtemp()
        self.network = SimulatedNetwork()
        self.nodes = []
        
    def tearDown(self):
        """Clean up."""
        try:
            shutil.rmtree(self.temp_dir)
        except:
            pass
        
    def test_gradual_node_join(self):
        """Test network grows as nodes join."""
        # Start with 2 nodes
        for i in range(2):
            node = VirtualNode(f"node_{i}", self.network, self.temp_dir)
            self.nodes.append(node)
            
        # Both nodes broadcast heartbeats (simulating periodic heartbeat)
        for node in self.nodes:
            node.broadcast_heartbeat()
            
        # Each should know the other
        self.assertIn("node_0", self.nodes[1].known_peers)
        self.assertIn("node_1", self.nodes[0].known_peers)
        
        # Add 3 more nodes one by one
        for i in range(2, 5):
            node = VirtualNode(f"node_{i}", self.network, self.temp_dir)
            self.nodes.append(node)
            
            # New node announces itself to existing nodes
            node.broadcast_heartbeat()
            
            # Existing nodes should discover new node
            for j in range(i):
                self.assertIn(f"node_{i}", self.nodes[j].known_peers)
                
            # Existing nodes also re-broadcast so new node discovers them
            for j in range(i):
                self.nodes[j].broadcast_heartbeat()
                
        # Final state: each node knows all others
        for i, node in enumerate(self.nodes):
            self.assertEqual(len(node.known_peers), 4)
            
    def test_layer_coverage_distribution(self):
        """Test layer coverage distributes across nodes."""
        # Create nodes with different layer responsibilities
        layer_assignments = [
            (0, 8), (8, 16), (16, 24), (0, 8), (8, 16), (16, 24)
        ]
        
        for i, layers in enumerate(layer_assignments):
            node = VirtualNode(f"node_{i}", self.network, self.temp_dir)
            self.nodes.append(node)
            node.broadcast_heartbeat(layer_range=layers)
            
        # Check router has diverse coverage
        test_node = self.nodes[0]
        
        # Should find candidates for each layer range
        for layer in [4, 12, 20]:
            candidates = test_node.router.get_candidates(layer)
            self.assertGreater(len(candidates), 0,
                             f"Should find peers for layer {layer}")


class TestEndToEndDecentralization(unittest.TestCase):
    """End-to-end decentralization scenarios."""
    
    def setUp(self):
        """Set up realistic network."""
        self.temp_dir = tempfile.mkdtemp()
        self.network = SimulatedNetwork()
        
        # Create 5 nodes with different layer assignments
        self.nodes = []
        layer_ranges = [(0, 8), (8, 16), (16, 24), (0, 12), (12, 24)]
        
        for i, layers in enumerate(layer_ranges):
            node = VirtualNode(f"node_{i}", self.network, self.temp_dir)
            node.layer_range = layers
            self.nodes.append(node)
            
        # All nodes discover each other
        for node in self.nodes:
            node.broadcast_heartbeat(layer_range=node.layer_range)
            
        # Register all public keys
        for i, node in enumerate(self.nodes):
            for j, other in enumerate(self.nodes):
                if i != j:
                    other.ledger.crypto.store_public_key(
                        node.crypto.node_id,
                        node.crypto.get_public_key_hex()
                    )
                    
    def tearDown(self):
        """Clean up."""
        try:
            shutil.rmtree(self.temp_dir)
        except:
            pass
            
    def test_training_round_with_gossip(self):
        """Test complete training round with proof gossip."""
        total_proofs_created = 0
        
        # Each node creates training proof
        for node in self.nodes:
            proof = node.ledger.create_proof(
                proof_type=ProofType.TRAINING,
                uptime_seconds=60,
                training_batches=5,
                layers_held=node.layer_range[1] - node.layer_range[0],
            )
            
            # Process locally
            success, reward, _ = node.ledger.process_proof(proof)
            self.assertTrue(success)
            total_proofs_created += 1
            
            # Gossip to network
            node.broadcast_proof(proof)
            
        # Each node should have received proofs from others
        for i, node in enumerate(self.nodes):
            # Should receive 4 proofs (from other 4 nodes)
            self.assertEqual(
                len(node.received_proofs), 4,
                f"Node {i} should receive 4 proofs"
            )
            
            # All should be verifiable
            for proof in node.received_proofs:
                is_valid, reason = node.ledger.verify_proof(proof)
                self.assertTrue(is_valid, f"Proof verification failed: {reason}")
                
    def test_routing_finds_path_through_network(self):
        """Test that routing can find complete layer coverage."""
        # Node 0 covers layers 0-8, needs to find path to complete model
        test_node = self.nodes[0]
        
        # Should find candidates for layers beyond its range
        for layer in [10, 20]:
            candidates = test_node.router.get_candidates(layer)
            self.assertGreater(
                len(candidates), 0,
                f"Should find route to layer {layer}"
            )
            
    def test_dht_stores_layer_info(self):
        """Test DHT stores layer availability info."""
        # Each node stores its layer availability
        for node in self.nodes:
            key = f"layer_{node.layer_range[0]}"
            value = {
                'node_id': node.node_id,
                'layer_range': node.layer_range,
                'grpc_addr': f'127.0.0.1:{50051 + hash(node.node_id) % 1000}'
            }
            node.store_in_dht(key, value)
            
        # All nodes should be able to look up layer info
        for node in self.nodes:
            self.assertIn("layer_0", node.dht_storage)
            self.assertIn("layer_8", node.dht_storage)
            self.assertIn("layer_16", node.dht_storage)


if __name__ == '__main__':
    unittest.main()

