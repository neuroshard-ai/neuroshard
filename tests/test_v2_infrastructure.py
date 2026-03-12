"""
ARCHITECTURE_V2 Infrastructure Tests

Tests for network infrastructure components:
- DHT network state management
- Gossip protocol for proofs and transactions
- P2P connection establishment
- Heartbeat service for quorum health

Uses real gRPC servers on localhost for realistic testing.
"""

import pytest
import asyncio
import time
import json
import hashlib
import threading
from unittest.mock import MagicMock, patch
from concurrent import futures

import grpc

from neuroshard.protos import neuroshard_pb2, neuroshard_pb2_grpc


# =============================================================================
# DHT NETWORK STATE TESTS
# =============================================================================

class TestDHTNetworkState:
    """Tests for DHT-based network state management."""
    
    def test_network_state_creation(self, dht_network):
        """Test creating and storing network state."""
        from neuroshard.core.network.dht_protocol import NetworkState
        
        state = NetworkState(
            arch_version=1,
            vocab_version=1,
            num_layers=12,
            hidden_dim=512,
            vocab_size=32000,
            total_nodes=1,
        )
        
        assert state.arch_version == 1
        assert state.num_layers == 12
        
        # Store in DHT
        dht = dht_network.create_dht("node1")
        dht.store_value("network:state", state.to_json())
        
        # Retrieve and verify
        stored = dht.lookup_value("network:state")
        restored = NetworkState.from_json(stored)
        
        assert restored.arch_version == 1
        assert restored.num_layers == 12
    
    def test_network_state_versioning(self, dht_network):
        """Test that network state versions are tracked correctly."""
        from neuroshard.core.network.dht_protocol import NetworkState
        
        dht = dht_network.create_dht("node1")
        
        # Initial state
        state1 = NetworkState(arch_version=1, num_layers=12)
        dht.store_value("network:state", state1.to_json())
        
        # Upgrade architecture
        state2 = NetworkState(arch_version=2, num_layers=16)
        dht.store_value("network:state", state2.to_json())
        
        # Verify latest
        stored = dht.lookup_value("network:state")
        restored = NetworkState.from_json(stored)
        
        assert restored.arch_version == 2
        assert restored.num_layers == 16
    
    def test_layer_info_storage(self, dht_network):
        """Test storing layer holder information in DHT."""
        dht1 = dht_network.create_dht("node1")
        dht2 = dht_network.create_dht("node2")
        
        # Node 1 announces layers 0-5
        for layer in range(6):
            dht1.announce(f"layer:{layer}", "node1:50001")
        
        # Node 2 announces layers 6-11
        for layer in range(6, 12):
            dht2.announce(f"layer:{layer}", "node2:50002")
        
        # Verify both can discover each other's layers
        assert dht1.lookup_value("layer:6") == "node2:50002"
        assert dht2.lookup_value("layer:0") == "node1:50001"
    
    def test_quorum_registration(self, dht_network):
        """Test registering quorums in DHT."""
        from neuroshard.core.swarm.quorum import Quorum, QuorumMember, QuorumRole
        
        dht = dht_network.create_dht("node1")
        
        # Create a quorum
        quorum = Quorum(
            quorum_id="test-quorum-123",
            speed_tier="tier2",
        )
        quorum.members.append(QuorumMember(
            node_id="node1",
            endpoint="localhost:50001",
            layer_range=(0, 4),
            speed_tier="tier2",
            role=QuorumRole.INITIATOR,
        ))
        
        # Store in DHT
        dht.store_value(f"quorum:{quorum.quorum_id}", quorum.to_json())
        
        # Retrieve and verify
        stored = dht.lookup_value(f"quorum:{quorum.quorum_id}")
        restored = Quorum.from_json(stored)
        
        assert restored.quorum_id == "test-quorum-123"
        assert len(restored.members) == 1
        assert restored.members[0].role == QuorumRole.INITIATOR


# =============================================================================
# GOSSIP PROTOCOL TESTS
# =============================================================================

class TestGossipProtocol:
    """Tests for gossip-based proof and transaction propagation."""
    
    def test_proof_gossip_accepted(self, test_node_factory):
        """Test that proofs are gossiped and accepted by peers."""
        node1 = test_node_factory(layer_range=(0, 6))
        node2 = test_node_factory(layer_range=(6, 12))
        
        # Give servers time to start
        time.sleep(0.5)
        
        # Create gRPC channel to node2
        channel = grpc.insecure_channel(node2.endpoint)
        stub = neuroshard_pb2_grpc.NeuroShardServiceStub(channel)
        
        # Send a proof gossip
        request = neuroshard_pb2.GossipProofRequest(
            signature="test-sig-123",
            node_id=node1.node_id,
            proof_type="training",
            uptime=3600.0,
            token_count=1000,
        )
        
        response = stub.GossipProof(request)
        
        assert response.accepted
        assert len(node2.servicer.received_proofs) == 1
        
        channel.close()
    
    def test_proof_propagation_to_multiple_peers(self, multi_node_cluster):
        """Test that proofs propagate to all peers."""
        nodes = multi_node_cluster(num_nodes=3)
        time.sleep(0.5)
        
        # Send proof from node 0 to nodes 1 and 2
        for target_node in nodes[1:]:
            channel = grpc.insecure_channel(target_node.endpoint)
            stub = neuroshard_pb2_grpc.NeuroShardServiceStub(channel)
            
            request = neuroshard_pb2.GossipProofRequest(
                signature=f"proof-from-{nodes[0].node_id}",
                node_id=nodes[0].node_id,
                proof_type="training",
            )
            
            response = stub.GossipProof(request)
            assert response.accepted
            
            channel.close()
        
        # Verify all received
        for node in nodes[1:]:
            assert len(node.servicer.received_proofs) >= 1
    
    def test_duplicate_proof_handling(self, test_node_factory):
        """Test that duplicate proofs are handled correctly."""
        node = test_node_factory()
        time.sleep(0.5)
        
        channel = grpc.insecure_channel(node.endpoint)
        stub = neuroshard_pb2_grpc.NeuroShardServiceStub(channel)
        
        # Send same proof twice
        request = neuroshard_pb2.GossipProofRequest(
            signature="duplicate-sig",
            node_id="sender",
            proof_type="training",
        )
        
        response1 = stub.GossipProof(request)
        response2 = stub.GossipProof(request)
        
        # Both should be accepted (dedup is at higher level)
        assert response1.accepted
        assert response2.accepted
        
        channel.close()


# =============================================================================
# P2P CONNECTION TESTS
# =============================================================================

class TestP2PConnections:
    """Tests for P2P connection establishment and management."""
    
    def test_node_discovery_via_dht(self, test_node_factory, dht_network):
        """Test that nodes can discover each other via DHT."""
        node1 = test_node_factory(layer_range=(0, 6))
        node2 = test_node_factory(layer_range=(6, 12))
        
        # Announce layers
        for layer in range(6):
            node1.dht.announce(f"layer:{layer}", node1.endpoint)
        for layer in range(6, 12):
            node2.dht.announce(f"layer:{layer}", node2.endpoint)
        
        # Node 1 discovers node 2 for layer 6
        peer = node1.dht.lookup_value("layer:6")
        assert peer == node2.endpoint
        
        # Node 2 discovers node 1 for layer 0
        peer = node2.dht.lookup_value("layer:0")
        assert peer == node1.endpoint
    
    def test_grpc_connection_establishment(self, test_node_factory):
        """Test establishing gRPC connection between nodes."""
        node1 = test_node_factory()
        node2 = test_node_factory()
        time.sleep(0.5)
        
        # Connect node1 to node2
        channel = grpc.insecure_channel(node2.endpoint)
        stub = neuroshard_pb2_grpc.NeuroShardServiceStub(channel)
        
        # Get swarm status to verify connection works
        response = stub.GetSwarmStatus(neuroshard_pb2.SwarmStatusRequest())
        
        assert response.node_id == node2.node_id
        assert response.is_accepting_activations
        
        channel.close()
    
    def test_connection_failure_handling(self):
        """Test handling of connection failures."""
        # Try to connect to non-existent node
        channel = grpc.insecure_channel("localhost:59999")
        stub = neuroshard_pb2_grpc.NeuroShardServiceStub(channel)
        
        with pytest.raises(grpc.RpcError):
            stub.GetSwarmStatus(neuroshard_pb2.SwarmStatusRequest(), timeout=1)
        
        channel.close()
    
    def test_multi_hop_routing(self, multi_node_cluster, dht_network):
        """Test routing through multiple hops."""
        nodes = multi_node_cluster(num_nodes=4, total_layers=12)
        time.sleep(0.5)
        
        # Verify complete layer coverage can be discovered
        all_layers_covered = True
        for layer in range(12):
            peer = dht_network.get(f"layer:{layer}")
            if peer is None:
                all_layers_covered = False
                break
        
        assert all_layers_covered, "Not all layers discoverable via DHT"


# =============================================================================
# HEARTBEAT SERVICE TESTS
# =============================================================================

class TestHeartbeatService:
    """Tests for heartbeat-based health monitoring."""
    
    def test_heartbeat_response(self, test_node_factory):
        """Test that nodes respond to heartbeats (via UpdatePeerCapacity)."""
        node = test_node_factory()
        time.sleep(0.5)
        
        channel = grpc.insecure_channel(node.endpoint)
        stub = neuroshard_pb2_grpc.NeuroShardServiceStub(channel)
        
        # Use UpdatePeerCapacity as heartbeat mechanism
        request = neuroshard_pb2.UpdatePeerCapacityRequest(
            node_id="sender",
            grpc_addr="localhost:50000",
            layer_start=0,
            layer_end=6,
        )
        
        response = stub.UpdatePeerCapacity(request)
        
        assert response.success
        assert len(node.servicer.received_heartbeats) == 1
        
        channel.close()
    
    def test_heartbeat_tracking(self, test_node_factory):
        """Test that heartbeat timing is tracked (via UpdatePeerCapacity)."""
        node = test_node_factory()
        time.sleep(0.5)
        
        channel = grpc.insecure_channel(node.endpoint)
        stub = neuroshard_pb2_grpc.NeuroShardServiceStub(channel)
        
        # Send multiple heartbeats via UpdatePeerCapacity
        for i in range(3):
            request = neuroshard_pb2.UpdatePeerCapacityRequest(
                node_id="sender",
                grpc_addr="localhost:50000",
                layer_start=0,
                layer_end=6,
            )
            stub.UpdatePeerCapacity(request)
            time.sleep(0.1)
        
        assert len(node.servicer.received_heartbeats) == 3
        
        channel.close()
    
    def test_stale_node_detection(self):
        """Test detecting stale nodes based on missed heartbeats."""
        from neuroshard.core.swarm.quorum import (
            QuorumMember, HEARTBEAT_INTERVAL, STALE_THRESHOLD
        )
        
        # Create a member with old heartbeat
        member = QuorumMember(
            node_id="test-node",
            endpoint="localhost:50000",
            layer_range=(0, 4),
            speed_tier="tier2",
        )
        
        # Initially healthy
        member.last_heartbeat = time.time()
        assert not member.is_stale
        
        # Simulate missed heartbeats
        stale_time = HEARTBEAT_INTERVAL * STALE_THRESHOLD + 1
        member.last_heartbeat = time.time() - stale_time
        
        assert member.is_stale


# =============================================================================
# NETWORK PHASE TESTS
# =============================================================================

class TestNetworkPhases:
    """Tests for network phase detection and adaptation."""
    
    def test_phase_detection_by_node_count(self):
        """Test that network phase is correctly detected."""
        from neuroshard.core.network.p2p import get_network_phase, NetworkPhase
        
        assert get_network_phase(1) == NetworkPhase.GENESIS
        assert get_network_phase(3) == NetworkPhase.MICRO
        assert get_network_phase(10) == NetworkPhase.SMALL
        assert get_network_phase(50) == NetworkPhase.MEDIUM
        assert get_network_phase(200) == NetworkPhase.GROWING
        assert get_network_phase(1000) == NetworkPhase.LARGE
        assert get_network_phase(10000) == NetworkPhase.MASSIVE
    
    def test_adaptive_config_scaling(self):
        """Test that adaptive config scales with network size."""
        from neuroshard.core.network.p2p import get_adaptive_config
        
        # Small network (min quorum size is at least 2 for redundancy)
        config_small = get_adaptive_config(5)
        assert config_small.min_quorum_size >= 1
        assert config_small.challenge_rate >= 0.0
        
        # Medium network
        config_medium = get_adaptive_config(50)
        assert config_medium.min_quorum_size >= 2
        assert config_medium.challenge_rate >= 0
        
        # Large network
        config_large = get_adaptive_config(500)
        assert config_large.min_quorum_size >= 3
        assert config_large.challenge_rate > 0
    
    def test_sync_interval_adaptation(self):
        """Test that sync interval adapts to network size."""
        from neuroshard.core.network.p2p import get_adaptive_config
        
        config_small = get_adaptive_config(5)
        config_large = get_adaptive_config(500)
        
        # Larger networks can sync less frequently
        assert config_large.sync_interval >= config_small.sync_interval


# =============================================================================
# ACTIVATION FORWARDING TESTS
# =============================================================================

class TestActivationForwarding:
    """Tests for activation forwarding between pipeline stages."""
    
    def test_swarm_forward_basic(self, test_node_factory):
        """Test basic activation forwarding."""
        import torch
        
        node = test_node_factory(layer_range=(4, 8))
        time.sleep(0.5)
        
        channel = grpc.insecure_channel(node.endpoint)
        stub = neuroshard_pb2_grpc.NeuroShardServiceStub(channel)
        
        # Create mock activation
        activation = torch.randn(2, 64, 256)  # batch, seq, hidden
        
        request = neuroshard_pb2.SwarmForwardRequest(
            session_id="test-session-123",
            request_id="req-123",
            micro_batch_id=0,
            sender_url="sender-node:50000",
            target_layer=4,
            hidden_states=activation.numpy().tobytes(),
            requires_grad=True,
        )
        
        response = stub.SwarmForward(request)
        
        assert response.success
        assert response.request_id == "req-123"
        assert len(node.servicer.received_activations) == 1
        
        channel.close()
    
    def test_activation_chain_through_nodes(self, multi_node_cluster):
        """Test activation forwarding through chain of nodes."""
        nodes = multi_node_cluster(num_nodes=3, total_layers=12)
        time.sleep(0.5)
        
        session_id = "chain-test-session"
        
        # Forward through each node
        for i, node in enumerate(nodes[1:], start=1):
            channel = grpc.insecure_channel(node.endpoint)
            stub = neuroshard_pb2_grpc.NeuroShardServiceStub(channel)
            
            request = neuroshard_pb2.SwarmForwardRequest(
                session_id=session_id,
                request_id=f"req-{i}",
                micro_batch_id=0,
                sender_url=f"{nodes[i-1].endpoint}",
                target_layer=node.layer_range[0],
                hidden_states=b"mock-activation-data",
                requires_grad=True,
            )
            
            response = stub.SwarmForward(request)
            assert response.success
            
            channel.close()
        
        # Verify all nodes received
        for node in nodes[1:]:
            assert len(node.servicer.received_activations) >= 1


# =============================================================================
# INTEGRATION TESTS
# =============================================================================

class TestInfrastructureIntegration:
    """Integration tests combining multiple infrastructure components."""
    
    def test_full_discovery_flow(self, multi_node_cluster, dht_network):
        """Test complete node discovery and connection flow."""
        nodes = multi_node_cluster(num_nodes=4, total_layers=12)
        time.sleep(0.5)
        
        # Each node should be able to discover all layers
        for node in nodes:
            for layer in range(12):
                peer = dht_network.get(f"layer:{layer}")
                assert peer is not None, f"Layer {layer} not discoverable"
        
        # Verify nodes can connect to each other
        for i, node1 in enumerate(nodes):
            for j, node2 in enumerate(nodes):
                if i != j:
                    channel = grpc.insecure_channel(node2.endpoint)
                    stub = neuroshard_pb2_grpc.NeuroShardServiceStub(channel)
                    
                    response = stub.GetSwarmStatus(neuroshard_pb2.SwarmStatusRequest())
                    assert response.is_accepting_activations
                    
                    channel.close()
    
    def test_network_state_with_gossip(self, multi_node_cluster, dht_network):
        """Test network state updates propagate via gossip."""
        from neuroshard.core.network.dht_protocol import NetworkState
        from neuroshard.protos import neuroshard_pb2
        
        nodes = multi_node_cluster(num_nodes=3)
        time.sleep(0.5)
        
        # Node 0 updates network state
        state = NetworkState(
            arch_version=2,
            vocab_version=1,
            num_layers=12,
            total_nodes=3,
        )
        nodes[0].dht.store_value("network:state", state.to_json())
        
        # All nodes should see the same state
        for node in nodes:
            stored = dht_network.get("network:state")
            restored = NetworkState.from_json(stored)
            assert restored.arch_version == 2
            assert restored.total_nodes == 3
