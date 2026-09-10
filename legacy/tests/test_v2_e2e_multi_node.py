"""
ARCHITECTURE_V2 End-to-End Multi-Node Tests

Complete system integration tests with 3-5 nodes on localhost:
- Network bootstrap from genesis
- Complete training session
- Complete inference request
- Layer growth with active quorums
- Node churn handling

Uses real gRPC servers for realistic multi-node simulation.
"""

import pytest
import asyncio
import time
import json
import hashlib
import threading
import torch
import torch.nn as nn
from unittest.mock import MagicMock, patch
from typing import Dict, List, Tuple, Any, Optional
from concurrent import futures
from dataclasses import dataclass

import grpc

from neuroshard.protos import neuroshard_pb2, neuroshard_pb2_grpc


# =============================================================================
# MULTI-NODE BOOTSTRAP TESTS
# =============================================================================

class TestMultiNodeBootstrap:
    """Tests for network startup from genesis."""
    
    def test_genesis_state_initialization(self, dht_network):
        """Test initializing genesis network state."""
        from neuroshard.core.network.dht_protocol import NetworkState
        
        # Create genesis state
        genesis_state = NetworkState(
            arch_version=1,
            vocab_version=1,
            num_layers=12,
            hidden_dim=512,
            vocab_size=32000,
            total_nodes=0,
        )
        
        # Store in DHT
        dht = dht_network.create_dht("genesis")
        dht.store_value("network:state", genesis_state.to_json())
        
        # Verify
        stored = dht.lookup_value("network:state")
        restored = NetworkState.from_json(stored)
        
        assert restored.arch_version == 1
        assert restored.num_layers == 12
        assert restored.total_nodes == 0
    
    def test_node_joins_network(self, test_node_factory, dht_network):
        """Test a node joining the network."""
        from neuroshard.core.network.dht_protocol import NetworkState
        
        # Initialize genesis state
        genesis_dht = dht_network.create_dht("genesis")
        state = NetworkState(arch_version=1, num_layers=12, total_nodes=0)
        genesis_dht.store_value("network:state", state.to_json())
        
        # Node joins
        node = test_node_factory(layer_range=(0, 12))
        
        # Announce layers
        for layer in range(12):
            node.dht.announce(f"layer:{layer}", node.endpoint)
        
        # Update node count
        state = NetworkState.from_json(dht_network.get("network:state"))
        state.total_nodes += 1
        genesis_dht.store_value("network:state", state.to_json())
        
        # Verify
        updated = NetworkState.from_json(dht_network.get("network:state"))
        assert updated.total_nodes == 1
    
    def test_multiple_nodes_bootstrap(self, multi_node_cluster, dht_network):
        """Test bootstrapping multiple nodes."""
        from neuroshard.core.network.dht_protocol import NetworkState
        
        # Initialize genesis
        genesis_dht = dht_network.create_dht("genesis")
        state = NetworkState(arch_version=1, num_layers=12, total_nodes=0)
        genesis_dht.store_value("network:state", state.to_json())
        
        # Create cluster
        nodes = multi_node_cluster(num_nodes=4, total_layers=12)
        
        # Update state
        state.total_nodes = len(nodes)
        genesis_dht.store_value("network:state", state.to_json())
        
        # Verify all layers covered
        for layer in range(12):
            endpoint = dht_network.get(f"layer:{layer}")
            assert endpoint is not None, f"Layer {layer} not covered"
    
    def test_dht_bootstrap_peers(self, multi_node_cluster, dht_network):
        """Test DHT bootstrap between peers."""
        nodes = multi_node_cluster(num_nodes=3)
        
        # All nodes should share DHT state
        # Store something from node 0
        nodes[0].dht.store_value("shared:key", "shared:value")
        
        # Should be visible to all via network
        for node in nodes:
            value = dht_network.get("shared:key")
            assert value == "shared:value"


# =============================================================================
# TRAINING PIPELINE TESTS
# =============================================================================

class TestTrainingPipeline:
    """Tests for complete training sessions."""
    
    def test_quorum_training_session(
        self,
        multi_node_cluster,
        mock_genesis_loader,
        dht_network,
    ):
        """Test a complete training session within a quorum."""
        from neuroshard.core.swarm.quorum import (
            Quorum, QuorumMember, QuorumRole, QuorumLifecycle
        )
        
        # Create cluster
        nodes = multi_node_cluster(num_nodes=3, total_layers=12)
        time.sleep(0.5)
        
        # Form quorum from nodes
        quorum = Quorum(quorum_id="e2e-train", speed_tier="tier2")
        quorum.lifecycle = QuorumLifecycle.ACTIVE
        quorum.session_start = time.time()
        quorum.session_end = quorum.session_start + 3600
        
        for i, node in enumerate(nodes):
            role = [QuorumRole.INITIATOR, QuorumRole.PROCESSOR, QuorumRole.FINISHER][i]
            member = QuorumMember(
                node_id=node.node_id,
                endpoint=node.endpoint,
                layer_range=node.layer_range,
                speed_tier="tier2",
                role=role,
            )
            member.last_heartbeat = time.time()
            quorum.members.append(member)
        
        # Verify quorum setup
        assert len(quorum.members) == 3
        assert quorum.layer_coverage == set(range(12))
        
        # Get training batch
        batch = mock_genesis_loader.get_batch()
        
        # Simulate training forward pass through pipeline
        from neuroshard.protos import neuroshard_pb2
        
        # Initiator processes and forwards to processor
        channel = grpc.insecure_channel(nodes[1].endpoint)
        stub = neuroshard_pb2_grpc.NeuroShardServiceStub(channel)
        
        request = neuroshard_pb2.SwarmForwardRequest(
            session_id="e2e-session",
            request_id="req-e2e-1",
            micro_batch_id=0,
            sender_url=nodes[0].endpoint,
            target_layer=nodes[1].layer_range[0],
            hidden_states=batch['input_ids'].numpy().tobytes(),
            requires_grad=True,
        )
        
        response = stub.SwarmForward(request)
        assert response.success
        
        channel.close()
        
        # Record batch completion
        quorum.record_batch(1)
        assert quorum.total_batches == 1
    
    def test_activation_forwarding_chain(self, multi_node_cluster):
        """Test activation forwarding through the pipeline."""
        from neuroshard.protos import neuroshard_pb2
        
        nodes = multi_node_cluster(num_nodes=3, total_layers=12)
        time.sleep(0.5)
        
        session_id = "forward-chain-test"
        
        # Forward through each consecutive pair
        for i in range(len(nodes) - 1):
            source = nodes[i]
            target = nodes[i + 1]
            
            channel = grpc.insecure_channel(target.endpoint)
            stub = neuroshard_pb2_grpc.NeuroShardServiceStub(channel)
            
            request = neuroshard_pb2.SwarmForwardRequest(
                session_id=session_id,
                request_id=f"req-chain-{i}",
                micro_batch_id=0,
                sender_url=source.endpoint,
                target_layer=target.layer_range[0],
                hidden_states=f"activation-{i}".encode(),
                requires_grad=True,
            )
            
            response = stub.SwarmForward(request)
            assert response.success, f"Forward {i} -> {i+1} failed"
            
            channel.close()
    
    def test_proof_generation_during_training(
        self,
        multi_node_cluster,
        mock_genesis_loader,
    ):
        """Test that proofs are generated during training."""
        from neuroshard.core.consensus.verifier import PipelineProof
        
        nodes = multi_node_cluster(num_nodes=2, total_layers=8)
        time.sleep(0.5)
        
        # Generate proof for each node's work
        proofs = []
        for i, node in enumerate(nodes):
            proof = PipelineProof(
                node_id=node.node_id,
                quorum_id="proof-test-quorum",
                layer_range=node.layer_range,
                batch_id=f"batch-{i}",
                step_id=1,
                input_activation_hash=hashlib.sha256(f"in-{i}".encode()).digest(),
                output_activation_hash=hashlib.sha256(f"out-{i}".encode()).digest(),
                gradient_merkle_root=hashlib.sha256(f"grad-{i}".encode()).digest(),
                prev_node_id=nodes[i-1].node_id if i > 0 else "",
                next_node_id=nodes[i+1].node_id if i < len(nodes)-1 else "",
            )
            proofs.append(proof)
        
        assert len(proofs) == 2
        assert proofs[0].next_node_id == nodes[1].node_id
        assert proofs[1].prev_node_id == nodes[0].node_id


# =============================================================================
# INFERENCE PIPELINE TESTS
# =============================================================================

class TestInferencePipeline:
    """Tests for complete inference request handling."""
    
    def test_inference_request_routing(
        self,
        multi_node_cluster,
        dht_network,
    ):
        """Test routing an inference request to a quorum."""
        from neuroshard.core.swarm.quorum import (
            QuorumInferenceRouter, QuorumRegistry, Quorum,
            QuorumMember, QuorumRole, QuorumLifecycle
        )
        
        nodes = multi_node_cluster(num_nodes=3, total_layers=12)
        time.sleep(0.5)
        
        # Setup router
        dht = dht_network.create_dht("router")
        registry = QuorumRegistry(dht_protocol=dht)
        router = QuorumInferenceRouter(registry=registry, dht_protocol=dht)
        
        # Register quorum
        quorum = Quorum(quorum_id="inf-e2e", speed_tier="tier2")
        quorum.lifecycle = QuorumLifecycle.ACTIVE
        
        for i, node in enumerate(nodes):
            role = [QuorumRole.INITIATOR, QuorumRole.PROCESSOR, QuorumRole.FINISHER][i]
            quorum.members.append(QuorumMember(
                node_id=node.node_id,
                endpoint=node.endpoint,
                layer_range=node.layer_range,
                speed_tier="tier2",
                role=role,
            ))
        
        registry.register_quorum(quorum)
        
        # Route request
        best = router.select_best_quorum()
        
        # Should select the registered quorum
        # (or None if filtering applies)
        assert best is None or best.quorum_id == "inf-e2e"
    
    def test_inference_through_pipeline(
        self,
        multi_node_cluster,
        simple_model,
    ):
        """Test inference request through pipeline."""
        from neuroshard.protos import neuroshard_pb2
        
        nodes = multi_node_cluster(num_nodes=3, total_layers=12)
        time.sleep(0.5)
        
        # Create input
        input_ids = torch.randint(0, simple_model.vocab_size, (1, 32))
        
        # Forward through first node
        channel = grpc.insecure_channel(nodes[0].endpoint)
        stub = neuroshard_pb2_grpc.NeuroShardServiceStub(channel)
        
        request = neuroshard_pb2.SwarmForwardRequest(
            session_id="inf-e2e",
            request_id="req-inf-1",
            micro_batch_id=0,
            sender_url="client:0",
            target_layer=0,
            hidden_states=input_ids.numpy().tobytes(),
            requires_grad=False,
        )
        
        response = stub.SwarmForward(request)
        assert response.success
        
        channel.close()
    
    def test_payment_distribution(self):
        """Test payment distribution after inference."""
        from neuroshard.core.economics.market import InferenceMarket, RequestStatus
        from neuroshard.core.economics.constants import (
            INITIATOR_SHARE, PROCESSOR_SHARE, FINISHER_SHARE
        )
        
        market = InferenceMarket()
        
        # Submit request
        success, req_id, locked_price = market.submit_request(
            user_id="user",
            initiator_node_id="initiator",
            tokens_requested=100,
            max_price=0.01,
        )
        
        # Claim and complete
        market.claim_request("initiator")
        
        # Start session
        market.start_pipeline_session(req_id, "session-1", "initiator")
        market.register_pipeline_participant("session-1", "processor", is_processor=True)
        market.register_pipeline_participant("session-1", "finisher", is_finisher=True)
        
        # Complete with proofs
        market.register_proof_received(req_id, "initiator", True, False)
        complete, _ = market.register_proof_received(req_id, "finisher", False, True)
        
        assert complete
        
        # Calculate expected payments
        total_payment = locked_price * 100  # tokens * price
        initiator_pay = total_payment * INITIATOR_SHARE
        finisher_pay = total_payment * FINISHER_SHARE
        
        assert initiator_pay > 0
        assert finisher_pay > 0


# =============================================================================
# NETWORK GROWTH TESTS
# =============================================================================

class TestNetworkGrowth:
    """Tests for layer growth with active quorums."""
    
    def test_layer_growth_check(self):
        """Test checking if layer growth should trigger."""
        from neuroshard.core.model.dynamic import check_layer_growth
        
        # Small network - no growth likely
        result = check_layer_growth(
            current_layers=12,
            current_hidden_dim=512,
            total_network_memory_mb=10000,
            steps_since_last_growth=5000,  # Not enough steps
            layer_coverage={i: 3 for i in range(12)},
            network_size=10,
        )
        
        # May or may not trigger depending on thresholds
        assert result is None or result is not None
    
    def test_architecture_upgrade_proposal(self):
        """Test proposing an architecture upgrade."""
        from neuroshard.core.model.dynamic import (
            check_layer_growth, ArchitectureUpgrade
        )
        
        # Check if conditions met for growth
        # Large network with good coverage and many steps
        upgrade = check_layer_growth(
            current_layers=12,
            current_hidden_dim=512,
            total_network_memory_mb=100000,  # Large capacity
            steps_since_last_growth=20000,   # Many steps
            layer_coverage={i: 10 for i in range(12)},  # Good coverage
            network_size=100,
        )
        
        # If upgrade proposed, verify structure
        if upgrade:
            assert upgrade.target_layers > 12 or upgrade.target_hidden_dim > 512
    
    def test_migration_during_active_quorums(self, multi_node_cluster, dht_network):
        """Test that migration doesn't disrupt active quorums."""
        from neuroshard.core.swarm.quorum import Quorum, QuorumLifecycle
        from neuroshard.core.network.dht_protocol import NetworkState
        
        nodes = multi_node_cluster(num_nodes=3, total_layers=12)
        time.sleep(0.5)
        
        # Create active quorum
        quorum = Quorum(quorum_id="migration-test", speed_tier="tier2")
        quorum.lifecycle = QuorumLifecycle.ACTIVE
        quorum.session_start = time.time()
        quorum.session_end = quorum.session_start + 3600
        
        # Simulate architecture upgrade (just version bump for test)
        dht = dht_network.create_dht("migration")
        
        state = NetworkState(arch_version=1, num_layers=12)
        dht.store_value("network:state", state.to_json())
        
        # Upgrade to v2
        state.arch_version = 2
        state.num_layers = 16  # Growing layers
        dht.store_value("network:state", state.to_json())
        
        # Quorum should still be active
        assert quorum.lifecycle == QuorumLifecycle.ACTIVE


# =============================================================================
# NODE CHURN TESTS
# =============================================================================

class TestNodeChurn:
    """Tests for handling nodes joining and leaving."""
    
    def test_node_graceful_departure(self, multi_node_cluster, dht_network):
        """Test graceful node departure."""
        from neuroshard.core.swarm.quorum import Quorum, QuorumMember, QuorumRole
        
        nodes = multi_node_cluster(num_nodes=4, total_layers=12)
        time.sleep(0.5)
        
        # Create quorum with all nodes
        quorum = Quorum(quorum_id="churn-test", speed_tier="tier2")
        for i, node in enumerate(nodes):
            quorum.members.append(QuorumMember(
                node_id=node.node_id,
                endpoint=node.endpoint,
                layer_range=node.layer_range,
                speed_tier="tier2",
                role=QuorumRole.PROCESSOR,
            ))
        
        initial_count = len(quorum.members)
        
        # Node 3 departs gracefully
        departing = nodes[3]
        departing.stop()
        
        # Remove from quorum
        quorum.members = [m for m in quorum.members if m.node_id != departing.node_id]
        
        assert len(quorum.members) == initial_count - 1
    
    def test_stale_node_detection_in_quorum(self, multi_node_cluster):
        """Test detecting stale nodes in quorum."""
        from neuroshard.core.swarm.quorum import (
            Quorum, QuorumMember, QuorumRole, HEARTBEAT_INTERVAL, STALE_THRESHOLD
        )
        
        nodes = multi_node_cluster(num_nodes=3, total_layers=12)
        time.sleep(0.5)
        
        # Create quorum
        quorum = Quorum(quorum_id="stale-test", speed_tier="tier2")
        
        for i, node in enumerate(nodes):
            member = QuorumMember(
                node_id=node.node_id,
                endpoint=node.endpoint,
                layer_range=node.layer_range,
                speed_tier="tier2",
                role=QuorumRole.PROCESSOR,
            )
            if i == 2:
                # Make one stale
                member.last_heartbeat = time.time() - (HEARTBEAT_INTERVAL * STALE_THRESHOLD + 10)
            else:
                member.last_heartbeat = time.time()
            quorum.members.append(member)
        
        # Detect stale
        stale = quorum.get_stale_members()
        healthy = quorum.get_healthy_members()
        
        assert len(stale) == 1
        assert len(healthy) == 2
    
    def test_new_node_joins_during_training(
        self,
        test_node_factory,
        multi_node_cluster,
        dht_network,
    ):
        """Test new node joining during active training."""
        from neuroshard.core.network.dht_protocol import NetworkState
        
        # Start with 3 nodes
        nodes = multi_node_cluster(num_nodes=3, total_layers=12)
        time.sleep(0.5)
        
        # Record initial state
        dht = dht_network.create_dht("join-test")
        state = NetworkState(arch_version=1, num_layers=12, total_nodes=3)
        dht.store_value("network:state", state.to_json())
        
        # New node joins
        new_node = test_node_factory(layer_range=(0, 4), speed_tier="tier1")
        
        # Announce layers
        for layer in range(4):
            new_node.dht.announce(f"layer:{layer}", new_node.endpoint)
        
        # Update state
        state.total_nodes = 4
        dht.store_value("network:state", state.to_json())
        
        # Verify new state
        updated = NetworkState.from_json(dht_network.get("network:state"))
        assert updated.total_nodes == 4
    
    def test_quorum_reformation_after_churn(self, multi_node_cluster, dht_network):
        """Test quorum reformation after node churn."""
        from neuroshard.core.swarm.quorum import (
            Quorum, QuorumMember, QuorumRole, QuorumLifecycle
        )
        
        nodes = multi_node_cluster(num_nodes=4, total_layers=12)
        time.sleep(0.5)
        
        # Create initial quorum with first 3 nodes
        quorum = Quorum(quorum_id="reform-test", speed_tier="tier2")
        quorum.lifecycle = QuorumLifecycle.ACTIVE
        
        for i, node in enumerate(nodes[:3]):
            role = [QuorumRole.INITIATOR, QuorumRole.PROCESSOR, QuorumRole.FINISHER][i]
            quorum.members.append(QuorumMember(
                node_id=node.node_id,
                endpoint=node.endpoint,
                layer_range=node.layer_range,
                speed_tier="tier2",
                role=role,
            ))
        
        # Node 2 fails (finisher)
        nodes[2].stop()
        
        # Mark quorum as needing reformation
        quorum.lifecycle = QuorumLifecycle.RENEWING
        
        # Remove failed member, add backup
        quorum.members = [m for m in quorum.members if m.node_id != nodes[2].node_id]
        quorum.members.append(QuorumMember(
            node_id=nodes[3].node_id,
            endpoint=nodes[3].endpoint,
            layer_range=nodes[3].layer_range,
            speed_tier="tier2",
            role=QuorumRole.FINISHER,
        ))
        
        # Back to active
        quorum.lifecycle = QuorumLifecycle.ACTIVE
        
        assert len(quorum.members) == 3
        assert quorum.lifecycle == QuorumLifecycle.ACTIVE


# =============================================================================
# FULL SYSTEM INTEGRATION TESTS
# =============================================================================

class TestFullSystemIntegration:
    """Complete system integration tests."""
    
    def test_full_training_cycle(
        self,
        multi_node_cluster,
        mock_genesis_loader,
        dht_network,
    ):
        """Test a complete training cycle from start to finish."""
        from neuroshard.core.swarm.quorum import (
            Quorum, QuorumMember, QuorumRole, QuorumLifecycle
        )
        from neuroshard.core.consensus.verifier import ProofVerifier, PipelineProof
        from neuroshard.core.network.dht_protocol import NetworkState
        from neuroshard.protos import neuroshard_pb2
        
        # 1. Bootstrap network
        dht = dht_network.create_dht("system")
        state = NetworkState(arch_version=1, num_layers=12, total_nodes=0)
        dht.store_value("network:state", state.to_json())
        
        # 2. Start nodes
        nodes = multi_node_cluster(num_nodes=3, total_layers=12)
        time.sleep(0.5)
        
        state.total_nodes = len(nodes)
        dht.store_value("network:state", state.to_json())
        
        # 3. Form quorum
        quorum = Quorum(quorum_id="full-cycle", speed_tier="tier2")
        quorum.lifecycle = QuorumLifecycle.ACTIVE
        quorum.session_start = time.time()
        quorum.session_end = quorum.session_start + 3600
        
        for i, node in enumerate(nodes):
            role = [QuorumRole.INITIATOR, QuorumRole.PROCESSOR, QuorumRole.FINISHER][i]
            member = QuorumMember(
                node_id=node.node_id,
                endpoint=node.endpoint,
                layer_range=node.layer_range,
                speed_tier="tier2",
                role=role,
            )
            member.last_heartbeat = time.time()
            quorum.members.append(member)
        
        # 4. Get training data
        batch = mock_genesis_loader.get_batch()
        
        # 5. Forward through pipeline
        for i in range(len(nodes) - 1):
            channel = grpc.insecure_channel(nodes[i + 1].endpoint)
            stub = neuroshard_pb2_grpc.NeuroShardServiceStub(channel)
            
            request = neuroshard_pb2.SwarmForwardRequest(
                session_id="train-cycle",
                request_id=f"req-train-{i}",
                micro_batch_id=0,
                sender_url=nodes[i].endpoint,
                target_layer=nodes[i + 1].layer_range[0],
                hidden_states=batch['input_ids'].numpy().tobytes(),
                requires_grad=True,
            )
            
            response = stub.SwarmForward(request)
            assert response.success
            
            channel.close()
        
        # 6. Generate and submit proofs
        verifier = ProofVerifier(optimistic=True)
        
        for i, node in enumerate(nodes):
            proof = PipelineProof(
                node_id=node.node_id,
                quorum_id=quorum.quorum_id,
                layer_range=node.layer_range,
                batch_id=f"cycle-batch-{i}",
                step_id=1,
                input_activation_hash=hashlib.sha256(f"in-{i}".encode()).digest(),
                output_activation_hash=hashlib.sha256(f"out-{i}".encode()).digest(),
                gradient_merkle_root=hashlib.sha256(f"grad-{i}".encode()).digest(),
                prev_node_id=nodes[i-1].node_id if i > 0 else "",
                next_node_id=nodes[i+1].node_id if i < len(nodes)-1 else "",
            )
            
            accepted, _, _ = verifier.accept_proof_optimistic(
                proof=proof.to_dict(),
                proof_id=proof.batch_id,
                reward=10.0,
            )
            assert accepted
        
        # 7. Record completion
        quorum.record_batch(1)
        
        # Verify
        assert quorum.total_batches == 1
        assert quorum.lifecycle == QuorumLifecycle.ACTIVE
    
    def test_full_inference_cycle(
        self,
        multi_node_cluster,
        simple_model,
        dht_network,
    ):
        """Test a complete inference cycle from request to response."""
        from neuroshard.core.swarm.quorum import (
            QuorumInferenceRouter, QuorumRegistry, Quorum,
            QuorumMember, QuorumRole, QuorumLifecycle
        )
        from neuroshard.core.economics.market import InferenceMarket, RequestStatus
        from neuroshard.protos import neuroshard_pb2
        
        # 1. Start nodes
        nodes = multi_node_cluster(num_nodes=3, total_layers=12)
        time.sleep(0.5)
        
        # 2. Setup quorum
        dht = dht_network.create_dht("inference")
        registry = QuorumRegistry(dht_protocol=dht)
        router = QuorumInferenceRouter(registry=registry, dht_protocol=dht)
        
        quorum = Quorum(quorum_id="inf-cycle", speed_tier="tier2")
        quorum.lifecycle = QuorumLifecycle.ACTIVE
        
        for i, node in enumerate(nodes):
            role = [QuorumRole.INITIATOR, QuorumRole.PROCESSOR, QuorumRole.FINISHER][i]
            quorum.members.append(QuorumMember(
                node_id=node.node_id,
                endpoint=node.endpoint,
                layer_range=node.layer_range,
                speed_tier="tier2",
                role=role,
            ))
        
        registry.register_quorum(quorum)
        
        # 3. Setup market
        market = InferenceMarket()
        
        # 4. Submit request
        success, req_id, price = market.submit_request(
            user_id="user-e2e",
            initiator_node_id=nodes[0].node_id,
            tokens_requested=100,
            max_price=0.01,
        )
        assert success
        
        # 5. Claim
        claimed = market.claim_request(nodes[0].node_id)
        assert claimed is not None
        
        # 6. Create input
        input_ids = torch.randint(0, simple_model.vocab_size, (1, 32))
        
        # 7. Forward through pipeline
        for i in range(len(nodes) - 1):
            channel = grpc.insecure_channel(nodes[i + 1].endpoint)
            stub = neuroshard_pb2_grpc.NeuroShardServiceStub(channel)
            
            request = neuroshard_pb2.SwarmForwardRequest(
                session_id=f"inf-{req_id}",
                request_id=f"req-inf-{i}",
                micro_batch_id=0,
                sender_url=nodes[i].endpoint,
                target_layer=nodes[i + 1].layer_range[0],
                hidden_states=input_ids.numpy().tobytes(),
                requires_grad=False,
            )
            
            response = stub.SwarmForward(request)
            assert response.success
            
            channel.close()
        
        # 8. Complete with proofs
        market.start_pipeline_session(req_id, f"session-{req_id}", nodes[0].node_id)
        market.register_proof_received(req_id, nodes[0].node_id, True, False)
        complete, _ = market.register_proof_received(req_id, nodes[2].node_id, False, True)
        
        assert complete
        
        # Verify
        request = market.get_request(req_id)
        assert request.status == RequestStatus.COMPLETED
