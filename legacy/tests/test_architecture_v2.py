"""
Tests for ARCHITECTURE_V2 Implementation

This module tests all new components added for Architecture V2:
- Speed Tiers and self-benchmarking
- Contribution Modes
- Network State with versioning
- Quorum System
- Optimistic PoNW verification
- sqrt(n) * freshness gradient weighting
- Scarcity-based incentives
- Layer growth system
- Network phase detection and adaptive parameters
"""

import pytest
import time
import math
import threading
from unittest.mock import MagicMock, patch


# =============================================================================
# SPEED TIER TESTS
# =============================================================================

class TestSpeedTiers:
    """Tests for speed tier classification."""
    
    def test_speed_tier_enum(self):
        """Test SpeedTier enum values."""
        from neuroshard.core.model.dynamic import SpeedTier
        
        assert SpeedTier.T1.value == "tier1"
        assert SpeedTier.T2.value == "tier2"
        assert SpeedTier.T3.value == "tier3"
        assert SpeedTier.T4.value == "tier4"
        assert SpeedTier.T5.value == "tier5"
    
    def test_classify_speed_tier(self):
        """Test speed tier classification by latency."""
        from neuroshard.core.model.dynamic import classify_speed_tier, SpeedTier
        
        # Fast GPUs
        assert classify_speed_tier(5) == SpeedTier.T1   # < 10ms
        assert classify_speed_tier(10) == SpeedTier.T2  # = 10ms
        assert classify_speed_tier(30) == SpeedTier.T2  # 10-50ms
        
        # Mid-range
        assert classify_speed_tier(50) == SpeedTier.T3  # = 50ms
        assert classify_speed_tier(100) == SpeedTier.T3  # 50-200ms
        
        # Slower
        assert classify_speed_tier(200) == SpeedTier.T4  # = 200ms
        assert classify_speed_tier(500) == SpeedTier.T4  # 200-1000ms
        
        # Very slow
        assert classify_speed_tier(1000) == SpeedTier.T5  # = 1000ms
        assert classify_speed_tier(5000) == SpeedTier.T5  # > 1000ms
    
    def test_tier_compatibility(self):
        """Test speed tier compatibility for quorum formation."""
        from neuroshard.core.model.dynamic import are_tiers_compatible, SpeedTier
        
        # Same tier always compatible
        assert are_tiers_compatible(SpeedTier.T1, SpeedTier.T1)
        assert are_tiers_compatible(SpeedTier.T3, SpeedTier.T3)
        
        # Adjacent tiers compatible
        assert are_tiers_compatible(SpeedTier.T1, SpeedTier.T2)
        assert are_tiers_compatible(SpeedTier.T2, SpeedTier.T1)
        
        # Non-adjacent tiers not compatible
        assert not are_tiers_compatible(SpeedTier.T1, SpeedTier.T3)
        assert not are_tiers_compatible(SpeedTier.T1, SpeedTier.T5)


# =============================================================================
# CONTRIBUTION MODE TESTS
# =============================================================================

class TestContributionModes:
    """Tests for contribution mode selection."""
    
    def test_contribution_mode_enum(self):
        """Test ContributionMode enum values."""
        from neuroshard.core.model.dynamic import ContributionMode
        
        assert ContributionMode.PIPELINE.value == "pipeline"
        assert ContributionMode.ASYNC.value == "async"
        assert ContributionMode.DATA.value == "data"
        assert ContributionMode.VERIFY.value == "verify"
        assert ContributionMode.INFERENCE.value == "inference"
        assert ContributionMode.IDLE.value == "idle"
    
    def test_select_contribution_mode_t5_async(self):
        """Test that T5 nodes default to async mode."""
        from neuroshard.core.model.dynamic import select_contribution_mode, ContributionMode, SpeedTier
        
        mode = select_contribution_mode(
            speed_tier=SpeedTier.T5,
            has_layers=True,
            has_quorum=False,
            has_genesis_data=False,
            has_stake=False,
        )
        assert mode == ContributionMode.ASYNC
    
    def test_select_contribution_mode_t5_data(self):
        """Test that T5 nodes with genesis data become data providers."""
        from neuroshard.core.model.dynamic import select_contribution_mode, ContributionMode, SpeedTier
        
        mode = select_contribution_mode(
            speed_tier=SpeedTier.T5,
            has_layers=False,
            has_quorum=False,
            has_genesis_data=True,
            has_stake=False,
        )
        assert mode == ContributionMode.DATA
    
    def test_select_contribution_mode_pipeline(self):
        """Test that nodes in quorum with layers become pipeline mode."""
        from neuroshard.core.model.dynamic import select_contribution_mode, ContributionMode, SpeedTier
        
        mode = select_contribution_mode(
            speed_tier=SpeedTier.T2,
            has_layers=True,
            has_quorum=True,
            has_genesis_data=False,
            has_stake=False,
        )
        assert mode == ContributionMode.PIPELINE
    
    def test_select_contribution_mode_idle(self):
        """Test that nodes without layers or roles go idle."""
        from neuroshard.core.model.dynamic import select_contribution_mode, ContributionMode, SpeedTier
        
        mode = select_contribution_mode(
            speed_tier=SpeedTier.T3,
            has_layers=False,
            has_quorum=False,
            has_genesis_data=False,
            has_stake=False,
        )
        assert mode == ContributionMode.IDLE


# =============================================================================
# NETWORK STATE TESTS
# =============================================================================

class TestNetworkState:
    """Tests for network state versioning."""
    
    def test_network_state_creation(self):
        """Test NetworkState dataclass creation."""
        from neuroshard.core.network.dht_protocol import NetworkState
        
        state = NetworkState(
            arch_version=1,
            vocab_version=1,
            num_layers=11,
            hidden_dim=512,
            vocab_size=32000,
        )
        
        assert state.arch_version == 1
        assert state.vocab_version == 1
        assert state.num_layers == 11
    
    def test_network_state_serialization(self):
        """Test NetworkState JSON serialization."""
        from neuroshard.core.network.dht_protocol import NetworkState
        
        state = NetworkState(
            arch_version=2,
            vocab_version=3,
            num_layers=24,
            hidden_dim=1024,
            vocab_size=50000,
        )
        
        json_str = state.to_json()
        restored = NetworkState.from_json(json_str)
        
        assert restored.arch_version == 2
        assert restored.vocab_version == 3
        assert restored.num_layers == 24
    
    def test_network_state_compatibility(self):
        """Test NetworkState compatibility checking."""
        from neuroshard.core.network.dht_protocol import NetworkState
        
        state1 = NetworkState(arch_version=1, vocab_version=1)
        state2 = NetworkState(arch_version=1, vocab_version=1)
        state3 = NetworkState(arch_version=2, vocab_version=1)
        
        assert state1.is_compatible_with(state2)
        assert not state1.is_compatible_with(state3)


# =============================================================================
# QUORUM TESTS
# =============================================================================

class TestQuorum:
    """Tests for quorum system."""
    
    def test_quorum_creation(self):
        """Test Quorum dataclass creation."""
        from neuroshard.core.swarm.quorum import Quorum, QuorumLifecycle
        
        quorum = Quorum(
            quorum_id="q-test123",
            speed_tier="tier2",
        )
        
        assert quorum.quorum_id == "q-test123"
        assert quorum.speed_tier == "tier2"
        assert quorum.lifecycle == QuorumLifecycle.FORMING
    
    def test_quorum_member_addition(self):
        """Test adding members to quorum."""
        from neuroshard.core.swarm.quorum import Quorum, QuorumMember, QuorumRole
        
        quorum = Quorum(quorum_id="q-test", speed_tier="tier2")
        
        member = QuorumMember(
            node_id="node1",
            endpoint="localhost:8000",
            layer_range=(0, 5),
            speed_tier="tier2",
        )
        
        assert quorum.add_member(member)
        assert len(quorum.members) == 1
        assert quorum.members[0].role == QuorumRole.INITIATOR  # Has layer 0
    
    def test_quorum_layer_coverage(self):
        """Test quorum layer coverage calculation."""
        from neuroshard.core.swarm.quorum import Quorum, QuorumMember
        
        quorum = Quorum(quorum_id="q-test", speed_tier="tier2")
        
        quorum.add_member(QuorumMember(
            node_id="node1",
            endpoint="localhost:8000",
            layer_range=(0, 5),
            speed_tier="tier2",
        ))
        quorum.add_member(QuorumMember(
            node_id="node2",
            endpoint="localhost:8001",
            layer_range=(5, 10),
            speed_tier="tier2",
        ))
        
        coverage = quorum.layer_coverage
        assert coverage == set(range(10))
    
    def test_quorum_serialization(self):
        """Test quorum JSON serialization."""
        from neuroshard.core.swarm.quorum import Quorum, QuorumMember
        
        quorum = Quorum(quorum_id="q-test", speed_tier="tier2")
        quorum.add_member(QuorumMember(
            node_id="node1",
            endpoint="localhost:8000",
            layer_range=(0, 5),
            speed_tier="tier2",
        ))
        
        json_str = quorum.to_json()
        restored = Quorum.from_json(json_str)
        
        assert restored.quorum_id == "q-test"
        assert len(restored.members) == 1
    
    def test_quorum_session_management(self):
        """Test quorum session timing."""
        from neuroshard.core.swarm.quorum import Quorum, BASE_SESSION_DURATION, RENEWAL_CHECK_RATIO
        
        quorum = Quorum(quorum_id="q-test", speed_tier="tier2")
        
        # Session should have remaining time
        assert quorum.session_remaining > 0
        assert quorum.session_remaining <= BASE_SESSION_DURATION
        
        # Not yet at renewal check
        assert not quorum.should_check_renewal


# =============================================================================
# OPTIMISTIC VERIFICATION TESTS
# =============================================================================

class TestOptimisticVerification:
    """Tests for optimistic PoNW verification."""
    
    def test_pending_proof_creation(self):
        """Test PendingProof dataclass."""
        from neuroshard.core.consensus.verifier import PendingProof, ProofStatus
        
        proof = PendingProof(
            proof_id="proof123",
            node_id="node1",
            proof_type="training",
            reward_credited=1.5,
            timestamp=time.time(),
        )
        
        assert proof.status == ProofStatus.PENDING
        assert proof.is_challengeable
        assert not proof.is_window_expired
    
    def test_proof_challenge_window(self):
        """Test proof challenge window timing."""
        from neuroshard.core.consensus.verifier import PendingProof, CHALLENGE_WINDOW
        
        # Create proof with old timestamp
        old_time = time.time() - CHALLENGE_WINDOW - 10
        proof = PendingProof(
            proof_id="proof123",
            node_id="node1",
            proof_type="training",
            reward_credited=1.5,
            timestamp=old_time,
        )
        
        assert proof.is_window_expired
        assert not proof.is_challengeable
    
    def test_pending_proof_store(self):
        """Test PendingProofStore operations."""
        from neuroshard.core.consensus.verifier import PendingProofStore, PendingProof
        
        store = PendingProofStore()
        
        proof = PendingProof(
            proof_id="proof123",
            node_id="node1",
            proof_type="training",
            reward_credited=1.5,
            timestamp=time.time(),
        )
        
        store.add_proof(proof)
        
        assert store.get_proof("proof123") is not None
        assert len(store.get_proofs_by_node("node1")) == 1
        assert len(store.get_challengeable_proofs()) == 1
    
    def test_verifier_optimistic_accept(self):
        """Test optimistic proof acceptance."""
        from neuroshard.core.consensus.verifier import ProofVerifier
        
        verifier = ProofVerifier(optimistic=True)
        
        class MockProof:
            proof_type = "training"
            uptime_seconds = 100
            training_batches = 10
            node_id = "node1"
            tokens_processed = 0
            model_hash = "abc123"
            signature = "sig123"
        
        proof = MockProof()
        accepted, reason, pending = verifier.accept_proof_optimistic(
            proof, "proof123", 1.5
        )
        
        assert accepted
        assert pending is not None
        assert pending.proof_id == "proof123"


# =============================================================================
# GRADIENT WEIGHTING TESTS
# =============================================================================

class TestGradientWeighting:
    """Tests for sqrt(n) * freshness gradient weighting."""
    
    def test_freshness_calculation(self):
        """Test freshness decay calculation."""
        from neuroshard.core.swarm.aggregation import GradientContribution
        import torch
        
        # Fresh contribution
        fresh = GradientContribution(
            peer_id="peer1",
            gradients={"layer": torch.zeros(10)},
            timestamp=time.time(),
            batches_processed=100,
        )
        assert fresh.freshness == 1.0
        
        # Old contribution (1.5 hours ago)
        old_time = time.time() - 1.5 * 3600
        old = GradientContribution(
            peer_id="peer2",
            gradients={"layer": torch.zeros(10)},
            timestamp=old_time,
            batches_processed=100,
        )
        assert old.freshness == 0.9
    
    def test_weight_calculation(self):
        """Test sqrt(n) * freshness weight calculation."""
        from neuroshard.core.swarm.aggregation import GradientContribution
        import torch
        
        contrib = GradientContribution(
            peer_id="peer1",
            gradients={"layer": torch.zeros(10)},
            timestamp=time.time(),
            batches_processed=100,
        )
        
        # Weight should be sqrt(100) * 1.0 = 10.0
        assert contrib.weight == pytest.approx(10.0)
    
    def test_compute_contribution_weight(self):
        """Test standalone weight computation function."""
        from neuroshard.core.swarm.aggregation import compute_contribution_weight
        
        # Fresh, many batches
        assert compute_contribution_weight(100, 0.5) == pytest.approx(10.0)  # sqrt(100) * 1.0
        
        # Old, few batches
        assert compute_contribution_weight(4, 48) == pytest.approx(1.4)  # sqrt(4) * 0.7


# =============================================================================
# SCARCITY INCENTIVES TESTS
# =============================================================================

class TestScarcityIncentives:
    """Tests for scarcity-based incentives."""
    
    def test_target_replicas(self):
        """Test target replica calculation."""
        from neuroshard.core.economics.constants import get_target_replicas
        
        # Small network
        assert get_target_replicas(10) == 3  # Minimum
        
        # Medium network
        assert get_target_replicas(50) == 5  # 50 * 0.1
        
        # Large network
        assert get_target_replicas(100) == 10
    
    def test_scarcity_bonus(self):
        """Test scarcity bonus calculation."""
        from neuroshard.core.economics.constants import compute_scarcity_bonus
        
        # Well-replicated layer (no bonus)
        bonus = compute_scarcity_bonus(5, 10, 100)
        assert bonus == 1.0
        
        # Under-replicated layer (bonus)
        bonus = compute_scarcity_bonus(5, 3, 100)  # Target is 10
        assert bonus > 1.0
        
        # Critical shortage (max bonus capped)
        bonus = compute_scarcity_bonus(5, 1, 100)
        assert bonus <= 2.0  # 1.0 + SCARCITY_BONUS_MAX
    
    def test_layer_scarcity_scores(self):
        """Test layer scarcity score calculation."""
        from neuroshard.core.economics.constants import get_layer_scarcity_scores
        
        layer_counts = {0: 10, 1: 5, 2: 2}
        scores = get_layer_scarcity_scores(layer_counts, 100)
        
        assert 0 in scores
        assert scores[0]["bonus"] == 1.0  # Well-replicated
        assert scores[2]["under_replicated"]  # Under-replicated


# =============================================================================
# LAYER GROWTH TESTS
# =============================================================================

class TestLayerGrowth:
    """Tests for layer growth system."""
    
    def test_growth_phase_enum(self):
        """Test GrowthPhase enum values."""
        from neuroshard.core.model.dynamic import GrowthPhase
        
        assert GrowthPhase.NONE.value == "none"
        assert GrowthPhase.ANNOUNCEMENT.value == "announcement"
        assert GrowthPhase.WARMUP.value == "warmup"
    
    def test_architecture_upgrade(self):
        """Test ArchitectureUpgrade dataclass."""
        from neuroshard.core.model.dynamic import ArchitectureUpgrade, GrowthPhase
        
        upgrade = ArchitectureUpgrade(
            upgrade_id="upgrade-123",
            target_layers=12,
            target_hidden_dim=512,
        )
        
        assert upgrade.target_layers == 12
        assert upgrade.phase == GrowthPhase.NONE
    
    def test_check_layer_growth_stability(self):
        """Test layer growth stability check."""
        from neuroshard.core.model.dynamic import check_layer_growth, GROWTH_STABILITY_STEPS
        
        # Not enough steps since last growth
        result = check_layer_growth(
            current_layers=11,
            current_hidden_dim=512,
            total_network_memory_mb=100000,
            steps_since_last_growth=1000,  # Too few
            layer_coverage={i: 10 for i in range(11)},
            network_size=100,
        )
        assert result is None
    
    def test_layer_growth_manager(self):
        """Test LayerGrowthManager."""
        from neuroshard.core.model.dynamic import LayerGrowthManager, ArchitectureUpgrade, GrowthPhase
        
        manager = LayerGrowthManager()
        
        upgrade = ArchitectureUpgrade(
            upgrade_id="upgrade-123",
            target_layers=12,
            target_hidden_dim=512,
        )
        
        assert manager.start_upgrade(upgrade)
        assert manager.is_upgrade_in_progress()
        assert manager.get_warmup_lr_multiplier() == 1.0  # Not in warmup yet


# =============================================================================
# NETWORK PHASE TESTS
# =============================================================================

class TestNetworkPhases:
    """Tests for network phase detection."""
    
    def test_network_phase_enum(self):
        """Test NetworkPhase enum values."""
        from neuroshard.core.network.p2p import NetworkPhase
        
        assert NetworkPhase.GENESIS.value == "genesis"
        assert NetworkPhase.MASSIVE.value == "massive"
    
    def test_get_network_phase(self):
        """Test network phase detection by node count."""
        from neuroshard.core.network.p2p import get_network_phase, NetworkPhase
        
        assert get_network_phase(1) == NetworkPhase.GENESIS
        assert get_network_phase(3) == NetworkPhase.MICRO
        assert get_network_phase(10) == NetworkPhase.SMALL
        assert get_network_phase(50) == NetworkPhase.MEDIUM
        assert get_network_phase(200) == NetworkPhase.GROWING
        assert get_network_phase(1000) == NetworkPhase.LARGE
        assert get_network_phase(10000) == NetworkPhase.MASSIVE
    
    def test_adaptive_config(self):
        """Test adaptive configuration by network size."""
        from neuroshard.core.network.p2p import get_adaptive_config
        
        # Genesis config
        config = get_adaptive_config(1)
        assert config.min_quorum_size == 1
        assert config.challenge_rate == 0.0
        
        # Medium config
        config = get_adaptive_config(50)
        assert config.min_quorum_size == 3
        assert config.challenge_rate > 0
    
    def test_network_phase_tracker(self):
        """Test NetworkPhaseTracker."""
        from neuroshard.core.network.p2p import NetworkPhaseTracker, NetworkPhase
        
        tracker = NetworkPhaseTracker()
        
        assert tracker.phase == NetworkPhase.GENESIS
        
        # Update to medium network
        changed = tracker.update(50)
        assert changed
        assert tracker.phase == NetworkPhase.MEDIUM
        
        # Same phase - no change
        changed = tracker.update(60)
        assert not changed
        assert tracker.num_nodes == 60


# =============================================================================
# INTEGRATION TESTS
# =============================================================================

class TestArchitectureV2Integration:
    """Integration tests for ARCHITECTURE_V2 components."""
    
    def test_quorum_with_speed_tiers(self):
        """Test quorum formation with speed tier matching."""
        from neuroshard.core.swarm.quorum import Quorum, QuorumMember, QuorumFormationService, QuorumRegistry
        from neuroshard.core.model.dynamic import SpeedTier, are_tiers_compatible
        
        registry = QuorumRegistry()
        formation = QuorumFormationService(registry)
        
        # Form a quorum
        quorum = formation.form_quorum(
            initiator_node_id="node1",
            initiator_endpoint="localhost:8000",
            initiator_layers=(0, 11),
            initiator_speed_tier="tier2",
            total_layers=11,
        )
        
        assert quorum is not None
        assert quorum.speed_tier == "tier2"
    
    def test_verifier_with_network_size(self):
        """Test verifier adapts to network size."""
        from neuroshard.core.consensus.verifier import ProofVerifier, get_verification_rate
        
        # Small network - verify all
        assert get_verification_rate(10) == 1.0
        
        # Large network - probabilistic
        assert get_verification_rate(1000) < 1.0
    
    def test_adaptive_params_with_quorum(self):
        """Test adaptive parameters affect quorum settings."""
        from neuroshard.core.network.p2p import get_adaptive_config
        
        # Small network - allow small quorums
        config = get_adaptive_config(5)
        assert config.min_quorum_size <= 2
        
        # Large network - require larger quorums
        config = get_adaptive_config(1000)
        assert config.min_quorum_size >= 3


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
