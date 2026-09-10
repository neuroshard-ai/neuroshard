"""
Test Hybrid Validator Consensus System

Tests the validator consensus implementation from the whitepaper (Section 7).

WHAT WE'RE TESTING:
==================
1. Validator registration and eligibility
2. Stake-weighted validator selection
3. Vote casting and consensus calculation
4. 66% threshold determination
5. Validator slashing for wrong votes
6. Dynamic stake requirements
7. Multiple validators scenario

From whitepaper Section 7 (Hybrid Validator System):
- Validators = nodes with Last Layer + LM Head + 100 NEURO staked
- 3 validators selected per proof (stake-weighted random)
- 66% stake threshold for consensus
- Validators earn 0.001 NEURO per proof validated
- Bad validators slashed at 2x rate
"""
import time
from unittest.mock import MagicMock


class TestValidatorEligibility:
    """Test validator eligibility requirements."""
    
    def test_eligible_validator(self):
        """A node with LM head + sufficient stake + memory is eligible."""
        from neuroshard.core.consensus.validator_consensus import ValidatorInfo
        from neuroshard.core.economics.constants import VALIDATOR_MIN_STAKE, VALIDATOR_MIN_MEMORY_MB
        
        validator = ValidatorInfo(
            node_id="validator_001",
            stake=VALIDATOR_MIN_STAKE + 50,  # More than required
            has_lm_head=True,
            memory_mb=VALIDATOR_MIN_MEMORY_MB + 1000,  # More than required
            url="http://validator:9000"
        )
        
        assert validator.is_eligible is True
        print(f"✓ Validator with stake={validator.stake}, has_lm_head=True, memory={validator.memory_mb}MB is ELIGIBLE")
    
    def test_ineligible_no_lm_head(self):
        """A node without LM head is NOT eligible."""
        from neuroshard.core.consensus.validator_consensus import ValidatorInfo
        from neuroshard.core.economics.constants import VALIDATOR_MIN_STAKE, VALIDATOR_MIN_MEMORY_MB
        
        validator = ValidatorInfo(
            node_id="worker_001",
            stake=VALIDATOR_MIN_STAKE + 50,
            has_lm_head=False,  # No LM head!
            memory_mb=VALIDATOR_MIN_MEMORY_MB + 1000,
            url="http://worker:9000"
        )
        
        assert validator.is_eligible is False
        print("✓ Node without LM head is NOT eligible")
    
    def test_ineligible_insufficient_stake(self):
        """A node with insufficient stake is NOT eligible."""
        from neuroshard.core.consensus.validator_consensus import ValidatorInfo
        from neuroshard.core.economics.constants import VALIDATOR_MIN_STAKE, VALIDATOR_MIN_MEMORY_MB
        
        validator = ValidatorInfo(
            node_id="poor_validator",
            stake=VALIDATOR_MIN_STAKE - 50,  # Below required!
            has_lm_head=True,
            memory_mb=VALIDATOR_MIN_MEMORY_MB + 1000,
            url="http://validator:9000"
        )
        
        assert validator.is_eligible is False
        print("✓ Node with insufficient stake is NOT eligible")
    
    def test_ineligible_insufficient_memory(self):
        """A node with insufficient memory is NOT eligible."""
        from neuroshard.core.consensus.validator_consensus import ValidatorInfo
        from neuroshard.core.economics.constants import VALIDATOR_MIN_STAKE, VALIDATOR_MIN_MEMORY_MB
        
        validator = ValidatorInfo(
            node_id="small_validator",
            stake=VALIDATOR_MIN_STAKE + 50,
            has_lm_head=True,
            memory_mb=VALIDATOR_MIN_MEMORY_MB - 500,  # Below required!
            url="http://validator:9000"
        )
        
        assert validator.is_eligible is False
        print("✓ Node with insufficient memory is NOT eligible")


class TestValidatorSelection:
    """Test stake-weighted validator selection."""
    
    def test_selection_prefers_higher_stake(self):
        """Higher stake = higher chance of selection."""
        from neuroshard.core.consensus.validator_consensus import ValidatorConsensus, ValidatorInfo
        from neuroshard.core.economics.constants import VALIDATOR_MIN_STAKE, VALIDATOR_MIN_MEMORY_MB
        
        consensus = ValidatorConsensus()
        
        # Register validators with different stakes
        consensus.register_validator("high_stake", stake=1000.0, has_lm_head=True, 
                                    memory_mb=4000, url="http://high:9000")
        consensus.register_validator("med_stake", stake=500.0, has_lm_head=True,
                                    memory_mb=4000, url="http://med:9000")
        consensus.register_validator("low_stake", stake=150.0, has_lm_head=True,
                                    memory_mb=4000, url="http://low:9000")
        
        # Run selection multiple times and count
        selection_counts = {"high_stake": 0, "med_stake": 0, "low_stake": 0}
        for _ in range(100):
            selected = consensus.select_validators(exclude_node_id="submitter", num_validators=1)
            if selected:
                selection_counts[selected[0].node_id] += 1
        
        # High stake should be selected most often
        assert selection_counts["high_stake"] > selection_counts["low_stake"]
        print(f"✓ Selection counts: high={selection_counts['high_stake']}, "
              f"med={selection_counts['med_stake']}, low={selection_counts['low_stake']}")
        print("✓ Higher stake validators are selected more often")
    
    def test_selection_includes_randomness(self):
        """Selection has randomness component (30%)."""
        from neuroshard.core.consensus.validator_consensus import ValidatorConsensus
        
        consensus = ValidatorConsensus()
        
        # Register validators
        consensus.register_validator("v1", stake=1000.0, has_lm_head=True, 
                                    memory_mb=4000, url="http://v1:9000")
        consensus.register_validator("v2", stake=1000.0, has_lm_head=True,
                                    memory_mb=4000, url="http://v2:9000")
        consensus.register_validator("v3", stake=1000.0, has_lm_head=True,
                                    memory_mb=4000, url="http://v3:9000")
        
        # With equal stakes, selection should vary due to randomness
        results = set()
        for _ in range(20):
            selected = consensus.select_validators(exclude_node_id="submitter", num_validators=1)
            if selected:
                results.add(selected[0].node_id)
        
        # Should see multiple different validators selected
        assert len(results) > 1
        print(f"✓ Different validators selected over time: {results}")
        print("✓ Randomness component (30%) ensures variety")


class TestConsensusVoting:
    """Test consensus voting and threshold."""
    
    def test_consensus_66_threshold_accept(self):
        """Proof accepted when 66%+ stake votes valid."""
        from neuroshard.core.consensus.validator_consensus import (
            ValidatorConsensus, ConsensusState
        )
        
        consensus = ValidatorConsensus()
        
        # Register validators with stakes that will produce 66.7% ratio
        consensus.register_validator("v1", stake=400.0, has_lm_head=True,
                                    memory_mb=4000, url="http://v1:9000")
        consensus.register_validator("v2", stake=200.0, has_lm_head=True,
                                    memory_mb=4000, url="http://v2:9000")
        
        # Submit proof
        consensus.submit_proof_for_consensus(
            proof_signature="test_proof_001",
            submitter_node_id="submitter"
        )
        
        # Cast invalid vote FIRST (smaller stake)
        consensus.cast_vote("test_proof_001", vote=False, validator_id="v2", stake=200.0)
        # Then valid vote (larger stake) - reaches 66.7% threshold
        consensus.cast_vote("test_proof_001", vote=True, validator_id="v1", stake=400.0)
        
        # Check result: 400/600 = 66.7% > 66% → ACCEPTED
        result = consensus.get_consensus_status("test_proof_001")
        
        assert result.consensus_reached is True
        assert result.consensus_result is True  # ACCEPTED
        assert result.valid_ratio >= 0.66
        
        print(f"✓ Consensus reached: {result.valid_ratio:.1%} valid stake → ACCEPTED")
    
    def test_consensus_66_threshold_reject(self):
        """Proof rejected when 66%+ stake votes invalid."""
        from neuroshard.core.consensus.validator_consensus import (
            ValidatorConsensus, ConsensusState
        )
        
        consensus = ValidatorConsensus()
        
        # Register validators
        consensus.register_validator("v1", stake=400.0, has_lm_head=True,
                                    memory_mb=4000, url="http://v1:9000")
        consensus.register_validator("v2", stake=200.0, has_lm_head=True,
                                    memory_mb=4000, url="http://v2:9000")
        
        # Submit proof
        consensus.submit_proof_for_consensus(
            proof_signature="test_proof_002",
            submitter_node_id="submitter"
        )
        
        # Cast valid vote FIRST (smaller stake)
        consensus.cast_vote("test_proof_002", vote=True, validator_id="v2", stake=200.0)
        # Then invalid vote (larger stake) - reaches 66.7% invalid threshold
        consensus.cast_vote("test_proof_002", vote=False, validator_id="v1", stake=400.0)
        
        # Check result: 400/600 = 66.7% invalid > 66% → REJECTED
        result = consensus.get_consensus_status("test_proof_002")
        
        assert result.consensus_reached is True
        assert result.consensus_result is False  # REJECTED
        assert (1 - result.valid_ratio) >= 0.66
        
        print(f"✓ Consensus reached: {1 - result.valid_ratio:.1%} invalid stake → REJECTED")
    
    def test_no_consensus_below_threshold(self):
        """No consensus when neither side has 66%."""
        from neuroshard.core.consensus.validator_consensus import (
            ValidatorConsensus, ConsensusState
        )
        
        consensus = ValidatorConsensus()
        
        # Register validators with equal stake
        consensus.register_validator("v1", stake=500.0, has_lm_head=True,
                                    memory_mb=4000, url="http://v1:9000")
        consensus.register_validator("v2", stake=500.0, has_lm_head=True,
                                    memory_mb=4000, url="http://v2:9000")
        
        # Submit proof
        consensus.submit_proof_for_consensus(
            proof_signature="test_proof_003",
            submitter_node_id="submitter"
        )
        
        # Cast votes: 50/50 split - no consensus!
        consensus.cast_vote("test_proof_003", vote=True, validator_id="v1", stake=500.0)
        consensus.cast_vote("test_proof_003", vote=False, validator_id="v2", stake=500.0)
        
        result = consensus.get_consensus_status("test_proof_003")
        
        # 50/50 split = neither side has 66%
        assert result.valid_ratio == 0.5
        assert result.consensus_reached is False
        assert result.consensus_result is None
        assert result.state == ConsensusState.COLLECTING_VOTES
        
        print(f"✓ No consensus: {result.valid_ratio:.1%} valid stake (need 66%)")


class TestValidatorSlashing:
    """Test validator slashing for voting against consensus."""
    
    def test_validator_slashed_for_wrong_vote(self):
        """Validator who votes against consensus is slashed."""
        from neuroshard.core.consensus.validator_consensus import (
            ValidatorConsensus, ConsensusState
        )
        
        consensus = ValidatorConsensus()
        
        # Register validators with stakes that show slashing behavior
        consensus.register_validator("v1", stake=500.0, has_lm_head=True,
                                    memory_mb=4000, url="http://v1:9000")
        consensus.register_validator("v2", stake=100.0, has_lm_head=True,
                                    memory_mb=4000, url="http://v2:9000")
        
        # Submit proof
        consensus.submit_proof_for_consensus(
            proof_signature="test_proof_004",
            submitter_node_id="submitter"
        )
        
        # v2 (100) votes INVALID first
        # v1 (500) votes VALID second → consensus VALID (500/600 = 83.3%)
        # v2 voted against consensus → should be slashed!
        consensus.cast_vote("test_proof_004", vote=False, validator_id="v2", stake=100.0)
        consensus.cast_vote("test_proof_004", vote=True, validator_id="v1", stake=500.0)
        
        result = consensus.get_consensus_status("test_proof_004")
        assert result.consensus_reached is True
        assert result.consensus_result is True  # ACCEPTED
        assert "v2" in result.validators_slashed
        
        print(f"✓ Validator v2 SLASHED for voting against consensus")
        print(f"  - v1 (500) voted VALID ✓")
        print(f"  - v2 (100) voted INVALID → SLASHED (2x penalty)")


class TestDynamicStakeRequirements:
    """Test dynamic stake requirements based on network size."""
    
    def test_bootstrap_phase_zero_stake(self):
        """During bootstrap (0-2 validators), stake requirement is 0."""
        from neuroshard.core.economics.constants import get_dynamic_validator_stake
        
        assert get_dynamic_validator_stake(0) == 0.0
        assert get_dynamic_validator_stake(1) == 0.0
        assert get_dynamic_validator_stake(2) == 0.0
        print("✓ Bootstrap (0-2 validators): 0 NEURO required")
    
    def test_early_phase_stake(self):
        """Early phase (3-10 validators): 100 NEURO required."""
        from neuroshard.core.economics.constants import get_dynamic_validator_stake
        
        assert get_dynamic_validator_stake(3) == 100.0
        assert get_dynamic_validator_stake(10) == 100.0
        print("✓ Early (3-10 validators): 100 NEURO required")
    
    def test_growing_phase_stake(self):
        """Growing phase (11-50 validators): 250 NEURO required."""
        from neuroshard.core.economics.constants import get_dynamic_validator_stake
        
        assert get_dynamic_validator_stake(11) == 250.0
        assert get_dynamic_validator_stake(50) == 250.0
        print("✓ Growing (11-50 validators): 250 NEURO required")
    
    def test_large_scale_stake(self):
        """Large scale (1000+ validators): 2500 NEURO required."""
        from neuroshard.core.economics.constants import get_dynamic_validator_stake
        
        assert get_dynamic_validator_stake(1001) == 2500.0
        assert get_dynamic_validator_stake(5000) == 2500.0
        print("✓ Large scale (1000+ validators): 2500 NEURO required")


class TestMultipleValidatorsScenario:
    """Test realistic multi-validator scenario."""
    
    def test_three_validator_consensus(self):
        """Standard 3-validator consensus flow."""
        from neuroshard.core.consensus.validator_consensus import ValidatorConsensus
        
        consensus = ValidatorConsensus(num_validators_per_proof=3)
        
        # Register 5 validators (3 will be selected)
        for i in range(5):
            consensus.register_validator(
                f"validator_{i}",
                stake=100.0 + i * 50,  # 100, 150, 200, 250, 300
                has_lm_head=True,
                memory_mb=4000,
                url=f"http://validator{i}:9000"
            )
        
        # Check eligible count
        eligible = consensus.get_eligible_validators()
        assert len(eligible) == 5
        
        # Select validators for a proof
        selected = consensus.select_validators(exclude_node_id="submitter")
        assert len(selected) == 3
        
        print(f"✓ 5 validators registered, 3 selected for proof validation")
        print(f"  Selected: {[v.node_id for v in selected]}")
        print(f"  Total stake: {sum(v.stake for v in selected):.2f} NEURO")
    
    def test_bootstrap_auto_accept(self):
        """During bootstrap (< 2 validators), proofs auto-accept."""
        from neuroshard.core.consensus.validator_consensus import (
            ValidatorConsensus, ConsensusState
        )
        
        consensus = ValidatorConsensus()
        
        # No validators registered (bootstrap)
        assert consensus.get_validator_count() == 0
        assert consensus.should_use_consensus() is False
        
        # Submit proof - should auto-accept
        result = consensus.submit_proof_for_consensus(
            proof_signature="bootstrap_proof",
            submitter_node_id="solo_node"
        )
        
        assert result.state == ConsensusState.ACCEPTED
        assert result.consensus_result is True
        assert result.reward_credited is True
        
        print("✓ Bootstrap mode: Proof auto-accepted (no validators available)")


if __name__ == "__main__":
    print("\n" + "="*70)
    print("HYBRID VALIDATOR CONSENSUS TESTS")
    print("From whitepaper Section 7")
    print("="*70 + "\n")
    
    print("--- Validator Eligibility ---")
    t1 = TestValidatorEligibility()
    t1.test_eligible_validator()
    t1.test_ineligible_no_lm_head()
    t1.test_ineligible_insufficient_stake()
    t1.test_ineligible_insufficient_memory()
    
    print("\n--- Validator Selection (Stake-Weighted) ---")
    t2 = TestValidatorSelection()
    t2.test_selection_prefers_higher_stake()
    t2.test_selection_includes_randomness()
    
    print("\n--- Consensus Voting (66% Threshold) ---")
    t3 = TestConsensusVoting()
    t3.test_consensus_66_threshold_accept()
    t3.test_consensus_66_threshold_reject()
    t3.test_no_consensus_below_threshold()
    
    print("\n--- Validator Slashing ---")
    t4 = TestValidatorSlashing()
    t4.test_validator_slashed_for_wrong_vote()
    
    print("\n--- Dynamic Stake Requirements ---")
    t5 = TestDynamicStakeRequirements()
    t5.test_bootstrap_phase_zero_stake()
    t5.test_early_phase_stake()
    t5.test_growing_phase_stake()
    t5.test_large_scale_stake()
    
    print("\n--- Multiple Validators Scenario ---")
    t6 = TestMultipleValidatorsScenario()
    t6.test_three_validator_consensus()
    t6.test_bootstrap_auto_accept()
    
    print("\n" + "="*70)
    print("ALL VALIDATOR CONSENSUS TESTS PASSED ✓")
    print("="*70 + "\n")
