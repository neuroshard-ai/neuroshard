"""
NEP Voting - Stake-Weighted Governance

Voting power is proportional to staked NEURO. This ensures that
participants with skin in the game make decisions about protocol changes.

Voting Rules:
- 1 NEURO staked = 1 vote
- Voting period: 7 days (configurable per NEP)
- Approval threshold: 66% of votes (stake-weighted)
- Quorum: 20% of total staked NEURO must vote
"""

import sqlite3
import time
import hashlib
import json
import logging
import threading
from dataclasses import dataclass, asdict
from enum import Enum
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


class VoteChoice(Enum):
    """Voting options."""
    YES = "yes"           # Approve the proposal
    NO = "no"             # Reject the proposal
    ABSTAIN = "abstain"   # Participate without opinion (counts for quorum)


@dataclass
class Vote:
    """A single vote on a NEP."""
    vote_id: str
    nep_id: str
    voter_node_id: str
    choice: VoteChoice
    stake_at_vote: float      # Stake when vote was cast (locked)
    timestamp: float
    signature: str            # ECDSA signature
    reason: str = ""          # Optional reason for vote
    
    def to_dict(self) -> Dict:
        return {
            "vote_id": self.vote_id,
            "nep_id": self.nep_id,
            "voter_node_id": self.voter_node_id,
            "choice": self.choice.value,
            "stake_at_vote": self.stake_at_vote,
            "timestamp": self.timestamp,
            "signature": self.signature,
            "reason": self.reason,
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'Vote':
        return cls(
            vote_id=data["vote_id"],
            nep_id=data["nep_id"],
            voter_node_id=data["voter_node_id"],
            choice=VoteChoice(data["choice"]),
            stake_at_vote=data["stake_at_vote"],
            timestamp=data["timestamp"],
            signature=data["signature"],
            reason=data.get("reason", ""),
        )


@dataclass
class VoteResult:
    """Aggregated voting results for a NEP."""
    nep_id: str
    
    # Vote counts (number of voters)
    yes_count: int = 0
    no_count: int = 0
    abstain_count: int = 0
    
    # Stake-weighted votes
    yes_stake: float = 0.0
    no_stake: float = 0.0
    abstain_stake: float = 0.0
    
    # Thresholds
    total_stake_voted: float = 0.0
    total_network_stake: float = 0.0
    approval_threshold: float = 0.66
    quorum_threshold: float = 0.20
    
    # Status
    quorum_reached: bool = False
    approved: bool = False
    
    @property
    def participation_rate(self) -> float:
        """Percentage of network stake that voted."""
        if self.total_network_stake == 0:
            return 0.0
        return self.total_stake_voted / self.total_network_stake
    
    @property
    def approval_rate(self) -> float:
        """Percentage of voting stake that approved."""
        yes_no_total = self.yes_stake + self.no_stake
        if yes_no_total == 0:
            return 0.0
        return self.yes_stake / yes_no_total
    
    def to_dict(self) -> Dict:
        return {
            "nep_id": self.nep_id,
            "yes_count": self.yes_count,
            "no_count": self.no_count,
            "abstain_count": self.abstain_count,
            "yes_stake": self.yes_stake,
            "no_stake": self.no_stake,
            "abstain_stake": self.abstain_stake,
            "total_stake_voted": self.total_stake_voted,
            "total_network_stake": self.total_network_stake,
            "participation_rate": self.participation_rate,
            "approval_rate": self.approval_rate,
            "quorum_reached": self.quorum_reached,
            "approved": self.approved,
        }


class VotingSystem:
    """
    Manages NEP voting with stake-weighted governance.
    """
    
    def __init__(
        self,
        db_path: str = "nep_votes.db",
        ledger=None,  # NEUROLedger for stake lookups
    ):
        self.db_path = db_path
        self.ledger = ledger
        self.lock = threading.Lock()
        
        self._init_db()
    
    def _init_db(self):
        """Initialize voting database."""
        with self.lock:
            with sqlite3.connect(self.db_path, timeout=60.0) as conn:
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS votes (
                        vote_id TEXT PRIMARY KEY,
                        nep_id TEXT NOT NULL,
                        voter_node_id TEXT NOT NULL,
                        choice TEXT NOT NULL,
                        stake_at_vote REAL NOT NULL,
                        timestamp REAL NOT NULL,
                        signature TEXT NOT NULL,
                        reason TEXT DEFAULT '',
                        UNIQUE(nep_id, voter_node_id)
                    )
                """)
                
                conn.execute("""
                    CREATE INDEX IF NOT EXISTS idx_votes_nep 
                    ON votes(nep_id)
                """)
    
    def cast_vote(
        self,
        nep_id: str,
        voter_node_id: str,
        choice: VoteChoice,
        signature: str,
        reason: str = "",
        nep_registry=None,
    ) -> Tuple[bool, str]:
        """
        Cast a vote on a NEP.
        
        Requirements:
        1. NEP must be in VOTING status
        2. Voting period must be active
        3. Voter must have stake > 0
        4. Vote must be signed
        
        Returns: (success, message)
        """
        # Check NEP status
        if nep_registry:
            nep = nep_registry.get_proposal(nep_id)
            if not nep:
                return False, "NEP not found"
            if not nep.is_voting_active():
                return False, f"Voting not active for {nep_id}"
        
        # Get voter's stake
        stake = 0.0
        if self.ledger:
            stake = self.ledger.get_local_stake(voter_node_id)
            if stake <= 0:
                return False, "Must have staked NEURO to vote"
        else:
            # Test mode - assume 100 stake
            stake = 100.0
        
        # Verify ECDSA signature before accepting the vote
        try:
            from neuroshard.core.crypto.ecdsa import verify_signature
            vote_payload = f"{nep_id}:{voter_node_id}:{choice.value}:{reason}"
            if not verify_signature(voter_node_id, vote_payload, signature):
                return False, "Invalid vote signature — ECDSA verification failed"
        except ImportError:
            logger.warning("ECDSA module unavailable — skipping vote signature verification")
        except Exception as e:
            logger.warning(f"Vote signature verification error: {e}")
            return False, f"Signature verification error: {e}"
        
        # Check if already voted
        with sqlite3.connect(self.db_path, timeout=60.0) as conn:
            existing = conn.execute(
                "SELECT vote_id FROM votes WHERE nep_id = ? AND voter_node_id = ?",
                (nep_id, voter_node_id)
            ).fetchone()
            
            if existing:
                return False, "Already voted on this NEP"
        
        # Create vote
        vote_id = hashlib.sha256(
            f"{nep_id}:{voter_node_id}:{time.time()}".encode()
        ).hexdigest()[:32]
        
        vote = Vote(
            vote_id=vote_id,
            nep_id=nep_id,
            voter_node_id=voter_node_id,
            choice=choice,
            stake_at_vote=stake,
            timestamp=time.time(),
            signature=signature,
            reason=reason,
        )
        
        # Store vote
        with self.lock:
            with sqlite3.connect(self.db_path, timeout=60.0) as conn:
                conn.execute("""
                    INSERT INTO votes 
                    (vote_id, nep_id, voter_node_id, choice, stake_at_vote, 
                     timestamp, signature, reason)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    vote.vote_id,
                    vote.nep_id,
                    vote.voter_node_id,
                    vote.choice.value,
                    vote.stake_at_vote,
                    vote.timestamp,
                    vote.signature,
                    vote.reason,
                ))
        
        logger.info(
            f"Vote cast: {voter_node_id[:16]}... voted {choice.value} "
            f"on {nep_id} with {stake:.2f} stake"
        )
        
        return True, f"Vote recorded: {choice.value} with {stake:.2f} stake"
    
    def get_vote_tally(
        self,
        nep_id: str,
        approval_threshold: float = 0.66,
        quorum_threshold: float = 0.20,
    ) -> VoteResult:
        """
        Get the current vote tally for a NEP.
        """
        result = VoteResult(
            nep_id=nep_id,
            approval_threshold=approval_threshold,
            quorum_threshold=quorum_threshold,
        )
        
        with sqlite3.connect(self.db_path, timeout=60.0) as conn:
            votes = conn.execute(
                "SELECT choice, stake_at_vote FROM votes WHERE nep_id = ?",
                (nep_id,)
            ).fetchall()
            
            for choice_str, stake in votes:
                choice = VoteChoice(choice_str)
                
                if choice == VoteChoice.YES:
                    result.yes_count += 1
                    result.yes_stake += stake
                elif choice == VoteChoice.NO:
                    result.no_count += 1
                    result.no_stake += stake
                else:
                    result.abstain_count += 1
                    result.abstain_stake += stake
        
        result.total_stake_voted = (
            result.yes_stake + result.no_stake + result.abstain_stake
        )
        
        # Get total network stake
        if self.ledger:
            # Query total staked from ledger
            result.total_network_stake = self._get_total_network_stake()
        else:
            # Test mode - assume voted stake is 50% of network
            result.total_network_stake = result.total_stake_voted * 2
        
        # Check quorum
        if result.total_network_stake > 0:
            result.quorum_reached = (
                result.participation_rate >= quorum_threshold
            )
        
        # Check approval (only if quorum reached)
        if result.quorum_reached:
            result.approved = result.approval_rate >= approval_threshold
        
        return result
    
    def _get_total_network_stake(self) -> float:
        """Get total staked NEURO in the network."""
        if not self.ledger:
            return 0.0
        
        try:
            with sqlite3.connect(self.ledger.db_path, timeout=60.0) as conn:
                row = conn.execute(
                    "SELECT SUM(amount) FROM stakes WHERE locked_until > ?",
                    (time.time(),)
                ).fetchone()
                return row[0] if row and row[0] else 0.0
        except Exception as e:
            logger.error(f"Failed to get network stake: {e}")
            return 0.0
    
    def get_voter_history(self, voter_node_id: str) -> List[Vote]:
        """Get all votes by a node."""
        with sqlite3.connect(self.db_path, timeout=60.0) as conn:
            rows = conn.execute(
                "SELECT vote_id, nep_id, voter_node_id, choice, "
                "stake_at_vote, timestamp, signature, reason "
                "FROM votes WHERE voter_node_id = ? ORDER BY timestamp DESC",
                (voter_node_id,)
            ).fetchall()
            
            return [
                Vote(
                    vote_id=row[0],
                    nep_id=row[1],
                    voter_node_id=row[2],
                    choice=VoteChoice(row[3]),
                    stake_at_vote=row[4],
                    timestamp=row[5],
                    signature=row[6],
                    reason=row[7],
                )
                for row in rows
            ]
    
    def finalize_voting(
        self,
        nep_id: str,
        nep_registry,
    ) -> Tuple[bool, str, VoteResult]:
        """
        Finalize voting and update NEP status.
        
        Called when voting period ends.
        Returns: (success, message, final_result)
        """
        nep = nep_registry.get_proposal(nep_id)
        if not nep:
            return False, "NEP not found", None
        
        # Get final tally
        result = self.get_vote_tally(
            nep_id,
            approval_threshold=nep.approval_threshold,
            quorum_threshold=nep.quorum_threshold,
        )
        
        if result.approved:
            # Calculate activation block (e.g., 10000 blocks from now)
            activation_block = self._get_current_block() + nep.upgrade_path.activation_delay_blocks
            
            nep_registry.update_status(
                nep_id,
                NEPStatus.SCHEDULED,
                activation_block=activation_block,
            )
            
            message = (
                f"Approved! Activation scheduled at block {activation_block}. "
                f"Approval: {result.approval_rate:.1%}, "
                f"Participation: {result.participation_rate:.1%}"
            )
            
        elif result.quorum_reached:
            nep_registry.update_status(nep_id, NEPStatus.REJECTED)
            message = (
                f"Rejected. Approval: {result.approval_rate:.1%} "
                f"(needed {nep.approval_threshold:.1%})"
            )
            
        else:
            nep_registry.update_status(nep_id, NEPStatus.REJECTED)
            message = (
                f"Failed to reach quorum. Participation: {result.participation_rate:.1%} "
                f"(needed {nep.quorum_threshold:.1%})"
            )
        
        logger.info(f"Voting finalized for {nep_id}: {message}")
        
        return True, message, result
    
    def _get_current_block(self) -> int:
        """
        Get current block number.
        
        In a true blockchain, this would query the chain.
        For NeuroShard, we use a logical block based on proof count.
        """
        if self.ledger:
            try:
                stats = self.ledger.get_global_stats()
                # Use proof count as "block height"
                return stats.total_proofs_processed
            except:
                pass
        return int(time.time() / 60)  # Fallback: 1 "block" per minute


# Module-level convenience functions
from .proposal import NEPStatus

def cast_vote(
    voting_system: VotingSystem,
    nep_id: str,
    voter_node_id: str,
    choice: VoteChoice,
    signature: str,
    reason: str = "",
    nep_registry=None,
) -> Tuple[bool, str]:
    """Cast a vote on a NEP."""
    return voting_system.cast_vote(
        nep_id, voter_node_id, choice, signature, reason, nep_registry
    )


def get_vote_tally(
    voting_system: VotingSystem,
    nep_id: str,
) -> VoteResult:
    """Get current vote tally for a NEP."""
    return voting_system.get_vote_tally(nep_id)
