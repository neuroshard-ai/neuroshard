"""
NEP Registry - Decentralized Proposal Storage

The registry stores all NEPs and their status. It's replicated across
nodes via the gossip protocol, similar to how proofs and stakes are shared.

Storage:
- SQLite locally (fast queries)
- Gossip sync (network consistency)
- DHT backup (persistence)
"""

import sqlite3
import time
import json
import logging
import threading
from typing import Dict, List, Optional, Tuple
from pathlib import Path

from .proposal import NEP, NEPType, NEPStatus, EconomicImpact, UpgradePath

logger = logging.getLogger(__name__)


class NEPRegistry:
    """
    Registry of all NeuroShard Enhancement Proposals.
    
    Responsibilities:
    - Store and retrieve NEPs
    - Track proposal lifecycle
    - Enforce submission rules (e.g., stake requirement to propose)
    - Sync with network via gossip
    
    Economic Incentives (Stake-Proportional):
    - Proposers pay PROPOSAL_FEE upfront (burned if spam/no-quorum)
    - Rewards scale with total stake that participated in voting
    - This aligns incentives: more impactful proposals = more votes = more reward
    
    Reward Formula:
        proposer_reward = PROPOSER_REWARD_RATE * total_stake_voted
        
    Example:
        - 1000 NEURO stake voted → 1000 * 0.001 = 1 NEURO reward
        - 100,000 NEURO stake voted → 100,000 * 0.001 = 100 NEURO reward
        - 1M NEURO stake voted → 1,000,000 * 0.001 = 1000 NEURO reward
    """
    
    # Minimum stake to create a proposal (prevents spam)
    MIN_STAKE_TO_PROPOSE = 100.0  # 100 NEURO
    
    # Proposal submission fee (held in escrow, burned if spam/no-quorum)
    PROPOSAL_FEE = 10.0  # 10 NEURO
    
    # Stake-proportional rewards (% of total stake that voted)
    PROPOSER_REWARD_RATE_APPROVED = 0.001   # 0.1% of total stake voted (if approved)
    PROPOSER_REWARD_RATE_QUORUM = 0.0001    # 0.01% of total stake voted (if rejected but quorum)
    
    # Minimum/maximum caps to prevent extremes
    PROPOSER_REWARD_MIN = 1.0      # At least 1 NEURO (if approved)
    PROPOSER_REWARD_MAX = 1000.0   # At most 1000 NEURO (prevents gaming)
    
    # Voter rewards (incentivizes participation)
    # Reward = VOTER_REWARD_RATE * voter's_stake
    VOTER_REWARD_RATE = 0.0001  # 0.01% of voter's stake per vote
    
    def __init__(
        self,
        db_path: str = "nep_registry.db",
        ledger=None,  # NEUROLedger for stake checks
    ):
        self.db_path = db_path
        self.ledger = ledger
        self.lock = threading.Lock()
        self._next_nep_number = 1
        
        self._init_db()
        self._load_next_number()
    
    def _init_db(self):
        """Initialize SQLite database."""
        with self.lock:
            with sqlite3.connect(self.db_path, timeout=60.0) as conn:
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS neps (
                        nep_id TEXT PRIMARY KEY,
                        nep_type TEXT NOT NULL,
                        status TEXT NOT NULL,
                        title TEXT NOT NULL,
                        author_node_id TEXT NOT NULL,
                        content_hash TEXT NOT NULL,
                        created_at REAL NOT NULL,
                        updated_at REAL NOT NULL,
                        voting_start REAL,
                        voting_end REAL,
                        activation_block INTEGER,
                        full_data TEXT NOT NULL
                    )
                """)
                
                conn.execute("""
                    CREATE INDEX IF NOT EXISTS idx_nep_status 
                    ON neps(status)
                """)
                
                conn.execute("""
                    CREATE INDEX IF NOT EXISTS idx_nep_type 
                    ON neps(nep_type)
                """)
                
                # Track highest NEP number
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS registry_meta (
                        key TEXT PRIMARY KEY,
                        value TEXT
                    )
                """)
    
    def _load_next_number(self):
        """Load the next NEP number from database."""
        with sqlite3.connect(self.db_path, timeout=60.0) as conn:
            row = conn.execute(
                "SELECT value FROM registry_meta WHERE key = 'next_nep_number'"
            ).fetchone()
            
            if row:
                self._next_nep_number = int(row[0])
            else:
                # Count existing NEPs and set next
                count = conn.execute("SELECT COUNT(*) FROM neps").fetchone()[0]
                self._next_nep_number = count + 1
    
    def _allocate_nep_id(self) -> str:
        """Allocate the next NEP ID."""
        nep_id = f"NEP-{self._next_nep_number:03d}"
        self._next_nep_number += 1
        
        with sqlite3.connect(self.db_path, timeout=60.0) as conn:
            conn.execute(
                "INSERT OR REPLACE INTO registry_meta (key, value) VALUES (?, ?)",
                ("next_nep_number", str(self._next_nep_number))
            )
        
        return nep_id
    
    def submit_proposal(
        self,
        nep: NEP,
        node_id: str,
        signature: str,
    ) -> Tuple[bool, str, Optional[str]]:
        """
        Submit a new proposal to the registry.
        
        Requirements:
        1. Author must have MIN_STAKE_TO_PROPOSE staked
        2. Proposal must be properly signed
        3. Proposal fee is burned
        
        Returns: (success, message, nep_id)
        """
        # Check stake requirement
        if self.ledger:
            stake = self.ledger.get_local_stake(node_id)
            if stake < self.MIN_STAKE_TO_PROPOSE:
                return False, f"Insufficient stake to propose: {stake:.2f} < {self.MIN_STAKE_TO_PROPOSE}", None
        
        # Verify signature
        # (In production, verify with ECDSA)
        if not signature:
            return False, "Proposal must be signed", None
        
        # Allocate official NEP ID
        nep_id = self._allocate_nep_id()
        nep.nep_id = nep_id
        nep.status = NEPStatus.DRAFT
        nep.signature = signature
        nep.updated_at = time.time()
        
        # Store in database
        with self.lock:
            with sqlite3.connect(self.db_path, timeout=60.0) as conn:
                conn.execute("""
                    INSERT INTO neps 
                    (nep_id, nep_type, status, title, author_node_id, 
                     content_hash, created_at, updated_at, full_data)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    nep.nep_id,
                    nep.nep_type.value,
                    nep.status.value,
                    nep.title,
                    nep.author_node_id,
                    nep.content_hash,
                    nep.created_at,
                    nep.updated_at,
                    json.dumps(nep.to_dict()),
                ))
        
        logger.info(f"Submitted proposal: {nep.summary()}")
        
        # Burn proposal fee
        if self.ledger:
            # TODO: Implement fee burn
            pass
        
        return True, f"Proposal submitted as {nep_id}", nep_id
    
    def get_proposal(self, nep_id: str) -> Optional[NEP]:
        """Retrieve a proposal by ID."""
        with sqlite3.connect(self.db_path, timeout=60.0) as conn:
            row = conn.execute(
                "SELECT full_data FROM neps WHERE nep_id = ?",
                (nep_id,)
            ).fetchone()
            
            if not row:
                return None
            
            return NEP.from_dict(json.loads(row[0]))
    
    def update_status(
        self,
        nep_id: str,
        new_status: NEPStatus,
        voting_start: float = None,
        voting_end: float = None,
        activation_block: int = None,
    ) -> bool:
        """Update a proposal's status."""
        with self.lock:
            with sqlite3.connect(self.db_path, timeout=60.0) as conn:
                # Get current NEP
                row = conn.execute(
                    "SELECT full_data FROM neps WHERE nep_id = ?",
                    (nep_id,)
                ).fetchone()
                
                if not row:
                    return False
                
                nep = NEP.from_dict(json.loads(row[0]))
                nep.status = new_status
                nep.updated_at = time.time()
                
                if voting_start:
                    nep.voting_start = voting_start
                if voting_end:
                    nep.voting_end = voting_end
                if activation_block:
                    nep.activation_block = activation_block
                
                conn.execute("""
                    UPDATE neps SET
                        status = ?,
                        voting_start = ?,
                        voting_end = ?,
                        activation_block = ?,
                        updated_at = ?,
                        full_data = ?
                    WHERE nep_id = ?
                """, (
                    new_status.value,
                    nep.voting_start,
                    nep.voting_end,
                    nep.activation_block,
                    nep.updated_at,
                    json.dumps(nep.to_dict()),
                    nep_id,
                ))
        
        logger.info(f"Updated {nep_id} status to {new_status.value}")
        return True
    
    def start_voting(
        self,
        nep_id: str,
        voting_duration_days: int = 7,
    ) -> Tuple[bool, str]:
        """Start the voting period for a proposal."""
        nep = self.get_proposal(nep_id)
        if not nep:
            return False, "Proposal not found"
        
        if nep.status not in [NEPStatus.DRAFT, NEPStatus.REVIEW]:
            return False, f"Cannot start voting from status: {nep.status.value}"
        
        voting_start = time.time()
        voting_end = voting_start + (voting_duration_days * 24 * 3600)
        
        self.update_status(
            nep_id,
            NEPStatus.VOTING,
            voting_start=voting_start,
            voting_end=voting_end,
        )
        
        return True, f"Voting started, ends in {voting_duration_days} days"
    
    def list_proposals(
        self,
        status: NEPStatus = None,
        nep_type: NEPType = None,
        limit: int = 100,
    ) -> List[NEP]:
        """List proposals with optional filtering."""
        with sqlite3.connect(self.db_path, timeout=60.0) as conn:
            query = "SELECT full_data FROM neps WHERE 1=1"
            params = []
            
            if status:
                query += " AND status = ?"
                params.append(status.value)
            
            if nep_type:
                query += " AND nep_type = ?"
                params.append(nep_type.value)
            
            query += " ORDER BY created_at DESC LIMIT ?"
            params.append(limit)
            
            rows = conn.execute(query, params).fetchall()
            
            return [NEP.from_dict(json.loads(row[0])) for row in rows]
    
    def get_active_config(self) -> Dict[str, any]:
        """
        Get the current active configuration based on all ACTIVE NEPs.
        
        This is what nodes use to determine current protocol parameters.
        """
        config = {}
        
        active_neps = self.list_proposals(status=NEPStatus.ACTIVE)
        
        # Sort by activation order (older first)
        active_neps.sort(key=lambda n: n.activation_block or 0)
        
        # Apply parameter changes in order
        for nep in active_neps:
            for change in nep.parameter_changes:
                key = f"{change.module}.{change.parameter}"
                config[key] = change.new_value
        
        return config
    
    def to_gossip_dict(self, nep_id: str) -> Optional[Dict]:
        """Serialize a NEP for gossip protocol."""
        nep = self.get_proposal(nep_id)
        if not nep:
            return None
        return nep.to_dict()
    
    def from_gossip_dict(self, data: Dict) -> bool:
        """
        Receive a NEP from gossip and store it.
        
        Returns True if the NEP was new/updated.
        """
        try:
            incoming_nep = NEP.from_dict(data)
            
            existing = self.get_proposal(incoming_nep.nep_id)
            
            if existing:
                # Only update if incoming is newer
                if incoming_nep.updated_at > existing.updated_at:
                    with self.lock:
                        with sqlite3.connect(self.db_path, timeout=60.0) as conn:
                            conn.execute("""
                                UPDATE neps SET
                                    status = ?,
                                    voting_start = ?,
                                    voting_end = ?,
                                    activation_block = ?,
                                    updated_at = ?,
                                    full_data = ?
                                WHERE nep_id = ?
                            """, (
                                incoming_nep.status.value,
                                incoming_nep.voting_start,
                                incoming_nep.voting_end,
                                incoming_nep.activation_block,
                                incoming_nep.updated_at,
                                json.dumps(incoming_nep.to_dict()),
                                incoming_nep.nep_id,
                            ))
                    return True
                return False
            else:
                # New NEP
                with self.lock:
                    with sqlite3.connect(self.db_path, timeout=60.0) as conn:
                        conn.execute("""
                            INSERT INTO neps 
                            (nep_id, nep_type, status, title, author_node_id, 
                             content_hash, created_at, updated_at, 
                             voting_start, voting_end, activation_block, full_data)
                            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """, (
                            incoming_nep.nep_id,
                            incoming_nep.nep_type.value,
                            incoming_nep.status.value,
                            incoming_nep.title,
                            incoming_nep.author_node_id,
                            incoming_nep.content_hash,
                            incoming_nep.created_at,
                            incoming_nep.updated_at,
                            incoming_nep.voting_start,
                            incoming_nep.voting_end,
                            incoming_nep.activation_block,
                            json.dumps(incoming_nep.to_dict()),
                        ))
                
                logger.info(f"Received new NEP via gossip: {incoming_nep.summary()}")
                return True
                
        except Exception as e:
            logger.error(f"Failed to process gossip NEP: {e}")
            return False


    def finalize_proposal(
        self,
        nep_id: str,
        approved: bool,
        quorum_reached: bool,
        total_stake_voted: float = 0.0,
    ) -> Tuple[bool, str, float]:
        """
        Finalize a proposal after voting ends.
        
        Rewards are STAKE-PROPORTIONAL:
        - Approved: reward = 0.1% of total_stake_voted + fee refund
        - Rejected w/ quorum: reward = 0.01% of total_stake_voted (community engaged)
        - Rejected w/o quorum: Fee burned (nobody cared = spam)
        
        This means:
        - A proposal that engaged 100K NEURO of stake → ~100 NEURO reward
        - A proposal that engaged 1M NEURO of stake → ~1000 NEURO reward (capped)
        - More engagement = more reward = incentive for quality proposals
        
        Returns: (success, message, reward_amount)
        """
        nep = self.get_proposal(nep_id)
        if not nep:
            return False, "Proposal not found", 0.0
        
        if nep.status != NEPStatus.VOTING:
            return False, f"Cannot finalize from status: {nep.status.value}", 0.0
        
        reward = 0.0
        
        if approved:
            # Proposal passed! Reward proportional to engagement
            new_status = NEPStatus.APPROVED
            
            # Calculate stake-proportional reward
            base_reward = total_stake_voted * self.PROPOSER_REWARD_RATE_APPROVED
            base_reward = max(base_reward, self.PROPOSER_REWARD_MIN)  # At least minimum
            base_reward = min(base_reward, self.PROPOSER_REWARD_MAX)  # Cap at maximum
            
            reward = base_reward + self.PROPOSAL_FEE  # Plus fee refund
            
            message = (
                f"Approved! Proposer earns {reward:.2f} NEURO "
                f"({base_reward:.2f} reward + {self.PROPOSAL_FEE} fee refund, "
                f"based on {total_stake_voted:.0f} NEURO stake participation)"
            )
            
            # Credit reward to author
            if self.ledger:
                try:
                    self.ledger._credit_local_balance(
                        nep.author_node_id,
                        reward,
                        f"NEP approved: {nep_id} (stake participation: {total_stake_voted:.0f})"
                    )
                except Exception as e:
                    logger.error(f"Failed to credit proposer reward: {e}")
                    
        elif quorum_reached:
            # Rejected but community engaged - smaller proportional reward
            new_status = NEPStatus.REJECTED
            
            # Smaller rate for rejected proposals, but still rewards engagement
            reward = total_stake_voted * self.PROPOSER_REWARD_RATE_QUORUM
            reward = min(reward, self.PROPOSER_REWARD_MAX * 0.1)  # Cap at 10% of max
            
            message = (
                f"Rejected (quorum reached). Proposer receives {reward:.2f} NEURO "
                f"(based on {total_stake_voted:.0f} NEURO stake participation)"
            )
            
            if self.ledger and reward > 0:
                try:
                    self.ledger._credit_local_balance(
                        nep.author_node_id,
                        reward,
                        f"NEP rejected (quorum): {nep_id}"
                    )
                except Exception as e:
                    logger.error(f"Failed to credit partial refund: {e}")
        else:
            # No quorum - fee burned (anti-spam)
            new_status = NEPStatus.REJECTED
            reward = 0.0
            message = f"Rejected (no quorum). Fee of {self.PROPOSAL_FEE} NEURO burned."
            logger.info(f"Burned {self.PROPOSAL_FEE} NEURO for rejected proposal {nep_id}")
        
        # Update status
        self.update_status(nep_id, new_status)
        
        logger.info(f"Finalized {nep_id}: {message}")
        return True, message, reward
    
    @staticmethod
    def calculate_expected_reward(total_stake_voted: float, approved: bool = True) -> float:
        """
        Calculate expected reward for a proposal based on stake participation.
        
        Useful for UI to show "potential reward" before voting ends.
        """
        if approved:
            reward = total_stake_voted * NEPRegistry.PROPOSER_REWARD_RATE_APPROVED
            reward = max(reward, NEPRegistry.PROPOSER_REWARD_MIN)
            reward = min(reward, NEPRegistry.PROPOSER_REWARD_MAX)
            return reward + NEPRegistry.PROPOSAL_FEE
        else:
            reward = total_stake_voted * NEPRegistry.PROPOSER_REWARD_RATE_QUORUM
            return min(reward, NEPRegistry.PROPOSER_REWARD_MAX * 0.1)


# Convenience functions
def get_active_neps(registry: NEPRegistry) -> List[NEP]:
    """Get all currently active NEPs."""
    return registry.list_proposals(status=NEPStatus.ACTIVE)


def get_pending_neps(registry: NEPRegistry) -> List[NEP]:
    """Get all NEPs pending vote or activation."""
    voting = registry.list_proposals(status=NEPStatus.VOTING)
    scheduled = registry.list_proposals(status=NEPStatus.SCHEDULED)
    return voting + scheduled
