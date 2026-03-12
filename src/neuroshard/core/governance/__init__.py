"""
NeuroShard Governance System

This module implements decentralized governance for protocol upgrades.

Philosophy:
===========
LLM architecture and economics are DEEPLY COUPLED. You cannot change one
without affecting the other. This governance system ensures:

1. TRANSPARENCY: All changes are proposed, reviewed, and voted on
2. ECONOMIC PARITY: Upgrade paths include economic adjustments
3. BACKWARD COMPATIBILITY: Grace periods for nodes to upgrade
4. STAKE-WEIGHTED VOTING: Validators with skin in the game decide

NEP (NeuroShard Enhancement Proposal) Types:
============================================
- NEP-ARCH: Architecture changes (MLA, MTP, new layers)
- NEP-ECON: Economic parameter changes (reward rates, fees)
- NEP-TRAIN: Training algorithm changes (DiLoCo params, gradient handling)
- NEP-NET: Network protocol changes (gossip, routing)

Lifecycle:
==========
1. DRAFT → Proposal created, open for discussion
2. REVIEW → Formal review period (7 days)
3. VOTING → Stake-weighted voting (7 days)
4. SCHEDULED → Approved, waiting for activation block
5. ACTIVE → Live on network
6. DEPRECATED → Superseded by newer NEP
"""

from .proposal import (
    NEP,
    NEPType,
    NEPStatus,
    create_proposal,
    EconomicImpact,
)

from .registry import (
    NEPRegistry,
    get_active_neps,
    get_pending_neps,
)

from .voting import (
    cast_vote,
    get_vote_tally,
    VoteResult,
)

from .versioning import (
    ProtocolVersion,
    get_current_version,
    is_compatible,
)

__all__ = [
    'NEP',
    'NEPType', 
    'NEPStatus',
    'create_proposal',
    'EconomicImpact',
    'NEPRegistry',
    'get_active_neps',
    'get_pending_neps',
    'cast_vote',
    'get_vote_tally',
    'VoteResult',
    'ProtocolVersion',
    'get_current_version',
    'is_compatible',
]
