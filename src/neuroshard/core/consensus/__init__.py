"""
NeuroShard Consensus Module

Implements Proof of Neural Work (PoNW) with optimistic verification:

1. **ProofVerifier**: Universal constraints + Optimistic Verification
   - Physical rate limits (hardware capabilities)
   - Format and sanity checks
   - Optimistic accept-then-challenge model
   - Challenge window management

2. **Optimistic Verification Flow**:
   - Proofs accepted IMMEDIATELY with rewards credited
   - Challenge window (10 minutes) for verification
   - Any node can challenge by staking NEURO
   - Fraud detection → challenger wins, prover slashed
   - No fraud → challenger loses stake to prover

3. **Proof Types**:
   - PipelineProof: Batch-level proof in quorum training
   - CohortSyncProof: DiLoCo sync round participation

See: docs/ARCHITECTURE_V2.md Section 10 (Proof of Neural Work v2)
"""

from neuroshard.core.consensus.verifier import (
    ProofVerifier,
    PendingProof,
    PendingProofStore,
    ProofStatus,
    PipelineProof,
    CohortSyncProof,
    CHALLENGE_WINDOW,
    MIN_CHALLENGE_STAKE,
    SLASH_MULTIPLIER,
    get_verification_rate,
)

__all__ = [
    "ProofVerifier",
    "PendingProof",
    "PendingProofStore",
    "ProofStatus",
    "PipelineProof",
    "CohortSyncProof",
    "CHALLENGE_WINDOW",
    "MIN_CHALLENGE_STAKE",
    "SLASH_MULTIPLIER",
    "get_verification_rate",
]
