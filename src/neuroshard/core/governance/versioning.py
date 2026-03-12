"""
Protocol Versioning - Compatibility Management

Nodes must be able to determine if they're compatible with the network.
This module tracks protocol versions and upgrade requirements.

Version Format: MAJOR.MINOR.PATCH
- MAJOR: Breaking changes (hard fork, must upgrade)
- MINOR: New features (backward compatible)
- PATCH: Bug fixes (fully compatible)
"""

import json
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Set
from enum import Enum

logger = logging.getLogger(__name__)


class CompatibilityLevel(Enum):
    """How compatible two versions are."""
    IDENTICAL = "identical"           # Same version
    COMPATIBLE = "compatible"         # Can communicate
    UPGRADE_RECOMMENDED = "upgrade"   # Works but should upgrade
    INCOMPATIBLE = "incompatible"     # Cannot communicate


@dataclass
class ProtocolVersion:
    """
    Protocol version with feature flags.
    
    The version determines what features/parameters a node supports.
    Feature flags allow granular compatibility checking.
    """
    major: int = 1
    minor: int = 0
    patch: int = 0
    
    # Feature flags (what this version supports)
    features: Set[str] = field(default_factory=set)
    
    # Active NEPs at this version
    active_neps: List[str] = field(default_factory=list)
    
    @property
    def version_string(self) -> str:
        return f"{self.major}.{self.minor}.{self.patch}"
    
    def __str__(self) -> str:
        return self.version_string
    
    def __hash__(self):
        return hash(self.version_string)
    
    def __eq__(self, other):
        if isinstance(other, ProtocolVersion):
            return (
                self.major == other.major and
                self.minor == other.minor and
                self.patch == other.patch
            )
        return False
    
    def __lt__(self, other):
        if isinstance(other, ProtocolVersion):
            return (self.major, self.minor, self.patch) < (other.major, other.minor, other.patch)
        return NotImplemented
    
    def __le__(self, other):
        return self == other or self < other
    
    def to_dict(self) -> Dict:
        return {
            "major": self.major,
            "minor": self.minor,
            "patch": self.patch,
            "features": list(self.features),
            "active_neps": self.active_neps,
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'ProtocolVersion':
        return cls(
            major=data.get("major", 1),
            minor=data.get("minor", 0),
            patch=data.get("patch", 0),
            features=set(data.get("features", [])),
            active_neps=data.get("active_neps", []),
        )
    
    @classmethod
    def from_string(cls, version_str: str) -> 'ProtocolVersion':
        """Parse version string like '1.2.3'."""
        parts = version_str.split(".")
        return cls(
            major=int(parts[0]) if len(parts) > 0 else 1,
            minor=int(parts[1]) if len(parts) > 1 else 0,
            patch=int(parts[2]) if len(parts) > 2 else 0,
        )
    
    def check_compatibility(self, other: 'ProtocolVersion') -> CompatibilityLevel:
        """
        Check compatibility between this version and another.
        """
        if self == other:
            return CompatibilityLevel.IDENTICAL
        
        # Major version mismatch = incompatible
        if self.major != other.major:
            return CompatibilityLevel.INCOMPATIBLE
        
        # Minor version difference = compatible but upgrade recommended
        if self.minor != other.minor:
            return CompatibilityLevel.UPGRADE_RECOMMENDED
        
        # Patch difference = fully compatible
        return CompatibilityLevel.COMPATIBLE
    
    def supports_feature(self, feature: str) -> bool:
        """Check if this version supports a feature."""
        return feature in self.features


# =============================================================================
# CURRENT PROTOCOL VERSION
# =============================================================================

# This is the current protocol version
# Updated when NEPs are activated
CURRENT_VERSION = ProtocolVersion(
    major=1,
    minor=0,
    patch=0,
    features={
        "diloco",           # DiLoCo training
        "dynamic_layers",   # Dynamic layer assignment
        "ponw",             # Proof of Neural Work
        "gossip_proofs",    # Gossip-based proof sharing
        "ecdsa_signatures", # ECDSA cryptographic signatures
        "inference_market", # Dynamic inference pricing
        "robust_aggregation", # Byzantine-tolerant gradient aggregation
    },
    active_neps=[],  # No NEPs activated yet
)


# =============================================================================
# UPCOMING FEATURES (from potential NEPs)
# =============================================================================

FEATURE_REGISTRY = {
    "mtp": {
        "name": "Multi-Token Prediction",
        "description": "Predict multiple future tokens for faster training",
        "min_version": ProtocolVersion(1, 1, 0),
        "required_nep": "NEP-001",
    },
    "mla": {
        "name": "Multi-Head Latent Attention",
        "description": "Compressed KV cache for 20x memory reduction",
        "min_version": ProtocolVersion(2, 0, 0),  # Breaking change
        "required_nep": "NEP-002",
    },
    "aux_free_moe": {
        "name": "Auxiliary-Loss-Free MoE Balancing",
        "description": "Dynamic bias for expert load balancing",
        "min_version": ProtocolVersion(1, 1, 0),
        "required_nep": "NEP-003",
    },
    "fp8_training": {
        "name": "FP8 Mixed Precision Training",
        "description": "8-bit floating point for faster training",
        "min_version": ProtocolVersion(1, 2, 0),
        "required_nep": "NEP-004",
    },
    "dualpipe": {
        "name": "DualPipe Parallelism",
        "description": "Overlap compute and communication",
        "min_version": ProtocolVersion(1, 1, 0),
        "required_nep": "NEP-005",
    },
}


# =============================================================================
# VERSION MANAGER
# =============================================================================

class VersionManager:
    """
    Manages protocol versions and feature activation.
    
    Responsibilities:
    - Track current version
    - Activate features from NEPs
    - Check compatibility with peers
    - Enforce version requirements
    """
    
    def __init__(
        self,
        current_version: ProtocolVersion = None,
        nep_registry=None,
    ):
        self.version = current_version or CURRENT_VERSION
        self.nep_registry = nep_registry
        
        # Version history
        self.version_history: List[Tuple[ProtocolVersion, float]] = []
    
    def get_current_version(self) -> ProtocolVersion:
        """Get the current protocol version."""
        return self.version
    
    def activate_nep(self, nep_id: str) -> Tuple[bool, str]:
        """
        Activate a NEP and update version accordingly.
        
        This is called when an approved NEP reaches its activation block.
        """
        if not self.nep_registry:
            return False, "No NEP registry configured"
        
        nep = self.nep_registry.get_proposal(nep_id)
        if not nep:
            return False, f"NEP {nep_id} not found"
        
        from .proposal import NEPStatus
        if nep.status != NEPStatus.SCHEDULED:
            return False, f"NEP {nep_id} not scheduled (status: {nep.status.value})"
        
        # Parse target version from upgrade path
        target_version = ProtocolVersion.from_string(nep.upgrade_path.target_version)
        
        # Update version
        old_version = self.version
        self.version = target_version
        
        # Add features from NEP
        for change in nep.parameter_changes:
            if change.parameter.startswith("feature_"):
                feature_name = change.parameter[8:]  # Remove "feature_" prefix
                self.version.features.add(feature_name)
        
        # Track in version history
        import time
        self.version_history.append((old_version, time.time()))
        
        # Update NEP status
        self.nep_registry.update_status(nep_id, NEPStatus.ACTIVE)
        
        logger.info(
            f"Activated {nep_id}: {old_version} → {self.version} "
            f"(features: {self.version.features})"
        )
        
        return True, f"Activated {nep_id}, now at version {self.version}"
    
    def check_peer_compatibility(
        self,
        peer_version: ProtocolVersion,
    ) -> Tuple[CompatibilityLevel, str]:
        """
        Check if we can communicate with a peer.
        
        Returns: (compatibility_level, message)
        """
        level = self.version.check_compatibility(peer_version)
        
        messages = {
            CompatibilityLevel.IDENTICAL: "Identical version",
            CompatibilityLevel.COMPATIBLE: "Compatible versions",
            CompatibilityLevel.UPGRADE_RECOMMENDED: (
                f"Upgrade recommended: you have {self.version}, "
                f"peer has {peer_version}"
            ),
            CompatibilityLevel.INCOMPATIBLE: (
                f"Incompatible versions: you have {self.version}, "
                f"peer has {peer_version}. Major version mismatch."
            ),
        }
        
        return level, messages[level]
    
    def can_participate_in_training(self) -> Tuple[bool, str]:
        """
        Check if we meet minimum requirements for training.
        """
        required_features = {"diloco", "ponw", "robust_aggregation"}
        missing = required_features - self.version.features
        
        if missing:
            return False, f"Missing required features: {missing}"
        
        return True, "Ready for training"
    
    def can_participate_in_inference(self) -> Tuple[bool, str]:
        """
        Check if we meet minimum requirements for inference.
        """
        required_features = {"ponw", "inference_market"}
        missing = required_features - self.version.features
        
        if missing:
            return False, f"Missing required features: {missing}"
        
        return True, "Ready for inference"


# =============================================================================
# MODULE-LEVEL FUNCTIONS
# =============================================================================

def get_current_version() -> ProtocolVersion:
    """Get the current protocol version."""
    return CURRENT_VERSION


def is_compatible(
    local_version: ProtocolVersion,
    peer_version: ProtocolVersion,
) -> bool:
    """Check if two versions can communicate."""
    level = local_version.check_compatibility(peer_version)
    return level in [
        CompatibilityLevel.IDENTICAL,
        CompatibilityLevel.COMPATIBLE,
        CompatibilityLevel.UPGRADE_RECOMMENDED,
    ]
