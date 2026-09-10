# Protocol Versioning

NeuroShard uses semantic versioning to manage protocol compatibility. Understanding versions helps you know when to upgrade and whether you can communicate with other nodes.

## Version Format

```
MAJOR.MINOR.PATCH

Example: 1.2.3
         │ │ │
         │ │ └── Patch: Bug fixes (fully compatible)
         │ └──── Minor: New features (backward compatible)
         └────── Major: Breaking changes (must upgrade)
```

## Current Version

```python
from neuroshard.core.governance import get_current_version

version = get_current_version()
print(version)  # "1.0.0"
print(version.features)  # {'diloco', 'ponw', 'dynamic_layers', ...}
```

## Version History

| Version | Date | Description |
|---------|------|-------------|
| **1.0.0** | Launch | Initial release with DiLoCo, PoNW, dynamic layers |

*Future versions will be added as NEPs are activated.*

## Compatibility Levels

When two nodes with different versions try to communicate:

| Level | Meaning | Action |
|-------|---------|--------|
| **IDENTICAL** | Same version | Full compatibility |
| **COMPATIBLE** | Patch difference | Can communicate |
| **UPGRADE_RECOMMENDED** | Minor difference | Works but should upgrade |
| **INCOMPATIBLE** | Major difference | Cannot communicate |

### Checking Compatibility

```python
from neuroshard.core.governance import ProtocolVersion, is_compatible

my_version = ProtocolVersion(1, 0, 0)
peer_version = ProtocolVersion(1, 1, 0)

if is_compatible(my_version, peer_version):
    print("Can communicate with peer")
else:
    print("Must upgrade to communicate")
```

## Feature Flags

Beyond the version number, nodes advertise **feature flags** to indicate capabilities:

```python
version = get_current_version()

# Check if a feature is supported
if version.supports_feature("mtp"):
    print("Multi-Token Prediction enabled")
else:
    print("MTP not available in this version")
```

### Current Features

| Feature | Description | Since |
|---------|-------------|-------|
| `diloco` | DiLoCo distributed training | 1.0.0 |
| `dynamic_layers` | Dynamic layer assignment | 1.0.0 |
| `ponw` | Proof of Neural Work | 1.0.0 |
| `gossip_proofs` | Gossip-based proof sharing | 1.0.0 |
| `ecdsa_signatures` | ECDSA cryptographic signatures | 1.0.0 |
| `inference_market` | Dynamic inference pricing | 1.0.0 |
| `robust_aggregation` | Byzantine-tolerant gradient aggregation | 1.0.0 |

### Future Features (Pending NEPs)

| Feature | Description | Required NEP |
|---------|-------------|--------------|
| `mtp` | Multi-Token Prediction | TBD |
| `mla` | Multi-Head Latent Attention | TBD |
| `aux_free_moe` | Auxiliary-Loss-Free MoE Balancing | TBD |
| `fp8_training` | FP8 Mixed Precision Training | TBD |
| `dualpipe` | DualPipe Parallelism | TBD |

## Upgrade Process

### 1. Check Current Version

```bash
neuroshard version
# NeuroShard v1.0.0
# Features: diloco, ponw, dynamic_layers, ...
```

### 2. Check Network Requirements

```bash
neuroshard network status
# Network minimum version: 1.0.0
# Your version: 1.0.0
# Status: Compatible ✓
```

### 3. Upgrade When Required

```bash
# Download new version
pip install --upgrade neuroshard

# Or with Docker
docker pull neuroshard/node:latest

# Restart node
neuroshard node start
```

## Breaking Changes (Major Version)

When a major version change is approved:

1. **Announcement**: 30+ days before activation
2. **Grace Period**: Old version continues working
3. **Activation Block**: Old nodes can no longer participate
4. **Migration**: Follow specific upgrade instructions

### Example: Hypothetical v2.0.0

```
┌──────────────────────────────────────────────────────────────┐
│                    MAJOR VERSION UPGRADE                      │
├──────────────────────────────────────────────────────────────┤
│                                                              │
│   Day 0          Day 14         Day 30         Day 31        │
│     │              │              │               │          │
│     ▼              ▼              ▼               ▼          │
│  ┌──────┐     ┌──────────┐   ┌──────────┐   ┌──────────┐    │
│  │ NEP  │     │  Grace   │   │ Upgrade  │   │  v2.0.0  │    │
│  │Approved│   │  Period  │   │ Deadline │   │  Active  │    │
│  └──────┘     └──────────┘   └──────────┘   └──────────┘    │
│                                                              │
│  v1.x nodes:  ✓ Working     ✓ Working      ✗ Disconnected   │
│  v2.x nodes:  ✓ Working     ✓ Working      ✓ Working        │
│                                                              │
└──────────────────────────────────────────────────────────────┘
```

## Active NEPs Per Version

Each version tracks which NEPs are active:

```python
version = get_current_version()
print(f"Active NEPs: {version.active_neps}")
# ['NEP-001', 'NEP-003', ...]  (empty at launch)
```

This creates an auditable trail of all protocol changes.

## Version Manager API

```python
from neuroshard.core.governance import VersionManager

manager = VersionManager()

# Get current version
version = manager.get_current_version()

# Check peer compatibility
level, message = manager.check_peer_compatibility(peer_version)
print(message)

# Check if we can train/infer
can_train, reason = manager.can_participate_in_training()
can_infer, reason = manager.can_participate_in_inference()

# Activate a scheduled NEP
success, message = manager.activate_nep("NEP-001")
```

## Best Practices

### For Node Operators

1. **Subscribe to announcements** on Discord/Twitter
2. **Check version weekly** against network requirements
3. **Upgrade during low-activity periods** to minimize disruption
4. **Test in staging** before upgrading production nodes

### For Developers

1. **Check compatibility** before peer communication
2. **Gracefully handle** version mismatches
3. **Feature-gate** new functionality
4. **Log version info** for debugging
