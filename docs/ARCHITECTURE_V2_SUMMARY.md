# NeuroShard v2 Architecture - Quick Reference

> One-page summary of the v2.1 architecture. See [ARCHITECTURE_V2.md](./ARCHITECTURE_V2.md) for full details.

---

## Core Philosophy: Everything is Dynamic

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                     NOTHING IS FIXED                                        │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  • Network size: 1 to 1,000,000+ nodes                                      │
│  • Model depth: 11 to 256+ layers                                           │
│  • Vocabulary: 266 to 1,000,000+ tokens                                     │
│  • Node power: Raspberry Pi to H100 clusters                                │
│  • Quorums: Form, adapt, dissolve dynamically                               │
│  • Prices: Market-driven with demand response                               │
│                                                                             │
│  The only constants are the RULES, not the state.                           │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Speed Tiers

| Tier | Hardware | Speed | Can Hold | Contribution Mode |
|------|----------|-------|----------|-------------------|
| T1 | H100/A100 | <10ms/layer | 50+ layers | Pipeline (fast quorums) |
| T2 | RTX 4090 | 10-50ms | 15-30 layers | Pipeline |
| T3 | RTX 3060 | 50-200ms | 5-15 layers | Pipeline |
| T4 | CPU (32GB) | 200-1000ms | 3-8 layers | Pipeline (slow quorums) |
| T5 | Raspberry Pi | >1000ms | 1-2 layers | Async only |

---

## Quorum-Based Training

```
FAST QUORUM (T1-T2):         SLOW QUORUM (T3-T4):         ASYNC (T5):
┌─────────────────────┐      ┌─────────────────────┐      ┌──────────────┐
│ 4 fast nodes        │      │ 6 slower nodes      │      │ Train offline│
│ ~16 batches/sec     │      │ ~0.5 batches/sec    │      │ Submit grads │
└─────────────────────┘      └─────────────────────┘      └──────────────┘
         │                            │                          │
         └────────────────────────────┼──────────────────────────┘
                                      ▼
                           DiLoCo Cross-Quorum Sync
                        (sqrt-weighted by batches × freshness)
```

**Quorum Lifecycle**: Form → Train (~1 hour session) → Renew or Dissolve

---

## Gradient Weighting

```python
weight = sqrt(batches) × freshness

freshness:
  < 1 hour  → 1.0
  < 1 day   → 0.9
  < 1 week  → 0.7
  older     → 0.3
```

Fast nodes: more batches = more influence (but diminishing returns via sqrt)
Slow nodes: still contribute meaningfully

---

## Vocabulary Governance (No Coordinator)

```
PROPOSAL → DISCUSSION (7d) → VOTING (7d) → IMPLEMENTATION (7d)
    │
    └── Requires: 100 NEURO stake, 66% approval, 30% quorum
```

**Why?** Prevents malicious merge attacks. Community reviews all vocab changes.

---

## Inference Pricing

```
price = BASE × demand × speed × reputation

BASE = 0.0001 NEURO/token
demand = 1.0 to 2.0 (based on utilization)
speed = 0.8 (slow) to 1.5 (fast)
reputation = 0.8 (new) to 1.2 (proven)
```

---

## Scale Adaptation

| Nodes | Phase | Key Behavior |
|-------|-------|--------------|
| 1 | Genesis | Solo training (only time allowed) |
| 2-4 | Micro | Single quorum, everyone knows everyone |
| 5-19 | Small | 1-3 quorums, first layer growth |
| 20-99 | Medium | Speed tiers active, async contributors |
| 100-499 | Growing | Full adversarial resistance |
| 500+ | Large | Regional clustering, 100B+ params |
| 5000+ | Massive | Hierarchical coordination |

**Same code at all phases.** Behavior emerges from conditions.

---

## Adversarial Resistance

| Attack | Defense |
|--------|---------|
| Lazy compute | PoNW spot-checks, 2× stake slash |
| Slow griefing | Speed tiers, quorum kick |
| Gradient poison | Robust aggregation (trimmed mean) |
| Quorum collusion | Cross-quorum audits |
| Vocab attack | Governance voting |

---

## Contribution Modes

Every node contributes SOMETHING:

| Mode | Who | What |
|------|-----|------|
| **Pipeline** | T1-T4 in quorums | Real-time training |
| **Async** | T5 or unmatched | Offline gradients |
| **Data** | Anyone with storage | Serve Genesis data |
| **Verify** | Anyone with stake | Challenge proofs |
| **Inference** | Any quorum | Serve user requests |

---

## Key Files to Modify

| File | Changes |
|------|---------|
| `core/model/dynamic.py` | SpeedTier, ContributionMode, quorum logic |
| `core/network/quorum.py` | New file - quorum formation/lifecycle |
| `core/swarm/diloco.py` | Weighted aggregation, cohort sync |
| `core/consensus/ponw.py` | New file - optimistic verification |
| `core/economics/` | Dynamic rewards, pricing |

---

## Implementation Timeline

| Phase | Duration | Focus |
|-------|----------|-------|
| 1 | Week 1-2 | Versioning, speed tiers, modes |
| 2 | Week 2-3 | Quorum system |
| 3 | Week 3-4 | Cross-quorum sync |
| 4 | Week 4-5 | Layer growth |
| 5 | Week 5-6 | Vocab governance |
| 6 | Week 6-7 | Inference, rewards |
| 7 | Week 7-8 | Adversarial resistance |
| 8 | Week 8-10 | Testing |

**Total: ~10 weeks**

---

## Quick Links

- [Full Architecture Doc](./ARCHITECTURE_V2.md)
- [Implementation TODO](./TODO.md)
- [Current Whitepaper](./whitepaper/neuroshard_whitepaper.pdf)
