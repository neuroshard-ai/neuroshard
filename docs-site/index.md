---
layout: home

hero:
  name: "NeuroShard"
  text: "Decentralized AI Training Network"
  tagline: Train models together. Earn NEURO. Own collective intelligence.
  image:
    src: /logo.svg
    alt: NeuroShard
  actions:
    - theme: brand
      text: Get Started
      link: /guide/introduction
    - theme: alt
      text: Run a Node
      link: /guide/quick-start
    - theme: alt
      text: GitHub
      link: https://github.com/neuroshard-ai/neuroshard

features:
  - title: Decentralized Training
    details: Train large language models across thousands of consumer GPUs using our DiLoCo protocol. No data center required.
  - title: Earn NEURO
    details: Contribute compute power and earn NEURO tokens through Proof of Neural Work. Fair rewards based on real contributions.
  - title: Byzantine Robust
    details: Advanced aggregation algorithms protect against malicious nodes. Your contributions are cryptographically verified.
  - title: Dynamic Scaling
    details: The model grows organically as the network expands. More nodes = larger, more capable AI.
  - title: Efficient Protocol
    details: DiLoCo reduces network communication by 500x. Train effectively even on home internet connections.
  - title: Open and Permissionless
    details: Anyone can join. No approval needed. Contribute from anywhere in the world with any GPU.
---

<style>
:root {
  --vp-home-hero-name-color: transparent;
  --vp-home-hero-name-background: -webkit-linear-gradient(120deg, #06b6d4 30%, #8b5cf6);
  --vp-home-hero-image-background-image: linear-gradient(-45deg, #06b6d450 50%, #8b5cf650 50%);
  --vp-home-hero-image-filter: blur(44px);
}

@media (min-width: 640px) {
  :root {
    --vp-home-hero-image-filter: blur(56px);
  }
}

@media (min-width: 960px) {
  :root {
    --vp-home-hero-image-filter: blur(68px);
  }
}
</style>

## Quick Start

Get up and running in 3 minutes:

```bash
# Install NeuroShard
pip install neuroshard-ai

# Run a node
neuroshard --token YOUR_WALLET_TOKEN
```

Your node will start on port 8000 (HTTP) and 9000 (gRPC).

Get your token at [neuroshard.com/signup](https://neuroshard.com/signup)

1. **Run a Node**: Install the software and contribute your GPU
2. **Train Together**: Your compute helps train a shared AI model
3. **Earn NEURO**: Get rewarded for every contribution you make

## Documentation

<div class="doc-cards">
  <a href="/guide/introduction" class="doc-card">
    <div class="doc-title">Guide</div>
    <div class="doc-desc">Learn the basics and get started</div>
  </a>
  <a href="/architecture/overview" class="doc-card">
    <div class="doc-title">Architecture</div>
    <div class="doc-desc">Deep dive into how it works</div>
  </a>
  <a href="/economics/overview" class="doc-card">
    <div class="doc-title">Economics</div>
    <div class="doc-desc">NEURO token and rewards</div>
  </a>
  <a href="/api/overview" class="doc-card">
    <div class="doc-title">API Reference</div>
    <div class="doc-desc">HTTP, gRPC, and SDK docs</div>
  </a>
</div>

<style>
.doc-cards {
  display: grid;
  grid-template-columns: repeat(4, 1fr);
  gap: 16px;
  margin: 40px 24px;
}

.doc-card {
  padding: 24px;
  background: var(--vp-c-bg-soft);
  border-radius: 12px;
  text-decoration: none;
  transition: transform 0.2s, box-shadow 0.2s;
}

.doc-card:hover {
  transform: translateY(-4px);
  box-shadow: 0 12px 24px -8px rgba(0, 0, 0, 0.1);
}

.doc-title {
  font-size: 1.1rem;
  font-weight: 600;
  color: var(--vp-c-text-1);
  margin-bottom: 8px;
}

.doc-desc {
  font-size: 0.85rem;
  color: var(--vp-c-text-2);
}

@media (max-width: 768px) {
  .doc-cards {
    grid-template-columns: repeat(2, 1fr);
  }
}

@media (max-width: 480px) {
  .doc-cards {
    grid-template-columns: 1fr;
  }
}
</style>

## Join the Community

- **GitHub**: [github.com/neuroshard-ai/neuroshard](https://github.com/neuroshard-ai/neuroshard)
- **Discord**: [discord.gg/4R49xpj7vn](https://discord.gg/4R49xpj7vn)
- **X (Twitter)**: [@shardneuro](https://x.com/shardneuro)
- **Website**: [neuroshard.com](https://neuroshard.com)
