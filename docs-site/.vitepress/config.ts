import { defineConfig } from 'vitepress'
import { withMermaid } from 'vitepress-plugin-mermaid'

export default withMermaid(
  defineConfig({
    title: "NeuroShard",
    description: "Decentralized AI Training Network - Complete Documentation",
    
    head: [
      ['link', { rel: 'icon', href: '/favicon.ico' }],
      ['meta', { name: 'theme-color', content: '#06b6d4' }],
      ['meta', { property: 'og:type', content: 'website' }],
      ['meta', { property: 'og:title', content: 'NeuroShard Documentation' }],
      ['meta', { property: 'og:description', content: 'Build the future of decentralized AI. Train models, earn NEURO, own collective intelligence.' }],
      ['meta', { property: 'og:image', content: 'https://neuroshard.com/og-image.png' }],
    ],

themeConfig: {
    logo: '/logo.png',
    siteTitle: 'NeuroShard Docs',
      
      nav: [
        { text: 'Guide', link: '/guide/introduction' },
        { text: 'Architecture', link: '/architecture/overview' },
        { text: 'Economics', link: '/economics/overview' },
        { text: 'Governance', link: '/governance/overview' },
        { text: 'API Reference', link: '/api/overview' },
        {
          text: 'Resources',
          items: [
            { text: 'Whitepaper', link: 'https://neuroshard.com/whitepaper' },
            { text: 'Ledger Explorer', link: 'https://neuroshard.com/ledger' },
            { text: 'Governance Portal', link: 'https://neuroshard.com/governance' },
            { text: 'Download Node', link: 'https://neuroshard.com/download' },
            { text: 'Main Website', link: 'https://neuroshard.com' },
          ]
        }
      ],

      sidebar: {
        '/guide/': [
          {
            text: 'Getting Started',
            collapsed: false,
            items: [
              { text: 'Introduction', link: '/guide/introduction' },
              { text: 'Quick Start', link: '/guide/quick-start' },
              { text: 'Installation', link: '/guide/installation' },
              { text: 'Running a Node', link: '/guide/running-a-node' },
            ]
          },
          {
            text: 'Core Concepts',
            collapsed: false,
            items: [
              { text: 'How It Works', link: '/guide/how-it-works' },
              { text: 'Network Roles', link: '/guide/network-roles' },
              { text: 'Training Pipeline', link: '/guide/training-pipeline' },
              { text: 'Proof of Neural Work', link: '/guide/proof-of-neural-work' },
            ]
          },
          {
            text: 'Reference',
            collapsed: false,
            items: [
              { text: 'CLI Reference', link: '/guide/cli-reference' },
              { text: 'Configuration', link: '/guide/configuration' },
              { text: 'Troubleshooting', link: '/guide/troubleshooting' },
              { text: 'FAQ', link: '/guide/faq' },
            ]
          }
        ],
        '/architecture/': [
          {
            text: 'System Architecture',
            collapsed: false,
            items: [
              { text: 'Overview', link: '/architecture/overview' },
              { text: 'NeuroLLM Model', link: '/architecture/neurollm' },
              { text: 'Dynamic Scaling', link: '/architecture/dynamic-scaling' },
            ]
          },
          {
            text: 'Data Pipeline',
            collapsed: false,
            items: [
              { text: 'Genesis Data', link: '/architecture/genesis-data' },
              { text: 'Tokenization (BPE)', link: '/architecture/tokenization' },
            ]
          },
          {
            text: 'Distributed Training',
            collapsed: false,
            items: [
              { text: 'DiLoCo Protocol', link: '/architecture/diloco' },
              { text: 'Byzantine-Robust Aggregation', link: '/architecture/aggregation' },
            ]
          },
          {
            text: 'Network Layer',
            collapsed: false,
            items: [
              { text: 'P2P Network', link: '/architecture/p2p-network' },
            ]
          },
          {
            text: 'Theory',
            collapsed: false,
            items: [
              { text: 'Mathematical Foundations', link: '/architecture/mathematical-foundations' },
            ]
          }
        ],
        '/economics/': [
          {
            text: 'NEURO Token',
            collapsed: false,
            items: [
              { text: 'Token Overview', link: '/economics/overview' },
              { text: 'Staking Guide', link: '/economics/staking' },
              { text: 'Reward System', link: '/economics/rewards' },
            ]
          }
        ],
        '/governance/': [
          {
            text: 'Governance',
            collapsed: false,
            items: [
              { text: 'Overview', link: '/governance/overview' },
              { text: 'Creating Proposals', link: '/governance/proposals' },
              { text: 'Voting Guide', link: '/governance/voting' },
            ]
          },
          {
            text: 'Protocol',
            collapsed: false,
            items: [
              { text: 'Versioning', link: '/governance/versioning' },
              { text: 'Active NEPs', link: '/governance/active' },
            ]
          }
        ],
        '/api/': [
          {
            text: 'API Reference',
            collapsed: false,
            items: [
              { text: 'Overview', link: '/api/overview' },
              { text: 'HTTP Endpoints', link: '/api/http-endpoints' },
              { text: 'gRPC Services', link: '/api/grpc-services' },
              { text: 'WebSocket Events', link: '/api/websocket-events' },
            ]
          },
          {
            text: 'Python SDK',
            collapsed: false,
            items: [
              { text: 'Getting Started', link: '/api/python-sdk' },
              { text: 'NeuroNode Class', link: '/api/neuronode-class' },
              { text: 'NEUROLedger Class', link: '/api/ledger-class' },
            ]
          }
        ]
      },

      socialLinks: [
        { icon: 'github', link: 'https://github.com/neuroshard-ai/neuroshard' },
        { icon: 'twitter', link: 'https://x.com/shardneuro' },
        { icon: 'discord', link: 'https://discord.gg/4R49xpj7vn' }
      ],

      footer: {
        message: 'Released under the Apache License 2.0.',
        copyright: 'Copyright © 2024-2025 NeuroShard'
      },

      search: {
        provider: 'local'
      },

    outline: {
        level: [2, 3],
        label: 'On this page'
      }
    },

    markdown: {
      lineNumbers: true,
      math: true
    },

    ignoreDeadLinks: true,
    lastUpdated: false
  }),
  {
    // Mermaid plugin options
    mermaidConfig: {
      theme: 'neutral'
    }
  }
)
