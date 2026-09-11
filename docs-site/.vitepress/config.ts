import { defineConfig } from 'vitepress'
export default defineConfig({
  title: 'NeuroShard Docs',
  description: 'Operate, verify, and improve the experimental native training network.',
  cleanUrls: true,
  head: [['link', { rel: 'icon', href: '/favicon.ico' }]],
  themeConfig: {
    logo: '/logo.png',
    nav: [{ text: 'Network', link: 'https://neuroshard.com' }, { text: 'Participate', link: '/generated/PUBLIC_TESTNET' }],
    sidebar: [{ text: 'Operate', items: [
      { text: 'Start here', link: '/' },
      { text: 'Node and worker guide', link: '/generated/PUBLIC_TESTNET' },
      { text: 'Model card', link: '/generated/MODEL_CARD' },
      { text: 'Public API', link: '/generated/API' },
      { text: 'Dataset pipeline', link: '/generated/DATA_PIPELINE' },
      { text: 'Deployment', link: '/generated/DEPLOYMENT' },
    ]}, { text: 'Protocol and evidence', items: [
      { text: 'Protocol specification', link: '/generated/LLM_PROTOCOL' },
      { text: 'Experiments', link: '/generated/LLM_EXPERIMENTS' },
      { text: 'Continual model evolution', link: '/generated/EVOLUTION_PROTOCOL' },
      { text: 'Fundamentals review', link: '/generated/FUNDAMENTALS_REVIEW' },
      { text: 'Research roadmap', link: '/generated/RESEARCH_ROADMAP' },
    ]}, { text: 'Contribute', items: [
      { text: 'Contribution guide', link: '/generated/CONTRIBUTING' },
      { text: 'Governance', link: '/generated/GOVERNANCE' },
      { text: 'Security', link: '/generated/SECURITY' },
      { text: 'Releases', link: '/generated/RELEASES' },
    ]}],
    socialLinks: [{ icon: 'github', link: 'https://github.com/neuroshard-ai/neuroshard' }],
    search: { provider: 'local' },
    outline: [2, 3],
    footer: { message: 'Experimental network. Source code licensed under Apache 2.0.' },
  },
  markdown: { math: true },
})
