import { useState, useEffect } from 'react';
import { Vote, FileText, Clock, Shield, Users, RefreshCw, ExternalLink } from 'lucide-react';
import { motion } from 'framer-motion';
import { SEO } from './SEO';

interface ProtocolVersion {
  major: number;
  minor: number;
  patch: number;
  features: string[];
  active_neps: string[];
}

interface GovernanceStats {
  total_proposals: number;
  active_votes: number;
  total_votes_cast: number;
  total_stake_participated: number;
  protocol_version: ProtocolVersion;
}

export const Governance = () => {
  const [activeTab, setActiveTab] = useState<'active' | 'pending' | 'history' | 'propose'>('active');
  const [stats, setStats] = useState<GovernanceStats | null>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    loadGovernanceData();
  }, [activeTab]);

  const loadGovernanceData = async () => {
    setLoading(true);

    try {
      // In production, these would be real API calls
      // For now, show placeholder data
      
      // Simulated stats
      setStats({
        total_proposals: 0,
        active_votes: 0,
        total_votes_cast: 0,
        total_stake_participated: 0,
        protocol_version: {
          major: 1,
          minor: 0,
          patch: 0,
          features: ['diloco', 'ponw', 'dynamic_layers', 'gossip_proofs', 'ecdsa_signatures', 'inference_market', 'robust_aggregation'],
          active_neps: [],
        },
      });
      
    } catch (err) {
      console.error('Governance Load Error:', err);
    } finally {
      setLoading(false);
    }
  };

  return (
    <>
      <SEO 
        title="Governance" 
        description="Participate in NeuroShard governance. Vote on protocol changes, propose improvements, and shape the future of decentralized AI." 
      />
      <div className="min-h-screen bg-neutral-950 pt-20 sm:pt-28 pb-12 px-4 sm:px-6">
        <div className="container mx-auto max-w-7xl">
          {/* Header */}
          <div className="mb-8 flex justify-between items-end">
            <div>
              <h1 className="text-4xl font-bold text-white mb-2 font-display">Governance</h1>
              <p className="text-neutral-400">
                Shape the future of NeuroShard through decentralized governance
              </p>
            </div>
            <button 
              onClick={loadGovernanceData}
              className="p-2 bg-neutral-800 hover:bg-neutral-700 text-neutral-300 hover:text-white transition-colors"
              title="Refresh Data"
            >
              <RefreshCw className={`w-5 h-5 ${loading ? 'animate-spin' : ''}`} />
            </button>
          </div>

          {/* Stats Cards */}
          <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mb-8">
            {[
              { 
                icon: FileText, 
                label: 'Total Proposals', 
                value: stats?.total_proposals ?? 0, 
                color: 'text-accent' 
              },
              { 
                icon: Vote, 
                label: 'Active Votes', 
                value: stats?.active_votes ?? 0, 
                color: 'text-yellow-400' 
              },
              { 
                icon: Shield, 
                label: 'Protocol Version', 
                value: stats ? `v${stats.protocol_version.major}.${stats.protocol_version.minor}.${stats.protocol_version.patch}` : '-', 
                color: 'text-purple-400' 
              },
              { 
                icon: Users, 
                label: 'Active Features', 
                value: stats?.protocol_version.features.length ?? 0, 
                color: 'text-green-400' 
              },
            ].map((item, idx) => (
              <motion.div
                key={item.label}
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ delay: idx * 0.1 }}
                className="bg-neutral-900/50 border border-neutral-800 p-4"
              >
                <div className="flex items-center gap-2 mb-1">
                  <item.icon className={`w-4 h-4 ${item.color}`} />
                  <span className="text-neutral-400 text-xs">{item.label}</span>
                </div>
                <p className="text-xl font-bold text-white">{item.value}</p>
              </motion.div>
            ))}
          </div>

          {/* Features List */}
          {stats && (
            <div className="bg-neutral-900/50 border border-neutral-800 p-6 mb-8">
              <h3 className="text-lg font-semibold text-white mb-4 flex items-center gap-2 font-display">
                <Shield className="w-5 h-5 text-purple-400" />
                Active Protocol Features
              </h3>
              <div className="flex flex-wrap gap-2">
                {stats.protocol_version.features.map((feature) => (
                  <span 
                    key={feature}
                    className="px-3 py-1 bg-neutral-800 text-neutral-300 text-sm"
                  >
                    {feature}
                  </span>
                ))}
              </div>
            </div>
          )}

          {/* Tabs */}
          <div className="flex gap-2 mb-6 border-b border-neutral-800 overflow-x-auto">
            {[
              { id: 'active', label: 'Active Votes', icon: Vote },
              { id: 'pending', label: 'Pending', icon: Clock },
              { id: 'history', label: 'History', icon: FileText },
              { id: 'propose', label: 'Create Proposal', icon: FileText },
            ].map((tab) => (
              <button
                key={tab.id}
                onClick={() => setActiveTab(tab.id as 'active' | 'pending' | 'history' | 'propose')}
                className={`px-4 py-3 font-semibold transition-colors whitespace-nowrap flex items-center gap-2 ${
                  activeTab === tab.id
                    ? 'text-accent border-b-2 border-accent'
                    : 'text-neutral-400 hover:text-white'
                }`}
              >
                <tab.icon className="w-4 h-4" />
                {tab.label}
              </button>
            ))}
          </div>

          {/* Content Area */}
          <div className="bg-neutral-900/50 border border-neutral-800 overflow-hidden min-h-[400px]">
            {loading ? (
              <div className="flex flex-col items-center justify-center h-96">
                <div className="inline-block animate-spin h-12 w-12 border-b-2 border-accent mb-4"></div>
                <p className="text-neutral-400">Loading governance data...</p>
              </div>
            ) : activeTab === 'active' ? (
              <div className="p-8 text-center">
                <Vote className="w-16 h-16 text-neutral-700 mx-auto mb-4" />
                <h3 className="text-xl font-semibold text-white mb-2 font-display">No Active Votes</h3>
                <p className="text-neutral-400 mb-6 max-w-md mx-auto">
                  There are no proposals currently in the voting phase. 
                  Check back soon or create a proposal to improve the network.
                </p>
                <button 
                  onClick={() => setActiveTab('propose')}
                  className="px-6 py-3 bg-accent hover:bg-accent/90 text-neutral-950 font-semibold transition-colors"
                >
                  Create a Proposal
                </button>
              </div>
            ) : activeTab === 'pending' ? (
              <div className="p-8 text-center">
                <Clock className="w-16 h-16 text-neutral-700 mx-auto mb-4" />
                <h3 className="text-xl font-semibold text-white mb-2 font-display">No Pending Proposals</h3>
                <p className="text-neutral-400 max-w-md mx-auto">
                  No proposals are currently in draft or review phase.
                </p>
              </div>
            ) : activeTab === 'history' ? (
              <div className="p-8 text-center">
                <FileText className="w-16 h-16 text-neutral-700 mx-auto mb-4" />
                <h3 className="text-xl font-semibold text-white mb-2 font-display">No Proposal History</h3>
                <p className="text-neutral-400 max-w-md mx-auto">
                  The governance system was just launched. 
                  Past proposals will appear here once voting concludes.
                </p>
              </div>
            ) : activeTab === 'propose' ? (
              <div className="p-8">
                <h3 className="text-xl font-semibold text-white mb-6 font-display">Create a Proposal</h3>
                
                {/* Wallet Setup */}
                <div className="bg-accent/10 border border-accent/30 p-6 mb-6">
                  <h4 className="text-lg font-semibold text-white mb-3 flex items-center gap-2">
                    <Shield className="w-5 h-5 text-accent" />
                    Wallet Authentication
                  </h4>
                  <p className="text-neutral-400 text-sm mb-4">
                    Proposing and voting requires your wallet token (same as node registration).
                    Your wallet signs proposals/votes cryptographically to prove ownership.
                  </p>
                  <pre className="bg-neutral-900 p-3 text-sm text-neutral-300 overflow-x-auto mb-3">
                    <code># Setup wallet (one-time){'\n'}neuroshard-governance --token YOUR_TOKEN wallet --save{'\n'}{'\n'}# Now future commands auto-authenticate</code>
                  </pre>
                </div>

                {/* CLI/SDK Instructions */}
                <div className="grid md:grid-cols-2 gap-6 mb-6">
                  <div className="bg-neutral-800/30 p-6 border border-neutral-700 hover:border-accent/50 transition-colors">
                    <h4 className="text-lg font-semibold text-white mb-2">Create Proposal</h4>
                    <p className="text-neutral-400 text-sm mb-4">Install CLI and submit a proposal:</p>
                    <pre className="bg-neutral-900 p-3 text-sm text-neutral-300 overflow-x-auto mb-3">
                      <code>pip install nexaroa</code>
                    </pre>
                    <pre className="bg-neutral-900 p-3 text-sm text-neutral-300 overflow-x-auto">
                      <code>neuroshard-governance propose \{'\n'}  --title "Add MTP Training" \{'\n'}  --type train</code>
                    </pre>
                  </div>
                  
                  <div className="bg-neutral-800/30 p-6 border border-neutral-700 hover:border-accent/50 transition-colors">
                    <h4 className="text-lg font-semibold text-white mb-2">Vote on Proposals</h4>
                    <p className="text-neutral-400 text-sm mb-4">List proposals and cast your vote:</p>
                    <pre className="bg-neutral-900 p-3 text-sm text-neutral-300 overflow-x-auto mb-3">
                      <code>neuroshard-governance list --status voting</code>
                    </pre>
                    <pre className="bg-neutral-900 p-3 text-sm text-neutral-300 overflow-x-auto">
                      <code>neuroshard-governance vote NEP-001 yes \{'\n'}  --reason "Improves efficiency"</code>
                    </pre>
                  </div>
                </div>

                {/* Quick CLI Reference */}
                <div className="bg-neutral-800/30 p-6 border border-neutral-700">
                  <h4 className="text-lg font-semibold text-white mb-4">Quick Reference</h4>
                  <div className="grid sm:grid-cols-2 gap-3 text-sm">
                    <div className="flex items-center gap-2">
                      <code className="bg-neutral-900 px-2 py-1 text-accent">list</code>
                      <span className="text-neutral-400">List all proposals</span>
                    </div>
                    <div className="flex items-center gap-2">
                      <code className="bg-neutral-900 px-2 py-1 text-accent">show NEP-001</code>
                      <span className="text-neutral-400">View proposal details</span>
                    </div>
                    <div className="flex items-center gap-2">
                      <code className="bg-neutral-900 px-2 py-1 text-accent">propose</code>
                      <span className="text-neutral-400">Create new proposal</span>
                    </div>
                    <div className="flex items-center gap-2">
                      <code className="bg-neutral-900 px-2 py-1 text-accent">vote</code>
                      <span className="text-neutral-400">Cast your vote</span>
                    </div>
                    <div className="flex items-center gap-2">
                      <code className="bg-neutral-900 px-2 py-1 text-accent">results NEP-001</code>
                      <span className="text-neutral-400">View voting results</span>
                    </div>
                    <div className="flex items-center gap-2">
                      <code className="bg-neutral-900 px-2 py-1 text-accent">--help</code>
                      <span className="text-neutral-400">Show all options</span>
                    </div>
                  </div>
                </div>

                <div className="mt-8 text-center">
                  <a 
                    href="https://docs.neuroshard.com/governance/proposals"
                    target="_blank"
                    rel="noopener noreferrer"
                    className="inline-flex items-center gap-2 text-accent hover:text-accent"
                  >
                    Read the full proposal guide
                    <ExternalLink className="w-4 h-4" />
                  </a>
                </div>
              </div>
            ) : null}
          </div>

          {/* Info Box */}
          <div className="mt-8 bg-neutral-900/50 border border-neutral-800 p-6">
            <h3 className="text-lg font-semibold text-white mb-3 flex items-center gap-2 font-display">
              <Shield className="w-5 h-5 text-accent" />
              About NeuroShard Governance
            </h3>
            <div className="text-neutral-400 space-y-2 text-sm">
              <p>
                <strong className="text-white">NEP (NeuroShard Enhancement Proposal):</strong> Formal proposals for protocol changes. 
                All changes to the LLM architecture, training algorithms, or economics must go through the NEP process.
              </p>
              <p>
                <strong className="text-white">Stake-Weighted Voting:</strong> Voting power is proportional to staked NEURO. 
                1 NEURO staked = 1 vote. Proposals need 66% approval and 20% quorum to pass.
              </p>
              <p>
                <strong className="text-white">Stake-Proportional Rewards:</strong> Proposer rewards scale with engagement—0.1% of 
                total stake that voted (if approved). More impactful proposals = more votes = higher reward. Capped at 1000 NEURO.
              </p>
              <p>
                <strong className="text-white">Economic Impact Analysis:</strong> Every proposal must include quantified economic impacts. 
                This ensures miners understand how changes affect their earnings before voting.
              </p>
            </div>
            
            <div className="mt-4 pt-4 border-t border-neutral-800">
              <a 
                href="https://docs.neuroshard.com/governance/overview"
                target="_blank"
                rel="noopener noreferrer"
                className="inline-flex items-center gap-2 text-accent hover:text-accent text-sm"
              >
                Learn more about governance
                <ExternalLink className="w-4 h-4" />
              </a>
            </div>
          </div>
        </div>
      </div>
    </>
  );
};
