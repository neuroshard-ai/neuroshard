import { useState, useEffect } from 'react';
import { Search, Clock, Coins, Network, TrendingUp, Hash, AlertCircle, RefreshCw, Flame, ArrowRightLeft, Shield, Zap } from 'lucide-react';
import { motion } from 'framer-motion';

interface Epoch {
  epoch_id: number;
  epoch_number: number;  // Relative (1, 2, 3... from genesis)
  absolute_epoch: number;  // Unix minutes
  timestamp_start: number;
  timestamp_end: number;
  datetime_start: string;
  datetime_end: string;
  proof_count: number;
  unique_nodes: number;
  inference_proofs: number;
  training_proofs: number;
  total_tokens_processed: number;
  total_training_batches: number;
  total_uptime_seconds: number;
  total_neuro_minted: number;
  hash: string;
}

interface EpochDetail {
  epoch_id: number;
  epoch_number: number;
  absolute_epoch: number;
  timestamp_start: number;
  timestamp_end: number;
  datetime_start: string;
  datetime_end: string;
  duration_seconds: number;
  summary: {
    proof_count: number;
    unique_nodes: number;
    total_tokens_processed: number;
    total_training_batches: number;
    total_uptime_seconds: number;
    total_neuro_minted: number;
  };
  reward_breakdown: Record<string, { count: number; total_reward: number }>;
  participants: Array<{ node_id: string; proof_count: number; reward_neuro: number }>;
  proofs: Proof[];
}

interface Proof {
  signature: string;
  node_id: string;
  proof_type: string;
  timestamp: number;
  datetime: string;
  uptime_seconds: number;
  tokens_processed: number;
  training_batches: number;
  reward_neuro: number;
  epoch_id: number;
}

interface Balance {
  node_id: string;
  address: string;
  balance_neuro: number;
  total_earned: number;
  total_spent: number;
  staked_neuro: number;
  stake_multiplier: number;
  proof_count: number;
  last_activity: string | null;
  is_burn_address: boolean;
}

interface Transaction {
  tx_id: string;
  from_id: string;
  to_id: string;
  amount: number;
  fee: number;
  burn_amount: number;
  timestamp: number;
  datetime: string;
  memo: string;
  type: string;
}

interface LedgerStats {
  total_nodes: number;
  total_neuro_supply: number;
  total_minted: number;
  total_burned: number;
  circulating_supply: number;
  total_staked: number;
  total_proofs: number;
  total_transactions: number;
  total_tokens_processed: number;
  latest_epoch: number;
  burn_rate: string;
  ledger_type: string;
  consensus: string;
}

interface BurnStats {
  total_burned: number;
  total_minted: number;
  circulating_supply: number;
  burn_percentage: number;
  burn_rate: string;
  deflationary_effect: string;
}

import { SEO } from './SEO';

export const LedgerExplorer = () => {
  const [activeTab, setActiveTab] = useState<'epochs' | 'proofs' | 'balances' | 'transactions'>('epochs');
  const [searchQuery, setSearchQuery] = useState('');
  const [epochs, setEpochs] = useState<Epoch[]>([]);
  const [proofs, setProofs] = useState<Proof[]>([]);
  const [balances, setBalances] = useState<Balance[]>([]);
  const [transactions, setTransactions] = useState<Transaction[]>([]);
  const [stats, setStats] = useState<LedgerStats | null>(null);
  const [burnStats, setBurnStats] = useState<BurnStats | null>(null);
  const [loading, setLoading] = useState(true);
  const [selectedNode, setSelectedNode] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [proofsTotal, setProofsTotal] = useState(0);
  const [proofsOffset, setProofsOffset] = useState(0);
  const [epochsTotal, setEpochsTotal] = useState(0);
  const [epochsOffset, setEpochsOffset] = useState(0);
  const [selectedEpoch, setSelectedEpoch] = useState<EpochDetail | null>(null);
  const [loadingEpoch, setLoadingEpoch] = useState(false);
  const PROOFS_LIMIT = 50;
  const EPOCHS_LIMIT = 50;

  const API_BASE = import.meta.env.VITE_API_URL || '';

  useEffect(() => {
    loadAllData();
  }, [activeTab, proofsOffset, epochsOffset]);

  const fetchWithTimeout = async (url: string, timeout = 5000) => {
    const controller = new AbortController();
    const id = setTimeout(() => controller.abort(), timeout);
    try {
      const response = await fetch(url, { signal: controller.signal });
      clearTimeout(id);
      return response;
    } catch (error) {
      clearTimeout(id);
      throw error;
    }
  };

  const loadAllData = async () => {
    setLoading(true);
    setError(null);

    try {
      // Always fetch stats and burn stats
      const [statsRes, burnRes] = await Promise.all([
        fetchWithTimeout(`${API_BASE}/api/ledger/stats`),
        fetchWithTimeout(`${API_BASE}/api/ledger/burn`)
      ]);

      if (statsRes.ok) {
        setStats(await statsRes.json());
      }
      if (burnRes.ok) {
        setBurnStats(await burnRes.json());
      }

      // Fetch tab-specific data
      let dataRes;
      if (activeTab === 'epochs') {
        dataRes = await fetchWithTimeout(`${API_BASE}/api/ledger/epochs?limit=${EPOCHS_LIMIT}&offset=${epochsOffset}`);
        if (dataRes.ok) {
          const data = await dataRes.json();
          setEpochs(data.epochs || []);
          setEpochsTotal(data.total || 0);
        }
      } else if (activeTab === 'proofs') {
        const params = selectedNode 
          ? `?node_id=${selectedNode}&limit=${PROOFS_LIMIT}&offset=${proofsOffset}` 
          : `?limit=${PROOFS_LIMIT}&offset=${proofsOffset}`;
        dataRes = await fetchWithTimeout(`${API_BASE}/api/ledger/proofs${params}`);
        if (dataRes.ok) {
          const data = await dataRes.json();
          setProofs(data.proofs || []);
          setProofsTotal(data.total || 0);
        }
      } else if (activeTab === 'balances') {
        dataRes = await fetchWithTimeout(`${API_BASE}/api/ledger/balances?limit=100`);
        if (dataRes.ok) {
          const data = await dataRes.json();
          setBalances(data.balances || []);
        }
      } else if (activeTab === 'transactions') {
        const params = selectedNode ? `?node_id=${selectedNode}&limit=50` : '?limit=50';
        dataRes = await fetchWithTimeout(`${API_BASE}/api/ledger/transactions${params}`);
        if (dataRes.ok) {
          const data = await dataRes.json();
          setTransactions(data.transactions || []);
        }
      }

    } catch (err: any) {
      console.error('Ledger Load Error:', err);
      setError(err.message === 'Aborted' ? 'Connection timed out.' : 'Unable to connect to NeuroShard Node.');
    } finally {
      setLoading(false);
    }
  };

  const handleSearch = async () => {
    if (!searchQuery.trim()) return;
    
    setLoading(true);
    setError(null);
    try {
      const response = await fetchWithTimeout(`${API_BASE}/api/ledger/node/${searchQuery}`);
      if (response.ok) {
        const data = await response.json();
        setSelectedNode(data.node_id);
        setActiveTab('proofs');
        await loadAllData();
      } else {
        const proofResponse = await fetchWithTimeout(`${API_BASE}/api/ledger/proof/${searchQuery}`);
        if (proofResponse.ok) {
          const proofData = await proofResponse.json();
          setSelectedNode(proofData.node_id);
          setActiveTab('proofs');
          await loadAllData();
        } else {
          setError('No node or proof found with that ID.');
          setLoading(false);
        }
      }
    } catch (error) {
      console.error('Search error:', error);
      setError('Search failed. Ledger might be offline.');
      setLoading(false);
    }
  };

  const formatAddress = (addr: string) => {
    if (addr.length > 20) {
      return `${addr.slice(0, 10)}...${addr.slice(-8)}`;
    }
    return addr;
  };

  const formatTime = (timestamp: number) => {
    return new Date(timestamp * 1000).toLocaleString();
  };

  const formatNeuro = (amount: number) => {
    if (amount >= 1000000) return `${(amount / 1000000).toFixed(2)}M`;
    if (amount >= 1000) return `${(amount / 1000).toFixed(2)}K`;
    return amount.toFixed(6);
  };

  const loadEpochDetails = async (epochId: number) => {
    setLoadingEpoch(true);
    try {
      const response = await fetchWithTimeout(`${API_BASE}/api/ledger/epoch/${epochId}`);
      if (response.ok) {
        const data = await response.json();
        setSelectedEpoch(data);
      }
    } catch (err) {
      console.error('Failed to load epoch details:', err);
    } finally {
      setLoadingEpoch(false);
    }
  };

  return (
    <>
      <SEO title="Ledger Explorer" description="View real-time blocks, transactions, and node statistics on the NeuroShard network." />
      <div className="min-h-screen bg-neutral-950 pt-20 sm:pt-28 pb-12 px-4 sm:px-6">
      <div className="container mx-auto max-w-7xl">
        {/* Header */}
        <div className="mb-8 flex justify-between items-end">
          <div>
            <h1 className="text-4xl font-bold text-white mb-2 font-display">Ledger Explorer</h1>
            <p className="text-neutral-400">
              Explore the NeuroShard distributed ledger: PoNW proofs, NEURO balances, and burn statistics
            </p>
          </div>
          <button 
            onClick={loadAllData}
            className="p-2 bg-neutral-800 hover:bg-neutral-700 text-neutral-300 hover:text-white transition-colors"
            title="Refresh Data"
          >
            <RefreshCw className={`w-5 h-5 ${loading ? 'animate-spin' : ''}`} />
          </button>
        </div>

        {/* Error Banner */}
        {error && (
          <motion.div 
            initial={{ opacity: 0, y: -10 }}
            animate={{ opacity: 1, y: 0 }}
            className="mb-8 bg-red-900/20 border border-red-500/50 p-6 flex items-start gap-4"
          >
            <AlertCircle className="w-6 h-6 text-red-400 shrink-0 mt-0.5" />
            <div>
              <h3 className="text-red-400 font-bold mb-1 font-display">Ledger Connection Failed</h3>
              <p className="text-neutral-300 text-sm mb-4">{error}</p>
              <p className="text-neutral-400 text-xs">
                Ensure your NeuroShard Node is running locally on port 8000, or update VITE_API_URL.
              </p>
            </div>
          </motion.div>
        )}

        {/* Stats Cards - Top Row */}
        <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mb-4">
          {[
            { icon: Network, label: 'Active Nodes', value: stats?.total_nodes.toLocaleString() || '-', color: 'text-accent' },
            { icon: Coins, label: 'Circulating Supply', value: stats ? `${formatNeuro(stats.circulating_supply)} NEURO` : '-', color: 'text-yellow-400' },
            { icon: Hash, label: 'Total Proofs', value: stats?.total_proofs?.toLocaleString() ?? '-', color: 'text-purple-400' },
            { icon: TrendingUp, label: 'Latest Epoch', value: stats && stats.latest_epoch !== null ? `#${stats.latest_epoch}` : '-', color: 'text-green-400' }
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
              <p className="text-xl font-bold text-white">
                {loading && !stats ? <span className="inline-block w-20 h-6 bg-neutral-800 animate-pulse"></span> : item.value}
              </p>
            </motion.div>
          ))}
        </div>

        {/* Stats Cards - Burn Row (Highlighted) */}
        <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mb-8">
          {[
            { icon: Flame, label: 'Total Burned', value: burnStats ? `${formatNeuro(burnStats.total_burned)} NEURO` : '-', color: 'text-orange-500', highlight: true },
            { icon: Zap, label: 'Total Minted', value: stats ? `${formatNeuro(stats.total_minted)} NEURO` : '-', color: 'text-emerald-400' },
            { icon: Shield, label: 'Total Staked', value: stats ? `${formatNeuro(stats.total_staked || 0)} NEURO` : '-', color: 'text-blue-400' },
            { icon: ArrowRightLeft, label: 'Transactions', value: stats?.total_transactions?.toLocaleString() ?? '-', color: 'text-pink-400' }
          ].map((item, idx) => (
            <motion.div
              key={item.label}
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: 0.4 + idx * 0.1 }}
              className={`bg-neutral-900/50 border p-4 ${item.highlight ? 'border-orange-500/50 bg-orange-950/20' : 'border-neutral-800'}`}
            >
              <div className="flex items-center gap-2 mb-1">
                <item.icon className={`w-4 h-4 ${item.color}`} />
                <span className="text-neutral-400 text-xs">{item.label}</span>
                {item.highlight && <span className="text-[10px] bg-orange-500/20 text-orange-400 px-1.5 py-0.5">5% Fee</span>}
              </div>
              <p className={`text-xl font-bold ${item.highlight ? 'text-orange-400' : 'text-white'}`}>
                {loading && !stats ? <span className="inline-block w-20 h-6 bg-neutral-800 animate-pulse"></span> : item.value}
              </p>
            </motion.div>
          ))}
        </div>

        {/* Search Bar */}
        <div className="mb-6">
          <div className="flex gap-2">
            <div className="flex-1 relative">
              <Search className="absolute left-4 top-1/2 transform -translate-y-1/2 w-5 h-5 text-neutral-400" />
              <input
                type="text"
                placeholder="Search by Node ID, Proof Signature, or Transaction ID..."
                value={searchQuery}
                onChange={(e) => setSearchQuery(e.target.value)}
                onKeyPress={(e) => e.key === 'Enter' && handleSearch()}
                disabled={loading && !stats}
                className="w-full pl-12 pr-4 py-3 bg-neutral-900 border border-neutral-800 text-white placeholder-neutral-500 focus:outline-none focus:border-accent disabled:opacity-50"
              />
            </div>
            <button
              onClick={handleSearch}
              disabled={loading && !stats}
              className="px-6 py-3 bg-accent hover:bg-accent/90 text-neutral-950 font-semibold transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
            >
              Search
            </button>
          </div>
        </div>

        {/* Tabs */}
        <div className="flex gap-2 mb-6 border-b border-neutral-800 overflow-x-auto">
          {[
            { id: 'epochs', label: 'Epochs', icon: Clock },
            { id: 'proofs', label: 'PoNW Proofs', icon: Hash },
            { id: 'balances', label: 'Balances', icon: Coins },
            { id: 'transactions', label: 'Transactions', icon: ArrowRightLeft }
          ].map((tab) => (
            <button
              key={tab.id}
              onClick={() => {
                if (tab.id === 'epochs') setSelectedNode(null);
                setActiveTab(tab.id as any);
              }}
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
        <div className="bg-neutral-900/50 border border-neutral-800 overflow-hidden min-h-[300px] relative">
          {loading && !epochs.length && !proofs.length && !balances.length ? (
            <div className="absolute inset-0 flex flex-col items-center justify-center">
              <div className="inline-block animate-spin h-12 w-12 border-b-2 border-accent mb-4"></div>
              <p className="text-neutral-400">Loading ledger data...</p>
            </div>
          ) : error && !epochs.length && !proofs.length && !balances.length ? (
            <div className="absolute inset-0 flex flex-col items-center justify-center p-8 text-center">
              <AlertCircle className="w-16 h-16 text-neutral-700 mb-4" />
              <h3 className="text-xl font-semibold text-white mb-2 font-display">Ledger Unavailable</h3>
              <p className="text-neutral-400 max-w-md">
                We couldn't connect to the ledger. Please check if your local NeuroShard node is running.
              </p>
            </div>
          ) : (
            <>
              {/* Epochs Table */}
              {activeTab === 'epochs' && (
                epochs.length === 0 ? (
                  <div className="text-center py-16">
                    <Network className="w-16 h-16 text-neutral-700 mx-auto mb-4" />
                    <h3 className="text-xl font-semibold text-white mb-2 font-display">No Epochs Found</h3>
                    <p className="text-neutral-400 mb-4">The ledger is empty. Start a node to begin generating epochs.</p>
                  </div>
                ) : selectedEpoch ? (
                  /* Epoch Detail View */
                  <div className="p-6">
                    <button 
                      onClick={() => setSelectedEpoch(null)}
                      className="mb-6 text-accent hover:text-accent flex items-center gap-2"
                    >
                      ← Back to Epochs
                    </button>
                    
                    <div className="mb-6">
                      <h2 className="text-2xl font-bold text-white mb-2 font-display">
                        Epoch #{selectedEpoch.epoch_number}
                      </h2>
                      <p className="text-neutral-400 text-sm">
                        {new Date(selectedEpoch.timestamp_start * 1000).toLocaleString()} • 
                        Duration: {selectedEpoch.duration_seconds.toFixed(1)}s
                      </p>
                    </div>
                    
                    {/* Summary Cards */}
                    <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mb-8">
                      <div className="bg-neutral-800/50 p-4">
                        <div className="text-neutral-400 text-xs mb-1">Proofs</div>
                        <div className="text-2xl font-bold text-white">{selectedEpoch.summary.proof_count}</div>
                      </div>
                      <div className="bg-neutral-800/50 p-4">
                        <div className="text-neutral-400 text-xs mb-1">Nodes</div>
                        <div className="text-2xl font-bold text-white">{selectedEpoch.summary.unique_nodes}</div>
                      </div>
                      <div className="bg-neutral-800/50 p-4">
                        <div className="text-neutral-400 text-xs mb-1">Tokens Processed</div>
                        <div className="text-2xl font-bold text-white">{selectedEpoch.summary.total_tokens_processed.toLocaleString()}</div>
                      </div>
                      <div className="bg-emerald-900/30 border border-emerald-500/30 p-4">
                        <div className="text-emerald-400 text-xs mb-1">NEURO Minted</div>
                        <div className="text-2xl font-bold text-emerald-400">{selectedEpoch.summary.total_neuro_minted.toFixed(6)}</div>
                      </div>
                    </div>
                    
                    {/* Reward Breakdown */}
                    <div className="mb-8">
                      <h3 className="text-lg font-semibold text-white mb-4 font-display">Reward Breakdown</h3>
                      <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
                        {Object.entries(selectedEpoch.reward_breakdown).map(([type, data]) => (
                          <div key={type} className="bg-neutral-800/30 p-3">
                            <div className="flex items-center gap-2 mb-2">
                              <span className={`px-2 py-0.5 text-xs font-semibold ${
                                type === 'uptime' ? 'bg-blue-500/20 text-blue-400' :
                                type === 'inference' ? 'bg-purple-500/20 text-purple-400' :
                                type === 'training' ? 'bg-orange-500/20 text-orange-400' :
                                'bg-neutral-500/20 text-neutral-400'
                              }`}>
                                {type.toUpperCase()}
                              </span>
                            </div>
                            <div className="text-white font-semibold">{data.count} proofs</div>
                            <div className="text-green-400 text-sm">+{data.total_reward.toFixed(6)} NEURO</div>
                          </div>
                        ))}
                      </div>
                    </div>
                    
                    {/* Participants */}
                    <div className="mb-8">
                      <h3 className="text-lg font-semibold text-white mb-4 font-display">Participating Nodes</h3>
                      <div className="space-y-2">
                        {selectedEpoch.participants.map((p, idx) => (
                          <div key={p.node_id} className="flex items-center justify-between bg-neutral-800/30 p-3">
                            <div className="flex items-center gap-3">
                              <span className="text-neutral-500 text-sm">#{idx + 1}</span>
                              <code className="text-accent text-sm font-mono">{formatAddress(p.node_id)}</code>
                            </div>
                            <div className="flex items-center gap-4">
                              <span className="text-neutral-400 text-sm">{p.proof_count} proofs</span>
                              <span className="text-green-400 font-semibold">+{p.reward_neuro.toFixed(6)} NEURO</span>
                            </div>
                          </div>
                        ))}
                      </div>
                    </div>
                    
                    {/* Proofs in Epoch */}
                    <div>
                      <h3 className="text-lg font-semibold text-white mb-4 font-display">Proofs in Epoch</h3>
                      <div className="overflow-x-auto">
                        <table className="w-full whitespace-nowrap text-sm">
                          <thead className="bg-neutral-800/50">
                            <tr>
                              <th className="px-4 py-3 text-left text-neutral-300">Time</th>
                              <th className="px-4 py-3 text-left text-neutral-300">Type</th>
                              <th className="px-4 py-3 text-left text-neutral-300">Node</th>
                              <th className="px-4 py-3 text-left text-neutral-300">Uptime</th>
                              <th className="px-4 py-3 text-left text-neutral-300">Tokens</th>
                              <th className="px-4 py-3 text-left text-neutral-300">Reward</th>
                            </tr>
                          </thead>
                          <tbody>
                            {selectedEpoch.proofs.map((proof) => (
                              <tr key={proof.signature} className="border-t border-neutral-800">
                                <td className="px-4 py-3 text-neutral-400">{new Date(proof.timestamp * 1000).toLocaleTimeString()}</td>
                                <td className="px-4 py-3">
                                  <span className={`px-2 py-0.5 text-xs font-semibold ${
                                    proof.proof_type === 'uptime' ? 'bg-blue-500/20 text-blue-400' :
                                    proof.proof_type === 'inference' ? 'bg-purple-500/20 text-purple-400' :
                                    proof.proof_type === 'training' ? 'bg-orange-500/20 text-orange-400' :
                                    'bg-neutral-500/20 text-neutral-400'
                                  }`}>
                                    {proof.proof_type?.toUpperCase() || 'UPTIME'}
                                  </span>
                                </td>
                                <td className="px-4 py-3"><code className="text-accent font-mono">{formatAddress(proof.node_id)}</code></td>
                                <td className="px-4 py-3 text-neutral-300">{proof.uptime_seconds}s</td>
                                <td className="px-4 py-3 text-neutral-300">{(proof.tokens_processed ?? 0).toLocaleString()}</td>
                                <td className="px-4 py-3 text-green-400 font-semibold">+{proof.reward_neuro.toFixed(6)}</td>
                              </tr>
                            ))}
                          </tbody>
                        </table>
                      </div>
                    </div>
                  </div>
                ) : (
                  <div>
                    <div className="overflow-x-auto">
                      <table className="w-full whitespace-nowrap">
                        <thead className="bg-neutral-800/50">
                          <tr>
                            <th className="px-6 py-4 text-left text-neutral-300 font-semibold">Epoch</th>
                            <th className="px-6 py-4 text-left text-neutral-300 font-semibold">Time</th>
                            <th className="px-6 py-4 text-left text-neutral-300 font-semibold">Proofs</th>
                            <th className="px-6 py-4 text-left text-neutral-300 font-semibold">Nodes</th>
                            <th className="px-6 py-4 text-left text-neutral-300 font-semibold">Tokens</th>
                            <th className="px-6 py-4 text-left text-neutral-300 font-semibold">Minted</th>
                          </tr>
                        </thead>
                        <tbody>
                          {epochs.map((epoch, idx) => (
                            <motion.tr
                              key={epoch.epoch_id}
                              initial={{ opacity: 0 }}
                              animate={{ opacity: 1 }}
                              transition={{ delay: idx * 0.02 }}
                              className={`border-t border-neutral-800 hover:bg-neutral-800/30 cursor-pointer ${
                                epoch.epoch_number === 1 ? 'bg-emerald-950/20' : ''
                              }`}
                              onClick={() => loadEpochDetails(epoch.epoch_id)}
                            >
                              <td className="px-6 py-4">
                                <span className={`font-mono ${epoch.epoch_number === 1 ? 'text-emerald-400' : 'text-accent'}`}>
                                  #{epoch.epoch_number}
                                </span>
                                {epoch.epoch_number === 1 && (
                                  <span className="ml-2 text-xs bg-emerald-500/20 text-emerald-400 px-2 py-0.5">FIRST</span>
                                )}
                              </td>
                              <td className="px-6 py-4 text-neutral-300 text-sm">{formatTime(epoch.timestamp_start)}</td>
                              <td className="px-6 py-4 text-neutral-300">{epoch.proof_count}</td>
                              <td className="px-6 py-4 text-neutral-300">{epoch.unique_nodes}</td>
                              <td className="px-6 py-4 text-neutral-300">{(epoch.total_tokens_processed ?? 0).toLocaleString()}</td>
                              <td className="px-6 py-4 text-green-400 font-semibold">{(epoch.total_neuro_minted ?? 0).toFixed(4)}</td>
                            </motion.tr>
                          ))}
                        </tbody>
                      </table>
                    </div>
                    
                    {/* Loading overlay for epoch details */}
                    {loadingEpoch && (
                      <div className="absolute inset-0 bg-neutral-900/80 flex items-center justify-center">
                        <div className="inline-block animate-spin h-8 w-8 border-b-2 border-accent"></div>
                      </div>
                    )}
                    
                    {/* Epochs Pagination */}
                    <div className="flex items-center justify-between px-6 py-4 bg-neutral-800/30 border-t border-neutral-800">
                      <div className="text-neutral-400 text-sm">
                        Showing {epochsOffset + 1} - {Math.min(epochsOffset + epochs.length, epochsTotal)} of {epochsTotal} epochs
                      </div>
                      <div className="flex gap-2">
                        <button
                          onClick={() => { setEpochsOffset(0); }}
                          disabled={epochsOffset === 0}
                          className="px-3 py-1 bg-neutral-700 hover:bg-neutral-600 disabled:opacity-50 disabled:cursor-not-allowed text-sm text-white"
                        >
                          Latest
                        </button>
                        <button
                          onClick={() => { setEpochsOffset(Math.max(0, epochsOffset - EPOCHS_LIMIT)); }}
                          disabled={epochsOffset === 0}
                          className="px-3 py-1 bg-neutral-700 hover:bg-neutral-600 disabled:opacity-50 disabled:cursor-not-allowed text-sm text-white"
                        >
                          ← Newer
                        </button>
                        <button
                          onClick={() => { setEpochsOffset(epochsOffset + EPOCHS_LIMIT); }}
                          disabled={epochsOffset + EPOCHS_LIMIT >= epochsTotal}
                          className="px-3 py-1 bg-neutral-700 hover:bg-neutral-600 disabled:opacity-50 disabled:cursor-not-allowed text-sm text-white"
                        >
                          Older →
                        </button>
                        <button
                          onClick={() => { setEpochsOffset(Math.max(0, epochsTotal - EPOCHS_LIMIT)); }}
                          disabled={epochsOffset + EPOCHS_LIMIT >= epochsTotal}
                          className="px-3 py-1 bg-accent hover:bg-accent/90 disabled:opacity-50 disabled:cursor-not-allowed text-sm text-neutral-950 font-semibold"
                        >
                          First Epoch ⟶
                        </button>
                      </div>
                    </div>
                  </div>
                )
              )}

              {/* Proofs Table */}
              {activeTab === 'proofs' && (
                proofs.length === 0 ? (
                  <div className="text-center py-16">
                    <Hash className="w-16 h-16 text-neutral-700 mx-auto mb-4" />
                    <h3 className="text-xl font-semibold text-white mb-2 font-display">No Proofs Found</h3>
                    <p className="text-neutral-400 mb-4">
                      {selectedNode ? `No proofs found for node ${formatAddress(selectedNode)}.` : 'The ledger is empty.'}
                    </p>
                    {selectedNode && (
                      <button onClick={() => { setSelectedNode(null); setSearchQuery(''); setProofsOffset(0); loadAllData(); }} className="text-accent hover:text-accent text-sm">
                        Clear filter
                      </button>
                    )}
                  </div>
                ) : (
                  <div>
                    <div className="overflow-x-auto">
                      <table className="w-full whitespace-nowrap">
                        <thead className="bg-neutral-800/50">
                          <tr>
                            <th className="px-6 py-4 text-left text-neutral-300 font-semibold">Signature</th>
                            <th className="px-6 py-4 text-left text-neutral-300 font-semibold">Node</th>
                            <th className="px-6 py-4 text-left text-neutral-300 font-semibold">Type</th>
                            <th className="px-6 py-4 text-left text-neutral-300 font-semibold">Time</th>
                            <th className="px-6 py-4 text-left text-neutral-300 font-semibold">Tokens</th>
                            <th className="px-6 py-4 text-left text-neutral-300 font-semibold">Reward</th>
                            <th className="px-6 py-4 text-left text-neutral-300 font-semibold">Epoch</th>
                          </tr>
                        </thead>
                        <tbody>
                          {proofs.map((proof, idx) => (
                            <motion.tr
                              key={proof.signature}
                              initial={{ opacity: 0 }}
                              animate={{ opacity: 1 }}
                              transition={{ delay: idx * 0.02 }}
                              className={`border-t border-neutral-800 hover:bg-neutral-800/30 cursor-pointer ${
                                proof.node_id === 'GENESIS' ? 'bg-emerald-950/30' : ''
                              }`}
                            >
                              <td className="px-6 py-4">
                                <code className={`text-sm font-mono ${proof.node_id === 'GENESIS' ? 'text-emerald-400' : 'text-accent'}`}>
                                  {proof.node_id === 'GENESIS' ? 'GENESIS_BLOCK' : formatAddress(proof.signature)}
                                </code>
                              </td>
                              <td className="px-6 py-4">
                                <code className={`text-sm font-mono ${proof.node_id === 'GENESIS' ? 'text-emerald-400' : 'text-purple-400'}`}>
                                  {proof.node_id === 'GENESIS' ? 'GENESIS' : formatAddress(proof.node_id)}
                                </code>
                              </td>
                              <td className="px-6 py-4">
                                <span className={`px-2 py-1 text-xs font-semibold ${
                                  proof.proof_type === 'GENESIS' ? 'bg-emerald-500/20 text-emerald-400' :
                                  proof.proof_type === 'uptime' ? 'bg-blue-500/20 text-blue-400' :
                                  proof.proof_type === 'inference' ? 'bg-purple-500/20 text-purple-400' :
                                  proof.proof_type === 'training' ? 'bg-orange-500/20 text-orange-400' :
                                  'bg-neutral-500/20 text-neutral-400'
                                }`}>
                                  {proof.proof_type?.toUpperCase() || 'UNKNOWN'}
                                </span>
                              </td>
                              <td className="px-6 py-4 text-neutral-300 text-sm">{formatTime(proof.timestamp)}</td>
                              <td className="px-6 py-4 text-neutral-300">{(proof.tokens_processed ?? 0).toLocaleString()}</td>
                              <td className="px-6 py-4 text-green-400 font-semibold">{(proof.reward_neuro ?? 0).toFixed(6)}</td>
                              <td className="px-6 py-4 text-neutral-400">#{proof.epoch_id}</td>
                            </motion.tr>
                          ))}
                        </tbody>
                      </table>
                    </div>
                    
                    {/* Pagination */}
                    <div className="flex items-center justify-between px-6 py-4 bg-neutral-800/30 border-t border-neutral-800">
                      <div className="text-neutral-400 text-sm">
                        Showing {proofsOffset + 1} - {Math.min(proofsOffset + proofs.length, proofsTotal)} of {proofsTotal} proofs
                      </div>
                      <div className="flex gap-2">
                        <button
                          onClick={() => { setProofsOffset(0); }}
                          disabled={proofsOffset === 0}
                          className="px-3 py-1 bg-neutral-700 hover:bg-neutral-600 disabled:opacity-50 disabled:cursor-not-allowed text-sm text-white"
                        >
                          First
                        </button>
                        <button
                          onClick={() => { setProofsOffset(Math.max(0, proofsOffset - PROOFS_LIMIT)); }}
                          disabled={proofsOffset === 0}
                          className="px-3 py-1 bg-neutral-700 hover:bg-neutral-600 disabled:opacity-50 disabled:cursor-not-allowed text-sm text-white"
                        >
                          ← Newer
                        </button>
                        <button
                          onClick={() => { setProofsOffset(proofsOffset + PROOFS_LIMIT); }}
                          disabled={proofsOffset + PROOFS_LIMIT >= proofsTotal}
                          className="px-3 py-1 bg-neutral-700 hover:bg-neutral-600 disabled:opacity-50 disabled:cursor-not-allowed text-sm text-white"
                        >
                          Older →
                        </button>
                        <button
                          onClick={() => { setProofsOffset(Math.max(0, proofsTotal - PROOFS_LIMIT)); }}
                          disabled={proofsOffset + PROOFS_LIMIT >= proofsTotal}
                          className="px-3 py-1 bg-accent hover:bg-accent/90 disabled:opacity-50 disabled:cursor-not-allowed text-sm text-neutral-950 font-semibold"
                        >
                          Genesis ⟶
                        </button>
                      </div>
                    </div>
                  </div>
                )
              )}

              {/* Balances Table */}
              {activeTab === 'balances' && (
                balances.length === 0 ? (
                  <div className="text-center py-16">
                    <Coins className="w-16 h-16 text-neutral-700 mx-auto mb-4" />
                    <h3 className="text-xl font-semibold text-white mb-2 font-display">No Balances Found</h3>
                    <p className="text-neutral-400">The ledger is empty.</p>
                  </div>
                ) : (
                  <div className="overflow-x-auto">
                    <table className="w-full whitespace-nowrap">
                      <thead className="bg-neutral-800/50">
                        <tr>
                          <th className="px-6 py-4 text-left text-neutral-300 font-semibold">Rank</th>
                          <th className="px-6 py-4 text-left text-neutral-300 font-semibold">Node Address</th>
                          <th className="px-6 py-4 text-left text-neutral-300 font-semibold">Balance</th>
                          <th className="px-6 py-4 text-left text-neutral-300 font-semibold">Earned</th>
                          <th className="px-6 py-4 text-left text-neutral-300 font-semibold">Spent</th>
                          <th className="px-6 py-4 text-left text-neutral-300 font-semibold">Staked</th>
                          <th className="px-6 py-4 text-left text-neutral-300 font-semibold">Multiplier</th>
                        </tr>
                      </thead>
                      <tbody>
                        {balances.map((balance, idx) => (
                          <motion.tr
                            key={balance.node_id}
                            initial={{ opacity: 0 }}
                            animate={{ opacity: 1 }}
                            transition={{ delay: idx * 0.02 }}
                            className={`border-t border-neutral-800 hover:bg-neutral-800/30 cursor-pointer ${balance.is_burn_address ? 'bg-orange-950/20' : ''}`}
                            onClick={() => {
                              if (!balance.is_burn_address) {
                                setSearchQuery(balance.node_id);
                                setSelectedNode(balance.node_id);
                                setActiveTab('proofs');
                              }
                            }}
                          >
                            <td className="px-6 py-4 text-neutral-400">
                              {balance.is_burn_address ? <Flame className="w-4 h-4 text-orange-500" /> : `#${idx + 1}`}
                            </td>
                            <td className="px-6 py-4">
                              <code className={`text-sm font-mono ${balance.is_burn_address ? 'text-orange-400' : 'text-accent'}`}>
                                {balance.is_burn_address ? 'BURN ADDRESS' : formatAddress(balance.address)}
                              </code>
                            </td>
                            <td className={`px-6 py-4 font-semibold text-lg ${balance.is_burn_address ? 'text-orange-400' : 'text-green-400'}`}>
                              {(balance.balance_neuro ?? 0).toFixed(6)}
                            </td>
                            <td className="px-6 py-4 text-emerald-400">{balance.total_earned?.toFixed(4) || '-'}</td>
                            <td className="px-6 py-4 text-red-400">{balance.total_spent?.toFixed(4) || '-'}</td>
                            <td className="px-6 py-4 text-yellow-400 font-semibold">
                              {(balance.staked_neuro ?? 0) > 0 ? (balance.staked_neuro ?? 0).toFixed(2) : '-'}
                            </td>
                            <td className="px-6 py-4 text-blue-400">
                              {(balance.stake_multiplier ?? 0) > 1 ? `${(balance.stake_multiplier ?? 1).toFixed(2)}x` : '-'}
                            </td>
                          </motion.tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                )
              )}

              {/* Transactions Table */}
              {activeTab === 'transactions' && (
                transactions.length === 0 ? (
                  <div className="text-center py-16">
                    <ArrowRightLeft className="w-16 h-16 text-neutral-700 mx-auto mb-4" />
                    <h3 className="text-xl font-semibold text-white mb-2 font-display">No Transactions Found</h3>
                    <p className="text-neutral-400">No transfers have been recorded yet.</p>
                  </div>
                ) : (
                  <div className="overflow-x-auto">
                    <table className="w-full whitespace-nowrap">
                      <thead className="bg-neutral-800/50">
                        <tr>
                          <th className="px-6 py-4 text-left text-neutral-300 font-semibold">TX ID</th>
                          <th className="px-6 py-4 text-left text-neutral-300 font-semibold">From</th>
                          <th className="px-6 py-4 text-left text-neutral-300 font-semibold">To</th>
                          <th className="px-6 py-4 text-left text-neutral-300 font-semibold">Amount</th>
                          <th className="px-6 py-4 text-left text-neutral-300 font-semibold">Fee</th>
                          <th className="px-6 py-4 text-left text-neutral-300 font-semibold">Burned</th>
                          <th className="px-6 py-4 text-left text-neutral-300 font-semibold">Time</th>
                        </tr>
                      </thead>
                      <tbody>
                        {transactions.map((tx, idx) => (
                          <motion.tr
                            key={tx.tx_id}
                            initial={{ opacity: 0 }}
                            animate={{ opacity: 1 }}
                            transition={{ delay: idx * 0.02 }}
                            className="border-t border-neutral-800 hover:bg-neutral-800/30"
                          >
                            <td className="px-6 py-4"><code className="text-accent text-sm font-mono">{formatAddress(tx.tx_id)}</code></td>
                            <td className="px-6 py-4"><code className="text-purple-400 text-sm font-mono">{formatAddress(tx.from_id)}</code></td>
                            <td className="px-6 py-4">
                              <code className={`text-sm font-mono ${tx.type === 'burn' ? 'text-orange-400' : 'text-green-400'}`}>
                                {tx.type === 'burn' ? 'BURN' : formatAddress(tx.to_id)}
                              </code>
                            </td>
                            <td className="px-6 py-4 text-white font-semibold">{(tx.amount ?? 0).toFixed(6)}</td>
                            <td className="px-6 py-4 text-neutral-400">{(tx.fee ?? 0).toFixed(6)}</td>
                            <td className="px-6 py-4 text-orange-400 font-semibold">{(tx.burn_amount ?? 0).toFixed(6)}</td>
                            <td className="px-6 py-4 text-neutral-300 text-sm">{formatTime(tx.timestamp)}</td>
                          </motion.tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                )
              )}

            </>
          )}
        </div>

        {/* Info Box */}
        <div className="mt-8 bg-neutral-900/50 border border-neutral-800 p-6">
          <h3 className="text-lg font-semibold text-white mb-3 flex items-center gap-2 font-display">
            <Clock className="w-5 h-5 text-accent" />
            About the NEURO Ledger
          </h3>
          <div className="text-neutral-400 space-y-2 text-sm">
            <p>
              <strong className="text-white">Proof of Neural Work (PoNW):</strong> Nodes earn NEURO by performing actual neural network computation. 
              Rewards: 0.1 NEURO/minute uptime + 0.9 NEURO per million tokens processed.
            </p>
            <p>
              <strong className="text-white">Staking Multiplier:</strong> Stake NEURO to earn bonus rewards. 
              Formula: 1 + (0.1 × Stake / 1000). Staking 1000 NEURO = 1.1x multiplier.
            </p>
            <p>
              <strong className="text-orange-400">5% Fee Burn:</strong> All spending (inference requests, transfers) includes a 5% fee that is permanently burned. 
              This creates deflationary pressure as network usage grows.
            </p>
            <p>
              <strong className="text-white">Distributed Ledger:</strong> Unlike a blockchain, NeuroShard uses a gossip-based distributed ledger. 
              Each node maintains a local copy with cryptographic proof verification.
            </p>
          </div>
        </div>
      </div>
    </div>
    </>
  );
};
