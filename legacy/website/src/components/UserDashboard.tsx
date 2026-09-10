import { useState, useEffect } from 'react';
import { useAuth } from '../contexts/AuthContext';
import { motion } from 'framer-motion';
import { Copy, Check, Activity, Download, Lock, RefreshCw, Coins, ShieldCheck, Send, ArrowRight, TrendingUp, Clock, Unlock, Wallet } from 'lucide-react';
import axios from 'axios';
import { API_URL } from '../config/api';

export const UserDashboard = () => {
  const { user, token, refreshUser, hasWallet } = useAuth();
  const [nodeStatus, setNodeStatus] = useState<{ active: boolean; detail?: string }>({ active: false });
  const [neuroBalance, setNeuroBalance] = useState<number | null>(null);
  const [stakedBalance, setStakedBalance] = useState<number | null>(null);
  const [stakeMultiplier, setStakeMultiplier] = useState<number>(1.0);
  const [stakeLockUntil, setStakeLockUntil] = useState<number | null>(null);
  const [copied, setCopied] = useState<string>('');
  const [refreshing, setRefreshing] = useState(false);
  
  // Transfer State
  const [transferRecipient, setTransferRecipient] = useState("");
  const [transferAmount, setTransferAmount] = useState("");
  const [transferStatus, setTransferStatus] = useState<'idle' | 'processing' | 'success' | 'error'>('idle');
  const [transferMessage, setTransferMessage] = useState("");

  // Staking State
  const [stakeAmount, setStakeAmount] = useState("");
  const [stakeDays, setStakeDays] = useState("30");
  const [stakeStatus, setStakeStatus] = useState<'idle' | 'processing' | 'success' | 'error'>('idle');
  const [stakeMessage, setStakeMessage] = useState("");

  useEffect(() => {
    if (user && hasWallet) {
      checkNodeStatus();
      fetchNeuroBalance();
    }
  }, [token, user, hasWallet]);

  const fetchNeuroBalance = async () => {
    if (!user?.node_id) return;
    try {
      const response = await axios.get(`${API_URL}/api/node/neuro`, {
        params: { node_id: user.node_id }
      });
      const neuroBalance = response.data.neuro_balance || 0;
      const stakedBalance = response.data.staked_balance || 0;
      const stakeMultiplier = response.data.stake_multiplier || 1.0;
      const stakeLockUntil = response.data.stake_locked_until || null;
      
      const finalStakedBalance = stakedBalance > 0 ? stakedBalance : 0;
      const finalStakeMultiplier = finalStakedBalance > 0 ? stakeMultiplier : 1.0;
      
      setNeuroBalance(neuroBalance);
      setStakedBalance(finalStakedBalance);
      setStakeMultiplier(finalStakeMultiplier);
      setStakeLockUntil(finalStakedBalance > 0 ? stakeLockUntil : null);
    } catch (error) {
      console.error("Failed to fetch NEURO balance", error);
      setNeuroBalance(0);
      setStakedBalance(0);
      setStakeMultiplier(1.0);
      setStakeLockUntil(null);
    }
  };

  const handleRefresh = async () => {
    setRefreshing(true);
    await Promise.all([
      checkNodeStatus(),
      refreshUser(),
      fetchNeuroBalance()
    ]);
    setRefreshing(false);
  };

  const checkNodeStatus = async () => {
    if (!token) return;
    try {
      const response = await axios.get(`${API_URL}/api/users/me/node_status`, {
        headers: { Authorization: `Bearer ${token}` }
      });
      setNodeStatus(response.data);
    } catch (error) {
      console.error("Failed to check node status", error);
      setNodeStatus({ active: false, detail: "Connection Error" });
    }
  };

  const copyToClipboard = (text: string, label: string) => {
    navigator.clipboard.writeText(text);
    setCopied(label);
    setTimeout(() => setCopied(''), 2000);
  };

  const handleTransfer = async (e: React.FormEvent) => {
    e.preventDefault();
    setTransferStatus('processing');
    setTransferMessage("");

    try {
        if (!user?.node_id) throw new Error("No wallet found");
        const amount = parseFloat(transferAmount);
        if (isNaN(amount) || amount <= 0) throw new Error("Invalid amount");
        if (amount > (neuroBalance || 0)) throw new Error("Insufficient balance");

        await axios.post(`${API_URL}/api/ledger/transfer`, {
            sender_node_id: user.node_id,
            recipient_address: transferRecipient,
            amount: amount
        }, {
            headers: { Authorization: `Bearer ${token}` }
        });

        setTransferStatus('success');
        setTransferMessage(`Successfully sent ${amount} NEURO!`);
        setTransferAmount("");
        setTransferRecipient("");
        fetchNeuroBalance();
        
        setTimeout(() => setTransferStatus('idle'), 3000);
    } catch (err: any) {
        setTransferStatus('error');
        setTransferMessage(err.response?.data?.detail || err.message || "Transfer failed");
    }
  };

  const handleStake = async (e: React.FormEvent) => {
    e.preventDefault();
    setStakeStatus('processing');
    setStakeMessage("");

    try {
        if (!user?.node_id) throw new Error("No wallet found");
        const amount = parseFloat(stakeAmount);
        const days = parseInt(stakeDays);
        
        if (isNaN(amount) || amount <= 0) throw new Error("Invalid amount");
        if (amount > (neuroBalance || 0)) throw new Error("Insufficient balance");
        if (isNaN(days) || days < 1 || days > 365) throw new Error("Duration must be 1-365 days");

        await axios.post(`${API_URL}/api/ledger/stake`, {
            sender_node_id: user.node_id,
            amount: amount,
            duration_days: days
        }, {
            headers: { Authorization: `Bearer ${token}` }
        });

        setStakeStatus('success');
        const newMultiplier = 1.0 + (0.1 * ((stakedBalance || 0) + amount) / 1000);
        setStakeMessage(`Staked ${amount} NEURO! New multiplier: ${newMultiplier.toFixed(2)}x`);
        setStakeAmount("");
        fetchNeuroBalance();
        
        setTimeout(() => setStakeStatus('idle'), 3000);
    } catch (err: any) {
        setStakeStatus('error');
        setStakeMessage(err.response?.data?.detail || err.message || "Staking failed");
    }
  };

  const handleUnstake = async () => {
    setStakeStatus('processing');
    setStakeMessage("");

    try {
        if (!user?.node_id) throw new Error("No wallet found");
        
        if (stakeLockUntil && stakeLockUntil > Date.now() / 1000) {
            const lockDate = new Date(stakeLockUntil * 1000);
            throw new Error(`Stake locked until ${lockDate.toLocaleDateString()}`);
        }

        await axios.post(`${API_URL}/api/ledger/unstake`, {
            sender_node_id: user.node_id
        }, {
            headers: { Authorization: `Bearer ${token}` }
        });

        setStakeStatus('success');
        setStakeMessage(`Unstaked ${stakedBalance} NEURO! Returned to balance.`);
        fetchNeuroBalance();
        
        setTimeout(() => setStakeStatus('idle'), 3000);
    } catch (err: any) {
        setStakeStatus('error');
        setStakeMessage(err.response?.data?.detail || err.message || "Unstaking failed");
    }
  };

  const isStakeLocked = stakeLockUntil && stakeLockUntil > Date.now() / 1000;

  if (!user) return null;

  // No wallet = invalid account, redirect to signup
  if (!hasWallet) {
    window.location.href = '/signup';
    return null;
  }

  return (
    <section className="pt-20 sm:pt-32 pb-24 min-h-screen bg-neutral-950 relative overflow-hidden">

      <div className="max-w-7xl mx-auto px-4 sm:px-6 relative z-10">
        
        {/* Welcome Header */}
        <motion.div 
          initial={{ opacity: 0, y: -20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.5 }}
          className="mb-12 flex flex-col sm:flex-row justify-between items-start sm:items-center gap-4"
        >
          <div>
            <h1 className="text-5xl font-bold text-white mb-3 text-white font-display">
              Dashboard
            </h1>
            <p className="text-neutral-400 text-lg">Manage your NeuroShard wallet and view earnings.</p>
          </div>
          <button
            onClick={handleRefresh}
            disabled={refreshing}
            className="flex items-center gap-2 px-5 py-2.5 bg-neutral-800 hover:bg-neutral-700 text-white border border-neutral-700/50 transition-all duration-200 disabled:opacity-50"
          >
            <RefreshCw className={`w-4 h-4 ${refreshing ? 'animate-spin' : ''}`} />
            <span className="text-sm font-semibold">Refresh</span>
          </button>
        </motion.div>

        {/* Stats Grid */}
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6 mb-8">
          
          {/* NEURO Balance Card */}
          <motion.div 
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.4 }}
            className="group relative bg-neutral-900 border border-neutral-800/50 p-6 overflow-hidden transition-all duration-300"
          >
                        <div className="relative z-10">
              <div className="flex items-center justify-between mb-4">
                <h3 className="text-neutral-400 text-xs font-semibold uppercase tracking-wider flex items-center gap-2 font-display">
                  <div className="p-1.5 bg-yellow-500/10 border border-yellow-500/20">
                    <Coins className="w-3.5 h-3.5 text-yellow-400" />
                  </div>
                  Liquid NEURO
                </h3>
              </div>
              <div className="text-4xl font-bold text-white mb-2 font-mono tracking-tight">
                {neuroBalance !== null ? neuroBalance.toFixed(4) : '...'}
              </div>
              <p className="text-xs text-neutral-500 font-medium">Available to spend</p>
            </div>
          </motion.div>

          {/* Staked Balance Card */}
          <motion.div 
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.05, duration: 0.4 }}
            className="group relative bg-neutral-900 border border-neutral-800/50 p-6 overflow-hidden transition-all duration-300"
          >
                        <div className="relative z-10">
              <div className="flex items-center justify-between mb-4">
                <h3 className="text-neutral-400 text-xs font-semibold uppercase tracking-wider flex items-center gap-2 font-display">
                  <div className="p-1.5 bg-accent/10 border border-accent/20">
                    <ShieldCheck className="w-3.5 h-3.5 text-accent" />
                  </div>
                  Staked NEURO
                </h3>
              </div>
              <div className="flex items-baseline gap-2 mb-2">
                <div className="text-4xl font-bold text-white font-mono tracking-tight">
                  {stakedBalance !== null ? stakedBalance.toFixed(2) : '...'}
                </div>
                <div className="px-2 py-1 bg-accent/10 border border-accent/20">
                  <span className="text-sm font-bold text-accent">{stakeMultiplier.toFixed(2)}x</span>
                </div>
              </div>
              <p className="text-xs text-neutral-500 font-medium">
                {isStakeLocked 
                  ? `Locked until ${new Date(stakeLockUntil! * 1000).toLocaleDateString()}`
                  : stakedBalance && stakedBalance > 0 
                    ? 'Unlocked - can unstake'
                    : 'Stake to earn multiplier'
                }
              </p>
            </div>
          </motion.div>

          {/* Node Status Card */}
          <motion.div 
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.1, duration: 0.4 }}
            className="group relative bg-neutral-900 border border-neutral-800/50 p-6 overflow-hidden transition-all duration-300 lg:col-span-2"
          >
                        <div className="relative z-10">
              <div className="flex items-center justify-between mb-4">
                <h3 className="text-neutral-400 text-xs font-semibold uppercase tracking-wider flex items-center gap-2 font-display">
                  <div className="p-1.5 bg-accent/10 border border-accent/20">
                    <Activity className="w-3.5 h-3.5 text-accent" />
                  </div>
                  Node Status
                </h3>
                <a 
                  href="/download" 
                  className="flex items-center gap-2 px-3 py-1.5 bg-neutral-800/50 hover:bg-neutral-700/50 text-xs font-semibold text-neutral-300 transition-all border border-neutral-700/50 hover:border-accent/30"
                >
                  <Download className="w-3 h-3" /> Download Node
                </a>
              </div>
              <div className="flex items-center gap-4">
                {nodeStatus.active ? (
                  <div className="flex items-center gap-3 px-5 py-3 bg-green-500/10 border border-green-500/30 text-green-400">
                    <span className="relative flex h-3 w-3">
                      <span className="animate-ping absolute inline-flex h-full w-full bg-green-400 opacity-75"></span>
                      <span className="relative inline-flex h-3 w-3 bg-green-500"></span>
                    </span>
                    <span className="font-bold text-sm">Active & Earning</span>
                  </div>
                ) : (
                  <div className="flex items-center gap-3 px-5 py-3 bg-red-500/10 border border-red-500/30 text-red-400">
                    <div className="w-3 h-3 bg-red-500"></div>
                    <span className="font-bold text-sm">Offline</span>
                  </div>
                )}
              </div>
              {!nodeStatus.active && (
                <p className="mt-4 text-xs text-neutral-500 font-medium">
                  Download the node app and use your recovery phrase to start earning.
                </p>
              )}
            </div>
          </motion.div>
        </div>

        {/* Wallet Address Section */}
        <motion.div 
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.15, duration: 0.4 }}
          className="relative bg-neutral-900 border border-neutral-800/50 p-6 mb-8"
        >
          <div className="flex items-center gap-3 mb-4">
            <div className="p-2 bg-accent/10 border border-accent/20">
              <Wallet className="w-5 h-5 text-accent" />
            </div>
            <div>
              <h2 className="text-lg font-bold text-white font-display">Wallet Address</h2>
              <p className="text-xs text-neutral-400">Your NEURO earning address & network identity</p>
            </div>
          </div>
          
          <div className="flex items-center gap-3">
            <div className="flex-1 bg-neutral-950/50 border border-neutral-700/50 p-4 font-mono text-accent text-sm break-all">
              {user.node_id}
            </div>
            <button 
              onClick={() => copyToClipboard(user.node_id!, 'node')}
              className="p-3 bg-neutral-800 hover:bg-neutral-700 text-white transition-all border border-neutral-700/50"
            >
              {copied === 'node' ? <Check className="w-4 h-4 text-green-400" /> : <Copy className="w-4 h-4" />}
            </button>
          </div>
          <p className="mt-3 text-xs text-neutral-500">Share this to receive NEURO • Used for PoNW verification</p>
        </motion.div>

        {/* Staking Section */}
        <div className="mb-8">
          <motion.div 
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.25, duration: 0.4 }}
            className="relative bg-neutral-900 border border-accent/30 p-8 md:p-10 overflow-hidden"
          >
            <div className="absolute top-0 left-0 w-full h-1 bg-accent"></div>
                        
            <div className="relative z-10">
              <div className="flex items-center gap-3 mb-8">
                <div className="p-2 bg-accent/10 border border-accent/20">
                  <TrendingUp className="w-6 h-6 text-accent" />
                </div>
                <div>
                  <h2 className="text-2xl font-bold text-white font-display">Stake NEURO</h2>
                  <p className="text-neutral-400 text-sm">Earn reward multipliers on all your earnings</p>
                </div>
              </div>
              
              <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
                {/* Stake Form */}
                <form onSubmit={handleStake} className="space-y-5">
                  <div className="bg-neutral-900/80 p-6 border border-neutral-700/50">
                    <div className="mb-4 p-3 bg-accent/5 border border-accent/10">
                      <p className="text-sm text-neutral-300 leading-relaxed">
                        Stake NEURO to earn a reward multiplier on all your earnings.
                      </p>
                      <p className="text-sm font-semibold text-accent mt-2">
                        +10% bonus per 1,000 NEURO staked!
                      </p>
                    </div>
                    
                    <div className="grid grid-cols-2 gap-4">
                      <div>
                        <label className="block text-xs font-semibold text-neutral-400 mb-2 uppercase tracking-wider">Amount</label>
                        <input 
                          type="number" 
                          step="0.01"
                          min="0"
                          placeholder="0.00"
                          value={stakeAmount}
                          onChange={(e) => setStakeAmount(e.target.value)}
                          className="w-full bg-neutral-950/50 border border-neutral-700/50 p-3.5 text-white placeholder-neutral-600 focus:outline-none focus:ring-0 focus:border-accent transition-all font-mono text-sm"
                        />
                      </div>
                      <div>
                        <label className="block text-xs font-semibold text-neutral-400 mb-2 uppercase tracking-wider">Lock Period</label>
                        <select 
                          value={stakeDays}
                          onChange={(e) => setStakeDays(e.target.value)}
                          className="w-full bg-neutral-950/50 border border-neutral-700/50 p-3.5 text-white focus:outline-none focus:ring-0 focus:border-accent transition-all text-sm"
                        >
                          <option value="7">7 days</option>
                          <option value="30">30 days</option>
                          <option value="90">90 days</option>
                          <option value="180">180 days</option>
                          <option value="365">365 days</option>
                        </select>
                      </div>
                    </div>
                    
                    <div className="mt-4 flex items-center justify-between p-2 bg-neutral-950/30">
                      <span className="text-xs text-neutral-500 font-medium">Available Balance</span>
                      <span className="text-sm font-bold text-white font-mono">{neuroBalance?.toFixed(4)} NEURO</span>
                    </div>
                  </div>
                  
                  <button 
                    type="submit"
                    disabled={stakeStatus === 'processing' || !stakeAmount}
                    className={`w-full py-4 font-bold flex items-center justify-center gap-2 transition-all duration-200 ${
                      stakeStatus === 'processing' 
                        ? 'bg-neutral-800 text-neutral-400 cursor-not-allowed'
                        : 'bg-accent hover:bg-accent/90 text-neutral-950'
                    }`}
                  >
                    {stakeStatus === 'processing' ? (
                      <><RefreshCw className="w-5 h-5 animate-spin" /> Processing...</>
                    ) : (
                      <><Lock className="w-5 h-5" /> Stake NEURO</>
                    )}
                  </button>
                </form>
                
                {/* Current Stake Info */}
                <div className="bg-neutral-900/80 p-6 border border-neutral-700/50">
                  <div className="flex items-center gap-2 mb-6">
                    <div className="p-1.5 bg-neutral-800/30">
                      <Clock className="w-4 h-4 text-neutral-400" />
                    </div>
                    <h3 className="text-lg font-bold text-white font-display">Current Stake</h3>
                  </div>
                  
                  <div className="space-y-4">
                    <div className="flex justify-between items-center p-3 bg-neutral-950/30">
                      <span className="text-neutral-400 text-sm font-medium">Staked Amount</span>
                      <span className="text-white font-bold font-mono text-lg">{stakedBalance?.toFixed(2) || '0.00'} NEURO</span>
                    </div>
                    <div className="flex justify-between items-center p-3 bg-neutral-950/30">
                      <span className="text-neutral-400 text-sm font-medium">Reward Multiplier</span>
                      <span className="px-3 py-1 bg-accent/10 border border-accent/30 text-accent font-bold">{stakeMultiplier.toFixed(2)}x</span>
                    </div>
                    <div className="flex justify-between items-center p-3 bg-neutral-950/30">
                      <span className="text-neutral-400 text-sm font-medium">Status</span>
                      <span className={`text-sm font-semibold px-3 py-1 ${
                        isStakeLocked 
                          ? 'bg-yellow-500/10 text-yellow-400 border border-yellow-500/30' 
                          : stakedBalance && stakedBalance > 0
                            ? 'bg-green-500/10 text-green-400 border border-green-500/30'
                            : 'bg-neutral-800/30 text-neutral-500'
                      }`}>
                        {isStakeLocked 
                          ? `Locked until ${new Date(stakeLockUntil! * 1000).toLocaleDateString()}`
                          : stakedBalance && stakedBalance > 0 
                            ? 'Unlocked'
                            : 'No stake'
                        }
                      </span>
                    </div>
                    
                    {(stakedBalance ?? 0) > 0 && (
                      <button 
                        onClick={handleUnstake}
                        disabled={stakeStatus === 'processing' || !!isStakeLocked}
                        className={`w-full mt-4 py-3 font-semibold text-sm flex items-center justify-center gap-2 transition-all duration-200 ${
                          isStakeLocked
                            ? 'bg-neutral-800/50 text-neutral-500 cursor-not-allowed border border-neutral-700/50'
                            : 'bg-neutral-800/50 hover:bg-neutral-700/50 text-white border border-neutral-700/50 hover:border-neutral-700'
                        }`}
                      >
                        <Unlock className="w-4 h-4" />
                        {isStakeLocked ? 'Stake Locked' : 'Unstake All'}
                      </button>
                    )}
                  </div>
                </div>
              </div>
              
              {stakeMessage && (
                <motion.div 
                  initial={{ opacity: 0, y: 10 }}
                  animate={{ opacity: 1, y: 0 }}
                  className={`mt-6 p-4 text-sm text-center font-medium ${
                    stakeStatus === 'success' 
                      ? 'bg-green-500/10 text-green-400 border border-green-500/30' 
                      : 'bg-red-500/10 text-red-400 border border-red-500/30'
                  }`}
                >
                  {stakeMessage}
                </motion.div>
              )}
            </div>
          </motion.div>
        </div>

        {/* Transfer Section */}
        <motion.div 
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.3, duration: 0.4 }}
          className="relative bg-neutral-900 border border-neutral-800/50 p-8 transition-all duration-300 overflow-hidden"
        >
                    <div className="relative z-10">
            <div className="flex items-center gap-3 mb-6">
              <div className="p-2 bg-green-500/10 border border-green-500/20">
                <Send className="w-5 h-5 text-green-400" />
              </div>
              <h2 className="text-xl font-bold text-white font-display">Send NEURO</h2>
            </div>
            
            <form onSubmit={handleTransfer} className="grid grid-cols-1 lg:grid-cols-3 gap-6">
              <div className="lg:col-span-2">
                <label className="block text-sm font-semibold text-neutral-400 mb-2 uppercase tracking-wider">Recipient Address</label>
                <input 
                  type="text" 
                  placeholder="Enter wallet address (Node ID)"
                  value={transferRecipient}
                  onChange={(e) => setTransferRecipient(e.target.value)}
                  className="w-full bg-neutral-950/50 border border-neutral-700/50 p-4 text-white placeholder-neutral-600 focus:outline-none focus:ring-0 focus:border-accent transition-all font-mono text-sm"
                  required
                />
              </div>
              
              <div>
                <label className="block text-sm font-semibold text-neutral-400 mb-2 uppercase tracking-wider">Amount</label>
                <div className="relative">
                  <input 
                    type="number" 
                    step="0.000001"
                    min="0.000001"
                    placeholder="0.00"
                    value={transferAmount}
                    onChange={(e) => setTransferAmount(e.target.value)}
                    className="w-full bg-neutral-950/50 border border-neutral-700/50 p-4 pr-20 text-white placeholder-neutral-600 focus:outline-none focus:ring-0 focus:border-accent transition-all font-mono text-lg"
                    required
                  />
                  <div className="absolute right-4 top-1/2 -translate-y-1/2 text-neutral-500 font-semibold text-sm">NEURO</div>
                </div>
              </div>

              <div className="lg:col-span-3 flex flex-wrap gap-4 items-center justify-between">
                <div className="flex items-center gap-2">
                  <span className="text-xs text-neutral-500 font-medium">Available:</span>
                  <span className="text-sm font-bold text-white font-mono">{neuroBalance?.toFixed(4)}</span>
                  <button 
                    type="button"
                    onClick={() => setTransferAmount(neuroBalance?.toString() || "")}
                    className="text-xs px-2 py-1 bg-accent/10 hover:bg-accent/10 text-accent font-semibold transition-colors border border-accent/20"
                  >
                    Max
                  </button>
                </div>
                
                <button 
                  type="submit"
                  disabled={transferStatus === 'processing'}
                  className={`px-8 py-3 font-bold flex items-center gap-2 transition-all duration-200 ${
                    transferStatus === 'processing' 
                      ? 'bg-neutral-800 text-neutral-400 cursor-not-allowed'
                      : 'bg-accent hover:bg-accent/90 text-neutral-950'
                  }`}
                >
                  {transferStatus === 'processing' ? (
                    <><RefreshCw className="w-5 h-5 animate-spin" /> Processing...</>
                  ) : (
                    <>Send <ArrowRight className="w-5 h-5" /></>
                  )}
                </button>
              </div>

              {transferMessage && (
                <motion.div 
                  initial={{ opacity: 0, y: 10 }}
                  animate={{ opacity: 1, y: 0 }}
                  className={`lg:col-span-3 p-4 text-sm text-center font-medium ${
                    transferStatus === 'success' 
                      ? 'bg-green-500/10 text-green-400 border border-green-500/30' 
                      : 'bg-red-500/10 text-red-400 border border-red-500/30'
                  }`}
                >
                  {transferMessage}
                </motion.div>
              )}
            </form>
          </div>
        </motion.div>

      </div>
    </section>
  );
};
