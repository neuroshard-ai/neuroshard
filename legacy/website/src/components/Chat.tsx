import { useState, useRef, useEffect, useCallback } from 'react';
import { motion } from 'framer-motion';
import { Send, Terminal, Server, Lock, Activity, LayoutDashboard, Coins, X, AlertCircle } from 'lucide-react';
import { API_URL } from '../config/api';
import axios, { CancelTokenSource } from 'axios';
import { useAuth } from '../contexts/AuthContext';
import { useNavigate, Link } from 'react-router-dom';

const CHAT_TIMEOUT_MS = 45_000;

export const Chat = () => {
  const { user, token, refreshUser } = useAuth();
  const navigate = useNavigate();
  const [input, setInput] = useState('');
  const [messages, setMessages] = useState<Array<{ role: 'user' | 'assistant'; content: string; error?: boolean }>>([
    { role: 'assistant', content: 'Hello! I am NeuroShard, a distributed AI running across a global swarm of consumer devices. How can I help you today?' }
  ]);
  const [loading, setLoading] = useState(false);
  const [nodeStatus, setNodeStatus] = useState(false);
  const [neuroBalance, setNeuroBalance] = useState<number | null>(null);
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const cancelSourceRef = useRef<CancelTokenSource | null>(null);
  const inputRef = useRef<HTMLInputElement>(null);

  useEffect(() => {
    if (user) {
      checkNodeStatus();
      fetchNeuroBalance();
    }
  }, [user]);

  const fetchNeuroBalance = async () => {
    if (!user?.node_id) return;
    try {
      const response = await axios.get(`${API_URL}/api/node/neuro`, {
        params: { node_id: user.node_id },
        timeout: 5000,
      });
      setNeuroBalance(response.data.neuro_balance || 0);
    } catch {
      setNeuroBalance(0);
    }
  };

  const checkNodeStatus = async () => {
    if (!token) return;
    try {
      const res = await axios.get(`${API_URL}/api/users/me/node_status`, {
        headers: { Authorization: `Bearer ${token}` },
        timeout: 5000,
      });
      setNodeStatus(res.data.active);
    } catch {
      setNodeStatus(false);
    }
  };

  const scrollToBottom = useCallback(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
  }, []);

  useEffect(() => { scrollToBottom(); }, [messages, scrollToBottom]);
  useEffect(() => { window.scrollTo(0, 0); }, []);

  const cancelRequest = useCallback(() => {
    if (cancelSourceRef.current) {
      cancelSourceRef.current.cancel('User cancelled');
      cancelSourceRef.current = null;
    }
    setLoading(false);
  }, []);

  useEffect(() => {
    return () => {
      cancelRequest();
    };
  }, [cancelRequest]);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!input.trim() || loading) return;

    if (!user) {
      navigate('/login');
      return;
    }

    const userMsg = input.trim();
    setMessages(prev => [...prev, { role: 'user', content: userMsg }]);
    setInput('');
    setLoading(true);

    const source = axios.CancelToken.source();
    cancelSourceRef.current = source;

    try {
      const response = await axios.post(`${API_URL}/api/chat`, {
        prompt: userMsg,
        max_new_tokens: 50
      }, {
        headers: { Authorization: `Bearer ${token}` },
        timeout: CHAT_TIMEOUT_MS,
        cancelToken: source.token,
      });

      const generatedText = response.data.text || response.data.result || 'No response generated.';

      setMessages(prev => [...prev, {
        role: 'assistant',
        content: generatedText
      }]);

      refreshUser();
      fetchNeuroBalance();

    } catch (err: any) {
      if (axios.isCancel(err)) {
        setMessages(prev => [...prev, {
          role: 'assistant',
          content: 'Request cancelled.',
          error: true,
        }]);
      } else {
        let errorMsg = 'Could not connect to the swarm. Please try again.';
        if (err.response?.status === 401) {
          errorMsg = 'Session expired. Please login again.';
        } else if (err.response?.status === 402) {
          errorMsg = `Insufficient NEURO: ${err.response.data.detail}`;
        } else if (err.response?.status === 403) {
          errorMsg = 'You must have an active NeuroShard Node running to use the chat. Start your node with your token.';
        } else if (err.response?.status === 429) {
          errorMsg = err.response.data.detail || 'Rate limit exceeded. Please wait before trying again.';
        } else if (err.response?.status === 503) {
          errorMsg = 'No nodes available in the swarm. Please wait for nodes to come online.';
        } else if (err.code === 'ECONNABORTED') {
          errorMsg = 'Request timed out. The swarm may be busy -- try again in a moment.';
        }
        setMessages(prev => [...prev, {
          role: 'assistant',
          content: errorMsg,
          error: true,
        }]);
      }
    } finally {
      cancelSourceRef.current = null;
      setLoading(false);
      setTimeout(() => inputRef.current?.focus(), 100);
    }
  };

  if (!user) {
    return (
      <section className="pt-20 sm:pt-32 pb-24 min-h-screen bg-neutral-950 relative overflow-hidden flex items-center justify-center">
        <div className="text-center p-6 sm:p-8 max-w-md">
          <div className="inline-flex p-4 bg-neutral-900 mb-6 border border-neutral-800">
            <Lock className="w-8 h-8 text-accent" />
          </div>
          <h2 className="text-2xl sm:text-3xl font-bold text-white mb-4 font-display">Access Restricted</h2>
          <p className="text-neutral-400 mb-8 text-sm sm:text-base">
            The NeuroShard Swarm Chat is currently in closed beta.
            Only active node operators who have contributed to the network can access the live model.
          </p>
          <div className="flex flex-col gap-3">
            <button
              onClick={() => navigate('/login')}
              className="w-full py-3 bg-accent hover:bg-accent/90 text-neutral-950 font-bold text-sm transition-colors"
            >
              Log In
            </button>
            <button
              onClick={() => navigate('/signup')}
              className="w-full py-3 bg-neutral-800 hover:bg-neutral-700 text-white font-medium text-sm transition-colors border border-neutral-700"
            >
              Create Account & Join Swarm
            </button>
          </div>
        </div>
      </section>
    );
  }

  return (
    <section className="pt-14 sm:pt-16 h-[100dvh] flex flex-col bg-neutral-950 relative overflow-hidden">
      <div className="flex-1 flex gap-0 lg:gap-4 min-h-0 max-w-7xl mx-auto w-full px-0 lg:px-4">

        {/* Sidebar */}
        <div className="w-64 hidden lg:flex flex-col gap-3 py-3 overflow-y-auto flex-shrink-0">
          <div className="bg-neutral-900 border border-neutral-800 p-4">
            <p className="text-[10px] font-mono uppercase tracking-widest text-neutral-500 mb-1">Logged in as</p>
            <p className="font-medium text-white truncate text-sm">{user.email}</p>

            <div className="mt-4 bg-neutral-950 p-3 border border-neutral-800">
              <p className="text-[10px] font-mono uppercase tracking-widest text-neutral-500 mb-1 flex items-center gap-1.5">
                <Coins className="w-3 h-3 text-yellow-400" />
                NEURO Balance
              </p>
              <p className="text-xl font-bold text-white font-mono">
                {neuroBalance !== null ? neuroBalance.toFixed(4) : '...'}
              </p>
            </div>

            <div className="flex items-center gap-2 mt-4 px-0.5">
              <div className={`w-2 h-2 ${nodeStatus ? 'bg-green-500' : 'bg-red-500'}`} />
              <span className={`text-xs font-medium ${nodeStatus ? 'text-green-400' : 'text-red-400'}`}>
                {nodeStatus ? 'Node Active' : 'Node Offline'}
              </span>
            </div>

            <Link to="/dashboard" className="flex items-center justify-center gap-2 w-full py-2 mt-4 bg-neutral-800 hover:bg-neutral-700 text-white transition-colors border border-neutral-700 text-xs font-medium">
              <LayoutDashboard className="w-3.5 h-3.5" />
              Dashboard
            </Link>
          </div>

          <div className="bg-neutral-900 border border-neutral-800 p-4 flex-1">
            <h3 className="font-bold text-white mb-3 flex items-center gap-2 text-xs font-display">
              <Activity className="w-3.5 h-3.5 text-accent" />
              Pricing
            </h3>
            <div className="space-y-2 text-[11px]">
              <div className="flex justify-between">
                <span className="text-neutral-500">Inference</span>
                <span className="text-neutral-300 font-mono">0.1 NEURO/1M tok</span>
              </div>
              <div className="flex justify-between">
                <span className="text-neutral-500">Uptime</span>
                <span className="text-neutral-300 font-mono">0.0005/min</span>
              </div>
              <div className="flex justify-between">
                <span className="text-neutral-500">Training</span>
                <span className="text-neutral-300 font-mono">0.0001/batch</span>
              </div>
            </div>
          </div>
        </div>

        {/* Main Chat */}
        <div className="flex-1 flex flex-col min-h-0 lg:py-3">
          <div className="flex-1 flex flex-col bg-neutral-900 border-x border-neutral-800 lg:border lg:border-neutral-800 overflow-hidden">

            {/* Chat header */}
            <div className="px-4 sm:px-5 py-3 border-b border-neutral-800 bg-neutral-950/50 flex items-center justify-between flex-shrink-0">
              <div>
                <h1 className="font-bold text-white text-sm sm:text-base leading-none mb-0.5 font-display">Swarm Chat</h1>
                <p className="text-neutral-500 text-[10px] sm:text-xs flex items-center gap-1.5">
                  <span className="w-1.5 h-1.5 bg-green-500" />
                  NeuroLLM-v1 &middot; Decentralized
                </p>
              </div>
              <div className="flex items-center gap-3">
                {/* Mobile balance */}
                <div className="lg:hidden flex items-center gap-1.5 bg-neutral-950 px-2 py-1 border border-neutral-800 text-[10px]">
                  <Coins className="w-3 h-3 text-yellow-400" />
                  <span className="text-white font-mono font-medium">{neuroBalance !== null ? neuroBalance.toFixed(2) : '...'}</span>
                </div>
                <div className="lg:hidden flex items-center gap-1.5">
                  <div className={`w-2 h-2 ${nodeStatus ? 'bg-green-500' : 'bg-red-500'}`} />
                  <span className={`text-[10px] font-medium ${nodeStatus ? 'text-green-400' : 'text-neutral-500'}`}>
                    {nodeStatus ? 'Online' : 'Offline'}
                  </span>
                </div>
              </div>
            </div>

            {/* Messages */}
            <div className="flex-1 overflow-y-auto p-4 sm:p-5 space-y-4">
              {messages.map((msg, idx) => (
                <motion.div
                  key={idx}
                  initial={{ opacity: 0, y: 8 }}
                  animate={{ opacity: 1, y: 0 }}
                  transition={{ duration: 0.2 }}
                  className={`flex ${msg.role === 'user' ? 'justify-end' : 'justify-start'}`}
                >
                  <div
                    className={`max-w-[88%] sm:max-w-[75%] p-3 sm:p-4 ${msg.role === 'user'
                      ? 'bg-accent text-neutral-950'
                      : msg.error
                        ? 'bg-red-950/30 border border-red-900/50 text-neutral-300'
                        : 'bg-neutral-800 text-neutral-200 border border-neutral-700/50'
                    }`}
                  >
                    {msg.role === 'assistant' && (
                      <div className={`flex items-center gap-1.5 mb-1.5 text-[10px] font-mono uppercase tracking-wider font-bold ${
                        msg.error ? 'text-red-400' : 'text-accent'
                      }`}>
                        {msg.error ? <AlertCircle className="w-3 h-3" /> : <Server className="w-3 h-3" />}
                        {msg.error ? 'Error' : 'NeuroShard'}
                      </div>
                    )}
                    <p className="leading-relaxed whitespace-pre-wrap break-words text-sm" style={{ wordBreak: 'break-word', overflowWrap: 'anywhere' }}>
                      {msg.content}
                    </p>
                  </div>
                </motion.div>
              ))}

              {loading && (
                <motion.div
                  initial={{ opacity: 0 }}
                  animate={{ opacity: 1 }}
                  className="flex justify-start"
                >
                  <div className="bg-neutral-800 border border-neutral-700/50 px-4 py-3 flex items-center gap-1">
                    <div className="w-1.5 h-1.5 bg-accent animate-bounce" style={{ animationDelay: '0ms' }} />
                    <div className="w-1.5 h-1.5 bg-accent animate-bounce" style={{ animationDelay: '150ms' }} />
                    <div className="w-1.5 h-1.5 bg-accent animate-bounce" style={{ animationDelay: '300ms' }} />
                  </div>
                </motion.div>
              )}
              <div ref={messagesEndRef} />
            </div>

            {/* Input */}
            <div className="p-3 sm:p-4 bg-neutral-950/50 border-t border-neutral-800 flex-shrink-0">
              <form onSubmit={handleSubmit} className="flex gap-2 max-w-4xl mx-auto">
                <input
                  ref={inputRef}
                  type="text"
                  value={input}
                  onChange={(e) => setInput(e.target.value)}
                  placeholder={loading ? 'Waiting for response...' : 'Ask the swarm anything...'}
                  className="flex-1 bg-neutral-900 border border-neutral-800 py-3 px-4 text-white text-sm placeholder:text-neutral-600 focus:outline-none focus:border-accent transition-colors disabled:opacity-60"
                  disabled={loading}
                  autoComplete="off"
                />
                {loading ? (
                  <button
                    type="button"
                    onClick={cancelRequest}
                    className="px-4 py-3 bg-red-950 border border-red-900/50 text-red-400 hover:bg-red-900/30 transition-colors flex items-center gap-1.5 text-sm font-medium flex-shrink-0"
                  >
                    <X className="w-4 h-4" />
                    <span className="hidden sm:inline">Cancel</span>
                  </button>
                ) : (
                  <button
                    type="submit"
                    disabled={!input.trim()}
                    className="px-4 py-3 bg-accent text-neutral-950 hover:bg-accent/90 transition-colors disabled:opacity-30 disabled:cursor-not-allowed flex items-center gap-1.5 text-sm font-bold flex-shrink-0"
                  >
                    <Send className="w-4 h-4" />
                    <span className="hidden sm:inline">Send</span>
                  </button>
                )}
              </form>
              <p className="text-[10px] text-neutral-600 text-center mt-2">
                <Terminal className="w-3 h-3 inline mr-1" />
                Powered by NeuroLLM &middot; Early model &middot; Quality improves as network grows
              </p>
            </div>
          </div>
        </div>
      </div>
    </section>
  );
};
