import { useState, useEffect } from 'react';
import { useAuth } from '../contexts/AuthContext';
import { useNavigate, Link } from 'react-router-dom';
import axios from 'axios';
import { motion } from 'framer-motion';
import { API_URL } from '../config/api';
import { AlertTriangle, Copy, Check, Eye, EyeOff, ArrowRight, Terminal, ExternalLink, Sparkles } from 'lucide-react';

type Step = 'account' | 'wallet' | 'import' | 'mnemonic' | 'complete';

interface WalletData {
  mnemonic: string;
  token: string;
  node_id: string;
  wallet_id: string;
}

export const Signup = () => {
  const [step, setStep] = useState<Step>('account');
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [error, setError] = useState('');
  const [loading, setLoading] = useState(false);
  const [wallet, setWallet] = useState<WalletData | null>(null);
  const [copiedMnemonic, setCopiedMnemonic] = useState(false);
  const [copiedToken, setCopiedToken] = useState(false);
  const [showMnemonic, setShowMnemonic] = useState(true);
  const [showToken, setShowToken] = useState(false);
  const [confirmed, setConfirmed] = useState(false);
  const [importSecret, setImportSecret] = useState('');
  const [acceptedTerms, setAcceptedTerms] = useState(false);
  const { login, refreshUser, isAuthenticated, hasWallet, isLoading } = useAuth();
  const navigate = useNavigate();

  useEffect(() => {
    if (!isLoading && isAuthenticated && hasWallet) {
      navigate('/dashboard', { replace: true });
    }
    if (!isLoading && isAuthenticated && !hasWallet && step === 'account') {
      setStep('wallet');
    }
  }, [isAuthenticated, hasWallet, isLoading, navigate, step]);

  const handleCreateAccount = async (e: React.FormEvent) => {
    e.preventDefault();
    setError('');
    setLoading(true);

    try {
      await axios.post(`${API_URL}/api/auth/signup`, { email, password });
      const formData = new FormData();
      formData.append('username', email);
      formData.append('password', password);
      const response = await axios.post(`${API_URL}/api/auth/token`, formData);
      await login(response.data);
      setStep('wallet');
    } catch (err: any) {
      const detail = err.response?.data?.detail || '';
      if (err.response?.status === 400 && detail === 'Email already registered') {
        setError('Email already registered. Please use a different email or log in.');
      } else if (err.response?.status === 403 && detail.includes('waitlist')) {
        setError(detail);
      } else {
        setError(detail || 'Failed to create account');
      }
    } finally {
      setLoading(false);
    }
  };

  const handleCreateWallet = async () => {
    setError('');
    setLoading(true);
    try {
      const token = localStorage.getItem('token');
      const response = await axios.post(`${API_URL}/api/wallet/create`, {}, {
        headers: { Authorization: `Bearer ${token}` }
      });
      setWallet(response.data);
      setStep('mnemonic');
    } catch (err: any) {
      setError(err.response?.data?.detail || 'Failed to create wallet');
    } finally {
      setLoading(false);
    }
  };

  const copyMnemonic = () => {
    if (wallet?.mnemonic) {
      navigator.clipboard.writeText(wallet.mnemonic);
      setCopiedMnemonic(true);
      setTimeout(() => setCopiedMnemonic(false), 2000);
    }
  };

  const copyToken = () => {
    if (wallet?.token) {
      navigator.clipboard.writeText(wallet.token);
      setCopiedToken(true);
      setTimeout(() => setCopiedToken(false), 2000);
    }
  };

  const handleImportWallet = async () => {
    if (!importSecret.trim()) {
      setError('Please enter your recovery phrase or token');
      return;
    }
    setError('');
    setLoading(true);
    try {
      const token = localStorage.getItem('token');
      const response = await axios.post(`${API_URL}/api/wallet/connect`,
        { secret: importSecret.trim() },
        { headers: { Authorization: `Bearer ${token}` } }
      );
      setWallet({
        mnemonic: importSecret.trim().split(' ').length >= 12 ? importSecret.trim() : '',
        token: response.data.token || importSecret.trim(),
        node_id: response.data.node_id,
        wallet_id: response.data.wallet_id
      });
      navigate('/dashboard');
    } catch (err: any) {
      setError(err.response?.data?.detail || 'Failed to import wallet');
    } finally {
      setLoading(false);
    }
  };

  const handleComplete = async () => {
    if (!confirmed) return;
    await refreshUser();
    navigate('/dashboard');
  };

  const ProgressBar = ({ current }: { current: number }) => (
    <div className="flex items-center gap-0 mb-8">
      {[1, 2, 3].map((s) => (
        <div key={s} className="flex items-center">
          <div className={`w-7 h-7 flex items-center justify-center text-xs font-mono font-bold border ${
            s < current ? 'bg-accent text-neutral-950 border-accent' :
            s === current ? 'border-accent text-accent' :
            'border-neutral-700 text-neutral-600'
          }`}>
            {s < current ? '\u2713' : s}
          </div>
          {s < 3 && <div className={`w-10 h-px ${s < current ? 'bg-accent' : 'bg-neutral-800'}`} />}
        </div>
      ))}
    </div>
  );

  if (isLoading || (isAuthenticated && hasWallet)) {
    return (
      <div className="min-h-screen flex items-center justify-center bg-neutral-950">
        <div className="w-6 h-6 border-2 border-accent border-t-transparent animate-spin" />
      </div>
    );
  }

  if (step === 'account') {
    return (
      <div className="min-h-screen flex items-center justify-center bg-neutral-950 px-4 pt-20">
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          className="max-w-md w-full bg-neutral-900 border border-neutral-800 p-8"
        >
          <ProgressBar current={1} />
          <h2 className="font-display text-2xl font-bold text-white mb-1">Create Account</h2>
          <p className="text-neutral-500 text-sm mb-6">Step 1 of 3: Your login credentials</p>

          {error && (
            <div className="mb-4 bg-red-950/30 border border-red-900/50 p-4">
              <p className="text-red-300 text-sm">{error}</p>
              {error.includes('waitlist') && (
                <Link
                  to="/join"
                  className="mt-3 inline-flex items-center gap-2 px-4 py-2 bg-neutral-800 hover:bg-neutral-700 text-accent text-sm font-medium transition-colors"
                >
                  <Sparkles className="w-4 h-4" />
                  Join the Waitlist
                  <ArrowRight className="w-4 h-4" />
                </Link>
              )}
            </div>
          )}

          <form onSubmit={handleCreateAccount} className="space-y-5">
            <div>
              <label className="block text-xs font-mono uppercase tracking-widest text-neutral-500 mb-2">Email</label>
              <input
                type="email"
                value={email}
                onChange={(e) => setEmail(e.target.value)}
                className="w-full px-4 py-3 bg-neutral-950 border border-neutral-800 text-white focus:outline-none focus:border-accent transition-colors"
                required
                disabled={loading}
              />
            </div>
            <div>
              <label className="block text-xs font-mono uppercase tracking-widest text-neutral-500 mb-2">Password</label>
              <input
                type="password"
                value={password}
                onChange={(e) => setPassword(e.target.value)}
                className="w-full px-4 py-3 bg-neutral-950 border border-neutral-800 text-white focus:outline-none focus:border-accent transition-colors"
                required
                disabled={loading}
              />
            </div>

            <div className="border border-neutral-800 p-4">
              <label className="flex items-start gap-3 cursor-pointer">
                <input
                  type="checkbox"
                  checked={acceptedTerms}
                  onChange={(e) => setAcceptedTerms(e.target.checked)}
                  className="mt-1 w-4 h-4 accent-accent bg-neutral-900"
                  disabled={loading}
                />
                <span className="text-neutral-400 text-sm leading-relaxed">
                  I am at least <strong className="text-white">18 years old</strong> and I have read and agree to the{' '}
                  <Link to="/legal" target="_blank" className="text-accent hover:underline inline-flex items-center gap-1">
                    Terms of Service
                    <ExternalLink className="w-3 h-3" />
                  </Link>
                  .
                </span>
              </label>
            </div>

            <button
              type="submit"
              disabled={loading || !acceptedTerms}
              className="w-full py-3 bg-white text-neutral-950 font-bold text-sm hover:bg-neutral-200 transition-colors disabled:opacity-50 disabled:cursor-not-allowed flex items-center justify-center gap-2"
            >
              {loading ? 'Creating...' : 'Continue to Wallet Setup'}
              <ArrowRight className="w-4 h-4" />
            </button>
          </form>

          <p className="mt-6 text-center text-neutral-500 text-sm">
            Already have an account?{' '}
            <Link to="/login" className="text-accent hover:underline">Log in</Link>
          </p>
        </motion.div>
      </div>
    );
  }

  if (step === 'wallet') {
    return (
      <div className="min-h-screen flex items-center justify-center bg-neutral-950 px-4 pt-20">
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          className="max-w-md w-full bg-neutral-900 border border-neutral-800 p-8"
        >
          <ProgressBar current={2} />
          <h2 className="font-display text-2xl font-bold text-white mb-1">Connect Your Wallet</h2>
          <p className="text-neutral-500 text-sm mb-8">Step 2 of 3: Secure your NEURO earnings</p>

          {error && <p className="text-red-300 text-sm mb-4 bg-red-950/30 border border-red-900/50 p-3">{error}</p>}

          <button
            onClick={handleCreateWallet}
            disabled={loading}
            className="w-full py-3.5 bg-accent text-neutral-950 font-bold text-sm hover:bg-accent/90 transition-colors disabled:opacity-50 disabled:cursor-not-allowed flex items-center justify-center gap-2 mb-4"
          >
            {loading ? 'Generating...' : 'Create New Wallet'}
          </button>

          <div className="flex items-center gap-4 my-6">
            <div className="flex-1 h-px bg-neutral-800" />
            <span className="text-neutral-600 text-xs font-mono">or</span>
            <div className="flex-1 h-px bg-neutral-800" />
          </div>

          <button
            onClick={() => setStep('import')}
            className="w-full py-3.5 border border-neutral-700 text-white font-bold text-sm hover:bg-neutral-800 transition-colors flex items-center justify-center gap-2"
          >
            Import Existing Wallet
          </button>

          <p className="text-neutral-600 text-xs text-center mt-4 font-mono">
            Have a 12-word recovery phrase? Import your existing wallet.
          </p>
        </motion.div>
      </div>
    );
  }

  if (step === 'import') {
    const isValidFormat = importSecret.trim().split(/\s+/).length >= 12 ||
                          (importSecret.trim().length === 64 && /^[a-f0-9]+$/i.test(importSecret.trim()));

    return (
      <div className="min-h-screen flex items-center justify-center bg-neutral-950 px-4 pt-20">
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          className="max-w-md w-full bg-neutral-900 border border-neutral-800 p-8"
        >
          <ProgressBar current={2} />
          <h2 className="font-display text-xl font-bold text-white mb-1">Import Existing Wallet</h2>
          <p className="text-neutral-500 text-sm mb-6">Enter your 12-word recovery phrase or secret token</p>

          {error && <p className="text-red-300 text-sm mb-4 bg-red-950/30 border border-red-900/50 p-3">{error}</p>}

          <div className="mb-6">
            <textarea
              value={importSecret}
              onChange={(e) => setImportSecret(e.target.value)}
              placeholder="Enter your 12-word recovery phrase or 64-character token..."
              className="w-full bg-neutral-950 border border-neutral-800 px-4 py-3 text-white placeholder-neutral-600 focus:outline-none focus:border-accent resize-none font-mono text-sm h-32 transition-colors"
              autoComplete="off"
              spellCheck={false}
            />
            {importSecret && (
              <p className={`text-xs mt-2 ${isValidFormat ? 'text-accent' : 'text-amber-400'}`}>
                {importSecret.trim().split(/\s+/).length >= 12
                  ? `\u2713 Mnemonic detected (${importSecret.trim().split(/\s+/).length} words)`
                  : importSecret.trim().length === 64
                    ? '\u2713 Token format detected'
                    : '\u26A0 Enter 12 words or 64-char token'}
              </p>
            )}
          </div>

          <button
            onClick={handleImportWallet}
            disabled={loading || !isValidFormat}
            className="w-full py-3.5 bg-accent text-neutral-950 font-bold text-sm hover:bg-accent/90 transition-colors disabled:opacity-50 disabled:cursor-not-allowed flex items-center justify-center gap-2 mb-4"
          >
            {loading ? 'Importing...' : 'Import Wallet'}
            <ArrowRight className="w-4 h-4" />
          </button>

          <button
            onClick={() => { setStep('wallet'); setError(''); setImportSecret(''); }}
            className="w-full py-3 text-neutral-500 hover:text-white transition-colors text-sm"
          >
            &larr; Back to wallet options
          </button>
        </motion.div>
      </div>
    );
  }

  if (step === 'mnemonic' && wallet) {
    return (
      <div className="min-h-screen flex items-center justify-center bg-neutral-950 px-4 pt-20 pb-12">
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          className="max-w-2xl w-full bg-neutral-900 border border-neutral-800 p-8"
        >
          <ProgressBar current={3} />
          <h2 className="font-display text-2xl font-bold text-white mb-1">Wallet Created</h2>
          <p className="text-neutral-500 text-sm mb-6">Step 3 of 3: Save your recovery phrase</p>

          <div className="bg-red-950/30 border border-red-900/50 p-5 mb-6">
            <div className="flex items-start gap-3">
              <AlertTriangle className="text-red-400 flex-shrink-0 mt-0.5" size={20} />
              <div>
                <p className="text-red-300 font-bold text-sm mb-2">
                  SAVE THIS NOW -- YOU WON'T SEE IT AGAIN
                </p>
                <ul className="text-red-300/70 text-xs space-y-1 list-disc list-inside">
                  <li>Write it down on paper and store it safely</li>
                  <li>Never share it with anyone -- not even NeuroShard support</li>
                  <li>Anyone with this phrase can access your NEURO</li>
                  <li>We cannot recover your wallet if you lose this phrase</li>
                </ul>
              </div>
            </div>
          </div>

          <div className="border border-neutral-800 p-6 mb-6">
            <div className="flex items-center justify-between mb-4">
              <h3 className="text-white font-bold text-sm">Your 12-Word Recovery Phrase</h3>
              <button onClick={() => setShowMnemonic(!showMnemonic)} className="text-neutral-500 hover:text-white transition-colors">
                {showMnemonic ? <EyeOff size={18} /> : <Eye size={18} />}
              </button>
            </div>

            <div className={`grid grid-cols-2 sm:grid-cols-3 gap-2 mb-4 ${!showMnemonic ? 'blur-md select-none' : ''}`}>
              {wallet.mnemonic.split(' ').map((word: string, i: number) => (
                <div key={i} className="bg-neutral-950 border border-neutral-800 p-3">
                  <span className="text-neutral-600 text-xs font-mono mr-2">{i + 1}.</span>
                  <span className="text-accent font-mono font-bold text-sm">{word}</span>
                </div>
              ))}
            </div>

            <button
              onClick={copyMnemonic}
              className="w-full bg-neutral-800 hover:bg-neutral-700 text-white font-medium py-3 text-sm transition-colors flex items-center justify-center gap-2"
            >
              {copiedMnemonic ? <><Check size={16} className="text-accent" /><span>Copied!</span></> : <><Copy size={16} /><span>Copy Mnemonic</span></>}
            </button>
          </div>

          <div className="border border-accent/20 bg-accent/5 p-6 mb-6">
            <div className="flex items-center gap-3 mb-3">
              <Terminal className="text-accent" size={18} />
              <h3 className="text-white font-bold text-sm">Secret Token (For Node Runner)</h3>
            </div>
            <p className="text-neutral-500 text-xs mb-4">
              Copy this token to paste into the NeuroShard Node Runner. It's derived from your mnemonic.
            </p>

            <div className="flex items-center gap-2 mb-3">
              <div className={`flex-1 bg-neutral-950 border border-neutral-800 p-3 font-mono text-xs break-all ${!showToken ? 'blur-sm select-none' : 'text-accent'}`}>
                {wallet.token}
              </div>
              <button onClick={() => setShowToken(!showToken)} className="p-3 border border-neutral-800 text-neutral-500 hover:text-white transition-colors">
                {showToken ? <EyeOff size={16} /> : <Eye size={16} />}
              </button>
            </div>

            <button
              onClick={copyToken}
              className="w-full bg-accent hover:bg-accent/90 text-neutral-950 font-medium py-3 text-sm transition-colors flex items-center justify-center gap-2"
            >
              {copiedToken ? <><Check size={16} /><span>Token Copied!</span></> : <><Copy size={16} /><span>Copy Token for Runner</span></>}
            </button>
          </div>

          <div className="border border-neutral-800 p-4 mb-6">
            <label className="flex items-start gap-3 cursor-pointer">
              <input
                type="checkbox"
                checked={confirmed}
                onChange={(e) => setConfirmed(e.target.checked)}
                className="mt-1 w-4 h-4 accent-accent"
              />
              <span className="text-neutral-400 text-sm">
                I have <strong className="text-white">written down and securely stored</strong> my 12-word recovery phrase.
                I understand that <strong className="text-red-400">I cannot recover my wallet</strong> without it.
              </span>
            </label>
          </div>

          <button
            onClick={handleComplete}
            disabled={!confirmed}
            className="w-full py-3.5 bg-white text-neutral-950 font-bold text-sm hover:bg-neutral-200 transition-colors disabled:opacity-50 disabled:cursor-not-allowed flex items-center justify-center gap-2"
          >
            Complete Registration & Go to Dashboard
            <ArrowRight className="w-4 h-4" />
          </button>
        </motion.div>
      </div>
    );
  }

  return null;
};
