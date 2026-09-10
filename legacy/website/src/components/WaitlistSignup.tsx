import { useState, useEffect } from 'react';
import { useSearchParams } from 'react-router-dom';
import { motion } from 'framer-motion';
import {
  Cpu, Zap, Wifi, Monitor, ArrowRight,
  Check, Copy, Link2,
  AlertCircle, Share2, RefreshCw
} from 'lucide-react';
import axios from 'axios';
import { API_URL } from '../config/api';
import { SEO } from './SEO';

type Step = 'hardware' | 'calculating' | 'results';

interface HardwareData {
  email: string;
  gpu_model: string;
  gpu_vram: number | null;
  ram_gb: number;
  internet_speed: number | null;
  operating_system: string;
  referral_code: string;
}

interface WaitlistResult {
  id: number;
  email: string;
  referral_code: string;
  referral_url: string;
  position: number;
  estimated_daily_neuro: number;
  hardware_tier: string;
  hardware_score: number;
  priority_score: number;
  status: string;
  message: string;
}

const GPU_OPTIONS = [
  { value: '', label: 'Select your GPU...' },
  { value: 'none', label: 'No GPU / CPU Only' },
  { value: 'RTX 4090', label: 'NVIDIA RTX 4090' },
  { value: 'RTX 4080', label: 'NVIDIA RTX 4080' },
  { value: 'RTX 4070 Ti', label: 'NVIDIA RTX 4070 Ti' },
  { value: 'RTX 4070', label: 'NVIDIA RTX 4070' },
  { value: 'RTX 4060 Ti', label: 'NVIDIA RTX 4060 Ti' },
  { value: 'RTX 4060', label: 'NVIDIA RTX 4060' },
  { value: 'RTX 3090 Ti', label: 'NVIDIA RTX 3090 Ti' },
  { value: 'RTX 3090', label: 'NVIDIA RTX 3090' },
  { value: 'RTX 3080 Ti', label: 'NVIDIA RTX 3080 Ti' },
  { value: 'RTX 3080', label: 'NVIDIA RTX 3080' },
  { value: 'RTX 3070 Ti', label: 'NVIDIA RTX 3070 Ti' },
  { value: 'RTX 3070', label: 'NVIDIA RTX 3070' },
  { value: 'RTX 3060 Ti', label: 'NVIDIA RTX 3060 Ti' },
  { value: 'RTX 3060', label: 'NVIDIA RTX 3060' },
  { value: 'RTX 2080 Ti', label: 'NVIDIA RTX 2080 Ti' },
  { value: 'RTX 2080', label: 'NVIDIA RTX 2080' },
  { value: 'RTX 2070', label: 'NVIDIA RTX 2070' },
  { value: 'RTX 2060', label: 'NVIDIA RTX 2060' },
  { value: 'GTX 1080 Ti', label: 'NVIDIA GTX 1080 Ti' },
  { value: 'GTX 1080', label: 'NVIDIA GTX 1080' },
  { value: 'GTX 1070', label: 'NVIDIA GTX 1070' },
  { value: 'GTX 1060', label: 'NVIDIA GTX 1060' },
  { value: 'RX 7900 XTX', label: 'AMD RX 7900 XTX' },
  { value: 'RX 7900 XT', label: 'AMD RX 7900 XT' },
  { value: 'RX 7800 XT', label: 'AMD RX 7800 XT' },
  { value: 'RX 7700 XT', label: 'AMD RX 7700 XT' },
  { value: 'RX 7600', label: 'AMD RX 7600' },
  { value: 'RX 6900 XT', label: 'AMD RX 6900 XT' },
  { value: 'RX 6800 XT', label: 'AMD RX 6800 XT' },
  { value: 'RX 6700 XT', label: 'AMD RX 6700 XT' },
  { value: 'M3 Ultra', label: 'Apple M3 Ultra' },
  { value: 'M3 Max', label: 'Apple M3 Max' },
  { value: 'M3 Pro', label: 'Apple M3 Pro' },
  { value: 'M3', label: 'Apple M3' },
  { value: 'M2 Ultra', label: 'Apple M2 Ultra' },
  { value: 'M2 Max', label: 'Apple M2 Max' },
  { value: 'M2 Pro', label: 'Apple M2 Pro' },
  { value: 'M2', label: 'Apple M2' },
  { value: 'M1 Ultra', label: 'Apple M1 Ultra' },
  { value: 'M1 Max', label: 'Apple M1 Max' },
  { value: 'M1 Pro', label: 'Apple M1 Pro' },
  { value: 'M1', label: 'Apple M1' },
  { value: 'A100', label: 'NVIDIA A100' },
  { value: 'H100', label: 'NVIDIA H100' },
  { value: 'A6000', label: 'NVIDIA A6000' },
  { value: 'T4', label: 'NVIDIA T4' },
  { value: 'other', label: 'Other GPU' },
];

const RAM_OPTIONS = [
  { value: 8, label: '8 GB' },
  { value: 16, label: '16 GB' },
  { value: 32, label: '32 GB' },
  { value: 64, label: '64 GB' },
  { value: 128, label: '128 GB' },
  { value: 256, label: '256 GB+' },
];

const INTERNET_OPTIONS = [
  { value: 25, label: '25 Mbps (Basic)' },
  { value: 50, label: '50 Mbps' },
  { value: 100, label: '100 Mbps' },
  { value: 200, label: '200 Mbps' },
  { value: 500, label: '500 Mbps' },
  { value: 1000, label: '1 Gbps+' },
];

const OS_OPTIONS = [
  { value: 'Windows', label: 'Windows' },
  { value: 'macOS', label: 'macOS' },
  { value: 'Linux', label: 'Linux' },
];

const inputStyles = "w-full px-4 py-3 bg-neutral-950 border border-neutral-800 text-white focus:outline-none focus:border-accent transition-colors text-sm";
const selectStyles = `${inputStyles} appearance-none cursor-pointer`;
const labelStyles = "block text-xs font-mono uppercase tracking-widest text-neutral-500 mb-2";

export const WaitlistSignup = () => {
  const [searchParams] = useSearchParams();

  const [step, setStep] = useState<Step>('hardware');
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');
  const [copied, setCopied] = useState(false);

  const [referralValid, setReferralValid] = useState<boolean | null>(null);
  const [referralInfo, setReferralInfo] = useState<string>('');

  const [hardware, setHardware] = useState<HardwareData>({
    email: '',
    gpu_model: '',
    gpu_vram: null,
    ram_gb: 16,
    internet_speed: 100,
    operating_system: 'Windows',
    referral_code: searchParams.get('ref') || '',
  });

  const [result, setResult] = useState<WaitlistResult | null>(null);

  useEffect(() => {
    const checkReferral = async () => {
      if (hardware.referral_code && hardware.referral_code.length >= 6) {
        try {
          const response = await axios.get(
            `${API_URL}/api/waitlist/check-referral?code=${hardware.referral_code}`
          );
          setReferralValid(response.data.valid);
          setReferralInfo(response.data.message);
        } catch {
          setReferralValid(false);
          setReferralInfo('Invalid referral code');
        }
      } else {
        setReferralValid(null);
        setReferralInfo('');
      }
    };

    const timer = setTimeout(checkReferral, 500);
    return () => clearTimeout(timer);
  }, [hardware.referral_code]);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setError('');
    setStep('calculating');
    setLoading(true);

    await new Promise(resolve => setTimeout(resolve, 2500));

    try {
      const response = await axios.post(`${API_URL}/api/waitlist/signup`, {
        email: hardware.email,
        gpu_model: hardware.gpu_model || null,
        gpu_vram: hardware.gpu_vram,
        ram_gb: hardware.ram_gb,
        internet_speed: hardware.internet_speed,
        operating_system: hardware.operating_system,
        referral_code: hardware.referral_code || null,
      });

      setResult(response.data);
      setStep('results');
    } catch (err: any) {
      setError(err.response?.data?.detail || 'Failed to join waitlist. Please try again.');
      setStep('hardware');
    } finally {
      setLoading(false);
    }
  };

  const copyReferralLink = () => {
    if (result?.referral_url) {
      navigator.clipboard.writeText(result.referral_url);
      setCopied(true);
      setTimeout(() => setCopied(false), 2000);
    }
  };

  const shareOnTwitter = () => {
    if (result) {
      const text = `I just reserved my spot to contribute to @NeuroShardAI - the decentralized AI network. Join the distributed intelligence revolution:`;
      const url = result.referral_url;
      window.open(`https://twitter.com/intent/tweet?text=${encodeURIComponent(text)}&url=${encodeURIComponent(url)}`, '_blank');
    }
  };

  if (step === 'hardware') {
    return (
      <>
        <SEO title="Join the Waitlist" description="Register your hardware and join the NeuroShard waitlist to earn NEURO tokens." />
        <div className="min-h-screen bg-neutral-950 pt-20 sm:pt-28 pb-12 px-4 sm:px-6">
          <div className="max-w-2xl mx-auto">
            <div className="mb-8">
              <h1 className="font-display text-3xl font-bold text-white mb-2">Join the Network</h1>
              <p className="text-neutral-500 text-sm">Register your hardware to reserve your spot in the NeuroShard network.</p>
            </div>

            <div className="bg-neutral-900 border border-neutral-800 p-6">
              {error && (
                <motion.div
                  initial={{ opacity: 0, y: -10 }}
                  animate={{ opacity: 1, y: 0 }}
                  className="mb-6 p-4 bg-red-950/30 border border-red-900/50 flex items-start gap-3"
                >
                  <AlertCircle className="w-5 h-5 text-red-400 flex-shrink-0 mt-0.5" />
                  <p className="text-red-300 text-sm">{error}</p>
                </motion.div>
              )}

              <form onSubmit={handleSubmit} className="space-y-6">
                <div>
                  <label className={labelStyles}>Email Address</label>
                  <input
                    type="email"
                    value={hardware.email}
                    onChange={(e) => setHardware({ ...hardware, email: e.target.value })}
                    className={inputStyles}
                    placeholder="you@example.com"
                    required
                  />
                </div>

                <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                  <div>
                    <label className={`flex items-center gap-2 ${labelStyles}`}>
                      <Cpu className="w-3.5 h-3.5 text-accent" />
                      GPU Model
                    </label>
                    <select
                      value={hardware.gpu_model}
                      onChange={(e) => setHardware({ ...hardware, gpu_model: e.target.value })}
                      className={selectStyles}
                    >
                      {GPU_OPTIONS.map(opt => (
                        <option key={opt.value} value={opt.value}>{opt.label}</option>
                      ))}
                    </select>
                  </div>

                  <div>
                    <label className={`flex items-center gap-2 ${labelStyles}`}>
                      <Monitor className="w-3.5 h-3.5 text-neutral-500" />
                      VRAM
                    </label>
                    <select
                      value={hardware.gpu_vram || ''}
                      onChange={(e) => setHardware({ ...hardware, gpu_vram: e.target.value ? parseInt(e.target.value) : null })}
                      className={selectStyles}
                      disabled={!hardware.gpu_model || hardware.gpu_model === 'none'}
                    >
                      <option value="">Select VRAM...</option>
                      {[4, 6, 8, 10, 12, 16, 24, 32, 48, 80].map(v => (
                        <option key={v} value={v}>{v === 80 ? '80 GB+' : `${v} GB`}</option>
                      ))}
                    </select>
                  </div>
                </div>

                <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                  <div>
                    <label className={`flex items-center gap-2 ${labelStyles}`}>
                      <Zap className="w-3.5 h-3.5 text-neutral-500" />
                      System RAM
                    </label>
                    <select
                      value={hardware.ram_gb}
                      onChange={(e) => setHardware({ ...hardware, ram_gb: parseInt(e.target.value) })}
                      className={selectStyles}
                      required
                    >
                      {RAM_OPTIONS.map(opt => (
                        <option key={opt.value} value={opt.value}>{opt.label}</option>
                      ))}
                    </select>
                  </div>

                  <div>
                    <label className={`flex items-center gap-2 ${labelStyles}`}>
                      <Wifi className="w-3.5 h-3.5 text-neutral-500" />
                      Internet Speed
                    </label>
                    <select
                      value={hardware.internet_speed || ''}
                      onChange={(e) => setHardware({ ...hardware, internet_speed: e.target.value ? parseInt(e.target.value) : null })}
                      className={selectStyles}
                    >
                      {INTERNET_OPTIONS.map(opt => (
                        <option key={opt.value} value={opt.value}>{opt.label}</option>
                      ))}
                    </select>
                  </div>
                </div>

                <div>
                  <label className={labelStyles}>Operating System</label>
                  <div className="flex gap-2">
                    {OS_OPTIONS.map(opt => (
                      <button
                        key={opt.value}
                        type="button"
                        onClick={() => setHardware({ ...hardware, operating_system: opt.value })}
                        className={`flex-1 py-3 px-4 border font-medium text-sm transition-all ${
                          hardware.operating_system === opt.value
                            ? 'bg-accent/10 border-accent/30 text-accent'
                            : 'bg-neutral-950 border-neutral-800 text-neutral-500 hover:border-neutral-700'
                        }`}
                      >
                        {opt.label}
                      </button>
                    ))}
                  </div>
                </div>

                <div>
                  <label className={`flex items-center gap-2 ${labelStyles}`}>
                    <Link2 className="w-3.5 h-3.5 text-neutral-500" />
                    Referral Code (Optional)
                  </label>
                  <div className="relative">
                    <input
                      type="text"
                      value={hardware.referral_code}
                      onChange={(e) => setHardware({ ...hardware, referral_code: e.target.value.toUpperCase() })}
                      className={`${inputStyles} ${
                        referralValid === true ? 'border-accent/50' :
                        referralValid === false ? 'border-red-900/50' :
                        ''
                      }`}
                      placeholder="Enter friend's code for priority boost"
                      maxLength={12}
                    />
                    {referralValid !== null && (
                      <div className={`absolute right-3 top-1/2 -translate-y-1/2 text-sm ${
                        referralValid ? 'text-accent' : 'text-red-400'
                      }`}>
                        {referralValid ? <Check className="w-4 h-4" /> : <AlertCircle className="w-4 h-4" />}
                      </div>
                    )}
                  </div>
                  {referralInfo && (
                    <p className={`mt-1.5 text-xs ${referralValid ? 'text-accent' : 'text-red-400'}`}>
                      {referralInfo}
                    </p>
                  )}
                </div>

                <button
                  type="submit"
                  disabled={loading || !hardware.email}
                  className="w-full py-3 bg-accent text-neutral-950 font-bold text-sm hover:bg-accent/90 transition-colors disabled:opacity-50 disabled:cursor-not-allowed flex items-center justify-center gap-2"
                >
                  Reserve My Spot
                  <ArrowRight className="w-4 h-4" />
                </button>
              </form>

              <p className="text-center text-neutral-600 text-sm mt-6">
                Already on the waitlist? <a href="/login" className="text-accent hover:underline">Check your status</a>
              </p>
            </div>
          </div>
        </div>
      </>
    );
  }

  if (step === 'calculating') {
    return (
      <div className="min-h-screen bg-neutral-950 pt-20 sm:pt-28 pb-12 px-4 sm:px-6 flex items-center justify-center">
        <div className="text-center">
          <RefreshCw className="w-6 h-6 text-accent animate-spin mx-auto mb-4" />
          <p className="text-white font-medium text-sm mb-1">Registering your hardware...</p>
          <p className="text-neutral-600 text-xs font-mono">This will only take a moment.</p>
        </div>
      </div>
    );
  }

  if (step === 'results' && result) {
    return (
      <>
        <SEO title="Waitlist Confirmed" description="You are on the list! Share your referral link to boost your priority." />
        <div className="min-h-screen bg-neutral-950 pt-20 sm:pt-28 pb-12 px-4 sm:px-6">
          <div className="max-w-2xl mx-auto">
            <div className="mb-8">
              <h1 className="font-display text-3xl font-bold text-white mb-2">You're on the List</h1>
            </div>

            <div className="bg-neutral-900 border border-neutral-800 p-6 mb-4">
              <div className="text-center py-4">
                <p className="text-xs font-mono uppercase tracking-widest text-neutral-500 mb-3">Estimated Daily Earnings</p>
                <div className="flex items-baseline justify-center gap-2">
                  <span className="font-display text-4xl font-bold text-white">
                    {result.estimated_daily_neuro.toFixed(2)}
                  </span>
                  <span className="text-neutral-500 font-mono text-sm">NEURO</span>
                </div>
                <p className="text-neutral-600 text-xs mt-2 font-mono">
                  Based on your hardware configuration
                </p>
              </div>
            </div>

            <div className="bg-neutral-900 border border-neutral-800 p-6 mb-4">
              <div className="flex items-center gap-2 mb-4">
                <Link2 className="w-4 h-4 text-accent" />
                <h3 className="text-white font-bold text-sm">Your Neuro Link</h3>
              </div>

              <p className="text-neutral-500 text-sm mb-4">
                Share this link to move up in the queue. Each referral gives you +10 priority points.
              </p>

              <div className="flex gap-2 mb-4">
                <div className="flex-1 bg-neutral-950 border border-neutral-800 px-4 py-3 font-mono text-accent text-sm truncate">
                  {result.referral_url}
                </div>
                <button
                  onClick={copyReferralLink}
                  className={`px-4 py-3 font-medium transition-all flex items-center gap-2 border ${
                    copied
                      ? 'bg-accent/10 text-accent border-accent/30'
                      : 'bg-neutral-800 text-neutral-300 hover:bg-neutral-700 border-neutral-700'
                  }`}
                >
                  {copied ? <Check className="w-4 h-4" /> : <Copy className="w-4 h-4" />}
                </button>
              </div>

              <button
                onClick={shareOnTwitter}
                className="w-full py-3 border border-neutral-800 text-neutral-300 hover:bg-neutral-800 text-sm font-medium transition-colors flex items-center justify-center gap-2"
              >
                <Share2 className="w-4 h-4" />
                Share on X
              </button>
            </div>

            <div className="bg-neutral-900 border border-neutral-800 p-6">
              <h3 className="text-white font-bold text-sm mb-4">What happens next?</h3>
              <div className="space-y-4 text-sm">
                {[
                  "We'll review your hardware and approve your node for the network.",
                  "You'll receive an email when approved with instructions to complete registration.",
                  'Download the NeuroShard node, create your wallet, and start earning NEURO.',
                ].map((text, i) => (
                  <div key={i} className="flex items-start gap-3">
                    <span className="w-5 h-5 bg-neutral-800 text-neutral-500 flex items-center justify-center text-xs font-mono flex-shrink-0 border border-neutral-700">
                      {i + 1}
                    </span>
                    <p className="text-neutral-500">{text}</p>
                  </div>
                ))}
              </div>
            </div>

            <p className="text-center text-neutral-700 text-xs mt-6 font-mono">
              Earnings estimates are projections based on current network conditions. Actual earnings may vary.
            </p>
          </div>
        </div>
      </>
    );
  }

  return null;
};
