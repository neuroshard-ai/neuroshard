import { useState } from 'react';
import { motion } from 'framer-motion';
import { 
 Terminal,
 Copy,
 Check,
 Cpu,
 Zap,
 Monitor,
 Globe,
 Shield,
 ChevronRight,
 ExternalLink,
 Server
} from 'lucide-react';
import { SEO } from './SEO';

type Platform = 'any' | 'nvidia' | 'apple' | 'jetson';

export const DownloadPage = () => {
 const [selectedPlatform, setSelectedPlatform] = useState<Platform>('any');
 const [copiedCommand, setCopiedCommand] = useState<string | null>(null);

 const copyToClipboard = (text: string, id: string) => {
 navigator.clipboard.writeText(text);
 setCopiedCommand(id);
 setTimeout(() => setCopiedCommand(null), 2000);
 };

 const platforms = [
 {
 id: 'any' as Platform,
 name: 'CPU / Any',
 icon: <Cpu className="w-6 h-6" />,
 description: 'Works everywhere',
 flatBg: 'bg-blue-500',
 },
 {
 id: 'nvidia' as Platform,
 name: 'NVIDIA GPU',
 icon: <Zap className="w-6 h-6" />,
 description: '10x faster training',
 flatBg: 'bg-green-500',
 },
 {
 id: 'apple' as Platform,
 name: 'Apple Silicon',
 icon: <Monitor className="w-6 h-6" />,
 description: 'M1/M2/M3/M4 GPU',
 flatBg: 'bg-purple-500',
 },
 {
 id: 'jetson' as Platform,
 name: 'NVIDIA Jetson',
 icon: <Server className="w-6 h-6" />,
 description: 'ARM64 Edge AI',
 flatBg: 'bg-orange-500',
 },
 ];

 const getInstallCommands = () => {
 switch (selectedPlatform) {
 case 'nvidia':
 return [
 { id: 'neuroshard', label: 'Install with GPU support', cmd: 'pip install nexaroa[gpu]' },
 ];
 case 'apple':
 return [
 { id: 'neuroshard', label: 'Install with GPU support (MPS auto-detected)', cmd: 'pip install nexaroa[gpu]' },
 ];
 case 'jetson':
 return [
 { id: 'torch-jetson', label: 'Install PyTorch from NVIDIA (JetPack 6.x)', cmd: 'pip install torch torchvision --extra-index-url https://developer.download.nvidia.com/compute/redist/jp/v60' },
 { id: 'neuroshard', label: 'Install NeuroShard', cmd: 'pip install nexaroa' },
 ];
 default:
 return [
 { id: 'neuroshard', label: 'Install NeuroShard', cmd: 'pip install nexaroa' },
 ];
 }
 };

 const runCommand = selectedPlatform === 'jetson' 
 ? 'neuroshard --token YOUR_TOKEN --device cuda'
 : 'neuroshard --token YOUR_TOKEN';

 return (
 <>
 <SEO title="Download" description="Download and install the NeuroShard node to start contributing to the network." />
 <div className="min-h-screen bg-neutral-950">

 <div className="relative z-10 container mx-auto px-4 sm:px-6 lg:px-8 pt-20 sm:pt-24 pb-20">
 {/* Hero Section */}
 <motion.div
 initial={{ opacity: 0, y: 20 }}
 animate={{ opacity: 1, y: 0 }}
 className="text-center mb-16"
 >
 <div className="inline-flex items-center gap-2 px-4 py-2 bg-neutral-800/50 border border-neutral-700 mb-6">
 <div className="relative flex h-2 w-2">
 <span className="animate-ping absolute inline-flex h-full w-full bg-accent opacity-75"></span>
 <span className="relative inline-flex h-2 w-2 bg-accent"></span>
 </div>
 <span className="text-sm font-medium text-accent">
 Install via pip
 </span>
 </div>

 <h1 className="text-5xl md:text-6xl lg:text-7xl font-bold text-white mb-6 tracking-tight font-display">
 Get Started with
 <span className="block text-accent">
 NeuroShard
 </span>
 </h1>

 <p className="text-xl text-neutral-400 max-w-2xl mx-auto">
 One command to join the decentralized AI network. Start earning NEURO tokens by sharing your computing power.
 </p>
 </motion.div>

 {/* Platform Selection */}
 <motion.div
 initial={{ opacity: 0, y: 30 }}
 animate={{ opacity: 1, y: 0 }}
 transition={{ delay: 0.1 }}
 className="max-w-4xl mx-auto mb-8"
 >
 <h2 className="text-lg font-semibold text-white mb-4 text-center font-display">Select your platform</h2>
 <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
 {platforms.map((platform) => (
 <button
 key={platform.id}
 onClick={() => setSelectedPlatform(platform.id)}
 className={`p-4 border transition-all ${
 selectedPlatform === platform.id
 ? `${platform.flatBg} border-transparent text-white `
 : 'bg-neutral-900/50 border-neutral-800 text-neutral-400 hover:border-neutral-700'
 }`}
 >
 <div className="flex flex-col items-center gap-2">
 {platform.icon}
 <span className="font-semibold text-sm">{platform.name}</span>
 <span className="text-xs opacity-80">{platform.description}</span>
 </div>
 </button>
 ))}
 </div>
 </motion.div>

 {/* Installation Commands */}
 <motion.div
 initial={{ opacity: 0, y: 30 }}
 animate={{ opacity: 1, y: 0 }}
 transition={{ delay: 0.15 }}
 className="max-w-4xl mx-auto mb-12"
 >
 <div className="bg-neutral-900 border border-neutral-800 overflow-hidden">
 {/* Terminal Header */}
 <div className="flex items-center gap-2 px-4 py-3 bg-neutral-950 border-b border-neutral-800">
 <div className="flex gap-1.5">
 <div className="w-3 h-3 bg-red-500"></div>
 <div className="w-3 h-3 bg-yellow-500"></div>
 <div className="w-3 h-3 bg-green-500"></div>
 </div>
 <span className="text-xs text-neutral-500 ml-2">Terminal</span>
 </div>

 {/* Commands */}
 <div className="p-6 space-y-4">
 {/* Step 1: Install */}
 <div>
 <div className="flex items-center gap-2 mb-2">
 <span className="w-6 h-6 bg-accent/10 text-accent flex items-center justify-center text-xs font-bold">1</span>
 <span className="text-sm text-neutral-400">Install</span>
 </div>
 {getInstallCommands().map((command, idx) => (
 <div key={command.id} className={`${idx > 0 ? 'mt-2' : ''}`}>
 {getInstallCommands().length > 1 && (
 <p className="text-xs text-neutral-500 mb-1 ml-8">{command.label}</p>
 )}
 <div className="flex items-center gap-2 bg-neutral-950 p-3 border border-neutral-800 overflow-x-auto">
 <span className="text-green-400 font-mono">$</span>
 <code className="flex-1 text-accent font-mono text-sm">{command.cmd}</code>
 <button
 onClick={() => copyToClipboard(command.cmd, command.id)}
 className="p-1.5 hover:bg-neutral-800 text-neutral-400 hover:text-white transition-colors"
 >
 {copiedCommand === command.id ? (
 <Check className="w-4 h-4 text-green-400" />
 ) : (
 <Copy className="w-4 h-4" />
 )}
 </button>
 </div>
 </div>
 ))}
 </div>

 {/* Step 2: Run */}
 <div>
 <div className="flex items-center gap-2 mb-2">
 <span className="w-6 h-6 bg-purple-500/20 text-purple-400 flex items-center justify-center text-xs font-bold">2</span>
 <span className="text-sm text-neutral-400">Run your node</span>
 </div>
 <div className="flex items-center gap-2 bg-neutral-950 p-3 border border-neutral-800">
 <span className="text-green-400 font-mono">$</span>
 <code className="flex-1 text-accent font-mono text-sm">{runCommand}</code>
 <button
 onClick={() => copyToClipboard(runCommand, 'run')}
 className="p-1.5 hover:bg-neutral-800 text-neutral-400 hover:text-white transition-colors"
 >
 {copiedCommand === 'run' ? (
 <Check className="w-4 h-4 text-green-400" />
 ) : (
 <Copy className="w-4 h-4" />
 )}
 </button>
 </div>
 <p className="text-xs text-neutral-500 mt-2 ml-8">
 Get your token at <a href="https://neuroshard.com/register" className="text-accent hover:underline">neuroshard.com/register</a>
 </p>
 </div>

 {/* Step 3: Dashboard */}
 <div>
 <div className="flex items-center gap-2 mb-2">
 <span className="w-6 h-6 bg-green-500/20 text-green-400 flex items-center justify-center text-xs font-bold">3</span>
 <span className="text-sm text-neutral-400">View your dashboard</span>
 </div>
 <div className="flex items-center gap-2 bg-neutral-950 p-3 border border-neutral-800">
 <Globe className="w-4 h-4 text-neutral-500" />
 <code className="flex-1 text-neutral-300 font-mono text-sm">http://localhost:8000</code>
 <span className="text-xs text-neutral-500">Opens automatically</span>
 </div>
 </div>
 </div>
 </div>
 </motion.div>

 {/* CLI Options */}
 <motion.div
 initial={{ opacity: 0, y: 20 }}
 animate={{ opacity: 1, y: 0 }}
 transition={{ delay: 0.2 }}
 className="max-w-4xl mx-auto mb-16"
 >
 <h2 className="text-2xl font-bold text-white mb-6 text-center font-display">Command Options</h2>

 <div className="bg-neutral-900/50 border border-neutral-800 overflow-hidden">
 <table className="w-full text-sm">
 <thead>
 <tr className="border-b border-neutral-800">
 <th className="text-left text-neutral-400 font-medium px-6 py-3">Option</th>
 <th className="text-left text-neutral-400 font-medium px-6 py-3">Description</th>
 <th className="text-left text-neutral-400 font-medium px-6 py-3">Default</th>
 </tr>
 </thead>
 <tbody className="divide-y divide-neutral-800">
 <tr>
 <td className="px-6 py-3 font-mono text-accent">--token</td>
 <td className="px-6 py-3 text-neutral-300">Your wallet recovery phrase</td>
 <td className="px-6 py-3 text-neutral-500">Required</td>
 </tr>
 <tr>
 <td className="px-6 py-3 font-mono text-accent">--device</td>
 <td className="px-6 py-3 text-neutral-300">Compute device (auto, cuda, mps, cpu)</td>
 <td className="px-6 py-3 text-neutral-500">auto</td>
 </tr>
 <tr>
 <td className="px-6 py-3 font-mono text-accent">--port</td>
 <td className="px-6 py-3 text-neutral-300">HTTP/Dashboard port</td>
 <td className="px-6 py-3 text-neutral-500">8000</td>
 </tr>
 <tr>
 <td className="px-6 py-3 font-mono text-accent">--memory</td>
 <td className="px-6 py-3 text-neutral-300">Max memory limit (MB)</td>
 <td className="px-6 py-3 text-neutral-500">4096</td>
 </tr>
 <tr>
 <td className="px-6 py-3 font-mono text-accent">--cpu-threads</td>
 <td className="px-6 py-3 text-neutral-300">Max CPU threads</td>
 <td className="px-6 py-3 text-neutral-500">4</td>
 </tr>
 <tr>
 <td className="px-6 py-3 font-mono text-accent">--max-storage</td>
 <td className="px-6 py-3 text-neutral-300">Disk space for training data (MB)</td>
 <td className="px-6 py-3 text-neutral-500">100</td>
 </tr>
 <tr>
 <td className="px-6 py-3 font-mono text-accent">--no-training</td>
 <td className="px-6 py-3 text-neutral-300">Inference only mode</td>
 <td className="px-6 py-3 text-neutral-500">false</td>
 </tr>
 <tr>
 <td className="px-6 py-3 font-mono text-accent">--headless</td>
 <td className="px-6 py-3 text-neutral-300">Don't auto-open browser</td>
 <td className="px-6 py-3 text-neutral-500">false</td>
 </tr>
 <tr>
 <td className="px-6 py-3 font-mono text-accent">--daemon</td>
 <td className="px-6 py-3 text-neutral-300">Run as background service</td>
 <td className="px-6 py-3 text-neutral-500">false</td>
 </tr>
 <tr>
 <td className="px-6 py-3 font-mono text-accent">--stop</td>
 <td className="px-6 py-3 text-neutral-300">Stop the background daemon</td>
 <td className="px-6 py-3 text-neutral-500">-</td>
 </tr>
 <tr>
 <td className="px-6 py-3 font-mono text-accent">--status</td>
 <td className="px-6 py-3 text-neutral-300">Check if daemon is running</td>
 <td className="px-6 py-3 text-neutral-500">-</td>
 </tr>
 <tr>
 <td className="px-6 py-3 font-mono text-accent">--logs</td>
 <td className="px-6 py-3 text-neutral-300">View daemon logs</td>
 <td className="px-6 py-3 text-neutral-500">-</td>
 </tr>
 </tbody>
 </table>
 </div>

 <p className="text-center text-sm text-neutral-500 mt-4">
 Full documentation at{' '}
 <a href="https://docs.neuroshard.com" className="text-accent hover:underline inline-flex items-center gap-1">
 docs.neuroshard.com <ExternalLink className="w-3 h-3" />
 </a>
 </p>
 </motion.div>

 {/* Features Grid */}
 <motion.div
 initial={{ opacity: 0, y: 20 }}
 animate={{ opacity: 1, y: 0 }}
 transition={{ delay: 0.3 }}
 className="max-w-4xl mx-auto"
 >
 <h2 className="text-2xl font-bold text-white mb-8 text-center font-display">What You Get</h2>

 <div className="grid md:grid-cols-2 gap-6">
 <div className="bg-neutral-900/50 border border-neutral-800 p-6">
 <div className="w-12 h-12 bg-accent flex items-center justify-center mb-4">
 <Terminal className="w-6 h-6 text-white" />
 </div>
 <h3 className="font-semibold text-white mb-2 font-display">Web Dashboard</h3>
 <p className="text-sm text-neutral-400">
 Monitor your node status, training progress, NEURO balance, and resource usage from a beautiful local dashboard.
 </p>
 </div>

 <div className="bg-neutral-900/50 border border-neutral-800 p-6">
 <div className="w-12 h-12 bg-green-500 flex items-center justify-center mb-4">
 <Zap className="w-6 h-6 text-white" />
 </div>
 <h3 className="font-semibold text-white mb-2 font-display">Automatic GPU Detection</h3>
 <p className="text-sm text-neutral-400">
 NVIDIA CUDA, Apple Metal, and Jetson ARM64 are auto-detected. Training is 10x faster with GPU acceleration.
 </p>
 </div>

 <div className="bg-neutral-900/50 border border-neutral-800 p-6">
 <div className="w-12 h-12 bg-purple-500 flex items-center justify-center mb-4">
 <Globe className="w-6 h-6 text-white" />
 </div>
 <h3 className="font-semibold text-white mb-2 font-display">Swarm Architecture</h3>
 <p className="text-sm text-neutral-400">
 Fault-tolerant multipath routing ensures your work is never lost. DiLoCo reduces sync bandwidth by 90%.
 </p>
 </div>

 <div className="bg-neutral-900/50 border border-neutral-800 p-6">
 <div className="w-12 h-12 bg-orange-500 flex items-center justify-center mb-4">
 <Shield className="w-6 h-6 text-white" />
 </div>
 <h3 className="font-semibold text-white mb-2 font-display">ECDSA Cryptography</h3>
 <p className="text-sm text-neutral-400">
 All Proof of Neural Work is cryptographically signed. Your rewards are verifiable and tamper-proof.
 </p>
 </div>
 </div>
 </motion.div>

 {/* Resource Configuration */}
 <motion.div
 initial={{ opacity: 0, y: 20 }}
 animate={{ opacity: 1, y: 0 }}
 transition={{ delay: 0.35 }}
 className="max-w-4xl mx-auto mt-16"
 >
 <h3 className="text-xl font-bold text-white mb-6 text-center font-display">Resource Configuration</h3>
 
 <div className="bg-neutral-900/50 border border-neutral-800 p-6 space-y-4">
 <div className="grid md:grid-cols-3 gap-4">
 <div className="bg-neutral-950/50 p-4">
 <h4 className="text-accent font-mono text-sm mb-2">--memory</h4>
 <p className="text-neutral-400 text-sm">
 Memory for model layers. More memory = more layers = higher rewards.
 </p>
 <p className="text-neutral-500 text-xs mt-2">
 CPU: Use ~30% of RAM • GPU: Can use more
 </p>
 </div>
 
 <div className="bg-neutral-950/50 p-4">
 <h4 className="text-accent font-mono text-sm mb-2">--max-storage</h4>
 <p className="text-neutral-400 text-sm">
 Disk space for training data shards (~10MB each). More storage = faster shard rotation.
 </p>
 <p className="text-neutral-500 text-xs mt-2">
 Default: 100MB (10 shards) • Suggested: 500MB+
 </p>
 </div>
 
 <div className="bg-neutral-950/50 p-4">
 <h4 className="text-accent font-mono text-sm mb-2">--cpu-threads</h4>
 <p className="text-neutral-400 text-sm">
 CPU cores for training. Lower values reduce system impact during training.
 </p>
 <p className="text-neutral-500 text-xs mt-2">
 Default: 4 • Use fewer for background operation
 </p>
 </div>
 </div>
 
 <div className="border-t border-neutral-800 pt-4 mt-4">
 <p className="text-sm text-neutral-400">
 <span className="text-accent font-semibold">💡 Tip:</span> All these settings can be adjusted live from the dashboard without restarting your node.
 </p>
 </div>
 </div>
 </motion.div>

 {/* System Requirements */}
 <motion.div
 initial={{ opacity: 0, y: 20 }}
 animate={{ opacity: 1, y: 0 }}
 transition={{ delay: 0.4 }}
 className="max-w-4xl mx-auto mt-16"
 >
 <h3 className="text-xl font-bold text-white mb-6 text-center font-display">System Requirements</h3>

 <div className="grid md:grid-cols-2 gap-6">
 <div className="bg-neutral-900/50 border border-neutral-800 p-6">
 <h4 className="font-semibold text-white mb-4 flex items-center gap-2">
 <Cpu className="w-5 h-5 text-accent" />
 Minimum
 </h4>
 <ul className="space-y-2 text-sm text-neutral-400">
 <li className="flex items-center gap-2">
 <ChevronRight className="w-4 h-4 text-accent" />
 Python 3.9+
 </li>
 <li className="flex items-center gap-2">
 <ChevronRight className="w-4 h-4 text-accent" />
 4GB RAM
 </li>
 <li className="flex items-center gap-2">
 <ChevronRight className="w-4 h-4 text-accent" />
 Dual-core CPU
 </li>
 <li className="flex items-center gap-2">
 <ChevronRight className="w-4 h-4 text-accent" />
 Stable internet connection
 </li>
 </ul>
 </div>

 <div className="bg-neutral-900/50 border border-neutral-800 p-6">
 <h4 className="font-semibold text-white mb-4 flex items-center gap-2">
 <Zap className="w-5 h-5 text-green-400" />
 Recommended
 </h4>
 <ul className="space-y-2 text-sm text-neutral-400">
 <li className="flex items-center gap-2">
 <ChevronRight className="w-4 h-4 text-green-400" />
 8GB+ RAM
 </li>
 <li className="flex items-center gap-2">
 <ChevronRight className="w-4 h-4 text-green-400" />
 NVIDIA GPU, Apple Silicon, or Jetson
 </li>
 <li className="flex items-center gap-2">
 <ChevronRight className="w-4 h-4 text-green-400" />
 SSD storage
 </li>
 <li className="flex items-center gap-2">
 <ChevronRight className="w-4 h-4 text-green-400" />
 Always-on machine
 </li>
 </ul>
 </div>
 </div>
 </motion.div>

 {/* PyPI Badge */}
 <motion.div
 initial={{ opacity: 0 }}
 animate={{ opacity: 1 }}
 transition={{ delay: 0.5 }}
 className="text-center mt-16"
 >
 <a
 href="https://pypi.org/project/nexaroa/"
 target="_blank"
 rel="noopener noreferrer"
 className="inline-flex items-center gap-3 px-6 py-3 bg-neutral-900/50 border border-neutral-800 hover:border-neutral-700 transition-colors"
 >
 <img src="https://badge.fury.io/py/nexaroa.svg" alt="PyPI version" className="h-5" />
 <span className="text-neutral-400 text-sm">View on PyPI</span>
 <ExternalLink className="w-4 h-4 text-neutral-500" />
 </a>
 </motion.div>
 </div>
 </div>
 </>
 );
};
