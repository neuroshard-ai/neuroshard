import { motion } from 'framer-motion';
import { Network, ShieldCheck, BrainCircuit, Zap, Wallet, TrendingUp } from 'lucide-react';

const features = [
  {
    icon: <Network className="w-8 h-8" />,
    title: 'Distributed Architecture',
    description: 'No single server holds the entire model. Layers are spread across thousands of devices, your laptop could be running layer 47 right now.',
    stat: '1000x',
    statLabel: 'More resilient than centralized AI',
    color: 'cyan',
  },
  {
    icon: <ShieldCheck className="w-8 h-8" />,
    title: 'Proof of Neural Work',
    description: 'Forget proof-of-stake theater. PoNW rewards actual computation - every token processed, every gradient computed is verified and paid.',
    stat: '100%',
    statLabel: 'Rewards for real work only',
    color: 'green',
  },
  {
    icon: <BrainCircuit className="w-8 h-8" />,
    title: 'Living Intelligence',
    description: 'The network never stops learning. Distributed training means the model improves continuously from user interactions worldwide.',
    stat: '24/7',
    statLabel: 'Continuous learning',
    color: 'purple',
  },
  {
    icon: <Zap className="w-8 h-8" />,
    title: 'Real-Time Inference',
    description: 'Speculative decoding + pipeline parallelism = GPT-level responses without the corporate cloud. Internet-distributed, but feels local.',
    stat: '<2s',
    statLabel: 'Response latency',
    color: 'yellow',
  },
  {
    icon: <Wallet className="w-8 h-8" />,
    title: 'BIP39 Wallet Security',
    description: 'Your keys, your NEURO. 12-word recovery phrase, ECDSA signatures, the same battle-tested crypto that secures billions in Bitcoin.',
    stat: '256-bit',
    statLabel: 'Cryptographic security',
    color: 'blue',
  },
  {
    icon: <TrendingUp className="w-8 h-8" />,
    title: 'Deflationary Economics',
    description: '5% of every transaction is burned forever. As the network grows, NEURO becomes scarcer. Early contributors benefit most.',
    stat: '5%',
    statLabel: 'Burn rate per transfer',
    color: 'orange',
  },
];

const colorClasses = {
  cyan: { bg: 'bg-cyan-500/10', border: 'border-cyan-500/30', text: 'text-cyan-400', iconBg: 'bg-cyan-500' },
  green: { bg: 'bg-green-500/10', border: 'border-green-500/30', text: 'text-green-400', iconBg: 'bg-green-500' },
  purple: { bg: 'bg-purple-500/10', border: 'border-purple-500/30', text: 'text-purple-400', iconBg: 'bg-purple-500' },
  yellow: { bg: 'bg-yellow-500/10', border: 'border-yellow-500/30', text: 'text-yellow-400', iconBg: 'bg-yellow-500' },
  blue: { bg: 'bg-blue-500/10', border: 'border-blue-500/30', text: 'text-blue-400', iconBg: 'bg-blue-500' },
  orange: { bg: 'bg-orange-500/10', border: 'border-orange-500/30', text: 'text-orange-400', iconBg: 'bg-orange-500' },
};

export const Features = () => {
  return (
    <section id="features" className="py-32 bg-slate-950 relative">

      <div className="container mx-auto px-6 relative z-10">
        <motion.div 
          initial={{ opacity: 0, y: 30 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true }}
          className="text-center mb-20"
        >
          <h2 className="text-4xl md:text-5xl font-bold text-white mb-6 font-display">
            Built Different.{' '}
            <span className="text-cyan-400">
              Built Better.
            </span>
          </h2>
          <p className="text-slate-400 text-lg max-w-3xl mx-auto leading-relaxed">
            NeuroShard combines breakthroughs in distributed systems, cryptography, and machine learning 
            into a unified protocol that makes global AI possible.
          </p>
        </motion.div>

        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
          {features.map((feature, index) => {
            const colors = colorClasses[feature.color as keyof typeof colorClasses];
            return (
              <motion.div
                key={index}
                initial={{ opacity: 0, y: 30 }}
                whileInView={{ opacity: 1, y: 0 }}
                viewport={{ once: true }}
                transition={{ delay: index * 0.1, duration: 0.5 }}
                className={`group relative bg-slate-900/50 border ${colors.border} p-8 hover:bg-slate-900/80 transition-all duration-300`}
              >
                {/* Icon */}
                <div className={`inline-flex p-4 ${colors.iconBg} text-white mb-6 shadow-lg group-hover:scale-110 transition-transform duration-300`}>
                  {feature.icon}
                </div>

                {/* Content */}
                <h3 className="text-xl font-bold text-white mb-3 font-display">{feature.title}</h3>
                <p className="text-slate-400 leading-relaxed mb-6">
                  {feature.description}
                </p>

                {/* Stat */}
                <div className={`flex items-baseline gap-2 pt-4 border-t border-slate-800`}>
                  <span className={`text-3xl font-black ${colors.text}`}>{feature.stat}</span>
                  <span className="text-slate-500 text-sm">{feature.statLabel}</span>
                </div>
              </motion.div>
            );
          })}
        </div>
      </div>
    </section>
  );
};
