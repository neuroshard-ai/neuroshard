import { motion } from 'framer-motion';
import { Cpu, Zap, Shield, Layers } from 'lucide-react';

const technologies = [
  {
    icon: <Cpu className="w-6 h-6" />,
    title: 'Speculative Decoding',
    description: 'Clients generate draft tokens locally, which are verified in batches by the network. This "hides" network latency, making distributed inference feel real-time.',
  },
  {
    icon: <Layers className="w-6 h-6" />,
    title: 'Smart Sharding & Caching',
    description: 'Nodes maintain KV caches for active sessions ("Session Affinity"). Only new tokens are transmitted, reducing complexity from O(N\u00B2) to O(N).',
  },
  {
    icon: <Zap className="w-6 h-6" />,
    title: '8x Bandwidth Reduction',
    description: 'Activations are quantized to INT8 and compressed with Zlib before transmission via gRPC/Protobuf, enabling high-speed relay over consumer internet.',
  },
  {
    icon: <Shield className="w-6 h-6" />,
    title: 'Proof of Neural Work',
    description: 'A revolutionary consensus mechanism that rewards useful computation instead of idle uptime. Active nodes earn significantly more, creating a robust "Give-to-Get" economy.',
  },
];

export const Technology = () => {
  return (
    <section className="py-16 sm:py-24 bg-neutral-950 relative overflow-hidden">
      <div className="max-w-7xl mx-auto px-4 sm:px-6">
        <div className="mb-16">
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            whileInView={{ opacity: 1, y: 0 }}
            viewport={{ once: true }}
          >
            <p className="text-[10px] font-mono uppercase tracking-widest text-accent mb-4">Technology</p>
            <h2 className="font-display text-3xl md:text-5xl font-bold text-white mb-4 tracking-tight">
              Powered by <span className="text-accent">NeuroShard</span> Protocol
            </h2>
          </motion.div>
          <motion.p
            initial={{ opacity: 0, y: 10 }}
            whileInView={{ opacity: 1, y: 0 }}
            viewport={{ once: true }}
            transition={{ delay: 0.1 }}
            className="text-neutral-400 text-base max-w-2xl"
          >
            We've solved the latency and bandwidth challenges of distributed AI through a novel combination of pipeline parallelism and optimistic consensus.
          </motion.p>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-2 gap-px bg-neutral-800">
          {technologies.map((tech, index) => (
            <motion.div
              key={index}
              initial={{ opacity: 0, y: 20 }}
              whileInView={{ opacity: 1, y: 0 }}
              viewport={{ once: true }}
              transition={{ delay: index * 0.08 }}
              className="bg-neutral-950 p-8 group hover:bg-neutral-900/50 transition-colors"
            >
              <div className="flex flex-col sm:flex-row gap-5">
                <div className="flex-shrink-0 text-neutral-600 group-hover:text-accent transition-colors">
                  {tech.icon}
                </div>
                <div>
                  <h3 className="font-display text-lg font-bold text-white mb-2">{tech.title}</h3>
                  <p className="text-neutral-500 text-sm leading-relaxed">{tech.description}</p>
                </div>
              </div>
            </motion.div>
          ))}
        </div>
      </div>
    </section>
  );
};
