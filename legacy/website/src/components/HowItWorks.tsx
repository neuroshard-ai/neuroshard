import { motion } from 'framer-motion';

export const HowItWorks = () => {
  return (
    <section className="py-16 sm:py-24 bg-neutral-900 border-y border-neutral-800 overflow-hidden">
      <div className="max-w-7xl mx-auto px-4 sm:px-6">
        <div className="flex flex-col lg:flex-row gap-16">
          <div className="lg:w-1/2">
            <motion.div
              initial={{ opacity: 0, y: 20 }}
              whileInView={{ opacity: 1, y: 0 }}
              viewport={{ once: true }}
            >
              <p className="text-[10px] font-mono uppercase tracking-widest text-accent mb-4">How It Works</p>
              <h2 className="font-display text-3xl md:text-4xl font-bold text-white mb-6 tracking-tight">
                The Global Relay Race
              </h2>
            </motion.div>

            <motion.p
              initial={{ opacity: 0, y: 10 }}
              whileInView={{ opacity: 1, y: 0 }}
              viewport={{ once: true }}
              transition={{ delay: 0.1 }}
              className="text-neutral-400 text-base mb-10 leading-relaxed"
            >
              Instead of one massive server holding the entire model, NeuroShard distributes layers across the network. Intelligence emerges from the collective relay of tokens.
            </motion.p>

            <ul className="space-y-8">
              {[
                { title: 'Model Sharding', desc: 'Node A loads Layers 0-4. Node B loads Layers 4-8. Memory requirements are slashed, enabling consumer hardware participation.' },
                { title: 'Token Relay', desc: 'Activations are quantized, compressed, and relayed via gRPC. "Session Affinity" ensures consistent routing for caching.' },
                { title: 'Speculative Execution', desc: 'Clients generate draft tokens locally. The network verifies them in parallel, hiding internet latency.' },
                { title: 'Proof of Neural Work', desc: 'Nodes are rewarded for verified computation. Audits ensure trust without redundant execution, creating a fair economy.' },
              ].map((item, idx) => (
                <motion.li
                  key={idx}
                  initial={{ opacity: 0, y: 10 }}
                  whileInView={{ opacity: 1, y: 0 }}
                  viewport={{ once: true }}
                  transition={{ delay: 0.15 + idx * 0.08 }}
                  className="flex gap-4"
                >
                  <span className="flex-shrink-0 w-8 h-8 bg-neutral-800 text-accent flex items-center justify-center font-mono text-sm font-bold border border-neutral-700">
                    {idx + 1}
                  </span>
                  <div>
                    <h4 className="text-white font-bold mb-1">{item.title}</h4>
                    <p className="text-neutral-500 text-sm leading-relaxed">{item.desc}</p>
                  </div>
                </motion.li>
              ))}
            </ul>
          </div>

          <div className="lg:w-1/2">
            <motion.div
              initial={{ opacity: 0, scale: 0.95 }}
              whileInView={{ opacity: 1, scale: 1 }}
              viewport={{ once: true }}
              className="bg-neutral-950 border border-neutral-800 p-8"
            >
              <div className="flex flex-col md:flex-row justify-between items-center mb-8 relative gap-8 md:gap-0">
                <div className="hidden md:block absolute top-1/2 left-0 w-full h-px bg-neutral-800 -z-10" />
                <div className="md:hidden absolute left-1/2 top-0 w-px h-full bg-neutral-800 -z-10 -translate-x-1/2" />

                {['Alice', 'Bob', 'Charlie'].map((name, i) => (
                  <div key={name} className="flex flex-col items-center gap-2 bg-neutral-900 border border-neutral-800 p-4 z-10 min-w-[100px] w-full md:w-auto">
                    <div className={`w-2.5 h-2.5 ${i === 1 ? 'bg-accent' : 'bg-neutral-600'}`} />
                    <span className="font-mono text-xs text-neutral-500">Node {name}</span>
                    <span className="text-xs text-accent font-bold font-mono">L{i * 4}—{(i + 1) * 4}</span>
                  </div>
                ))}
              </div>

              <div className="font-mono text-xs bg-neutral-900 border border-neutral-800 p-4 text-neutral-500 space-y-1 overflow-x-auto">
                <p className="opacity-50">{`> Session ID: 8f92-a3b1...`}</p>
                <p className="text-neutral-300">{`> Node Alice: Processing Batch (Draft K=5)...`}</p>
                <p className="text-neutral-300">{`> Node Alice: Forward Pass Layers 0-4 [OK]`}</p>
                <p className="text-accent">{`> Relaying 0.37MB compressed tensor to Bob...`}</p>
                <p className="animate-pulse text-neutral-400">{`> Node Bob: Receiving...`}</p>
              </div>
            </motion.div>
          </div>
        </div>
      </div>
    </section>
  );
};
