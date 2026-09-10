import { motion } from 'framer-motion';
import { Globe, Coins, Brain, ArrowRight } from 'lucide-react';

export const Vision = () => {
  return (
    <section className="py-20 sm:py-32 bg-neutral-950 relative">
      <div className="max-w-7xl mx-auto px-4 sm:px-6">
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true }}
          className="mb-20"
        >
          <p className="text-[10px] font-mono uppercase tracking-widest text-accent mb-4">The Vision</p>
          <h2 className="font-display text-3xl sm:text-4xl md:text-6xl font-bold text-white mb-6 leading-[1.05] tracking-tightest max-w-3xl">
            What If AI Belonged To Everyone?
          </h2>
          <p className="text-lg text-neutral-400 max-w-3xl leading-relaxed">
            Today's AI is locked in corporate data centers. We're building something different --
            a <span className="text-white">globally distributed intelligence</span> that
            anyone can contribute to, and everyone can benefit from.
          </p>
        </motion.div>

        <div className="grid lg:grid-cols-2 gap-px bg-neutral-800 mb-24">
          <motion.div
            initial={{ opacity: 0, x: -20 }}
            whileInView={{ opacity: 1, x: 0 }}
            viewport={{ once: true }}
            className="bg-neutral-950 p-8 md:p-10"
          >
            <p className="text-[10px] font-mono uppercase tracking-widest text-red-400 mb-6">The Problem</p>
            <h3 className="font-display text-2xl font-bold text-white mb-6">
              AI Is Centralized & Fragile
            </h3>
            <ul className="space-y-4 text-neutral-400">
              {[
                'Trillion-dollar models locked behind corporate APIs',
                'Massive GPU clusters consume as much power as small cities',
                'Single points of failure - if OpenAI goes down, millions are affected',
                'You pay to use it, but never own any of it',
              ].map((item, i) => (
                <li key={i} className="flex items-start gap-3">
                  <span className="text-red-400 mt-0.5 text-sm">&times;</span>
                  <span>{item}</span>
                </li>
              ))}
            </ul>
          </motion.div>

          <motion.div
            initial={{ opacity: 0, x: 20 }}
            whileInView={{ opacity: 1, x: 0 }}
            viewport={{ once: true }}
            className="bg-neutral-950 p-8 md:p-10"
          >
            <p className="text-[10px] font-mono uppercase tracking-widest text-accent mb-6">Our Solution</p>
            <h3 className="font-display text-2xl font-bold text-white mb-6">
              A Global Brain, Owned by All
            </h3>
            <ul className="space-y-4 text-neutral-400">
              {[
                'Run AI on millions of devices worldwide - laptops, servers, phones',
                'No single point of failure - the network is the computer',
                'Contribute compute, earn NEURO tokens - real ownership',
                'The model learns continuously from collective training',
              ].map((item, i) => (
                <li key={i} className="flex items-start gap-3">
                  <span className="text-accent mt-0.5 text-sm">&check;</span>
                  <span>{item}</span>
                </li>
              ))}
            </ul>
          </motion.div>
        </div>

        <motion.div
          initial={{ opacity: 0, y: 20 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true }}
          className="mb-12"
        >
          <h3 className="font-display text-2xl md:text-3xl font-bold text-white mb-3">
            Three Pillars of the New AI Economy
          </h3>
          <p className="text-neutral-500 max-w-2xl">
            NeuroShard isn't just distributed computing -- it's a new paradigm where participation creates value.
          </p>
        </motion.div>

        <div className="grid md:grid-cols-3 gap-px bg-neutral-800">
          {[
            {
              icon: <Brain className="w-8 h-8" />,
              title: 'Contribute Intelligence',
              description: 'Run a node. Your device becomes part of a global neural network. Even a laptop can hold layers of the model and process tokens.',
            },
            {
              icon: <Coins className="w-8 h-8" />,
              title: 'Earn NEURO Tokens',
              description: 'Every forward pass, every gradient computed -- you get paid. Proof of Neural Work ensures only real work is rewarded.',
            },
            {
              icon: <Globe className="w-8 h-8" />,
              title: 'Shape the Future',
              description: "This isn't someone else's AI. Stake NEURO to vote on model upgrades. The community decides how the network evolves.",
            },
          ].map((pillar, index) => (
            <motion.div
              key={index}
              initial={{ opacity: 0, y: 20 }}
              whileInView={{ opacity: 1, y: 0 }}
              viewport={{ once: true }}
              transition={{ delay: index * 0.1 }}
              className="bg-neutral-950 p-8 group"
            >
              <div className="text-neutral-600 group-hover:text-accent transition-colors mb-6">
                {pillar.icon}
              </div>
              <h4 className="font-display text-lg font-bold text-white mb-3">{pillar.title}</h4>
              <p className="text-neutral-500 text-sm leading-relaxed">{pillar.description}</p>
            </motion.div>
          ))}
        </div>

        <motion.div
          initial={{ opacity: 0, y: 20 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true }}
          className="mt-16 flex flex-col sm:flex-row items-start gap-4"
        >
          <a
            href="/signup"
            className="group px-8 py-3.5 bg-white text-neutral-950 font-bold text-sm flex items-center gap-2 hover:bg-neutral-200 transition-colors"
          >
            Join the Network
            <ArrowRight className="w-4 h-4 group-hover:translate-x-0.5 transition-transform" />
          </a>
          <span className="text-neutral-600 text-sm self-center">
            Free to join &middot; Earn from day one
          </span>
        </motion.div>
      </div>
    </section>
  );
};
