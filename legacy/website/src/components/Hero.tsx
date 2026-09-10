import { motion } from 'framer-motion';
import { FileText, Activity, ArrowRight } from 'lucide-react';
import { useNavigate } from 'react-router-dom';
import { NetworkBackground } from './NetworkBackground';
import { LiveStats } from './LiveStats';

export const Hero = () => {
  const navigate = useNavigate();

  return (
    <section className="relative min-h-screen flex items-center justify-center pt-14 sm:pt-16 overflow-hidden bg-neutral-950">
      <NetworkBackground />
      <div className="absolute inset-0 bg-neutral-950/70" />

      <div className="max-w-7xl mx-auto px-4 sm:px-6 relative z-10 w-full py-12 sm:py-0">
        <div className="max-w-4xl">
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.6 }}
          >
            <div className="inline-flex items-center gap-2 py-1.5 px-3 border border-neutral-700 text-neutral-400 text-[10px] sm:text-xs font-mono uppercase tracking-widest mb-6 sm:mb-8">
              <Activity className="w-3 h-3 text-accent" />
              Limited Spots Available
            </div>

            <h1 className="font-display text-4xl sm:text-5xl md:text-7xl lg:text-8xl font-bold text-white tracking-tightest leading-[0.95] mb-6 sm:mb-8">
              The Global Brain.{' '}
              <span className="text-accent">Powered by Everyone.</span>
            </h1>

            <p className="text-base sm:text-lg md:text-xl text-neutral-400 max-w-2xl mb-8 sm:mb-12 leading-relaxed">
              Join the distributed AI revolution. Register your hardware, reserve your mining node, and earn NEURO tokens.
            </p>

            <div className="flex flex-col sm:flex-row items-stretch sm:items-start gap-3">
              <button
                onClick={() => navigate('/join')}
                className="group px-6 sm:px-8 py-3.5 bg-accent text-neutral-950 font-bold text-sm flex items-center justify-center gap-2 hover:bg-accent/90 transition-colors"
              >
                Reserve Your Node
                <ArrowRight className="w-4 h-4 group-hover:translate-x-0.5 transition-transform" />
              </button>
              <button
                onClick={() => navigate('/whitepaper')}
                className="px-6 sm:px-8 py-3.5 border border-neutral-700 text-white font-medium text-sm flex items-center justify-center gap-2 hover:bg-neutral-900 transition-colors"
              >
                <FileText className="w-4 h-4" />
                Read Whitepaper
              </button>
            </div>
          </motion.div>

          <LiveStats />

          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            transition={{ delay: 0.8, duration: 0.6 }}
            className="mt-12 sm:mt-16 flex flex-col sm:flex-row gap-4 sm:gap-12 text-neutral-500 text-[10px] sm:text-xs uppercase tracking-widest font-mono"
          >
            <div className="flex items-center gap-2">
              <span className="w-1.5 h-1.5 bg-accent" />
              100% Decentralized
            </div>
            <div className="flex items-center gap-2">
              <span className="w-1.5 h-1.5 bg-accent" />
              Limitless Scale
            </div>
            <div className="flex items-center gap-2">
              <span className="w-1.5 h-1.5 bg-accent" />
              Pure Intelligence
            </div>
          </motion.div>
        </div>
      </div>
    </section>
  );
};
