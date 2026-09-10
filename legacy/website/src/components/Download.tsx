import { motion } from 'framer-motion';
import { ArrowRight, Laptop, Zap, TrendingUp, Users } from 'lucide-react';
import { useNavigate } from 'react-router-dom';

export const Download = () => {
  const navigate = useNavigate();

  return (
    <section className="py-16 sm:py-24 bg-neutral-900 border-t border-neutral-800">
      <div className="max-w-7xl mx-auto px-4 sm:px-6">
        <div className="border border-neutral-800 bg-neutral-950 p-6 sm:p-8 md:p-16">
          <div className="border-b border-neutral-800 pb-10 mb-10">
            <div className="inline-flex items-center gap-2 py-1.5 px-3 border border-neutral-700 text-neutral-400 text-xs font-mono uppercase tracking-widest mb-6">
              <Users className="w-3 h-3 text-accent" />
              Limited Spots &middot; Join the Waitlist
            </div>

            <h2 className="font-display text-3xl md:text-5xl font-bold text-white mb-4 tracking-tight">
              Reserve Your Node
            </h2>
            <p className="text-neutral-400 text-base max-w-2xl">
              Register your hardware and secure your spot in the NeuroShard network. Get your estimated earnings and a unique referral link to boost your priority.
            </p>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-3 gap-px bg-neutral-800 mb-12">
            {[
              { icon: Laptop, title: 'Register Hardware', desc: 'Tell us your GPU, RAM, and internet specs.' },
              { icon: TrendingUp, title: 'See Your Earnings', desc: 'Get estimated daily NEURO based on your setup.' },
              { icon: Zap, title: 'Get Your Neuro Link', desc: 'Share to boost priority & earn referral bonuses.' },
            ].map((item, i) => (
              <div key={i} className="bg-neutral-950 p-6 group">
                <item.icon className="w-6 h-6 text-neutral-600 group-hover:text-accent transition-colors mb-4" />
                <h3 className="text-white font-bold text-sm mb-1">{item.title}</h3>
                <p className="text-neutral-500 text-sm">{item.desc}</p>
              </div>
            ))}
          </div>

          <div className="flex flex-col sm:flex-row items-start gap-4">
            <motion.button
              whileHover={{ scale: 1.02 }}
              whileTap={{ scale: 0.98 }}
              className="group bg-accent text-neutral-950 px-10 py-3.5 font-bold text-sm flex items-center gap-2 hover:bg-accent/90 transition-colors"
              onClick={() => navigate('/join')}
            >
              Join the Waitlist
              <ArrowRight className="w-4 h-4 group-hover:translate-x-0.5 transition-transform" />
            </motion.button>
          </div>

          <p className="mt-8 text-xs text-neutral-600 font-mono">
            Windows, macOS, Linux & Jetson &middot; CPU, NVIDIA GPU, Apple Silicon, or ARM64
          </p>
        </div>
      </div>
    </section>
  );
};
