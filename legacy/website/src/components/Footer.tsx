import { Link } from 'react-router-dom';
import { Github, Twitter, FileText, ExternalLink, MessageCircle } from 'lucide-react';
import logo from '../assets/logo_white.png';

export const Footer = () => {
  return (
    <footer className="bg-neutral-950 border-t border-neutral-800">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 py-12 sm:py-16">
        <div className="grid grid-cols-2 md:grid-cols-4 gap-8 md:gap-8">
          <div className="col-span-2">
            <div className="flex items-center gap-2 text-lg font-display font-bold text-white mb-4">
              <img src={logo} alt="NeuroShard" className="h-6 w-auto" />
              <span>Neuro<span className="text-accent">Shard</span></span>
            </div>
            <p className="text-neutral-500 text-sm leading-relaxed max-w-sm mb-6">
              Decentralized AI training network. Contributing compute,
              earning NEURO, building collective intelligence.
            </p>
            <div className="flex items-center gap-2">
              {[
                { href: 'https://github.com/neuroshard-ai/neuroshard', icon: Github, label: 'GitHub' },
                { href: 'https://x.com/shardneuro', icon: Twitter, label: 'X' },
                { href: 'https://discord.gg/4R49xpj7vn', icon: MessageCircle, label: 'Discord' },
                { href: 'https://docs.neuroshard.com', icon: FileText, label: 'Docs' },
              ].map((link) => (
                <a
                  key={link.label}
                  href={link.href}
                  target="_blank"
                  rel="noopener noreferrer"
                  className="p-2 border border-neutral-800 text-neutral-500 hover:text-white hover:border-neutral-600 transition-all"
                  aria-label={link.label}
                >
                  <link.icon className="w-4 h-4" />
                </a>
              ))}
            </div>
          </div>

          <div>
            <h4 className="text-[10px] font-mono uppercase tracking-widest text-neutral-500 mb-4 sm:mb-5">Resources</h4>
            <ul className="space-y-2.5">
              <li>
                <Link to="/whitepaper" className="text-neutral-400 hover:text-white text-sm transition-colors">
                  Whitepaper
                </Link>
              </li>
              <li>
                <a href="https://docs.neuroshard.com" target="_blank" rel="noopener noreferrer" className="text-neutral-400 hover:text-white text-sm transition-colors inline-flex items-center gap-1">
                  Documentation
                  <ExternalLink className="w-3 h-3 opacity-50" />
                </a>
              </li>
              <li>
                <Link to="/download" className="text-neutral-400 hover:text-white text-sm transition-colors">
                  Download Node
                </Link>
              </li>
              <li>
                <Link to="/ledger" className="text-neutral-400 hover:text-white text-sm transition-colors">
                  Ledger Explorer
                </Link>
              </li>
            </ul>
          </div>

          <div>
            <h4 className="text-[10px] font-mono uppercase tracking-widest text-neutral-500 mb-4 sm:mb-5">Legal</h4>
            <ul className="space-y-2.5">
              <li>
                <Link to="/legal" className="text-neutral-400 hover:text-white text-sm transition-colors">
                  Terms of Service
                </Link>
              </li>
              <li>
                <Link to="/legal" className="text-neutral-400 hover:text-white text-sm transition-colors">
                  Privacy Policy
                </Link>
              </li>
              <li>
                <Link to="/legal" className="text-neutral-400 hover:text-white text-sm transition-colors">
                  Token Disclaimer
                </Link>
              </li>
              <li>
                <Link to="/legal" className="text-neutral-400 hover:text-white text-sm transition-colors">
                  Risk Disclosure
                </Link>
              </li>
            </ul>
          </div>
        </div>
      </div>

      <div className="border-t border-neutral-800">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 py-4 sm:py-5">
          <div className="flex flex-col sm:flex-row justify-between items-center gap-3">
            <p className="text-neutral-600 text-xs font-mono">
              &copy; {new Date().getFullYear()} Nexaroa &middot; Apache 2.0
            </p>
            <p className="text-neutral-700 text-xs text-center sm:text-right max-w-xl">
              NEURO is a utility token, not an investment. AI outputs may be inaccurate.{' '}
              <Link to="/legal" className="text-neutral-500 hover:text-white">
                Full disclaimer &rarr;
              </Link>
            </p>
          </div>
        </div>
      </div>
    </footer>
  );
};
