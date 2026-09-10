import { useState, useEffect } from 'react';
import { Menu, X, LogOut, Shield, ChevronDown, LayoutDashboard, Download, Hexagon, Zap } from 'lucide-react';
import { motion, AnimatePresence } from 'framer-motion';
import { Link, useNavigate } from 'react-router-dom';
import { useAuth } from '../contexts/AuthContext';
import logo from '../assets/logo_white.png';

export const Header = () => {
  const [isOpen, setIsOpen] = useState(false);
  const [isProfileOpen, setIsProfileOpen] = useState(false);
  const [scrolled, setScrolled] = useState(false);
  const { user, logout, isLoading } = useAuth();
  const navigate = useNavigate();

  useEffect(() => {
    const handleScroll = () => {
      setScrolled(window.scrollY > 50);
    };
    window.addEventListener('scroll', handleScroll);
    return () => window.removeEventListener('scroll', handleScroll);
  }, []);

  useEffect(() => {
    if (isOpen) {
      document.body.style.overflow = 'hidden';
    } else {
      document.body.style.overflow = '';
    }
    return () => { document.body.style.overflow = ''; };
  }, [isOpen]);

  const handleLogout = () => {
    logout();
    navigate('/');
    setIsProfileOpen(false);
  };

  const navLinks = [
    { name: 'Homepage', href: '/' },
    { name: 'Training', href: '/training' },
    { name: 'Ledger', href: '/ledger' },
    { name: 'Governance', href: '/governance' },
    { name: 'Whitepaper', href: '/whitepaper' },
  ];

  return (
    <header
      className={`fixed top-0 w-full z-50 transition-all duration-300 ${
        scrolled || isOpen
          ? 'bg-neutral-950 border-b border-neutral-800'
          : 'bg-transparent'
      }`}
    >
      <div className="max-w-7xl mx-auto px-4 sm:px-6 h-14 sm:h-16 flex justify-between items-center">
        <Link to="/" className="flex items-center gap-2 text-white z-50">
          <img src={logo} alt="NeuroShard" className="h-7 sm:h-8 w-auto" />
          <span className="font-display font-bold text-base sm:text-lg tracking-tight">
            Neuro<span className="text-accent">Shard</span>
          </span>
        </Link>

        <nav className="hidden md:flex items-center gap-1">
          {navLinks.map((link) => (
            <Link
              key={link.name}
              to={link.href}
              className="px-3 py-1.5 text-sm text-neutral-400 hover:text-white transition-colors"
            >
              {link.name}
            </Link>
          ))}

          <div className="w-px h-5 bg-neutral-800 mx-3" />

          <div className="flex items-center gap-3">
            {isLoading ? (
              <div className="h-8 w-20 bg-neutral-800 animate-pulse" />
            ) : user ? (
              <div className="flex items-center gap-3">
                <Link
                  to="/chat"
                  className="bg-accent text-neutral-950 px-4 py-1.5 text-sm font-semibold hover:bg-accent/90 transition-colors flex items-center gap-1.5"
                >
                  <Zap className="w-3.5 h-3.5" />
                  Live Demo
                </Link>

                <div className="relative">
                  <button
                    onClick={() => setIsProfileOpen(!isProfileOpen)}
                    className="flex items-center gap-1.5 text-sm text-neutral-400 hover:text-white transition-colors"
                  >
                    <Hexagon className="w-5 h-5 text-accent" />
                    <span className="hidden lg:inline">Account</span>
                    <ChevronDown className={`w-3.5 h-3.5 transition-transform ${isProfileOpen ? 'rotate-180' : ''}`} />
                  </button>

                  {isProfileOpen && (
                    <div
                      className="fixed inset-0 z-40"
                      onClick={() => setIsProfileOpen(false)}
                    />
                  )}

                  <AnimatePresence>
                    {isProfileOpen && (
                      <motion.div
                        initial={{ opacity: 0, y: 8 }}
                        animate={{ opacity: 1, y: 0 }}
                        exit={{ opacity: 0, y: 8 }}
                        transition={{ duration: 0.15 }}
                        className="absolute right-0 mt-3 w-60 bg-neutral-900 border border-neutral-800 z-50"
                      >
                        <div className="px-4 py-3 border-b border-neutral-800">
                          <p className="text-[10px] text-neutral-500 font-mono uppercase tracking-widest mb-1">Signed in as</p>
                          <p className="text-sm text-white truncate" title={user.email}>{user.email}</p>
                        </div>

                        <div className="py-1">
                          <Link
                            to="/dashboard"
                            className="flex items-center gap-3 px-4 py-2.5 text-sm text-neutral-400 hover:text-white hover:bg-neutral-800/60 transition-colors"
                            onClick={() => setIsProfileOpen(false)}
                          >
                            <LayoutDashboard className="w-4 h-4" />
                            Dashboard
                          </Link>

                          <Link
                            to="/download"
                            className="flex items-center gap-3 px-4 py-2.5 text-sm text-neutral-400 hover:text-white hover:bg-neutral-800/60 transition-colors"
                            onClick={() => setIsProfileOpen(false)}
                          >
                            <Download className="w-4 h-4" />
                            Downloads
                          </Link>

                          {user.is_admin && (
                            <Link
                              to="/admin"
                              className="flex items-center gap-3 px-4 py-2.5 text-sm text-neutral-400 hover:text-white hover:bg-neutral-800/60 transition-colors"
                              onClick={() => setIsProfileOpen(false)}
                            >
                              <Shield className="w-4 h-4" />
                              Admin Console
                            </Link>
                          )}
                        </div>

                        <div className="border-t border-neutral-800">
                          <button
                            onClick={handleLogout}
                            className="w-full flex items-center gap-3 px-4 py-2.5 text-sm text-red-400 hover:text-red-300 hover:bg-red-950/20 transition-colors text-left"
                          >
                            <LogOut className="w-4 h-4" />
                            Sign Out
                          </button>
                        </div>
                      </motion.div>
                    )}
                  </AnimatePresence>
                </div>
              </div>
            ) : (
              <>
                <Link
                  to="/login"
                  className="text-sm text-neutral-400 hover:text-white transition-colors"
                >
                  Log In
                </Link>
                <Link
                  to="/signup"
                  className="bg-white text-neutral-950 px-4 py-1.5 text-sm font-semibold hover:bg-neutral-200 transition-colors"
                >
                  Sign Up
                </Link>
              </>
            )}
          </div>
        </nav>

        <button
          className="md:hidden text-white p-1 z-50"
          onClick={() => setIsOpen(!isOpen)}
          aria-label={isOpen ? 'Close menu' : 'Open menu'}
        >
          {isOpen ? <X size={22} /> : <Menu size={22} />}
        </button>
      </div>

      <AnimatePresence>
        {isOpen && (
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            transition={{ duration: 0.2 }}
            className="md:hidden fixed inset-0 top-0 bg-neutral-950 z-40 flex flex-col"
          >
            <div className="h-14 flex-shrink-0" />

            <nav className="flex-1 flex flex-col px-6 py-6 overflow-y-auto">
              <div className="flex-1">
                {navLinks.map((link, i) => (
                  <motion.div
                    key={link.name}
                    initial={{ opacity: 0, x: -10 }}
                    animate={{ opacity: 1, x: 0 }}
                    transition={{ delay: i * 0.05 }}
                  >
                    <Link
                      to={link.href}
                      className="text-white text-2xl font-display font-bold py-4 block border-b border-neutral-800/50"
                      onClick={() => setIsOpen(false)}
                    >
                      {link.name}
                    </Link>
                  </motion.div>
                ))}
              </div>

              <motion.div
                initial={{ opacity: 0, y: 10 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ delay: 0.25 }}
                className="mt-8 pt-6 border-t border-neutral-800"
              >
                {user ? (
                  <div className="space-y-4">
                    <p className="text-xs font-mono uppercase tracking-widest text-neutral-500">
                      Signed in as
                    </p>
                    <p className="text-white text-sm truncate">{user.email}</p>

                    <Link
                      to="/chat"
                      className="bg-accent text-neutral-950 py-3.5 font-bold text-sm w-full flex items-center justify-center gap-2"
                      onClick={() => setIsOpen(false)}
                    >
                      <Zap className="w-4 h-4" />
                      Live Demo
                    </Link>

                    <div className="grid grid-cols-2 gap-2">
                      <Link
                        to="/dashboard"
                        className="border border-neutral-800 text-neutral-300 py-3 text-sm flex items-center justify-center gap-2"
                        onClick={() => setIsOpen(false)}
                      >
                        <LayoutDashboard className="w-4 h-4" />
                        Dashboard
                      </Link>
                      <Link
                        to="/download"
                        className="border border-neutral-800 text-neutral-300 py-3 text-sm flex items-center justify-center gap-2"
                        onClick={() => setIsOpen(false)}
                      >
                        <Download className="w-4 h-4" />
                        Downloads
                      </Link>
                    </div>

                    {user.is_admin && (
                      <Link
                        to="/admin"
                        className="border border-neutral-800 text-neutral-300 py-3 text-sm flex items-center justify-center gap-2 w-full"
                        onClick={() => setIsOpen(false)}
                      >
                        <Shield className="w-4 h-4" />
                        Admin
                      </Link>
                    )}

                    <button
                      onClick={() => { handleLogout(); setIsOpen(false); }}
                      className="text-red-400 text-sm py-3 flex items-center justify-center gap-2 w-full border border-neutral-800"
                    >
                      <LogOut className="w-4 h-4" />
                      Sign Out
                    </button>
                  </div>
                ) : (
                  <div className="space-y-3">
                    <Link
                      to="/signup"
                      className="bg-white text-neutral-950 py-3.5 font-bold text-sm w-full block text-center"
                      onClick={() => setIsOpen(false)}
                    >
                      Sign Up
                    </Link>
                    <Link
                      to="/login"
                      className="border border-neutral-800 text-neutral-300 py-3.5 text-sm w-full block text-center"
                      onClick={() => setIsOpen(false)}
                    >
                      Log In
                    </Link>
                  </div>
                )}
              </motion.div>
            </nav>
          </motion.div>
        )}
      </AnimatePresence>
    </header>
  );
};
