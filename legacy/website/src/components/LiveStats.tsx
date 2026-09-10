import { useState, useEffect } from 'react';
import { motion } from 'framer-motion';
import { Cpu, Globe, Zap, Users } from 'lucide-react';
import axios from 'axios';
import { API_URL } from '../config/api';

const StatItem = ({ icon: Icon, label, value, delay }: any) => (
  <motion.div
    initial={{ opacity: 0, y: 10 }}
    animate={{ opacity: 1, y: 0 }}
    transition={{ delay, duration: 0.4 }}
    className="border border-neutral-800 bg-neutral-950/50 backdrop-blur-sm px-4 sm:px-5 py-3 sm:py-4 flex-1 min-w-[120px]"
  >
    <div className="flex items-center gap-1.5 mb-1">
      <Icon className="w-3 h-3 sm:w-3.5 sm:h-3.5 text-neutral-500" />
      <span className="text-[9px] sm:text-[10px] font-mono uppercase tracking-widest text-neutral-500">{label}</span>
    </div>
    <div className="text-lg sm:text-xl font-display font-bold text-white">
      {value}
    </div>
  </motion.div>
);

export const LiveStats = () => {
  const [stats, setStats] = useState({
    nodes: '0',
    quorums: '0',
    params: '142B',
    tps: '0',
    latency: '--'
  });

  useEffect(() => {
    const fetchStats = async () => {
      try {
        const [statsResponse, networkResponse] = await Promise.all([
          axios.get(`${API_URL}/api/stats`).catch(() => ({ data: {} })),
          axios.get(`${API_URL}/api/network/status`).catch(() => ({ data: {} }))
        ]);

        const statsData = statsResponse.data;
        const networkData = networkResponse.data;

        setStats({
          nodes: statsData.active_nodes?.toLocaleString() || networkData.nodes?.active?.toLocaleString() || '0',
          quorums: networkData.quorums?.active?.toString() || '0',
          params: statsData.model_size || '142B',
          tps: statsData.total_tps?.toLocaleString() || '0',
          latency: statsData.avg_latency || '--'
        });
      } catch (error) {
        console.error('Failed to fetch live stats:', error);
      }
    };

    fetchStats();
    const interval = setInterval(fetchStats, 5000);
    return () => clearInterval(interval);
  }, []);

  return (
    <div className="grid grid-cols-2 sm:flex sm:flex-wrap gap-2 sm:gap-3 mt-10 sm:mt-16">
      <StatItem icon={Globe} label="Nodes" value={stats.nodes} delay={0.9} />
      <StatItem icon={Users} label="Quorums" value={stats.quorums} delay={0.95} />
      <StatItem icon={Cpu} label="Model" value={stats.params} delay={1.0} />
      <StatItem icon={Zap} label="TPS" value={stats.tps} delay={1.05} />
    </div>
  );
};
