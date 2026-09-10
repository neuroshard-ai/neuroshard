import { useState, useEffect } from 'react';
import { useAuth } from '../contexts/AuthContext';
import { useNavigate } from 'react-router-dom';
import { 
 RefreshCw, Server, Users, Shield, ShieldOff, 
 UserCheck, UserX, Crown, Clock,
 ChevronDown, ChevronUp, Mail, Key,
 ListChecks, Cpu, TrendingUp, Link2, Check, X, Trash2
} from 'lucide-react';

const API_BASE = import.meta.env.VITE_API_URL || '';

interface User {
 id: number;
 email: string;
 is_active: boolean;
 is_admin: boolean;
 node_id: string | null;
 wallet_id: string | null;
 created_at: string | null;
 last_login: string | null;
 // Rate limiting
 rate_limit_tier: string;
 is_rate_limited: boolean;
 // Chat stats
 chat_count: number;
 total_tokens_used: number;
 last_chat_at: string | null;
}

interface Peer {
 url: string;
 shard_range: string;
 last_seen: number;
 tps: number;
 latency: number;
}

interface AdminStats {
 total_users: number;
 active_users: number;
 admin_users: number;
 users_with_wallets: number;
}

interface WaitlistEntry {
 id: number;
 email: string;
 gpu_model: string | null;
 gpu_vram: number | null;
 ram_gb: number;
 internet_speed: number | null;
 operating_system: string | null;
 estimated_daily_neuro: number;
 hardware_tier: string;
 hardware_score: number;
 referral_code: string;
 referred_by: string | null;
 referral_count: number;
 referral_bonus_percent: number;
 status: string;
 position: number | null;
 priority_score: number;
 created_at: string;
 approved_at: string | null;
 converted_at: string | null;
 admin_notes: string | null;
 confirmation_email_sent: boolean;
 approval_email_sent: boolean;
}

interface WaitlistStats {
 total_entries: number;
 pending_entries: number;
 approved_entries: number;
 rejected_entries: number;
 converted_entries: number;
 total_referrals: number;
 avg_hardware_score: number;
 tier_distribution: Record<string, number>;
}

export const AdminDashboard = () => {
 const { user, isAuthenticated } = useAuth();
 const navigate = useNavigate();
 
 const [isAdmin, setIsAdmin] = useState<boolean | null>(null);
 const [loading, setLoading] = useState(true);
 const [error, setError] = useState('');
 
 // Data states
 const [users, setUsers] = useState<User[]>([]);
 const [peers, setPeers] = useState<Peer[]>([]);
 const [stats, setStats] = useState<AdminStats | null>(null);
 const [waitlistEntries, setWaitlistEntries] = useState<WaitlistEntry[]>([]);
 const [waitlistStats, setWaitlistStats] = useState<WaitlistStats | null>(null);
 const [waitlistFilter, setWaitlistFilter] = useState<string>('pending');
 
 // UI states
 const [activeTab, setActiveTab] = useState<'users' | 'nodes' | 'waitlist'>('waitlist');
 const [expandedUser, setExpandedUser] = useState<number | null>(null);
 const [expandedWaitlist, setExpandedWaitlist] = useState<number | null>(null);
 const [processingIds, setProcessingIds] = useState<Set<number>>(new Set());
 const [deleteConfirm, setDeleteConfirm] = useState<{id: number, email: string} | null>(null);
 const [deleting, setDeleting] = useState(false);

 // Check if user is admin
 useEffect(() => {
 const checkAdmin = async () => {
 if (!isAuthenticated) {
 navigate('/login');
 return;
 }
 
 try {
 const token = localStorage.getItem('token');
 const response = await fetch(`${API_BASE}/api/users/me/is_admin`, {
 headers: { 'Authorization': `Bearer ${token}` }
 });
 
 if (response.ok) {
 const data = await response.json();
 setIsAdmin(data.is_admin);
 if (!data.is_admin) {
 setError('You do not have admin access');
 }
 } else {
 setIsAdmin(false);
 setError('Failed to verify admin status');
 }
 } catch (err) {
 setIsAdmin(false);
 setError('Failed to verify admin status');
 } finally {
 setLoading(false);
 }
 };
 
 checkAdmin();
 }, [isAuthenticated, navigate]);

 // Fetch data when admin is confirmed
 useEffect(() => {
 if (isAdmin) {
 fetchUsers();
 fetchStats();
 fetchPeers();
 fetchWaitlistEntries();
 fetchWaitlistStats();
 }
 }, [isAdmin]);
 
 // Refetch waitlist when filter changes
 useEffect(() => {
 if (isAdmin) {
 fetchWaitlistEntries();
 }
 }, [waitlistFilter]);

 const getAuthHeaders = () => ({
 'Authorization': `Bearer ${localStorage.getItem('token')}`,
 'Content-Type': 'application/json'
 });

 const fetchUsers = async () => {
 try {
 const response = await fetch(`${API_BASE}/api/admin/users`, {
 headers: getAuthHeaders()
 });
 if (response.ok) {
 const data = await response.json();
 setUsers(data);
 }
 } catch (err) {
 console.error('Failed to fetch users:', err);
 }
 };

 const fetchStats = async () => {
 try {
 const response = await fetch(`${API_BASE}/api/admin/stats`, {
 headers: getAuthHeaders()
 });
 if (response.ok) {
 const data = await response.json();
 setStats(data);
 }
 } catch (err) {
 console.error('Failed to fetch stats:', err);
 }
 };

 const fetchPeers = async () => {
 try {
 const response = await fetch(`${API_BASE}/api/admin/peers`, {
 headers: getAuthHeaders()
 });
 if (response.ok) {
 const data = await response.json();
 setPeers(data);
 }
 } catch (err) {
 console.error('Failed to fetch peers:', err);
 }
 };
 
 const fetchWaitlistEntries = async () => {
 try {
 const params = new URLSearchParams();
 if (waitlistFilter && waitlistFilter !== 'all') {
 params.append('status_filter', waitlistFilter);
 }
 const response = await fetch(`${API_BASE}/api/waitlist/admin/list?${params}`, {
 headers: getAuthHeaders()
 });
 if (response.ok) {
 const data = await response.json();
 setWaitlistEntries(data);
 }
 } catch (err) {
 console.error('Failed to fetch waitlist:', err);
 }
 };
 
 const fetchWaitlistStats = async () => {
 try {
 const response = await fetch(`${API_BASE}/api/waitlist/admin/stats`, {
 headers: getAuthHeaders()
 });
 if (response.ok) {
 const data = await response.json();
 setWaitlistStats(data);
 }
 } catch (err) {
 console.error('Failed to fetch waitlist stats:', err);
 }
 };
 
 const approveWaitlist = async (entryId: number) => {
 setProcessingIds(prev => new Set(prev).add(entryId));
 try {
 const response = await fetch(`${API_BASE}/api/waitlist/admin/approve`, {
 method: 'POST',
 headers: getAuthHeaders(),
 body: JSON.stringify({ waitlist_id: entryId, action: 'approve' })
 });
 if (response.ok) {
 fetchWaitlistEntries();
 fetchWaitlistStats();
 } else {
 const data = await response.json();
 alert(data.detail || 'Failed to approve');
 }
 } catch (err) {
 console.error('Failed to approve:', err);
 } finally {
 setProcessingIds(prev => {
 const newSet = new Set(prev);
 newSet.delete(entryId);
 return newSet;
 });
 }
 };
 
 const rejectWaitlist = async (entryId: number) => {
 setProcessingIds(prev => new Set(prev).add(entryId));
 try {
 const response = await fetch(`${API_BASE}/api/waitlist/admin/approve`, {
 method: 'POST',
 headers: getAuthHeaders(),
 body: JSON.stringify({ waitlist_id: entryId, action: 'reject' })
 });
 if (response.ok) {
 fetchWaitlistEntries();
 fetchWaitlistStats();
 } else {
 const data = await response.json();
 alert(data.detail || 'Failed to reject');
 }
 } catch (err) {
 console.error('Failed to reject:', err);
 } finally {
 setProcessingIds(prev => {
 const newSet = new Set(prev);
 newSet.delete(entryId);
 return newSet;
 });
 }
 };
 
 const bulkApprove = async (count: number) => {
 if (!confirm(`Approve top ${count} entries by priority score?`)) return;
 try {
 const response = await fetch(`${API_BASE}/api/waitlist/admin/bulk-approve?count=${count}`, {
 method: 'POST',
 headers: getAuthHeaders()
 });
 if (response.ok) {
 const data = await response.json();
 alert(data.message);
 fetchWaitlistEntries();
 fetchWaitlistStats();
 } else {
 const data = await response.json();
 alert(data.detail || 'Failed to bulk approve');
 }
 } catch (err) {
 console.error('Failed to bulk approve:', err);
 }
 };

 const toggleAdmin = async (userId: number) => {
 try {
 const response = await fetch(`${API_BASE}/api/admin/users/${userId}/toggle-admin`, {
 method: 'PATCH',
 headers: getAuthHeaders()
 });
 if (response.ok) {
 fetchUsers();
 fetchStats();
 } else {
 const data = await response.json();
 alert(data.detail || 'Failed to toggle admin status');
 }
 } catch (err) {
 console.error('Failed to toggle admin:', err);
 }
 };

 const toggleActive = async (userId: number) => {
 try {
 const response = await fetch(`${API_BASE}/api/admin/users/${userId}/toggle-active`, {
 method: 'PATCH',
 headers: getAuthHeaders()
 });
 if (response.ok) {
 fetchUsers();
 fetchStats();
 } else {
 const data = await response.json();
 alert(data.detail || 'Failed to toggle active status');
 }
 } catch (err) {
 console.error('Failed to toggle active:', err);
 }
 };

 const changeUserTier = async (userId: number, tier: string) => {
 try {
 const response = await fetch(`${API_BASE}/api/admin/users/${userId}/rate-limit?tier=${tier}`, {
 method: 'PATCH',
 headers: getAuthHeaders()
 });
 if (response.ok) {
 fetchUsers();
 } else {
 const data = await response.json();
 alert(data.detail || 'Failed to change tier');
 }
 } catch (err) {
 console.error('Failed to change tier:', err);
 }
 };

 const deleteUser = async (userId: number) => {
 setDeleting(true);
 try {
 const response = await fetch(`${API_BASE}/api/admin/users/${userId}`, {
 method: 'DELETE',
 headers: getAuthHeaders()
 });
 if (response.ok) {
 setDeleteConfirm(null);
 setExpandedUser(null);
 fetchUsers();
 fetchStats();
 } else {
 const data = await response.json();
 alert(data.detail || 'Failed to delete user');
 }
 } catch (err) {
 console.error('Failed to delete user:', err);
 alert('Failed to delete user');
 } finally {
 setDeleting(false);
 }
 };

 const refreshAll = () => {
 fetchUsers();
 fetchStats();
 fetchPeers();
 fetchWaitlistEntries();
 fetchWaitlistStats();
 };
 
 const getTierColor = (tier: string) => {
 switch (tier) {
 case 'elite': return 'bg-amber-500/10 text-amber-400 border-amber-500/20';
 case 'pro': return 'bg-purple-500/10 text-purple-400 border-purple-500/20';
 case 'standard': return 'bg-blue-500/10 text-blue-400 border-blue-500/20';
 default: return 'bg-neutral-500/10 text-neutral-400 border-neutral-500/20';
 }
 };
 
 const getStatusColor = (status: string) => {
 switch (status) {
 case 'approved': return 'bg-green-500/10 text-green-400 border-green-500/20';
 case 'rejected': return 'bg-red-500/10 text-red-400 border-red-500/20';
 case 'converted': return 'bg-accent/10 text-accent border-accent/20';
 default: return 'bg-yellow-500/10 text-yellow-400 border-yellow-500/20';
 }
 };

 // Loading state
 if (loading) {
 return (
 <div className="min-h-screen bg-neutral-950 flex items-center justify-center">
 <div className="text-center">
 <RefreshCw className="w-8 h-8 text-accent animate-spin mx-auto mb-4" />
 <p className="text-neutral-400">Verifying admin access...</p>
 </div>
 </div>
 );
 }

 // Not admin
 if (!isAdmin) {
 return (
 <div className="min-h-screen bg-neutral-950 flex items-center justify-center px-4">
 <div className="max-w-md w-full bg-neutral-900 p-8 border border-red-500/20 text-center">
 <div className="p-4 bg-red-500/10 w-fit mx-auto mb-6">
 <ShieldOff className="w-8 h-8 text-red-400" />
 </div>
 <h2 className="text-2xl font-bold text-white mb-2 font-display">Access Denied</h2>
 <p className="text-neutral-400 mb-6">
 {error || 'You do not have permission to access the admin dashboard.'}
 </p>
 <button
 onClick={() => navigate('/dashboard')}
 className="px-6 py-2 bg-neutral-800 hover:bg-neutral-700 text-white transition-colors"
 >
 Go to Dashboard
 </button>
 </div>
 </div>
 );
 }

 return (
    <div className="min-h-screen bg-neutral-950 pt-20 sm:pt-24 pb-12 px-4 sm:px-6">
 <div className="container mx-auto max-w-7xl">
 {/* Header */}
 <div className="flex flex-col md:flex-row justify-between items-start md:items-center mb-8 gap-4">
 <div>
 <div className="flex items-center gap-3 mb-2">
 <div className="p-2 bg-accent/10">
 <Shield className="w-6 h-6 text-accent" />
 </div>
 <h1 className="text-3xl font-bold text-white font-display">Admin Dashboard</h1>
 </div>
 <p className="text-neutral-400">Manage users and monitor network health</p>
 </div>
 <button 
 onClick={refreshAll}
 className="flex items-center gap-2 px-4 py-2 bg-neutral-800 hover:bg-neutral-700 text-white transition-colors border border-neutral-700"
 >
 <RefreshCw className="w-4 h-4" />
 Refresh All
 </button>
 </div>

 {/* Stats Cards */}
 <div className="grid grid-cols-2 md:grid-cols-4 lg:grid-cols-6 gap-4 mb-8">
 {waitlistStats && (
 <>
 <div className="bg-neutral-900 border border-neutral-800 p-4">
 <div className="flex items-center gap-3 mb-2">
 <ListChecks className="w-5 h-5 text-yellow-400" />
 <span className="text-neutral-400 text-sm">Pending</span>
 </div>
 <p className="text-2xl font-bold text-white">{waitlistStats.pending_entries}</p>
 </div>
 <div className="bg-neutral-900 border border-neutral-800 p-4">
 <div className="flex items-center gap-3 mb-2">
 <UserCheck className="w-5 h-5 text-green-400" />
 <span className="text-neutral-400 text-sm">Approved</span>
 </div>
 <p className="text-2xl font-bold text-white">{waitlistStats.approved_entries}</p>
 </div>
 <div className="bg-neutral-900 border border-neutral-800 p-4">
 <div className="flex items-center gap-3 mb-2">
 <TrendingUp className="w-5 h-5 text-accent" />
 <span className="text-neutral-400 text-sm">Converted</span>
 </div>
 <p className="text-2xl font-bold text-white">{waitlistStats.converted_entries}</p>
 </div>
 </>
 )}
 {stats && (
 <>
 <div className="bg-neutral-900 border border-neutral-800 p-4">
 <div className="flex items-center gap-3 mb-2">
 <Users className="w-5 h-5 text-accent" />
 <span className="text-neutral-400 text-sm">Total Users</span>
 </div>
 <p className="text-2xl font-bold text-white">{stats.total_users}</p>
 </div>
 <div className="bg-neutral-900 border border-neutral-800 p-4">
 <div className="flex items-center gap-3 mb-2">
 <Crown className="w-5 h-5 text-yellow-400" />
 <span className="text-neutral-400 text-sm">Admins</span>
 </div>
 <p className="text-2xl font-bold text-white">{stats.admin_users}</p>
 </div>
 <div className="bg-neutral-900 border border-neutral-800 p-4">
 <div className="flex items-center gap-3 mb-2">
 <Server className="w-5 h-5 text-purple-400" />
 <span className="text-neutral-400 text-sm">With Wallets</span>
 </div>
 <p className="text-2xl font-bold text-white">{stats.users_with_wallets}</p>
 </div>
 </>
 )}
 </div>

 {/* Tabs */}
 <div className="flex flex-wrap gap-2 mb-6">
 <button
 onClick={() => setActiveTab('waitlist')}
 className={`px-4 py-2 font-medium transition-colors ${
 activeTab === 'waitlist'
 ? 'bg-accent text-neutral-950'
 : 'bg-neutral-800 text-neutral-400 hover:bg-neutral-700'
 }`}
 >
 <ListChecks className="w-4 h-4 inline mr-2" />
 Waitlist ({waitlistEntries.length})
 </button>
 <button
 onClick={() => setActiveTab('users')}
 className={`px-4 py-2 font-medium transition-colors ${
 activeTab === 'users'
 ? 'bg-accent text-neutral-950'
 : 'bg-neutral-800 text-neutral-400 hover:bg-neutral-700'
 }`}
 >
 <Users className="w-4 h-4 inline mr-2" />
 Users ({users.length})
 </button>
 <button
 onClick={() => setActiveTab('nodes')}
 className={`px-4 py-2 font-medium transition-colors ${
 activeTab === 'nodes'
 ? 'bg-accent text-neutral-950'
 : 'bg-neutral-800 text-neutral-400 hover:bg-neutral-700'
 }`}
 >
 <Server className="w-4 h-4 inline mr-2" />
 Network Nodes ({peers.length})
 </button>
 </div>

 {/* Waitlist Tab */}
 {activeTab === 'waitlist' && (
 <div className="space-y-4">
 {/* Waitlist Controls */}
 <div className="flex flex-wrap items-center justify-between gap-4 bg-neutral-900 border border-neutral-800 p-4">
 <div className="flex flex-wrap gap-2">
 {['all', 'pending', 'approved', 'rejected', 'converted'].map((status) => (
 <button
 key={status}
 onClick={() => setWaitlistFilter(status)}
 className={`px-3 py-1.5 text-sm font-medium transition-colors ${
 waitlistFilter === status
 ? 'bg-accent/10 text-accent border border-accent/30'
 : 'bg-neutral-800 text-neutral-400 hover:bg-neutral-700'
 }`}
 >
 {status.charAt(0).toUpperCase() + status.slice(1)}
 </button>
 ))}
 </div>
 
 <div className="flex gap-2">
 <button
 onClick={() => bulkApprove(10)}
 className="px-4 py-2 bg-green-500/20 text-green-400 hover:bg-green-500/30 text-sm font-medium transition-colors border border-green-500/30 flex items-center gap-2"
 >
 <Check className="w-4 h-4" />
 Approve Top 10
 </button>
 <button
 onClick={() => bulkApprove(50)}
 className="px-4 py-2 bg-green-500/20 text-green-400 hover:bg-green-500/30 text-sm font-medium transition-colors border border-green-500/30 flex items-center gap-2"
 >
 <Check className="w-4 h-4" />
 Approve Top 50
 </button>
 </div>
 </div>
 
 {/* Waitlist Table */}
 <div className="bg-neutral-900 border border-neutral-800 overflow-hidden ">
 <div className="overflow-x-auto">
 <table className="w-full text-left border-collapse">
 <thead>
 <tr className="bg-neutral-800/50 text-neutral-400 text-sm uppercase tracking-wider">
 <th className="p-4 font-medium">Email</th>
 <th className="p-4 font-medium">Hardware</th>
 <th className="p-4 font-medium">Tier</th>
 <th className="p-4 font-medium">Score</th>
 <th className="p-4 font-medium">Est. Daily</th>
 <th className="p-4 font-medium">Referrals</th>
 <th className="p-4 font-medium">Status</th>
 <th className="p-4 font-medium text-right">Actions</th>
 </tr>
 </thead>
 <tbody className="divide-y divide-neutral-800">
 {waitlistEntries.map((entry) => (
 <>
 <tr 
 key={entry.id}
 className="hover:bg-neutral-800/30 transition-colors cursor-pointer"
 onClick={() => setExpandedWaitlist(expandedWaitlist === entry.id ? null : entry.id)}
 >
 <td className="p-4">
 <div className="flex items-center gap-3">
 <div className="p-2 bg-neutral-800">
 <Mail className="w-4 h-4 text-neutral-400" />
 </div>
 <div>
 <span className="text-white font-medium">{entry.email}</span>
 <div className="text-xs text-neutral-500">
 {new Date(entry.created_at).toLocaleDateString()}
 </div>
 </div>
 </div>
 </td>
 <td className="p-4">
 <div className="text-sm">
 <div className="flex items-center gap-2 text-neutral-300">
 <Cpu className="w-3 h-3 text-accent" />
 {entry.gpu_model || 'CPU Only'}
 </div>
 <div className="flex items-center gap-2 text-neutral-500 text-xs mt-1">
 <span>{entry.ram_gb} GB RAM</span>
 {entry.internet_speed && (
 <>
 <span>•</span>
 <span>{entry.internet_speed} Mbps</span>
 </>
 )}
 </div>
 </div>
 </td>
 <td className="p-4">
 <span className={`px-2.5 py-1 text-xs font-bold uppercase border ${getTierColor(entry.hardware_tier)}`}>
 {entry.hardware_tier}
 </span>
 </td>
 <td className="p-4">
 <div className="flex items-center gap-2">
 <div className="w-16 h-2 bg-neutral-800 overflow-hidden">
 <div 
 className="h-full bg-accent"
 style={{ width: `${entry.hardware_score}%` }}
 />
 </div>
 <span className="text-white font-mono text-sm">{entry.hardware_score}</span>
 </div>
 </td>
 <td className="p-4">
 <div className="flex items-center gap-1">
 <TrendingUp className="w-4 h-4 text-green-400" />
 <span className="text-green-400 font-medium">{entry.estimated_daily_neuro.toFixed(2)}</span>
 <span className="text-neutral-500 text-xs">NEURO</span>
 </div>
 </td>
 <td className="p-4">
 <div className="flex items-center gap-2">
 <Link2 className="w-4 h-4 text-amber-400" />
 <span className="text-white">{entry.referral_count}</span>
 {entry.referred_by && (
 <span className="text-xs text-neutral-500">(via {entry.referred_by})</span>
 )}
 </div>
 </td>
 <td className="p-4">
 <span className={`inline-flex items-center gap-1.5 px-2.5 py-1 text-xs font-medium border ${getStatusColor(entry.status)}`}>
 <div className={`w-1.5 h-1.5 ${
 entry.status === 'pending' ? 'bg-yellow-500 animate-pulse' :
 entry.status === 'approved' ? 'bg-green-500' :
 entry.status === 'converted' ? 'bg-accent' :
 'bg-red-500'
 }`} />
 {entry.status}
 </span>
 </td>
 <td className="p-4 text-right">
 {entry.status === 'pending' ? (
 <div className="flex items-center justify-end gap-2">
 <button
 onClick={(e) => {
 e.stopPropagation();
 approveWaitlist(entry.id);
 }}
 disabled={processingIds.has(entry.id)}
 className="p-2 bg-green-500/20 text-green-400 hover:bg-green-500/30 transition-colors disabled:opacity-50"
 title="Approve"
 >
 <Check className="w-4 h-4" />
 </button>
 <button
 onClick={(e) => {
 e.stopPropagation();
 rejectWaitlist(entry.id);
 }}
 disabled={processingIds.has(entry.id)}
 className="p-2 bg-red-500/20 text-red-400 hover:bg-red-500/30 transition-colors disabled:opacity-50"
 title="Reject"
 >
 <X className="w-4 h-4" />
 </button>
 </div>
 ) : (
 <div className="flex items-center justify-end gap-2">
 {expandedWaitlist === entry.id ? (
 <ChevronUp className="w-4 h-4 text-neutral-400" />
 ) : (
 <ChevronDown className="w-4 h-4 text-neutral-400" />
 )}
 </div>
 )}
 </td>
 </tr>
 {expandedWaitlist === entry.id && (
 <tr className="bg-neutral-800/20">
 <td colSpan={8} className="p-4">
 <div className="grid grid-cols-2 md:grid-cols-4 gap-4 text-sm">
 <div>
 <span className="text-neutral-500">Referral Code:</span>
 <span className="ml-2 text-accent font-mono">{entry.referral_code}</span>
 </div>
 <div>
 <span className="text-neutral-500">Priority Score:</span>
 <span className="ml-2 text-white font-bold">{entry.priority_score}</span>
 </div>
 <div>
 <span className="text-neutral-500">OS:</span>
 <span className="ml-2 text-white">{entry.operating_system || 'Unknown'}</span>
 </div>
 <div>
 <span className="text-neutral-500">VRAM:</span>
 <span className="ml-2 text-white">{entry.gpu_vram ? `${entry.gpu_vram} GB` : 'N/A'}</span>
 </div>
 {entry.approved_at && (
 <div>
 <span className="text-neutral-500">Approved:</span>
 <span className="ml-2 text-green-400">{new Date(entry.approved_at).toLocaleString()}</span>
 </div>
 )}
 {entry.converted_at && (
 <div>
 <span className="text-neutral-500">Converted:</span>
 <span className="ml-2 text-accent">{new Date(entry.converted_at).toLocaleString()}</span>
 </div>
 )}
 <div className="col-span-2">
 <span className="text-neutral-500">Email Status:</span>
 <span className="ml-2">
 {entry.confirmation_email_sent && <span className="text-green-400 mr-2">✓ Confirmation</span>}
 {entry.approval_email_sent && <span className="text-green-400">✓ Approval</span>}
 {!entry.confirmation_email_sent && !entry.approval_email_sent && <span className="text-neutral-500">None sent</span>}
 </span>
 </div>
 {entry.admin_notes && (
 <div className="col-span-4">
 <span className="text-neutral-500">Admin Notes:</span>
 <span className="ml-2 text-neutral-300">{entry.admin_notes}</span>
 </div>
 )}
 </div>
 </td>
 </tr>
 )}
 </>
 ))}
 {waitlistEntries.length === 0 && (
 <tr>
 <td colSpan={8} className="p-12 text-center text-neutral-500">
 No waitlist entries found with status "{waitlistFilter}".
 </td>
 </tr>
 )}
 </tbody>
 </table>
 </div>
 </div>
 </div>
 )}

 {/* Users Tab */}
 {activeTab === 'users' && (
 <div className="bg-neutral-900 border border-neutral-800 overflow-hidden ">
 <div className="overflow-x-auto">
 <table className="w-full text-left border-collapse">
 <thead>
 <tr className="bg-neutral-800/50 text-neutral-400 text-sm uppercase tracking-wider">
 <th className="p-4 font-medium">User</th>
 <th className="p-4 font-medium">Status</th>
 <th className="p-4 font-medium">Role</th>
 <th className="p-4 font-medium">Rate Tier</th>
 <th className="p-4 font-medium">Chat Usage</th>
 <th className="p-4 font-medium">Wallet</th>
 <th className="p-4 font-medium">Joined</th>
 <th className="p-4 font-medium text-right">Actions</th>
 </tr>
 </thead>
 <tbody className="divide-y divide-neutral-800">
 {users.map((u) => (
 <>
 <tr 
 key={u.id} 
 className="hover:bg-neutral-800/30 transition-colors cursor-pointer"
 onClick={() => setExpandedUser(expandedUser === u.id ? null : u.id)}
 >
 <td className="p-4">
 <div className="flex items-center gap-3">
 <div className={`p-2 ${u.is_admin ? 'bg-yellow-500/20' : 'bg-neutral-800'}`}>
 {u.is_admin ? (
 <Crown className="w-4 h-4 text-yellow-400" />
 ) : (
 <Mail className="w-4 h-4 text-neutral-400" />
 )}
 </div>
 <div>
 <span className="text-white font-medium">{u.email}</span>
 {u.id === user?.id && (
 <span className="ml-2 text-xs text-accent">(you)</span>
 )}
 </div>
 </div>
 </td>
 <td className="p-4">
 {u.is_active ? (
 <span className="inline-flex items-center gap-1.5 px-2.5 py-0.5 bg-green-500/10 text-green-400 text-xs font-medium border border-green-500/20">
 <div className="w-1.5 h-1.5 bg-green-500" />
 Active
 </span>
 ) : (
 <span className="inline-flex items-center gap-1.5 px-2.5 py-0.5 bg-red-500/10 text-red-400 text-xs font-medium border border-red-500/20">
 <div className="w-1.5 h-1.5 bg-red-500" />
 Inactive
 </span>
 )}
 </td>
 <td className="p-4">
 {u.is_admin ? (
 <span className="px-2 py-1 bg-yellow-500/10 text-yellow-400 text-xs font-bold border border-yellow-500/20">
 ADMIN
 </span>
 ) : (
 <span className="px-2 py-1 bg-neutral-800 text-neutral-400 text-xs font-medium">
 User
 </span>
 )}
 </td>
 <td className="p-4">
 <select
 value={u.rate_limit_tier || 'standard'}
 onChange={(e) => {
 e.stopPropagation();
 changeUserTier(u.id, e.target.value);
 }}
 onClick={(e) => e.stopPropagation()}
 className={`px-2 py-1 text-xs font-bold border cursor-pointer ${
 u.rate_limit_tier === 'premium' 
 ? 'bg-purple-500/10 text-purple-400 border-purple-500/30'
 : u.rate_limit_tier === 'unlimited'
 ? 'bg-yellow-500/10 text-yellow-400 border-yellow-500/30'
 : 'bg-neutral-800 text-neutral-300 border-neutral-700'
 }`}
 >
 <option value="standard">Standard</option>
 <option value="premium">Premium</option>
 <option value="unlimited">Unlimited</option>
 </select>
 {u.is_rate_limited && (
 <span className="ml-2 px-1.5 py-0.5 bg-red-500/20 text-red-400 text-xs">
 BLOCKED
 </span>
 )}
 </td>
 <td className="p-4">
 <div className="text-xs">
 <div className="text-white font-medium">{u.chat_count || 0} chats</div>
 <div className="text-neutral-500">{((u.total_tokens_used || 0) / 1000).toFixed(1)}k tokens</div>
 </div>
 </td>
 <td className="p-4">
 {u.wallet_id ? (
 <div className="flex items-center gap-2">
 <div className="p-1.5 bg-accent/10">
 <Key className="w-4 h-4 text-accent" />
 </div>
 <div>
 <div className="font-mono text-xs text-accent">{u.wallet_id}</div>
 <div className="font-mono text-xs text-neutral-500">{u.node_id?.slice(0, 16)}...</div>
 </div>
 </div>
 ) : (
 <span className="text-neutral-500 text-sm">No wallet</span>
 )}
 </td>
 <td className="p-4 text-neutral-400 text-sm">
 {u.created_at ? new Date(u.created_at).toLocaleDateString() : 'Unknown'}
 </td>
 <td className="p-4 text-right">
 <div className="flex items-center justify-end gap-2">
 {expandedUser === u.id ? (
 <ChevronUp className="w-4 h-4 text-neutral-400" />
 ) : (
 <ChevronDown className="w-4 h-4 text-neutral-400" />
 )}
 </div>
 </td>
 </tr>
 {expandedUser === u.id && (
 <tr className="bg-neutral-800/20">
 <td colSpan={8} className="p-4">
 <div className="flex flex-wrap gap-3">
 <button
 onClick={(e) => {
 e.stopPropagation();
 toggleAdmin(u.id);
 }}
 disabled={u.id === user?.id}
 className={`px-4 py-2 text-sm font-medium transition-colors flex items-center gap-2 ${
 u.id === user?.id
 ? 'bg-neutral-800 text-neutral-500 cursor-not-allowed'
 : u.is_admin
 ? 'bg-yellow-500/10 text-yellow-400 hover:bg-yellow-500/20 border border-yellow-500/20'
 : 'bg-neutral-800 text-neutral-300 hover:bg-neutral-700'
 }`}
 >
 <Crown className="w-4 h-4" />
 {u.is_admin ? 'Remove Admin' : 'Make Admin'}
 </button>
 <button
 onClick={(e) => {
 e.stopPropagation();
 toggleActive(u.id);
 }}
 disabled={u.id === user?.id}
 className={`px-4 py-2 text-sm font-medium transition-colors flex items-center gap-2 ${
 u.id === user?.id
 ? 'bg-neutral-800 text-neutral-500 cursor-not-allowed'
 : u.is_active
 ? 'bg-red-500/10 text-red-400 hover:bg-red-500/20 border border-red-500/20'
 : 'bg-green-500/10 text-green-400 hover:bg-green-500/20 border border-green-500/20'
 }`}
 >
 {u.is_active ? (
 <>
 <UserX className="w-4 h-4" />
 Deactivate
 </>
 ) : (
 <>
 <UserCheck className="w-4 h-4" />
 Activate
 </>
 )}
 </button>
 <button
 onClick={(e) => {
 e.stopPropagation();
 setDeleteConfirm({ id: u.id, email: u.email });
 }}
 disabled={u.id === user?.id}
 className={`px-4 py-2 text-sm font-medium transition-colors flex items-center gap-2 ${
 u.id === user?.id
 ? 'bg-neutral-800 text-neutral-500 cursor-not-allowed'
 : 'bg-red-500/10 text-red-400 hover:bg-red-500/30 border border-red-500/20'
 }`}
 >
 <Trash2 className="w-4 h-4" />
 Delete
 </button>
 {u.last_login && (
 <div className="flex items-center gap-2 px-4 py-2 bg-neutral-800 text-sm text-neutral-400">
 <Clock className="w-4 h-4" />
 Last login: {new Date(u.last_login).toLocaleString()}
 </div>
 )}
 </div>
 </td>
 </tr>
 )}
 </>
 ))}
 {users.length === 0 && (
 <tr>
 <td colSpan={6} className="p-12 text-center text-neutral-500">
 No users found.
 </td>
 </tr>
 )}
 </tbody>
 </table>
 </div>
 </div>
 )}

 {/* Nodes Tab */}
 {activeTab === 'nodes' && (
 <div className="bg-neutral-900 border border-neutral-800 overflow-hidden ">
 <div className="overflow-x-auto">
 <table className="w-full text-left border-collapse">
 <thead>
 <tr className="bg-neutral-800/50 text-neutral-400 text-sm uppercase tracking-wider">
 <th className="p-4 font-medium">Node Address</th>
 <th className="p-4 font-medium">Shards</th>
 <th className="p-4 font-medium">Last Seen</th>
 <th className="p-4 font-medium text-right">TPS</th>
 <th className="p-4 font-medium text-right">Latency</th>
 <th className="p-4 font-medium text-center">Status</th>
 </tr>
 </thead>
 <tbody className="divide-y divide-neutral-800">
 {peers.map((peer) => {
 const isOnline = (Date.now() / 1000) - peer.last_seen < 60;
 return (
 <tr key={peer.url} className="hover:bg-neutral-800/30 transition-colors">
 <td className="p-4">
 <div className="flex items-center gap-3">
 <div className="p-2 bg-neutral-800 border border-neutral-700">
 <Server className="w-4 h-4 text-accent" />
 </div>
 <span className="font-mono text-neutral-300">{peer.url}</span>
 </div>
 </td>
 <td className="p-4">
 <span className="px-2 py-1 bg-purple-500/10 text-purple-400 text-xs font-bold border border-purple-500/20">
 {peer.shard_range}
 </span>
 </td>
 <td className="p-4 text-neutral-400 text-sm font-mono">
 {new Date(peer.last_seen * 1000).toLocaleTimeString()}
 </td>
 <td className="p-4 text-right font-mono text-white">
 {peer.tps.toFixed(1)}
 </td>
 <td className="p-4 text-right font-mono text-white">
 {peer.latency.toFixed(0)}ms
 </td>
 <td className="p-4 text-center">
 {isOnline ? (
 <div className="inline-flex items-center gap-1.5 px-2.5 py-0.5 bg-green-500/10 text-green-400 text-xs font-medium border border-green-500/20">
 <div className="w-1.5 h-1.5 bg-green-500 animate-pulse" />
 Online
 </div>
 ) : (
 <div className="inline-flex items-center gap-1.5 px-2.5 py-0.5 bg-red-500/10 text-red-400 text-xs font-medium border border-red-500/20">
 <div className="w-1.5 h-1.5 bg-red-500" />
 Offline
 </div>
 )}
 </td>
 </tr>
 );
 })}
 {peers.length === 0 && (
 <tr>
 <td colSpan={6} className="p-12 text-center text-neutral-500">
 No active nodes found in the network.
 </td>
 </tr>
 )}
 </tbody>
 </table>
 </div>
 </div>
 )}
 </div>

 {/* Delete Confirmation Modal */}
 {deleteConfirm && (
 <div className="fixed inset-0 bg-black/60 backdrop-blur-sm flex items-center justify-center z-50 p-4">
 <div className="bg-neutral-900 border border-neutral-700 p-6 max-w-md w-full ">
 <div className="flex items-center gap-4 mb-4">
 <div className="p-3 bg-red-500/10">
 <Trash2 className="w-6 h-6 text-red-400" />
 </div>
 <div>
 <h3 className="text-lg font-bold text-white font-display">Delete User</h3>
 <p className="text-neutral-400 text-sm">This action cannot be undone</p>
 </div>
 </div>
 
 <div className="bg-neutral-800 p-4 mb-6">
 <p className="text-neutral-300">
 Are you sure you want to permanently delete <span className="text-white font-semibold">{deleteConfirm.email}</span>?
 </p>
 <p className="text-neutral-500 text-sm mt-2">
 This will remove the user account and all associated data.
 </p>
 </div>
 
 <div className="flex gap-3">
 <button
 onClick={() => setDeleteConfirm(null)}
 disabled={deleting}
 className="flex-1 px-4 py-2.5 bg-neutral-800 hover:bg-neutral-700 text-white font-medium transition-colors disabled:opacity-50"
 >
 Cancel
 </button>
 <button
 onClick={() => deleteUser(deleteConfirm.id)}
 disabled={deleting}
 className="flex-1 px-4 py-2.5 bg-red-500 hover:bg-red-600 text-white font-medium transition-colors flex items-center justify-center gap-2 disabled:opacity-50"
 >
 {deleting ? (
 <>
 <RefreshCw className="w-4 h-4 animate-spin" />
 Deleting...
 </>
 ) : (
 <>
 <Trash2 className="w-4 h-4" />
 Delete User
 </>
 )}
 </button>
 </div>
 </div>
 </div>
 )}
 </div>
 );
};
