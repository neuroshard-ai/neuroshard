import { useEffect, useState } from 'react';
import { ArrowRight } from 'lucide-react';
import { get, neuro, short } from './api';

type Sponsor = { remaining_tasks: number; workers_by_stage: number[]; active_operations: number; sponsor: string };
export function SponsorStatus() {
  const [status, setStatus] = useState<Sponsor | null>(null), [error, setError] = useState(false);
  useEffect(() => {
    let alive = true;
    const refresh = () => { void get<Sponsor>('/work/status').then(s => { if (alive) { setStatus(s); setError(false); } }).catch(() => { if (alive) setError(true); }); };
    refresh(); const timer = setInterval(refresh, 10000); return () => { alive = false; clearInterval(timer); };
  }, []);
  return <div className="sponsor-status" aria-live="polite"><strong>{error ? 'Sponsor currently unavailable' : status ? status.active_operations ? 'A task is in progress' : status.remaining_tasks > 0 ? 'Sponsor is waiting for workers' : 'This sponsorship session is complete' : 'Checking work availability…'}</strong>
    {status && !error && <p>{status.remaining_tasks} task attempts remaining · {status.workers_by_stage[0]} recent stage-0 worker(s) · {status.workers_by_stage[1]} recent stage-1 worker(s)</p>}
    <p>{error ? 'Your full node can still follow the ledger. Try again later or connect to another sponsor.' : 'Availability is observed, not reserved for you. Payment requires both stages and finalized acceptance.'}</p></div>;
}

type Validator = { public_key: string; owner: string; power: number; bond: string };
export function Validators() {
  const [values, setValues] = useState<Validator[]>([]), [after, setAfter] = useState<string | null>(null);
  const [total, setTotal] = useState(0), [error, setError] = useState('');
  const load = async (cursor?: string) => {
    try { const v = await get<{validators: Validator[]; total: number; next_after: string | null}>(`/api/validators?limit=100${cursor ? `&after=${cursor}` : ''}`); setValues(v.validators); setTotal(v.total); setAfter(v.next_after); setError(''); }
    catch (e) { setError(e instanceof Error ? e.message : 'Validators unavailable'); }
  };
  useEffect(() => { void load(); }, []);
  return <section className="validators"><div className="ledger-heading"><h2>Active validators{total ? ` · ${total}` : ''}</h2><button className="native-button secondary compact" onClick={() => void load()}>Refresh</button></div>
    <p className="muted">Voting power comes from activated bonds. The launch validators are controlled by one operator across two machines; independent ownership is still needed.</p>
    {error && <div className="native-notice" role="alert">{error}</div>}<div className="block-table"><table><thead><tr><th>CONSENSUS KEY</th><th>OWNER ACCOUNT</th><th>VOTING POWER</th><th>BOND · NEURO</th></tr></thead><tbody>{values.map(v => <tr key={v.public_key}><td className="mono" title={v.public_key}>{short(v.public_key)}</td><td className="mono"><a href={`?account=${v.owner}`} title={v.owner}>{short(v.owner)}</a></td><td>{v.power.toLocaleString()}</td><td>{neuro(v.bond)}</td></tr>)}</tbody></table></div>
    {after && <div className="ledger-pagination"><button onClick={() => void load(after)}>More validators <ArrowRight size={15} /></button></div>}</section>;
}
