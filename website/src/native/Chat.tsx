import { FormEvent, useEffect, useRef, useState } from 'react';
import { Link } from 'react-router-dom';
import { ArrowDown, ArrowRight, RefreshCw } from 'lucide-react';
import { asset, get, neuro, short } from './api';
import { Wallet, createWallet, importWallet, sign, requestId, encodedTransaction, validatePublicKey } from './wallet';
import releaseNetwork from '../../../src/neuroshard/client/networks/llm-testnet.json';

type Quote = {chain_id: string; genesis_sha256: string; model_root: string; model_name: string;
  provider: string | null; provider_online: boolean; fee_atoms: string; price_per_max_token_atoms: string;
  max_tokens: number; pending_requests: number; request_lifetime_blocks: number};
type Account = {balance: string; locked: string; nonce: number};
type Result = {id: string; status: string; model_root: string; height?: number; output?: {text: string}; refunded?: string; provider_paid?: string};
const message = (e: unknown) => e instanceof Error ? e.message : 'Request failed';

export default function Chat() {
  const [wallet, setWallet] = useState<Wallet | null>(null), [backedUp, setBackedUp] = useState(false);
  const [quote, setQuote] = useState<Quote | null>(null), [account, setAccount] = useState<Account | null>(null);
  const [prompt, setPrompt] = useState(''), [tokens, setTokens] = useState(32), [consent, setConsent] = useState(false);
  const [error, setError] = useState(''), [busy, setBusy] = useState(false), [result, setResult] = useState<Result | null>(null);
  const [pending, setPending] = useState(() => sessionStorage.getItem('neuroshard-pending-' + releaseNetwork.chain_id) || '');
  const mounted = useRef(true), submitting = useRef(false);
  const refresh = async (key = wallet?.public_key) => {
    const q = await get<Quote>('/api/inference');
    if (q.chain_id !== releaseNetwork.chain_id || q.genesis_sha256 !== releaseNetwork.genesis_sha256) throw new Error('The connected network differs from this release. Reload after checking the published release.');
    setQuote(q);
    if (key) setAccount(await get<Account>('/api/account?public_key=' + key));
    return q;
  };
  useEffect(() => {
    mounted.current = true;
    void refresh().catch(e => setError(message(e)));
    return () => { mounted.current = false; };
    // Refresh again explicitly before every payment; never trust an old quote.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);
  useEffect(() => {
    if (!pending) return;
    let cancelled = false;
    const poll = async () => {
      try {
        const value = await get<Result>('/api/inference/request?id=' + pending);
        if (cancelled) return;
        setResult(value);
        if (value.status === 'completed' || value.status === 'expired') {
          setPending(''); sessionStorage.removeItem('neuroshard-pending-' + releaseNetwork.chain_id);
          setError(''); void refresh().catch(() => {});
        }
      } catch { /* A broadcast can be unconfirmed; keep its ID without resubmitting. */ }
    };
    void poll(); const timer = setInterval(() => void poll(), 4000);
    return () => { cancelled = true; clearInterval(timer); };
  }, [pending, wallet]);
  const selectWallet = async (value: Wallet, backup: boolean) => {
    setWallet(value); setBackedUp(backup); setAccount(null); setError('');
    await refresh(value.public_key);
  };
  const backup = () => {
    if (!wallet) return;
    const url = URL.createObjectURL(new Blob([JSON.stringify(wallet) + '\n'], {type: 'application/json'}));
    const link = document.createElement('a'); link.href = url; link.download = 'neuroshard-key.json'; link.click();
    setTimeout(() => URL.revokeObjectURL(url), 1000); setBackedUp(true);
  };
  const total = quote ? BigInt(quote.fee_atoms) + BigInt(quote.price_per_max_token_atoms) * BigInt(tokens) : 0n;
  const funded = account && BigInt(account.balance) >= total;
  const submit = async (event: FormEvent) => {
    event.preventDefault();
    if (!wallet || !backedUp || !consent || submitting.current || pending || !quote) return;
    submitting.current = true; setBusy(true); setError(''); setResult(null);
    let id = '', submitted = false;
    try {
      const q = await refresh();
      if (!q.provider || !q.provider_online) throw new Error('The provider is currently unavailable. No payment was submitted.');
      validatePublicKey(q.provider);
      const liveTotal = BigInt(q.fee_atoms) + BigInt(q.price_per_max_token_atoms) * BigInt(tokens);
      if (liveTotal !== total || liveTotal > 100000n || tokens > q.max_tokens) throw new Error('The quote changed. Review the price and try again.');
      const a = await get<Account>('/api/account?public_key=' + wallet.public_key);
      const n = await get<{height: number; ready: boolean; chain_id: string; genesis_sha256: string}>('/api/network');
      if (!n.ready || n.chain_id !== q.chain_id || n.genesis_sha256 !== q.genesis_sha256) throw new Error('The native node is not ready for this network.');
      if (BigInt(a.balance) < liveTotal) throw new Error('Insufficient available NEURO. Contribute work or receive a native transfer.');
      if (new TextEncoder().encode(prompt).length > 2048) throw new Error('Keep the prompt within 2,048 UTF-8 bytes.');
      const envelope = await sign(wallet, {kind: 'infer', chain_id: q.chain_id, nonce: a.nonce,
        provider: q.provider, model_root: q.model_root, request: {prompt, max_tokens: tokens},
        price: Number(liveTotal - BigInt(q.fee_atoms)), expires: n.height + Math.min(120, q.request_lifetime_blocks)});
      id = await requestId(envelope);
      sessionStorage.setItem('neuroshard-pending-' + releaseNetwork.chain_id, id);
      setPending(id); submitted = true;
      const response = await fetch(asset('/rpc'), {method: 'POST', headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({jsonrpc: '2.0', id: 1, method: 'broadcast_tx_sync', params: {tx: encodedTransaction(envelope)}}),
        signal: AbortSignal.timeout(95000)});
      const value = await response.json();
      if (!response.ok || value.error) throw new Error(value.error?.message || 'Submission outcome is uncertain');
      if (value.result?.code) {
        submitted = false;
        throw new Error(value.result.log || 'Native node rejected the request');
      }
    } catch (e) {
      if (!submitted && id) {setPending(''); sessionStorage.removeItem('neuroshard-pending-' + releaseNetwork.chain_id);}
      setError(message(e) + (submitted ? ' Keep the request ID below; check its status before trying another payment.' : ''));
    } finally { submitting.current = false; if (mounted.current) setBusy(false); }
  };
  return <section className="native-wrap native-page"><div className="native-eyebrow">INFERENCE · NATIVE NEURO</div>
    <h1>Use the model.<br /><em>Pay for the work.</em></h1><p className="page-intro">Ask the network’s small instruction model. Your device signs a bounded payment; validators replay the response before the provider is paid.</p>
    <div className="chat-grid"><aside className="chat-wallet"><h2>Your native account</h2><p>Keys stay in this tab’s memory. Closing or reloading the page removes the key from the app. Import the same backup used by your worker to spend its rewards.</p>
      {!wallet ? <><label className="native-button secondary" htmlFor="key-import">Import key backup</label><input id="key-import" aria-label="Import key backup" type="file" accept="application/json,.json" onChange={e => {
        const file = e.target.files?.[0]; if (!file) return;
        if (file.size > 2048) {setError('Key file is too large'); return;}
        void file.text().then(importWallet).then(w => selectWallet(w, true)).catch(e => setError(message(e)));
      }} /><button className="native-text-link" onClick={() => void createWallet().then(w => selectWallet(w, false)).catch(e => setError(message(e)))}>Create a new local key <ArrowRight size={15} /></button>
      <p className="muted">Export a worker key with <code>neuroshard wallet export neuroshard-key.json</code>. A new key starts with zero NEURO.</p></> : <>
        <p className="mono public-key" title={wallet.public_key}>{wallet.public_key}</p><strong className="wallet-balance">{account ? neuro(account.balance) : '…'} NEURO</strong><p className="muted">{account ? neuro(account.locked || '0') : '…'} locked in pending requests</p>
        <button className="native-button secondary compact" onClick={backup}><ArrowDown size={16} /> Download key backup</button>
        {!backedUp && <p role="status">Save your key backup before making a payment.</p>}
        <button className="native-text-link" disabled={busy} onClick={() => {setWallet(null); setAccount(null); setBackedUp(false);}}>Remove key from this tab</button></>}
      <hr /><Link className="native-text-link" to="/join">Earn NEURO by contributing <ArrowRight size={15} /></Link><p className="muted">Experimental balances belong to this chain. Keys from older networks do not transfer their balances.</p></aside>
      <div className="chat-panel"><div className="ledger-heading"><h2>{quote?.model_name || 'Network model'}</h2><button className="native-button secondary compact" onClick={() => void refresh().then(() => setError('')).catch(e => setError(message(e)))} aria-label="Refresh inference status"><RefreshCw size={16} /></button></div>
      <p className="muted">{quote ? quote.provider_online ? 'Provider connected · ' + quote.pending_requests + ' pending requests' : 'Provider currently unavailable' : 'Reading provider availability…'}</p>
      <form onSubmit={e => void submit(e)}><label htmlFor="prompt">Your prompt</label><textarea id="prompt" value={prompt} onChange={e => setPrompt(e.target.value)} maxLength={2048} rows={5} required placeholder="Explain how a blockchain records a payment." />
        <label htmlFor="output-limit">Maximum response length</label><select id="output-limit" value={tokens} onChange={e => setTokens(Number(e.target.value))}><option value={16}>16 tokens · brief</option><option value={32}>32 tokens · short</option><option value={64}>64 tokens · longer</option></select>
        <div className="chat-price"><strong>{quote ? neuro(total.toString()) : '—'} NEURO total</strong><span>{quote ? neuro(quote.fee_atoms) : '—'} submission fee included. Fixed price for this token limit, even if the response ends earlier.</span></div>
        <label className="chat-consent"><input type="checkbox" checked={consent} onChange={e => setConsent(e.target.checked)} />I understand my prompt and response will be public on the ledger.</label>
        <button className="native-button" disabled={!wallet || !backedUp || !funded || !consent || !quote?.provider_online || busy || !!pending} type="submit">{busy ? 'Submitting…' : pending ? 'Waiting for settlement' : 'Sign and pay'} <ArrowRight size={16} /></button>
        {wallet && account && !funded && <p role="status">This account needs more available NEURO. Contribute work or receive a transfer first.</p>}
      </form>{error && <div className="native-notice" role="alert">{error}</div>}
      {pending && <div className="chat-result" role="status"><h3>Request pending</h3><p className="mono public-key">{pending}</p><p>Checking native settlement. If the provider misses the deadline, the inference budget unlocks automatically; the submission fee remains spent.</p><a className="native-text-link" href={asset('/api/inference/request?id=' + pending)}>Inspect request status <ArrowRight size={15} /></a><p className="muted">If it never appears, inspect the ledger and run <code>neuroshard request {pending}</code> before retrying. This tab remembers the ID, never your key.</p><button className="native-text-link" disabled={busy} onClick={() => {
        if (window.confirm('Have you checked this request on the ledger? Clearing this notice does not cancel a submitted payment. Save the request ID before continuing.')) {
          sessionStorage.removeItem('neuroshard-pending-' + releaseNetwork.chain_id); setPending(''); setError('');
        }
      }}>Clear tracking after checking the ledger</button></div>}
      {result?.status === 'completed' && <div className="chat-result"><div className="native-eyebrow">SETTLED IN BLOCK {result.height}</div><p className="model-response">{result.output?.text || '(The model produced an end-of-sequence token.)'}</p><p className="muted">Checkpoint {short(result.model_root)} · provider paid {neuro(result.provider_paid || '0')} NEURO</p><Link className="native-text-link" to={'/ledger?height=' + result.height}>Inspect settlement <ArrowRight size={15} /></Link></div>}
      {result?.status === 'expired' && <div className="chat-result" role="status">The provider did not finish in time. {neuro(result.refunded || '0')} NEURO unlocked; the submission fee was spent.</div>}
      <p className="muted chat-limit">The 135M-parameter base model is frozen; network training updates a 4,608-parameter adapter. Responses can be inaccurate or incomplete. The latest accepted training checkpoint is served only when its fixed validation score improves.</p></div></div></section>;
}
