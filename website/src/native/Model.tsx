import { useCallback, useEffect, useState } from 'react';
import { ArrowDown, ArrowRight, RefreshCw } from 'lucide-react';
import { asset, get, short } from './api';

type ModelState = { chain_id: string; height: number; round: number; model_root: string;
  parameter_count: number; base_parameter_count?: number; trainable_parameter_count?: number; model_name?: string; serving_root?: string; validation_loss?: number; serving_validation_loss?: number; last_training_loss: number | null; execution: {
    dataset_sha256: string; optimizer: string; learning_rate: number; batch_size: number; sequence_length: number;
    numerics: string; model: { num_layers: number; hidden_dim: number; vocab_size: number } } };
type Step = { round: number; height: number; loss: number; time: string; model_root: string; transaction_hash: string };
type History = { records: Step[]; indexed_height: number; chain_height: number; indexed_rounds: number;
  next_before: number | null; error: string | null };

export default function Model() {
  const [model, setModel] = useState<ModelState | null>(null), [history, setHistory] = useState<History | null>(null);
  const [error, setError] = useState(''), [selected, setSelected] = useState<Step | null>(null);
  const [before, setBefore] = useState<number | undefined>();
  const refresh = useCallback(async () => {
    try {
      const [m, h] = await Promise.all([get<ModelState>('/api/model'), get<History>(`/api/training?limit=50${before ? `&before=${before}` : ''}`)]);
      setModel(m); setHistory(h); setError('');
    } catch (e) { setError(e instanceof Error ? e.message : 'Model data unavailable'); }
  }, [before]);
  useEffect(() => { void refresh(); const id = setInterval(() => void refresh(), 10000); return () => clearInterval(id); }, [refresh]);
  const points = [...(history?.records || [])].reverse();
  const low = Math.floor(Math.min(...points.map(p => p.loss), 10) * 2) / 2;
  const high = Math.max(low + 1, Math.ceil(Math.max(...points.map(p => p.loss), 0) * 2) / 2);
  const x = (i: number) => 48 + (i / Math.max(points.length - 1, 1)) * 820;
  const y = (loss: number) => 185 - ((loss - low) / (high - low)) * 150;
  return <section className="native-wrap native-page"><div className="native-eyebrow">MODEL & TRAINING</div>
    <h1>See what<br /><em>the work produces.</em></h1><p className="page-intro">A shared checkpoint advances when validators accept a complete training update. Inspect the receipts, training adapter, and checkpoint selected for inference.</p>
    {error && <div className="native-notice" role="alert">{error} <button onClick={() => void refresh()}>Retry</button></div>}
    {model ? <><div className="native-stats"><div><span>PARAMETERS</span><strong>{model.parameter_count.toLocaleString()}</strong></div><div><span>ACCEPTED UPDATES</span><strong>{model.round.toLocaleString()}</strong></div><div><span>LAST TRAINING LOSS</span><strong>{model.last_training_loss?.toFixed(4) ?? '—'}</strong></div><div><span>TRAINABLE PARAMETERS</span><strong>{(model.trainable_parameter_count ?? model.parameter_count).toLocaleString()}</strong></div></div>
      <div className="ledger-heading"><h2>Accepted training history</h2><button className="native-button secondary compact" onClick={() => { setBefore(undefined); void refresh(); }}><RefreshCw size={15} /> Latest</button></div>
      <p className="muted">Minibatch cross-entropy before each update. Batches change between steps; this curve does not measure held-out quality.</p>
      {history?.error && <div className="native-notice">{history.error}. The records below are the last successfully indexed data.</div>}
      {points.length ? <div className="training-chart"><svg viewBox="0 0 900 225" role="img" aria-label="Training loss by accepted update; exact values are listed below">
        {[low, (low + high) / 2, high].map(v => <g key={v}><line x1="45" x2="875" y1={y(v)} y2={y(v)} stroke="#30362a" /><text x="0" y={y(v) + 4}>{v.toFixed(1)}</text></g>)}
        <polyline points={points.map((p, i) => `${x(i)},${y(p.loss)}`).join(' ')} fill="none" stroke="#c8ff00" strokeWidth="2" />
        {points.map((p, i) => <circle key={p.height} cx={x(i)} cy={y(p.loss)} r={selected?.height === p.height ? 6 : 4} fill="#c8ff00" onMouseEnter={() => setSelected(p)}><title>Step {p.round}: {p.loss.toFixed(6)}</title></circle>)}
        <text x="48" y="218">Step {points[0].round}</text><text x="815" y="218">Step {points[points.length - 1].round}</text>
      </svg><p>{selected ? `Step ${selected.round} · loss ${selected.loss.toFixed(6)} · block ${selected.height.toLocaleString()}` : 'Select a step below to inspect its exact value.'}</p></div> : <div className="native-notice">{history && history.indexed_height < history.chain_height ? 'Reconstructing training history from native blocks…' : 'No accepted training updates in this page yet.'}</div>}
      <div className="block-table"><table><thead><tr><th>UPDATE</th><th>TRAINING LOSS</th><th>BLOCK</th><th>MODEL ROOT</th></tr></thead><tbody>{history?.records.map(p => <tr key={p.height}><td><button className="block-height" onClick={() => setSelected(p)}>Step {p.round}</button></td><td>{p.loss.toFixed(6)}</td><td><a href={asset(`/ledger?height=${p.height}`)}>#{p.height.toLocaleString()}</a></td><td className="mono" title={p.model_root}>{short(p.model_root)}</td></tr>)}</tbody></table></div>
      <div className="ledger-pagination"><span>Indexed through #{history?.indexed_height.toLocaleString() ?? '—'} of #{history?.chain_height.toLocaleString() ?? '—'} · {history?.indexed_rounds ?? 0} accepted updates</span><button disabled={!history?.next_before} onClick={() => { setBefore(history?.next_before ?? undefined); setSelected(null); }}>Earlier updates <ArrowRight size={15} /></button></div>
      <section className="model-grid"><article><div className="native-eyebrow">CURRENT CHECKPOINT</div><h2>Public model state</h2><p className="muted">{model.model_name || "Reference language model"}. {model.execution.model.hidden_dim} hidden dimensions, two CPU stages. Network training updates the adapter; the pretrained backbone stays frozen.</p><dl><dt>Chain</dt><dd>{model.chain_id}</dd><dt>Observed model root</dt><dd>{model.model_root}</dd><dt>Serving checkpoint</dt><dd>{model.serving_root || model.model_root}</dd><dt>Serving validation loss</dt><dd>{model.serving_validation_loss?.toFixed(6) ?? "Unavailable for this profile"}</dd><dt>Optimizer</dt><dd>{model.execution.optimizer} · learning rate {model.execution.learning_rate}</dd><dt>Batch</dt><dd>{model.execution.batch_size} × {model.execution.sequence_length} tokens</dd></dl><a className="native-button" href={asset('/api/model/checkpoint.json')} download="neuroshard-checkpoint.json">Download latest checkpoint <ArrowDown size={16} /></a><p className="muted">Adapter tensors with chain identity and model root. The 135M-parameter base model is downloaded separately by the client and checked against genesis. This download is the latest training state; the serving checkpoint can be older.</p></article>
      <article><div className="native-eyebrow">DATA & EVALUATION</div><h2>A reproducible starting point</h2><p>The LLM profile uses a pinned Apache-2.0 snapshot of Smol-SmolTalk, rendered with the pinned model tokenizer. The initial execution dataset contains 128 training and four held-out 64-token sequences. Immutable snapshot and execution-data hashes are bound by genesis.</p><dl><dt>Corpus SHA-256</dt><dd>{model.execution.dataset_sha256}</dd><dt>Runtime</dt><dd>{model.execution.numerics}</dd></dl><a className="native-text-link" href="https://docs.neuroshard.com/generated/MODEL_CARD">Read the model card <ArrowRight size={16} /></a><p className="muted">A checkpoint is promoted for inference only when loss improves on the four fixed public validation sequences. This small, public gate can be overfit; it does not establish broad model quality. Broader evaluation and economical verification remain research work.</p></article></section>
    </> : !error && <p className="empty-state">Reading model state…</p>}</section>;
}
