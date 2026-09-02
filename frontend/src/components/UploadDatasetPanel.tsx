import { useState, type ChangeEvent } from 'react';
import { API } from '../lib/api';
import type { EventItem, Summary } from '../types';

type Props = { onComplete: (summary: Summary, events: EventItem[]) => void };

const numericEventFields = [
  'amount', 'engagement_score', 'historical_failure_count', 'days_since_last_failure',
  'historical_payment_count', 'day_of_month', 'tenure_days', 'prob_none', 'prob_retry',
  'prob_whatsapp', 'inc_prob_retry', 'inc_prob_whatsapp', 'eniv_none', 'eniv_retry',
  'eniv_whatsapp', 'uncertainty_retry', 'uncertainty_whatsapp',
] as const;

function normalizeEvent(item: { event: EventItem; decision: Record<string, unknown> }): EventItem {
  const merged = { ...item.event, ...item.decision, source: 'dynamic' as const } as Record<string, unknown>;
  for (const field of numericEventFields) {
    if (field in merged) {
      const numeric = Number(merged[field]);
      merged[field] = Number.isFinite(numeric) ? numeric : undefined;
    }
  }
  return merged as EventItem;
}

export function UploadDatasetPanel({ onComplete }: Props) {
  const [file, setFile] = useState<File | null>(null);
  const [message, setMessage] = useState('');
  const [error, setError] = useState('');
  const [busy, setBusy] = useState(false);
  const choose = (event: ChangeEvent<HTMLInputElement>) => setFile(event.target.files?.[0] ?? null);
  const upload = async () => {
    if (!file) return setError('Choose a CSV file first.');
    setBusy(true); setError(''); setMessage('');
    try {
      const body = new FormData(); body.append('file', file);
      const response = await fetch(`${API}/api/upload-dataset`, { method: 'POST', body });
      const data = await response.json();
      if (!response.ok) throw new Error(data?.detail || 'Dataset upload failed.');
      const events = (data.results ?? []).map((item: { event: EventItem; decision: Record<string, unknown> }) => normalizeEvent(item));
      onComplete({ total_failed_payments: data.processed_rows, gross_recovered: data.summary.gross_recovery, incremental_recovered: data.summary.incremental_recovery, intervention_cost: data.summary.intervention_cost, policy_value: data.summary.eniv, action_distribution: { retry: data.summary.retry, whatsapp: data.summary.whatsapp, none: data.summary.none, abstained: data.summary.abstained }, dataset_id: `${data.dataset_id} · ${file.name}`, source: 'uploaded' }, events);
      setMessage(`Uploaded ${data.processed_rows} of ${data.total_rows} rows. Dataset: ${data.dataset_id}`);
    } catch (cause) { setError(cause instanceof Error ? cause.message : 'Dataset upload failed.'); } finally { setBusy(false); }
  };
  return <section className="panel upload-panel"><div className="panel-head"><span className="panel-kicker">Phase 10</span><h2>Upload Dataset</h2><p>Upload failed-payment CSV rows for real causal decisions. Required model fields are validated row by row.</p></div><label className="upload-drop">Choose CSV file<input type="file" accept=".csv,text/csv" onChange={choose} /></label>{file ? <p className="upload-meta">{file.name} · selected</p> : null}<button className="primary-btn" onClick={() => void upload()} disabled={busy || !file}>{busy ? 'Processing…' : 'Process Dataset'}</button>{message ? <p className="upload-success">{message}</p> : null}{error ? <p className="auth-error">{error}</p> : null}<p className="upload-hint">Columns: event_id, customer_id, amount, plan_tier, payment_method, failure_context, decline_signal_bucket, engagement_score, historical_failure_count, whatsapp_opted_in, email_verified, days_since_last_failure, historical_payment_count, day_of_month, tenure_days.</p></section>;
}
