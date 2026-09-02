import { useState, type FormEvent } from 'react';
import { API } from '../lib/api';
import { actionLabel, inr, pct } from '../lib/format';
import type { Decision, EventItem } from '../types';

type Props = { onComplete: (event: EventItem, decision: Decision) => void };

const defaults = {
  event_id: `judge_event_${Date.now()}`, customer_id: 'judge_customer_001', payment_id: '',
  amount: '499', plan_tier: 'Pro', payment_method: 'Card', failure_context: 'insufficient_funds',
  decline_signal_bucket: 'soft', engagement_score: '0.55', historical_failure_count: '1',
  whatsapp_opted_in: 'true', email_verified: 'true', days_since_last_failure: '14',
  historical_payment_count: '8', day_of_month: '15', tenure_days: '180',
};

const fields = [
  ['event_id', 'Event ID'], ['customer_id', 'Customer ID'], ['payment_id', 'Payment ID'],
  ['amount', 'Amount'], ['engagement_score', 'Engagement score'], ['historical_failure_count', 'Historical failures'],
  ['days_since_last_failure', 'Days since last failure'], ['historical_payment_count', 'Historical payments'],
  ['day_of_month', 'Day of month'], ['tenure_days', 'Tenure days'],
] as const;

export function TryDataPanel({ onComplete }: Props) {
  const [form, setForm] = useState(defaults);
  const [decision, setDecision] = useState<Decision | null>(null);
  const [error, setError] = useState('');
  const [busy, setBusy] = useState(false);
  const update = (key: keyof typeof defaults, value: string) => setForm((current) => ({ ...current, [key]: value }));

  const submit = async (event: FormEvent) => {
    event.preventDefault();
    setError('');
    if (!form.event_id.trim() || !form.customer_id.trim() || !form.amount || Number(form.amount) <= 0) {
      setError('Enter an event ID, customer ID, and a positive amount.');
      return;
    }
    setBusy(true);
    try {
      const payload = {
        ...form, payment_id: form.payment_id || `pay_${form.event_id}`,
        amount: Number(form.amount), engagement_score: Number(form.engagement_score),
        historical_failure_count: Number(form.historical_failure_count), days_since_last_failure: Number(form.days_since_last_failure),
        historical_payment_count: Number(form.historical_payment_count), day_of_month: Number(form.day_of_month), tenure_days: Number(form.tenure_days),
        whatsapp_opted_in: form.whatsapp_opted_in === 'true', email_verified: form.email_verified === 'true',
      };
      const response = await fetch(`${API}/api/decision`, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(payload) });
      const data = await response.json();
      if (!response.ok) throw new Error(data?.detail || 'The backend rejected this payment event.');
      const result = data as Decision;
      setDecision(result);
      onComplete({ ...payload, event_id: String(payload.event_id), amount: payload.amount, source: 'dynamic', recommended_action: result.recommended_action, is_abstain: result.is_abstain }, result);
    } catch (cause) {
      setError(cause instanceof Error ? cause.message : 'Unable to analyze payment event.');
    } finally {
      setBusy(false);
    }
  };

  return <section className="panel try-data-panel">
    <div className="panel-head"><span className="panel-kicker">Real backend analysis</span><h2>Try Your Own Payment Event</h2><p>Enter a failed payment and CausaPay will score it through the live decision API.</p></div>
    <form className="try-data-form" onSubmit={submit}>
      {fields.map(([key, label]) => <label key={key}>{label}<input value={form[key]} type={['amount', 'engagement_score', 'historical_failure_count', 'days_since_last_failure', 'historical_payment_count', 'day_of_month', 'tenure_days'].includes(key) ? 'number' : 'text'} step={key === 'engagement_score' ? '0.01' : undefined} onChange={(e) => update(key, e.target.value)} /></label>)}
      {(['plan_tier', 'payment_method', 'failure_context', 'decline_signal_bucket'] as const).map((key) => <label key={key}>{key.replaceAll('_', ' ')}<select value={form[key]} onChange={(e) => update(key, e.target.value)}>{({ plan_tier: ['Basic', 'Pro', 'Enterprise'], payment_method: ['Card', 'UPI Autopay', 'Netbanking'], failure_context: ['insufficient_funds', 'technical_timeout'], decline_signal_bucket: ['soft', 'technical'] }[key]).map((value) => <option key={value}>{value}</option>)}</select></label>)}
      {(['whatsapp_opted_in', 'email_verified'] as const).map((key) => <label key={key}>{key.replaceAll('_', ' ')}<select value={form[key]} onChange={(e) => update(key, e.target.value)}><option value="true">Yes</option><option value="false">No</option></select></label>)}
      {error ? <p className="auth-error">{error}</p> : null}
      <button className="primary-btn" type="submit" disabled={busy}>{busy ? 'Analyzing…' : 'Analyze Payment'}</button>
    </form>
    {decision ? <div className="try-result"><span className="panel-kicker">Live decision result</span><h3>{actionLabel(decision.recommended_action)} {decision.is_abstain ? '· Abstained' : ''}</h3><p>Probabilities: none {pct(decision.prob_none)} · retry {pct(decision.prob_retry)} · WhatsApp {pct(decision.prob_whatsapp)}</p><p>Incremental lift: retry {pct(decision.inc_prob_retry)} · WhatsApp {pct(decision.inc_prob_whatsapp)}</p><p>ENIV: retry {inr(decision.eniv_retry, 2)} · WhatsApp {inr(decision.eniv_whatsapp, 2)}</p><p>{decision.explanation}</p></div> : null}
  </section>;
}
