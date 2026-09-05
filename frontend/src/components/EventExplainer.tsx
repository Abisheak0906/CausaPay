import { useState } from 'react';
import { ArrowRight, Check, X } from 'lucide-react';
import clsx from 'clsx';
import { actionLabel, inr, pct } from '../lib/format';

type ExplorerEvent = {
  event_id: string;
  customer_id?: string;
  amount: number;
  plan_tier?: string;
  payment_method?: string;
  failure_context?: string;
  engagement_score?: number;
  historical_failure_count?: number;
  whatsapp_opted_in?: boolean;
  recommended_action: string;
  baseline_action: string;
  is_abstain: boolean;
  prob_none: number;
  prob_retry: number;
  prob_whatsapp: number;
  inc_prob_retry: number;
  inc_prob_whatsapp: number;
  eniv_none: number;
  eniv_retry: number;
  eniv_whatsapp: number;
  uncertainty_retry?: number;
  uncertainty_whatsapp?: number;
  uncertainty_threshold?: number;
  whatsapp_eligible?: boolean;
  explanation?: string;
};

export default function EventExplainer({ events }: { events: ExplorerEvent[] }) {
  const [selectedEventId, setSelectedEventId] = useState<string | null>(events[0]?.event_id ?? null);
  const selectedEvent = events.find((e) => e.event_id === selectedEventId);

  if (!selectedEvent) {
    return (
      <div className="explainer-layout">
        <div className="panel event-pane">
          <div className="panel-head"><h2>Sample events</h2></div>
          <div className="event-list">
            {events.map((e) => (
              <button
                key={e.event_id}
                onClick={() => setSelectedEventId(e.event_id)}
                className={clsx('event-button', selectedEventId === e.event_id && 'selected')}
              >
                <span><strong>{e.customer_id}</strong><small>{e.failure_context} · {e.plan_tier}</small></span>
                <em>{inr(e.amount)}</em>
              </button>
            ))}
          </div>
        </div>
        <div className="panel"><p className="muted">Select an event to view details</p></div>
      </div>
    );
  }

  return (
    <div className="explainer-layout">
      <div className="panel event-pane">
        <div className="panel-head"><h2>Sample events</h2></div>
        <div className="event-list">
          {events.map((e) => (
            <button
              key={e.event_id}
              onClick={() => setSelectedEventId(e.event_id)}
              className={clsx('event-button', selectedEventId === e.event_id && 'selected')}
            >
              <span><strong>{e.customer_id}</strong><small>{e.failure_context} · {e.plan_tier}</small></span>
              <em>{inr(e.amount)}</em>
            </button>
          ))}
        </div>
      </div>

      <div className="panel decision-pane">
        <div className="observed-head">
          <div>
            <h2>Customer #{selectedEvent.customer_id}</h2>
            <p>{selectedEvent.failure_context} · {selectedEvent.plan_tier} · {selectedEvent.payment_method}</p>
          </div>
          <div className="amount-block">
            <span>Failed amount</span>
            <strong className="amount-highlight">{inr(selectedEvent.amount)}</strong>
          </div>
        </div>

        <div className="meta-grid">
          <div className="meta-item"><span className="meta-label">Engagement</span><b className="meta-val">{selectedEvent.engagement_score ?? '—'}</b></div>
          <div className="meta-item"><span className="meta-label">Historical failures</span><b className="meta-val">{selectedEvent.historical_failure_count ?? '—'}</b></div>
          <div className="meta-item">
            <span className="meta-label">WhatsApp opt-in</span>
            <b className="meta-val">{selectedEvent.whatsapp_opted_in ? <Check size={16} className="text-green" /> : <X size={16} className="text-red" />}</b>
          </div>
        </div>

        <div className="decision-flow">
          <div className="flow-step">
            <div className="step-label">1. Counterfactual Recovery Probabilities</div>
            <div className="step-content">
              {[
                { label: 'None', prob: selectedEvent.prob_none, color: 'gray' },
                { label: 'Retry', prob: selectedEvent.prob_retry, color: 'blue' },
                { label: 'WhatsApp', prob: selectedEvent.prob_whatsapp, color: 'green' },
              ].map((item) => (
                <div key={item.label} className="prob-row">
                  <span className="prob-label">{item.label}</span>
                  <div className="prob-bar-bg">
                    <div className={`prob-bar-fill ${item.color}`} style={{ width: `${item.prob * 100}%` }} />
                  </div>
                  <span className="prob-val">{pct(item.prob)}</span>
                </div>
              ))}
            </div>
          </div>

          <div className="flow-step">
            <div className="step-label">2. Incremental Lift (vs None)</div>
            <div className="step-content">
              <div className="lift-grid">
                <div className="lift-item">
                  <span className="lift-label">Retry</span>
                  <strong className={clsx(selectedEvent.inc_prob_retry > 0 ? 'text-green' : 'text-red')}>
                    {selectedEvent.inc_prob_retry > 0 ? `+${pct(selectedEvent.inc_prob_retry)}` : pct(selectedEvent.inc_prob_retry)}
                  </strong>
                </div>
                <div className="lift-item">
                  <span className="lift-label">WhatsApp</span>
                  <strong className={clsx(selectedEvent.inc_prob_whatsapp > 0 ? 'text-green' : 'text-red')}>
                    {selectedEvent.inc_prob_whatsapp > 0 ? `+${pct(selectedEvent.inc_prob_whatsapp)}` : pct(selectedEvent.inc_prob_whatsapp)}
                  </strong>
                </div>
              </div>
            </div>
          </div>

          <div className="flow-step">
            <div className="step-label">3. Expected Net Incremental Value (ENIV)</div>
            <div className="step-content">
              <div className="eniv-grid">
                <div className="eniv-item">
                  <span className="eniv-label">Retry</span>
                  <strong className={clsx(selectedEvent.eniv_retry > 0 ? 'text-green' : 'text-red')}>
                    {inr(selectedEvent.eniv_retry, 2)}
                  </strong>
                </div>
                <div className="eniv-item">
                  <span className="eniv-label">WhatsApp</span>
                  <strong className={clsx(selectedEvent.eniv_whatsapp > 0 ? 'text-green' : 'text-red')}>
                    {inr(selectedEvent.eniv_whatsapp, 2)}
                  </strong>
                </div>
              </div>
              <p className="eniv-formula">
                Formula: {inr(selectedEvent.amount)} × Lift − Cost
              </p>
            </div>
          </div>

          <div className="flow-step">
            <div className="step-label">4. Constraints & Uncertainty</div>
            <div className="step-content">
              <div className="constraint-item">
                <span className="constraint-label">WhatsApp Consent</span>
                <span className={clsx('constraint-val', selectedEvent.whatsapp_opted_in ? 'text-green' : 'text-red')}>
                  {selectedEvent.whatsapp_opted_in ? 'Eligible' : 'Not Eligible'}
                </span>
              </div>
              <div className="constraint-item">
                <span className="constraint-label">Uncertainty Proxy</span>
                <span className="constraint-val">
                  {selectedEvent.is_abstain ? 'Too High (Abstain)' : 'Acceptable'}
                </span>
              </div>
              <p className="uncertainty-disclaimer">
                Heuristic uncertainty proxy — not a calibrated confidence interval.
              </p>
            </div>
          </div>

          <div className="recommendation-final">
            <div className="final-label">Final Recommendation</div>
            <div className="final-action-box">
              <div className="final-baseline">
                <span className="baseline-tag">Gross Baseline</span>
                <strong>{actionLabel(selectedEvent.baseline_action)}</strong>
              </div>
              <div className="final-arrow"><ArrowRight size={20} /></div>
              <div className="final-result">
                <span className="result-tag">CausaPay Decision</span>
                <strong className="result-val">{actionLabel(selectedEvent.recommended_action)}</strong>
              </div>
            </div>
            <p className="explanation-text">{selectedEvent.explanation}</p>
          </div>
        </div>
      </div>
    </div>
  );
}
