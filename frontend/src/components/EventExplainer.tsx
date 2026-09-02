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
  inc_prob_retry: number;
  eniv_retry: number;
  prob_wa: number;
  inc_prob_wa: number;
  eniv_wa: number;
};

export default function EventExplainer({ events }: { events: ExplorerEvent[] }) {
  const [selectedEventId, setSelectedEventId] = useState<string | null>(events[0]?.event_id ?? null);
  const selectedEvent = events.find((e) => e.event_id === selectedEventId);

  return (
    <div className="explainer-layout">
      <div className="panel event-pane">
        <div className="panel-head">
          <h2>Sample events</h2>
        </div>
        <div className="event-list">
          {events.map((e) => (
            <button
              key={e.event_id}
              onClick={() => setSelectedEventId(e.event_id)}
              className={clsx('event-button', selectedEventId === e.event_id && 'selected')}
            >
              <span>
                <strong>{e.customer_id}</strong>
                <small>{e.failure_context} · {e.plan_tier}</small>
              </span>
              <em>{inr(e.amount)}</em>
            </button>
          ))}
        </div>
      </div>

      {selectedEvent ? (
        <div className="panel decision-pane">
          <div className="observed-head">
            <div>
              <h2>Customer #{selectedEvent.customer_id}</h2>
              <p>{selectedEvent.failure_context} · {selectedEvent.plan_tier} · {selectedEvent.payment_method}</p>
            </div>
            <div className="amount-block">
              <span>Failed amount</span>
              <strong>{inr(selectedEvent.amount)}</strong>
            </div>
          </div>
          <div className="meta-grid">
            <div><span>Engagement</span><b>{selectedEvent.engagement_score ?? '—'}</b></div>
            <div><span>Historical failures</span><b>{selectedEvent.historical_failure_count ?? '—'}</b></div>
            <div>
              <span>WhatsApp opt-in</span>
              <b>{selectedEvent.whatsapp_opted_in ? <Check size={16} /> : <X size={16} />}</b>
            </div>
          </div>
          <div className="recommendation-bar">
            <div>
              <span>Gross baseline</span>
              <strong>{actionLabel(selectedEvent.baseline_action)}</strong>
            </div>
            <ArrowRight size={16} />
            <div>
              <span>Recommended</span>
              <strong>{actionLabel(selectedEvent.recommended_action)}</strong>
            </div>
          </div>
          <p className="muted">
            Retry incremental {pct(selectedEvent.inc_prob_retry)} · ENIV {inr(selectedEvent.eniv_retry, 2)} ·
            WhatsApp incremental {pct(selectedEvent.inc_prob_wa)} · ENIV {inr(selectedEvent.eniv_wa, 2)} ·
            P(none) {pct(selectedEvent.prob_none)} · P(retry) {pct(selectedEvent.prob_retry)} · P(WA) {pct(selectedEvent.prob_wa)}
          </p>
        </div>
      ) : (
        <div className="panel"><p className="muted">Select an event to view details</p></div>
      )}
    </div>
  );
}
