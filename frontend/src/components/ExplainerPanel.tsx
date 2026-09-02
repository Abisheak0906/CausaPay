import { useEffect, useMemo, useState } from 'react';
import { Activity, ArrowRight, Check, ShieldCheck, Workflow, X } from 'lucide-react';
import clsx from 'clsx';
import { API } from '../lib/api';
import { actionLabel, inr, pct } from '../lib/format';
import type { Decision, EventItem } from '../types';

type AuditResponse = {
  event_id: string;
  decision_id?: string | null;
  timestamp?: string | null;
  audit?: Record<string, unknown>;
  recovery_lifecycle?: Array<Record<string, unknown>>;
  action_history?: Array<Record<string, unknown>>;
  workflow_events?: Array<Record<string, unknown>>;
};

type Scenario = {
  id: string;
  label: string;
  description: string;
  payload: Record<string, unknown>;
};

const DEMO_SCENARIOS: Scenario[] = [
  {
    id: 'retry',
    label: 'RETRY — recommended',
    description: 'Deterministic model fixture: RETRY is recommended without abstention.',
    payload: {
      event_id: '43',
      customer_id: 22777,
      amount: 1129.56,
      plan_tier: 'Basic',
      payment_method: 'UPI Autopay',
      failure_context: 'technical_timeout',
      decline_signal_bucket: 'technical',
      engagement_score: 0.2792202347426931,
      historical_failure_count: 1,
      whatsapp_opted_in: false,
      email_verified: true,
      days_since_last_failure: 240,
      historical_payment_count: 12,
      day_of_month: 21,
      tenure_days: 368,
    },
  },
  {
    id: 'whatsapp',
    label: 'WHATSAPP — recommended',
    description: 'Deterministic model fixture: WHATSAPP is recommended with consent.',
    payload: {
      event_id: '9',
      customer_id: 17850,
      amount: 284.14,
      plan_tier: 'Basic',
      payment_method: 'UPI Autopay',
      failure_context: 'insufficient_funds',
      decline_signal_bucket: 'soft',
      engagement_score: 0.3252198409964951,
      historical_failure_count: 2,
      whatsapp_opted_in: true,
      email_verified: true,
      days_since_last_failure: 346,
      historical_payment_count: 10,
      day_of_month: 8,
      tenure_days: 983,
    },
  },
  {
    id: 'abstention_none',
    label: 'ABSTENTION / NONE',
    description: 'Deterministic model fixture: uncertainty causes abstention and NONE.',
    payload: {
      event_id: '0',
      customer_id: 24654,
      amount: 214.77,
      plan_tier: 'Pro',
      payment_method: 'Card',
      failure_context: 'insufficient_funds',
      decline_signal_bucket: 'soft',
      engagement_score: 0.2622742530538169,
      historical_failure_count: 2,
      whatsapp_opted_in: false,
      email_verified: false,
      days_since_last_failure: 76,
      historical_payment_count: 5,
      day_of_month: 15,
      tenure_days: 111,
    },
  },
  {
    id: 'safeguard_blocked_whatsapp',
    label: 'SAFEGUARD — WhatsApp consent=false → RETRY',
    description: 'Model recommends WHATSAPP, then consent safeguard falls back to executable RETRY.',
    payload: {
      event_id: 'phase9-safeguard-consent-false',
      customer_id: 17850,
      amount: 284.14,
      plan_tier: 'Basic',
      payment_method: 'UPI Autopay',
      failure_context: 'insufficient_funds',
      decline_signal_bucket: 'soft',
      engagement_score: 0.3252198409964951,
      historical_failure_count: 2,
      whatsapp_opted_in: false,
      email_verified: true,
      days_since_last_failure: 346,
      historical_payment_count: 10,
      day_of_month: 8,
      tenure_days: 983,
    },
  },
];

type Props = {
  events: EventItem[];
  selected: EventItem | null;
  decision: Decision | null;
  onSelect: (event: EventItem) => void;
};

export function ExplainerPanel({ events, selected, decision, onSelect }: Props) {
  const [localEvents, setLocalEvents] = useState<EventItem[]>(events);
  const [localDecision, setLocalDecision] = useState<Decision | null>(decision);
  const [selectedEventId, setSelectedEventId] = useState<string | null>(selected?.event_id ?? events[0]?.event_id ?? null);
  const [demoScenarioId, setDemoScenarioId] = useState<string>(DEMO_SCENARIOS[0].id);
  const [audit, setAudit] = useState<AuditResponse | null>(null);
  const [actionBusy, setActionBusy] = useState<string | null>(null);
  const [notice, setNotice] = useState<string>('');
  const [finalAction, setFinalAction] = useState<string | null>(null);

  useEffect(() => {
    setLocalEvents(events);
  }, [events]);

  useEffect(() => {
    setLocalDecision(decision);
  }, [decision]);

  useEffect(() => {
    if (selected?.event_id) {
      setSelectedEventId(selected.event_id);
    }
  }, [selected]);

  const visibleEvents = useMemo(() => (localEvents.length > 0 ? localEvents : events), [localEvents, events]);
  const selectedEvent = visibleEvents.find((event) => event.event_id === selectedEventId) ?? selected ?? visibleEvents[0] ?? null;

  const currentLifecycleState = useMemo(() => {
    if (!audit || !Array.isArray(audit.recovery_lifecycle) || audit.recovery_lifecycle.length === 0) {
      return 'PENDING';
    }

    const latest = audit.recovery_lifecycle[audit.recovery_lifecycle.length - 1];
    return typeof latest?.event === 'string' && latest.event.trim().length > 0 ? latest.event : 'PENDING';
  }, [audit]);

  useEffect(() => {
    if (!selectedEvent) return;

    const loadAudit = async () => {
      try {
        const response = await fetch(`${API}/api/events/${encodeURIComponent(selectedEvent.event_id)}/audit`);
        if (!response.ok) {
          setAudit(null);
          return;
        }
        const data = (await response.json()) as AuditResponse;
        setAudit(data);
      } catch {
        setAudit(null);
      }
    };

    void loadAudit();
  }, [selectedEvent]);

  const refreshEventDetail = async (eventId: string) => {
    try {
      const response = await fetch(`${API}/api/events/${encodeURIComponent(eventId)}`);
      if (!response.ok) return null;
      const data = (await response.json()) as Decision & EventItem;
      setLocalDecision(data as Decision);

      const eventListResponse = await fetch(`${API}/api/events?limit=80`);
      if (eventListResponse.ok) {
        const list = (await eventListResponse.json()) as { events?: EventItem[] };
        setLocalEvents(list.events ?? []);
      }
      return data;
    } catch {
      return null;
    }
  };

  const runScenarioDecision = async () => {
    const scenario = DEMO_SCENARIOS.find((item) => item.id === demoScenarioId);
    if (!scenario) return;

    setActionBusy('decision');
    setNotice('Sending decision to the causal engine…');

    try {
      const response = await fetch(`${API}/api/decision`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(scenario.payload),
      });

      const data = await response.json();
      if (!response.ok) {
        throw new Error(data?.detail || 'Decision failed');
      }

      const eventId = String(data.event_id);
      setLocalDecision(data as Decision);
      setFinalAction(null);
      setSelectedEventId(eventId);
      onSelect({
        ...scenario.payload,
        event_id: eventId,
        amount: Number(scenario.payload.amount ?? 0),
        recommended_action: data.recommended_action,
        baseline_action: data.baseline_action ?? 'none',
        is_abstain: Boolean(data.is_abstain),
        source: 'dynamic',
      } as EventItem);

      await refreshEventDetail(eventId);
      setNotice(`Decision created for ${eventId}. Recommended action: ${data.recommended_action}.`);
    } catch (error) {
      setNotice(error instanceof Error ? error.message : 'Decision request failed.');
    } finally {
      setActionBusy(null);
    }
  };

  const runNextAction = async () => {
    if (!selectedEvent) return;

    setActionBusy('next-action');
    setNotice('Executing the recommended next action…');

    try {
      const response = await fetch(`${API}/api/events/${encodeURIComponent(selectedEvent.event_id)}/next-action`, { method: 'POST' });
      const data = await response.json();
      if (!response.ok) {
        throw new Error(data?.detail || data?.message || 'Next action failed');
      }

      setFinalAction(String(data.recommended_action ?? 'STOP'));
      setNotice(`Workflow prepared for ${selectedEvent.event_id}: ${data.recommended_action ?? 'STOP'}.`);
      await refreshEventDetail(selectedEvent.event_id);
    } catch (error) {
      setNotice(error instanceof Error ? error.message : 'Next action failed.');
    } finally {
      setActionBusy(null);
    }
  };

  const simulateOutcome = async (status: 'PENDING' | 'FAILED' | 'RECOVERED') => {
    if (!selectedEvent) return;

    setActionBusy(`outcome-${status.toLowerCase()}`);
    setNotice(`Simulating ${status.toLowerCase()} outcome for ${selectedEvent.event_id}…`);

    try {
      const response = await fetch(`${API}/api/events/${encodeURIComponent(selectedEvent.event_id)}/outcome`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ status, source: 'demo_simulation' }),
      });
      const data = await response.json();
      if (!response.ok) {
        throw new Error(data?.detail || data?.message || 'Demo outcome failed');
      }
      setNotice(`Outcome recorded: ${data.status}.`);
      await refreshEventDetail(selectedEvent.event_id);
    } catch (error) {
      setNotice(error instanceof Error ? error.message : 'Outcome simulation failed.');
    } finally {
      setActionBusy(null);
    }
  };

  const eventOptions = localDecision
    ? [
        { key: 'none', label: 'No intervention', prob: localDecision.prob_none, inc: 0, eniv: localDecision.eniv_none, cost: 0 },
        { key: 'retry', label: 'Retry', prob: localDecision.prob_retry, inc: localDecision.inc_prob_retry, eniv: localDecision.eniv_retry, cost: 2 },
        { key: 'whatsapp', label: 'WhatsApp', prob: localDecision.prob_whatsapp, inc: localDecision.inc_prob_whatsapp, eniv: localDecision.eniv_whatsapp, cost: 15 },
      ]
    : [];

  const selectedDecision = localDecision ?? decision;

  const safeguardSummary = selectedEvent
    ? [
        selectedEvent.whatsapp_opted_in === false ? 'WhatsApp consent missing' : null,
        selectedDecision?.is_abstain ? 'Causal uncertainty too high' : null,
        selectedDecision?.recommended_action === 'none' ? 'No intervention recommended by ENIV policy' : null,
        selectedEvent.source === 'dynamic' ? 'Persisted dynamic event' : 'Seeded demo event',
      ].filter(Boolean) as string[]
    : [];

  const timeline = [
    ...(audit?.action_history ?? []).map((entry) => ({
      label: String(entry.action ?? 'ACTION'),
      time: String(entry.timestamp ?? entry.created_at ?? ''),
    })),
    ...(audit?.recovery_lifecycle ?? []).map((entry) => ({
      label: String(entry.event ?? entry.action ?? entry.current_state ?? 'STATE'),
      time: String(entry.timestamp ?? entry.created_at ?? ''),
    })),
    ...(audit?.workflow_events ?? []).map((entry) => ({
      label: `Workflow ${String(entry.status ?? 'UPDATE')}`,
      time: String(entry.dispatch_timestamp ?? entry.completion_timestamp ?? ''),
    })),
  ];

  return (
    <div className="explainer-layout">
      <div className="panel event-pane">
        <div className="panel-head">
          <span className="panel-kicker">Demo runner</span>
          <h2>Scenario playback</h2>
        </div>

        <div className="demo-shell">
          <label className="demo-label" htmlFor="demo-scenario">Scenario</label>
          <select id="demo-scenario" value={demoScenarioId} onChange={(event) => setDemoScenarioId(event.target.value)} className="demo-select">
            {DEMO_SCENARIOS.map((scenario) => (
              <option key={scenario.id} value={scenario.id}>{scenario.label}</option>
            ))}
          </select>
          <p className="demo-description">
            {DEMO_SCENARIOS.find((scenario) => scenario.id === demoScenarioId)?.description}
          </p>
          <div className="demo-actions">
            <button type="button" className="primary-btn" onClick={runScenarioDecision} disabled={Boolean(actionBusy)}>
              {actionBusy === 'decision' ? 'Running…' : 'Run Decision'}
            </button>
            <button type="button" className="secondary-btn" onClick={runNextAction} disabled={Boolean(actionBusy) || !selectedEvent}>
              {actionBusy === 'next-action' ? 'Processing…' : 'Execute Next Action'}
            </button>
          </div>
          <div className="demo-outcome-wrap">
            <span className="panel-kicker">Demo Simulation</span>
            <div className="demo-actions outcome-actions">
              <button type="button" className="secondary-btn" onClick={() => void simulateOutcome('PENDING')} disabled={Boolean(actionBusy) || !selectedEvent}>Pending</button>
              <button type="button" className="secondary-btn" onClick={() => void simulateOutcome('FAILED')} disabled={Boolean(actionBusy) || !selectedEvent}>Failed</button>
              <button type="button" className="secondary-btn" onClick={() => void simulateOutcome('RECOVERED')} disabled={Boolean(actionBusy) || !selectedEvent}>Recovered</button>
            </div>
          </div>
          {notice ? <p className="demo-notice">{notice}</p> : null}
        </div>

        <div className="panel-head demo-list-head">
          <h2>Failed payments</h2>
        </div>
        <div className="event-list">
          {visibleEvents.map((event) => (
            <button
              key={event.event_id}
              onClick={() => {
                setSelectedEventId(event.event_id);
                onSelect(event);
              }}
              className={clsx('event-button', selectedEventId === event.event_id && 'selected')}
            >
              <span>
                <strong>{event.customer_id ?? 'unknown customer'}</strong>
                <small>{event.failure_context ?? 'unknown failure'} · {event.plan_tier ?? 'unknown plan'}</small>
              </span>
              <em>{inr(event.amount)}</em>
            </button>
          ))}
        </div>
      </div>

      {selectedEvent ? (
        <div className="panel decision-pane">
          <div className="observed-head">
            <div>
              <span className="panel-kicker">{selectedEvent.source === 'dynamic' ? 'Dynamic event' : 'Seeded demo'}</span>
              <h2>{selectedEvent.customer_id ? `Customer ${selectedEvent.customer_id}` : `Event ${selectedEvent.event_id}`}</h2>
              <p>{selectedEvent.failure_context ?? 'Unknown failure'} · {selectedEvent.plan_tier ?? 'unknown plan'} · {selectedEvent.payment_method ?? 'unknown method'}</p>
            </div>
            <div className="amount-block">
              <span>Failed amount</span>
              <strong>{inr(selectedEvent.amount)}</strong>
            </div>
          </div>

          <div className="meta-grid">
            <div><span>Event ID</span><b>{selectedEvent.event_id}</b></div>
            <div><span>Payment ID</span><b>{selectedEvent.payment_id ?? '—'}</b></div>
            <div><span>Engagement</span><b>{Number.isFinite(Number(selectedEvent.engagement_score)) ? Number(selectedEvent.engagement_score).toFixed(2) : '—'}</b></div>
            <div><span>Historical failures</span><b>{selectedEvent.historical_failure_count ?? '—'}</b></div>
            <div><span>WhatsApp opt-in</span><b>{selectedEvent.whatsapp_opted_in ? <Check size={16} /> : <X size={16} />}</b></div>
            <div><span>Decline bucket</span><b>{selectedEvent.decline_signal_bucket ?? '—'}</b></div>
            <div><span>Source</span><b>{selectedEvent.source ?? 'seeded_demo'}</b></div>
            <div><span>Abstain</span><b>{selectedDecision?.is_abstain ? 'Yes' : 'No'}</b></div>
          </div>

          <div className="story-panels">
            <div className="story-panel">
              <div className="section-kicker"><Activity size={14} /> Failed payment</div>
              <ul>
                <li><strong>Event ID:</strong> {selectedEvent.event_id}</li>
                <li><strong>Payment ID:</strong> {selectedEvent.payment_id ?? '—'}</li>
                <li><strong>Amount:</strong> {inr(selectedEvent.amount)}</li>
                <li><strong>Payment method:</strong> {selectedEvent.payment_method ?? 'unknown'}</li>
                <li><strong>Failure context:</strong> {selectedEvent.failure_context ?? 'unknown'}</li>
                <li><strong>Customer / plan:</strong> {selectedEvent.customer_id ?? 'unknown'} · {selectedEvent.plan_tier ?? 'unknown'}</li>
              </ul>
            </div>

            <div className="story-panel">
              <div className="section-kicker"><ArrowRight size={14} /> Causal decision</div>
              {selectedDecision ? (
                <>
                  <div className="decision-pill">Recommended: {actionLabel(selectedDecision.recommended_action)}</div>
                  <p>Recovery with no intervention: <strong>{pct(selectedDecision.prob_none)}</strong></p>
                  <p>Recovery with RETRY: <strong>{pct(selectedDecision.prob_retry)}</strong></p>
                  <p>Recovery with WHATSAPP: <strong>{pct(selectedDecision.prob_whatsapp)}</strong></p>
                  <p>Incremental lift: <strong>{inr(selectedDecision.inc_prob_retry, 2)} / {inr(selectedDecision.inc_prob_whatsapp, 2)}</strong></p>
                  <p>ENIV: <strong>{inr(selectedDecision.eniv_retry, 2)} / {inr(selectedDecision.eniv_whatsapp, 2)}</strong></p>
                  <p>{selectedDecision.explanation}</p>
                </>
              ) : (
                <p className="muted">No decision available yet for this event.</p>
              )}
            </div>

            <div className="story-panel">
              <div className="section-kicker"><ShieldCheck size={14} /> Safeguards</div>
              {safeguardSummary.length > 0 ? (
                <ul>
                  {safeguardSummary.map((reason) => <li key={reason}>{reason}</li>)}
                </ul>
              ) : (
                <p>Intervention passed safeguards.</p>
              )}
            </div>

            <div className="story-panel">
              <div className="section-kicker"><ArrowRight size={14} /> Execution path</div>
              <p>Model recommendation: <strong>{selectedDecision ? actionLabel(selectedDecision.recommended_action) : 'Pending'}</strong></p>
              <p>Safeguard decision: <strong>{selectedDecision?.recommended_action === 'whatsapp' && selectedEvent.whatsapp_opted_in === false ? 'WHATSAPP blocked — fallback RETRY' : 'Passed'}</strong></p>
              <p>Final executable action: <strong>{finalAction ? actionLabel(finalAction) : 'Not executed'}</strong></p>
            </div>

            <div className="story-panel">
              <div className="section-kicker"><Workflow size={14} /> Workflow execution</div>
              <p>Workflow required: <strong>{selectedDecision?.workflow?.required ? 'Yes' : 'No'}</strong></p>
              <p>Execution mode: <strong>{selectedDecision?.workflow?.mode ?? 'local'}</strong></p>
              <p>Status: <strong>{selectedDecision?.workflow?.status ?? 'NOT_REQUIRED'}</strong></p>
              <p>Workflow ID: <strong>{selectedDecision?.workflow?.workflow_id ?? '—'}</strong></p>
              <p>Last update: <strong>{selectedDecision?.workflow?.last_updated ?? '—'}</strong></p>
              <p>Failure reason: <strong>{selectedDecision?.workflow?.failure_reason ?? 'none'}</strong></p>
            </div>

            <div className="story-panel">
              <div className="section-kicker"><Check size={14} /> Outcome</div>
              <p>Current state: <strong>{currentLifecycleState}</strong></p>
              <p>Outcome status: <strong>{audit && audit.recovery_lifecycle?.length ? 'Recorded' : 'Pending'}</strong></p>
              <p>Action history: <strong>{audit?.action_history?.length ?? 0}</strong></p>
            </div>
          </div>

          <div className="recommendation-bar">
            <div>
              <span>Gross baseline</span>
              <strong>{actionLabel(selectedEvent.baseline_action ?? 'none')}</strong>
            </div>
            <ArrowRight size={16} />
            <div>
              <span>Recommended</span>
              <strong>{selectedDecision ? actionLabel(selectedDecision.recommended_action) : actionLabel(selectedEvent.recommended_action ?? 'none')}</strong>
              {selectedDecision?.is_abstain ? <em>Abstained — low confidence, baseline used</em> : null}
            </div>
          </div>

          {selectedDecision ? (
            <>
              <div className="decision-grid">
                {eventOptions.map((option) => {
                  const recommended = selectedDecision.recommended_action === option.key;
                  return (
                    <article key={option.key} className={`decision-option ${recommended ? 'recommended' : ''}`}>
                      <header>
                        <h3>{option.label}</h3>
                        {recommended ? <span>Recommended</span> : null}
                      </header>
                      <p>Recovery probability <strong>{pct(option.prob)}</strong></p>
                      <p>Incremental vs none <strong>{option.key === 'none' ? '—' : pct(option.inc)}</strong></p>
                      <p>Intervention cost <strong>{inr(option.cost, 0)}</strong></p>
                      <p className="eniv">ENIV <strong>{inr(option.eniv, 2)}</strong></p>
                    </article>
                  );
                })}
              </div>
              <p className="explanation">{selectedDecision.explanation}</p>
            </>
          ) : null}

          <div className="audit-panel">
            <div className="panel-head">
              <span className="panel-kicker">Audit timeline</span>
              <h2>Decision and execution trail</h2>
            </div>
            {timeline.length > 0 ? (
              <ol className="audit-timeline">
                {timeline.map((entry, index) => (
                  <li key={`${entry.label}-${entry.time}-${index}`}>
                    <span>{entry.label}</span>
                    <small>{entry.time || 'timestamp unavailable'}</small>
                  </li>
                ))}
              </ol>
            ) : (
              <p className="muted">No audit timeline is available yet for this event.</p>
            )}
          </div>
        </div>
      ) : (
        <div className="panel"><p className="muted">Select a failed payment to inspect the causal story.</p></div>
      )}
    </div>
  );
}
