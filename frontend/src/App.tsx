import { useEffect, useState, type FormEvent, type MouseEvent } from 'react';
import {
  Activity,
  BarChart3,
  CircleDollarSign,
  FlaskConical,
  LayoutDashboard,
  LogOut,
  Network,
  Search,
  Upload,
} from 'lucide-react';
import './App.css';
import { ActionDistribution } from './components/ActionDistribution';
import { CannibalChart } from './components/CannibalChart';
import CausalityExplainer from './components/CausalityExplainer';
import { DiagnosticMetrics } from './components/DiagnosticMetrics';
import ExecutiveSummary from './components/ExecutiveSummary';
import { ExplainerPanel } from './components/ExplainerPanel';
import { LoadingSpinner } from './components/LoadingSpinner';
import { OverviewCard } from './components/OverviewCard';
import { PolicyTable } from './components/PolicyTable';
import { RecoveryImpactPanel } from './components/RecoveryImpactPanel';
import { TryDataPanel } from './components/TryDataPanel';
import { UploadDatasetPanel } from './components/UploadDatasetPanel';
import { API } from './lib/api';
import { inr } from './lib/format';
import type { Decision, Diagnostics, EventItem, PageId, Policy, Summary } from './types';

const navigation: Array<{ href: string; label: string; icon: typeof LayoutDashboard; page: PageId }> = [
  { href: '/', label: 'Overview', icon: LayoutDashboard, page: 'overview' },
  { href: '/policy', label: 'Policy Intelligence', icon: BarChart3, page: 'policy' },
  { href: '/payment', label: 'Payment Explorer', icon: Search, page: 'payment' },
  { href: '/try-data', label: 'Try Your Data', icon: Activity, page: 'try-data' },
  { href: '/upload', label: 'Upload Dataset', icon: Upload, page: 'upload' },
  { href: '/cannibal', label: 'Recovery Attribution', icon: CircleDollarSign, page: 'cannibal' },
  { href: '/diagnostics', label: 'Diagnostics', icon: FlaskConical, page: 'diagnostics' },
];

function pageFromPath(pathname: string): PageId {
  if (pathname.startsWith('/policy')) return 'policy';
  if (pathname.startsWith('/payment')) return 'payment';
  if (pathname.startsWith('/try-data')) return 'try-data';
  if (pathname.startsWith('/upload')) return 'upload';
  if (pathname.startsWith('/cannibal')) return 'cannibal';
  if (pathname.startsWith('/diagnostics')) return 'diagnostics';
  return 'overview';
}

function policiesFromUploadedEvents(events: EventItem[]): Policy[] {
  const totals = events.reduce((result, event) => {
    const action = event.is_abstain ? 'none' : (event.recommended_action ?? 'none').toLowerCase();
    const amount = Number(event.amount) || 0;
    const probability = action === 'retry' ? Number(event.prob_retry ?? 0) : action === 'whatsapp' ? Number(event.prob_whatsapp ?? 0) : Number(event.prob_none ?? 0);
    const incremental = action === 'retry' ? Number(event.inc_prob_retry ?? 0) : action === 'whatsapp' ? Number(event.inc_prob_whatsapp ?? 0) : 0;
    result.gross += probability * amount;
    result.incremental += incremental * amount;
    result.cost += action === 'retry' ? 2 : action === 'whatsapp' ? 15 : 0;
    return result;
  }, { gross: 0, incremental: 0, cost: 0 });
  return [{
    policy_name: 'Uploaded dataset (incrementality-aware)',
    gross_recovered: totals.gross,
    true_incremental_recovered: totals.incremental,
    intervention_cost: totals.cost,
    policy_value: totals.incremental - totals.cost,
    recovery_rate: totals.gross / (events.reduce((sum, event) => sum + (Number(event.amount) || 0), 0) || 1),
    action_distribution: events.reduce<Record<string, number>>((counts, event) => {
      const action = event.is_abstain ? 'abstained' : (event.recommended_action ?? 'none').toLowerCase();
      counts[action] = (counts[action] ?? 0) + 1;
      return counts;
    }, {}),
  }];
}

const headings: Record<PageId, [string, string]> = {
  overview: [
    'Recover revenue without paying for self-cures.',
    'Failed payments are scored through counterfactuals. CausaPay estimates the incremental effect of each intervention, subtracts cost, and only acts when ENIV is positive.',
  ],
  policy: [
    'Policy Intelligence',
    'Compare gross-recovery targeting with incrementality-aware allocation and the oracle upper bound on the same held-out events.',
  ],
  payment: [
    'Payment Explorer',
    'Inspect an observed failure, its counterfactual recovery probabilities, incremental effect, ENIV, and the recommended action.',
  ],
  'try-data': ['Try Your Own Payment Event', 'Submit a failed payment to the real CausaPay decision endpoint.'],
  upload: ['Upload Dataset', 'Process a CSV of failed payments through the real causal decision engine.'],
  cannibal: [
    'Recovery Attribution',
    'Separate intervention-driven recovery from payments that would have succeeded anyway.',
  ],
  diagnostics: [
    'Decision Quality',
    'Outcome fit, CATE calibration, overlap, and the limitations that should constrain how these estimates are used.',
  ],
};

function App() {
  const [authenticated, setAuthenticated] = useState(() => localStorage.getItem('causapay_demo_session') === 'active');
  const [authMode, setAuthMode] = useState<'login' | 'signup'>('login');
  const [authError, setAuthError] = useState('');
  const [authName, setAuthName] = useState('');
  const [authEmail, setAuthEmail] = useState('');
  const [authPassword, setAuthPassword] = useState('');
  const [summary, setSummary] = useState<Summary | null>(null);
  const [policies, setPolicies] = useState<Policy[]>([]);
  const [events, setEvents] = useState<EventItem[]>([]);
  const [diagnostics, setDiagnostics] = useState<Diagnostics | null>(null);
  const [selected, setSelected] = useState<EventItem | null>(null);
  const [decision, setDecision] = useState<Decision | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState('');
  const [current, setCurrent] = useState<PageId>(() => pageFromPath(window.location.pathname));
  const [activeDataset, setActiveDataset] = useState('demo');
  const [demoSnapshot, setDemoSnapshot] = useState<{ summary: Summary | null; policies: Policy[]; events: EventItem[]; diagnostics: Diagnostics | null }>({ summary: null, policies: [], events: [], diagnostics: null });

  useEffect(() => {
    if (!authenticated) {
      setLoading(false);
      return;
    }

    const onPop = () => setCurrent(pageFromPath(window.location.pathname));
    window.addEventListener('popstate', onPop);
    return () => window.removeEventListener('popstate', onPop);
  }, [authenticated]);

  useEffect(() => {
    const read = async <T,>(path: string): Promise<T | null> => {
      try {
        const response = await fetch(`${API}${path}`);
        if (!response.ok) return null;
        return (await response.json()) as T;
      } catch {
        return null;
      }
    };

    Promise.all([
      read<Summary>('/api/summary'),
      read<Policy[]>('/api/policy-comparison'),
      read<{ events: EventItem[] }>('/api/events?limit=80'),
      read<Diagnostics>('/api/diagnostics'),
    ])
      .then(([s, p, e, d]) => {
        const policiesData = p ?? [];
        const ours = policiesData.find((row) => row.policy_name.toLowerCase().includes('increment'));
        setPolicies(policiesData);
        setSummary(
          s ??
            (ours
              ? {
                  total_failed_payments: 0,
                  gross_recovered: ours.gross_recovered,
                  incremental_recovered: ours.true_incremental_recovered,
                  intervention_cost: ours.intervention_cost,
                  policy_value: ours.policy_value,
                  action_distribution: ours.action_distribution,
                }
              : null),
        );
        setEvents(e?.events ?? []);
        setDemoSnapshot({ summary: s ?? null, policies: policiesData, events: e?.events ?? [], diagnostics: d });
        setDiagnostics(d);
        setSelected(e?.events[0] ?? null);
        if (!s && policiesData.length === 0 && !e && !d) {
          setError('CausaPay could not reach the API. Start FastAPI on port 8000 and refresh.');
        }
      })
      .finally(() => setLoading(false));
  }, []);

  useEffect(() => {
    if (!selected) return;
    setDecision(null);
    fetch(`${API}/api/events/${encodeURIComponent(selected.event_id)}`)
      .then((r) => (r.ok ? r.json() : Promise.reject()))
      .then(setDecision)
      .catch(() => setDecision(null));
  }, [selected]);

  const submitAuth = (event: FormEvent<HTMLFormElement>) => {
    event.preventDefault();
    setAuthError('');
    const email = authEmail.trim();
    if (authMode === 'signup' && !authName.trim()) {
      setAuthError('Please enter your name.');
      return;
    }
    if (!email || !email.includes('@')) {
      setAuthError('Enter a valid email address.');
      return;
    }
    if (authPassword.length < 6) {
      setAuthError('Password must be at least 6 characters.');
      return;
    }
    localStorage.setItem('causapay_demo_user', JSON.stringify({ name: authName.trim() || email.split('@')[0], email }));
    localStorage.setItem('causapay_demo_session', 'active');
    setAuthenticated(true);
  };

  if (!authenticated) {
    return (
      <main className="auth-shell">
        <section className="auth-card">
          <div className="brand auth-brand">
            <div className="brand-mark">C</div>
            <div className="brand-text"><strong>CausaPay</strong><small>Causal Recovery Intelligence</small></div>
          </div>
          <span className="auth-kicker">Demo workspace</span>
          <h1>{authMode === 'login' ? 'Welcome back' : 'Create your workspace'}</h1>
          <p className="auth-subtitle">{authMode === 'login' ? 'Sign in to continue to your recovery dashboard.' : 'Set up a local demo account to explore CausaPay.'}</p>
          <form onSubmit={submitAuth} className="auth-form">
            {authMode === 'signup' ? <label>Name<input value={authName} onChange={(e) => setAuthName(e.target.value)} placeholder="Your name" autoComplete="name" /></label> : null}
            <label>Email<input type="email" value={authEmail} onChange={(e) => setAuthEmail(e.target.value)} placeholder="you@company.com" autoComplete="email" /></label>
            <label>Password<input type="password" value={authPassword} onChange={(e) => setAuthPassword(e.target.value)} placeholder="At least 6 characters" autoComplete={authMode === 'login' ? 'current-password' : 'new-password'} /></label>
            {authError ? <p className="auth-error" role="alert">{authError}</p> : null}
            <button type="submit" className="primary-btn auth-submit">{authMode === 'login' ? 'Log in' : 'Sign up'}</button>
          </form>
          <p className="auth-switch">
            {authMode === 'login' ? 'New to CausaPay?' : 'Already have an account?'}{' '}
            <button type="button" onClick={() => { setAuthMode(authMode === 'login' ? 'signup' : 'login'); setAuthError(''); }}>
              {authMode === 'login' ? 'Create an account' : 'Log in'}
            </button>
          </p>
        </section>
      </main>
    );
  }

  if (loading) return <LoadingSpinner label="Loading causal recovery intelligence…" />;
  if (error) {
    return (
      <main className="app-shell error-shell">
        <div className="error-panel">
          <Network size={22} />
          <div>
            <strong>Connection unavailable</strong>
            <p>{error}</p>
          </div>
        </div>
      </main>
    );
  }

  const [title, subtitle] = headings[current];
  const selfCure = summary ? summary.gross_recovered - summary.incremental_recovered : 0;

  const go = (href: string, page: PageId) => (event: MouseEvent<HTMLAnchorElement>) => {
    event.preventDefault();
    window.history.pushState({}, '', href);
    setCurrent(page);
  };
  const onManualDecision = (event: EventItem, result: Decision) => {
    setEvents((items) => [event, ...items.filter((item) => item.event_id !== event.event_id)]);
    setSelected(event);
    setDecision(result);
    window.history.pushState({}, '', '/payment');
    setCurrent('payment');
  };
  const onUploadComplete = (uploadedSummary: Summary, uploadedEvents: EventItem[]) => {
    setSummary(uploadedSummary);
    setPolicies(policiesFromUploadedEvents(uploadedEvents));
    setEvents(uploadedEvents);
    setSelected(uploadedEvents[0] ?? null);
    setDecision(null);
    setDiagnostics(null);
    window.history.pushState({}, '', '/payment');
    setCurrent('payment');
    setActiveDataset(uploadedSummary.dataset_id ?? 'uploaded');
  };
  const switchToDemo = () => {
    setSummary(demoSnapshot.summary);
    setPolicies(demoSnapshot.policies);
    setEvents(demoSnapshot.events);
    setSelected(demoSnapshot.events[0] ?? null);
    setDecision(null);
    setDiagnostics(demoSnapshot.diagnostics);
    setActiveDataset('demo');
  };

  const overview = (
    <>
      <section className="overview-intro">
        <div>
          <span className="badge"><Activity size={13} /> Revenue Recovery Intelligence</span>
          <h1>{title}</h1>
          <p>{subtitle}</p>
        </div>
        <div className="causal-chain" aria-label="Causal decision pipeline">
          <span>Failed payment</span>
          <i>→</i>
          <span>Counterfactuals</span>
          <i>→</i>
          <span>Causal effect</span>
          <i>→</i>
          <span>Incremental recovery</span>
          <i>→</i>
          <span>Cost</span>
          <i>→</i>
          <strong>ENIV → action</strong>
        </div>
      </section>
      <RecoveryImpactPanel events={events} policies={policies} uploaded={activeDataset !== 'demo'} />
      <section className="kpi-grid">
        <OverviewCard
          label="Incremental recovery"
          value={summary?.incremental_recovered ?? 0}
          currency
          accent
          primary
          hint="Revenue caused by intervention — the primary objective"
        />
        <OverviewCard label="Gross recovery" value={summary?.gross_recovered ?? 0} currency hint="Includes payments that would have self-cured" />
        <OverviewCard label="Implied self-cure" value={selfCure} currency hint="Gross minus incremental on the production policy" />
        <OverviewCard label="Policy value (ENIV)" value={summary?.policy_value ?? 0} currency accent hint="Incremental recovery minus intervention cost" />
        <OverviewCard label="Intervention cost" value={summary?.intervention_cost ?? 0} currency />
        <OverviewCard label="Failed payments" value={summary?.total_failed_payments ?? 0} hint="Held-out evaluation events" />
      </section>
      <ExecutiveSummary results={policies} />
      <section className="overview-analytics">
        <CannibalChart policies={policies} compact />
        <ActionDistribution distribution={summary?.action_distribution ?? {}} />
      </section>
      <CausalityExplainer />
      <section className="policy-section">
        <div className="section-heading">
          <div>
            <span>Policy intelligence</span>
            <h2>Choose the value created by intervention.</h2>
          </div>
          <p>Production policy value on this slice is {inr(summary?.policy_value ?? 0)} after cost.</p>
        </div>
        <PolicyTable policies={policies} compact />
      </section>
    </>
  );

  const content =
    current === 'try-data' ? (
      <TryDataPanel onComplete={onManualDecision} />
    ) : current === 'upload' ? (
      <UploadDatasetPanel onComplete={onUploadComplete} />
    ) : current === 'policy' ? (
      <section className="page-stack">
        <CannibalChart policies={policies} />
        <PolicyTable policies={policies} />
      </section>
    ) : current === 'payment' ? (
      <ExplainerPanel events={events} selected={selected} decision={decision} onSelect={setSelected} />
    ) : current === 'cannibal' ? (
      <CannibalChart policies={policies} presentation />
    ) : current === 'diagnostics' ? (
      <DiagnosticMetrics diagnostics={diagnostics} />
    ) : (
      overview
    );

  return (
    <div className="product-shell">
      <header className="topnav" role="banner">
        <div className="topnav-inner">
          <div className="brand">
            <div className="brand-mark">C</div>
            <div className="brand-text">
              <strong>CausaPay</strong>
              <small>Causal Recovery Intelligence</small>
            </div>
          </div>

          <nav className="primary-nav" aria-label="Primary navigation">
            {navigation.map((item) => {
              const Icon = item.icon;
              return (
                <a
                  key={item.page}
                  className={`nav-link ${current === item.page ? 'active' : ''}`}
                  href={item.href}
                  onClick={go(item.href, item.page)}
                >
                  <Icon size={16} />
                  <span>{item.label}</span>
                </a>
              );
            })}
          </nav>

          <div className="topnav-actions">
            <div className="topbar-kicker">{current === 'overview' ? 'Executive overview' : 'CausaPay intelligence'}</div>
            <div className="dataset-indicator">Viewing: {activeDataset === 'demo' ? 'Demo Dataset' : `Uploaded Dataset — ${activeDataset}`}</div>
            <div className="topbar-status"><i className="online" /> API connected · live evaluation</div>
            {activeDataset !== 'demo' ? <button type="button" className="dataset-switch" onClick={switchToDemo}>Demo Dataset</button> : null}
            <button type="button" className="logout-btn" onClick={() => { localStorage.removeItem('causapay_demo_session'); setAuthenticated(false); }}> <LogOut size={14} /> Log out</button>
          </div>
        </div>
      </header>

      <main className="app-content">
        {current !== 'overview' ? (
          <section className={`page-hero ${current}`}>
            <span className="page-eyebrow">Causal AI analytics</span>
            <h1>{title}</h1>
            <p>{subtitle}</p>
          </section>
        ) : (
          <section className="overview-hero">
            <div className="overview-intro">
              <div>
                <span className="badge"><Activity size={13} /> Revenue Recovery Intelligence</span>
                <h1>{title}</h1>
                <p>{subtitle}</p>
              </div>
              <div className="causal-chain" aria-label="Causal decision pipeline">
                <span>Failed payment</span>
                <i>→</i>
                <span>Counterfactuals</span>
                <i>→</i>
                <span>Causal effect</span>
                <i>→</i>
                <span>Incremental recovery</span>
                <i>→</i>
                <span>Cost</span>
                <i>→</i>
                <strong>ENIV → action</strong>
              </div>
            </div>
          </section>
        )}

        <div className="content-inner">{content}</div>
      </main>
    </div>
  );
}

export default App;
