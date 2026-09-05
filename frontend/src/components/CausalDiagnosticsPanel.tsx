import type { ModelDiagnostics } from '../types';
import { num, pct } from '../lib/format';

export function CausalDiagnosticsPanel({ data }: { data: ModelDiagnostics | null }) {
  if (!data) return <section className="panel"><p className="muted">Live overlap and uncertainty diagnostics are unavailable.</p></section>;
  const uncertainty = data.uncertainty;
  return <section className="page-stack">
    <section className="panel">
      <div className="panel-head"><span className="panel-kicker">Positivity diagnostic</span><h2>Observed propensity overlap</h2><p>{data.overlap.interpretation}</p></div>
      <div className="table-wrap"><table className="policy-table"><thead><tr><th>Assigned action</th><th>Historical rows</th><th>Mean propensity</th><th>P05</th><th>Median</th><th>P95</th></tr></thead><tbody>{data.overlap.groups.map((group) => <tr key={group.treatment}><td><strong>{group.treatment}</strong></td><td>{group.assigned_count.toLocaleString()}</td><td>{num(group.all_rows.mean, 3)}</td><td>{num(group.all_rows.p05, 3)}</td><td>{num(group.all_rows.median, 3)}</td><td>{num(group.all_rows.p95, 3)}</td></tr>)}</tbody></table></div>
      <p className="muted evaluation-note">Central 90% common interval: {num(data.overlap.central_90_percent_common_interval.lower, 3)}–{num(data.overlap.central_90_percent_common_interval.upper, 3)}. {data.overlap.limitation}</p>
    </section>
    <section className="panel">
      <div className="panel-head">
        <span className="panel-kicker">Abstention diagnostic</span>
        <h2>Uncertainty distribution and abstention threshold</h2>
        <p>Threshold: <strong>{num(uncertainty.threshold, 2)}</strong> · {pct(uncertainty.abstention_proxy_rate)} of held-out rows exceed it for at least one intervention.</p>
      </div>
      <div className="table-wrap">
        <table className="policy-table">
          <thead>
            <tr><th>Action</th><th>Mean</th><th>P05</th><th>Median</th><th>P95</th><th>Max</th></tr>
          </thead>
          <tbody>
            {Object.entries(uncertainty.distributions).map(([name, d]) => (
              <tr key={name}>
                <td><strong>{name}</strong></td>
                <td>{num(d.mean, 3)}</td>
                <td>{num(d.p05, 3)}</td>
                <td>{num(d.median, 3)}</td>
                <td>{num(d.p95, 3)}</td>
                <td>{num(d.max, 3)}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      <p className="muted evaluation-note">{uncertainty.limitation}</p>
    </section>

    <section className="panel" style={{ borderLeft: '4px solid #f59e0b' }}>
      <div className="panel-head">
        <span className="panel-kicker" style={{ color: '#f59e0b' }}>Critical Methodology Limitations</span>
        <h2>Model Assumptions & Deployment Caveats</h2>
        <p>Transparent disclosure of econometric and statistical limits. These boundaries are essential for scientific integrity.</p>
      </div>
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(280px, 1fr))', gap: '16px', marginTop: '12px' }}>
        <div style={{ padding: '12px', background: 'rgba(255,255,255,0.03)', borderRadius: '8px', border: '1px solid var(--line)' }}>
          <strong style={{ display: 'block', marginBottom: '4px', color: '#f59e0b' }}>1. Synthetic Evaluation ≠ Real-World Proof</strong>
          <p className="muted" style={{ margin: 0, fontSize: '0.85rem', lineHeight: '1.4' }}>
            Validation relies on synthetic potential outcomes where true counterfactuals are generated. This tests algorithmic consistency, not verified real-world revenue lift.
          </p>
        </div>
        <div style={{ padding: '12px', background: 'rgba(255,255,255,0.03)', borderRadius: '8px', border: '1px solid var(--line)' }}>
          <strong style={{ display: 'block', marginBottom: '4px', color: '#f59e0b' }}>2. Identification Assumptions</strong>
          <p className="muted" style={{ margin: 0, fontSize: '0.85rem', lineHeight: '1.4' }}>
            Causal estimates require conditional unconfoundedness (no unmeasured confounders) and positivity. In real-world payment flows, hidden variables (e.g. bank outages) can bias estimates.
          </p>
        </div>
        <div style={{ padding: '12px', background: 'rgba(255,255,255,0.03)', borderRadius: '8px', border: '1px solid var(--line)' }}>
          <strong style={{ display: 'block', marginBottom: '4px', color: '#f59e0b' }}>3. Positivity & Overlap Boundaries</strong>
          <p className="muted" style={{ margin: 0, fontSize: '0.85rem', lineHeight: '1.4' }}>
            Regions with limited overlap contain fewer comparable examples across actions. Positivity overlap does NOT prove causal validity—it only checks that extreme inverse propensity weights do not explode.
          </p>
        </div>
        <div style={{ padding: '12px', background: 'rgba(255,255,255,0.03)', borderRadius: '8px', border: '1px solid var(--line)' }}>
          <strong style={{ display: 'block', marginBottom: '4px', color: '#f59e0b' }}>4. Uncertainty Heuristic ≠ Confidence Interval</strong>
          <p className="muted" style={{ margin: 0, fontSize: '0.85rem', lineHeight: '1.4' }}>
            Uncertainty is measured as random-forest tree dispersion across final-stage learners. It is an operational abstention heuristic, NOT a frequentist or Bayesian calibrated confidence interval.
          </p>
        </div>
        <div style={{ padding: '12px', background: 'rgba(255,255,255,0.03)', borderRadius: '8px', border: '1px solid var(--line)' }}>
          <strong style={{ display: 'block', marginBottom: '4px', color: '#f59e0b' }}>5. Production Deployment Requirement</strong>
          <p className="muted" style={{ margin: 0, fontSize: '0.85rem', lineHeight: '1.4' }}>
            Prior to full production deployment, any causal recovery engine must be validated via randomized A/B holdouts or carefully designed phased rollouts to measure true incremental revenue lift.
          </p>
        </div>
      </div>
    </section>
  </section>;
}
