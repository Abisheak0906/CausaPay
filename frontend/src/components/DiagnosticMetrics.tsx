import { Line, LineChart, ResponsiveContainer, Tooltip, XAxis, YAxis, CartesianGrid, ScatterChart, Scatter, Legend } from 'recharts';
import type { Diagnostics } from '../types';
import { inr, num } from '../lib/format';

export function DiagnosticMetrics({ diagnostics, activeDataset }: { diagnostics: Diagnostics | null; activeDataset: string }) {
  if (!diagnostics) {
    return (
      <section className="panel">
        <p className="muted">
          {activeDataset === 'demo'
            ? 'No diagnostic artifact was returned by the API.'
            : 'Diagnostics are currently unavailable for uploaded datasets. Causal diagnostics require a validated synthetic ground-truth or a large-scale randomized trial to be methodologically sound.'}
        </p>
      </section>
    );
  }

  const outcomes = Object.entries(diagnostics.outcome_model_validation ?? {});
  const cate = Object.entries(diagnostics.true_vs_estimated_cate ?? {});
  const learners = Object.entries(diagnostics.final_stage_learner_comparison ?? {});
  const curve = Object.entries(diagnostics.learning_curve ?? {}).map(([n, row]) => ({
    n: Number(n),
    AIPW: row.AIPW_policy_value,
    Baseline: row.Baseline_policy_value,
    Oracle: row.Oracle_policy_value,
    CATE_MAE: row.CATE_MAE,
  }));
  const diagnosis = diagnostics.final_diagnosis;
  const audit = diagnostics.cross_fitting_audit;

  return (
    <section className="page-stack">
      <div className="limitation-banner">
        <span>Honest limitations</span>
        <p>
          These diagnostics are evaluation artifacts from the causal engine. Weak outcome fit, near-zero CATE rank correlation,
          and an unverified cross-fitting audit are shown as-is — they are reasons to treat individual CATE estimates cautiously.
        </p>
        {diagnosis ? <p className="diagnosis-line">Dominant issue {diagnosis.dominant_issue}: {diagnosis.reasoning}</p> : null}
      </div>

      <div className="diagnostic-grid">
        <article className="panel">
          <div className="panel-head">
            <span className="panel-kicker">Identifiability hygiene</span>
            <h2>Cross-fitting audit</h2>
          </div>
          <p className="status-pill warn">{audit?.passed == null ? 'Inconclusive' : audit.passed ? 'Passed' : 'Failed'}</p>
          <p>{audit?.message ?? 'No audit message.'}</p>
        </article>
        <article className="panel">
          <div className="panel-head">
            <span className="panel-kicker">Outcome models</span>
            <h2>Out-of-sample fit</h2>
          </div>
          <div className="mini-table">
            {outcomes.map(([arm, metrics]) => (
              <div key={arm}>
                <strong>{arm}</strong>
                <span>R² {num(metrics.r2, 3)}</span>
                <span>MAE {num(metrics.mae, 3)}</span>
              </div>
            ))}
          </div>
          <p className="muted">R² around 0.11–0.16 means the outcome models explain little of recovery variation.</p>
        </article>
        <article className="panel">
          <div className="panel-head">
            <span className="panel-kicker">Final stage</span>
            <h2>Learner comparison</h2>
          </div>
          <div className="mini-table">
            {learners.map(([name, metrics]) => (
              <div key={name}>
                <strong>{name}{diagnostics.best_learner === name ? ' · best' : ''}</strong>
                <span>MAE {num(metrics.mae, 3)}</span>
                <span>RMSE {num(metrics.rmse, 3)}</span>
              </div>
            ))}
          </div>
        </article>
      </div>

      <section className="panel">
        <div className="panel-head">
          <span className="panel-kicker">CATE quality</span>
          <h2>True vs estimated treatment effect</h2>
          <p>Available because the environment is synthetic. In production this comparison would be unobservable.</p>
        </div>
        <div className="cate-row">
          {cate.map(([arm, block]) => (
            <div key={arm} className="cate-card">
              <h3>{arm}</h3>
              <p>MAE {num(block.mae, 3)} · RMSE {num(block.rmse, 3)} · Corr {num(block.corr, 3)}</p>
              <div className="chart-frame compact">
                <ResponsiveContainer width="100%" height="100%">
                  <ScatterChart margin={{ top: 8, right: 8, bottom: 8, left: 8 }}>
                    <CartesianGrid stroke="rgba(148,163,184,0.12)" />
                    <XAxis type="number" dataKey="mean_est" name="Estimated" tick={{ fill: '#93a4bb', fontSize: 11 }} />
                    <YAxis type="number" dataKey="mean_true" name="True" tick={{ fill: '#93a4bb', fontSize: 11 }} />
                    <Tooltip
                      content={({ payload }) => {
                        const point = payload?.[0]?.payload as { mean_est?: number; mean_true?: number } | undefined;
                        if (!point) return null;
                        return (
                          <div className="chart-tooltip">
                            <p>Estimated <b>{num(point.mean_est ?? 0, 3)}</b></p>
                            <p>True <b>{num(point.mean_true ?? 0, 3)}</b></p>
                          </div>
                        );
                      }}
                    />
                    <Scatter
                      data={Object.values(block.calibration)}
                      fill="#3ecfad"
                    />
                  </ScatterChart>
                </ResponsiveContainer>
              </div>
            </div>
          ))}
        </div>
      </section>

      <section className="panel">
        <div className="panel-head">
          <span className="panel-kicker">Sample size</span>
          <h2>Learning curve — policy value</h2>
        </div>
        <div className="chart-frame">
          <ResponsiveContainer width="100%" height="100%">
            <LineChart data={curve} margin={{ top: 8, right: 16, left: 8, bottom: 0 }}>
              <CartesianGrid stroke="rgba(148,163,184,0.12)" vertical={false} />
              <XAxis dataKey="n" tick={{ fill: '#93a4bb', fontSize: 12 }} />
              <YAxis tick={{ fill: '#93a4bb', fontSize: 11 }} tickFormatter={(v: number) => inr(v)} width={88} />
              <Legend />
              <Tooltip
                content={({ active, payload, label }) => {
                  if (!active || !payload?.length) return null;
                  return (
                    <div className="chart-tooltip">
                      <strong>{label} events</strong>
                      {payload.map((item) => (
                        <p key={String(item.dataKey)}><span>{item.name}</span><b>{inr(Number(item.value))}</b></p>
                      ))}
                    </div>
                  );
                }}
              />
              <Line type="monotone" dataKey="AIPW" stroke="#3ecfad" strokeWidth={2} dot={false} />
              <Line type="monotone" dataKey="Baseline" stroke="#8ea0b8" strokeWidth={2} dot={false} />
              <Line type="monotone" dataKey="Oracle" stroke="#7aa2ff" strokeWidth={2} dot={false} />
            </LineChart>
          </ResponsiveContainer>
        </div>
      </section>

      <section className="panel">
        <div className="panel-head">
          <span className="panel-kicker">Overlap</span>
          <h2>Propensity diagnostics</h2>
        </div>
        <div className="table-wrap">
          <table className="policy-table">
            <thead>
              <tr>
                <th>Treatment</th>
                <th>% propensity &lt; 0.05</th>
                <th>% propensity &gt; 0.95</th>
                <th>ESS</th>
              </tr>
            </thead>
            <tbody>
              {(diagnostics.propensity_diagnostics ?? []).map((row) => (
                <tr key={row.treatment}>
                  <td><strong>{row.treatment}</strong></td>
                  <td>{num(Number(row['pct_below_0.05']) * 100, 1)}%</td>
                  <td>{num(Number(row['pct_above_0.95']) * 100, 1)}%</td>
                  <td>{num(Number(row.ESS), 1)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </section>
    </section>
  );
}
