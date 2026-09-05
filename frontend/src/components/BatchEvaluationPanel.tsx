import type { BatchEvaluation } from '../types';
import { inr, num, pct } from '../lib/format';

type Props = {
  evaluation: BatchEvaluation | null;
  loading: boolean;
  error: string;
  onRun: () => void;
};

function count(distribution: Record<string, number>, action: string) {
  return Number(distribution[action] ?? 0);
}

export function BatchEvaluationPanel({ evaluation, loading, error, onRun }: Props) {
  const isUploaded = evaluation?.evaluation_type === 'uploaded_dataset';
  const baseline = evaluation?.baseline;
  const causapay = evaluation?.causapay;
  const retry    = causapay ? count(causapay.action_distribution, 'retry')    : 0;
  const whatsapp = causapay ? count(causapay.action_distribution, 'whatsapp') : 0;
  const noneAct  = causapay ? count(causapay.action_distribution, 'none')     : 0;
  const acted =
    causapay?.action_distribution.acted_upon ??
    (causapay ? retry + whatsapp : 0);
  const abstained = causapay ? causapay.action_distribution.abstained : undefined;

  const datasetLabel = isUploaded
    ? `${evaluation?.dataset_filename ?? 'Uploaded dataset'} (${evaluation?.dataset_id ?? ''})`
    : 'Synthetic held-out batch';

  return (
    <section className="page-stack">
      <div className="panel evaluation-banner">
        <div>
          <span className="panel-kicker">
            {isUploaded ? 'Uploaded Dataset Evaluation' : 'Synthetic Held-Out Evaluation'}
          </span>
          <h2>Batch Evaluation and Model Validation</h2>
          <p>
            {isUploaded
              ? 'Evaluating the uploaded CSV dataset through the causal decision engine.'
              : 'Run the existing causal evaluation engine on a reproducible held-out synthetic batch.'}
          </p>
        </div>
        <button type="button" className="primary-btn" onClick={onRun} disabled={loading}>
          {loading ? 'Running evaluation…' : 'Run Evaluation'}
        </button>
      </div>

      {error ? <div className="evaluation-error" role="alert">{error}</div> : null}
      {!evaluation ? (
        <section className="panel"><p className="muted">No evaluation has been run yet.</p></section>
      ) : (
        <>
          <section className="panel">
            <div className="panel-head">
              <span className="panel-kicker">
                {isUploaded ? 'Uploaded Dataset Evaluation' : 'Synthetic Held-Out Evaluation'}
              </span>
              <h2>Policy results</h2>
              <p>
                {evaluation.batch_size.toLocaleString('en-IN')} events evaluated
                {' · '}
                <strong>{datasetLabel}</strong>
              </p>
            </div>
            <div className="evaluation-kpi-grid">
              <div><span>Batch size</span><strong>{evaluation.batch_size.toLocaleString('en-IN')}</strong></div>
              {baseline != null ? (
                <>
                  <div><span>Baseline gross recovered</span><strong>{inr(baseline.gross_recovered)}</strong></div>
                  <div><span>Baseline intervention cost</span><strong>{inr(baseline.intervention_cost)}</strong></div>
                  <div><span>Baseline policy value</span><strong>{inr(baseline.policy_value)}</strong></div>
                </>
              ) : null}
              <div><span>CausaPay gross recovered</span><strong>{inr(causapay?.gross_recovered ?? 0)}</strong></div>
              <div><span>CausaPay intervention cost</span><strong>{inr(causapay?.intervention_cost ?? 0)}</strong></div>
              <div><span>CausaPay policy value</span><strong className="accent">{inr(causapay?.policy_value ?? 0)}</strong></div>
              {evaluation.incremental_value_vs_baseline != null ? (
                <div>
                  <span>Incremental value vs baseline</span>
                  <strong className="accent">{inr(evaluation.incremental_value_vs_baseline)}</strong>
                </div>
              ) : null}
              {evaluation.naive_likelihood != null ? (
                <div><span>Naive likelihood policy value</span><strong>{inr(evaluation.naive_likelihood.policy_value)}</strong></div>
              ) : null}
              <div><span>Payments acted upon (retry + WhatsApp)</span><strong>{acted.toLocaleString('en-IN')}</strong></div>
              <div><span>No intervention (ENIV ≤ 0)</span><strong>{noneAct.toLocaleString('en-IN')}</strong></div>
              <div><span>Abstained (high uncertainty → baseline)</span><strong>{abstained == null ? '—' : Number(abstained).toLocaleString('en-IN')}</strong></div>
            </div>
          </section>

          {evaluation.validation != null ? (
            <section className="panel">
              <div className="panel-head">
                <span className="panel-kicker">Synthetic Held-Out Evaluation</span>
                <h2>Model Validation</h2>
                <p>AIPW estimates compared with simulator ground truth on the held-out split.</p>
              </div>
              <div className="table-wrap">
                <table className="policy-table">
                  <thead>
                    <tr>
                      <th>Outcome</th>
                      <th>Ground Truth</th>
                      <th>AIPW Estimate</th>
                      <th>Absolute Error</th>
                    </tr>
                  </thead>
                  <tbody>
                    {(['retry', 'whatsapp'] as const).map((treatment) => {
                      const metric = evaluation.validation![treatment];
                      return (
                        <tr key={treatment}>
                          <td><strong>{treatment === 'retry' ? 'Retry vs None' : 'WhatsApp vs None'}</strong></td>
                          <td>{num(metric.ground_truth_effect, 4)}</td>
                          <td>{num(metric.aipw_estimate, 4)}</td>
                          <td>{num(metric.absolute_error, 4)}</td>
                        </tr>
                      );
                    })}
                  </tbody>
                </table>
              </div>
              <p className="muted evaluation-note">
                CausaPay recovery rate {pct(causapay?.recovery_rate ?? 0)}
                {evaluation.oracle != null
                  ? ` · Oracle policy value ${inr(evaluation.oracle.policy_value)}`
                  : null}
                .
              </p>
              {evaluation.naive_likelihood != null ? <p className="muted evaluation-note">Naive comparator: {evaluation.naive_likelihood.description}</p> : null}
            </section>
          ) : (
            <section className="panel">
              <p className="muted">
                Model validation (AIPW vs ground truth) is not available for uploaded datasets
                because uploaded CSVs do not contain simulator ground-truth outcomes.
              </p>
            </section>
          )}
        </>
      )}
    </section>
  );
}
