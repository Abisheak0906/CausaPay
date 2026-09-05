import type { Policy } from '../types';
import { inr, pct, policyShort } from '../lib/format';

export function PolicyTable({ policies, compact }: { policies: Policy[]; compact?: boolean }) {
  // Determine whether any policy carries estimated (uploaded) vs true (synthetic ground-truth) incremental
  const hasEstimated = policies.some((p) => p.estimated_incremental_recovered != null);
  const incrementalLabel = hasEstimated ? 'Est. incremental' : 'True incremental';
  const incrementalHint = hasEstimated
    ? 'AIPW model estimate — no simulator ground truth for uploaded datasets'
    : 'Realized from simulator potential outcomes (synthetic evaluation only)';

  return (
    <section className="panel">
      <div className="panel-head">
        <span className="panel-kicker">Held-out evaluation</span>
        <h2>{compact ? 'Policy scorecard' : 'Policy comparison'}</h2>
      </div>
      <div className="table-wrap">
        <table className="policy-table">
          <thead>
            <tr>
              <th>Policy</th>
              <th>Gross recovered</th>
              <th title={incrementalHint}>{incrementalLabel}</th>
              <th>Intervention cost</th>
              <th>Policy value</th>
              <th>Recovery rate</th>
            </tr>
          </thead>
          <tbody>
            {policies.map((policy) => {
              const focus = policy.policy_name.toLowerCase().includes('increment');
              const incrementalValue = policy.estimated_incremental_recovered ?? policy.true_incremental_recovered;
              return (
                <tr key={policy.policy_name} className={focus ? 'focus' : ''}>
                  <td>
                    <strong>{policyShort(policy.policy_name)}</strong>
                    <small>{policy.policy_name}</small>
                  </td>
                  <td>{inr(policy.gross_recovered)}</td>
                  <td>{inr(incrementalValue)}</td>
                  <td>{inr(policy.intervention_cost, 2)}</td>
                  <td>{inr(policy.policy_value)}</td>
                  <td>{pct(policy.recovery_rate)}</td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
      {hasEstimated && (
        <p className="muted" style={{ fontSize: '0.75rem', marginTop: '0.5rem' }}>
          ⚠ Incremental column shows AIPW model estimates (no ground truth available for uploaded datasets).
        </p>
      )}
    </section>
  );
}
