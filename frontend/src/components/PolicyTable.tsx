import type { Policy } from '../types';
import { inr, pct, policyShort } from '../lib/format';

export function PolicyTable({ policies, compact }: { policies: Policy[]; compact?: boolean }) {
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
              <th>True incremental</th>
              <th>Intervention cost</th>
              <th>Policy value</th>
              <th>Recovery rate</th>
            </tr>
          </thead>
          <tbody>
            {policies.map((policy) => {
              const focus = policy.policy_name.toLowerCase().includes('increment');
              return (
                <tr key={policy.policy_name} className={focus ? 'focus' : ''}>
                  <td>
                    <strong>{policyShort(policy.policy_name)}</strong>
                    <small>{policy.policy_name}</small>
                  </td>
                  <td>{inr(policy.gross_recovered)}</td>
                  <td>{inr(policy.true_incremental_recovered)}</td>
                  <td>{inr(policy.intervention_cost, 2)}</td>
                  <td>{inr(policy.policy_value)}</td>
                  <td>{pct(policy.recovery_rate)}</td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
    </section>
  );
}
