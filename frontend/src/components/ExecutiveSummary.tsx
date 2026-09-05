import { TrendingUp, AlertTriangle } from 'lucide-react';
import type { Policy } from '../types';
import { inr, policyShort } from '../lib/format';
import EvidencePanel from './EvidencePanel';

export default function ExecutiveSummary({ results }: { results: Policy[] }) {
  const baseline = results.find((r) => r.policy_name.toLowerCase().includes('baseline'));
  const ours = results.find((r) => r.policy_name.toLowerCase().includes('increment'));
  if (!baseline || !ours) return null;

  const improvement = ours.policy_value - baseline.policy_value;
  const isPositive = improvement > 0;

  return (
    <div className="exec-summary-stack">
      <div className="exec-grid">
        <article className="panel">
          <span className="metric-label">{policyShort(baseline.policy_name)} value</span>
          <strong className="metric-value">{inr(baseline.policy_value)}</strong>
          <p className="metric-hint">Incremental revenue minus intervention cost</p>
        </article>
        <article className="panel highlight">
          <span className="metric-label">{policyShort(ours.policy_name)} value</span>
          <strong className="metric-value accent">{inr(ours.policy_value)}</strong>
          <p className={`delta ${isPositive ? 'up' : 'down'}`}>
            {isPositive ? <TrendingUp size={14} /> : <AlertTriangle size={14} />}
            {isPositive ? '+' : ''}{inr(improvement)} vs baseline
          </p>
        </article>
      </div>
      <EvidencePanel />
    </div>
  );
}
