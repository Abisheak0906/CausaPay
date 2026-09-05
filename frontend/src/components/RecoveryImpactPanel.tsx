import type { EventItem, Policy } from '../types';
import { inr } from '../lib/format';

type Props = { events: EventItem[]; policies: Policy[]; uploaded: boolean };

export function RecoveryImpactPanel({ events, policies, uploaded }: Props) {
  const baseline = policies.find((policy) => policy.policy_name.toLowerCase().includes('baseline'));
  const causa = policies.find((policy) => policy.policy_name.toLowerCase().includes('increment')) ?? policies[0];
  const impact = uploaded
    ? events.reduce((result, event) => {
        const amount = Number(event.amount) || 0;
        const baselineAction = (event.baseline_action ?? 'none').toLowerCase();
        const action = event.is_abstain ? 'none' : (event.recommended_action ?? 'none').toLowerCase();
        const probability = (name: string) => name === 'retry' ? Number(event.prob_retry ?? 0) : name === 'whatsapp' ? Number(event.prob_whatsapp ?? 0) : Number(event.prob_none ?? 0);
        result.baseline += probability(baselineAction) * amount;
        result.recovery += probability(action) * amount;
        result.incremental += (action === 'retry' ? Number(event.inc_prob_retry ?? 0) : action === 'whatsapp' ? Number(event.inc_prob_whatsapp ?? 0) : 0) * amount;
        result.cost += action === 'retry' ? 2 : action === 'whatsapp' ? 15 : 0;
        result.acted += Number(action === 'retry' || action === 'whatsapp');
        result.noAction += Number(action === 'none' || Boolean(event.is_abstain));
        return result;
      }, { baseline: 0, recovery: 0, incremental: 0, cost: 0, acted: 0, noAction: 0 })
    : {
        baseline: baseline?.gross_recovered ?? 0,
        recovery: causa?.gross_recovered ?? 0,
        incremental: causa?.estimated_incremental_recovered ?? causa?.true_incremental_recovered ?? 0,
        cost: causa?.intervention_cost ?? 0,
        acted: events.filter((event) => ['retry', 'whatsapp'].includes((event.recommended_action ?? '').toLowerCase()) && !event.is_abstain).length,
        noAction: events.filter((event) => event.is_abstain || (event.recommended_action ?? 'none').toLowerCase() === 'none').length,
      };

  return (
    <section className="panel impact-panel">
      <div className="panel-head">
        <span className="panel-kicker">Recovery impact</span>
        <h2>Baseline Recovery vs CausaPay Estimated Recovery</h2>
        <p>Estimated from the currently active dataset and the decisions produced for its payment events.</p>
      </div>
      <div className="impact-compare">
        <div className="impact-side baseline-impact"><span>Baseline Recovery</span><strong>{inr(impact.baseline)}</strong></div>
        <div className="impact-arrow">→</div>
        <div className="impact-side causa-impact"><span>CausaPay Estimated Recovery</span><strong>{inr(impact.recovery)}</strong></div>
      </div>
      <div className="impact-grid">
        <div><span>Incremental Recovery</span><strong>{inr(impact.incremental)}</strong></div>
        <div><span>Intervention Cost</span><strong>{inr(impact.cost)}</strong></div>
        <div><span>Net Incremental Value (ENIV)</span><strong>{inr(impact.incremental - impact.cost)}</strong></div>
        <div><span>Payments acted upon</span><strong>{impact.acted.toLocaleString('en-IN')}</strong></div>
        <div><span>Abstained or no action</span><strong>{impact.noAction.toLocaleString('en-IN')}</strong></div>
      </div>
    </section>
  );
}
