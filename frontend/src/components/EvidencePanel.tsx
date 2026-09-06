import { CheckCircle2, AlertCircle } from 'lucide-react';

export default function EvidencePanel() {
  const supports = [
    { label: 'Causal Framework', description: 'Implemented a cross-fitted AIPW estimator for incremental recovery.' },
    { label: 'Economic Logic', description: ' incorporated intervention costs and expected net incremental value (ENIV).' },
    { label: 'Consent Safety', description: 'Hard-enforced WhatsApp opt-in constraints across all policy tiers.' },
    { label: 'Robust Evaluation', description: 'Evaluated via repeated-seed studies (0-9) on held-out synthetic data.' },
    { label: 'Fair Comparators', description: 'Matched intervention volume for naive likelihood targeting comparison.' },
    { label: 'Overlap Diagnostics', description: 'Exposed propensity overlap and positivity checks for transparency.' },
  ];

  const doesNotSupport = [
    { label: 'Real-world Uplift', description: 'Current results are based on synthetic simulations, not production A/B tests.' },
    { label: 'Universal Superiority', description: 'CausaPay beats naive targeting but underperforms the gross-recovery baseline.' },
    { label: 'Calibrated Uncertainty', description: 'Abstention is based on a heuristic proxy, not a statistical confidence interval.' },
    { label: 'Individual Accuracy', description: 'CATE rankings show low correlation with truth; individual predictions are noisy.' },
  ];

  return (
    <div className="evidence-container">
      <div className="evidence-grid">
        <div className="evidence-column">
          <div className="evidence-header">
            <CheckCircle2 size={20} className="text-green" />
            <h2>What the evidence supports</h2>
          </div>
          <div className="evidence-list">
            {supports.map((item, i) => (
              <div key={i} className="evidence-item">
                <strong className="evidence-label">{item.label}</strong>
                <span className="evidence-desc">{item.description}</span>
              </div>
            ))}
          </div>
        </div>

        <div className="evidence-column">
          <div className="evidence-header">
            <AlertCircle size={20} className="text-warn" />
            <h2>What it does not yet support</h2>
          </div>
          <div className="evidence-list">
            {doesNotSupport.map((item, i) => (
              <div key={i} className="evidence-item">
                <strong className="evidence-label">{item.label}</strong>
                <span className="evidence-desc">{item.description}</span>
              </div>
            ))}
          </div>
        </div>
      </div>
    </div>
  );
}
