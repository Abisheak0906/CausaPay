import { compactInr, inr } from '../lib/format';

type Props = {
  label: string;
  value: number;
  currency?: boolean;
  accent?: boolean;
  primary?: boolean;
  hint?: string;
};

export function OverviewCard({ label, value, currency, accent, primary, hint }: Props) {
  const formatted = currency ? (primary ? inr(value) : compactInr(value)) : value.toLocaleString('en-IN');
  return (
    <article className={`metric-card ${primary ? 'primary' : ''} ${accent ? 'accent' : ''}`}>
      <span className="metric-label">{label}</span>
      <strong className={`metric-value ${accent ? 'accent' : ''}`}>{formatted}</strong>
      {hint ? <p className="metric-hint">{hint}</p> : null}
    </article>
  );
}
