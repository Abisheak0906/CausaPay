import { actionLabel } from '../lib/format';

export function ActionDistribution({ distribution }: { distribution: Record<string, number> }) {
  const entries = Object.entries(distribution);
  const total = entries.reduce((sum, [, count]) => sum + Number(count), 0) || 1;
  const order = ['none', 'retry', 'whatsapp', 'abstained'];
  const sorted = [...entries].sort((a, b) => order.indexOf(a[0]) - order.indexOf(b[0]));

  return (
    <section className="panel">
      <div className="panel-head">
        <span className="panel-kicker">Intervention mix</span>
        <h2>Recommended actions</h2>
      </div>
      <div className="mix-stack">
        {sorted.map(([action, count]) => (
          <div className="mix-row" key={action}>
            <div>
              <strong>{actionLabel(action)}</strong>
              <span>{Number(count).toLocaleString('en-IN')} events</span>
            </div>
            <div className="bar-track">
              <div className={`bar mix-${action}`} style={{ width: `${(Number(count) / total) * 100}%` }} />
            </div>
            <em>{((Number(count) / total) * 100).toFixed(1)}%</em>
          </div>
        ))}
      </div>
    </section>
  );
}
