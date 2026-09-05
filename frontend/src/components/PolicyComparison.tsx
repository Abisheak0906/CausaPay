import { Bar, BarChart, CartesianGrid, Cell, Legend, ResponsiveContainer, Tooltip, XAxis, YAxis } from 'recharts';
import type { Policy } from '../types';
import { inr } from '../lib/format';

export default function PolicyComparison({ results }: { results: Policy[] }) {
  const isEstimated = results.some((r) => r.estimated_incremental_recovered != null);
  const incrementalKey = isEstimated ? 'Est. Incremental' : 'True Incremental';
  const data = results.map((r) => ({
    name: r.policy_name,
    'Gross Recovery': r.gross_recovered,
    [incrementalKey]: r.estimated_incremental_recovered ?? r.true_incremental_recovered,
    'Net Policy Value': r.policy_value,
    color: r.policy_name.toLowerCase().includes('baseline')
      ? '#8ea0b8'
      : r.policy_name.toLowerCase().includes('increment')
        ? '#3ecfad'
        : '#7aa2ff',
  }));

  return (
    <div className="panel">
      <div className="chart-frame">
        <ResponsiveContainer width="100%" height="100%">
          <BarChart data={data} margin={{ top: 5, right: 20, left: 8, bottom: 5 }} barGap={8}>
            <CartesianGrid strokeDasharray="3 3" stroke="#1d304a" vertical={false} />
            <XAxis dataKey="name" stroke="#8ea0b8" tick={{ fill: '#8ea0b8', fontSize: 12 }} />
            <YAxis stroke="#8ea0b8" tick={{ fill: '#8ea0b8' }} tickFormatter={(v: number) => `₹${v / 1000}k`} />
            <Tooltip
              cursor={{ fill: '#112238' }}
              content={({ active, payload, label }) => {
                if (!active || !payload?.length) return null;
                return (
                  <div className="chart-tooltip">
                    <strong>{String(label)}</strong>
                    {payload.map((p) => (
                      <p key={String(p.dataKey)}>
                        <span style={{ color: String(p.color) }}>{p.name}</span>
                        <b>{inr(Number(p.value))}</b>
                      </p>
                    ))}
                  </div>
                );
              }}
            />
            <Legend wrapperStyle={{ paddingTop: '20px' }} />
            <Bar dataKey="Gross Recovery" fill="#3d5270" radius={[4, 4, 0, 0]} />
            <Bar dataKey={incrementalKey} fill="#3ecfad" radius={[4, 4, 0, 0]} />
            <Bar dataKey="Net Policy Value" radius={[4, 4, 0, 0]}>
              {data.map((entry) => (
                <Cell key={entry.name} fill={entry.color} />
              ))}
            </Bar>
          </BarChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
}
