import { Bar, BarChart, CartesianGrid, Legend, ResponsiveContainer, Tooltip, XAxis, YAxis } from 'recharts';
import type { Policy } from '../types';
import { inr, policyShort } from '../lib/format';

type Props = { policies: Policy[]; compact?: boolean; presentation?: boolean };

export function CannibalChart({ policies, compact, presentation }: Props) {
  const data = policies.map((policy) => ({
    name: policyShort(policy.policy_name),
    Gross: policy.gross_recovered,
    Incremental: policy.true_incremental_recovered,
    Cost: policy.intervention_cost,
    Value: policy.policy_value,
  }));

  const chart = (
    <div className={compact ? 'chart-frame compact' : 'chart-frame'}>
      <ResponsiveContainer width="100%" height="100%">
        <BarChart data={data} margin={{ top: 8, right: 8, left: 8, bottom: 0 }} barGap={6}>
          <CartesianGrid stroke="var(--line)" vertical={false} />
          <XAxis dataKey="name" tick={{ fill: 'var(--muted)', fontSize: 12 }} axisLine={false} tickLine={false} />
          <YAxis tick={{ fill: 'var(--muted)', fontSize: 11 }} axisLine={false} tickLine={false} tickFormatter={(v: number) => `₹${(v / 1000).toFixed(0)}k`} />
          <Tooltip
            cursor={{ fill: 'var(--accent-soft)' }}
            content={({ active, payload, label }) => {
              if (!active || !payload?.length) return null;
              return (
                <div className="chart-tooltip">
                  <strong>{label}</strong>
                  {payload.map((item) => (
                    <p key={String(item.dataKey)}>
                      <span style={{ color: String(item.color) }}>{item.name}</span>
                      <b>{inr(Number(item.value))}</b>
                    </p>
                  ))}
                </div>
              );
            }}
          />
          {!compact && <Legend wrapperStyle={{ color: 'var(--muted)', fontSize: 12, paddingTop: 12 }} iconType="square" />}
          <Bar dataKey="Gross" name="Gross recovery" fill="var(--line-strong)" radius={[4, 4, 0, 0]} />
          <Bar dataKey="Incremental" name="True incremental" fill="var(--accent)" radius={[4, 4, 0, 0]} />
          <Bar dataKey="Value" name="Policy value" fill="var(--accent-2)" radius={[4, 4, 0, 0]} />
        </BarChart>
      </ResponsiveContainer>
    </div>
  );

  return (
    <section className={`panel ${presentation ? 'presentation' : ''}`}>
      <div className="panel-head">
        <span className="panel-kicker">Attribution</span>
        <h2>Gross recovery vs incremental recovery</h2>
        <p>Gross includes self-cures. Incremental is the revenue that would not have occurred without the intervention.</p>
      </div>
      {chart}
    </section>
  );
}
