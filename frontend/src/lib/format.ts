export const inr = (value: number, digits = 0) =>
  new Intl.NumberFormat('en-IN', {
    style: 'currency',
    currency: 'INR',
    maximumFractionDigits: digits,
    minimumFractionDigits: digits,
  }).format(value);

export const compactInr = (value: number) => {
  const abs = Math.abs(value);
  const sign = value < 0 ? '-' : '';
  if (abs >= 1_00_00_000) return `${sign}₹${(abs / 1_00_00_000).toFixed(2)} Cr`;
  if (abs >= 1_00_000) return `${sign}₹${(abs / 1_00_000).toFixed(2)} L`;
  if (abs >= 1_000) return `${sign}₹${(abs / 1_000).toFixed(1)}k`;
  return inr(value);
};

export const pct = (value: number, digits = 1) => `${(value * 100).toFixed(digits)}%`;

export const num = (value: number, digits = 3) =>
  Number.isFinite(value) ? value.toFixed(digits) : '—';

export const actionLabel = (action: string) => {
  if (action === 'none') return 'No intervention';
  if (action === 'retry') return 'Retry';
  if (action === 'whatsapp') return 'WhatsApp';
  return action;
};

export const policyShort = (name: string) => {
  const lower = name.toLowerCase();
  if (lower.includes('oracle')) return 'Oracle';
  if (lower.includes('increment')) return 'Incrementality-aware';
  if (lower.includes('baseline')) return 'Gross baseline';
  return name;
};
