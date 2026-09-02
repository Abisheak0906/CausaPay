const safeNumber = (value: unknown) => {
  const numeric = Number(value);
  return Number.isFinite(numeric) ? numeric : 0;
};

export const inr = (value: number, digits = 0) =>
  new Intl.NumberFormat('en-IN', {
    style: 'currency',
    currency: 'INR',
    maximumFractionDigits: digits,
    minimumFractionDigits: digits,
  }).format(safeNumber(value));

export const compactInr = (value: number) => {
  const numeric = safeNumber(value);
  const abs = Math.abs(numeric);
  const sign = numeric < 0 ? '-' : '';
  if (abs >= 1_00_00_000) return `${sign}₹${(abs / 1_00_00_000).toFixed(2)} Cr`;
  if (abs >= 1_00_000) return `${sign}₹${(abs / 1_00_000).toFixed(2)} L`;
  if (abs >= 1_000) return `${sign}₹${(abs / 1_000).toFixed(1)}k`;
  return inr(numeric);
};

export const pct = (value: number, digits = 1) => `${(safeNumber(value) * 100).toFixed(digits)}%`;

export const num = (value: number, digits = 3) =>
  Number.isFinite(Number(value)) ? Number(value).toFixed(digits) : '—';

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
