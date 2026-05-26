export function formatPercent(v: number | null): string {
  if (v === null || Number.isNaN(v)) return "—";
  const pct = v * 100;
  return `${pct >= 0 ? "+" : ""}${pct.toFixed(2)}%`;
}
export function formatPrice(v: number | null): string {
  if (v === null || Number.isNaN(v)) return "—";
  return v.toFixed(2);
}
export function formatLargeCurrency(v: number | null): string {
  if (v === null || Number.isNaN(v)) return "N/A";
  const abs = Math.abs(v);
  const sign = v < 0 ? "-" : "";
  if (abs >= 1e12) return `${sign}$${(abs / 1e12).toFixed(2)}T`;
  if (abs >= 1e9) return `${sign}$${(abs / 1e9).toFixed(2)}B`;
  if (abs >= 1e6) return `${sign}$${(abs / 1e6).toFixed(2)}M`;
  return `${sign}$${abs.toLocaleString("en-US")}`;
}
export function formatMultiple(v: number | null): string {
  if (v === null || Number.isNaN(v)) return "N/A";
  return `${v.toFixed(2)}×`;
}
export function formatRatioPercent(v: number | null): string {
  if (v === null || Number.isNaN(v)) return "N/A";
  return `${(v * 100).toFixed(1)}%`;
}
