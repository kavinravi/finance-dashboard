// Sample standard deviation of daily returns over a rolling window.
export function rollingVolatility(closes: number[], window = 5): (number | null)[] {
  const returns: number[] = [];
  for (let i = 1; i < closes.length; i++) {
    returns.push(closes[i - 1] === 0 ? 0 : (closes[i] - closes[i - 1]) / closes[i - 1]);
  }
  const out: (number | null)[] = closes.map(() => null);
  for (let i = window; i < closes.length; i++) {
    const slice = returns.slice(i - window, i);
    const mean = slice.reduce((a, b) => a + b, 0) / slice.length;
    const variance = slice.reduce((a, b) => a + (b - mean) ** 2, 0) / (slice.length - 1);
    out[i] = Math.sqrt(variance);
  }
  return out;
}
