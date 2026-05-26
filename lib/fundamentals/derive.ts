import type { FundamentalConcepts, FundamentalsView } from "@/lib/types";

function div(a: number | null, b: number | null): number | null {
  if (a === null || b === null || b === 0) return null;
  return a / b;
}

export function deriveMetrics(c: FundamentalConcepts, latestClose: number | null): FundamentalsView {
  const marketCap = latestClose !== null && c.sharesOutstanding !== null
    ? latestClose * c.sharesOutstanding
    : null;
  return {
    marketCap,
    peRatio: div(latestClose, c.eps),
    psRatio: div(marketCap, c.revenue),
    grossMargin: div(c.grossProfit, c.revenue),
    roe: div(c.netIncome, c.equity),
    roa: div(c.netIncome, c.assets),
    operatingIncome: c.operatingIncome,
    currentRatio: div(c.currentAssets, c.currentLiabilities),
    debtToEquity: div(c.liabilities, c.equity),
    assets: c.assets,
    liabilities: c.liabilities,
    equity: c.equity,
    revenue: c.revenue,
    netIncome: c.netIncome,
    eps: c.eps,
  };
}
