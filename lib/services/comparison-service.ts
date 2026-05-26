import { getTickerData, type Range } from "./price-service";
import { rollingVolatility } from "@/lib/indicators";
import type { PriceBar } from "@/lib/types";

export type CompareSeries = { ticker: string; normalized: number[]; volatility: number | null; maxDrawdown: number };
export type ComparisonResult = {
  dates: string[];
  primary: CompareSeries;
  comparison: CompareSeries;
  relativeReturn: number | null;
};

function closeByDate(bars: PriceBar[]): Map<string, number> {
  return new Map(bars.map((b) => [b.date, b.close]));
}

function maxDrawdown(values: number[]): number {
  let peak = -Infinity, mdd = 0;
  for (const v of values) {
    peak = Math.max(peak, v);
    if (peak > 0) mdd = Math.min(mdd, v / peak - 1);
  }
  return mdd;
}

export async function compareTickers(primary: string, comparison: string, range: Range): Promise<ComparisonResult> {
  const [a, b] = await Promise.all([getTickerData(primary, range), getTickerData(comparison, range)]);
  const ma = closeByDate(a.bars), mb = closeByDate(b.bars);
  const dates = [...ma.keys()].filter((d) => mb.has(d)).sort();

  const aCloses = dates.map((d) => ma.get(d)!);
  const bCloses = dates.map((d) => mb.get(d)!);

  const norm = (xs: number[]) => (xs.length ? xs.map((x) => (x / xs[0]) * 100) : []);
  const lastVol = (xs: number[]) => rollingVolatility(xs, 5).at(-1) ?? null;
  const windowReturn = (xs: number[]) => (xs.length >= 2 ? xs.at(-1)! / xs[0] - 1 : null);

  const aRet = windowReturn(aCloses);
  const bRet = windowReturn(bCloses);

  return {
    dates,
    primary: { ticker: a.ticker, normalized: norm(aCloses), volatility: lastVol(aCloses), maxDrawdown: maxDrawdown(aCloses) },
    comparison: { ticker: b.ticker, normalized: norm(bCloses), volatility: lastVol(bCloses), maxDrawdown: maxDrawdown(bCloses) },
    relativeReturn: aRet !== null && bRet !== null ? aRet - bRet : null,
  };
}
