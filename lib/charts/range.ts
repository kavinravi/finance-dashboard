import type { PriceBar } from "@/lib/types";

export type ChartRange = "1m" | "3m" | "6m" | "ytd" | "1y" | "all" | { from: string; to: string };

export type SliceableIndicators = {
  ma20: (number | null)[];
  ma50: (number | null)[];
  rsi14: (number | null)[];
  macdLine: number[];
  macdSignal: number[];
  macdHistogram: number[];
};

function bounds(range: ChartRange, bars: PriceBar[], today: string): { start: string; end: string } {
  const first = bars[0]?.date ?? today;
  const last = bars[bars.length - 1]?.date ?? today;
  if (typeof range === "object") {
    return { start: range.from < first ? first : range.from, end: range.to > last ? last : range.to };
  }
  if (range === "all") return { start: first, end: last };
  if (range === "ytd") return { start: `${today.slice(0, 4)}-01-01`, end: last };
  const months = { "1m": 1, "3m": 3, "6m": 6, "1y": 12 }[range];
  const d = new Date(`${today}T00:00:00Z`);
  d.setUTCMonth(d.getUTCMonth() - months);
  return { start: d.toISOString().slice(0, 10), end: last };
}

export function rangeStartDate(range: ChartRange, bars: PriceBar[], today: string): string {
  return bounds(range, bars, today).start;
}

export function sliceByRange(
  bars: PriceBar[],
  ind: SliceableIndicators,
  range: ChartRange,
  today: string,
): { bars: PriceBar[]; indicators: SliceableIndicators } {
  if (bars.length === 0) return { bars, indicators: ind };
  const { start, end } = bounds(range, bars, today);
  let lo = bars.findIndex((b) => b.date >= start);
  if (lo === -1) lo = bars.length;
  let hiIdx = bars.length - 1;
  while (hiIdx >= 0 && bars[hiIdx].date > end) hiIdx--;
  const hi = hiIdx + 1;
  return {
    bars: bars.slice(lo, hi),
    indicators: {
      ma20: ind.ma20.slice(lo, hi), ma50: ind.ma50.slice(lo, hi), rsi14: ind.rsi14.slice(lo, hi),
      macdLine: ind.macdLine.slice(lo, hi), macdSignal: ind.macdSignal.slice(lo, hi), macdHistogram: ind.macdHistogram.slice(lo, hi),
    },
  };
}
