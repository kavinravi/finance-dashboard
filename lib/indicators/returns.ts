import type { PriceBar, PeriodReturns } from "@/lib/types";

function pct(curr: number, prev: number): number | null {
  if (prev === 0) return null;
  return (curr - prev) / prev;
}

function shiftMonths(iso: string, months: number): string {
  const d = new Date(iso + "T00:00:00Z");
  d.setUTCMonth(d.getUTCMonth() - months);
  return d.toISOString().slice(0, 10);
}

function closeOnOrBefore(bars: PriceBar[], targetIso: string): number | null {
  for (let i = bars.length - 1; i >= 0; i--) {
    if (bars[i].date <= targetIso) return bars[i].close;
  }
  return null;
}

export function computeReturns(bars: PriceBar[]): PeriodReturns {
  const empty: PeriodReturns = {
    oneDay: null, fiveDay: null, oneMonth: null, threeMonth: null,
    sixMonth: null, ytd: null, oneYear: null,
  };
  if (bars.length === 0) return empty;

  const last = bars[bars.length - 1];
  const latest = last.close;
  const latestDate = last.date;
  const latestYear = latestDate.slice(0, 4);

  const byCount = (n: number) =>
    bars.length > n ? pct(latest, bars[bars.length - 1 - n].close) : null;

  const byDate = (months: number) => {
    const ref = closeOnOrBefore(bars, shiftMonths(latestDate, months));
    return ref === null ? null : pct(latest, ref);
  };

  const ytdRef = closeOnOrBefore(bars, `${Number(latestYear) - 1}-12-31`);

  return {
    oneDay: byCount(1),
    fiveDay: byCount(5),
    oneMonth: byDate(1),
    threeMonth: byDate(3),
    sixMonth: byDate(6),
    ytd: ytdRef === null ? null : pct(latest, ytdRef),
    oneYear: byDate(12),
  };
}
