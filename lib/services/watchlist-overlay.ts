export type OverlayInput = { ticker: string; bars: { date: string; close: number }[] };
export type Overlay = { dates: string[]; series: { ticker: string; normalized: number[] }[] };

export function buildOverlay(items: OverlayInput[]): Overlay {
  const valid = items.filter((it) => it.bars.length > 0);
  if (valid.length === 0) return { dates: [], series: [] };

  const maps = valid.map((it) => new Map(it.bars.map((b) => [b.date, b.close])));
  let common = [...maps[0].keys()];
  for (let i = 1; i < maps.length; i++) common = common.filter((d) => maps[i].has(d));
  common.sort();

  const series = valid.map((it, i) => {
    const closes = common.map((d) => maps[i].get(d)!);
    const base = closes[0];
    return { ticker: it.ticker, normalized: base ? closes.map((c) => Math.round((c / base) * 100 * 1e10) / 1e10) : [] };
  });
  return { dates: common, series };
}
