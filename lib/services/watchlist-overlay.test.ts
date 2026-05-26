import { describe, it, expect } from "vitest";
import { buildOverlay } from "./watchlist-overlay";

const bar = (date: string, close: number) => ({ date, close });

describe("buildOverlay", () => {
  it("normalizes each series to 100 at the first common date", () => {
    const o = buildOverlay([
      { ticker: "A", bars: [bar("2026-01-01", 10), bar("2026-01-02", 11), bar("2026-01-03", 12)] },
      { ticker: "B", bars: [bar("2026-01-01", 20), bar("2026-01-02", 25), bar("2026-01-03", 20)] },
    ]);
    expect(o.dates).toEqual(["2026-01-01", "2026-01-02", "2026-01-03"]);
    expect(o.series[0]).toEqual({ ticker: "A", normalized: [100, 110, 120] });
    expect(o.series[1]).toEqual({ ticker: "B", normalized: [100, 125, 100] });
  });

  it("intersects on common dates only", () => {
    const o = buildOverlay([
      { ticker: "A", bars: [bar("2026-01-01", 10), bar("2026-01-02", 11)] },
      { ticker: "B", bars: [bar("2026-01-02", 20), bar("2026-01-03", 22)] },
    ]);
    expect(o.dates).toEqual(["2026-01-02"]);
    expect(o.series[0].normalized).toEqual([100]);
    expect(o.series[1].normalized).toEqual([100]);
  });

  it("drops empty-bar tickers and returns empty for all-empty input", () => {
    expect(buildOverlay([{ ticker: "A", bars: [] }])).toEqual({ dates: [], series: [] });
    expect(buildOverlay([])).toEqual({ dates: [], series: [] });
  });
});
