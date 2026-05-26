import { describe, it, expect } from "vitest";
import { computeReturns } from "./returns";
import type { PriceBar } from "@/lib/types";

const bar = (date: string, close: number): PriceBar => ({
  date, open: close, high: close, low: close, close, adjClose: null, volume: 0,
});

describe("computeReturns", () => {
  it("computes 1D and 5D returns by trading-day count", () => {
    const bars: PriceBar[] = [
      bar("2025-01-02", 100), bar("2025-01-03", 101), bar("2025-01-06", 102),
      bar("2025-01-07", 103), bar("2025-01-08", 104), bar("2025-01-09", 110),
    ];
    const r = computeReturns(bars);
    expect(r.oneDay).toBeCloseTo((110 - 104) / 104, 10);
    expect(r.fiveDay).toBeCloseTo((110 - 100) / 100, 10);
  });

  it("computes YTD vs the last close of the previous year", () => {
    const bars: PriceBar[] = [
      bar("2024-12-31", 200), bar("2025-01-02", 210), bar("2025-01-03", 220),
    ];
    const r = computeReturns(bars);
    expect(r.ytd).toBeCloseTo((220 - 200) / 200, 10);
  });

  it("returns null when there is insufficient history", () => {
    const r = computeReturns([bar("2025-01-02", 100)]);
    expect(r.oneDay).toBeNull();
    expect(r.oneYear).toBeNull();
  });
});
