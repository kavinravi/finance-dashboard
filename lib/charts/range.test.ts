import { describe, it, expect } from "vitest";
import { rangeStartDate, sliceByRange, downsample, type SliceableIndicators } from "./range";
import type { PriceBar } from "@/lib/types";

const bar = (date: string, close: number): PriceBar => ({ date, open: close, high: close, low: close, close, adjClose: null, volume: 0 });

// Five consecutive daily bars.
const bars: PriceBar[] = [
  bar("2026-05-18", 10), bar("2026-05-19", 11), bar("2026-05-20", 12), bar("2026-05-21", 13), bar("2026-05-22", 14),
];
const ind: SliceableIndicators = {
  ma20: [1, 2, 3, 4, 5], ma50: [1, 2, 3, 4, 5], rsi14: [1, 2, 3, 4, 5],
  macdLine: [1, 2, 3, 4, 5], macdSignal: [1, 2, 3, 4, 5], macdHistogram: [1, 2, 3, 4, 5],
};

describe("rangeStartDate", () => {
  const wide = [bar("2021-01-04", 1), bar("2026-05-26", 2)];
  it("computes preset cutoffs against today", () => {
    expect(rangeStartDate("1y", wide, "2026-05-26")).toBe("2025-05-26");
    expect(rangeStartDate("6m", wide, "2026-05-26")).toBe("2025-11-26");
    expect(rangeStartDate("ytd", wide, "2026-05-26")).toBe("2026-01-01");
    expect(rangeStartDate("all", wide, "2026-05-26")).toBe("2021-01-04");
  });
  it("clamps a custom start before the first bar to the first bar", () => {
    expect(rangeStartDate({ from: "2019-01-01", to: "2026-05-26" }, wide, "2026-05-26")).toBe("2021-01-04");
  });
});

describe("sliceByRange", () => {
  it("slices bars and every indicator array to a custom window", () => {
    const out = sliceByRange(bars, ind, { from: "2026-05-19", to: "2026-05-21" }, "2026-05-22");
    expect(out.bars.map((b) => b.date)).toEqual(["2026-05-19", "2026-05-20", "2026-05-21"]);
    expect(out.indicators.ma20).toEqual([2, 3, 4]);
    expect(out.indicators.macdHistogram).toEqual([2, 3, 4]);
  });
  it("returns everything for 'all'", () => {
    const out = sliceByRange(bars, ind, "all", "2026-05-22");
    expect(out.bars).toHaveLength(5);
    expect(out.indicators.rsi14).toHaveLength(5);
  });
  it("handles empty input", () => {
    const out = sliceByRange([], ind, "1y", "2026-05-22");
    expect(out.bars).toEqual([]);
  });
});

describe("downsample", () => {
  it("returns input unchanged when under the cap", () => {
    const out = downsample(bars, ind, 800);
    expect(out.bars).toHaveLength(5);
  });

  it("caps the point count and always keeps the latest bar", () => {
    const many = Array.from({ length: 5000 }, (_, i) => bar(`2026-${String((i % 12) + 1).padStart(2, "0")}-01`, i));
    const manyInd: SliceableIndicators = {
      ma20: many.map((_, i) => i), ma50: many.map((_, i) => i), rsi14: many.map((_, i) => i),
      macdLine: many.map((_, i) => i), macdSignal: many.map((_, i) => i), macdHistogram: many.map((_, i) => i),
    };
    const out = downsample(many, manyInd, 800);
    expect(out.bars.length).toBeLessThanOrEqual(801);
    expect(out.bars.length).toBeGreaterThan(0);
    expect(out.bars.at(-1)).toEqual(many.at(-1)); // latest preserved
    expect(out.indicators.ma20).toHaveLength(out.bars.length); // alignment preserved
  });
});
