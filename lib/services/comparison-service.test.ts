import { describe, it, expect, vi, beforeEach } from "vitest";

const { getTickerData } = vi.hoisted(() => ({ getTickerData: vi.fn() }));
vi.mock("./price-service", () => ({ getTickerData }));

import { compareTickers } from "./comparison-service";
import type { PriceBar } from "@/lib/types";

const bars = (closes: [string, number][]): PriceBar[] =>
  closes.map(([date, c]) => ({ date, open: c, high: c, low: c, close: c, adjClose: null, volume: 0 }));

beforeEach(() => getTickerData.mockReset());

describe("compareTickers", () => {
  it("aligns on common dates and computes normalized series + relative return", async () => {
    getTickerData.mockImplementation((t: string) => ({
      ticker: t,
      bars: t === "A"
        ? bars([["2025-01-02", 100], ["2025-01-03", 110], ["2025-01-06", 120]])
        : bars([["2025-01-03", 50], ["2025-01-06", 55]]),
    }));
    const out = await compareTickers("A", "B", "1y");
    expect(out.dates).toEqual(["2025-01-03", "2025-01-06"]);
    expect(out.primary.normalized[0]).toBeCloseTo(100, 6);
    expect(out.primary.normalized[1]).toBeCloseTo((120 / 110) * 100, 6);
    expect(out.comparison.normalized[1]).toBeCloseTo((55 / 50) * 100, 6);
    expect(out.relativeReturn).toBeCloseTo((120 / 110 - 1) - (55 / 50 - 1), 6);
  });

  it("returns empty alignment and null relative return when there is no date overlap", async () => {
    getTickerData.mockImplementation((t: string) => ({
      ticker: t,
      bars: t === "A"
        ? bars([["2025-01-02", 100], ["2025-01-03", 110]])
        : bars([["2025-02-03", 50], ["2025-02-04", 55]]),
    }));
    const out = await compareTickers("A", "B", "1y");
    expect(out.dates).toEqual([]);
    expect(out.primary.normalized).toEqual([]);
    expect(out.comparison.normalized).toEqual([]);
    expect(out.relativeReturn).toBeNull();
  });
});
