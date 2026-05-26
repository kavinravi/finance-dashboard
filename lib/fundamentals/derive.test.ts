import { describe, it, expect } from "vitest";
import { deriveMetrics } from "./derive";
import type { FundamentalConcepts } from "@/lib/types";

const C: FundamentalConcepts = {
  revenue: 1000, netIncome: 100, eps: 5, operatingIncome: 200, grossProfit: 400,
  assets: 2000, liabilities: 1200, equity: 800, currentAssets: 600, currentLiabilities: 300,
  sharesOutstanding: 50,
};

describe("deriveMetrics", () => {
  it("computes price-derived multiples from the latest close", () => {
    const v = deriveMetrics(C, 10);
    expect(v.marketCap).toBe(500);       // 10 * 50
    expect(v.peRatio).toBe(2);           // 10 / 5
    expect(v.psRatio).toBe(0.5);         // 500 / 1000
  });
  it("computes price-independent ratios", () => {
    const v = deriveMetrics(C, 10);
    expect(v.grossMargin).toBe(0.4);     // 400 / 1000
    expect(v.roe).toBe(0.125);           // 100 / 800
    expect(v.roa).toBe(0.05);            // 100 / 2000
    expect(v.currentRatio).toBe(2);      // 600 / 300
    expect(v.debtToEquity).toBe(1.5);    // 1200 / 800
  });
  it("passes through raw financials", () => {
    const v = deriveMetrics(C, 10);
    expect(v.revenue).toBe(1000);
    expect(v.netIncome).toBe(100);
    expect(v.eps).toBe(5);
    expect(v.operatingIncome).toBe(200);
  });
  it("returns null for divide-by-zero / null inputs", () => {
    const v = deriveMetrics({ ...C, equity: 0, eps: null }, 10);
    expect(v.roe).toBeNull();            // /0
    expect(v.debtToEquity).toBeNull();   // /0
    expect(v.peRatio).toBeNull();        // eps null
    const noClose = deriveMetrics(C, null);
    expect(noClose.marketCap).toBeNull();
    expect(noClose.peRatio).toBeNull();
    expect(noClose.psRatio).toBeNull();
  });
});
