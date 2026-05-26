import { describe, it, expect } from "vitest";
import { parseSearch, parseProfile, parseBars } from "./fmp";

describe("FMP parsers", () => {
  it("maps search results", () => {
    const raw = [{ symbol: "AAPL", name: "Apple Inc.", currency: "USD",
      exchangeFullName: "NASDAQ Global Select", exchange: "NASDAQ" }];
    expect(parseSearch(raw)).toEqual([
      { symbol: "AAPL", name: "Apple Inc.", exchange: "NASDAQ", assetType: "stock", source: "fmp" },
    ]);
  });

  it("maps a profile and infers ETF asset type", () => {
    const raw = [{ symbol: "SPY", companyName: "SPDR S&P 500 ETF Trust", currency: "USD",
      exchange: "NYSE", sector: "", industry: "", isEtf: true, isFund: false }];
    expect(parseProfile(raw)).toEqual({
      ticker: "SPY", name: "SPDR S&P 500 ETF Trust", assetType: "etf",
      exchange: "NYSE", sector: null, industry: null, currency: "USD",
    });
  });

  it("maps EOD bars and sorts ascending by date", () => {
    const raw = [
      { date: "2025-05-02", open: 2, high: 3, low: 1, close: 2.5, volume: 100 },
      { date: "2025-05-01", open: 1, high: 2, low: 0.5, close: 1.5, volume: 50 },
    ];
    const bars = parseBars(raw);
    expect(bars.map((b) => b.date)).toEqual(["2025-05-01", "2025-05-02"]);
    expect(bars[0]).toEqual({ date: "2025-05-01", open: 1, high: 2, low: 0.5,
      close: 1.5, adjClose: null, volume: 50 });
  });
});
