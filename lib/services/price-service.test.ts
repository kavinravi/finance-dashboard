import { describe, it, expect, vi, beforeEach } from "vitest";

const {
  getCompanyByTicker, upsertCompany, getBars, upsertBars,
  fmpProfile, fmpPrices, yahooPrices,
  canCall, recordSuccess, recordError,
} = vi.hoisted(() => ({
  getCompanyByTicker: vi.fn(),
  upsertCompany: vi.fn(),
  getBars: vi.fn(),
  upsertBars: vi.fn(),
  fmpProfile: vi.fn(),
  fmpPrices: vi.fn(),
  yahooPrices: vi.fn(),
  canCall: vi.fn(),
  recordSuccess: vi.fn(),
  recordError: vi.fn(),
}));

vi.mock("@/lib/db/companies", () => ({ getCompanyByTicker, upsertCompany }));
vi.mock("@/lib/db/price-bars", () => ({ getBars, upsertBars }));
vi.mock("@/lib/providers/fmp", () => ({ fmp: { profile: fmpProfile, dailyPrices: fmpPrices } }));
vi.mock("@/lib/providers/yahoo", () => ({ yahoo: { dailyPrices: yahooPrices } }));
vi.mock("@/lib/db/provider-state", () => ({ canCall, recordSuccess, recordError }));

import { getTickerData } from "./price-service";

const company = { id: "c1", ticker: "NVDA", name: "NVIDIA", assetType: "stock" };
const freshBars = (lastDate: string) => [
  { date: "2025-05-01", open: 1, high: 1, low: 1, close: 1, adjClose: null, volume: 1 },
  { date: lastDate, open: 2, high: 2, low: 2, close: 2, adjClose: null, volume: 1 },
];

beforeEach(() => {
  [getCompanyByTicker, upsertCompany, getBars, upsertBars, fmpProfile, fmpPrices, yahooPrices,
    canCall, recordSuccess, recordError].forEach((m) => m.mockReset());
  upsertCompany.mockResolvedValue(company);
  getCompanyByTicker.mockResolvedValue(company);
  canCall.mockResolvedValue(true);
});

describe("getTickerData", () => {
  it("serves cache without calling a provider when bars are fresh", async () => {
    const today = new Date().toISOString().slice(0, 10);
    getBars.mockResolvedValue(freshBars(today));
    const out = await getTickerData("NVDA", "1y");
    expect(fmpPrices).not.toHaveBeenCalled();
    expect(out.source).toBe("cache");
    expect(out.bars.length).toBe(2);
  });

  it("fetches from FMP when cache is stale, then upserts and serves", async () => {
    getBars.mockResolvedValueOnce([]).mockResolvedValueOnce(freshBars("2020-01-01"));
    fmpProfile.mockResolvedValue({ ticker: "NVDA", name: "NVIDIA", assetType: "stock",
      exchange: "NASDAQ", sector: null, industry: null, currency: "USD" });
    fmpPrices.mockResolvedValue(freshBars("2020-01-01"));
    const out = await getTickerData("NVDA", "1y");
    expect(fmpPrices).toHaveBeenCalled();
    expect(upsertBars).toHaveBeenCalledWith("c1", expect.any(Array), "fmp");
    expect(recordSuccess).toHaveBeenCalled();
    expect(out.source).toBe("fmp");
  });

  it("falls back to Yahoo when FMP fails, marking the source", async () => {
    getBars.mockResolvedValueOnce([]).mockResolvedValueOnce(freshBars("2020-01-01"));
    fmpProfile.mockResolvedValue(null);
    fmpPrices.mockRejectedValue(new Error("fmp 429"));
    yahooPrices.mockResolvedValue(freshBars("2020-01-01"));
    const out = await getTickerData("NVDA", "1y");
    expect(recordError).toHaveBeenCalled();
    expect(upsertBars).toHaveBeenCalledWith("c1", expect.any(Array), "yahoo");
    expect(out.source).toBe("yahoo");
  });

  it("degrades to cache (does not throw) when both providers fail and cache is empty", async () => {
    getBars.mockResolvedValue([]);
    fmpProfile.mockResolvedValue(null);
    fmpPrices.mockRejectedValue(new Error("fmp down"));
    yahooPrices.mockRejectedValue(new Error("yahoo down"));
    const out = await getTickerData("NVDA", "1y");
    expect(out.source).toBe("cache");
    expect(out.bars).toEqual([]);
    expect(out.returns.oneDay).toBeNull();
  });

  it("skips FMP and uses Yahoo when the FMP budget is exhausted", async () => {
    getBars.mockResolvedValueOnce([]).mockResolvedValueOnce(freshBars("2020-01-01"));
    canCall.mockResolvedValue(false);
    yahooPrices.mockResolvedValue(freshBars("2020-01-01"));
    const out = await getTickerData("NVDA", "1y");
    expect(fmpPrices).not.toHaveBeenCalled();
    expect(upsertBars).toHaveBeenCalledWith("c1", expect.any(Array), "yahoo");
    expect(out.source).toBe("yahoo");
  });
});
