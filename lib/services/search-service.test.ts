import { describe, it, expect, vi, beforeEach } from "vitest";

const { fmpSearch, yahooSearch, logSearch } = vi.hoisted(() => ({
  fmpSearch: vi.fn(),
  yahooSearch: vi.fn(),
  logSearch: vi.fn(),
}));

vi.mock("@/lib/providers/fmp", () => ({ fmp: { search: fmpSearch } }));
vi.mock("@/lib/providers/yahoo", () => ({ yahoo: { search: yahooSearch } }));
vi.mock("@/lib/db/recent-searches", () => ({ logSearch }));

import { resolveQuery } from "./search-service";

beforeEach(() => { fmpSearch.mockReset(); yahooSearch.mockReset(); logSearch.mockReset(); });

describe("resolveQuery", () => {
  it("returns FMP results and logs the top resolved ticker", async () => {
    fmpSearch.mockResolvedValue([{ symbol: "NVDA", name: "NVIDIA", exchange: "NASDAQ", assetType: "stock", source: "fmp" }]);
    const out = await resolveQuery("nvidia");
    expect(out[0].symbol).toBe("NVDA");
    expect(logSearch).toHaveBeenCalledWith("nvidia", "NVDA");
  });

  it("falls back to Yahoo when FMP throws, and still logs", async () => {
    fmpSearch.mockRejectedValue(new Error("fmp down"));
    yahooSearch.mockResolvedValue([{ symbol: "AMD", name: "AMD", exchange: "NMS", assetType: "stock", source: "yahoo" }]);
    const out = await resolveQuery("amd");
    expect(out[0].source).toBe("yahoo");
    expect(logSearch).toHaveBeenCalledWith("amd", "AMD");
  });

  it("logs a null ticker when nothing resolves", async () => {
    fmpSearch.mockResolvedValue([]);
    yahooSearch.mockResolvedValue([]);
    const out = await resolveQuery("zzzzz");
    expect(out).toEqual([]);
    expect(logSearch).toHaveBeenCalledWith("zzzzz", null);
  });

  it("returns [] when both providers throw", async () => {
    fmpSearch.mockRejectedValue(new Error("fmp down"));
    yahooSearch.mockRejectedValue(new Error("yahoo down"));
    const out = await resolveQuery("xyz");
    expect(out).toEqual([]);
    expect(logSearch).toHaveBeenCalledWith("xyz", null);
  });

  it("short-circuits an empty query without calling providers or logging", async () => {
    const out = await resolveQuery("   ");
    expect(out).toEqual([]);
    expect(fmpSearch).not.toHaveBeenCalled();
    expect(logSearch).not.toHaveBeenCalled();
  });
});
