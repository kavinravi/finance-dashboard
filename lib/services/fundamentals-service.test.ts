import { describe, it, expect, vi, beforeEach } from "vitest";

// Mock the SEC provider — no live HTTP.
vi.mock("@/lib/providers/sec", () => ({
  SEC_DAILY_LIMIT: 100000,
  findCik: () => null,
  sec: {
    name: "sec",
    resolveCik: vi.fn(async () => "0000000001"),
    fetchCompanyFacts: vi.fn(async () => ({
      facts: {
        "us-gaap": {
          Revenues: { units: { USD: [
            { start: "2023-01-01", end: "2023-12-31", val: 1000, fy: 2023, fp: "FY", form: "10-K", filed: "2024-02-01" },
          ] } },
          NetIncomeLoss: { units: { USD: [
            { start: "2023-01-01", end: "2023-12-31", val: 100, fy: 2023, fp: "FY", form: "10-K", filed: "2024-02-01" },
          ] } },
          Assets: { units: { USD: [{ end: "2023-12-31", val: 2000, fy: 2023, fp: "FY", form: "10-K", filed: "2024-02-01" }] } },
          StockholdersEquity: { units: { USD: [{ end: "2023-12-31", val: 800, fy: 2023, fp: "FY", form: "10-K", filed: "2024-02-01" }] } },
        },
        dei: { EntityCommonStockSharesOutstanding: { units: { shares: [
          { end: "2023-12-31", val: 50, fy: 2023, fp: "FY", form: "10-K", filed: "2024-02-01" },
        ] } } },
      },
    })),
  },
}));

// Mock price-service so the company is ensured with a known close + asset type.
vi.mock("@/lib/services/price-service", () => ({
  getTickerData: vi.fn(async () => ({ bars: [{ date: "2024-01-02", open: 0, high: 0, low: 0, close: 10, adjClose: null, volume: 0 }] })),
}));

import { getFundamentals } from "./fundamentals-service";
import { db } from "@/lib/db/client";
import { companies, companyFundamentals } from "@/lib/db/schema";
import { eq } from "drizzle-orm";
import { sec } from "@/lib/providers/sec";

async function reseed(ticker: string, assetType: string) {
  const [c] = await db.select().from(companies).where(eq(companies.ticker, ticker));
  if (c) {
    await db.delete(companyFundamentals).where(eq(companyFundamentals.companyId, c.id));
    await db.delete(companies).where(eq(companies.id, c.id));
  }
  const [row] = await db.insert(companies)
    .values({ ticker, name: ticker, assetType, currency: "USD" }).returning();
  return row;
}

describe("fundamentals-service (live Neon, SEC mocked)", () => {
  beforeEach(() => vi.clearAllMocks());

  it("returns not_applicable for ETFs/indexes", async () => {
    await reseed("TST_ETF", "etf");
    const r = await getFundamentals("TST_ETF");
    expect(r.status).toBe("not_applicable");
    expect(sec.fetchCompanyFacts).not.toHaveBeenCalled();
  });

  it("resolves CIK, fetches, extracts, derives, and caches (cache-first on 2nd call)", async () => {
    await reseed("TST_STK", "stock");
    const r1 = await getFundamentals("TST_STK");
    expect(r1.status).toBe("ok");
    expect(r1.view?.revenue).toBe(1000);
    expect(r1.view?.marketCap).toBe(500);      // close 10 * shares 50
    expect(r1.view?.peRatio).toBeNull();        // no EPS in fixture
    expect(r1.asOf?.fiscalYear).toBe(2023);
    expect(r1.asOf?.edgarUrl).toContain("CIK=0000000001");
    expect(sec.fetchCompanyFacts).toHaveBeenCalledTimes(1);

    const r2 = await getFundamentals("TST_STK");
    expect(r2.status).toBe("ok");
    expect(sec.fetchCompanyFacts).toHaveBeenCalledTimes(1); // served from cache — no refetch
  });

  it("returns unavailable when no CIK can be resolved and there is no cache", async () => {
    await reseed("TST_NOCIK", "stock");
    (sec.resolveCik as ReturnType<typeof vi.fn>).mockResolvedValueOnce(null);
    const r = await getFundamentals("TST_NOCIK");
    expect(r.status).toBe("unavailable");
  });
});
