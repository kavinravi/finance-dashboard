import { describe, it, expect, beforeEach } from "vitest";
import { db } from "@/lib/db/client";
import { companies, priceBarsDaily } from "@/lib/db/schema";
import { eq } from "drizzle-orm";
import { upsertCompany, getCompanyByTicker } from "@/lib/db/companies";
import { upsertBars, getBars } from "@/lib/db/price-bars";
import { logSearch, listRecentSearches as listRecent } from "@/lib/db/recent-searches";

async function cleanup(ticker: string) {
  const c = await getCompanyByTicker(ticker);
  if (c) {
    await db.delete(priceBarsDaily).where(eq(priceBarsDaily.companyId, c.id));
    await db.delete(companies).where(eq(companies.id, c.id));
  }
}

describe("repositories", () => {
  beforeEach(() => cleanup("TEST"));

  it("upserts a company idempotently by ticker", async () => {
    const a = await upsertCompany({ ticker: "TEST", name: "Test One", assetType: "stock",
      exchange: "NYSE", sector: null, industry: null, currency: "USD" });
    const b = await upsertCompany({ ticker: "TEST", name: "Test Two", assetType: "stock",
      exchange: "NYSE", sector: null, industry: null, currency: "USD" });
    expect(a.id).toBe(b.id);
    expect((await getCompanyByTicker("TEST"))!.name).toBe("Test Two");
  });

  it("upserts bars without duplicating on (company,date,source) and reads them ascending", async () => {
    const c = await upsertCompany({ ticker: "TEST", name: "Test", assetType: "stock",
      exchange: null, sector: null, industry: null, currency: "USD" });
    const bars = [
      { date: "2025-05-02", open: 2, high: 3, low: 1, close: 2.5, adjClose: null, volume: 100 },
      { date: "2025-05-01", open: 1, high: 2, low: 0.5, close: 1.5, adjClose: null, volume: 50 },
    ];
    await upsertBars(c.id, bars, "fmp");
    await upsertBars(c.id, bars, "fmp"); // repeat → no dupes
    const read = await getBars(c.id);
    expect(read.map((b) => b.date)).toEqual(["2025-05-01", "2025-05-02"]);
  });

  it("logs and lists recent searches (most recent first)", async () => {
    await logSearch("nvidia", "NVDA");
    const rows = await listRecent(5);
    expect(rows[0].query).toBe("nvidia");
  });
});
