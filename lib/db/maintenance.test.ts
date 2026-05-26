import { describe, it, expect, afterAll } from "vitest";
import { eq } from "drizzle-orm";
import { db } from "@/lib/db/client";
import { companies, articles, companyFundamentals } from "@/lib/db/schema";
import { pruneExpired } from "@/lib/db/maintenance";
import { pruneExpiredForCompany } from "@/lib/db/articles";
import { getAllProviderStates } from "@/lib/db/provider-state";

const TICKER = `ZZPRUNE${Date.now()}`;
const past = new Date(Date.now() - 86_400_000);
const future = new Date(Date.now() + 86_400_000);
let companyId = "";

afterAll(async () => {
  if (companyId) {
    await db.delete(articles).where(eq(articles.companyId, companyId));
    await db.delete(companyFundamentals).where(eq(companyFundamentals.companyId, companyId));
    await db.delete(companies).where(eq(companies.id, companyId));
  }
});

describe("pruning (integration, live Neon)", () => {
  it("pruneExpired deletes expired articles + fundamentals and keeps fresh ones", async () => {
    const [c] = await db.insert(companies).values({ ticker: TICKER, name: "Prune Test Co" }).returning({ id: companies.id });
    companyId = c.id;

    await db.insert(articles).values([
      { companyId, source: "finnhub", url: "https://ex.com/expired", urlHash: `${TICKER}-exp`, title: "Expired", publishedAt: past, expiresAt: past },
      { companyId, source: "finnhub", url: "https://ex.com/fresh", urlHash: `${TICKER}-fresh`, title: "Fresh", publishedAt: past, expiresAt: future },
    ]);
    await db.insert(companyFundamentals).values([
      { companyId, conceptsJson: {}, source: "sec_edgar", expiresAt: past },
    ]);

    const counts = await pruneExpired();
    expect(counts.articles).toBeGreaterThanOrEqual(1);
    expect(counts.fundamentals).toBeGreaterThanOrEqual(1);

    const remaining = await db.select({ urlHash: articles.urlHash }).from(articles).where(eq(articles.companyId, companyId));
    expect(remaining.map((r) => r.urlHash)).toEqual([`${TICKER}-fresh`]);

    const fund = await db.select({ id: companyFundamentals.id }).from(companyFundamentals).where(eq(companyFundamentals.companyId, companyId));
    expect(fund).toHaveLength(0);
  });

  it("pruneExpiredForCompany removes only that company's expired rows", async () => {
    await db.insert(articles).values([
      { companyId, source: "finnhub", url: "https://ex.com/expired2", urlHash: `${TICKER}-exp2`, title: "Expired2", publishedAt: past, expiresAt: past },
    ]);
    const n = await pruneExpiredForCompany(companyId);
    expect(n).toBe(1);
    const remaining = await db.select({ urlHash: articles.urlHash }).from(articles).where(eq(articles.companyId, companyId));
    expect(remaining.map((r) => r.urlHash)).toEqual([`${TICKER}-fresh`]);
  });

  it("getAllProviderStates returns rows whose provider is a string", async () => {
    const rows = await getAllProviderStates();
    expect(Array.isArray(rows)).toBe(true);
    for (const r of rows) expect(typeof r.provider).toBe("string");
  });
});
