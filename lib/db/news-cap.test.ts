import { describe, it, expect, afterAll } from "vitest";
import { eq } from "drizzle-orm";
import { db } from "@/lib/db/client";
import { companies, articles } from "@/lib/db/schema";
import { getRecentArticles } from "@/lib/db/articles";

const TICKER = `ZZNEWS${Date.now() % 100000}`;
let companyId = "";

afterAll(async () => {
  if (companyId) {
    await db.delete(articles).where(eq(articles.companyId, companyId));
    await db.delete(companies).where(eq(companies.id, companyId));
  }
});

describe("getRecentArticles cap (integration, live Neon)", () => {
  it("returns at most the limit, newest first", async () => {
    const [c] = await db.insert(companies).values({ ticker: TICKER, name: "News Cap Co" }).returning({ id: companies.id });
    companyId = c.id;
    const now = Date.now();
    const rows = Array.from({ length: 14 }, (_, i) => ({
      companyId, source: "finnhub", url: `https://ex.com/${TICKER}/${i}`, urlHash: `${TICKER}-${i}`,
      title: `Article ${i}`, publishedAt: new Date(now - i * 60_000),
    }));
    await db.insert(articles).values(rows);

    const since = new Date(now - 7 * 86_400_000).toISOString().slice(0, 10);
    const got = await getRecentArticles(companyId, since, 10);
    expect(got).toHaveLength(10);
    expect(got[0].title).toBe("Article 0"); // newest first
  });
});
