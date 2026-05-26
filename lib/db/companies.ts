import { db } from "./client";
import { companies } from "./schema";
import { eq } from "drizzle-orm";
import type { CompanyProfile } from "@/lib/types";

export type CompanyRow = typeof companies.$inferSelect;

export async function getCompanyByTicker(ticker: string): Promise<CompanyRow | undefined> {
  const [row] = await db.select().from(companies).where(eq(companies.ticker, ticker.toUpperCase()));
  return row;
}

export async function upsertCompany(p: CompanyProfile): Promise<CompanyRow> {
  const ticker = p.ticker.toUpperCase();
  await db.insert(companies)
    .values({ ticker, name: p.name, assetType: p.assetType, exchange: p.exchange,
      sector: p.sector, industry: p.industry, currency: p.currency,
      lastProfileRefreshAt: new Date(), updatedAt: new Date() })
    .onConflictDoUpdate({
      target: companies.ticker,
      set: { name: p.name, assetType: p.assetType, exchange: p.exchange, sector: p.sector,
        industry: p.industry, currency: p.currency, lastProfileRefreshAt: new Date(), updatedAt: new Date() },
    });
  return (await getCompanyByTicker(ticker))!;
}
