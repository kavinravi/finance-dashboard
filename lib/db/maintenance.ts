import { db } from "./client";
import { articles, companyFundamentals } from "./schema";
import { lt } from "drizzle-orm";

export async function pruneExpired(now: Date = new Date()): Promise<{ articles: number; fundamentals: number }> {
  const a = await db.delete(articles).where(lt(articles.expiresAt, now)).returning({ id: articles.id });
  const f = await db.delete(companyFundamentals).where(lt(companyFundamentals.expiresAt, now)).returning({ id: companyFundamentals.id });
  return { articles: a.length, fundamentals: f.length };
}
