import { db } from "./client";
import { articles } from "./schema";
import { and, eq, gte, gt, desc, inArray } from "drizzle-orm";
import { urlHash } from "@/lib/news/dedupe";
import type { NewsArticle } from "@/lib/types";

export type ArticleRow = typeof articles.$inferSelect;

const RETENTION_DAYS = 90;

export async function upsertArticles(companyId: string, items: NewsArticle[]): Promise<void> {
  if (items.length === 0) return;
  const expiresAt = new Date(Date.now() + RETENTION_DAYS * 86_400_000);
  const rows = items.map((a) => ({
    companyId, source: a.source, sourceArticleId: a.sourceArticleId, url: a.url,
    urlHash: urlHash(a.url), title: a.title, summary: a.summary,
    publishedAt: a.publishedAt, imageUrl: a.imageUrl, related: a.related, expiresAt,
  }));
  for (let i = 0; i < rows.length; i += 500) {
    await db.insert(articles).values(rows.slice(i, i + 500)).onConflictDoNothing();
  }
}

export async function getRecentArticles(companyId: string, sinceIso: string): Promise<ArticleRow[]> {
  return db.select().from(articles)
    .where(and(eq(articles.companyId, companyId), gte(articles.publishedAt, new Date(sinceIso))))
    .orderBy(desc(articles.publishedAt));
}

export async function newestArticleCreatedAt(companyId: string): Promise<Date | null> {
  const [row] = await db.select().from(articles)
    .where(eq(articles.companyId, companyId)).orderBy(desc(articles.createdAt)).limit(1);
  return row?.createdAt ?? null;
}

export async function hasArticleNewerThan(companyId: string, t: Date): Promise<boolean> {
  const [row] = await db.select().from(articles)
    .where(and(eq(articles.companyId, companyId), gt(articles.createdAt, t))).limit(1);
  return !!row;
}

export async function getArticlesByIds(ids: string[]): Promise<ArticleRow[]> {
  if (ids.length === 0) return [];
  return db.select().from(articles).where(inArray(articles.id, ids));
}
