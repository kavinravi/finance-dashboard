import { db } from "./client";
import { recentSearches } from "./schema";
import { desc } from "drizzle-orm";

export async function logSearch(query: string, resolvedTicker: string | null): Promise<void> {
  await db.insert(recentSearches).values({ query, resolvedTicker });
}

export async function listRecentSearches(limit = 10) {
  return db.select().from(recentSearches).orderBy(desc(recentSearches.createdAt)).limit(limit);
}
