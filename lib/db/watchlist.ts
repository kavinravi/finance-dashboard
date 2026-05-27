import { db } from "./client";
import { watchlist } from "./schema";
import { and, eq, asc } from "drizzle-orm";

export type WatchlistRow = typeof watchlist.$inferSelect;

export async function getWatchlist(profileId: string): Promise<WatchlistRow[]> {
  return db.select().from(watchlist).where(eq(watchlist.profileId, profileId)).orderBy(asc(watchlist.createdAt));
}

export async function addToWatchlist(profileId: string, ticker: string): Promise<void> {
  await db.insert(watchlist).values({ profileId, ticker: ticker.toUpperCase() }).onConflictDoNothing();
}

export async function removeFromWatchlist(profileId: string, ticker: string): Promise<void> {
  await db.delete(watchlist).where(and(eq(watchlist.profileId, profileId), eq(watchlist.ticker, ticker.toUpperCase())));
}
