import { db } from "./client";
import { watchlist } from "./schema";
import { eq, asc } from "drizzle-orm";

export type WatchlistRow = typeof watchlist.$inferSelect;

export async function getWatchlist(): Promise<WatchlistRow[]> {
  return db.select().from(watchlist).orderBy(asc(watchlist.createdAt));
}

export async function addToWatchlist(ticker: string): Promise<void> {
  await db.insert(watchlist).values({ ticker: ticker.toUpperCase() }).onConflictDoNothing();
}

export async function removeFromWatchlist(ticker: string): Promise<void> {
  await db.delete(watchlist).where(eq(watchlist.ticker, ticker.toUpperCase()));
}
