import { describe, it, expect, afterAll } from "vitest";
import { getWatchlist, addToWatchlist, removeFromWatchlist } from "@/lib/db/watchlist";

const T = `ZZ${Date.now() % 100000}`; // short, regex-valid, uppercase

afterAll(async () => { await removeFromWatchlist(T); });

describe("watchlist repo (integration, live Neon)", () => {
  it("adds (uppercased), lists, dedupes, and removes", async () => {
    await addToWatchlist(T.toLowerCase());
    let tickers = (await getWatchlist()).map((w) => w.ticker);
    expect(tickers).toContain(T);

    await addToWatchlist(T); // dedupe — no throw, no duplicate
    tickers = (await getWatchlist()).map((w) => w.ticker);
    expect(tickers.filter((x) => x === T)).toHaveLength(1);

    await removeFromWatchlist(T);
    tickers = (await getWatchlist()).map((w) => w.ticker);
    expect(tickers).not.toContain(T);
  });
});
