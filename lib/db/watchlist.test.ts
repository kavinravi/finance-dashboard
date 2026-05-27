import { describe, it, expect, beforeAll, afterAll } from "vitest";
import { getWatchlist, addToWatchlist, removeFromWatchlist } from "@/lib/db/watchlist";
import { createProfile, deleteProfile } from "@/lib/db/profiles";

const T = `ZZ${Date.now() % 100000}`; // short, regex-valid, uppercase
let profileId: string;

beforeAll(async () => { profileId = (await createProfile(`wl-${Date.now()}`)).id; });
afterAll(async () => { await deleteProfile(profileId); }); // cascade removes its watchlist rows

describe("watchlist repo (integration, live Neon)", () => {
  it("adds (uppercased), lists, dedupes, and removes — scoped to a profile", async () => {
    await addToWatchlist(profileId, T.toLowerCase());
    let tickers = (await getWatchlist(profileId)).map((w) => w.ticker);
    expect(tickers).toContain(T);

    await addToWatchlist(profileId, T); // dedupe within the profile — no throw, no duplicate
    tickers = (await getWatchlist(profileId)).map((w) => w.ticker);
    expect(tickers.filter((x) => x === T)).toHaveLength(1);

    await removeFromWatchlist(profileId, T);
    tickers = (await getWatchlist(profileId)).map((w) => w.ticker);
    expect(tickers).not.toContain(T);
  });
});
