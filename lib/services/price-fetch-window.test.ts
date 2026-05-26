import { describe, it, expect } from "vitest";
import { fetchFromDate, HISTORY_FLOOR } from "./price-fetch-window";
import type { PriceBar } from "@/lib/types";

const bar = (date: string): PriceBar => ({ date, open: 1, high: 1, low: 1, close: 1, adjClose: null, volume: 0 });

describe("fetchFromDate", () => {
  const today = "2026-05-26";

  it("fetches full history when the cache is empty", () => {
    expect(fetchFromDate([], today)).toBe(HISTORY_FLOOR);
  });

  it("fetches full history when the cache is shallow (earliest bar within 3y)", () => {
    expect(fetchFromDate([bar("2024-06-01"), bar("2026-05-20")], today)).toBe(HISTORY_FLOOR);
  });

  it("fetches incrementally (last bar minus a buffer) once the cache is deep", () => {
    // earliest bar older than 3y → deep; last bar 2026-05-20 → from = 5 days earlier
    expect(fetchFromDate([bar("2018-01-02"), bar("2026-05-20")], today)).toBe("2026-05-15");
  });
});
