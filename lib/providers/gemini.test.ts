import { describe, it, expect } from "vitest";
import { memoOutputSchema, buildPrompt, type MemoInput } from "./gemini";

const validMemo = {
  ticker: "NVDA", date: "2026-05-25", one_sentence_takeaway: "Quiet week.",
  bullish_developments: [], bearish_developments: [], neutral_or_operational_updates: [],
  watch_items: [], caveats: ["Sparse coverage."],
  overall_news_tone: { label: "neutral", score: 50, rationale: "Few articles." },
};

describe("memoOutputSchema", () => {
  it("accepts a valid memo", () => {
    expect(memoOutputSchema.safeParse(validMemo).success).toBe(true);
  });
  it("rejects an out-of-range score and a bad label", () => {
    expect(memoOutputSchema.safeParse({ ...validMemo, overall_news_tone: { label: "neutral", score: 150, rationale: "x" } }).success).toBe(false);
    expect(memoOutputSchema.safeParse({ ...validMemo, overall_news_tone: { label: "great", score: 50, rationale: "x" } }).success).toBe(false);
  });
});

describe("buildPrompt", () => {
  it("includes every article id and forbids buy/sell/hold", () => {
    const input: MemoInput = {
      ticker: "NVDA", companyName: "NVIDIA", date: "2026-05-25",
      priceContext: { latestClose: 215.3, currency: "USD", returns: { d1: -0.019, d5: 0.02, m1: 0.05, y1: 0.62 } },
      articles: [
        { id: "a1", source: "finnhub", publishedAt: "2026-05-24T12:00:00Z", headline: "Chip demand", summary: "Up", related: "NVDA" },
        { id: "a2", source: "yahoo_rss", publishedAt: "2026-05-23T12:00:00Z", headline: "Supply news", summary: null, related: null },
      ],
    };
    const p = buildPrompt(input);
    expect(p).toContain("[a1]");
    expect(p).toContain("[a2]");
    expect(p.toLowerCase()).toContain("never output buy");
  });
});
