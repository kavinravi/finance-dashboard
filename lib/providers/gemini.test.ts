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
  it("coerces out-of-range scores, unknown labels, and odd confidence values to safe defaults", () => {
    const r1 = memoOutputSchema.safeParse({ ...validMemo, overall_news_tone: { label: "neutral", score: 150, rationale: "x" } });
    expect(r1.success).toBe(true);
    if (r1.success) expect(r1.data.overall_news_tone.score).toBe(100); // clamped

    const r2 = memoOutputSchema.safeParse({ ...validMemo, overall_news_tone: { label: "great", score: 62.5, rationale: "x" } });
    expect(r2.success).toBe(true);
    if (r2.success) {
      expect(r2.data.overall_news_tone.label).toBe("neutral"); // unknown → caught
      expect(r2.data.overall_news_tone.score).toBe(63);          // float → rounded
    }

    const r3 = memoOutputSchema.safeParse({
      ...validMemo,
      neutral_or_operational_updates: [{ claim: "c", why_it_matters: "w", source_article_ids: ["a1"], confidence: "informational" }],
    });
    expect(r3.success).toBe(true);
    if (r3.success) expect(r3.data.neutral_or_operational_updates[0].confidence).toBe("medium"); // unknown → caught

    const r4 = memoOutputSchema.safeParse({ ...validMemo, overall_news_tone: { label: "Somewhat Bullish", score: 70, rationale: "x" } });
    expect(r4.success && r4.data.overall_news_tone.label).toBe("somewhat_bullish");
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
