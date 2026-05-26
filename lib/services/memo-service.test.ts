import { describe, it, expect, vi, beforeEach } from "vitest";

const {
  getTickerData, getCompanyByTicker, getNews,
  getRecentArticles, hasArticleNewerThan, getArticlesByIds,
  getMemoForDate, upsertMemo, canCall, generateMemo,
} = vi.hoisted(() => ({
  getTickerData: vi.fn(), getCompanyByTicker: vi.fn(), getNews: vi.fn(),
  getRecentArticles: vi.fn(), hasArticleNewerThan: vi.fn(), getArticlesByIds: vi.fn(),
  getMemoForDate: vi.fn(), upsertMemo: vi.fn(), canCall: vi.fn(), generateMemo: vi.fn(),
}));

vi.mock("@/lib/services/price-service", () => ({ getTickerData }));
vi.mock("@/lib/db/companies", () => ({ getCompanyByTicker }));
vi.mock("@/lib/services/news-service", () => ({ getNews }));
vi.mock("@/lib/db/articles", () => ({ getRecentArticles, hasArticleNewerThan, getArticlesByIds }));
vi.mock("@/lib/db/daily-memos", () => ({ getMemoForDate, upsertMemo }));
vi.mock("@/lib/db/provider-state", () => ({ canCall }));
vi.mock("@/lib/providers/gemini", () => ({ generateMemo }));

import { getMemo } from "./memo-service";

const company = { id: "c1", ticker: "NVDA", name: "NVIDIA", currency: "USD" };
const priceData = { bars: [{ close: 215.3 }], returns: { oneDay: -0.019, fiveDay: 0.02, oneMonth: 0.05, oneYear: 0.62 } };
const articleRow = (id: string) => ({ id, source: "finnhub", url: `https://ex.com/${id}`, title: `Title ${id}`, summary: "s", related: null, publishedAt: new Date("2026-05-24T12:00:00Z") });

const geminiOut = {
  ticker: "NVDA", date: "2026-05-25", one_sentence_takeaway: "Busy week.",
  bullish_developments: [
    { claim: "Real", why_it_matters: "x", source_article_ids: ["a1"], confidence: "medium" },
    { claim: "Hallucinated cite only", why_it_matters: "y", source_article_ids: ["a999"], confidence: "low" },
  ],
  bearish_developments: [], neutral_or_operational_updates: [],
  watch_items: [], caveats: [],
  overall_news_tone: { label: "somewhat_bullish", score: 64, rationale: "Mostly positive." },
};

beforeEach(() => {
  [getTickerData, getCompanyByTicker, getNews, getRecentArticles, hasArticleNewerThan,
    getArticlesByIds, getMemoForDate, upsertMemo, canCall, generateMemo].forEach((m) => m.mockReset());
  getTickerData.mockResolvedValue(priceData);
  getCompanyByTicker.mockResolvedValue(company);
  getNews.mockResolvedValue({ articles: [], asOf: null });
  getRecentArticles.mockResolvedValue([articleRow("uuid-1")]);
  hasArticleNewerThan.mockResolvedValue(false);
  getArticlesByIds.mockImplementation(async (ids: string[]) => ids.map((id) => articleRow(id)));
  canCall.mockResolvedValue(true);
  generateMemo.mockResolvedValue(geminiOut);
  process.env.GEMINI_API_KEY = "test-key";
});

describe("getMemo", () => {
  it("serves a fresh cached memo without calling Gemini", async () => {
    getMemoForDate.mockResolvedValue({
      summaryJson: geminiOut, generatedAt: new Date(), model: "gemini-3.5-flash",
      basedOnArticleCount: 3, sourceArticleIds: ["uuid-1"],
    });
    const out = await getMemo("NVDA");
    expect(out.status).toBe("ok");
    expect(generateMemo).not.toHaveBeenCalled();
  });

  it("regenerates when a newer article exists", async () => {
    getMemoForDate.mockResolvedValue({
      summaryJson: geminiOut, generatedAt: new Date(Date.now() - 3600_000),
      model: "gemini-3.5-flash", basedOnArticleCount: 1, sourceArticleIds: ["uuid-1"],
    });
    hasArticleNewerThan.mockResolvedValue(true);
    const out = await getMemo("NVDA");
    expect(generateMemo).toHaveBeenCalled();
    expect(out.status).toBe("ok");
  });

  it("returns no_news when there are no recent articles", async () => {
    getMemoForDate.mockResolvedValue(undefined);
    getRecentArticles.mockResolvedValue([]);
    const out = await getMemo("NVDA");
    expect(out.status).toBe("no_news");
    expect(generateMemo).not.toHaveBeenCalled();
  });

  it("returns unavailable when the Gemini budget is exhausted", async () => {
    getMemoForDate.mockResolvedValue(undefined);
    canCall.mockResolvedValue(false);
    const out = await getMemo("NVDA");
    expect(out.status).toBe("unavailable");
  });

  it("drops developments citing unknown ids and persists the memo", async () => {
    getMemoForDate.mockResolvedValue(undefined);
    const out = await getMemo("NVDA");
    expect(out.status).toBe("ok");
    // 'a1' maps to the one input article; 'a999' is hallucinated → that development dropped
    expect(out.memo!.bullish_developments).toHaveLength(1);
    expect(upsertMemo).toHaveBeenCalledTimes(1);
  });

  it("returns error when Gemini throws", async () => {
    getMemoForDate.mockResolvedValue(undefined);
    generateMemo.mockRejectedValue(new Error("gemini 500"));
    const out = await getMemo("NVDA");
    expect(out.status).toBe("error");
  });
});
