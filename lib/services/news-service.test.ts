import { describe, it, expect, vi, beforeEach } from "vitest";

const {
  getCompanyByTicker, upsertArticles, getRecentArticles, newestArticleCreatedAt,
  finnhubNews, yahooRssNews, pruneExpiredForCompany,
} = vi.hoisted(() => ({
  getCompanyByTicker: vi.fn(),
  upsertArticles: vi.fn(),
  getRecentArticles: vi.fn(),
  newestArticleCreatedAt: vi.fn(),
  finnhubNews: vi.fn(),
  yahooRssNews: vi.fn(),
  pruneExpiredForCompany: vi.fn(),
}));

vi.mock("@/lib/db/companies", () => ({ getCompanyByTicker }));
vi.mock("@/lib/db/articles", () => ({ upsertArticles, getRecentArticles, newestArticleCreatedAt, pruneExpiredForCompany }));
vi.mock("@/lib/providers/finnhub", () => ({ finnhub: { companyNews: finnhubNews } }));
vi.mock("@/lib/providers/yahoo-rss", () => ({ yahooRss: { companyNews: yahooRssNews } }));

import { getNews } from "./news-service";

const company = { id: "c1", ticker: "NVDA", name: "NVIDIA", currency: "USD" };
const article = (url: string, when: string) => ({
  source: "finnhub", sourceArticleId: "1", url, title: "T " + url, summary: null,
  publishedAt: new Date(when), imageUrl: null, related: null,
});

beforeEach(() => {
  [getCompanyByTicker, upsertArticles, getRecentArticles, newestArticleCreatedAt, finnhubNews, yahooRssNews, pruneExpiredForCompany]
    .forEach((m) => m.mockReset());
  getCompanyByTicker.mockResolvedValue(company);
  getRecentArticles.mockResolvedValue([]);
  finnhubNews.mockResolvedValue([]);
  yahooRssNews.mockResolvedValue([]);
  pruneExpiredForCompany.mockResolvedValue(0);
});

describe("getNews", () => {
  it("returns empty when the company is unknown", async () => {
    getCompanyByTicker.mockResolvedValue(undefined);
    const out = await getNews("ZZZZ");
    expect(out.articles).toEqual([]);
    expect(finnhubNews).not.toHaveBeenCalled();
  });

  it("serves cache without calling providers when news is fresh", async () => {
    newestArticleCreatedAt.mockResolvedValue(new Date()); // just now → fresh
    await getNews("NVDA");
    expect(finnhubNews).not.toHaveBeenCalled();
    expect(yahooRssNews).not.toHaveBeenCalled();
  });

  it("fetches, dedupes, and upserts when stale", async () => {
    newestArticleCreatedAt.mockResolvedValue(new Date(Date.now() - 5 * 3600_000)); // 5h → stale
    finnhubNews.mockResolvedValue([article("https://ex.com/a", "2026-05-25T08:00:00Z")]);
    yahooRssNews.mockResolvedValue([article("https://www.ex.com/a/?utm_source=z", "2026-05-25T10:00:00Z")]);
    await getNews("NVDA");
    expect(upsertArticles).toHaveBeenCalledTimes(1);
    expect(upsertArticles.mock.calls[0][1]).toHaveLength(1); // deduped to one
  });

  it("force-refreshes even when fresh", async () => {
    newestArticleCreatedAt.mockResolvedValue(new Date());
    await getNews("NVDA", { force: true });
    expect(finnhubNews).toHaveBeenCalled();
  });
});
