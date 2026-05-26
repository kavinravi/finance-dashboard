import { describe, it, expect } from "vitest";
import { canonicalizeUrl, urlHash, normalizeTitle, dedupeArticles } from "./dedupe";
import type { NewsArticle } from "@/lib/types";

describe("canonicalizeUrl", () => {
  it("lowercases host, drops www, strips tracking params and trailing slash", () => {
    expect(canonicalizeUrl("https://WWW.Ex.com/a/?utm_source=x&id=5#frag"))
      .toBe("https://ex.com/a?id=5");
    expect(canonicalizeUrl("https://ex.com/a/")).toBe("https://ex.com/a");
  });
  it("falls back to the trimmed string for non-URLs", () => {
    expect(canonicalizeUrl("  not a url ")).toBe("not a url");
  });
});

describe("urlHash", () => {
  it("is stable for URLs that canonicalize equally", () => {
    expect(urlHash("https://www.ex.com/a?utm_medium=y")).toBe(urlHash("https://ex.com/a"));
  });
});

describe("normalizeTitle", () => {
  it("lowercases and strips punctuation", () => {
    expect(normalizeTitle("Acme, Inc. Beats!")).toBe("acme inc beats");
  });
});

const mk = (over: Partial<NewsArticle>): NewsArticle => ({
  source: "finnhub", sourceArticleId: null, url: "https://ex.com/x", title: "T",
  summary: null, publishedAt: new Date("2026-05-25T00:00:00Z"), imageUrl: null, related: null, ...over,
});

describe("dedupeArticles", () => {
  it("collapses same canonical URL across sources, keeping the newest", () => {
    const out = dedupeArticles([
      mk({ source: "finnhub", url: "https://ex.com/a", publishedAt: new Date("2026-05-25T08:00:00Z") }),
      mk({ source: "yahoo_rss", url: "https://www.ex.com/a/?utm_source=z", publishedAt: new Date("2026-05-25T10:00:00Z") }),
    ]);
    expect(out).toHaveLength(1);
    expect(out[0].source).toBe("yahoo_rss"); // newer kept
  });
  it("collapses near-identical titles with different URLs", () => {
    const out = dedupeArticles([
      mk({ url: "https://a.com/1", title: "Acme beats earnings!" }),
      mk({ url: "https://b.com/2", title: "Acme Beats Earnings" }),
    ]);
    expect(out).toHaveLength(1);
  });
});
