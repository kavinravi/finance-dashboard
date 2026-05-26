import { describe, it, expect } from "vitest";
import { parseFinnhubNews } from "./finnhub";

describe("parseFinnhubNews", () => {
  it("maps raw items, converting unix seconds to a Date", () => {
    const raw = [{
      id: 12345, datetime: 1716595200, headline: "Acme beats earnings",
      source: "MarketWatch", summary: "Strong quarter.", url: "https://ex.com/a",
      image: "https://ex.com/a.jpg", related: "ACME", category: "company",
    }];
    const out = parseFinnhubNews(raw);
    expect(out).toEqual([{
      source: "finnhub", sourceArticleId: "12345", url: "https://ex.com/a",
      title: "Acme beats earnings", summary: "Strong quarter.",
      publishedAt: new Date(1716595200 * 1000), imageUrl: "https://ex.com/a.jpg", related: "ACME",
    }]);
  });

  it("drops items missing headline or url and blanks empty optional fields", () => {
    const raw = [
      { id: 1, datetime: 1, headline: "", url: "https://x" },
      { id: 2, datetime: 2, headline: "Has title", url: "", summary: "" },
      { id: 3, datetime: 3, headline: "Keep", url: "https://y", summary: "  ", image: "", related: "" },
    ];
    const out = parseFinnhubNews(raw);
    expect(out).toHaveLength(1);
    expect(out[0]).toMatchObject({ title: "Keep", summary: null, imageUrl: null, related: null });
  });
});
