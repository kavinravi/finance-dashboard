import { describe, it, expect } from "vitest";
import { parseYahooRss } from "./yahoo-rss";

const xml = `<?xml version="1.0"?><rss version="2.0"><channel>
  <item><title>Acme &amp; Co rallies</title><link>https://ex.com/1</link>
    <pubDate>Mon, 25 May 2026 12:00:00 GMT</pubDate>
    <description>&lt;p&gt;Shares up &lt;b&gt;5%&lt;/b&gt;.&lt;/p&gt;</description></item>
  <item><title>Second story</title><link>https://ex.com/2</link>
    <pubDate>Mon, 25 May 2026 09:00:00 GMT</pubDate></item>
</channel></rss>`;

describe("parseYahooRss", () => {
  it("maps items, strips HTML, and decodes entities", () => {
    const out = parseYahooRss(xml);
    expect(out).toHaveLength(2);
    expect(out[0]).toMatchObject({
      source: "yahoo_rss", sourceArticleId: null, url: "https://ex.com/1",
      title: "Acme & Co rallies", summary: "Shares up 5%.", imageUrl: null, related: null,
    });
    expect(out[0].publishedAt instanceof Date).toBe(true);
    expect(out[1].summary).toBeNull();
  });

  it("returns [] for empty or malformed feeds", () => {
    expect(parseYahooRss("<rss><channel></channel></rss>")).toEqual([]);
    expect(parseYahooRss("not xml")).toEqual([]);
  });

  it("falls back to a valid date when pubDate is unparseable", () => {
    const bad = `<rss version="2.0"><channel>
      <item><title>Bad date</title><link>https://ex.com/x</link><pubDate>not-a-date</pubDate></item>
    </channel></rss>`;
    const out = parseYahooRss(bad);
    expect(out).toHaveLength(1);
    expect(isNaN(out[0].publishedAt.getTime())).toBe(false);
  });
});
