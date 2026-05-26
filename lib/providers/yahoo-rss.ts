import { XMLParser } from "fast-xml-parser";
import { recordSuccess, recordError } from "@/lib/db/provider-state";
import type { NewsArticle } from "@/lib/types";

const YAHOO_RSS_LIMIT = 100000; // health-row only; not gated
const parser = new XMLParser({ ignoreAttributes: true, htmlEntities: true });

function stripHtml(s: string): string {
  return s.replace(/<[^>]*>/g, "").replace(/\s+/g, " ").trim();
}

export function parseYahooRss(xml: string): NewsArticle[] {
  let doc: any;
  try { doc = parser.parse(xml); } catch { return []; }
  const items = doc?.rss?.channel?.item;
  const arr = Array.isArray(items) ? items : items ? [items] : [];
  return arr
    .filter((it: any) => it && it.title && it.link)
    .map((it: any) => {
      const summary = it.description != null ? stripHtml(String(it.description)) : "";
      return {
        source: "yahoo_rss" as const,
        sourceArticleId: null,
        url: String(it.link),
        title: stripHtml(String(it.title)),
        summary: summary === "" ? null : summary,
        publishedAt: it.pubDate ? new Date(String(it.pubDate)) : new Date(),
        imageUrl: null,
        related: null,
      };
    });
}

export const yahooRss = {
  name: "yahoo_rss",
  async companyNews(ticker: string): Promise<NewsArticle[]> {
    const url = `https://finance.yahoo.com/rss/headline?s=${encodeURIComponent(ticker)}`;
    try {
      const res = await fetch(url, { headers: { Accept: "application/rss+xml, application/xml" } });
      if (!res.ok) throw new Error(`Yahoo RSS ${res.status}`);
      const data = parseYahooRss(await res.text());
      await recordSuccess("yahoo_rss", YAHOO_RSS_LIMIT);
      return data;
    } catch (e) {
      await recordError("yahoo_rss", YAHOO_RSS_LIMIT, String(e));
      return []; // scrape-fragile — degrade silently
    }
  },
};
