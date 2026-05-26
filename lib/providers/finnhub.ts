import { env } from "@/lib/env";
import { recordSuccess, recordError } from "@/lib/db/provider-state";
import type { NewsArticle } from "@/lib/types";

const BASE = "https://finnhub.io/api/v1";
// Not hard-gated — Finnhub free is 60/min with no daily cap, ample for one user.
// The value only feeds the provider_state health row.
export const FINNHUB_DAILY_LIMIT = 100000;

function nonEmpty(v: unknown): string | null {
  return typeof v === "string" && v.trim() !== "" ? v : null;
}

export function parseFinnhubNews(raw: any[]): NewsArticle[] {
  return (raw ?? [])
    .filter((r) => r && r.headline && r.url)
    .map((r) => ({
      source: "finnhub" as const,
      sourceArticleId: r.id != null ? String(r.id) : null,
      url: String(r.url),
      title: String(r.headline),
      summary: nonEmpty(r.summary),
      publishedAt: new Date(Number(r.datetime) * 1000),
      imageUrl: nonEmpty(r.image),
      related: nonEmpty(r.related),
    }));
}

export const finnhub = {
  name: "finnhub",
  async companyNews(ticker: string, fromIso: string, toIso: string): Promise<NewsArticle[]> {
    if (!env.FINNHUB_API_KEY) return [];
    const url = `${BASE}/company-news?symbol=${encodeURIComponent(ticker)}&from=${fromIso}&to=${toIso}`;
    try {
      const res = await fetch(url, { headers: { "X-Finnhub-Token": env.FINNHUB_API_KEY } });
      if (!res.ok) throw new Error(`Finnhub ${res.status}`);
      const data = parseFinnhubNews(await res.json());
      await recordSuccess("finnhub", FINNHUB_DAILY_LIMIT);
      return data;
    } catch (e) {
      await recordError("finnhub", FINNHUB_DAILY_LIMIT, String(e));
      return []; // degrade — never throw
    }
  },
};
