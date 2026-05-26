import { getCompanyByTicker } from "@/lib/db/companies";
import { upsertArticles, getRecentArticles, newestArticleCreatedAt, pruneExpiredForCompany, type ArticleRow } from "@/lib/db/articles";
import { finnhub } from "@/lib/providers/finnhub";
import { yahooRss } from "@/lib/providers/yahoo-rss";
import { dedupeArticles } from "@/lib/news/dedupe";

const LOOKBACK_DAYS = 7;
const CACHE_TTL_MS = 3 * 60 * 60 * 1000; // 3h

function isoDaysAgo(n: number): string {
  const d = new Date();
  d.setUTCDate(d.getUTCDate() - n);
  return d.toISOString().slice(0, 10);
}

export async function getNews(
  ticker: string,
  opts: { force?: boolean } = {},
): Promise<{ articles: ArticleRow[]; asOf: Date | null }> {
  const company = await getCompanyByTicker(ticker);
  if (!company) return { articles: [], asOf: null };

  const newest = await newestArticleCreatedAt(company.id);
  const fresh = newest !== null && Date.now() - newest.getTime() < CACHE_TTL_MS;

  if (opts.force || !fresh) {
    const fromIso = isoDaysAgo(LOOKBACK_DAYS);
    const toIso = new Date().toISOString().slice(0, 10);
    const [fh, yr] = await Promise.all([
      finnhub.companyNews(company.ticker, fromIso, toIso), // each degrades to [] internally
      yahooRss.companyNews(company.ticker),
    ]);
    await upsertArticles(company.id, dedupeArticles([...fh, ...yr]));
    await pruneExpiredForCompany(company.id).catch(() => 0); // best-effort; never break the news path
  }

  const rows = await getRecentArticles(company.id, isoDaysAgo(LOOKBACK_DAYS), 10);
  return { articles: rows, asOf: rows[0]?.publishedAt ?? null };
}
