import { env } from "@/lib/env";
import { getCompanyByTicker } from "@/lib/db/companies";
import { getTickerData } from "@/lib/services/price-service";
import { getNews } from "@/lib/services/news-service";
import { getRecentArticles, hasArticleNewerThan, getArticlesByIds, type ArticleRow } from "@/lib/db/articles";
import { getMemoForDate, upsertMemo } from "@/lib/db/daily-memos";
import { canCall } from "@/lib/db/provider-state";
import { generateMemo, type MemoInput, type MemoOutput } from "@/lib/providers/gemini";

export type MemoStatus = "ok" | "no_news" | "unavailable" | "error";
export type CitedArticle = { id: string; title: string; url: string; source: string };
export type MemoView = MemoOutput & { generatedAt: string; model: string; basedOnArticleCount: number };
export type MemoResult = { status: MemoStatus; memo: MemoView | null; citedArticles: CitedArticle[] };

const LOOKBACK_DAYS = 7;
const todayIso = () => new Date().toISOString().slice(0, 10);
function isoDaysAgo(n: number): string {
  const d = new Date();
  d.setUTCDate(d.getUTCDate() - n);
  return d.toISOString().slice(0, 10);
}

async function citedFrom(ids: string[]): Promise<CitedArticle[]> {
  const rows = await getArticlesByIds(ids);
  const byId = new Map(rows.map((r) => [r.id, r] as const));
  return ids
    .map((id) => byId.get(id))
    .filter((r): r is ArticleRow => !!r)
    .map((r) => ({ id: r.id, title: r.title, url: r.url, source: r.source }));
}

function view(memo: MemoOutput, generatedAt: Date, model: string, count: number): MemoView {
  return { ...memo, generatedAt: generatedAt.toISOString(), model, basedOnArticleCount: count };
}

export async function getMemo(ticker: string, opts: { force?: boolean } = {}): Promise<MemoResult> {
  const priceData = await getTickerData(ticker, "1y"); // ensures the company exists + price context
  const company = await getCompanyByTicker(ticker);
  if (!company) return { status: "no_news", memo: null, citedArticles: [] };
  const today = todayIso();

  const cached = await getMemoForDate(company.id, today);
  if (cached && !opts.force && !(await hasArticleNewerThan(company.id, cached.generatedAt))) {
    const memo = cached.summaryJson as MemoOutput;
    return {
      status: "ok",
      memo: view(memo, cached.generatedAt, cached.model, cached.basedOnArticleCount),
      citedArticles: await citedFrom(cached.sourceArticleIds as string[]),
    };
  }

  await getNews(ticker, { force: opts.force });
  const rows = await getRecentArticles(company.id, isoDaysAgo(LOOKBACK_DAYS));
  if (rows.length === 0) return { status: "no_news", memo: null, citedArticles: [] };

  if (!env.GEMINI_API_KEY || !(await canCall("gemini", env.GEMINI_DAILY_LIMIT))) {
    return { status: "unavailable", memo: null, citedArticles: [] };
  }

  const idMap = new Map<string, ArticleRow>();
  const input: MemoInput = {
    ticker: company.ticker,
    companyName: company.name,
    date: today,
    priceContext: {
      latestClose: priceData.bars.at(-1)?.close ?? null,
      currency: company.currency,
      returns: {
        d1: priceData.returns.oneDay, d5: priceData.returns.fiveDay,
        m1: priceData.returns.oneMonth, y1: priceData.returns.oneYear,
      },
    },
    articles: rows.map((r, i) => {
      const shortId = `a${i + 1}`;
      idMap.set(shortId, r);
      return { id: shortId, source: r.source, publishedAt: r.publishedAt.toISOString(), headline: r.title, summary: r.summary, related: r.related };
    }),
  };

  let out: MemoOutput;
  try {
    out = await generateMemo(input);
  } catch {
    return { status: "error", memo: null, citedArticles: [] };
  }

  // Citation guard: resolve short ids → UUIDs, drop hallucinated ids, drop zero-cite developments.
  const resolve = (ids: string[]) => ids.map((s) => idMap.get(s)?.id).filter((x): x is string => !!x);
  const fixGroup = (g: MemoOutput["bullish_developments"]) =>
    g.map((d) => ({ ...d, source_article_ids: resolve(d.source_article_ids) }))
     .filter((d) => d.source_article_ids.length > 0);

  const resolved: MemoOutput = {
    ...out,
    bullish_developments: fixGroup(out.bullish_developments),
    bearish_developments: fixGroup(out.bearish_developments),
    neutral_or_operational_updates: fixGroup(out.neutral_or_operational_updates),
  };

  const citedIds = [...new Set(
    [...resolved.bullish_developments, ...resolved.bearish_developments, ...resolved.neutral_or_operational_updates]
      .flatMap((d) => d.source_article_ids),
  )];

  await upsertMemo({
    companyId: company.id, memoDate: today, model: env.GEMINI_MODEL, summaryJson: resolved,
    toneLabel: resolved.overall_news_tone.label, toneScore: resolved.overall_news_tone.score,
    sourceArticleIds: citedIds, basedOnArticleCount: rows.length,
  });

  return {
    status: "ok",
    memo: view(resolved, new Date(), env.GEMINI_MODEL, rows.length),
    citedArticles: await citedFrom(citedIds),
  };
}
