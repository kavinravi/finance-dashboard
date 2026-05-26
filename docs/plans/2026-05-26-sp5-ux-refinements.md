# SP5 — UX Refinements Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Post-deploy UX refinements: cap news to the 10 most-recent, split the ticker page into two bookmarkable tabbed routes (Charts & Fundamentals / News & Memo), add RSI + MACD charts, and add a DB-backed multi-stock watchlist overlay alongside the existing 2-stock compare.

**Architecture:** A shared `app/ticker/[symbol]/layout.tsx` holds a persistent tab bar over two route segments (`page.tsx` = charts/fundamentals, `news/page.tsx` = news/memo) — the Gemini memo only runs when the News tab is opened. RSI/MACD reuse the already-computed indicator series (MACD line/signal newly surfaced). The watchlist is one global DB table; a pure `buildOverlay` normalizes N tickers' closes to 100 at the first common date and a client manager adds/removes via a gated API.

**Tech Stack:** Next.js 16 (App Router, nested routes), TypeScript, Recharts, Drizzle + Neon Postgres, Zod, Vitest, Playwright.

**Spec:** `docs/specs/2026-05-25-finance-dashboard-rebuild-design.md` §10.

**Standing constraints:**
- No AI-authorship traces in commits/docs/branches (no "Claude"/"Anthropic"/"Co-Authored-By"/tool names). Git author `kavinravi` is correct.
- `gemini-3.5-flash` is the verified 2026 stable model — don't let any reviewer "fix" it.
- Do NOT run `pnpm lint`/eslint locally (OOM-crashes); lint is verified in CI. Next 16 doesn't lint during `build`.
- The Next middleware convention is `proxy.ts` (not `middleware.ts`) in this repo — don't reintroduce `middleware.ts`.
- Tests: `pnpm test` (Vitest; integration hits live Neon via the `dotenv/config` setup), `pnpm test:e2e` (Playwright; the gate is ON via `playwright.config.ts` `webServer.env`, and a `setup` project saves an authenticated `storageState` — existing specs run authenticated). Type check: `pnpm exec tsc --noEmit` (ignore any errors under `.next/` — those are stale dev-server artifacts; a clean check is `rm -rf .next && pnpm build`). Migrations: `pnpm db:generate` then `pnpm db:migrate`.
- After the build, do a live smoke + screenshots before declaring done.

**Verified context (2026-05-26):**
- `lib/services/price-service.ts` `getTickerData` returns `indicators: { ma10, ma20, ma50, rsi14, macdHistogram, volatility5d }` — `rsi14` is a full `(number|null)[]`; `macd()` (in `lib/indicators/macd.ts`) returns `{ macdLine, signalLine, histogram }` (all `number[]`) but only `histogram` is currently surfaced. No consumer reads `indicators.macdHistogram` today.
- `lib/db/articles.ts` `getRecentArticles(companyId, sinceIso)` orders by `publishedAt desc`, no limit. Callers: `news-service.getNews` and `memo-service`.
- Existing charts (`components/price-chart.tsx`, `comparison-chart.tsx`) use Recharts; dark theme colors: grid `#262626`, axis text `#737373`, tooltip bg `#171717`/border `#404040`, series `#e5e5e5`/`#38bdf8`/`#f59e0b`.
- `components/app-nav.tsx` nav currently: Home / Health / Log out (hidden on `/login`).
- Ticker regex used elsewhere: `/^[A-Za-z.\-]{1,10}$/`.
- The current single `app/ticker/[symbol]/page.tsx` renders: back-link, symbol h1, price, StalenessBadge, ReturnsTable, PriceChart, compare form, MemoCard, FundamentalsCard, NewsTable.
- `tests/e2e/smoke.spec.ts` test 3 asserts memo + "Recent news" + fundamentals all on `/ticker/NVDA` — it MUST be updated when the page splits (Task 4).

---

## File Structure

**Create:**
- `components/rsi-chart.tsx` — RSI line chart (0–100, 30/70 bands).
- `components/macd-chart.tsx` — MACD line + signal + histogram.
- `app/ticker/[symbol]/layout.tsx` — shared header + tab bar.
- `components/ticker-tabs.tsx` — client tab bar (active via `usePathname`).
- `app/ticker/[symbol]/news/page.tsx` — News & Memo view.
- `lib/db/watchlist.ts` — watchlist repo.
- `lib/db/watchlist.test.ts` — repo integration test (live Neon).
- `lib/services/watchlist-overlay.ts` — pure `buildOverlay`.
- `lib/services/watchlist-overlay.test.ts` — unit test.
- `lib/services/watchlist-service.ts` — `getWatchlistOverlay` (repo + price-service + buildOverlay).
- `app/api/watchlist/route.ts` — gated GET/POST/DELETE.
- `components/overlay-chart.tsx` — N-series normalized overlay.
- `components/watchlist-manager.tsx` — client add/remove.
- `app/watchlist/page.tsx` — watchlist page.
- `lib/db/news-cap.test.ts` — `getRecentArticles` limit integration test.

**Modify:**
- `lib/db/articles.ts` — add `limit = 10` to `getRecentArticles`.
- `lib/services/price-service.ts` — surface `macdLine`/`macdSignal`/`macdHistogram`; update `TickerData` type.
- `app/ticker/[symbol]/page.tsx` — becomes the Charts & Fundamentals view (price/RSI/MACD/returns/compare/fundamentals); header/back-link move to the layout.
- `lib/db/schema.ts` — add `watchlist` table.
- `components/app-nav.tsx` — add "Watchlist" link.
- `tests/e2e/smoke.spec.ts` — update test 3 for the split routes; add tab-nav + watchlist coverage (Task 9).

---

## Task 1: Cap news to the 10 most-recent

**Files:**
- Modify: `lib/db/articles.ts`
- Test: `lib/db/news-cap.test.ts`

- [ ] **Step 1: Write the failing integration test** — Create `lib/db/news-cap.test.ts`:

```ts
import { describe, it, expect, afterAll } from "vitest";
import { eq } from "drizzle-orm";
import { db } from "@/lib/db/client";
import { companies, articles } from "@/lib/db/schema";
import { getRecentArticles } from "@/lib/db/articles";

const TICKER = `ZZNEWS${Date.now() % 100000}`;
let companyId = "";

afterAll(async () => {
  if (companyId) {
    await db.delete(articles).where(eq(articles.companyId, companyId));
    await db.delete(companies).where(eq(companies.id, companyId));
  }
});

describe("getRecentArticles cap (integration, live Neon)", () => {
  it("returns at most the limit, newest first", async () => {
    const [c] = await db.insert(companies).values({ ticker: TICKER, name: "News Cap Co" }).returning({ id: companies.id });
    companyId = c.id;
    const now = Date.now();
    const rows = Array.from({ length: 14 }, (_, i) => ({
      companyId, source: "finnhub", url: `https://ex.com/${TICKER}/${i}`, urlHash: `${TICKER}-${i}`,
      title: `Article ${i}`, publishedAt: new Date(now - i * 60_000),
    }));
    await db.insert(articles).values(rows);

    const since = new Date(now - 7 * 86_400_000).toISOString().slice(0, 10);
    const got = await getRecentArticles(companyId, since, 10);
    expect(got).toHaveLength(10);
    expect(got[0].title).toBe("Article 0"); // newest first
  });
});
```

- [ ] **Step 2: Run the test to verify it fails** — Run: `pnpm exec vitest run lib/db/news-cap.test.ts` — Expected: FAIL (`getRecentArticles` takes 2 args / extra arg ignored → returns 14, not 10).

- [ ] **Step 3: Add the limit** — In `lib/db/articles.ts`, change `getRecentArticles` to:

```ts
export async function getRecentArticles(companyId: string, sinceIso: string, limit = 10): Promise<ArticleRow[]> {
  return db.select().from(articles)
    .where(and(eq(articles.companyId, companyId), gte(articles.publishedAt, new Date(sinceIso))))
    .orderBy(desc(articles.publishedAt))
    .limit(limit);
}
```

(The `limit = 10` default means both callers — `news-service.getNews` and `memo-service` — automatically cap to 10 with no change. The news table and the memo's Gemini input both shrink to ≤10.)

- [ ] **Step 4: Run the test to verify it passes** — Run: `pnpm exec vitest run lib/db/news-cap.test.ts` — Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add lib/db/articles.ts lib/db/news-cap.test.ts
git commit -m "Cap recent-article queries to the 10 newest"
```

---

## Task 2: Surface MACD line + signal in the price service

**Files:**
- Modify: `lib/services/price-service.ts`

- [ ] **Step 1: Update the `TickerData` indicators type** — In `lib/services/price-service.ts`, change the `indicators` field of the `TickerData` type to:

```ts
  indicators: {
    ma10: (number | null)[]; ma20: (number | null)[]; ma50: (number | null)[];
    rsi14: (number | null)[];
    macdLine: number[]; macdSignal: number[]; macdHistogram: number[];
    volatility5d: (number | null)[];
  };
```

- [ ] **Step 2: Compute and return the full MACD triple** — In the same file, replace the `indicators:` object built in the return of `getTickerData` (currently `macdHistogram: macd(closes).histogram`) with a single `macd()` call surfaced as three series:

```ts
  const closes = bars.map((b) => b.close);
  const lastBarDate = bars.at(-1)?.date ?? null;
  const m = macd(closes);
  return {
    ticker: company.ticker,
    bars,
    returns: computeReturns(bars),
    indicators: {
      ma10: sma(closes, 10), ma20: sma(closes, 20), ma50: sma(closes, 50),
      rsi14: rsi(closes, 14),
      macdLine: m.macdLine, macdSignal: m.signalLine, macdHistogram: m.histogram,
      volatility5d: rollingVolatility(closes, 5),
    },
    source,
    lastBarDate,
    stale: lastBarDate !== null && lastBarDate < lastTradingDayIso(),
  };
```

- [ ] **Step 3: Type-check + confirm no existing test broke** — Run: `pnpm exec tsc --noEmit` then `pnpm exec vitest run lib/services lib/indicators` — Expected: tsc clean; existing service/indicator tests still pass (no consumer read `macdHistogram` before).

- [ ] **Step 4: Commit**

```bash
git add lib/services/price-service.ts
git commit -m "Surface MACD line and signal series from the price service"
```

---

## Task 3: RSI + MACD chart components

**Files:**
- Create: `components/rsi-chart.tsx`
- Create: `components/macd-chart.tsx`

- [ ] **Step 1: Implement `components/rsi-chart.tsx`**

```tsx
"use client";
import { LineChart, Line, XAxis, YAxis, Tooltip, ResponsiveContainer, CartesianGrid, ReferenceLine } from "recharts";
import type { PriceBar } from "@/lib/types";

type Props = { bars: PriceBar[]; rsi14: (number | null)[] };

export function RsiChart({ bars, rsi14 }: Props) {
  const data = bars.map((b, i) => ({ date: b.date, rsi: rsi14[i] }));
  return (
    <div className="h-40 w-full">
      <ResponsiveContainer>
        <LineChart data={data} margin={{ top: 8, right: 8, bottom: 0, left: 0 }}>
          <CartesianGrid stroke="#262626" vertical={false} />
          <XAxis dataKey="date" tick={{ fontSize: 10, fill: "#737373" }} minTickGap={48} />
          <YAxis domain={[0, 100]} ticks={[0, 30, 70, 100]} tick={{ fontSize: 10, fill: "#737373" }} width={48} />
          <Tooltip contentStyle={{ background: "#171717", border: "1px solid #404040", fontSize: 12 }} />
          <ReferenceLine y={70} stroke="#ef4444" strokeDasharray="3 3" />
          <ReferenceLine y={30} stroke="#22c55e" strokeDasharray="3 3" />
          <Line type="monotone" dataKey="rsi" stroke="#a78bfa" dot={false} strokeWidth={1.5} />
        </LineChart>
      </ResponsiveContainer>
    </div>
  );
}
```

- [ ] **Step 2: Implement `components/macd-chart.tsx`**

```tsx
"use client";
import { ComposedChart, Line, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer, CartesianGrid, Legend } from "recharts";
import type { PriceBar } from "@/lib/types";

type Props = { bars: PriceBar[]; macdLine: number[]; macdSignal: number[]; macdHistogram: number[] };

export function MacdChart({ bars, macdLine, macdSignal, macdHistogram }: Props) {
  const data = bars.map((b, i) => ({ date: b.date, macd: macdLine[i], signal: macdSignal[i], hist: macdHistogram[i] }));
  return (
    <div className="h-40 w-full">
      <ResponsiveContainer>
        <ComposedChart data={data} margin={{ top: 8, right: 8, bottom: 0, left: 0 }}>
          <CartesianGrid stroke="#262626" vertical={false} />
          <XAxis dataKey="date" tick={{ fontSize: 10, fill: "#737373" }} minTickGap={48} />
          <YAxis tick={{ fontSize: 10, fill: "#737373" }} width={48} />
          <Tooltip contentStyle={{ background: "#171717", border: "1px solid #404040", fontSize: 12 }} />
          <Legend wrapperStyle={{ fontSize: 12 }} />
          <Bar dataKey="hist" fill="#525252" />
          <Line type="monotone" dataKey="macd" stroke="#e5e5e5" dot={false} strokeWidth={1.5} />
          <Line type="monotone" dataKey="signal" stroke="#38bdf8" dot={false} strokeWidth={1} />
        </ComposedChart>
      </ResponsiveContainer>
    </div>
  );
}
```

- [ ] **Step 3: Type-check** — Run: `pnpm exec tsc --noEmit` — Expected: no errors.

- [ ] **Step 4: Commit**

```bash
git add components/rsi-chart.tsx components/macd-chart.tsx
git commit -m "Add RSI and MACD chart components"
```

---

## Task 4: Split the ticker page into tabbed routes

**Files:**
- Create: `app/ticker/[symbol]/layout.tsx`
- Create: `components/ticker-tabs.tsx`
- Create: `app/ticker/[symbol]/news/page.tsx`
- Modify: `app/ticker/[symbol]/page.tsx` (becomes the Charts & Fundamentals view)
- Modify: `tests/e2e/smoke.spec.ts` (update test 3 for the split)

- [ ] **Step 1: Implement `components/ticker-tabs.tsx`**

```tsx
"use client";
import Link from "next/link";
import { usePathname } from "next/navigation";

export function TickerTabs({ symbol }: { symbol: string }) {
  const pathname = usePathname();
  const onNews = pathname.endsWith("/news");
  const base = `/ticker/${symbol}`;
  const cls = (active: boolean) =>
    `pb-2 text-sm ${active ? "border-b-2 border-neutral-200 text-neutral-100" : "text-neutral-500 hover:text-neutral-300"}`;
  return (
    <nav className="mt-4 flex gap-6 border-b border-neutral-800">
      <Link href={base} className={cls(!onNews)}>Charts &amp; Fundamentals</Link>
      <Link href={`${base}/news`} className={cls(onNews)}>News &amp; Memo</Link>
    </nav>
  );
}
```

- [ ] **Step 2: Implement `app/ticker/[symbol]/layout.tsx`** (back-link + symbol + persistent tabs; no data fetch)

```tsx
import Link from "next/link";
import { TickerTabs } from "@/components/ticker-tabs";

export default async function TickerLayout({
  children, params,
}: { children: React.ReactNode; params: Promise<{ symbol: string }> }) {
  const { symbol } = await params;
  const ticker = symbol.toUpperCase();
  return (
    <main className="mx-auto max-w-4xl px-4 pb-24 pt-10">
      <Link href="/" className="text-sm text-neutral-500">← Search</Link>
      <h1 className="mt-4 font-mono text-3xl font-semibold">{ticker}</h1>
      <TickerTabs symbol={ticker} />
      {children}
    </main>
  );
}
```

- [ ] **Step 3: Rewrite `app/ticker/[symbol]/page.tsx`** as the Charts & Fundamentals view (back-link/symbol now live in the layout; adds RSI + MACD; memo + news removed)

```tsx
import { getTickerData } from "@/lib/services/price-service";
import { ReturnsTable } from "@/components/returns-table";
import { PriceChart } from "@/components/price-chart";
import { RsiChart } from "@/components/rsi-chart";
import { MacdChart } from "@/components/macd-chart";
import { StalenessBadge } from "@/components/staleness-badge";
import { FundamentalsCard } from "@/components/fundamentals-card";
import { formatPrice } from "@/lib/formatters";

export const dynamic = "force-dynamic";

export default async function TickerChartsPage({ params }: { params: Promise<{ symbol: string }> }) {
  const { symbol } = await params;
  const data = await getTickerData(symbol.toUpperCase(), "1y");
  const latest = data.bars.at(-1)?.close ?? null;

  if (data.bars.length === 0) {
    return (
      <p className="mt-8">
        Couldn&apos;t resolve <span className="font-mono">{symbol.toUpperCase()}</span>. Try another ticker.
      </p>
    );
  }

  return (
    <div className="mt-6">
      <div className="flex items-baseline justify-between">
        <div className="text-2xl">{formatPrice(latest)}</div>
        <StalenessBadge stale={data.stale} lastBarDate={data.lastBarDate} source={data.source} />
      </div>

      <div className="mt-4"><ReturnsTable returns={data.returns} /></div>
      <div className="mt-6"><PriceChart bars={data.bars} ma20={data.indicators.ma20} ma50={data.indicators.ma50} /></div>

      <h2 className="mt-8 text-sm font-medium text-neutral-400">RSI (14)</h2>
      <div className="mt-2"><RsiChart bars={data.bars} rsi14={data.indicators.rsi14} /></div>

      <h2 className="mt-8 text-sm font-medium text-neutral-400">MACD (12/26/9)</h2>
      <div className="mt-2">
        <MacdChart bars={data.bars} macdLine={data.indicators.macdLine} macdSignal={data.indicators.macdSignal} macdHistogram={data.indicators.macdHistogram} />
      </div>

      <form action="/compare" className="mt-8 flex items-center gap-2">
        <input type="hidden" name="primary" value={data.ticker} />
        <label className="text-sm text-neutral-400">Compare against</label>
        <input name="comparison" placeholder="e.g. SPY"
          className="rounded bg-neutral-900 px-2 py-1 font-mono uppercase ring-1 ring-neutral-800" />
        <button className="rounded bg-neutral-200 px-3 py-1 text-sm font-medium text-neutral-900">Go</button>
      </form>

      <section className="mt-10">
        <h2 className="text-sm font-medium text-neutral-400">Fundamentals</h2>
        <p className="mb-2 text-xs text-neutral-600">From official SEC filings.</p>
        <FundamentalsCard symbol={data.ticker} />
      </section>
    </div>
  );
}
```

- [ ] **Step 4: Implement `app/ticker/[symbol]/news/page.tsx`** (ensures the company exists so deep-linked news resolves, then renders memo + capped news)

```tsx
import { getTickerData } from "@/lib/services/price-service";
import { getNews } from "@/lib/services/news-service";
import { MemoCard } from "@/components/memo-card";
import { NewsTable } from "@/components/news-table";

export const dynamic = "force-dynamic";

export default async function TickerNewsPage({ params }: { params: Promise<{ symbol: string }> }) {
  const { symbol } = await params;
  const ticker = symbol.toUpperCase();
  await getTickerData(ticker, "1y"); // ensure the company row exists for deep links
  const news = await getNews(ticker);

  return (
    <div className="mt-6">
      <section>
        <h2 className="text-sm font-medium text-neutral-400">Daily memo</h2>
        <p className="mb-2 text-xs text-neutral-600">Research assistant, not investment advice.</p>
        <MemoCard symbol={ticker} />
      </section>

      <section className="mt-10">
        <h2 className="text-sm font-medium text-neutral-400">Recent news</h2>
        <div className="mt-2"><NewsTable articles={news.articles} /></div>
      </section>
    </div>
  );
}
```

- [ ] **Step 5: Update `tests/e2e/smoke.spec.ts` test 3** for the split — replace the third test (`"ticker page renders the news section and a memo from an intercepted response"`) with this version that keeps both interceptions but visits both routes (the first two tests are unchanged — test 1 still lands on `/ticker/NVDA` and sees the chart):

```ts
test("charts tab shows fundamentals; news tab shows the memo (intercepted)", async ({ page }) => {
  await page.route("**/api/memo/**", (route) =>
    route.fulfill({
      status: 200,
      contentType: "application/json",
      body: JSON.stringify({
        status: "ok",
        citedArticles: [{ id: "u1", title: "Source one", url: "https://ex.com/1", source: "finnhub" }],
        memo: {
          ticker: "NVDA", date: "2026-05-26", one_sentence_takeaway: "Fixture takeaway for NVDA.",
          bullish_developments: [{ claim: "Demand strong", why_it_matters: "revenue", source_article_ids: ["u1"], confidence: "medium" }],
          bearish_developments: [], neutral_or_operational_updates: [], watch_items: [], caveats: [],
          overall_news_tone: { label: "somewhat_bullish", score: 64, rationale: "Positive coverage." },
          generatedAt: new Date().toISOString(), model: "gemini-3.5-flash", basedOnArticleCount: 1,
        },
      }),
    }),
  );
  await page.route("**/api/fundamentals/**", (route) =>
    route.fulfill({
      status: 200,
      contentType: "application/json",
      body: JSON.stringify({
        status: "ok", source: "sec_edgar",
        asOf: { fiscalYear: 2024, incomePeriodEnd: "2024-09-28", balanceSheetAsOf: "2024-12-28", filingForm: "10-K", filedAt: "2024-11-01", edgarUrl: "https://www.sec.gov/cgi-bin/browse-edgar?action=getcompany&CIK=0000320193&type=10-K" },
        view: { marketCap: 3420000000000, peRatio: 28.41, psRatio: 8.7, grossMargin: 0.462, roe: 1.5, roa: 0.28, operatingIncome: 123216000000, currentRatio: 0.92, debtToEquity: 4.15, assets: 364980000000, liabilities: 308030000000, equity: 56950000000, revenue: 391035000000, netIncome: 93736000000, eps: 6.08 },
      }),
    }),
  );

  // Charts tab: fundamentals + price chart
  await page.goto("/ticker/NVDA");
  await expect(page.getByRole("heading", { name: "Fundamentals" })).toBeVisible();
  await expect(page.getByText("$3.42T")).toBeVisible();
  await expect(page.locator("svg .recharts-line").first()).toBeVisible();

  // Switch to the News & Memo tab
  await page.getByRole("link", { name: /News & Memo/i }).click();
  await expect(page).toHaveURL(/\/ticker\/NVDA\/news/);
  await expect(page.getByText("Fixture takeaway for NVDA.")).toBeVisible();
  await expect(page.getByText(/News Tone/).first()).toBeVisible();
  await expect(page.getByRole("heading", { name: "Recent news" })).toBeVisible();
});
```

- [ ] **Step 6: Type-check** — Run: `pnpm exec tsc --noEmit` — Expected: no errors.

- [ ] **Step 7: Commit**

```bash
git add app/ticker/[symbol]/layout.tsx components/ticker-tabs.tsx "app/ticker/[symbol]/news/page.tsx" app/ticker/[symbol]/page.tsx tests/e2e/smoke.spec.ts
git commit -m "Split ticker page into Charts and News tabbed routes with RSI/MACD"
```

(Full E2E run happens in Task 9.)

---

## Task 5: Watchlist table + repo

**Files:**
- Modify: `lib/db/schema.ts`
- Create: `lib/db/watchlist.ts`
- Test: `lib/db/watchlist.test.ts`
- Migration: generated under `drizzle/`

- [ ] **Step 1: Add the `watchlist` table to `lib/db/schema.ts`** — append:

```ts
export const watchlist = pgTable("watchlist", {
  id: uuid("id").primaryKey().defaultRandom(),
  ticker: text("ticker").notNull().unique(),
  createdAt: timestamp("created_at", { withTimezone: true }).defaultNow().notNull(),
});
```

(`pgTable`, `uuid`, `text`, `timestamp` are already imported at the top of the file.)

- [ ] **Step 2: Generate + apply the migration**

Run: `pnpm db:generate` (creates a new `drizzle/00NN_*.sql`), then `pnpm db:migrate` (applies it to Neon).
Expected: a new migration file is created and applied; no errors.

- [ ] **Step 3: Write the failing repo integration test** — Create `lib/db/watchlist.test.ts`:

```ts
import { describe, it, expect, afterAll } from "vitest";
import { getWatchlist, addToWatchlist, removeFromWatchlist } from "@/lib/db/watchlist";

const T = `ZZ${Date.now() % 100000}`; // short, regex-valid, uppercase

afterAll(async () => { await removeFromWatchlist(T); });

describe("watchlist repo (integration, live Neon)", () => {
  it("adds (uppercased), lists, dedupes, and removes", async () => {
    await addToWatchlist(T.toLowerCase());
    let tickers = (await getWatchlist()).map((w) => w.ticker);
    expect(tickers).toContain(T);

    await addToWatchlist(T); // dedupe — no throw, no duplicate
    tickers = (await getWatchlist()).map((w) => w.ticker);
    expect(tickers.filter((x) => x === T)).toHaveLength(1);

    await removeFromWatchlist(T);
    tickers = (await getWatchlist()).map((w) => w.ticker);
    expect(tickers).not.toContain(T);
  });
});
```

- [ ] **Step 4: Run the test to verify it fails** — Run: `pnpm exec vitest run lib/db/watchlist.test.ts` — Expected: FAIL (`Cannot find module '@/lib/db/watchlist'`).

- [ ] **Step 5: Implement `lib/db/watchlist.ts`**

```ts
import { db } from "./client";
import { watchlist } from "./schema";
import { eq, asc } from "drizzle-orm";

export type WatchlistRow = typeof watchlist.$inferSelect;

export async function getWatchlist(): Promise<WatchlistRow[]> {
  return db.select().from(watchlist).orderBy(asc(watchlist.createdAt));
}

export async function addToWatchlist(ticker: string): Promise<void> {
  await db.insert(watchlist).values({ ticker: ticker.toUpperCase() }).onConflictDoNothing();
}

export async function removeFromWatchlist(ticker: string): Promise<void> {
  await db.delete(watchlist).where(eq(watchlist.ticker, ticker.toUpperCase()));
}
```

- [ ] **Step 6: Run the test to verify it passes** — Run: `pnpm exec vitest run lib/db/watchlist.test.ts` — Expected: PASS.

- [ ] **Step 7: Commit**

```bash
git add lib/db/schema.ts lib/db/watchlist.ts lib/db/watchlist.test.ts drizzle/
git commit -m "Add watchlist table and repository"
```

---

## Task 6: Watchlist overlay (pure) + service

**Files:**
- Create: `lib/services/watchlist-overlay.ts`
- Test: `lib/services/watchlist-overlay.test.ts`
- Create: `lib/services/watchlist-service.ts`

- [ ] **Step 1: Write the failing unit test** — Create `lib/services/watchlist-overlay.test.ts`:

```ts
import { describe, it, expect } from "vitest";
import { buildOverlay } from "./watchlist-overlay";

const bar = (date: string, close: number) => ({ date, close });

describe("buildOverlay", () => {
  it("normalizes each series to 100 at the first common date", () => {
    const o = buildOverlay([
      { ticker: "A", bars: [bar("2026-01-01", 10), bar("2026-01-02", 11), bar("2026-01-03", 12)] },
      { ticker: "B", bars: [bar("2026-01-01", 20), bar("2026-01-02", 25), bar("2026-01-03", 20)] },
    ]);
    expect(o.dates).toEqual(["2026-01-01", "2026-01-02", "2026-01-03"]);
    expect(o.series[0]).toEqual({ ticker: "A", normalized: [100, 110, 120] });
    expect(o.series[1]).toEqual({ ticker: "B", normalized: [100, 125, 100] });
  });

  it("intersects on common dates only", () => {
    const o = buildOverlay([
      { ticker: "A", bars: [bar("2026-01-01", 10), bar("2026-01-02", 11)] },
      { ticker: "B", bars: [bar("2026-01-02", 20), bar("2026-01-03", 22)] },
    ]);
    expect(o.dates).toEqual(["2026-01-02"]);
    expect(o.series[0].normalized).toEqual([100]);
    expect(o.series[1].normalized).toEqual([100]);
  });

  it("drops empty-bar tickers and returns empty for all-empty input", () => {
    expect(buildOverlay([{ ticker: "A", bars: [] }])).toEqual({ dates: [], series: [] });
    expect(buildOverlay([])).toEqual({ dates: [], series: [] });
  });
});
```

- [ ] **Step 2: Run the test to verify it fails** — Run: `pnpm exec vitest run lib/services/watchlist-overlay.test.ts` — Expected: FAIL (`Cannot find module './watchlist-overlay'`).

- [ ] **Step 3: Implement `lib/services/watchlist-overlay.ts`**

```ts
export type OverlayInput = { ticker: string; bars: { date: string; close: number }[] };
export type Overlay = { dates: string[]; series: { ticker: string; normalized: number[] }[] };

export function buildOverlay(items: OverlayInput[]): Overlay {
  const valid = items.filter((it) => it.bars.length > 0);
  if (valid.length === 0) return { dates: [], series: [] };

  const maps = valid.map((it) => new Map(it.bars.map((b) => [b.date, b.close])));
  let common = [...maps[0].keys()];
  for (let i = 1; i < maps.length; i++) common = common.filter((d) => maps[i].has(d));
  common.sort();

  const series = valid.map((it, i) => {
    const closes = common.map((d) => maps[i].get(d)!);
    const base = closes[0];
    return { ticker: it.ticker, normalized: base ? closes.map((c) => (c / base) * 100) : [] };
  });
  return { dates: common, series };
}
```

- [ ] **Step 4: Run the test to verify it passes** — Run: `pnpm exec vitest run lib/services/watchlist-overlay.test.ts` — Expected: PASS.

- [ ] **Step 5: Implement `lib/services/watchlist-service.ts`** (wires the repo + price-service + pure overlay; degrades per-ticker)

```ts
import { getWatchlist } from "@/lib/db/watchlist";
import { getTickerData } from "@/lib/services/price-service";
import { buildOverlay, type Overlay, type OverlayInput } from "./watchlist-overlay";

const MAX_TICKERS = 10;

export async function getWatchlistOverlay(): Promise<{ tickers: string[]; overlay: Overlay }> {
  const rows = await getWatchlist();
  const tickers = rows.slice(0, MAX_TICKERS).map((r) => r.ticker);
  if (tickers.length === 0) return { tickers: [], overlay: { dates: [], series: [] } };

  const datas = await Promise.all(
    tickers.map((t) =>
      getTickerData(t, "1y")
        .then((d): OverlayInput => ({ ticker: d.ticker, bars: d.bars }))
        .catch(() => null),
    ),
  );
  const items = datas.filter((d): d is OverlayInput => d !== null);
  return { tickers, overlay: buildOverlay(items) };
}
```

- [ ] **Step 6: Type-check** — Run: `pnpm exec tsc --noEmit` — Expected: no errors.

- [ ] **Step 7: Commit**

```bash
git add lib/services/watchlist-overlay.ts lib/services/watchlist-overlay.test.ts lib/services/watchlist-service.ts
git commit -m "Add watchlist overlay normalization and service"
```

---

## Task 7: Watchlist API route

**Files:**
- Create: `app/api/watchlist/route.ts`

- [ ] **Step 1: Implement `app/api/watchlist/route.ts`** (gated by `proxy.ts`; not in the allow-list)

```ts
import { NextResponse } from "next/server";
import { z } from "zod";
import { getWatchlist, addToWatchlist, removeFromWatchlist } from "@/lib/db/watchlist";

export const runtime = "nodejs";
export const dynamic = "force-dynamic";

const schema = z.object({ ticker: z.string().regex(/^[A-Za-z.\-]{1,10}$/) });
const list = async () => ({ tickers: (await getWatchlist()).map((w) => w.ticker) });

export async function GET() {
  return NextResponse.json(await list());
}

export async function POST(req: Request) {
  const parsed = schema.safeParse(await req.json().catch(() => null));
  if (!parsed.success) return NextResponse.json({ ok: false, error: "bad_request" }, { status: 400 });
  await addToWatchlist(parsed.data.ticker);
  return NextResponse.json(await list());
}

export async function DELETE(req: Request) {
  const parsed = schema.safeParse(await req.json().catch(() => null));
  if (!parsed.success) return NextResponse.json({ ok: false, error: "bad_request" }, { status: 400 });
  await removeFromWatchlist(parsed.data.ticker);
  return NextResponse.json(await list());
}
```

- [ ] **Step 2: Type-check** — Run: `pnpm exec tsc --noEmit` — Expected: no errors.

- [ ] **Step 3: Commit**

```bash
git add app/api/watchlist/route.ts
git commit -m "Add gated watchlist API route"
```

---

## Task 8: Watchlist UI + nav

**Files:**
- Create: `components/overlay-chart.tsx`
- Create: `components/watchlist-manager.tsx`
- Create: `app/watchlist/page.tsx`
- Modify: `components/app-nav.tsx`

- [ ] **Step 1: Implement `components/overlay-chart.tsx`** (N normalized series)

```tsx
"use client";
import { LineChart, Line, XAxis, YAxis, Tooltip, ResponsiveContainer, CartesianGrid, Legend } from "recharts";

const COLORS = ["#e5e5e5", "#38bdf8", "#f59e0b", "#a78bfa", "#22c55e", "#ef4444", "#ec4899", "#14b8a6", "#eab308", "#8b5cf6"];

type Props = { dates: string[]; series: { ticker: string; normalized: number[] }[] };

export function OverlayChart({ dates, series }: Props) {
  const data = dates.map((date, i) => {
    const row: Record<string, string | number> = { date };
    for (const s of series) row[s.ticker] = s.normalized[i];
    return row;
  });
  return (
    <div className="h-96 w-full">
      <ResponsiveContainer>
        <LineChart data={data} margin={{ top: 8, right: 8, bottom: 0, left: 0 }}>
          <CartesianGrid stroke="#262626" vertical={false} />
          <XAxis dataKey="date" tick={{ fontSize: 10, fill: "#737373" }} minTickGap={48} />
          <YAxis tick={{ fontSize: 10, fill: "#737373" }} width={48} />
          <Tooltip contentStyle={{ background: "#171717", border: "1px solid #404040", fontSize: 12 }} />
          <Legend wrapperStyle={{ fontSize: 12 }} />
          {series.map((s, i) => (
            <Line key={s.ticker} type="monotone" dataKey={s.ticker} stroke={COLORS[i % COLORS.length]} dot={false} strokeWidth={1.5} />
          ))}
        </LineChart>
      </ResponsiveContainer>
    </div>
  );
}
```

- [ ] **Step 2: Implement `components/watchlist-manager.tsx`** (client add/remove; refreshes the server page)

```tsx
"use client";
import { useState } from "react";
import { useRouter } from "next/navigation";

export function WatchlistManager({ initialTickers }: { initialTickers: string[] }) {
  const router = useRouter();
  const [input, setInput] = useState("");
  const [busy, setBusy] = useState(false);

  async function add(e: React.FormEvent) {
    e.preventDefault();
    const ticker = input.trim().toUpperCase();
    if (!ticker) return;
    setBusy(true);
    await fetch("/api/watchlist", {
      method: "POST", headers: { "content-type": "application/json" }, body: JSON.stringify({ ticker }),
    });
    setBusy(false);
    setInput("");
    router.refresh();
  }

  async function remove(ticker: string) {
    setBusy(true);
    await fetch("/api/watchlist", {
      method: "DELETE", headers: { "content-type": "application/json" }, body: JSON.stringify({ ticker }),
    });
    setBusy(false);
    router.refresh();
  }

  return (
    <div className="space-y-3">
      <form onSubmit={add} className="flex items-center gap-2">
        <input value={input} onChange={(e) => setInput(e.target.value)} placeholder="Add ticker (e.g. MSFT)"
          className="rounded bg-neutral-900 px-2 py-1 font-mono uppercase ring-1 ring-neutral-800" />
        <button disabled={busy} className="rounded bg-neutral-200 px-3 py-1 text-sm font-medium text-neutral-900 disabled:opacity-50">Add</button>
      </form>
      <div className="flex flex-wrap gap-2">
        {initialTickers.map((t) => (
          <span key={t} className="flex items-center gap-1 rounded bg-neutral-800 px-2 py-1 text-sm">
            <span className="font-mono">{t}</span>
            <button onClick={() => remove(t)} disabled={busy} aria-label={`Remove ${t}`} className="text-neutral-500 hover:text-red-400">×</button>
          </span>
        ))}
      </div>
    </div>
  );
}
```

- [ ] **Step 3: Implement `app/watchlist/page.tsx`**

```tsx
import { getWatchlistOverlay } from "@/lib/services/watchlist-service";
import { OverlayChart } from "@/components/overlay-chart";
import { WatchlistManager } from "@/components/watchlist-manager";

export const dynamic = "force-dynamic";

export default async function WatchlistPage() {
  const { tickers, overlay } = await getWatchlistOverlay();
  return (
    <main className="mx-auto max-w-4xl px-4 pb-24 pt-10">
      <h1 className="text-2xl font-semibold">Watchlist</h1>
      <p className="mt-1 text-sm text-neutral-500">
        Normalized to 100 at the start of the common window. Research only — not advice.
      </p>
      <div className="mt-4"><WatchlistManager initialTickers={tickers} /></div>
      {overlay.series.length > 0 ? (
        <div className="mt-6"><OverlayChart dates={overlay.dates} series={overlay.series} /></div>
      ) : (
        <p className="mt-6 text-neutral-400">Add tickers above to see them overlaid here.</p>
      )}
    </main>
  );
}
```

- [ ] **Step 4: Add "Watchlist" to `components/app-nav.tsx`** — add the link between Home and Health:

```tsx
      <Link href="/" className="text-neutral-300 hover:text-white">Home</Link>
      <Link href="/watchlist" className="text-neutral-300 hover:text-white">Watchlist</Link>
      <Link href="/health" className="text-neutral-300 hover:text-white">Health</Link>
```

- [ ] **Step 5: Type-check** — Run: `pnpm exec tsc --noEmit` — Expected: no errors.

- [ ] **Step 6: Commit**

```bash
git add components/overlay-chart.tsx components/watchlist-manager.tsx app/watchlist/page.tsx components/app-nav.tsx
git commit -m "Add watchlist page, overlay chart, and nav link"
```

---

## Task 9: E2E coverage + full verification + live smoke

**Files:**
- Modify: `tests/e2e/smoke.spec.ts` (add a watchlist test)

- [ ] **Step 1: Add a watchlist E2E test** — append to `tests/e2e/smoke.spec.ts` (uses a regex-valid throwaway ticker `ZZ` that won't collide with a real watchlist entry, and cleans up by removing it):

```ts
test("watchlist add then remove updates the chips", async ({ page }) => {
  await page.goto("/watchlist");
  await page.getByPlaceholder(/Add ticker/i).fill("ZZ");
  await page.getByRole("button", { name: "Add" }).click();
  await expect(page.locator("span.font-mono", { hasText: /^ZZ$/ })).toBeVisible();

  await page.getByRole("button", { name: "Remove ZZ" }).click();
  await expect(page.locator("span.font-mono", { hasText: /^ZZ$/ })).toHaveCount(0);
});
```

- [ ] **Step 2: Run the full E2E suite** — Run: `pnpm test:e2e` — Expected: setup + all chromium specs pass (the two original smokes, the updated split-tab test, the auth specs, and the new watchlist test). Ensure nothing else is serving port 3000.

- [ ] **Step 3: Full unit + integration suite** — Run: `pnpm test` — Expected: all suites pass (new `news-cap`, `watchlist`, `watchlist-overlay` included; no regressions).

- [ ] **Step 4: Clean type-check + production build** — Run: `rm -rf .next && pnpm build` — Expected: build succeeds; route list includes `/ticker/[symbol]`, `/ticker/[symbol]/news`, `/watchlist`, `/api/watchlist`; no deprecation warnings (proxy convention intact).

- [ ] **Step 5: Live smoke + screenshots** — Start the gated prod server (`APP_PASSWORD=localtest SESSION_SECRET=$(openssl rand -hex 32) pnpm start`), authenticate in a throwaway Playwright script, and capture:
  - `/ticker/NVDA` (Charts tab) — price + RSI + MACD charts render.
  - `/ticker/NVDA/news` (News tab) — memo + ≤10 news rows.
  - `/watchlist` after adding 2–3 real tickers (e.g. NVDA, AMD, SPY) — overlay shows multiple normalized lines + legend.
  Confirm the news list shows at most 10 rows. Stop the server afterward (exit 143 from SIGTERM is expected).

- [ ] **Step 6: Commit (if Step 1 added the test or any verification fix was needed)**

```bash
git add tests/e2e/smoke.spec.ts
git commit -m "Add watchlist E2E coverage"
```

---

## Self-Review (completed during planning)

**1. Spec coverage (§10):**
- §10.2 news cap → Task 1 (default-10 limit covers both callers). ✅
- §10.3 tabbed routes + RSI/MACD → Task 2 (expose MACD line/signal), Task 3 (charts), Task 4 (layout/tabs/charts page/news page + smoke update). ✅
- §10.4 watchlist (table/repo/overlay/service/API/UI/nav) → Tasks 5–8. ✅
- §10.5 error handling (per-ticker skip, empty state) → `buildOverlay` drops empty + `getWatchlistOverlay` `.catch`, page empty state (Tasks 6, 8). ✅
- §10.6 testing → Tasks 1/5/6 (unit+integration) + Task 9 (E2E + live smoke). ✅
- §10.7 acceptance → exercised by Task 9.

**2. Placeholder scan:** none — every code step is complete; commands have expected output.

**3. Type consistency:** `getTickerData().indicators` gains `macdLine`/`macdSignal`/`macdHistogram` (Task 2) and is consumed exactly so in the Charts page + `MacdChart` (Tasks 3–4). `buildOverlay`/`OverlayInput`/`Overlay` (Task 6) match `getWatchlistOverlay` + `OverlayChart` props (Tasks 6, 8). `getRecentArticles(…, limit=10)` (Task 1) leaves existing call sites valid. `getWatchlist`/`addToWatchlist`/`removeFromWatchlist` (Task 5) match the API route + service (Tasks 6–7). `TickerTabs` route shape (`/ticker/[symbol]` + `/news`) matches the layout + news page (Task 4).

---

## Execution Handoff

Execute task-by-task. Recommended: subagent-driven development (fresh subagent per task + review between tasks), with the controller running the full `pnpm test` after Task 4 (the page split touches the E2E smoke) and personally doing the Task 9 live smoke + screenshots. Tasks 1, 5, 6 are TDD with real failing-first tests; Tasks 2–4, 7, 8 are write-then-typecheck (behavior verified by the Task 9 E2E + live smoke).
