# Finance Dashboard Rebuild — Design Spec

**Date:** 2026-05-25
**Branch:** `rebuild-nextjs-mvp`
**Status:** Approved design; ready for implementation planning.

This spec is the de-risked, authoritative architecture for rebuilding `finance-dashboard`. It supersedes the architecture in `plan.md` where they differ (notably: single app instead of a two-service split). `plan.md` remains the broader product-vision reference (page-level features, prompt rules, compliance posture, risk register).

---

## 1. Goal & posture

A deployed, personal investing-**research** dashboard: search a ticker or company name, open a research page with prices, technical indicators, a one-target comparison, recent news, source-linked sentiment, a Gemini-generated daily memo, and basic fundamentals.

- **Research assistant, not investment advice.** Never emits buy/sell/hold.
- **Trust through evidence.** Every memo claim cites source article IDs; sentiment is labeled "News Tone" with article count/confidence; data freshness is shown honestly. (A primary user distrusts AI summaries, so source-linking and transparency are core requirements, not polish.)
- **Personal / non-commercial use.** This is a hard constraint, not a preference — see §3.

## 2. What changed from `plan.md` and why

`plan.md` assumed a Next.js frontend + a separate Python FastAPI worker, with local FinBERT for sentiment, FMP for news + fundamentals, and Alpha Vantage as a news/sentiment source. Pre-build verification (2026-05-25) invalidated those assumptions:

| `plan.md` assumption | Verified reality (2026) | Decision |
|---|---|---|
| FMP free covers search, profile, prices, **news, fundamentals** | FMP free ("Basic", 250 calls/day + 500MB/30-day bandwidth) covers **search + profile + EOD prices only**; **news & fundamentals are paid** (Starter $22/mo+). Use `/stable/` routes (legacy `/api/v3/` is deprecated). | FMP = search/profile/prices only. News and fundamentals sourced elsewhere. |
| Alpha Vantage for news/sentiment | Free tier is now **25 requests/day**, 5/min — unusable for recurring ingestion. | **Dropped entirely.** No AV key needed. |
| Local FinBERT for sentiment | Needs **~2GB RAM**; won't fit free/cheap (512MB–1GB) containers; no reliable free hosted endpoint. FinBERT was the **only** reason for a separate Python service. | **Dropped from MVP.** Sentiment comes from Gemini (§4). FinBERT becomes an optional later "Labs" add-on if a paid host is ever justified. |
| `gemini-3-flash-preview` | Exists (preview); stable `gemini-3.5-flash` also available. Preview models can be deprecated. | Default to stable `gemini-3.5-flash`; keep preview configurable via env. |

**Net effect:** no Python, no second service, no PyTorch. The product is **one Next.js app on Vercel.** This removes two-service deployment, CORS, shared-OpenAPI-types coordination, and ~$5–10/mo of always-on ML hosting.

## 3. Data sources & the personal-use constraint

| Need | Source | Tier | Notes |
|---|---|---|---|
| Search, company profile, EOD daily prices | **FMP** (`/stable/`) | Free, 250/day | Binding limit is 250 **calls/day** + 500MB/30-day bandwidth. Cache hard. |
| Price + search fallback | **Yahoo** (`yahoo-finance2` + RSS) | Free, keyless | Scrape-based; periodic 401/crumb/429 breakage. Best-effort fallback only. |
| Company news | **Finnhub** `/company-news` | Free, 60/min, no daily cap | **Personal/non-commercial use only.** |
| News fallback | **Yahoo Finance RSS** (+ Google News RSS) | Free, keyless | Verified working 2026-05-25. Personal-use only. |
| Fundamentals & filings | **SEC EDGAR** (CompanyFacts/submissions JSON) | Free, official | Primary (FMP fundamentals are paid). Requires a contact `User-Agent`. |
| Sentiment (News Tone) + daily memo | **Gemini** (`gemini-3.5-flash` default) | Free tier (limits live in AI Studio) | Structured JSON output. |

**Hard constraint:** Finnhub-free, Yahoo, and Google News are all licensed for **personal/non-commercial use only**. The app therefore stays **private (password-gated)** and must not publicly redistribute provider data. Making it public/commercial later would require paid data tiers (FMP Starter+, a commercial news API, etc.). This aligns with `plan.md`'s "no public redistribution" non-goal.

**Deferred paid option:** Nasdaq Data Link "Sharadar Core US Fundamentals" (SF1, ~$50/mo individual, unconfirmed) — a clean normalized-fundamentals upgrade path if SEC XBRL parsing (SP3) proves too messy. Not used in the MVP.

## 4. Architecture (cross-cutting, all phases)

- **One Next.js app** (App Router + TypeScript) at the **repo root**, deployed on Vercel.
- **All provider calls are server-side** (route handlers / server components) so API keys never reach the client.
- **Sentiment + memo via Gemini:** the same batched, structured-JSON Gemini calls produce per-article News Tone *and* the source-linked daily memo. Tradeoff accepted: LLM-derived tone is less deterministic than a dedicated finance model and carries small per-call cost, in exchange for zero ML infra.
- **Indicators in TypeScript:** MA/RSI/MACD/volatility/returns are pure functions (math ported from `legacy/app.py`), computed on the fly from cached price bars (no precomputed indicator table).
- **Storage = bounded cache,** not a warehouse: **Neon Postgres + Drizzle ORM.** Same Postgres dialect locally (Docker or a Neon branch) and in prod — no SQLite/Postgres dialect drift. Freshness/retention rules per `plan.md` §2.
- **No scheduler in the MVP:** fetch/compute on page load when the cache is stale, then persist. (A daily refresh job can be added later if "what changed since yesterday" trends are wanted.)
- **Auth:** simple password gate (also enforces the private/personal-use posture).
- **Frontend libs:** Tailwind + shadcn/ui, Recharts + lightweight-charts, TanStack Query/Table, Zod for boundary validation.

### Known risks / watch-items (carried into implementation)
- **Vercel function duration:** on-demand memo generation (news fetch + Gemini) can be slow (SP2). Mitigate: cache-first, possibly generate-then-poll, set `maxDuration`.
- **FMP 250/day budget:** enforce via `provider_state` call counting + aggressive caching.
- **Yahoo fragility:** treat as fallback only; never the sole source for a page.

## 5. Phasing — sub-projects

Each sub-project gets its own spec → plan → build cycle so we never build on an unproven foundation. SP1–SP4 are the MVP through first deploy; SP5 is a post-deploy UX-refinement round driven by live-use feedback.

1. **SP1 — Foundation + Search + Prices + Comparison** *(detailed below; build first)*. Usable slice: search "NVIDIA" → NVDA research page with price chart + AMD comparison.
2. **SP2 — News + Sentiment + Memo:** Finnhub + Yahoo RSS ingestion, dedupe, Gemini News Tone + source-linked daily memo, news table + tone meter + memo UI. *(detailed in §7 below)*
3. **SP3 — Fundamentals:** SEC EDGAR CIK mapping + CompanyFacts (minimal concepts), fundamentals card. Adds `cik` to `companies`. *(detailed in §8 below)*
4. **SP4 — Deploy & harden:** Vercel deploy (reusing the existing Neon DB), app-level password gate, provider-health page (reads `provider_state`), opportunistic + manual cache pruning, security hardening. *(detailed in §9 below)*
5. **SP5 — UX refinements (post-deploy):** cap news volume, split the ticker page into tabbed routes (Charts & Fundamentals / News & Memo), add RSI/MACD charts, add a multi-stock watchlist overlay alongside the existing 2-stock compare. *(detailed in §10 below)*
6. **SP6 — Chart ranges, watchlist tab, global search (post-deploy):** customizable chart date ranges (presets + custom calendar), Watchlist moved into the ticker tab bar, and a persistent global search header. *(detailed in §11 below)*

---

## 6. SP1 detailed design

### 6.1 Structure & stack

Single app at repo root:

```
finance-dashboard/
  app/                       # Next.js App Router
    page.tsx                 # search-first homepage
    ticker/[symbol]/page.tsx # company research page
    compare/page.tsx
    api/                     # server-side route handlers
      search/route.ts
      prices/[symbol]/route.ts
      compare/route.ts
  components/                # SearchBar, PriceChart, ReturnsTable, ComparisonChart, RecentSearches, ...
  lib/
    providers/               # base.ts, fmp.ts, yahoo.ts
    indicators/              # returns.ts, moving-averages.ts, rsi.ts, macd.ts, volatility.ts
    db/                      # drizzle schema + client
    services/                # search-service, price-service, comparison-service
    types.ts, formatters.ts
  drizzle/                   # migrations
  legacy/                    # archived Streamlit app (done)
  .env.example
  package.json, next.config.ts, tsconfig.json, tailwind.config.ts
```

**Stack:** Next.js App Router, TypeScript, Tailwind, shadcn/ui, Drizzle + Neon Postgres, Zod, TanStack Query/Table, Recharts + lightweight-charts.

**Env (finalized at scaffold):** `FMP_API_KEY`, `FINNHUB_API_KEY`, `GEMINI_API_KEY`, `GEMINI_MODEL=gemini-3.5-flash`, `GEMINI_PREVIEW_MODEL=gemini-3-flash-preview` (optional), `SEC_USER_AGENT`, `DATABASE_URL` (Neon), `APP_PASSWORD` + `SESSION_SECRET` (SP4), `FMP_DAILY_LIMIT=250`. A committed `.env.example` documents all of these.

### 6.2 Data model (Drizzle / Postgres) — four tables

- **`companies`** — resolved securities, including ETFs/benchmarks (SPY/QQQ): `id`, `ticker` (unique, uppercase), `name`, `asset_type` (`stock`/`etf`/`index`), `exchange`, `sector`, `industry`, `currency`, `last_profile_refresh_at`, `created_at`, `updated_at`. (`cik` added in SP3.)
- **`price_bars_daily`** — `id`, `company_id` FK, `date`, `open`, `high`, `low`, `close`, `adj_close` (nullable), `volume`, `source`. **Unique `(company_id, date, source)`.**
- **`recent_searches`** — `id`, `query`, `resolved_ticker` (nullable), `created_at`.
- **`provider_state`** — `provider`, `calls_today`, `daily_limit`, `reset_at`, `last_success_at`, `last_error_at`, `last_error`. (DB-backed because serverless has no persistent memory to count the FMP budget; also feeds the SP4 health page.)

(`technical_indicators_daily` from `plan.md` §9 is intentionally omitted — indicators are computed on the fly. Saved watchlist is deferred to SP4; SP1 ships only `recent_searches`.)

### 6.3 Provider layer (`lib/providers/`, server-side)

- **`base.ts`** — shared types (`PriceBar`, `CompanyProfile`, `SearchResult`) and a thin `Provider` contract.
- **`fmp.ts`** — `search(query)`, `profile(ticker)`, `dailyPrices(ticker, from, to)` against FMP `/stable/` routes. Every call updates `provider_state` (increment `calls_today`, record success/error). Respects `FMP_DAILY_LIMIT`.
- **`yahoo.ts`** — `yahoo-finance2` for price + search **fallback**, wrapped in try/catch (scrape-fragile); failures recorded to `provider_state` and degraded silently.

### 6.4 Services (`lib/services/`)

- **`search-service`** — normalize query → exact ticker hit in `companies` → else FMP `search()` → log `recent_searches` → return candidates (ticker, name, exchange, asset type, source).
- **`price-service`** — `getPrices(ticker, range)`: check freshness (latest cached bar vs last trading day); if stale **and** FMP budget available → FMP `dailyPrices()`, else Yahoo, else serve cache; upsert bars; return series. Computes returns + indicators via `lib/indicators`.
- **`comparison-service`** — `getComparison(primary, comparison, range)`: fetch both series, align on common dates, compute relative return, volatility, drawdown, and a performance table.

### 6.5 Indicators (`lib/indicators/`)

Pure functions over a bar series, unit-tested test-first against golden values (cross-checked vs `legacy/app.py`, using `legacy/data/SPY.csv` as a fixture):
- **Returns:** 1D, 5D, 1M, 3M, 6M, YTD, 1Y (nearest bar at/just-before the lookback date → % change to latest).
- **Moving averages:** MA10, MA20, MA50.
- **RSI14**, **MACD(12/26/9)**, **5-day volatility** (rolling std of daily returns).

### 6.6 Data flows

1. **Search:** homepage `SearchBar` → `GET /api/search?q=` → `search-service` → candidates → user selects → navigate to `/ticker/[symbol]`.
2. **Ticker page** (server component): load profile (cache → FMP) + bars (cache → FMP → Yahoo) → compute returns + indicators → render header (price, name, period returns) + `PriceChart` (MA overlays, range selector 1M/3M/6M/YTD/1Y) + comparison selector.
3. **Compare:** `/compare?primary=&comparison=` → `comparison-service` → `ComparisonChart` overlay + relative return + volatility + drawdown + performance table.

### 6.7 Error handling — "degrade, never crash"

- **Fallback chain:** FMP → Yahoo → stale cache. A provider failure never breaks a page.
- **Staleness honesty:** when serving stale cache (provider down or budget spent), show a "data as of \<date\>" badge — never fail silently or misrepresent freshness.
- **FMP budget exhaustion:** when `calls_today >= FMP_DAILY_LIMIT`, skip FMP → Yahoo/cache; record state.
- **Unresolvable ticker:** empty search → friendly "no match"; unknown `/ticker/[symbol]` → clear "couldn't resolve" page.
- **Yahoo scrape failures** (401/crumb/429): caught, logged to `provider_state`, degraded to cache.
- **Boundary validation:** Zod on all API route params (query, ticker format); reject malformed input early.
- **Partial comparison:** if the comparison ticker fails but the primary succeeds, render the primary and note the comparison is unavailable.

### 6.8 Testing

- **Unit (Vitest):** indicators (priority; test-first, golden values, `legacy/data/SPY.csv` fixture); provider normalization + fallback decision logic (mocked `fetch`/`yahoo-finance2`); Zod schemas.
- **Integration:** services against a test Postgres (local Docker or Neon branch) — upsert/dedup on `(company_id, date, source)`, freshness logic, budget gating.
- **E2E smoke (Playwright, light):** "NVIDIA" → NVDA → chart renders; compare renders. Golden path only.

### 6.9 SP1 acceptance criteria

- Search "NVIDIA" resolves to NVDA; "NVDA" opens a ticker page.
- Ticker page shows a price chart, period returns, and indicators from cached/provider data.
- `NVDA` vs `AMD` and `AAPL` vs `SPY` comparisons render (overlay + relative return + volatility + drawdown + table).
- Provider failure or exhausted FMP budget degrades to Yahoo/cache with a visible staleness badge — no crash.
- Indicator unit tests pass against golden values.

## 7. SP2 detailed design — News + sentiment + memo

Build second. Usable slice: a ticker page that shows recent deduped news plus a source-linked daily memo and a News Tone meter, generated by Gemini and cached. Verified against 2026 API realities on 2026-05-25 (Finnhub `/company-news` still free at 60/min; Gemini `@google/genai` SDK with `gemini-3.5-flash` stable + structured JSON output; Yahoo Finance RSS `finance.yahoo.com/rss/headline?s=` working, non-commercial only).

### 7.1 Scope & what's deferred

**In scope (all on the existing `/ticker/[symbol]` page):**
- Recent-news ingestion from **Finnhub `/company-news`** (primary) + **Yahoo Finance RSS** (supplement/fallback), deduped and cached.
- A **news table** (time / source / headline / link) — no per-article tone.
- A **source-linked daily memo** + **News Tone meter**, both produced by one batched Gemini call; auto-loaded cache-first with a Regenerate button.

**Deferred (out of SP2):** standalone `/news` page; per-article sentiment / FinBERT; daily tone-trend history & "what changed since yesterday" (needs a scheduler); cross-ticker news filters; retention-pruning job + provider-health page (SP4); embedding/MinHash dedupe.

**Key product decisions (locked during brainstorming):**
- The aggregate News Tone score (0–100) is **emitted directly by Gemini** (label + score + rationale), not computed by a separate deterministic formula.
- The **news table carries no per-article tone**; tone evidence is conveyed by the memo's grouped sections, which cite article IDs.
- The memo is **auto, cache-first**: the page paints instantly with prices + news; the memo card client-fetches after paint and regenerates only when stale; a manual Regenerate button forces a new one. Gemini runs only when stale → low cost.

### 7.2 Data model — two new tables

Add to `lib/db/schema.ts` (imports add `jsonb` from `drizzle-orm/pg-core`):

- **`articles`** — `id` uuid PK · `companyId` uuid FK→`companies.id` · `source` text (`finnhub`|`yahoo_rss`) · `sourceArticleId` text? · `url` text · `urlHash` text (sha256 of canonicalized URL) · `title` text · `summary` text? · `publishedAt` timestamptz · `imageUrl` text? · `related` text? (comma-joined tickers from Finnhub; null for RSS) · `createdAt` timestamptz default now · `expiresAt` timestamptz? (set ~90d out; pruning is SP4). **Unique `(companyId, urlHash)`** → hard cross-source URL dedupe.
- **`daily_memos`** — `id` uuid PK · `companyId` uuid FK · `memoDate` date · `model` text · `summaryJson` jsonb (full structured output, with `source_article_ids` already resolved to article UUIDs) · `toneLabel` text · `toneScore` int (0–100) · `sourceArticleIds` jsonb (union of cited article UUIDs) · `basedOnArticleCount` int · `generatedAt` timestamptz default now. **Unique `(companyId, memoDate)`** → one memo per ticker per day; Regenerate upserts.

**Staleness trigger for regeneration:** no memo exists for `(company, today)` **or** an `articles` row for that company has `createdAt > memo.generatedAt`.

**`provider_state`:** add `finnhub` and `gemini` rows (reuse the existing table for health: `lastSuccessAt`/`lastErrorAt`/`lastError` + `callsToday`). `dailyLimit` stays `notNull`; set Finnhub to a high sentinel (not hard-gated — 60/min is ample for one user, and the ~3h news cache prevents hammering) and Gemini to `GEMINI_DAILY_LIMIT`. FMP stays hard-gated exactly as in SP1.

**`lib/env.ts` additions:** `GEMINI_MODEL` (`z.string().min(1).default("gemini-3.5-flash")`), `GEMINI_PREVIEW_MODEL` (`z.string().min(1).optional()`), `GEMINI_DAILY_LIMIT` (`z.coerce.number().int().positive().default(200)`). `FINNHUB_API_KEY` and `GEMINI_API_KEY` **stay optional** — the features degrade gracefully when a key is absent. News lookback (7 days) and the news cache TTL (~3h) live as code constants, not env.

A migration is generated via `drizzle-kit generate` (same flow as SP1) and applied to Neon.

### 7.3 Provider layer (`lib/providers/`, server-side)

Normalized type added to `lib/types.ts`:

```ts
export interface NewsArticle {
  source: "finnhub" | "yahoo_rss";
  sourceArticleId: string | null;
  url: string;
  title: string;
  summary: string | null;
  publishedAt: Date;
  imageUrl: string | null;
  related: string | null; // comma-joined tickers (Finnhub); null for RSS
}
```

- **`finnhub.ts`** — `companyNews(ticker, fromIso, toIso): Promise<NewsArticle[]>`. `GET https://finnhub.io/api/v1/company-news?symbol=&from=YYYY-MM-DD&to=YYYY-MM-DD`, auth via `X-Finnhub-Token` header. Maps raw `{id, datetime (unix s), headline, source, summary, url, image, related, category}` → `NewsArticle` (`sourceArticleId = String(id)`, `publishedAt = new Date(datetime*1000)`, `title = headline`, `imageUrl = image`). Records success/error to `provider_state`. North-America-only is acceptable (US-listed scope).
- **`yahoo-rss.ts`** — `companyNews(ticker): Promise<NewsArticle[]>`. Fetches `https://finance.yahoo.com/rss/headline?s=TICKER`, parses with **`fast-xml-parser`** (new dep). Maps each `<item>` `{title, link, pubDate, description}` → `NewsArticle` (`source = "yahoo_rss"`, `sourceArticleId = null`, `url = link`, `publishedAt = new Date(pubDate)`, `summary` = HTML-stripped description, `imageUrl = null`). Wrapped in try/catch (scrape-fragile); failures recorded and degraded silently. Returns ~25 most-recent items.
- **`gemini.ts`** — `generateMemo(input: MemoInput): Promise<MemoOutput>` via **`@google/genai`** (new dep). Structured JSON output, low temperature (~0.2), model from env. Records success/error + increments the `gemini` call count. Contract in §7.5.

### 7.4 Services (`lib/services/`)

- **`news-service.ts`** — `getNews(ticker, { force? }): Promise<{ articles: StoredArticle[]; asOf: Date }>`:
  1. Resolve `companyId` (reuse the companies repo).
  2. **Freshness:** if the newest `article.createdAt` for the company is younger than the ~3h TTL and not `force` → serve cache (last 7 days, newest first).
  3. Else fetch **Finnhub** (`from = today−7d`, `to = today`, when the key is present) **+ Yahoo RSS**, merge.
  4. **Dedupe:** canonicalize URL (lowercase host, strip `utm_*`/tracking query params, trim trailing slash) → `urlHash`; the unique constraint collapses exact cross-source dups; additionally soft-skip near-identical normalized titles (lowercase, strip punctuation + source suffixes) within the window.
  5. Upsert (`onConflict (companyId, urlHash) do nothing`), set `expiresAt = now + 90d`; return the 7-day window.
  - **Degrade:** a provider error is caught + recorded; if both providers fail, serve cache; if there's no cache, return an empty list (UI shows "news unavailable").
- **`memo-service.ts`** — `getMemo(ticker, { force? }): Promise<{ memo, status: "ok"|"no_news"|"unavailable"|"error", citedArticles }>`:
  1. **Cache-first:** load `daily_memos` for `(company, today)`; if present, not `force`, and no newer article → return it (with its cited articles).
  2. Else ensure news is fresh (`news-service.getNews`), gather the last-7-day articles. **Zero articles → return `no_news`** (no Gemini call).
  3. If `GEMINI_API_KEY` missing **or** the `GEMINI_DAILY_LIMIT` guardrail is exceeded → return `unavailable`.
  4. Build `MemoInput`: assign each article a short id (`a1…aN`) for citation; add price context (latest close + 1D/5D/1M/1Y returns via `price-service`).
  5. Call `gemini.generateMemo`, validate (§7.5). Map cited short-ids → article UUIDs, applying the citation guards. Upsert `daily_memos` (`onConflict (companyId, memoDate) do update`). Return `ok` with the memo + a `{id, title, url, source}` lookup of cited articles.

### 7.5 Gemini memo contract (trust-critical core)

**`MemoInput`:** `{ ticker, companyName, date, priceContext: { latestClose, currency, returns: {d1, d5, m1, y1} }, articles: [{ id: "a1", source, publishedAt, headline, summary, related }] }`.

**`MemoOutput`** (Zod schema; also the Gemini `responseSchema`):

```ts
const development = z.object({
  claim: z.string(),
  why_it_matters: z.string(),
  source_article_ids: z.array(z.string()),
  confidence: z.enum(["low", "medium", "high"]),
});
const memoOutput = z.object({
  ticker: z.string(),
  date: z.string(),
  one_sentence_takeaway: z.string(),
  bullish_developments: z.array(development),
  bearish_developments: z.array(development),
  neutral_or_operational_updates: z.array(development),
  watch_items: z.array(z.string()),
  caveats: z.array(z.string()),
  overall_news_tone: z.object({
    label: z.enum(["bearish", "somewhat_bearish", "neutral", "somewhat_bullish", "bullish"]),
    score: z.number().int().min(0).max(100),
    rationale: z.string(),
  }),
});
```

(Dropped from `plan.md` §H for SP2: `comparison_context` — single-ticker page; and a speculative `market_reaction` — price context stays factual in the takeaway, no causal guessing.)

**Prompt rules (ported from `plan.md` §H):** never output buy/sell/hold or price targets; every development must cite ≥1 provided article id; use only the supplied headlines/summaries — never invent facts or imply access to full article bodies; separate confirmed company events from analyst speculation; if evidence is thin/duplicated/stale, say so in `caveats`, lower `confidence`, and return mostly-empty arrays; the score is **News Tone** (coverage tone), not an investment outlook; the rationale must reference the actual articles.

**Model config:** `model = env.GEMINI_MODEL` (default `gemini-3.5-flash`; `GEMINI_PREVIEW_MODEL` configurable); `responseMimeType: "application/json"` + `responseSchema`; temperature ~0.2. The exact `@google/genai` config field names (and whether `zod-to-json-schema` is needed) are pinned against the installed SDK version at build time.

**Code-enforced trust guards (not prompt-trusted):**
- Validate with Zod; on failure → one retry, then degrade to `unavailable` (never store malformed memos).
- Map cited short-ids → article UUIDs; **drop any id not in the input set** (hallucination guard); if a development cites zero valid articles after mapping, **drop that development**. The "every claim is source-linked" guarantee is thus enforced in code, not just the prompt.
- Persist `summaryJson` with `source_article_ids` **resolved to article UUIDs** so the UI links each claim straight to its source rows.

### 7.6 API route & data flow (Approach 1 — two endpoints, client-orchestrated, simplified to one new route)

- **Ticker page (server component)** calls `news-service.getNews` + the existing `price-service` and **SSRs** the header, price chart, and **news table** — instant paint (news is cached/fast).
- **Memo card (client component)** fetches **`GET /api/memo/[symbol]`** after paint (spinner). The route is cache-first and generates only when stale. **Regenerate** = same route with `?force=1`. Response = `{ status, memo, citedArticles }`.
- The route Zod-validates the symbol (reuse SP1's ticker regex) and sets **`export const maxDuration = 60`** (news + Gemini can be slow); the page never waits on Gemini.
- No `/api/news` route — news is SSR'd and auto-refreshes on load when older than the TTL.

### 7.7 UI (`components/`, ticker-page additions)

Page order: existing header + price chart → **memo card** → **news table**. A small "Research assistant, not investment advice" note sits near the memo.

- **`memo-card.tsx` (client, centerpiece)** — fetches `/api/memo/[symbol]` on mount. States: `loading` (spinner, "Generating today's memo…"), `ok`, `no_news` ("No recent news in the last 7 days"), `unavailable` ("Memo unavailable — Gemini key missing or daily limit reached"), `error` (friendly retry). `ok` layout: prominent **one-sentence takeaway** → **tone meter** → **Bullish / Bearish / Neutral** sections (each claim shows `why_it_matters`, a confidence tag, and numbered **source chips** that link out to the cited articles) → **Watch items** → **Caveats**. A **Regenerate** button re-fetches with `?force=1`. **Transparency footer:** "Generated by {model} at {time} from {N} sources · every claim links to its source · News Tone reflects coverage tone, not a forecast · not investment advice."
- **`tone-meter.tsx` (presentational)** — 0–100 gauge with the label, colored across the bearish→bullish range; clearly labeled "News Tone." Lives inside the memo card (the score isn't known until the memo loads).
- **`news-table.tsx` (server-rendered)** — columns: **Time** (relative; absolute on hover) · **Source** (finnhub/yahoo + original publisher) · **Headline** (links out, `target="_blank" rel="noopener"`). Newest first, last 7 days. Empty state: "No recent news."

### 7.8 Error handling — "degrade, never crash"

- Missing `FINNHUB_API_KEY` → Yahoo RSS only. Missing `GEMINI_API_KEY` → memo `unavailable`. Neither crashes the page.
- Finnhub error → caught, recorded to `provider_state`, fall back to Yahoo RSS + cache. Yahoo RSS error (404/parse) → caught, recorded, degrade silently. Both fail → serve cached articles (with honest staleness); no cache → "news unavailable" empty state.
- Gemini failure (network/timeout/invalid JSON after one retry) → `error`; card offers retry; nothing malformed stored. `GEMINI_DAILY_LIMIT` exceeded → `unavailable`. Zero articles → `no_news` (no Gemini call).
- Zod-validate the `/api/memo` symbol param; `maxDuration = 60`; overflow → `error` → card retry. Citation guards block hallucinated/unsourced claims. `provider_state` updated for `finnhub`/`gemini` on every call (feeds the SP4 health page).

### 7.9 Testing

- **Unit (Vitest):** Finnhub normalization (unix→Date, field mapping; mocked fetch); Yahoo RSS parsing (sample XML → articles, HTML-strip, empty/bad feed → `[]`); dedupe (URL canonicalization + hash, cross-source collapse, soft title-dedupe); `MemoOutput` Zod schema (valid/invalid) + citation-guard mapping (hallucinated id dropped, zero-cite development dropped); tone-meter score→segment mapping.
- **Integration (live Neon test DB, as SP1):** `news-service` upsert/dedup on `(companyId, urlHash)`, freshness gating, 90d `expiresAt`; `memo-service` cache-first, staleness trigger, `force=1`, `no_news` and `unavailable` paths — **Gemini mocked**.
- **E2E smoke (Playwright, golden path):** ticker page renders the news table and the memo card transitions loading → rendered memo. The memo card's client fetch to **`/api/memo` is intercepted with a fixture** (deterministic, free — never hits live Gemini). The news table is server-rendered, so route interception doesn't apply; it's made deterministic by **seeding the test DB** with a couple of `articles` rows. Finnhub/Yahoo aren't called in the smoke run.
- Lint still OOM-crashes locally on this machine; verified in CI per the standing note.

### 7.10 SP2 acceptance criteria

- A searched ticker (e.g., NVDA) shows a news table populated from Finnhub and/or Yahoo RSS within the last 7 days, with duplicates collapsed.
- The memo card loads cache-first: first visit of the day generates a memo; later visits serve cache; Regenerate forces a fresh one.
- Every memo development links to ≥1 source article; no claim renders without a valid cited article (code-enforced).
- The News Tone meter shows Gemini's 0–100 score + label, is labeled "News Tone," carries the not-advice disclaimer, and no buy/sell/hold text appears anywhere.
- A missing Gemini/Finnhub key or a provider failure degrades to a clear `unavailable`/empty state without crashing; staleness is shown honestly.
- Unit + integration + E2E suites pass with Gemini mocked.

## 8. SP3 detailed design — Fundamentals (SEC EDGAR)

Build third. Usable slice: a stock's ticker page shows a fundamentals card (valuation, profitability, financial health, latest financials) sourced from official SEC filings, cached, with every figure traceable to a 10-K. Verified against live SEC endpoints on 2026-05-25: `company_tickers.json` maps ticker→CIK; `data.sec.gov` requires a contact `User-Agent` (403 without one); `companyfacts` returns current XBRL facts (AAPL: 3.75 MB, 503 us-gaap concepts, all needed concepts present and current to the latest 10-Q).

### 8.1 Scope & what's deferred

**In scope (on the existing `/ticker/[symbol]` page, stocks only):**
- SEC CIK resolution (ticker→CIK via `company_tickers.json`), cached on the company row.
- A single `companyfacts` fetch per company (cold cache only), normalized to ~12 concepts and cached as a small snapshot.
- A **fundamentals card** (client-fetched, mirrors the SP2 memo) showing latest-annual (10-K) figures + the latest balance sheet, with valuation multiples computed from the cached close.

**Deferred (out of SP3):** trailing-twelve-month (TTM) assembly; multi-period history/trends; EV/EBITDA and dividend yield (missing/messy XBRL tags); peer fundamentals comparison; segment data; the deferred paid Sharadar upgrade (§3); retention-pruning of snapshots (SP4).

**Key product decisions (locked during brainstorming):**
- **Data acquisition = one `companyfacts` call**, not per-concept (`companyconcept`) or `frames`. Simplest code, one round-trip, future-proof; the few-MB payload is a non-issue given a multi-day cache (fetch happens at most once per company per TTL window).
- **Period basis = latest annual (10-K).** Flow metrics (revenue, net income, EPS, operating income, gross profit) use the most-recent annual (`fp:"FY"`) fact; balance-sheet metrics + shares use the latest available instant (may be a 10-Q — more current, labeled honestly). TTM deferred (avoids fragile 4-quarter assembly).
- **Metrics = SEC-native + key valuation.** Price-derived Market Cap, P/E, P/S are computed at request time from the already-cached close (no extra provider call) so they never go stale; price-independent ratios derive purely from stored concepts.
- **Render = client-fetched** `/api/fundamentals/[symbol]`, mirroring the memo card so a cold SEC fetch never blocks page paint.

### 8.2 Data model

Add to `lib/db/schema.ts`:

- **`companies`** gains `cik text` (nullable; 10-digit zero-padded string, e.g. `"0000320193"`). Resolved once and stored; reused thereafter.
- **`company_fundamentals`** — one current snapshot per company (refresh upserts): `id` uuid PK · `companyId` uuid FK→`companies.id` **unique** · `conceptsJson` jsonb (normalized reported concepts + per-concept period/filing metadata — *not* the raw 3.75 MB blob) · `fiscalYear` int · `incomePeriodEnd` date (FY period end used for flow metrics) · `balanceSheetAsOf` date (latest balance-sheet instant) · `filingForm` text (`10-K`) · `filedAt` date · `source` text (`sec_edgar`) · `fetchedAt` timestamptz default now · `expiresAt` timestamptz (~7d out). **Unique `(companyId)`.**
- **`provider_state`** gains a `sec` row: high sentinel `dailyLimit` (not hard-gated — SEC has no daily cap, only a ≤10 req/s guideline that hard caching respects); `lastSuccessAt`/`lastErrorAt`/`lastError` feed the SP4 health page.

A migration is generated via `drizzle-kit generate` and applied to Neon (same flow as SP1/SP2).

**`lib/env.ts` addition:** `SEC_USER_AGENT` (`z.string().min(1).optional()`). SEC returns 403 without a contact `User-Agent`; if absent, the feature degrades to `unavailable` (consistent with optional `FINNHUB_API_KEY`/`GEMINI_API_KEY`). `.env.example` documents the format `"AppName contact@email"`.

### 8.3 Provider layer (`lib/providers/sec.ts`, server-side)

- `resolveCik(ticker): Promise<string | null>` — GET `https://www.sec.gov/files/company_tickers.json`, find the entry whose `ticker` matches (case-insensitive), zero-pad `cik_str` to 10 digits. Returns `null` if not found. Sends `User-Agent: env.SEC_USER_AGENT`; records `provider_state(sec)`.
- `fetchCompanyFacts(cik): Promise<RawCompanyFacts | null>` — GET `https://data.sec.gov/api/xbrl/companyfacts/CIK{cik}.json` with the User-Agent. Returns parsed JSON or `null` on 403/404/parse error. Records success/error.
- Both return `null` (never throw to the page) when `SEC_USER_AGENT` is missing.

### 8.4 Pure extraction & derivation (`lib/fundamentals/`)

- **`extract.ts`** — `extractConcepts(raw): FundamentalConcepts`. For each logical metric, walk a **fallback list** of XBRL tags and read the right **unit bucket**:
  - Revenue: `Revenues` → `RevenueFromContractWithCustomerExcludingAssessedTax` → `SalesRevenueNet` (USD)
  - Net Income: `NetIncomeLoss` (USD)
  - EPS: `EarningsPerShareDiluted` → `EarningsPerShareBasic` (USD/shares)
  - Operating Income: `OperatingIncomeLoss` (USD)
  - Gross Profit: `GrossProfit` (USD)
  - Assets / Liabilities / Equity: `Assets`, `Liabilities`, `StockholdersEquity` → `StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest` (USD)
  - Current Assets / Liabilities: `AssetsCurrent`, `LiabilitiesCurrent` (USD)
  - Shares outstanding: dei `EntityCommonStockSharesOutstanding` (shares)
  - **Period selection:** flow metrics → most-recent fact with `fp === "FY"` (annual); instant/balance-sheet metrics + shares → latest fact by `end`. Each chosen fact's `end`/`form`/`filed`/`fy` is recorded for honest labeling. Missing tag → `null`.
- **`derive.ts`** — `deriveMetrics(concepts, latestClose): FundamentalsView`. Computes: Gross Margin = grossProfit/revenue; ROE = netIncome/equity; ROA = netIncome/assets; Debt/Equity = totalLiabilities/equity; Current Ratio = currentAssets/currentLiabilities; Market Cap = latestClose×shares; P/E = latestClose/dilutedEps; P/S = marketCap/revenue. Every divisor guarded (null/0 → `null`). Pure; no I/O.

Both are pure and unit-tested with a trimmed AAPL `companyfacts` fixture.

### 8.5 Service & API

- **`lib/db/fundamentals.ts`** — `getFundamentalsSnapshot(companyId)`, `upsertFundamentalsSnapshot(...)` (`onConflict (companyId) do update`).
- **`lib/services/fundamentals-service.ts`** — `getFundamentals(ticker): Promise<{ status: "ok"|"not_applicable"|"unavailable"|"error", view, asOf, source }>`:
  1. Resolve company (reuse the companies repo / price-service company path). `assetType !== "stock"` → `not_applicable`.
  2. **Cache-first:** snapshot with `expiresAt > now` → use it.
  3. Else ensure `cik` (resolve + persist on the company if missing). No CIK → `unavailable`.
  4. `fetchCompanyFacts` → `extractConcepts` → upsert snapshot (`expiresAt = now + 7d`). SEC/extract failure → `error`; serve a stale snapshot if one exists; never store malformed.
  5. Read the latest cached close from `price_bars_daily` (no extra provider call) → `deriveMetrics` → return `ok` with the view, `asOf` (period/filing dates), and an EDGAR filings link.
- **`app/api/fundamentals/[symbol]/route.ts`** — `dynamic="force-dynamic"`, `maxDuration=30`; Zod-validate the symbol (reuse the SP1 ticker regex); returns `{ status, view, asOf, source }` at HTTP 200; catch → `{ status:"error", ... }`. The page never waits on SEC.

### 8.6 UI (`components/fundamentals-card.tsx`, ticker-page addition)

Client component; fetches `/api/fundamentals/[symbol]` on mount. States: `loading` (spinner), `ok`, `not_applicable` ("Fundamentals aren't available for ETFs/indexes"), `unavailable` ("Fundamentals unavailable — SEC data couldn't be fetched"), `error` (retry). `ok` groups (each value shows "N/A" when null):
- **Valuation:** Market Cap, P/E (FY EPS), P/S
- **Profitability:** Gross Margin, ROE, ROA, Operating Income
- **Financial Health:** Current Ratio, Debt/Equity, Assets, Liabilities, Equity
- **Latest Financials:** Revenue, Net Income, EPS (diluted)

**Trust footer:** "Source: SEC EDGAR · FY{year} {form} filed {date}; balance sheet as of {date}" + a link to the company's EDGAR filings page. Placed after the memo, before recent news. `lib/types.ts` adds `FundamentalConcepts` and `FundamentalsView`.

### 8.7 Error handling — "degrade, never crash"

- Missing `SEC_USER_AGENT` / SEC 403 / timeout / parse failure → `unavailable`; ETF/index → `not_applicable`; individual missing concepts → "N/A" rows only. Page and card never crash.
- A stale snapshot is served if a fresh fetch fails but an old snapshot exists (honest "as of" labeling). `provider_state(sec)` is updated on every call.

### 8.8 Testing

- **Unit (Vitest):** `extract` (trimmed AAPL fixture → concepts; FY-vs-latest period selection; tag fallback; missing tag → null); `derive` (ratios; divide-by-zero/null → null; margin %, market cap); `resolveCik` (found/not-found vs a small map fixture); route Zod param.
- **Integration (live Neon, SEC mocked):** cache-first, upsert, 7-day TTL staleness, `not_applicable` for an ETF, `unavailable` with no CIK.
- **E2E smoke (Playwright):** the fundamentals card transitions loading→rendered with `/api/fundamentals` intercepted by a fixture.
- **Live verification (standing rule):** real SEC fetch for NVDA + AAPL — confirm figures render sensibly + a screenshot; mocks can't catch real concept-tagging quirks. Lint stays verified in CI (OOM locally).

### 8.9 SP3 acceptance criteria

- A stock ticker (e.g. NVDA, AAPL) shows a fundamentals card with valuation, profitability, financial-health, and latest-financials values sourced from SEC EDGAR, cached after first load.
- Every figure traces to a filing: the card shows the fiscal year, form, filing date, and balance-sheet date, and links to EDGAR.
- Market Cap / P/E / P/S are computed from the latest cached close (no extra provider call) and update as prices refresh.
- An ETF/index (SPY/QQQ) shows `not_applicable`; a missing key/CIK or SEC failure shows `unavailable`; missing individual concepts show "N/A" — no crash.
- Unit + integration + E2E suites pass with SEC mocked.

## 9. SP4 detailed design — Deploy & harden

Build fourth (final). Usable outcome: the app runs at a private, password-gated `*.vercel.app` URL Kavin can share with his dad and a few others; internal health is visible at `/health`; the cache stays bounded over time. Verified realities (2026-05-26): Vercel Hobby is free and builds Next.js 16; Vercel's built-in **Password Protection is a paid (Pro) feature**, and its free **"Vercel Authentication" forces every viewer to hold a Vercel account with project access** — neither fits sharing with family, so the gate is **app-level**. Next.js edge middleware exposes Web Crypto (`crypto.subtle`), so cookie signing/verification needs no Node-only APIs.

### 9.1 Scope & what's deferred

**In scope:**
- App-level **shared-password gate**: `middleware.ts` + a styled `/login` page + an HMAC-signed session cookie.
- **Provider-health page** (`/health`, gated) reading `provider_state` (fmp/finnhub/gemini/sec).
- **Cache pruning**: opportunistic (during normal fetches) + a manual "Prune now" action.
- **Security hardening**: response headers + a disallow-all `robots.txt`.
- **Deploy** to Vercel, **reusing the existing Neon DB**, from branch `rebuild-nextjs-mvp`.
- A **deploy runbook** (§9.9) enumerating the user-only account/secret steps.

**Deferred (out of SP4):** per-user accounts/roles; login rate-limiting beyond constant-time compare + a small fixed delay; Vercel Cron / any scheduled data refresh; a strict Content-Security-Policy; a separate production database; a custom domain; merging `rebuild-nextjs-mvp` into `main`.

**Key product decisions (locked during brainstorming):**
- **Styled login page** over HTTP basic-auth (better UX for non-technical viewers; supports logout/styling) and over Vercel's paid/account-bound protection.
- **Opportunistic + manual pruning** over a cron job — no new infra or secret, honoring §4's "no scheduler in the MVP."
- **Reuse the existing Neon DB** for prod (simplest; accepted tradeoff that the test suite writes to the same cache DB and shares the FMP daily counter — don't run the full integration suite while the live FMP budget is relied upon).
- **Deploy from `rebuild-nextjs-mvp`** (Vercel "Production Branch") so shipping needs no disruptive merge; `main` keeps the Streamlit app as reference.

### 9.2 Env & config

`lib/env.ts` adds two optional vars:
- `APP_PASSWORD` (`z.string().min(1).optional()`) — the shared gate password.
- `SESSION_SECRET` (`z.string().min(1).optional()`) — HMAC key for signing the session cookie.

Both are optional in the schema; the **runtime gate** (§9.3) enforces them. `.env.example` documents both (`SESSION_SECRET` = a long random string, e.g. `openssl rand -hex 32`; `APP_PASSWORD` = the chosen shared password). No `vercel.json` is needed (no cron; per-route `maxDuration` already covers the slow routes).

### 9.3 Auth gate

**`lib/auth/session.ts` (pure, runtime-agnostic via Web Crypto):**
- `createSessionToken(secret, now?) → Promise<string>` — payload = expiry epoch ms (`now + 30d`); token = `${expiry}.${base64url(HMAC_SHA256(String(expiry), secret))}`.
- `verifySessionToken(token, secret, now?) → Promise<boolean>` — re-computes the HMAC (constant-time compare), rejects malformed/tampered tokens and `expiry <= now`.
- `safeNextPath(raw) → string` — returns `raw` only if it is a same-origin absolute path (starts with `/`, not `//` or `/\`); else `"/"`. Guards the post-login redirect against open-redirects.
- `constantTimeEqual(a, b) → boolean` — length-hardened byte compare for the password check.

**`lib/auth/gate.ts` — pure decision (unit-tested):**
`shouldAllow({ pathname, hasValidSession, appPassword, sessionSecret, onVercel }) → { allow: true } | { allow: false, reason: "login" | "misconfig" }`:
- Always allow `/login`, `/api/login`, `/api/logout`, `/_next/*`, `/favicon.ico`, `/robots.txt`, and other static assets.
- If `appPassword` **or** `sessionSecret` is empty: `onVercel` → `{ allow:false, reason:"misconfig" }` (fail closed); else (local dev) → `{ allow:true }` (gate off).
- Else (both present): `hasValidSession` → allow; otherwise `{ allow:false, reason:"login" }`.

**`middleware.ts` (thin edge wrapper):** reads the `fd_session` cookie, calls `verifySessionToken`, builds the `shouldAllow` input from `env.APP_PASSWORD`, `env.SESSION_SECRET`, and `process.env.VERCEL`, then: allow → `NextResponse.next()`; `login` → redirect to `/login?next=<pathname>`; `misconfig` → a 503 plain response ("App is not configured: APP_PASSWORD/SESSION_SECRET missing"). `matcher` excludes static assets for efficiency; the function re-checks anyway.

**Routes & page:**
- **`app/login/page.tsx`** (client) — a styled, centered password form; on submit POSTs `{ password, next }` to `/api/login`; shows an inline error on 401; on success navigates to the sanitized `next`.
- **`app/api/login/route.ts`** (Node) — Zod-validates the body; if `!APP_PASSWORD || !SESSION_SECRET` → 503; constant-time-compares the password; on match sets `fd_session` (HttpOnly, `Secure` when not local, SameSite=Lax, `Max-Age` 30d, `Path=/`) and returns `{ ok:true }`; on miss → a small fixed delay then 401.
- **`app/api/logout/route.ts`** (Node) — clears `fd_session`, returns `{ ok:true }`.
- A **logout control** in the app header (small client button) that POSTs `/api/logout` then reloads.

### 9.4 Provider-health page (`/health`)

- **`lib/db/provider-state.ts`** gains `getAllProviderStates(): Promise<ProviderStateRow[]>` (select all, ordered by provider).
- **`app/health/page.tsx`** — server component, `force-dynamic`, behind the gate. Renders a table of every `provider_state` row: provider, calls today / daily limit, last success, last error time + truncated message, reset time. Plus a small "configuration" block: booleans for whether `FINNHUB_API_KEY` / `GEMINI_API_KEY` / `SEC_USER_AGENT` / `APP_PASSWORD` are set (**never the values**), DB reachability (the query succeeded), and the current server time. Hosts the **Prune-now** control (§9.5).

### 9.5 Cache pruning

- **`lib/db/maintenance.ts` → `pruneExpired(): Promise<{ articles: number; fundamentals: number }>`** — deletes `articles` where `expiresAt < now` and `company_fundamentals` where `expiresAt < now`; returns row counts. (Price bars have no `expiresAt` — the durable cache — and are left intact. `daily_memos` are one-per-company-per-day and tiny — left for now.)
- **`lib/db/articles.ts` → `pruneExpiredForCompany(companyId)`** — deletes that company's expired articles; called opportunistically by `news-service` after upserting, keeping the hot path bounded at negligible cost.
- **`app/api/admin/prune/route.ts`** (POST, gated) → `pruneExpired()` → returns the counts.
- **`components/prune-button.tsx`** (client, on `/health`) → POSTs `/api/admin/prune`, then shows "Removed N articles, M snapshots."

### 9.6 Hardening

- **`next.config.ts` `async headers()`** applied to all routes: `X-Frame-Options: DENY`, `X-Content-Type-Options: nosniff`, `Referrer-Policy: strict-origin-when-cross-origin`, `Strict-Transport-Security: max-age=63072000; includeSubDomains; preload`. (A strict CSP is deferred — high breakage risk with Recharts/Next inline styles, low marginal value on a gated private app.)
- **`app/robots.ts`** → returns a disallow-all rule (`userAgent: "*"`, `disallow: "/"`) — private; prevents indexing alongside the gate.
- Boot-time env validation (existing) still throws on missing required vars; prod error pages don't leak stack traces (Next default).

### 9.7 Error handling — "degrade, never crash"

- Missing `APP_PASSWORD`/`SESSION_SECRET`: locally → gate off (dev convenience); on Vercel → fail-closed 503 (never silently public). The login route returns 503 in that state rather than setting an unsigned cookie.
- Wrong password → 401 + inline message; never reveals whether the password var is set.
- `/health` and `/api/admin/prune` are gated; a DB read error on `/health` renders a clear "database unreachable" state instead of crashing. `pruneExpired` failures are caught and surfaced as an error, not a thrown page.

### 9.8 Testing

- **Unit (Vitest):** `session.ts` — sign→verify round-trip, tampered token → false, expired → false, malformed → false; `constantTimeEqual`; `safeNextPath` (allows `/ticker/AAPL`; rejects `//evil.com`, `/\evil`, `https://…`; returns `/` for junk); `gate.shouldAllow` (login/static allowed; gated path w/o session → `login`; empty password + `onVercel` → `misconfig`; empty password local → allow; valid session → allow); `robots` output.
- **Integration (live Neon, as prior SPs):** `pruneExpired()` deletes expired articles + fundamentals and keeps fresh rows (seed one expired + one fresh of each); `pruneExpiredForCompany`; `getAllProviderStates()` returns seeded rows.
- **E2E (Playwright):** the dev server runs with `APP_PASSWORD` + `SESSION_SECRET` set (gate on). A **global-setup** logs in once via `/api/login` and saves `storageState`; existing smoke specs adopt it so they keep passing behind the gate. A new **`auth.spec.ts`**: unauthenticated `/` → redirected to `/login`; a wrong password → inline error, still on `/login`; the correct password → cookie set → lands on `/`; `/health` renders the provider table.
- **Live verification (standing rule):** `pnpm build` then `pnpm start` with the gate on → manually confirm: `/` redirects to `/login`; wrong password rejected; correct password → app usable; logout clears the cookie; `/health` shows real provider rows (fmp/finnhub/gemini/sec); `curl -I` shows the four security headers; `/robots.txt` disallows. Screenshot the login + health pages. Lint stays verified in CI (OOM locally).

### 9.9 Deploy runbook (user-only steps)

The assistant can write all code, run a local production build + smoke, generate a `SESSION_SECRET`, and push the branch on request. The following require Kavin's accounts/secrets and **cannot** be performed by the assistant:

1. **Authorize the push** of `rebuild-nextjs-mvp` to `origin` (GitHub).
2. **Vercel:** create a free Hobby account → *Add New → Project* → import `kavinravi/finance-dashboard` → set **Production Branch = `rebuild-nextjs-mvp`**; Next.js is auto-detected, root = repo root.
3. **Environment variables** (Vercel → Settings → Environment Variables), copying secret values from the local `.env`: `DATABASE_URL` (the same Neon pooled URL — reuse), `FMP_API_KEY`, `FINNHUB_API_KEY`, `GEMINI_API_KEY`, `GEMINI_MODEL` (`gemini-3.5-flash`), `SEC_USER_AGENT`, `APP_PASSWORD` (chosen shared password), `SESSION_SECRET` (generated), and optionally `FMP_DAILY_LIMIT` / `GEMINI_DAILY_LIMIT`.
4. **Migrations:** none — prod reuses the already-migrated Neon DB. (A fresh DB would need `pnpm db:migrate` against its URL.)
5. **Deploy** → Vercel builds and returns a `*.vercel.app` URL.
6. **Verify + share:** open the URL → confirm the login gate, a ticker page, and `/health` → send the URL + password to viewers. Keep it gated (personal-use licensing). Optional: add a custom domain.

### 9.10 SP4 acceptance criteria

- Visiting any app route unauthenticated redirects to a styled `/login`; the correct shared password sets a session and grants access; a wrong password is rejected; logout clears the session.
- On Vercel without `APP_PASSWORD`/`SESSION_SECRET`, the app fails closed (never serves content publicly); locally without them, dev is ungated.
- `/health` (gated) shows live `provider_state` for fmp/finnhub/gemini/sec plus key-configured booleans and DB reachability.
- Expired `articles`/`company_fundamentals` are pruned opportunistically during use and on demand via the `/health` "Prune now" button.
- Security headers and a disallow-all `robots.txt` are served; no strict CSP regressions.
- Unit + integration + E2E suites pass with the gate on; a local production build smoke passes.
- The app is deployable to a private `*.vercel.app` URL per the §9.9 runbook (account/secret steps performed by the user).

## 10. SP5 detailed design — UX refinements (post-deploy)

Added after the first Vercel deploy, from live-use feedback. Four changes: cap news volume, split the ticker page into tabbed routes, add RSI/MACD charts, and add a multi-stock watchlist overlay alongside the existing 2-stock compare. No new providers; reuses existing data + the already-computed indicators.

### 10.1 Scope & decisions (locked during brainstorming)

- **News cap = most-recent 10** (within the 7-day window), applied at the query level *and* to the memo's Gemini input (less noise + lower cost).
- **Ticker page split = separate URLs + a persistent tab bar** (not in-page tabs): `/ticker/[symbol]` (Charts & Fundamentals) + `/ticker/[symbol]/news` (News & Memo) under a shared `layout.tsx`. Chosen for shareable/bookmarkable views and load/cost isolation — the Gemini memo only generates when the News tab is opened.
- **RSI + MACD charts** on the Charts tab (Recharts, sharing the price date axis).
- **Watchlist = overlay-only, DB-shared, single global list.** No holdings/$/weights (that would push into the portfolio-advice posture the app avoids); no per-user scoping (single-tenant gated app). Capped at 10 tickers to bound fetch cost.
- **Compare stays as-is** (2-ticker `/compare`), reachable from the Charts tab's compare form.

**Deferred:** per-user watchlists; holdings/P&L tracking; watchlist reordering/notes; a range selector on the new charts (still full ~2y history per the §6 note); unifying the compare chart with the new overlay chart.

### 10.2 News cap

- `lib/db/articles.ts → getRecentArticles(companyId, sinceIso, limit = 10)` adds `.limit(limit)` (already newest-first).
- `news-service.getNews` passes the cap; `memo-service` feeds Gemini the same capped set. `NewsTable` is unchanged (renders what it's given).

### 10.3 Tabbed ticker routes + RSI/MACD

- `app/ticker/[symbol]/layout.tsx` — server component: back-link + ticker symbol + a client `TickerTabs` bar (`usePathname` highlights Charts vs News) + `{children}`. No data fetch (symbol from params) → no double-fetch across tabs.
- `app/ticker/[symbol]/page.tsx` (Charts & Fundamentals) — today's content moved here: header price + `StalenessBadge`, `ReturnsTable`, `PriceChart`, **`RsiChart`**, **`MacdChart`**, the compare form, `FundamentalsCard`. Fetches `getTickerData`.
- `app/ticker/[symbol]/news/page.tsx` (News & Memo) — `MemoCard` + capped `NewsTable` (SSR via `getNews`). Does **not** call `getTickerData` (no price series needed) → cheap.
- `price-service`: expose `macdLine` / `macdSignal` / `macdHistogram` in `TickerData.indicators` (today only the histogram is surfaced; `macd()` already returns all three). `rsi14` is already present.
- `components/rsi-chart.tsx` — Recharts line of `rsi14` over dates, 30/70 reference lines, y-axis 0–100.
- `components/macd-chart.tsx` — Recharts `macdLine` + `macdSignal` lines + `macdHistogram` bars on a shared date axis.
- `price-chart`, `returns-table`, `fundamentals-card`, `memo-card`, `news-table` are reused unchanged.

### 10.4 Watchlist (overlay-only, DB-shared)

- **Data model:** new `watchlist` table — `id` uuid PK · `ticker` text **unique** (uppercase) · `createdAt` timestamptz default now. One global list. Migration via `drizzle-kit generate` → applied to Neon.
- `lib/db/watchlist.ts` — `getWatchlist()`, `addToWatchlist(ticker)` (uppercase, `onConflictDoNothing`), `removeFromWatchlist(ticker)`.
- `lib/services/watchlist-service.ts → getWatchlistOverlay()`: read tickers (cap 10) → `getTickerData` each → align on the common-date intersection (sorted) → normalize each series to 100 at the first common date → return `{ tickers, dates, series: { ticker, normalized }[] }`. Generalizes `comparison-service`'s normalization to N; a ticker that fails to resolve or has no overlap is skipped (page never crashes).
- **API** `app/api/watchlist/route.ts` (gated, Node runtime): `GET` → current tickers; `POST {ticker}` (Zod ticker regex, reuse SP1's) → add; `DELETE {ticker}` → remove. Each returns the updated list.
- **UI** `app/watchlist/page.tsx` (server, `force-dynamic`) — the overlay chart + a client `WatchlistManager` (add-ticker input + ticker chips with remove buttons that call the API and refresh). `components/overlay-chart.tsx` — Recharts multi-line (N normalized series, legend, distinct colors, shared date axis). Empty state: a friendly prompt + the add box.
- **Nav:** add "Watchlist" to `components/app-nav.tsx`.
- **Cost note:** each overlay ticker calls `getTickerData` (DB-cache read; a cold/stale ticker may hit FMP once). The 10-ticker cap + existing FMP budget gating bound this.

### 10.5 Error handling — "degrade, never crash"

- Empty watchlist → friendly empty state. A ticker that won't resolve or shares no overlapping dates is dropped from the overlay (not fatal). The news cap doesn't change any degrade path. RSI/MACD lines simply start once their series become non-null (early bars are null by construction).

### 10.6 Testing

- **Unit (Vitest):** `getWatchlistOverlay` N-series normalization + date-intersection (incl. a non-overlapping ticker skipped); news-cap limit honored by `getRecentArticles`.
- **Integration (live Neon):** `watchlist` repo add/get/remove + uppercase dedupe; `getRecentArticles` returns ≤10 newest.
- **E2E (Playwright, gated via storageState):** ticker tab navigation (Charts ↔ News URLs); News tab shows ≤10 rows; watchlist add → chip + overlay line appears, remove → gone; RSI/MACD present on the Charts tab. Memo + fundamentals still intercepted by fixtures.
- **Live smoke + screenshots:** Charts tab (price + RSI + MACD), News tab, and the watchlist overlay with a few real tickers.

### 10.7 SP5 acceptance criteria

- A ticker exposes two bookmarkable views (Charts & Fundamentals / News & Memo) via a persistent tab bar; the memo only generates when the News tab is opened.
- The Charts tab shows price (with MAs), RSI, and MACD charts plus returns, fundamentals, and the compare entry.
- The news list and the memo's input are capped at the 10 most-recent articles.
- A DB-backed watchlist adds/removes tickers and overlays them as normalized lines on one chart (≤10), shared across devices; the 2-stock `/compare` is unchanged.
- Unit + integration + E2E pass; live smoke + screenshots confirm the charts, tabs, and overlay render.

## 11. SP6 detailed design — Chart ranges, watchlist tab, global search (post-deploy)

Second post-deploy refinement round, from continued live-use feedback. Three changes: customizable chart date ranges (presets + custom calendar), move Watchlist into the ticker tab bar, and a persistent global search header. No new providers; reuses existing data + the existing search.

### 11.1 Scope & decisions (locked during brainstorming)

- **Price history = max available.** `price-service` fetches each ticker's full history on a cold cache and refreshes **incrementally** (a warm-but-stale cache fetches only from the last cached bar forward). "ALL" = full history; custom ranges are bounded by it.
- **Range applied client-side.** The Charts page ships the full bars + indicator series; a pure `sliceByRange` slices both to the selected window. Indicators are computed on the **full** series first, so MA/RSI/MACD stay correct at the window's left edge. Default view = **1Y**.
- **Presets: 1M / 3M / 6M / YTD / 1Y / ALL**, plus a custom from–to via native `<input type="date">` (calendar dropdown, dark-styled, clamped to the available range). Only the three charts respond; the returns table stays fixed-period.
- **Watchlist = middle tab.** `TickerTabs` → Charts & Fundamentals · Watchlist · News & Memo; Watchlist links to the global `/watchlist` (kept in the header nav too).
- **Persistent global search header.** A `SiteHeader` (search + nav) replaces `AppNav`; the "← Search" back link is removed. The header search is hidden on `/` (the homepage keeps its own big search) and the whole header is hidden on `/login`.

**Deferred:** chart downsampling / on-demand range fetch for very long histories (ship-full is accepted at personal scale); making the returns table range-aware; intraday data.

### 11.2 Price history fetch (max + incremental)

- `lib/services/price-service.ts`: introduce `HISTORY_FLOOR = "1970-01-01"`. Replace the fixed 2-year `from` with: **cold cache** (no bars) → fetch `HISTORY_FLOOR`→today; **warm-but-stale** → fetch from `lastBarDate` (minus a few-day buffer for revisions) → today; upsert as today. The now-unused `_range` param is dropped (slicing is client-side). Extract a pure `fetchFromDate(bars, today)` helper (returns the `from` string) for unit testing.
- `fmp.dailyPrices` / `yahoo.dailyPrices` already accept `(ticker, from, to)`; no provider change beyond the wider `from`.

### 11.3 Range slicing (pure) + controls

- `lib/charts/range.ts` (pure): `export type ChartRange = "1m" | "3m" | "6m" | "ytd" | "1y" | "all" | { from: string; to: string }`. `rangeStartDate(range, bars, today): string` (preset → cutoff; YTD → Jan 1 of the current year; ALL → first bar's date; custom → its `from`). `sliceByRange(bars, indicators, range, today)` returns `{ bars, indicators }` sliced by the same `[startIndex, endIndex]` span across bars + every indicator array. Clamps to available data; empty input → empty.
- `components/ticker-charts.tsx` (client): holds the range state, renders the preset buttons (active one highlighted like the tabs) + the two date inputs (styled), calls `sliceByRange`, and renders `PriceChart` / `RsiChart` / `MacdChart` with the sliced data. Default `"1y"`.
- `app/ticker/[symbol]/page.tsx`: passes the full `data.bars` + `data.indicators` into `<TickerCharts/>` (replacing the three inline chart blocks). Header price, returns table, compare form, and fundamentals stay.

### 11.4 Watchlist tab

- `components/ticker-tabs.tsx`: add a middle `Link` to `/watchlist` ("Watchlist") between the Charts and News links. It links away from the ticker (the shared watchlist), so it is never the "active" tab on ticker routes.

### 11.5 Global search header

- `components/site-header.tsx` (client, replaces `app-nav.tsx`): renders the nav links (Home / Watchlist / Health / Log out — logout posts `/api/logout`) plus a compact `SearchBar`. Via `usePathname`: hide the whole header on `/login`; hide the `SearchBar` (keep nav) on `/`.
- `app/layout.tsx`: render `<SiteHeader/>` instead of `<AppNav/>`; delete `app-nav.tsx`. `<Analytics/>` stays.
- `app/ticker/[symbol]/layout.tsx`: remove the "← Search" link (keep the symbol heading + tabs).
- `app/page.tsx`: unchanged — keeps its big `SearchBar` + `RecentSearches`; the header search is simply hidden here.
- `SearchBar` is reused as-is (it already navigates to `/ticker/[symbol]` on select).

### 11.6 Error handling — "degrade, never crash"

- A custom range with `from > to` or outside the available data clamps to the valid window (or shows the full series); never crashes. Empty bars → the existing "couldn't resolve" path. `sliceByRange` is pure + total. The header search degrades exactly as today (a search-API failure → empty results).

### 11.7 Testing

- **Unit (Vitest):** `rangeStartDate` / `sliceByRange` (each preset, the YTD boundary, custom, clamping, empty + out-of-range); `fetchFromDate` (cold → floor, stale → last bar).
- **E2E (Playwright, gated):** range presets toggle active state and the custom date inputs accept values on a charts tab; the Watchlist tab sits between Charts and News and navigates to `/watchlist`; the header search is visible on a ticker page and hidden on `/`; submitting a header search lands on a ticker page.
- **Live smoke + screenshots:** AAPL at 1M / 1Y / ALL + a custom range; the header search on a ticker; the three-tab bar.

### 11.8 SP6 acceptance criteria

- The charts offer 1M/3M/6M/YTD/1Y/ALL presets + a custom calendar range, styled to match, switching instantly; indicators remain correct at the window's left edge; "ALL" shows full available history.
- The ticker tab bar reads Charts & Fundamentals · Watchlist · News & Memo, and Watchlist opens the shared watchlist.
- A search bar persists in the header on every page except the homepage (and login); the "← Search" link is gone; searching navigates to the ticker.
- Unit + E2E pass; live smoke + screenshots confirm the ranges, the tab, and the header search.

## 12. Out of scope (MVP)

Per `plan.md` non-goals: no scraping of paywalled article bodies, no trading execution, no portfolio optimization, no public redistribution of provider data, no delisted-company database, no buy/sell/hold output. Also out of scope for the MVP specifically: local FinBERT, a separate Python service, Alpha Vantage, paid data tiers, and a background scheduler.

## 13. Assumptions

- Personal/non-commercial use; app stays private behind a password gate.
- US-listed securities (FMP free is US-only; SEC is US-only).
- Daily (EOD) data is sufficient; no realtime/intraday in the MVP.
- Free-tier quotas (FMP 250/day, Gemini free tier, Finnhub 60/min) are adequate for one user at personal volume with caching.
